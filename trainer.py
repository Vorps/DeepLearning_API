import torch
from torch.utils.data import DataLoader
from torch.backends import cudnn
import tqdm
import numpy as np
import datetime
import os
from DeepLearning_API import MODELS_DIRECTORY, CHECKPOINTS_DIRECTORY, STATISTICS_DIRECTORY, SETUPS_DIRECTORY, CONFIG_FILE, URL_MODEL
from DeepLearning_API.dataset import DataTrain, DataSet
from DeepLearning_API.config import config
from DeepLearning_API.utils import gpuInfo, getDevice, State, logImageFormat
from DeepLearning_API.networks.network import Network, ModelLoader
import datetime
import dill
from typing import Union, Callable
from torch.cuda.amp.autocast_mode import autocast
import random
DATE = lambda : datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import shutil
import pynvml
from functools import partial
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.swa_utils import AveragedModel
import torch.distributed as dist


def setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class _Trainer():

    def __init__(self, train_name: str, images_log: Union[list[str], None] , save_checkpoint_mode: str, epochs: int, epoch: int, groupsInput: list[str], autocast: bool, it_validation: Union[int, None], it: int, rank: int, model: DDP, modelEMA: AveragedModel, dataloader_training: DataLoader, dataloader_validation: Union[DataLoader, None] = None) -> None:
        self.rank = rank
        self.save_checkpoint_mode = save_checkpoint_mode
        self.train_name = train_name
        self.epochs = epochs
        self.epoch = epoch
        self.groupsInput = groupsInput
        self.model = model
        self.dataloader_training = dataloader_training
        self.dataloader_validation = dataloader_validation
        self.autocast = autocast
        self.modelEMA = modelEMA
        if self.modelEMA is not None:
            self.modelEMA.eval()
        self.it_validation = it_validation
        if self.it_validation is None:
            self.it_validation = len(dataloader_training)
        self.it = it
        self.tb = SummaryWriter(log_dir = STATISTICS_DIRECTORY()+self.train_name+"/")
        self.images_log = images_log
        self.device = self.model.device

    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        if self.tb is not None:
            self.tb.close()

    def run(self) -> None:
        with tqdm.tqdm(iterable = range(self.epoch, self.epochs), total=self.epochs, initial=self.epoch, desc="Progress") as epoch_tqdm:
            for self.epoch in epoch_tqdm:
                self.train()
                
    def getInput(self, data_dict : dict[str, tuple[torch.Tensor, int, int, int]]) -> dict[tuple[str, bool], torch.Tensor]:
        inputs = {(k, True) : data_dict[k][0] for k in self.groupsInput}
        inputs.update({(k, False) : v[0] for k, v in data_dict.items() if k not in self.groupsInput})
        return inputs

    def train(self) -> None:
        self.model.train()
        description = lambda : "Training : Loss ("+" ".join(["{}({:.6f}) : {:.4f}".format(name, network.optimizer.param_groups[0]['lr'], network.measure.getLastValue()) for name, network in self.model.module.getNetworks().items() if network.measure is not None])+") "+("Loss_EMA ("+" ".join(["{} : {:.4f}".format(name, network.measure.getLastValue()) for name, network in self.modelEMA.module.getNetworks().items() if network.measure is not None])+") " if self.modelEMA is not None else "")+gpuInfo(self.device)
        
        with tqdm.tqdm(iterable = enumerate(self.dataloader_training), desc = description(), total=len(self.dataloader_training)) as batch_iter:
            for _, data_dict in batch_iter:
                with autocast(enabled=self.autocast):
                    input = self.getInput(data_dict)
                    self.model(input)
                    self.model.module.backward(self.model.module)
                    if self.rank == 0:
                        if self.modelEMA is not None:
                            self.modelEMA.update_parameters(self.model)
                            self.modelEMA.module(input)
                            
                    if (self.it+1) % self.it_validation == 0:
                        self.model.module.update_lr()
                        if self.rank == 0:
                            self._train_log(data_dict)
                        if self.dataloader_validation is not None:
                            self._validate()
                            self.model.train()
                        if self.rank == 0:
                            self.checkpoint_save(np.mean([network.measure.mean() for network in self.model.module.getNetworks().values() if network.measure is not None]))
                        self.model.module.measureClear()

                batch_iter.set_description(description()) 
                self.it += 1
    

    @torch.no_grad()
    def _validate(self) -> None:
        self.model.module.measureClear()
        self.model.eval()
        description = lambda : "Validation : Loss ("+" ".join(["{} : {:.4f}".format(name, network.measure.getLastValue()) for name, network in self.model.module.getNetworks().items() if network.measure is not None])+") "+("Loss_EMA ("+" ".join(["{} : {:.4f}".format(name, network.measure.getLastValue()) for name, network in self.modelEMA.module.getNetworks().items() if network.measure is not None])+") " if self.modelEMA is not None else "") +gpuInfo(self.device)
        data_dict = None
        with tqdm.tqdm(iterable = enumerate(self.dataloader_validation), desc = description(), total=len(self.dataloader_validation), leave=False) as batch_iter:
            for _, data_dict in batch_iter:
                input = self.getInput(data_dict)
                self.model(input)
                if self.rank == 0:
                    if self.modelEMA is not None:
                        self.modelEMA.update_parameters(self.model.module)
                        self.modelEMA.module(input)

                batch_iter.set_description(description())
            if self.rank == 0:
                self._validation_log(data_dict)

    def checkpoint_save(self, loss) -> None:
        path = CHECKPOINTS_DIRECTORY()+self.train_name+"/"
        last_loss = None
        if os.path.exists(path) and os.listdir(path):
            name = sorted(os.listdir(path))[-1]
            state_dict = torch.load(path+name)
            last_loss = state_dict["loss"]
            if self.save_checkpoint_mode == "BEST":
                if last_loss >= loss:
                    os.remove(path+name)

        if self.save_checkpoint_mode != "BEST" or (last_loss is None or last_loss >= loss):
            name = DATE()+".pt"
            if not os.path.exists(path):
                os.makedirs(path)
            
            save_dict = {
                "epoch": self.epoch,
                "it": self.it,
                "loss": loss,
                "Model": self.model.module.state_dict()}

            if self.modelEMA is not None:
                save_dict.update({"Model_EMA" : self.modelEMA.module.state_dict()})

            save_dict.update({'{}_optimizer_state_dict'.format(name): network.optimizer.state_dict() for name, network in self.model.module.getNetworks().items() if network.optimizer is not None})
            torch.save(save_dict, path+name)

    def _train_log(self, data_dict : dict[str, tuple[torch.Tensor, int, int, int]]):
        models: dict[str, Network] = {"" : self.model.module}
        if self.modelEMA is not None:
            models["_EMA"] = self.modelEMA.module

        for label, model in models.items():
            for name, network in model.getNetworks().items():
                if network.measure is not None:
                    self.tb.add_scalars("{}{}/Loss/Trainning".format(name, label), network.measure.format(isLoss=True), self.it)
                    self.tb.add_scalars("{}{}/Metric/Trainning".format(name, label), network.measure.format(isLoss=False), self.it)
            
            if self.images_log:
                images_log = []
                addImageFunction = lambda name, layer : self.tb.add_image("result/Train/{}{}".format(name, label), logImageFormat(layer), self.it, dataformats='HW' if layer.shape[1] == 1 or layer.shape[1] > 50 else 'CHW')

                for name in self.images_log:
                    if name in data_dict:
                        addImageFunction(name, data_dict[name][0])
                    else:
                        images_log.append(name.replace(":", "."))
                model.eval()
                
                for name, layer in model.get_layers([v.to(self.rank) for k, v in self.getInput(data_dict).items() if k[1]], images_log):
                    addImageFunction(name, layer)
                model.train()


        for name, network in self.model.module.getNetworks().items():
            if network.optimizer is not None:
                self.tb.add_scalar("{}/Learning Rate".format(name), network.optimizer.param_groups[0]['lr'], self.it)

    def _validation_log(self, data_dict : dict[str, tuple[torch.Tensor, int, int, int]]):
        models: dict[str, Network] = {"" : self.model.module}
        if self.modelEMA is not None:
            models["_EMA"] = self.modelEMA.module
        
        for label, model in models.items():
            for name, network in model.getNetworks().items():
                if network.measure is not None:
                    self.tb.add_scalars("{}{}/Loss/Validation".format(name, label), network.measure.format(isLoss=True), self.it)
                    self.tb.add_scalars("{}{}/Metric/Validation".format(name, label), network.measure.format(isLoss=False), self.it)
            
            
            addImageFunction = lambda name, layer : self.tb.add_image("result/Validation/{}{}".format(name, label), logImageFormat(layer), self.it, dataformats='HW' if layer.shape[1] == 1 or layer.shape[1] > 50 else 'CHW')

            if self.images_log:
                images_log = []
                for name in self.images_log:
                    if name in data_dict:
                        addImageFunction(name, data_dict[name][0])
                    else:
                        images_log.append(name.replace(":", "."))
                        
                for name, layer in model.get_layers([v.to(self.rank) for k, v in self.getInput(data_dict).items() if k[1]], images_log):
                    addImageFunction(name, layer)

def train(rank: int, world_size: int, size: int, manual_seed: int, trainer: Callable[[int, DDP, DataLoader, DataLoader], _Trainer], dataloaders_list: list[list[DataLoader]], model: Network, modelEMA: AveragedModel, devices: list[torch.device]):
    setup(rank, world_size)
    pynvml.nvmlInit()
    if manual_seed is not None:
        np.random.seed(manual_seed * world_size + rank)
        random.seed(manual_seed * world_size + rank)
        torch.manual_seed(manual_seed * world_size + rank)
    torch.backends.cudnn.benchmark = manual_seed is None
    torch.backends.cudnn.deterministic = manual_seed is not None
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dataloaders = dataloaders_list[rank]
    for dataloader in dataloaders:
        dataloader.dataset.to(devices[rank])
        dataloader.dataset.load()
    model = Network.to(model, rank*size)
    if modelEMA is not None:
        modelEMA.module = Network.to(modelEMA.module, rank*size)
    model = DDP(model)
    with trainer(rank*size, model, modelEMA, *dataloaders) as t:
        t.run()
    pynvml.nvmlShutdown()
    cleanup()
    
class Trainer():

    @config("Trainer")
    def __init__(   self,
                    model : ModelLoader = ModelLoader(),
                    dataset : DataTrain = DataTrain(),
                    groupsInput : list[str] = ["default"],
                    train_name : str = "default",
                    devices : Union[list[int], None] = [None],
                    manual_seed : Union[int, None] = None,
                    epochs: int = 100,
                    it_validation : Union[int, None] = None,
                    autocast : bool = False,
                    gradient_checkpoints: Union[list[str], None] = None,
                    gpu_checkpoints: Union[list[str], None] = None,
                    ema_decay : float = 0,
                    images_log: Union[list[str], None] = None,
                    save_checkpoint_mode: str= "BEST") -> None:
        if os.environ["DEEP_LEANING_API_CONFIG_MODE"] != "Done":
            exit(0)
        self.manual_seed = manual_seed        
        self.train_name = train_name
        self.dataset = dataset
        self.groupsInput = groupsInput
        self.autocast = autocast
        self.epochs = epochs
        self.epoch = 0
        self.it = 0
        self.it_validation = it_validation
        self.model = model.getModel(train=True)
        self.ema_decay = ema_decay
        self.modelEMA : Union[torch.optim.swa_utils.AveragedModel, None] = None
        self.images_log = images_log
        self.gradient_checkpoints = gradient_checkpoints
        self.gpu_checkpoints = gpu_checkpoints
        self.save_checkpoint_mode = save_checkpoint_mode
        self.devices = getDevice(devices)
        self.world_size = len(self.devices)//(len(self.gpu_checkpoints)+1 if self.gpu_checkpoints else 1)
    
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        self.save()

    def load(self) -> dict[str, dict[str, torch.Tensor]]:
        if URL_MODEL().startswith("https://"):
            try:
                state_dict = {URL_MODEL().split(":")[1]: torch.hub.load_state_dict_from_url(url=URL_MODEL().split(":")[0], map_location="cpu", check_hash=True)}
            except:
                raise Exception("Model : {} does not exist !".format(URL_MODEL())) 
        else:
            path = CHECKPOINTS_DIRECTORY()+self.train_name+"/"
            if os.path.exists(path) and os.listdir(path):
                name = sorted(os.listdir(path))[-1]
                state_dict = torch.load(path+name)
            else:
                raise Exception("Model : {} does not exist !".format(self.train_name))
        if "epoch" in state_dict:
            self.epoch = state_dict['epoch']
        if "it" in state_dict:
            self.it = state_dict['it']
        return state_dict

    def save(self) -> None:
        path_checkpoint = CHECKPOINTS_DIRECTORY()+self.train_name+"/"
        path_model = MODELS_DIRECTORY()+self.train_name+"/"
        if os.path.exists(path_checkpoint) and os.listdir(path_checkpoint):
            for dir in [path_model, "{}Serialized/".format(path_model), "{}StateDict/".format(path_model)]:
                if not os.path.exists(dir):
                    os.makedirs(dir)

            for name in sorted(os.listdir(path_checkpoint)):
                checkpoint = torch.load(path_checkpoint+name)
                self.model.load(checkpoint, init=False, ema=False)

                torch.save(self.model, "{}Serialized/{}".format(path_model, name), pickle_module=dill)
                torch.save({"Model" : self.model.state_dict()}, "{}StateDict/{}".format(path_model, name))
                
                if self.modelEMA is not None:
                    self.modelEMA.module.load(checkpoint, init=False, ema=True)
                    torch.save(self.modelEMA.module, "{}Serialized/{}".format(path_model, DATE()+"_EMA.pt"))
                    torch.save({"Model_EMA" : self.modelEMA.module.state_dict()}, "{}StateDict/{}".format(path_model, DATE()+"_EMA.pt"))

            os.rename(self.config_namefile, self.config_namefile.replace(".yml", "")+"_"+str(self.it)+".yml")
    
    def _avg_fn(self, averaged_model_parameter, model_parameter, num_averaged):
        return (1-self.ema_decay) * averaged_model_parameter + self.ema_decay * model_parameter
    
    def train(self, state : State) -> None:
        if state == State.TRAIN and os.path.exists(STATISTICS_DIRECTORY()+self.train_name+"/"):
            if os.environ["DL_API_OVERWRITE"] != "True":
                accept = input("The model {} already exists ! Do you want to overwrite it (yes,no) : ".format(self.train_name))
                if accept != "yes":
                    return
            for directory_path in [STATISTICS_DIRECTORY(), MODELS_DIRECTORY(), CHECKPOINTS_DIRECTORY(), SETUPS_DIRECTORY()]:
                if os.path.exists(directory_path+self.train_name+"/"):
                    shutil.rmtree(directory_path+self.train_name+"/")
        
        state_dict = {}
        if state != State.TRAIN:
            state_dict = self.load()


        self.model.init(self.autocast, state)
        self.model.init_outputsGroup()
        self.model._compute_channels_trace(self.model, self.model.in_channels, self.gradient_checkpoints, self.gpu_checkpoints)
        self.model.load(state_dict, init=True, ema=False)
        
        if self.ema_decay > 0:
            self.modelEMA = AveragedModel(self.model, avg_fn=self._avg_fn)
            if state_dict is not None:
                self.modelEMA.module.load(state_dict, init=False, ema=True)
        
        config_namefile_src = CONFIG_FILE().replace(".yml", "")
        self.config_namefile = SETUPS_DIRECTORY()+self.train_name+"/"+config_namefile_src.split("/")[-1]+"_"+str(self.it)+".yml"
        if not os.path.exists(SETUPS_DIRECTORY()+self.train_name+"/"):
            os.makedirs(SETUPS_DIRECTORY()+self.train_name+"/")
        shutil.copyfile(config_namefile_src+".yml", self.config_namefile)

        self.dataloader = self.dataset.getData(self.world_size)
        trainer = partial(_Trainer, self.train_name, self.images_log, self.save_checkpoint_mode, self.epochs, self.epoch, self.groupsInput, self.autocast, self.it_validation, self.it)
        mp.spawn(train, args=(self.world_size, (len(self.gpu_checkpoints)+1 if self.gpu_checkpoints else 1), self.manual_seed, trainer, self.dataloader, self.model, self.modelEMA, self.devices), nprocs=self.world_size)