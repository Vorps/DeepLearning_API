from functools import partial
import importlib
import torch
import tqdm
from typing import List, Dict, Tuple
import numpy as np
import datetime
import os
from enum import Enum
from DeepLearning_API import DataTrain, config, MODELS_DIRECTORY, CHECKPOINTS_DIRECTORY, STATISTICS_DIRECTORY, SETUPS_DIRECTORY, CONFIG_FILE, gpuInfo, getDevice, _getModule, Loss
from DeepLearning_API.networks import network
import datetime

DATE = lambda : datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

import shutil
import pynvml

from torch.utils.tensorboard import SummaryWriter

class State(Enum):
    TRAIN = "TRAIN"
    RESUME = "RESUME"
    PREDICTION = "PREDICTION"
    
    def __str__(self) -> str:
        return self.value
        
class Model():

    @config("Model")
    def __init__(self, classpath : str = "default:segmentation.UNet") -> None:
        self.module, self.name = _getModule(classpath.split(".")[-1] if len(classpath.split(".")) > 1 else classpath, "networks" + "."+".".join(classpath.split(".")[:-1]) if len(classpath.split(".")) > 1 else "")
        
    def getModel(self) -> network.Network:
        return getattr(importlib.import_module(self.module), self.name)(config = None, args="Trainer.Model")


class Trainer:

    @config("Trainer")
    def __init__(   self,
                    model : Model = Model(),
                    dataset : DataTrain = DataTrain(), 
                    groupsInput : List[str] = ["default"],
                    train_name : str = "default",
                    device : int = None,
                    nb_batch_per_step : int = 1,
                    manual_seed : int = None,
                    epochs: int = 100,
                    it_validation : int = None,
                    autocast : bool = False) -> None:
        if os.environ["DEEP_LEANING_API_CONFIG_MODE"] != "Done":
            exit(0)
            
        self.manual_seed = manual_seed

        if self.manual_seed is not None:
            torch.manual_seed(self.manual_seed)
        
        torch.backends.cudnn.deterministic = self.manual_seed is not None
        torch.backends.cudnn.benchmark = self.manual_seed is None
        
        self.device         = getDevice(device)
        self.train_name     = train_name
        
        self.dataset        = dataset
        self.groupsInput    = groupsInput
        self.autocast       = autocast
        
        
        self.epochs = epochs
        self.nb_batch_per_step = nb_batch_per_step
        self.epoch = 0
        self.it = 0
        self.epoch_it = 0
        self.it_validation = it_validation
        self.checkpoints : Dict[str, float] = {}
        self.tb = None
        
        self.model          = model.getModel().to(self.device)
        self.model.init(self.model.__class__.__name__, partial(Loss, device = self.device, nb_batch_per_step = self.nb_batch_per_step, groupsInput = self.groupsInput))
        

    def __enter__(self) -> None:
        pynvml.nvmlInit()
        self.dataset.__enter__()
        self.dataloader_training, self.dataloader_validation = self.dataset.getData(self.manual_seed)
        if self.it_validation is None:
            self.it_validation = len(self.dataloader_training) 
        return self
    
    def __exit__(self, type, value, traceback) -> None:
        if len(self.checkpoints) > 0 and self.epoch_it > 0:
            self.save(checkpoint_filename=list(self.checkpoints)[np.argmin(list(self.checkpoints.values()))])
        self.dataset.__exit__(type, value, traceback)
        if self.tb is not None:
            self.tb.close()
        pynvml.nvmlShutdown()

    def train(self, state : State) -> None:
        if state == State.RESUME:
            self.checkpoint_load()
        if state == State.TRAIN and os.path.exists(STATISTICS_DIRECTORY()+self.train_name+"/"):
            if os.environ["DL_API_OVERWRITE"] != "True":
                accept = input("The model {} already exists ! Do you want to overwrite it (yes,no) : ".format(self.train_name))
                if accept != "yes":
                    return
            for directory_path in [STATISTICS_DIRECTORY(), MODELS_DIRECTORY(), CHECKPOINTS_DIRECTORY(), SETUPS_DIRECTORY()]:
                if os.path.exists(directory_path+self.train_name+"/"):
                    shutil.rmtree(directory_path+self.train_name+"/")

        self.tb = SummaryWriter(log_dir = STATISTICS_DIRECTORY()+self.train_name+"/")
        
        config_namefile_src = CONFIG_FILE().replace(".yml", "")
        self.config_namefile = SETUPS_DIRECTORY()+self.train_name+"/"+config_namefile_src.split("/")[-1]+"_"+str(self.it)+".yml"
        if not os.path.exists(SETUPS_DIRECTORY()+self.train_name+"/"):
            os.makedirs(SETUPS_DIRECTORY()+self.train_name+"/")
        shutil.copyfile(config_namefile_src+".yml", self.config_namefile)

        self.dataset.load()
        self._run()

    def _run(self) -> None:
        with tqdm.tqdm(iterable = range(self.epoch, self.epochs), total=self.epochs, initial=self.epoch, desc="Progress") as epoch:
            for self.epoch in epoch:
                self._train()
    
    def getInput(self, data_dict : Dict[str, List[torch.Tensor]]) -> torch.Tensor:
        return torch.cat([torch.unsqueeze(data_dict[group], dim=1) for group in self.groupsInput], dim=1)

    def _train(self) -> None:
        self.model.train()
        scaler = torch.cuda.amp.GradScaler(enabled=self.autocast)

        for model in self.model.getSubModels():
            model.optimizer.zero_grad()

        description = lambda : "Training : Loss ("+" ".join(["{} : {:.4f}".format(model.getName(), model.loss.getLastValue()) for model in self.model.getSubModels()])+") "+gpuInfo(self.device)

        with tqdm.tqdm(iterable = enumerate(self.dataloader_training), desc = description(), total=len(self.dataloader_training), leave=False) as batch_iter:
            for i, data_dict in batch_iter:
                with torch.cuda.amp.autocast(enabled=self.autocast):
                    input = self.getInput(data_dict)
                    out = self.model.backward(input.to(self.device))
                    
                    if (i+1) % self.nb_batch_per_step == 0 or (i+1) % self.it_validation == 0:    
                        for model in self.model.getSubModels():
                            if model.loss.criterions:
                                scaler.scale(model.loss.loss).backward()
                                scaler.step(model.optimizer)
                                scaler.update()
                                model.optimizer.zero_grad()

                    if (self.it+1) % self.it_validation == 0:
                        if self.dataloader_validation is not None:
                            self._validate()
                        else:
                            self._update_lr()
                        
                        self._train_log(input, out)

                        for model in self.model.getSubModels():
                            model.loss.clear()

                        self.model.train()
                        self.epoch_it += 1

                batch_iter.set_description(description())
                
                self.it += 1

    def _update_lr(self):
        name = self.checkpoint_save()
        loss = []
        for model in self.model.getSubModels():
            value = model.loss.mean()
            if model.scheduler is not None:
                if model.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    model.scheduler.step(value)
                else:
                    model.scheduler.step()
            loss.append(value)

        self.checkpoints[name] = np.mean(np.asarray(loss))

    @torch.no_grad()
    def _validate(self) -> None:
        self.model.eval()
        description = lambda : "Validation : Loss ("+" ".join(["{} : {:.4f}".format(model.getName(), model.loss.getLastValue()) for model in self.model.getSubModels()])+") "+gpuInfo(self.device)

        with tqdm.tqdm(iterable = enumerate(self.dataloader_validation), desc = description(), total=len(self.dataloader_validation), leave=False) as batch_iter:
            for _, data_dict in batch_iter:
                input = self.getInput(data_dict)
                out = self.model.backward(input.to(self.device))
                batch_iter.set_description(description())

        self._update_lr()
        self._validation_log(input, out)
        for model in self.model.getSubModels():
            model.loss.clear()

    def checkpoint_save(self) -> None:
        path = CHECKPOINTS_DIRECTORY()+self.train_name+"/"
        name = DATE()+".pt"
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save({
            'epoch': self.epoch,
            'it': self.it,
            'checkpoints' : self.checkpoints,
            'model_state_dict': self.model.state_dict()}.update({'{}_optimizer_state_dict'.format(model.getName()): model.optimizer.state_dict() for model in self.model.getSubModels()}), 
            path+name)
        return path+name
        
    def save(self, checkpoint_filename) -> None:
        path = MODELS_DIRECTORY()+self.train_name+"/"
        name = DATE()+".pt"
        if not os.path.exists(path):
            os.makedirs(path)
        checkpoint = torch.load(checkpoint_filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        torch.save(self.model, path+name)
        os.rename(self.config_namefile, self.config_namefile.replace(".yml", "")+"_"+str(self.it)+".yml")

    def checkpoint_load(self) -> None:
        path = CHECKPOINTS_DIRECTORY()+self.train_name+"/"
        if os.path.exists(path) and os.listdir(path):
            name = sorted(os.listdir(path))[-1]
            checkpoint = torch.load(path+name)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            for model in self.model.getSubModels():
                model.optimizer.load_state_dict(checkpoint['{}_optimizer_state_dict'.format(model.getName())])
            self.epoch = checkpoint['epoch']
            self.it = checkpoint['it']
            self.checkpoints = checkpoint['checkpoints']
        else:
            raise Exception("Model : {} does not exist !".format(self.train_name))

    def load(self) -> None:
        path = MODELS_DIRECTORY()+self.train_name+"/"
        if os.path.exists(path) and os.listdir(path):
            name = sorted(os.listdir(path))[-1]
            self.model.load_state_dict(torch.load(path+name))
        else:
            raise Exception("Model : {} does not exist !".format(self.train_name))    
        
    def _train_log(self, input, out):
        for model in self.model.getSubModels():
            self.tb.add_scalars("{}/Loss/Trainning".format(model.getName()), model.loss.format(), self.it)
            self.tb.add_scalar("{}/Learning Rate".format(model.getName()), model.optimizer.param_groups[0]['lr'], self.it)
        for name, weight in self.model.named_parameters():
            self.tb.add_histogram(name, weight, self.it)
            if weight.grad is not None:
                self.tb.add_histogram(f'{name}.grad',weight.grad, self.it)
        images =  self.model.logImage(input, out)
        if images is not None:
            for name, image in images.items():
                self.tb.add_image("result/Train/"+name, image, self.it, dataformats='HW' if image.ndim == 2 else 'CHW')


    def _validation_log(self, input, out):
        for model in self.model.getSubModels():
            self.tb.add_scalars("{}/Loss/Validation".format(model.getName()), model.loss.format(), self.it)
        images =  self.model.logImage(input, out)
        if images is not None:
            for name, image in images.items():
                self.tb.add_image("result/Validation/"+name, image, self.it, dataformats='HW' if image.ndim == 2 else 'CHW')
