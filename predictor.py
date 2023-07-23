from abc import ABC, abstractmethod
import builtins
import importlib
import shutil
import torch
import tqdm
import os
import pynvml

from DeepLearning_API import MODELS_DIRECTORY, PREDICTIONS_DIRECTORY, CONFIG_FILE, URL_MODEL
from DeepLearning_API.config import config
from DeepLearning_API.utils import DatasetUtils, State, gpuInfo, getDevice, logImageFormat, Attribute, get_patch_slices_from_nb_patch_per_dim, NeedDevice, _getModule
from DeepLearning_API.dataset import DataPrediction, DataSet
from DeepLearning_API.HDF5 import Accumulator, PathCombine
from DeepLearning_API.networks.network import Measure, ModelLoader, Network
from DeepLearning_API.transform import Transform, TransformLoader

from torch.utils.tensorboard.writer import SummaryWriter
from typing import Union, Callable
import numpy as np
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from functools import partial
import torch.multiprocessing as mp
import importlib

def setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class OutDataset(DatasetUtils, NeedDevice, ABC):

    def __init__(self, filename: str, group: str, pre_transforms : dict[str, TransformLoader], post_transforms : dict[str, TransformLoader], patchCombine: Union[str, None]) -> None: 
        super().__init__(filename)
        self.group = group
        self._pre_transforms = pre_transforms
        self._post_transforms = post_transforms
        self._patchCombine = patchCombine
       
        self.pre_transforms : list[Transform] = []
        self.post_transforms : list[Transform] = []
        self.patchCombine: PathCombine = None

        self.output_layer_accumulator: dict[int, dict[int, Accumulator]] = {}
        self.attributes: dict[int, dict[int, dict[int, Attribute]]] = {}
        self.names: dict[int, str] = {}
        self.nb_data_augmentation = 0

    def load(self, name_layer: str):
        if self._pre_transforms is not None:
            for classpath, transform in self._pre_transforms.items():
                transform = transform.getTransform(classpath, DL_args =  "{}.outsDataset.{}.OutDataset.pre_transforms".format(os.environ["DEEP_LEARNING_API_ROOT"], name_layer))
                self.pre_transforms.append(transform)

        if self._post_transforms is not None:
            for classpath, transform in self._post_transforms.items():
                transform = transform.getTransform(classpath, DL_args =  "{}.outsDataset.{}.OutDataset.post_transforms".format(os.environ["DEEP_LEARNING_API_ROOT"], name_layer))
                self.post_transforms.append(transform)
        if self._patchCombine is not None:
            module, name = _getModule(self._patchCombine, "HDF5")
            self.patchCombine = getattr(importlib.import_module(module), name)(config = None, DL_args =  "{}.outsDataset.{}.OutDataset".format(os.environ["DEEP_LEARNING_API_ROOT"], name_layer))
    
    def setPatchConfig(self, patchSize: list[int], overlap: list[int], nb_data_augmentation: int) -> None:
        if self.patchCombine is not None:
            self.patchCombine.setPatchConfig(patchSize, overlap)
        self.nb_data_augmentation = nb_data_augmentation
    
    def setDevice(self, device: torch.device):
        super().setDevice(device)
        if self.pre_transforms is not None:
            for transform in self.pre_transforms:
                transform.setDevice(device)
                
        if self.post_transforms is not None:
            for transform in self.post_transforms:
                transform.setDevice(device)

    @abstractmethod
    def addLayer(self, index: int, index_patch: int, layer: torch.Tensor, dataset: DataSet):
        pass

    def isDone(self, index: int) -> bool:
        return len(self.output_layer_accumulator[index]) == self.nb_data_augmentation and all([acc.isFull() for acc in self.output_layer_accumulator[index].values()])

    @abstractmethod
    def getOutput(self, index: int, dataset: DataSet) -> torch.Tensor:
        pass

    def write(self, index: int, name: str, layer: torch.Tensor):
        super().write(self.group, name, layer.numpy(), self.attributes[index][0][0])
        self.attributes.pop(index)

class OutSameAsGroupDataset(OutDataset):

    @config("OutDataset")
    def __init__(self, dataset_filename: str = "Dataset.h5", group: str = "default", sameAsGroup: str = "default", pre_transforms : dict[str, TransformLoader] = {"default:Normalize": TransformLoader()}, post_transforms : dict[str, TransformLoader] = {"default:Normalize": TransformLoader()}, patchCombine: Union[str, None] = None, redution: str = "mean") -> None:
        super().__init__(dataset_filename, group, pre_transforms, post_transforms, patchCombine)
        self.group_src, self.group_dest = sameAsGroup.split("/")
        self.redution = redution

    def addLayer(self, index_dataset: int, index_augmentation: int, index_patch: int, layer: torch.Tensor, dataset: DataSet):
        if index_dataset not in self.output_layer_accumulator or index_augmentation not in self.output_layer_accumulator[index_dataset]:
            input_dataset = dataset.getDatasetFromIndex(self.group_dest, index_dataset)
            if index_dataset not in self.output_layer_accumulator:
                self.output_layer_accumulator[index_dataset] = {}
                self.attributes[index_dataset] = {}
                self.names[index_dataset] = input_dataset.name
            self.attributes[index_dataset][index_augmentation] = {}

            self.output_layer_accumulator[index_dataset][index_augmentation] = Accumulator(input_dataset.patch.getPatch_slices(index_augmentation), input_dataset.patch.patch_size, self.patchCombine, batch=False)

            for i in range(len(input_dataset.patch.getPatch_slices(index_augmentation))):
                self.attributes[index_dataset][index_augmentation][i] = Attribute(input_dataset.cache_attributes[0])

        for transform in self.pre_transforms:
            layer = transform(self.names[index_dataset], layer, self.attributes[index_dataset][index_augmentation][index_patch])

        for transform in reversed(dataset.groups_src[self.group_src][self.group_dest].post_transforms):
            layer = transform.inverse(self.names[index_dataset], layer, self.attributes[index_dataset][index_augmentation][index_patch])
        self.output_layer_accumulator[index_dataset][index_augmentation].addLayer(index_patch, layer)
    
    def _getOutput(self, index: int, index_augmentation: int, dataset: DataSet) -> torch.Tensor:
        layer = self.output_layer_accumulator[index][index_augmentation].assemble()
        name = self.names[index]
        if index_augmentation > 0:
            i = 0
            index_augmentation_tmp = index_augmentation-1
            for dataAugmentations in dataset.dataAugmentationsList:
                if index_augmentation_tmp < i+dataAugmentations.nb:
                    index_augmentation_tmp -= i
                    for dataAugmentation in reversed(dataAugmentations.dataAugmentations):
                        layer = dataAugmentation.inverse(index, index_augmentation_tmp, layer)
                i += dataAugmentations.nb

        for transform in reversed(dataset.groups_src[self.group_src][self.group_dest].pre_transforms):
            layer = transform.inverse(name, layer, self.attributes[index][index_augmentation][0])

        for transform in self.post_transforms:
            layer = transform(name, layer, self.attributes[index][index_augmentation][0])
        return layer

    def getOutput(self, index: int, dataset: DataSet) -> torch.Tensor:
        result = torch.cat([self._getOutput(index, index_augmentation, dataset).unsqueeze(0) for index_augmentation in self.output_layer_accumulator[index].keys()], dim=0)
        self.output_layer_accumulator.pop(index)
        if self.redution == "mean":
            result = torch.mean(result, dim=0)
        elif self.redution == "median":
            result, _ = torch.median(result, dim=0)
        else:
            raise NameError("Reduction method does not exist (mean, median)")
        return result

class OutLayerDataset(OutDataset):

    @config("OutDataset")
    def __init__(self, dataset_filename: str = "Dataset.h5", group: str = "default", overlap : Union[list[int], None] = None, pre_transforms : dict[str, TransformLoader] = {"default:Normalize": TransformLoader()}, post_transforms : dict[str, TransformLoader] = {"default:Normalize": TransformLoader()}, patchCombine: Union[str, None] = None) -> None:
        super().__init__(dataset_filename, group, pre_transforms, post_transforms, patchCombine)
        self.overlap = overlap
        
    def addLayer(self, index: int, index_patch: int, layer: torch.Tensor, dataset: DataSet):
        if index not in self.output_layer_accumulator:
            group = list(dataset.groups.keys())[0]
            patch_slices = get_patch_slices_from_nb_patch_per_dim(list(layer.shape[2:]), dataset.getDatasetFromIndex(group, index).patch.nb_patch_per_dim, self.overlap)
            self.output_layer_accumulator[index] = Accumulator(patch_slices, self.patchCombine, batch=False)
            self.attributes[index] = Attribute()
            self.names[index] = dataset.getDatasetFromIndex(group, index).name
            

        for transform in self.pre_transforms:
            layer = transform(layer, self.attributes[index])
        self.output_layer_accumulator[index].addLayer(index_patch, layer)

    def getOutput(self, index: int, dataset: DataSet) -> torch.Tensor:
        layer = self.output_layer_accumulator[index].assemble()
        name = self.names[index]
        for transform in self.post_transforms:
            layer = transform(name, layer, self.attributes[index])

        self.output_layer_accumulator.pop(index)
        return layer

class OutDatasetLoader():

    @config("OutDataset")
    def __init__(self, name_class: str = "OutSameAsGroupDataset") -> None:
        self.name_class = name_class

    def getOutDataset(self, layer_name: str) -> OutDataset:
        return getattr(importlib.import_module("DeepLearning_API.predictor"), self.name_class)(config = None, DL_args = "Predictor.outsDataset.{}".format(layer_name))

class _Predictor():

    def __init__(self, predict_path: str, images_log: Union[list[str], None], groupsInput: list[str], outsDataset: dict[str, OutDataset], rank: int, model: DDP, dataloader_prediction: DataLoader) -> None:
        self.rank = rank
        self.model = model
        self.dataloader_prediction = dataloader_prediction
        self.outsDataset = outsDataset
        self.groupsInput = groupsInput
        self.images_log = images_log
        self.tb = SummaryWriter(log_dir = predict_path+"Metric/")
        self.it = 0
        self.device = self.model.device
        self.dataset: DataSet = self.dataloader_prediction.dataset
        patch_size, overlap = self.dataset.getPatchConfig()
        for outDataset in self.outsDataset.values():
            outDataset.setPatchConfig(patch_size, overlap, np.max([int(np.sum([data_augmentation.nb for data_augmentation in self.dataset.dataAugmentationsList])+1), 1]))

    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        if self.tb is not None:
            self.tb.close()

    def getInput(self, data_dict : dict[str, tuple[torch.Tensor, int, int, int]]) -> dict[tuple[str, bool], torch.Tensor]:
        inputs = {(k, True) : data_dict[k][0] for k in self.groupsInput}
        inputs.update({(k, False) : v[0] for k, v in data_dict.items() if k not in self.groupsInput})
        return inputs
    
    @torch.no_grad()
    def run(self):
        self.model.eval()   
        description = lambda : "Prediction : Metric ("+" ".join(["{} : ".format(name)+" ".join(["{} : {:.4f}".format(nameMetric, value) for nameMetric, value in network.measure.getLastMetrics().items()]) for name, network in self.model.module.getNetworks().items() if network.measure is not None])+") "+gpuInfo(self.device)

        with tqdm.tqdm(iterable = enumerate(self.dataloader_prediction), desc = description(), total=len(self.dataloader_prediction)) as batch_iter:
            for _, data_dict in batch_iter:
                input = self.getInput(data_dict)
                for name, output in self.model(input, list(self.outsDataset.keys())):
                    if self.rank == 0:
                        self._predict_log(data_dict)
                    outDataset = self.outsDataset[name]
                    for i, (index, patch_augmentation, patch_index) in enumerate([(int(index), int(patch_augmentation), int(patch_index)) for index, patch_augmentation, patch_index in zip(list(data_dict.values())[0][1], list(data_dict.values())[0][2], list(data_dict.values())[0][3])]):    
                        outDataset.addLayer(index, patch_augmentation, patch_index, output[i].cpu(), self.dataset)
                        if outDataset.isDone(index):
                            name_data = self.dataset.getDatasetFromIndex(list(data_dict.keys())[0], index).name.split("/")[-1]
                            layer_output = outDataset.getOutput(index, self.dataset)
                            outDataset.write(index, name_data, layer_output)

                batch_iter.set_description(description())
                self.it += 1


    def _predict_log(self, data_dict : dict[str, tuple[torch.Tensor, int, int, int]]):
        for name, network in self.model.module.getNetworks().items():
            if network.measure is not None:
                self.tb.add_scalars("{}/Loss".format(name), network.measure.format(isLoss=True), self.it)
                self.tb.add_scalars("{}/Metric".format(name), network.measure.format(isLoss=False), self.it)
        
        if self.images_log:
            images_log = []
            addImageFunction = lambda name, output_layer: self.tb.add_image("Images/{}".format(name), logImageFormat(output_layer), self.it, dataformats='HW' if output_layer.shape[1] == 1 else 'CHW')
            
            for name in self.images_log:
                if name in data_dict:
                    addImageFunction(name, data_dict[name][0])
                else:
                    images_log.append(name.replace(":", "."))
            for name, layer in self.model.module.get_layers([v.to(self.rank) for k, v in self.getInput(data_dict).items() if k[1]], images_log):
                addImageFunction(name, layer)

def predict(rank: int, world_size: int, size: int, manual_seed: int, predictor: Callable[[int, DDP, DataLoader, DataLoader], _Predictor], dataloaders_list: list[list[DataLoader]], model: Network, devices: list[torch.device]):
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
    model = DDP(model)
    with predictor(rank*size, model, *dataloaders) as p:
        p.run()
    pynvml.nvmlShutdown()
    cleanup()

class Predictor():

    @config("Predictor")
    def __init__(self, 
                    model: ModelLoader = ModelLoader(),
                    dataset: DataPrediction = DataPrediction(),
                    groupsInput: list[str] = ["default"],
                    train_name: str = "name",
                    devices : Union[list[int], None] = [None],
                    manual_seed : Union[int, None] = None,
                    gpu_checkpoints: Union[list[str], None] = None,
                    outsDataset: Union[dict[str, OutDatasetLoader], None] = {"default:Default" : OutDatasetLoader()},
                    images_log: list[str] = []) -> None:
        if os.environ["DEEP_LEANING_API_CONFIG_MODE"] != "Done":
            exit(0)
        self.manual_seed = manual_seed 
        self.train_name = train_name
        self.dataset = dataset

        self.groupsInput = groupsInput
        self.model = model.getModel(train=False)
        self.it = 0
        self.outsDatasetLoader = outsDataset if outsDataset else {}
        self.outsDataset = {name.replace(":", ".") : value.getOutDataset(name) for name, value in self.outsDatasetLoader.items()}
        self.images_log = images_log
        self.predict_path = PREDICTIONS_DIRECTORY()+self.train_name+"/"
        for name, outDataset in self.outsDataset.items():
            outDataset.filename = "{}{}".format(self.predict_path, outDataset.filename)
            outDataset.load(name.replace(".", ":"))
        self.gpu_checkpoints = gpu_checkpoints
        self.devices = getDevice(devices)
        self.world_size = len(self.devices)//(len(self.gpu_checkpoints)+1 if self.gpu_checkpoints else 1)
        
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        pass

    def load(self) -> dict[str, dict[str, torch.Tensor]]:
        if URL_MODEL().startswith("https://"):
            try:
                state_dict = {URL_MODEL().split(":")[1]: torch.hub.load_state_dict_from_url(url=URL_MODEL().split(":")[0], map_location="cpu", check_hash=True)}
            except:
                raise Exception("Model : {} does not exist !".format(URL_MODEL())) 
        else:
            path = MODELS_DIRECTORY()+self.train_name.split("/")[0]+"/StateDict/"
            if os.path.exists(path):
                if len(self.train_name.split("/")) == 2:
                    state_dict = torch.load(path+self.train_name.split("/")[-1])
                elif os.listdir(path):
                    name = sorted(os.listdir(path))[-1]
                    state_dict = torch.load(path+name)
            else:
                raise Exception("Model : {} does not exist !".format(self.train_name))
        return state_dict
    
    def predict(self) -> None:
        if os.path.exists(self.predict_path):
            if os.environ["DL_API_OVERWRITE"] != "True":
                accept = builtins.input("The prediction {} {} already exists ! Do you want to overwrite it (yes,no) : ".format(self.train_name, self.dataset.dataset_filename.split("/")[-1]))
                if accept != "yes":
                    return
           
            if os.path.exists(self.predict_path):
                shutil.rmtree(self.predict_path) 

        self.model.init(autocast=False, state = State.PREDICTION)
        self.model.init_outputsGroup()
        self.model._compute_channels_trace(self.model, self.model.in_channels, None, self.gpu_checkpoints)
        self.model.load(self.load(), init=False)
        
        if len(list(self.outsDataset.keys())) == 0 and len([network for network in self.model.getNetworks().values() if network.measure is not None]) == 0:
            exit(0)

        if not os.path.exists(self.predict_path):
            os.makedirs(self.predict_path)
        shutil.copyfile(CONFIG_FILE(), self.predict_path+"Prediction.yml")

        self.dataloader = self.dataset.getData(self.world_size)
        trainer = partial(_Predictor, self.predict_path, self.images_log, self.groupsInput, self.outsDataset)
        mp.spawn(predict, args=(self.world_size, (len(self.gpu_checkpoints)+1 if self.gpu_checkpoints else 1), self.manual_seed, trainer, self.dataloader, self.model, self.devices), nprocs=self.world_size)