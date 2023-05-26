from abc import ABC, abstractmethod
import builtins
import importlib
import shutil
import torch
import tqdm
import os
import pynvml

from DeepLearning_API import MODELS_DIRECTORY, PREDICTIONS_DIRECTORY, CONFIG_FILE
from DeepLearning_API.config import config
from DeepLearning_API.utils import DatasetUtils, State, gpuInfo, getDevice, NeedDevice, logImageFormat, Attribute, get_patch_slices_from_nb_patch_per_dim
from DeepLearning_API.dataset import DataPrediction, DataSet
from DeepLearning_API.HDF5 import Accumulator
from DeepLearning_API.networks.network import Measure, ModelLoader
from DeepLearning_API.transform import Transform, TransformLoader

from torch.utils.tensorboard.writer import SummaryWriter
from typing import Union

class OutDataset(DatasetUtils, NeedDevice, ABC):

    def __init__(self, filename: str, group: str, pre_transforms : dict[str, TransformLoader], post_transforms : dict[str, TransformLoader], patchCombine: Union[str, None]) -> None: 
        super().__init__(filename, read=False)
        self.group = group
        self._pre_transforms = pre_transforms
        self._post_transforms = post_transforms
        self.patchCombine = patchCombine
        self.pre_transforms : list[Transform] = []
        self.post_transforms : list[Transform] = []
        self.output_layer_accumulator: dict[int, Accumulator] = {}
        self.attributes: dict[int, dict[int, Attribute]] = {}

    def load(self, name_layer: str):
        if self._pre_transforms is not None:
            for classpath, transform in self._pre_transforms.items():
                transform = transform.getTransform(classpath, DL_args =  "{}.outsDataset.{}.OutDataset.pre_transforms".format(os.environ["DEEP_LEARNING_API_ROOT"], name_layer))
                self.pre_transforms.append(transform)

        if self._post_transforms is not None:
            for classpath, transform in self._post_transforms.items():
                transform = transform.getTransform(classpath, DL_args =  "{}.outsDataset.{}.OutDataset.post_transforms".format(os.environ["DEEP_LEARNING_API_ROOT"], name_layer))
                self.post_transforms.append(transform)

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
        return self.output_layer_accumulator[index].isFull()

    @abstractmethod
    def getOutput(self, index: int, dataset: DataSet) -> torch.Tensor:
        pass

    def write(self, index: int, name: str, layer: torch.Tensor, measure: dict[str, float]):
        self.attributes[index].update({k : v for k, v in measure.items()})
        self.writeData(self.group, name, layer.numpy(), self.attributes[index][0])
        self.attributes.pop(index)

class OutSameAsGroupDataset(OutDataset):

    @config("OutDataset")
    def __init__(self, dataset_filename: str = "Dataset.h5", group: str = "default", sameAsGroup: str = "default", pre_transforms : dict[str, TransformLoader] = {"default:Normalize": TransformLoader()}, post_transforms : dict[str, TransformLoader] = {"default:Normalize": TransformLoader()}, patchCombine: Union[str, None] = None) -> None:
        super().__init__(dataset_filename, group, pre_transforms, post_transforms, patchCombine)
        self.group_src, self.group_dest = sameAsGroup.split("/")

    def addLayer(self, index: int, index_patch: int, layer: torch.Tensor, dataset: DataSet):
        if index not in self.output_layer_accumulator:
            input_dataset = dataset.getDatasetsFromIndex(self.group_dest, [index])[0]
            self.output_layer_accumulator[index] = Accumulator(input_dataset.patch.patch_slices, self.patchCombine, batch=False)
            self.attributes[index] = {}
            for i in range(len(input_dataset.patch.patch_slices)):
                self.attributes[index][i] = Attribute({k : v for k, v in input_dataset.cache_attribute.items()})
        
        for transform in self.pre_transforms:
            layer = transform(layer, self.attributes[index][index_patch])
        for transform in reversed(dataset.groups_src[self.group_src][self.group_dest].post_transforms):
            layer = transform.inverse(layer, self.attributes[index][index_patch])

        self.output_layer_accumulator[index].addLayer(index_patch, layer)
    
    def getOutput(self, index: int, dataset: DataSet) -> torch.Tensor:
        layer = self.output_layer_accumulator[index].assemble(device=self.device)
        for transform in reversed(dataset.groups_src[self.group_src][self.group_dest].pre_transforms):
            layer = transform.inverse(layer, self.attributes[index][0])

        for transform in self.post_transforms:
            layer = transform(layer, self.attributes[index][0])
        self.output_layer_accumulator.pop(index)
        return layer

class OutLayerDataset(OutDataset):

    @config("OutDataset")
    def __init__(self, dataset_filename: str = "Dataset.h5", group: str = "default", overlap : Union[list[int], None] = None, pre_transforms : dict[str, TransformLoader] = {"default:Normalize": TransformLoader()}, post_transforms : dict[str, TransformLoader] = {"default:Normalize": TransformLoader()}, patchCombine: Union[str, None] = None) -> None:
        super().__init__(dataset_filename, group, pre_transforms, post_transforms, patchCombine)
        self.overlap = overlap
        
    def addLayer(self, index: int, index_patch: int, layer: torch.Tensor, dataset: DataSet):
        if index not in self.output_layer_accumulator:
            group = list(dataset.groups.keys())[0]
            patch_slices = get_patch_slices_from_nb_patch_per_dim(list(layer.shape[2:]), dataset.getDatasetsFromIndex(group, [index])[0].patch.nb_patch_per_dim, self.overlap)
            self.output_layer_accumulator[index] = Accumulator(patch_slices, self.patchCombine, batch=False)
            self.attributes[index] = Attribute()

        for transform in self.pre_transforms:
            layer = transform(layer, self.attributes[index])
        self.output_layer_accumulator[index].addLayer(index_patch, layer)

    def getOutput(self, index: int, dataset: DataSet) -> torch.Tensor:
        layer = self.output_layer_accumulator[index].assemble(device=self.device)
        
        for transform in self.post_transforms:
            layer = transform(layer,  self.attributes[index])

        self.output_layer_accumulator.pop(index)
        return layer

class OutDatasetLoader():

    @config("OutDataset")
    def __init__(self, name_class: str = "OutSameAsGroupDataset") -> None:
        self.name_class = name_class

    def getOutDataset(self, layer_name: str) -> OutDataset:
        return getattr(importlib.import_module("DeepLearning_API.predictor"), self.name_class)(config = None, DL_args = "Predictor.outsDataset.{}".format(layer_name))
        
class Predictor(NeedDevice):

    @config("Predictor")
    def __init__(self, 
                    model: ModelLoader = ModelLoader(),
                    dataset: DataPrediction = DataPrediction(),
                    train_name: str = "name",
                    groupsInput: list[str] = ["default"],
                    device: Union[int, None] = None,
                    outsDataset: Union[dict[str, OutDatasetLoader], None] = {"default:Default" : OutDatasetLoader()},
                    images_log: list[str] = []) -> None:
        if os.environ["DEEP_LEANING_API_CONFIG_MODE"] != "Done":
            exit(0)

        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False # type: ignore

        self.groupsInput = groupsInput
        self.train_name = train_name
        self.dataset = dataset
        self.model = model.getModel(train=False)
        self.it = 0
        self.tb = None
        self.outsDatasetLoader = outsDataset if outsDataset else {}
        self.outsDataset = {name.replace(":", ".") : value.getOutDataset(name) for name, value in self.outsDatasetLoader.items()}
        self.predict_path = PREDICTIONS_DIRECTORY()+self.train_name+"/"+self.dataset.dataset_filename.split("/")[-1].replace(".h5", "")+"/"
        self.images_log = images_log
        for name, outDataset in self.outsDataset.items():
            outDataset.filename = self.predict_path+outDataset.filename
            outDataset.load(name.replace(".", ":"))
        self.setDevice(getDevice(device))
        
    def setDevice(self, device: torch.device):
        super().setDevice(device)
        self.dataset.setDevice(device)
        self.model.setDevice(device)
        for outDataset in self.outsDataset.values():
            outDataset.setDevice(device)

    def __enter__(self):
        pynvml.nvmlInit()
        self.dataset.__enter__()
        self.dataloader_prediction = self.dataset.getData()
        return self
    
    def __exit__(self, type, value, traceback):
        self.dataset.__exit__(type, value, traceback)
        for outDataset in self.outsDataset.values():
            outDataset.__exit__(type, value, traceback)
        if self.tb is not None:
            self.tb.close()
        pynvml.nvmlShutdown()

    def getInput(self, data_dict : dict[str, tuple[torch.Tensor, int, int, int]]) -> dict[tuple[str, bool], torch.Tensor]:
        inputs = {(k, True) : data_dict[k][0].to(self.device) for k in self.groupsInput}
        inputs.update({(k, False) : v[0].to(self.device) for k, v in data_dict.items() if k not in self.groupsInput})
        return inputs

    @torch.no_grad()
    def predict(self) -> None:
        if os.path.exists(self.predict_path):
            if os.environ["DL_API_OVERWRITE"] != "True":
                accept = builtins.input("The prediction {} {} already exists ! Do you want to overwrite it (yes,no) : ".format(self.train_name, self.dataset.dataset_filename.split("/")[-1]))
                if accept != "yes":
                    return
           
            if os.path.exists(self.predict_path):
                shutil.rmtree(self.predict_path)

        if not os.path.exists(self.predict_path):
            os.makedirs(self.predict_path)
        shutil.copyfile(CONFIG_FILE(), self.predict_path+"Prediction.yml")
        
        path = MODELS_DIRECTORY()+self.train_name.split("/")[0]+"/StateDict/"
        if os.path.exists(path):
            if len(self.train_name.split("/")) == 2:
                state_dict = torch.load(path+self.train_name.split("/")[-1])
            elif os.listdir(path):
                name = sorted(os.listdir(path))[-1]
                state_dict = torch.load(path+name)
        else:
            raise Exception("Model : {} does not exist !".format(self.train_name))

        for outDataset in self.outsDataset.values():    
            outDataset.__enter__()

        self.model.init(autocast=False, state = State.PREDICTION)

        self.model._compute_channels_trace(self.model, self.model.in_channels, None)

        self.model.load(state_dict, init=False)
        self.model.to(self.device)
        self.dataset.load()
        self.model.eval()

        if len(list(self.outsDataset.keys())) or len([network for network in self.model.getNetworks().values() if network.measure is not None]):
            self.tb = SummaryWriter(log_dir = self.predict_path+"/Metric/")
        
        description = lambda : "Prediction : Metric ("+" ".join(["{} : ".format(name)+" ".join(["{} : {:.4f}".format(nameMetric, value) for nameMetric, value in network.measure.getLastMetrics().items()]) for name, network in self.model.getNetworks().items() if network.measure is not None])+") "+gpuInfo(self.device)

        measures : list[Measure] = [network.measure for name, network in self.model.getNetworks().items() if network.measure is not None]
        
        dataset: DataSet = self.dataloader_prediction.dataset
        it_image = 0
        with tqdm.tqdm(iterable = enumerate(self.dataloader_prediction), desc = description(), total=len(self.dataloader_prediction)) as batch_iter:
            for _, data_dict in batch_iter:
                input = self.getInput(data_dict)
                for name, output in self.model(input, list(self.outsDataset.keys())):
                    self._predict_log(data_dict)
                    outDataset = self.outsDataset[name]
                    for i, (index, patch_index) in enumerate([(int(index), int(patch_index)) for index, patch_index in zip(list(data_dict.values())[0][1], list(data_dict.values())[0][2])]):    
                        outDataset.addLayer(index, patch_index, output[i], dataset)
                        if outDataset.isDone(index):
                            name_data = dataset.getDatasetsFromIndex(list(data_dict.keys())[0], [index])[0].name.split("/")[-1]
                            layer_output = outDataset.getOutput(index, dataset)
                            measure_result = {}
                            for measure in [measure for measure in measures if name in measure.outputsCriterions.keys()]:
                                measure.resetLoss()
                                target_data_dict = {target_group : dataset.getDatasetsFromIndex(target_group, [index])[0].data[0].unsqueeze(0) for target_group in measure.outputsCriterions[name]}
                                measure.update(name, layer_output, target_data_dict, 0)
                                measure_result.update(measure.getLastMetrics())
                                self.tb.add_scalars("{}_Prediction/Loss".format(name), measure.format(isLoss=True), it_image)
                                self.tb.add_scalars("{}_Prediction/Metric".format(name), measure.format(isLoss=False), it_image)
                            
                            outDataset.write(index, name_data, layer_output.cpu(), measure_result)
                            it_image+=1

                batch_iter.set_description(description())
                self.it += 1

    def _predict_log(self, data_dict : dict[str, tuple[torch.Tensor, int, int, int]]):
        assert self.tb, "SummaryWriter is None"
        for name, network in self.model.getNetworks().items():
            if network.measure is not None:
                self.tb.add_scalars("{}/Loss".format(name), network.measure.format(isLoss=True), self.it)
                self.tb.add_scalars("{}/Metric".format(name), network.measure.format(isLoss=False), self.it)
        
        if self.images_log:
            addImageFunction = lambda name, output_layer: self.tb.add_image("Images/{}".format(name), logImageFormat(output_layer), self.it, dataformats='HW' if output_layer.shape[1] == 1 else 'CHW')

            images_log = []
            
            for name in self.images_log:
                if name in data_dict:
                    addImageFunction(name, data_dict[name][0])
                else:
                    images_log.append(name.replace(":", "."))
            for name, layer in self.model.get_layers([v for k, v in self.getInput(data_dict).items() if k[1]], images_log):
                    addImageFunction(name, layer)
