from abc import ABC
import builtins
import importlib
from itertools import accumulate
import shutil
from typing import Dict, List, Optional, Tuple
import torch
import tqdm
import os
import pynvml

from DeepLearning_API import MODELS_DIRECTORY, PREDICTIONS_DIRECTORY, CONFIG_FILE
from DeepLearning_API.config import config
from DeepLearning_API.utils import DatasetUtils, State, gpuInfo, getDevice, NeedDevice, logImageFormat
from DeepLearning_API.dataset import DataPrediction, DataSet
from DeepLearning_API.HDF5 import HDF5, ModelPatch, Patch
from DeepLearning_API.networks.network import ModelLoader
from DeepLearning_API.transform import Transform, TransformLoader

from torch.utils.tensorboard.writer import SummaryWriter


class OutDataset(DatasetUtils):

    def __init__(self, filename: str = "Dataset.h5", group: str = "default", pre_transforms : Dict[str, TransformLoader] = {"default:Normalize": TransformLoader()}, post_transforms : Dict[str, TransformLoader] = {"default:Normalize": TransformLoader()}) -> None:
        super().__init__(filename, read=False)
        self.group = group
        self._pre_transforms = pre_transforms
        self._post_transforms = post_transforms
        self.pre_transforms : List[Transform] = []
        self.post_transforms : List[Transform] = []

    def load(self, name_layer: str, device: torch.device):
        if self._pre_transforms is not None:
            for classpath, transform in self._pre_transforms.items():
                transform = transform.getTransform(classpath, DL_args =  "{}.outsDataset.{}.OutDataset.pre_transforms".format(os.environ["DEEP_LEARNING_API_ROOT"], name_layer))
                transform.setDevice(device)
                self.pre_transforms.append(transform)

        if self._post_transforms is not None:
            for classpath, transform in self._post_transforms.items():
                transform = transform.getTransform(classpath, DL_args =  "{}.outsDataset.{}.OutDataset.post_transforms".format(os.environ["DEEP_LEARNING_API_ROOT"], name_layer))
                transform.setDevice(device)
                self.post_transforms.append(transform)
        

class OutSameAsGroupDataset(OutDataset):

    @config("OutDataset")
    def __init__(self, dataset_filename: str = "Dataset.h5", group: str = "default", sameAsGroup: str = "default" , patchCombine:Optional[str] = None, pre_transforms : Dict[str, TransformLoader] = {"default:Normalize": TransformLoader()}, post_transforms : Dict[str, TransformLoader] = {"default:Normalize": TransformLoader()}) -> None:
        super().__init__(dataset_filename, group, pre_transforms, post_transforms)
        self.sameAsGroup = sameAsGroup

class OutLayerDataset(OutDataset):

    @config("OutDataset")
    def __init__(self, dataset_filename: str = "Dataset.h5", group: str = "default", patch: ModelPatch = ModelPatch(), pre_transforms : Dict[str, TransformLoader] = {"default:Normalize": TransformLoader()}, post_transforms : Dict[str, TransformLoader] = {"default:Normalize": TransformLoader()}) -> None:
        super().__init__(dataset_filename, group, pre_transforms, post_transforms)
        self.patch = patch

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
                    groupsInput: List[str] = ["default"],
                    device: Optional[int] = None,
                    outsDataset: Dict[str, OutDatasetLoader] = {"default:Default" : OutDatasetLoader()}) -> None:
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
        self.outsDatasetLoader = outsDataset
        self.outsDataset: Dict[str, OutDataset]

        self.setDevice(getDevice(device))
        self.outsDataset = {name.replace(":", ".") : value.getOutDataset(name) for name, value in self.outsDatasetLoader.items()}
        for name, outDataset in self.outsDataset.items():
            outDataset.load(name.replace(".", ":"), self.device)
        
    def setDevice(self, device: torch.device):
        super().setDevice(device)
        self.dataset.setDevice(device)
        self.model.setDevice(device)
        
    def __enter__(self):
        pynvml.nvmlInit()
        self.dataset.__enter__()
        self.dataloader_prediction = self.dataset.getData()
        return self
    
    def __exit__(self, type, value, traceback):
        self.dataset.__exit__(type, value, traceback)
        if self.tb is not None:
            self.tb.close()
        pynvml.nvmlShutdown()
    
    """def _combine_patch(self, group, out_patch_accu, n_channel, x):
        out_dataset = DataSet.getDatasetsFromIndex(self.dataloader_prediction.dataset.data[self.groupsInput[0].replace("_0", "")], self.dataloader_prediction.dataset.mapping[x])[0]
        out_data = torch.zeros([n_channel]+list(out_dataset.getShape()))

        for index, patch_data in enumerate(out_patch_accu):
            if group in self.groups:
                if self.groups[group].patch_transform is not None:
                    for patch_transformFunction in self.groups[group].patch_transform:
                        patch_data = patch_transformFunction.loadDataset(out_dataset)(patch_data)
            src_slices = tuple([slice(n_channel)]+[slice(slice_tmp.stop-slice_tmp.start) for slice_tmp in out_dataset.patch_slices[index]])
            dest_slices = tuple([slice(n_channel)]+list(out_dataset.patch_slices[index]))
            out_data[dest_slices] += patch_data[src_slices]
        return out_data"""

    def getInput(self, data_dict : Dict[str, Tuple[torch.Tensor, int, int, int]]) -> Dict[Tuple[str, bool], torch.Tensor]:
        return {(k, k in self.groupsInput) : v[0].to(self.device) for k, v in data_dict.items()}

    @torch.no_grad()
    def predict(self) -> None:
        predict_path = PREDICTIONS_DIRECTORY()+self.train_name+"/"+self.dataset.dataset_filename.split("/")[-1].replace(".h5", "")+"/"
        if os.path.exists(predict_path):
            if os.environ["DL_API_OVERWRITE"] != "True":
                accept = builtins.input("The prediction {} {} already exists ! Do you want to overwrite it (yes,no) : ".format(self.train_name, self.dataset.dataset_filename.split("/")[-1]))
                if accept != "yes":
                    return
           
            if os.path.exists(predict_path):
                shutil.rmtree(predict_path)

        if not os.path.exists(predict_path):
            os.makedirs(predict_path)
        shutil.copyfile(CONFIG_FILE(), predict_path+"Prediction.yml")
        
        path = MODELS_DIRECTORY()+self.train_name+"/StateDict/"
        if os.path.exists(path) and os.listdir(path):
            name = sorted(os.listdir(path))[-1]
            state_dict : Dict[str, torch.Tensor] = torch.load(path+name)
        else:
            raise Exception("Model : {} does not exist !".format(self.train_name))

        self.model.init(autocast=False, state = State.PREDICTION)
        print(state_dict.keys())
        exit(0)
        self.model.load(state_dict)
        self.model.to(self.device)
        self.dataset.load()
        self.model.eval()

        self.tb = SummaryWriter(log_dir = predict_path+"/Metric/")
        
        description = lambda : "Prediction : Metric ("+" ".join(["{} : ".format(name)+" ".join(["{} : {:.4f}".format(nameMetric, value) for nameMetric, value in network.measure.getLastMetrics().items()]) for name, network in self.model.getNetworks().items() if network.measure is not None])+") "+gpuInfo(self.device)
        dataset: DataSet = self.dataloader_prediction.dataset 

        with tqdm.tqdm(iterable = enumerate(self.dataloader_prediction), desc = description(), total=len(self.dataloader_prediction)) as batch_iter:
            for _, data_dict in batch_iter:
                input = self.getInput(data_dict)
                for name, output in self.model(input, list(self.outsDataset.keys())):
                    outDataset = self.outsDataset[name]
                    if isinstance(outDataset, OutSameAsGroupDataset):
                        group = outDataset.sameAsGroup
                        if outDataset.sameAsGroup in data_dict:
                            for value in data_dict[group][1]:
                                input_dataset = dataset.getDatasetsFromIndex(group, [int(value)])[0]
                                input_dataset.patch.addLayer(output)
                                if input_dataset.patch.isFull():
                                    outDataset.writeData(group, input_dataset.name, input_dataset.patch.assemble(device=self.device).cpu().numpy(), dict(input_dataset._dataset.attrs))
                    else:
                        
                        pass
                exit(0)
                #self._predict_log(data_dict)
                
                for data in data_dict:
                    print()
                    exit(0)
                """for group in out:
                    for batch in range(out[group].shape[0]):
                        if group not in out_patch_accu:
                            out_patch_accu[group] = []
                        out_patch_accu[group].append(out[group][batch].cpu())

                        x, idx = self.dataloader_prediction.dataset.getMap(it)
                        if idx == self.dataloader_prediction.dataset.nb_patch[x]-1:
                            out_data[group] = self._combine_patch(group, out_patch_accu[group], out[group].shape[1], x)               
                            del out_patch_accu[group]
                    
                    if len(out_data) == len(out):
                        if self.metrics is not None:
                            value, values = self._metric(out_data, self.dataloader_prediction.dataset.getData(x))
                        
                        for group in out_data:
                            if group in self.groups:
                                if self.groups[group].transform is not None:
                                    for transformFunction in self.groups[group].transform:
                                        out_dataset = DataSet.getDatasetsFromIndex(self.dataloader_prediction.dataset.data[self.groupsInput[0].replace("_0", "")], self.dataloader_prediction.dataset.mapping[x])
                                        out_data[group] = transformFunction.loadDataset(out_dataset[0])(out_data[group])
                        
                        name_dataset = "_".join([dataset.name.split("/")[-1] for dataset in out_dataset])
                        with DatasetUtils(PREDICTIONS_DIRECTORY()+self.train_name+"/"+self.dataset.dataset_filename.split("/")[-1], read=False) as datasetUtils:
                            for group in out_data:
                                out = out_data[group].numpy()
                                datasetUtils.writeImage(group, name_dataset, out_dataset[0].to_image(out[0], out.dtype))
                                if self.metrics is not None:
                                    datasetUtils.h5.attrs["value"] = value
                                    for name in values:
                                        datasetUtils.h5.attrs[name] = values[name]
                        out_data.clear()"""
                
                batch_iter.set_description(description())
                self.it += 1

    def _predict_log(self, data_dict : Dict[Tuple[str, bool], torch.Tensor]):
        assert self.tb, "SummaryWriter is None"
        models = {"" : self.model}

        for label, model in models.items():
            for name, network in model.getNetworks().items():
                if network.measure is not None:
                    self.tb.add_scalars("{}{}/Loss".format(name, label), network.measure.format(isLoss=True), self.it)
                    self.tb.add_scalars("{}{}/Metric".format(name, label), network.measure.format(isLoss=False), self.it)

            if self.images_log:
                addImageFunction = lambda name, output_layer: self.tb.add_image("result/Train/{}{}".format(name, label), logImageFormat(output_layer), self.it, dataformats='HW' if output_layer.shape[1] == 1 else 'CHW')

                images_log = []
                
                data_dict_format = {k[0] : v for k, v in data_dict.values()}
                for name in self.images_log:
                    if name in data_dict_format:
                        addImageFunction(name, data_dict[name])
                    else:
                        images_log.append(name.replace(":", "."))
                model.apply_layers_function(data_dict, images_log, addImageFunction)
