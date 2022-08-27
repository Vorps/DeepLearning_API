import builtins
import shutil
from typing import Dict, List, Optional
import torch
import tqdm
import os
import pynvml

from DeepLearning_API import MODELS_DIRECTORY, PREDICTIONS_DIRECTORY, CONFIG_FILE
from DeepLearning_API.config import config
from DeepLearning_API.utils import State, gpuInfo, getDevice, NeedDevice
from DeepLearning_API.dataset import DataPrediction
from DeepLearning_API.networks.network import Network, ModelLoader

class Predictor(NeedDevice):

    @config("Predictor")
    def __init__(self, 
                    model : ModelLoader = ModelLoader(),
                    dataset : DataPrediction = DataPrediction(),
                    train_name : str = "name",
                    device : Optional[int] = None) -> None:
        if os.environ["DEEP_LEANING_API_CONFIG_MODE"] != "Done":
            exit(0)

        torch.backends.cudnn.deterministic = False  # type: ignore
        torch.backends.cudnn.benchmark = True # type: ignore
        
        self.train_name = train_name
        self.dataset = dataset
        self.model = model.getModel(train=False)
        self.model.init(autocast=False, state = State.PREDICTION)
        self.setDevice(getDevice(device))

    def setDevice(self, device: torch.device):
        super().setDevice(device)
        self.dataset.setDevice(device)
        self.model.setDevice(device)
        self.model.to(device)

    def __enter__(self):
        pynvml.nvmlInit()
        self.dataset.__enter__()
        self.dataloader_prediction = self.dataset.getData()
        return self
    
    def __exit__(self, type, value, traceback):
        self.dataset.__exit__(type, value, traceback)
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

    @torch.no_grad()
    def predict(self) -> None:
        config_file_dest = PREDICTIONS_DIRECTORY()+self.train_name+"/"+self.dataset.dataset_filename.split("/")[-1].replace(".h5", "")+".yml"
        if os.path.exists(config_file_dest):
            if os.environ["DL_API_OVERWRITE"] != "True":
                accept = builtins.input("The prediction {} {} already exists ! Do you want to overwrite it (yes,no) : ".format(self.train_name, self.dataset.dataset_filename.split("/")[-1]))
                if accept != "yes":
                    return
            
            if os.path.exists(config_file_dest):
                os.remove(config_file_dest)
        if not os.path.exists(PREDICTIONS_DIRECTORY()+self.train_name+"/"):
            os.makedirs(PREDICTIONS_DIRECTORY()+self.train_name+"/")
        shutil.copyfile(CONFIG_FILE(), config_file_dest)
        
        path = MODELS_DIRECTORY()+self.train_name+"/StateDict/"
        if os.path.exists(path) and os.listdir(path):
            name = sorted(os.listdir(path))[-1]
            state_dict : Dict[str, torch.Tensor] = torch.load(path+name)
        else:
            raise Exception("Model : {} does not exist !".format(self.train_name))

        self.model.load(state_dict)
        self.dataset.load()
        self.model.eval()

        description = lambda : "Prediction : Metric ("+" ".join(["{} : {:.4f}".format(name, network.measure.getLastValue()) for name, network in self.model.getNetworks().items()])+") "+gpuInfo(self.device)

        with tqdm.tqdm(iterable = enumerate(self.dataloader_prediction), desc = description(), total=len(self.dataloader_prediction)) as batch_iter:
            for _, data_dict in batch_iter:
                out = self.model(data_dict)
                self.model.measureUpdate(out, data_dict)
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
        