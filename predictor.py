import builtins
import importlib
import shutil
from typing import Dict, List, Tuple
import torch
import tqdm
import os

from DeepLearning_API import config, DataPrediction, MODELS_DIRECTORY, PREDICTIONS_DIRECTORY, CONFIG_FILE, gpuInfo, getDevice, DatasetUtils, TransformLoader, _getModule, DataSet
import pynvml

class Metric():

    @config(None)
    def __init__(self, group : str = "Default", l : float = 1) -> None:
        self.l = l
        self.group = group

    def getMetric(self, classpath : str, group : str) -> torch.nn.Module:
        module, name = _getModule(classpath, "metric")
        return config("Predictor.metrics."+group+".metric."+classpath)(getattr(importlib.import_module(module), name))(config = None)

class Metrics():

    @config(None)
    def __init__(self, metrics : Dict[str, Metric] = {"default:Dice" : Metric()}) -> None:
        self.metrics = metrics

    def getMetrics(self, group : str) -> Dict[str, torch.nn.Module]:
        for key in self.metrics:
            self.metrics[key] = (self.metrics[key].group, self.metrics[key].l, self.metrics[key].getMetric(key, group))
        return self.metrics

class Predictor():

    @config("Predictor")
    def __init__(self, 
                    dataset : DataPrediction = DataPrediction(),
                    groupsInput : List[str] = ["default"],
                    train_name : str = "name",
                    device : int = None,
                    metrics: Dict[str, Metrics] = {"default" : Metrics()},
                    patch_transform : Dict[str, TransformLoader] = {"default:GausianImportance"},
                    transform : Dict[str, TransformLoader] = {"default:ResampleResizeForDataset:ArgMax:NConnexeLabel:TensorCast": TransformLoader()}) -> None:
        if os.environ["DEEP_LEANING_API_CONFIG_MODE"] != "Done":
            exit(0)
        torch.backends.cudnn.benchmark = True
        
        self.device = getDevice(device)
        self.train_name = train_name
        self.dataset = dataset
        self.groupsInput = groupsInput
        self.metrics = metrics
        for classpath in self.metrics:
            self.metrics[classpath] = self.metrics[classpath].getMetrics(classpath)

        self.patch_transform = patch_transform
        self.transform = transform
        if self.patch_transform is not None:
            for key in self.patch_transform:
                self.patch_transform[key] = self.patch_transform[key].getTransform(key, args = "Predictor.patch_transform."+key)
        if self.transform is not None:
            for key in self.transform:
                self.transform[key] = self.transform[key].getTransform(key, args = "Predictor.transform."+key)

        path = MODELS_DIRECTORY()+self.train_name+"/"
        if os.path.exists(path) and os.listdir(path):
            name = sorted(os.listdir(path))[-1]
            self.model = torch.load(path+name).to(self.device)
        else:
            raise Exception("Model : {} does not exist !".format(self.train_name))

    def __enter__(self) -> None:
        pynvml.nvmlInit()
        self.dataset.__enter__()
        self.dataloader_prediction = self.dataset.getData()
        return self
    
    def __exit__(self, type, value, traceback) -> None:
        self.dataset.__exit__(type, value, traceback)
        pynvml.nvmlShutdown()

    def getInput(self, data_dict : Dict[str, List[torch.Tensor]]) -> torch.Tensor:    
        input = torch.cat([data_dict[group] for group in self.groupsInput], dim=0)
        return torch.unsqueeze(input, dim=0)
        
    def _metric(self, out_dict : torch.Tensor, data_dict : Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        value = 0
        values = dict()
        for group in self.metrics:
            output = out_dict[group] if group in out_dict else None
            for true_group, l, metric in self.metrics[group].values():
                target = data_dict[true_group] if true_group in data_dict else None
                result = metric(output, target)
                values[group+":"+metric.__class__.__name__] = result.item()
                value = value + l*result
        return value, values

    def _combine_patch(self, out_patch_accu, n_channel, x):
        out_dataset = DataSet.getDatasetsFromIndex(self.dataloader_prediction.dataset.data[self.groupsInput[0].replace("_0", "")], self.dataloader_prediction.dataset.mapping[x])[0]
        out_data = torch.zeros([n_channel]+list(out_dataset.getShape()))

        for index, patch_data in enumerate(out_patch_accu):
            if self.patch_transform is not None:
                for patch_transformFunction in self.patch_transform.values():
                    patch_data = patch_transformFunction.loadDataset(out_dataset)(patch_data)
            src_slices = tuple([slice(n_channel)]+[slice(slice_tmp.stop-slice_tmp.start) for slice_tmp in out_dataset.patch_slices[index]])
            dest_slices = tuple([slice(n_channel)]+list(out_dataset.patch_slices[index]))
            out_data[dest_slices] += patch_data[src_slices]
        return out_data

    @torch.no_grad()
    def predict(self) -> None:
        config_file_dest = PREDICTIONS_DIRECTORY()+self.train_name+"/"+self.dataset.dataset_filename.split("/")[-1].replace(".h5", "_")+self.groupsInput[0].replace("/", "_")+".yml"

        if os.path.exists(config_file_dest):
            accept = builtins.input("The prediction {} {} already exists ! Do you want to overwrite it (yes,no) : ".format(self.train_name, self.dataset.dataset_filename.split("/")[-1]))
            if accept != "yes":
                return
            else:
                if os.path.exists(config_file_dest):
                    os.remove(config_file_dest)
        if not os.path.exists(PREDICTIONS_DIRECTORY()+self.train_name+"/"):
            os.makedirs(PREDICTIONS_DIRECTORY()+self.train_name+"/")
        shutil.copyfile(CONFIG_FILE(), config_file_dest)
        self.dataset.load()
        self.model.eval()
        out_patch_accu : Dict[str, List[torch.Tensor]]= {}
        out_data : Dict[str, torch.Tensor] = {}
        it = 0

        description = lambda loss : "Prediction : (Metric {:.4f}) ".format(loss)+gpuInfo(self.device)
        with tqdm.tqdm(iterable = enumerate(self.dataloader_prediction), desc = description(0), total=len(self.dataloader_prediction)) as batch_iter:
            for _, data_dict in batch_iter:
                
                input = self.getInput(data_dict)
                out = self.model(input.to(self.device))
                for batch in range(input.shape[0]):
                    for group in out:                
                        if group not in out_patch_accu:
                            out_patch_accu[group] = []
                        out_patch_accu[group].append(self.model.getOutputModel(group, out[group][batch]).cpu())

                        x, idx = self.dataloader_prediction.dataset.getMap(it)
                        if idx == self.dataloader_prediction.dataset.nb_patch[x]-1:
                            out_data[group] = self._combine_patch(out_patch_accu[group], out[group].shape[1], x)               
                            del out_patch_accu[group]
                    
                    if len(out_data) == len(out):
                        if self.metrics is not None:
                            value, values = self._metric(out_data, self.dataloader_prediction.dataset.getData(x))
                        
                        """if self.transform is not None:
                            for transformFunction in self.transform.values():
                                out_data = transformFunction.loadDataset(out_dataset)(out_data)
                        out_data = out_data.numpy()"""
                        out_dataset = DataSet.getDatasetsFromIndex(self.dataloader_prediction.dataset.data[self.groupsInput[0].replace("_0", "")], self.dataloader_prediction.dataset.mapping[x])[0]
                        with DatasetUtils(PREDICTIONS_DIRECTORY()+self.train_name+"/"+self.dataset.dataset_filename.split("/")[-1], read=False) as datasetUtils:
                            for group in out_data:
                                datasetUtils.writeImage(group, out_dataset.name, out_dataset.to_image(out_data[group].numpy()))
                                if self.metrics is not None:
                                    datasetUtils.h5.attrs["value"] = value
                                    print(value)
                                    print(values)
                                    for name, value in values.items():
                                        datasetUtils.h5.attrs[name] = value
                        out_data.clear()

                    it += 1
                
                batch_iter.set_description(description(0))
        