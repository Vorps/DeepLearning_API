from DeepLearning_API.config import config
import os
from DeepLearning_API import METRICS_DIRECTORY, PREDICTIONS_DIRECTORY
import shutil
import builtins
import importlib
from DeepLearning_API.utils import _getModule
from DeepLearning_API.dataset import DataMetric
import torch
import tqdm
import numpy as np
import json

class CriterionsAttr():

    @config()
    def __init__(self) -> None:
        pass  

class CriterionsLoader():

    @config()
    def __init__(self, criterionsLoader: dict[str, CriterionsAttr] = {"default:torch_nn_CrossEntropyLoss:Dice:NCC": CriterionsAttr()}) -> None:
        self.criterionsLoader = criterionsLoader

    def getCriterions(self, output_group : str, target_group : str) -> dict[torch.nn.Module, CriterionsAttr]:
        criterions = {}
        for module_classpath, criterionsAttr in self.criterionsLoader.items():
            module, name = _getModule(module_classpath, "measure")
            criterions[config("{}.metrics.{}.targetsCriterions.{}.criterionsLoader.{}".format(os.environ["DEEP_LEARNING_API_ROOT"], output_group, target_group, module_classpath))(getattr(importlib.import_module(module), name))(config = None)] = criterionsAttr
        return criterions

class TargetCriterionsLoader():

    @config()
    def __init__(self, targetsCriterions : dict[str, CriterionsLoader] = {"default" : CriterionsLoader()}) -> None:
        self.targetsCriterions = targetsCriterions
        
    def getTargetsCriterions(self, output_group : str) -> dict[str, dict[torch.nn.Module, float]]:
        targetsCriterions = {}
        for target_group, criterionsLoader in self.targetsCriterions.items():
            targetsCriterions[target_group] = criterionsLoader.getCriterions(output_group, target_group)
        return targetsCriterions

class Statistics():

    def __init__(self, filename: str) -> None:
        self.measures: dict[str, list[float]] = {}
        self.filename = filename

    def add(self, values: dict[str, float]) -> None:
        for name, value in values.items():
            if name in self.measures:
                self.measures[name].append(value)
            else:
                self.measures[name] = [value]

    def getStatistic(self, values: list[float]):
        return {"max": np.max(values), "min": np.min(values), "std": np.std(values), "25pc": np.percentile(values, 25), "50pc": np.percentile(values, 50), "75pc": np.percentile(values, 75), "mean": np.mean(values), "count": len(values)}
        
    def write(self) -> str:
        result = {}
        result["case"] = {k : {str(i) : u for i, u in enumerate(v)} for k, v in self.measures.items()}
        result["aggregates"] = {k : self.getStatistic(v) for k, v in self.measures.items()}
        with open(self.filename, "w") as f:
            f.write(json.dumps(result, indent=4))

class Metric():

    @config("Metric")
    def __init__(self, train_name: str = "default:name", metrics: dict[str, TargetCriterionsLoader] = {"default": TargetCriterionsLoader()}, dataset : DataMetric = DataMetric(),) -> None:
        if os.environ["DEEP_LEANING_API_CONFIG_MODE"] != "Done":
            exit(0)

        self.train_name = train_name
        self.metric_path = METRICS_DIRECTORY()+self.train_name+"/"
        self.predict_path = PREDICTIONS_DIRECTORY()+self.train_name+"/"
        self.metricsLoader = metrics
        self.dataset = dataset
        self.metrics = {k: v.getTargetsCriterions(k) for k, v in self.metricsLoader.items()}
        self.statistics = Statistics(self.metric_path+"Metric.json")

    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        pass
    
    def update(self, data_dict: dict[str, torch.Tensor]) -> dict[str, float]:
        result = {}
        for output_group in self.metrics:
            for target_group in self.metrics[output_group]:
                for metric in self.metrics[output_group][target_group]:
                    result["{}:{}:{}".format(output_group, target_group, metric.__class__.__name__)] = metric(data_dict[output_group], data_dict[target_group]).item()
        self.statistics.add(result)
        return result

    def measure(self):
        if os.path.exists(self.metric_path):
            if os.environ["DL_API_OVERWRITE"] != "True":
                accept = builtins.input("The metric {} already exists ! Do you want to overwrite it (yes,no) : ".format(self.train_name))
                if accept != "yes":
                    return
           
            if os.path.exists(self.metric_path):
                shutil.rmtree(self.metric_path) 

        if not os.path.exists(self.metric_path):
            os.makedirs(self.metric_path)

        self.dataloader = self.dataset.getData(1)[0][0]
        self.dataloader.dataset.to(torch.device("cpu"))
        self.dataloader.dataset.load()
        
        description = lambda measure : "Metric : {} ".format(" | ".join("{}: {:.2f}".format(k, v) for k, v in measure.items()) if measure is not None else "")
        with tqdm.tqdm(iterable = enumerate(self.dataloader), desc = description(None), total=len(self.dataloader)) as batch_iter:
            for _, data_dict in batch_iter:
                batch_iter.set_description(description(self.update({k: v[0] for k,v in data_dict.items()})))
        self.statistics.write()