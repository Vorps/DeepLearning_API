import importlib
import torch
from torch.utils import data
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
import tqdm

from . import HDF5
from . import config
from . import Transform, Standardize

class TransformLoader:

    @config("Dataset.Param")
    def __init__(self, module : str = "DeepLearning_API.transform") -> None:
        self.module = module
    
    def getTransform(self, name : str, args : str):
        return getattr(importlib.import_module(self.module), name)(config = "Config.yml", args = args)

class Param:

    @config("Dataset.Param")
    def __init__(self, transformFunctions : Dict[str, TransformLoader] = {"Normalize": TransformLoader(), "Unsqueeze": TransformLoader()}) -> None:
        self.transformFunctions = transformFunctions
        self.loaded = False
        
    def load(self, name):
        for key in self.transformFunctions:
            self.transformFunctions[key] = self.transformFunctions[key].getTransform(key, args = "Dataset.params."+name+".transformFunctions."+key)
        self.loaded = True


class DataSet(data.Dataset):
    def __init__(self, hdf5 : HDF5, index : List[int], params : Dict[str, Param], nb_cut : int, use_cache = True) -> None:

        self.data = {}
        for param in params:
            if not params[param].loaded:
                params[param].load(param)
            self.data[param] = hdf5.getDatasets(param, index)
        
        self.size = len(index)
        
        self.params = params
        self.nb_cut = nb_cut
        self.use_cache = use_cache

        if self.use_cache:
            self.load()
    
    def load(self):
        with tqdm.tqdm(range(len(self)), desc='Caching') as progressbar:
            for index in progressbar:
                for group in self.params:
                    self.loadData(group, index)
    
    def loadData(self, group : str, index : int) -> None:
        self.data[group][index].read(index % self.nb_cut, self.nb_cut)
        for _, transformFunction in self.params[group].transformFunctions.items():
            self.data[group][index].data = transformFunction(self.data[group][index].data)

    def __len__(self) -> int:
        return self.size*self.nb_cut

    def __getitem__(self, index : int) -> Dict[str, torch.Tensor]:
        data = {}
        for group in self.params:
            value = self.data[group][index]
            if not value.loaded:
                self.loadData(group, index)
            data[group] = value.data
        return data


class Data:
    
    @config("Dataset")
    def __init__(self,  filename : str = "Dataset.h5", 
                        params : Dict[str, Param] = {"group" : Param()},
                        seed : int = 45,
                        shuffle : bool = True,
                        use_cache : bool = True,
                        nb_cut : int = 1,
                        train_size : float = 0.8,
                        batch_size : int = 1,
                        num_workers : int = 4,
                        pin_memory : bool = True) -> None:
        self.filename = filename
        self.train_test_split_args = dict(random_state=seed, train_size=train_size, shuffle=shuffle)
        self.segmentationDataSet_args = dict(params = params, nb_cut = nb_cut, use_cache = use_cache)
        self.dataLoader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle)

    def __enter__(self):
        self.hdf5 = HDF5(self.filename)
        self.hdf5.__enter__()
        return self
    
    def __exit__(self, type, value, traceback):
        self.hdf5.__exit__(type, value, traceback)

    def getData(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        index_train, index_valid = train_test_split(range(self.hdf5.size), **self.train_test_split_args)
        dataset_train = DataSet(self.hdf5, index = index_train, **self.segmentationDataSet_args)
        dataset_valid = DataSet(self.hdf5, index = index_valid, **self.segmentationDataSet_args)
        return torch.utils.data.DataLoader(dataset=dataset_train, **self.dataLoader_args), torch.utils.data.DataLoader(dataset=dataset_valid, **self.dataLoader_args)
