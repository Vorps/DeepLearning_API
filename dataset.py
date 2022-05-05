import importlib
import os
from sklearn import datasets
import torch
from torch.utils import data
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
import tqdm
import numpy as np
from abc import ABC, abstractmethod

from DeepLearning_API import HDF5, config, memoryInfo, cpuInfo, memoryForecast, getMemory, TransformLoader, Transform

class Group:

    @config(None)
    def __init__(self,  pre_transforms : Dict[str, TransformLoader] = {"default:Normalize:Standardize:Unsqueeze:TensorCast:ResampleIsotropic:ResampleResize": TransformLoader()},
                        post_transforms : Dict[str, TransformLoader] = {"default:Normalize:Standardize:Unsqueeze:TensorCast:ResampleIsotropic:ResampleResize": TransformLoader()}) -> None:
        self._pre_transforms = pre_transforms
        self._post_transforms = post_transforms
        self.pre_transforms : List[Transform] = []
        self.post_transforms : List[Transform] = []
        self.loaded = False
        
    def load(self, name):
        if self._pre_transforms is not None:
            for classpath, transform in self._pre_transforms.items():
                self.pre_transforms.append(transform.getTransform(classpath, args =  "{}.Dataset.groups.{}.pre_transforms".format(os.environ["DEEP_LEARNING_API_ROOT"], name)))
        if self._post_transforms is not None:
            for classpath, transform in self._post_transforms.items():
                self.post_transforms.append(transform.getTransform(classpath, args = "{}.Dataset.groups.{}.post_transforms".format(os.environ["DEEP_LEARNING_API_ROOT"], name)))
        self.loaded = True

class DataSet(data.Dataset):
    def __init__(self, hdf5 : HDF5, index : List[int], groups : Dict[str, Group], patch_size : List[int], overlap : List[int], use_cache = True) -> None:

        self.data : dict[str, List[HDF5.Dataset]] = {}
        self.nb_dataset : List[int] = []
        self.nb_patch : List[int] = []

        self.cache : Dict[str, List[int]] = {}

        for group in groups:
            self.cache[group] = []
            if not groups[group].loaded:
                groups[group].load(group)
            _, self.mapping, self.data[group] = hdf5.getDatasets(group, index, patch_size, overlap, groups[group].pre_transforms, [], 0, {})
            
            self.nb_dataset.append(len(self.mapping))
            self.nb_patch.append(np.array([[len(dataset) for dataset in DataSet.getDatasetsFromIndex(self.data[group], index)] for index in self.mapping.values()]).flatten())
        assert len(np.unique(self.nb_dataset)) == 1
        assert len(np.unique([np.sum(nb_patch) for nb_patch in self.nb_patch])) == 1
        self.nb_dataset = self.nb_dataset[0]
        self.nb_patch = self.nb_patch[0]

        self.groups = groups
        self.patch_size = patch_size
        self.use_cache = use_cache

        self.map : dict[int, tuple[int, int]] = {}
        i = 0
        for x in range(self.nb_dataset):
            for y in range(self.nb_patch[x]):
                self.map[i] = (x,y)
                i += 1
    @staticmethod
    def getDatasetsFromIndex(data, indexs):
        result = []
        last_index = None
        for index in indexs:
            if last_index is not None:
                data = data[last_index]

            result.append(data[index])
            last_index = index + 1
        return result

    def load(self):
        if self.use_cache:
            description = lambda memory_init, i : "Caching : {} | {} | {}".format(memoryInfo(), memoryForecast(memory_init, i, self.nb_dataset), cpuInfo())
            memory_init = getMemory()

            desc = description(memory_init, 0)
            with tqdm.tqdm(range(self.nb_dataset), desc=desc) as progressbar:
                for index in progressbar:
                    for group in self.groups:
                        self.loadData(group, index)

                    progressbar.set_description(description(memory_init, index))


    def loadData(self, group : str, index : int) -> None:
        self.cache[group].append(index)
        datasets = DataSet.getDatasetsFromIndex(self.data[group], self.mapping[index])
        for dataset in datasets:
            dataset.load(self.groups[group].pre_transforms)
    
    def unloadData(self, group : str, index : int) -> None:
        return self.data[group][index].unload()

    def getMap(self, index):
        return self.map[index][0], self.map[index][1]

    def __len__(self) -> int:
        return len(self.map)

    def getData(self, x):
        data = {}
        for group in self.groups:
            datasets = DataSet.getDatasetsFromIndex(self.data[group], self.mapping[x])
            for i, dataset in enumerate(datasets):
                if not dataset.loaded:
                    if len(self.cache[group]) > 5:
                        self.unloadData(group, self.cache[group].pop(0))
                    self.loadData(group, x)
                data_tmp = dataset.data
                for transformFunction in self.groups[group].post_transforms:
                    data_tmp = transformFunction.loadDataset(dataset)(data_tmp)
                data["{}_{}".format(group, i)] = data_tmp
        return data

                
    def __getitem__(self, index : int) -> Dict[str, torch.Tensor]:
        data = {}
        for group in self.groups:
            x, p = self.getMap(index)
            datasets = DataSet.getDatasetsFromIndex(self.data[group], self.mapping[x])
            for i, dataset in enumerate(datasets):
                if not dataset.loaded:
                    if len(self.cache[group]) > 5:
                        self.unloadData(group, self.cache[group].pop(0))
                    self.loadData(group, x)
                data_tmp = dataset.getData(self.patch_size, p)
                for transformFunction in self.groups[group].post_transforms:
                    data_tmp = transformFunction.loadDataset(dataset)(data_tmp)
                data["{}_{}".format(group, i)] = data_tmp
        return data

class Data(ABC):
    
    @config("Dataset")
    def __init__(self,  dataset_filename : str = "default", 
                        groups : Dict[str, Group] = {"default" : Group()},
                        patch_size : List[int] = [128, 128, 128],
                        overlap : List[int] = None,
                        use_cache : bool = True,
                        subset : List[int] = None,
                        num_workers : int = 4,
                        pin_memory : bool = True,
                        batch_size : int = 1,
                        shuffle : bool = True) -> None:
        self.dataset_filename = dataset_filename
        self.subset = subset
        self.groups = groups
        self.dataSet_args = dict(groups = groups, patch_size = np.asarray(patch_size), overlap = overlap, use_cache = use_cache)
        self.dataLoader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle)

    def __enter__(self):
        self.hdf5 = HDF5(self.dataset_filename)
        self.hdf5.__enter__()
        sizes = []
        for group in self.groups:
            sizes.append(self.hdf5.getSize(group))
        assert len(np.unique(sizes)) == 1
        if self.subset is None:
            self.subset = [0, sizes[0]]
        return self
    
    def __exit__(self, type, value, traceback):
        self.hdf5.__exit__(type, value, traceback)
    
    @abstractmethod
    def getData(self) -> object:
        pass

    @abstractmethod
    def load(self) -> None:
        pass
    
class DataTrain(Data):

    @config("Dataset")
    def __init__(self,  dataset_filename : str = "default", 
                        groups : Dict[str, Group] = {"default" : Group()},
                        patch_size : List[int] = [128, 128, 128],
                        overlap : List[int] = None,
                        use_cache : bool = True,
                        subset : List[int] = None,
                        num_workers : int = 4,
                        pin_memory : bool = True,
                        batch_size : int = 1,
                        shuffle : bool = True,
                        train_size : float = 0.8) -> None:
        super().__init__(dataset_filename, groups, patch_size, overlap, use_cache, subset, num_workers, pin_memory, batch_size, shuffle)
        self.train_test_split_args = dict(train_size=train_size, shuffle=shuffle)

    def getData(self, random_state) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        if self.train_test_split_args["train_size"] < 1:
            index_train, index_valid = train_test_split(range(self.subset[0], self.subset[1]), random_state=random_state, **self.train_test_split_args)
            self.dataset_train = DataSet(self.hdf5, index = index_train, **self.dataSet_args)
            self.dataset_validation = DataSet(self.hdf5, index = index_valid, **self.dataSet_args)
            return torch.utils.data.DataLoader(dataset=self.dataset_train, **self.dataLoader_args), torch.utils.data.DataLoader(dataset=self.dataset_validation, **self.dataLoader_args)
        else:
            self.dataset_train = DataSet(self.hdf5, index = range(self.subset[0], self.subset[1]), **self.dataSet_args)
            self.dataset_validation = None
            return torch.utils.data.DataLoader(dataset=self.dataset_train, **self.dataLoader_args), None

    def load(self) -> None:
        if self.dataset_train is not None:
            self.dataset_train.load()
        if self.dataset_validation is not None:
            self.dataset_validation.load()

class DataPrediction(Data):

    @config("Dataset")
    def __init__(self,  dataset_filename : str = "Dataset.h5", 
                        groups : Dict[str, Group] = {"default" : Group()},
                        patch_size : List[int] = [128, 128, 128],
                        overlap : List[int] = None,
                        use_cache : bool = True,
                        subset : List[int] = None,
                        num_workers : int = 4,
                        pin_memory : bool = True,
                        batch_size : int = 1) -> None:

        super().__init__(dataset_filename, groups, patch_size, overlap, use_cache, subset, num_workers, pin_memory, batch_size, False)
        
    def getData(self) -> torch.utils.data.DataLoader:
        self.dataset_prediction = DataSet(self.hdf5, index = range(self.subset[0], self.subset[1]), **self.dataSet_args)
        return torch.utils.data.DataLoader(dataset=self.dataset_prediction, **self.dataLoader_args)

    def load(self) -> None:
        self.dataset_prediction.load()