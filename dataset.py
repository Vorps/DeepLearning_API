import math
import os
import random
import torch
from torch.utils import data
from typing import List, Optional, Tuple, Dict, Union
from sklearn.model_selection import train_test_split
import tqdm
import numpy as np
from abc import ABC, abstractmethod

from DeepLearning_API.HDF5 import HDF5, Patch, Dataset
from DeepLearning_API.config import config
from DeepLearning_API.utils import memoryInfo, cpuInfo, memoryForecast, getMemory, NeedDevice
from DeepLearning_API.transform import TransformLoader, Transform
from DeepLearning_API.augmentation import DataAugmentationLoader, DataAugmentation
from torch.utils.data import DataLoader

class Group:

    @config(None)
    def __init__(self,  pre_transforms : Dict[str, TransformLoader] = {"default:Normalize:Standardize:Unsqueeze:TensorCast:ResampleIsotropic:ResampleResize": TransformLoader()},
                        post_transforms : Dict[str, TransformLoader] = {"default:Normalize:Standardize:Unsqueeze:TensorCast:ResampleIsotropic:ResampleResize": TransformLoader()}) -> None:
        self._pre_transforms = pre_transforms
        self._post_transforms = post_transforms
        self.pre_transforms : List[Transform] = []
        self.post_transforms : List[Transform] = []
        
    def load(self, name : str, device : torch.device):
        if self._pre_transforms is not None:
            for classpath, transform in self._pre_transforms.items():
                transform = transform.getTransform(classpath, DL_args =  "{}.Dataset.groups.{}.pre_transforms".format(os.environ["DEEP_LEARNING_API_ROOT"], name))
                transform.setDevice(device)
                self.pre_transforms.append(transform)

        if self._post_transforms is not None:
            for classpath, transform in self._post_transforms.items():
                transform = transform.getTransform(classpath, DL_args = "{}.Dataset.groups.{}.post_transforms".format(os.environ["DEEP_LEARNING_API_ROOT"], name))
                transform.setDevice(device)
                self.post_transforms.append(transform)

class DataSet(data.Dataset):

    def __init__(self, hdf5 : HDF5, index : List[int], groups : Dict[str, Group], dataAugmentations : List[DataAugmentation], patch : Patch, use_cache = True) -> None:
        self.data : dict[str, List[Dataset]] = {}
        self.dataAugmentations = dataAugmentations
        self.groups = groups

        nb_dataset_list = []
        nb_patch_list = []
        for group in self.groups:
            _, self.mapping, self.data[group] = hdf5.getDatasets(group, index, patch, self.groups[group].pre_transforms, [], 0, {})
            nb_dataset_list.append(len(self.mapping))
            nb_patch_list.append(np.array([[len(dataset) for dataset in DataSet.getDatasetsFromIndex(self.data[group], index)] for index in self.mapping.values()]).flatten())
        
        self.nb_dataset = nb_dataset_list[0]
        self.nb_patch = nb_patch_list[0]
        self.nb_augmentation = np.max([int(np.sum([data_augmentation.nb for data_augmentation in self.dataAugmentations])), 1])
        
        self.use_cache = use_cache

        self.map : dict[int, tuple[int, int, int]] = {}
        i = 0
        for x in range(self.nb_dataset):
            for y in range(self.nb_patch[x]):
                for z in range(self.nb_augmentation):
                    self.map[i] = (x,y,z)
                    i += 1

    @staticmethod
    def getDatasetsFromIndex(data, indexs) -> List[Dataset]:
        result : List[Dataset] = []
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
        datasets = DataSet.getDatasetsFromIndex(self.data[group], self.mapping[index])
        for dataset in datasets:
            dataset.load(index, self.groups[group].pre_transforms, self.dataAugmentations)
        

    def unloadData(self, group : str, index : int) -> None:
        return self.data[group][index].unload()

    def getMap(self, index) -> Tuple[int, int, int]:
        return self.map[index]

    def __len__(self) -> int:
        return len(self.map)

    def getData(self, x):
        data = {}
        for group in self.groups:
            datasets = DataSet.getDatasetsFromIndex(self.data[group], self.mapping[x])
            for i, dataset in enumerate(datasets):
                if not dataset.loaded:
                    self.loadData(group, x)
                data["{}".format(group) if len(datasets) == 1 else "{}_{}".format(group, i)] = dataset.data
        return data

    def __getitem__(self, index : int) -> Dict[str, torch.Tensor]:
        data = {}
        for group in self.groups:
            x, p, a = self.getMap(index)
            datasets = DataSet.getDatasetsFromIndex(self.data[group], self.mapping[x])
            for i, dataset in enumerate(datasets):
                if not dataset.loaded:
                    self.loadData(group, x)
                data["{}".format(group) if len(datasets) == 1 else "{}_{}".format(group, i)] = dataset.getData(p, a, self.groups[group].post_transforms)
        return data

class Data(NeedDevice, ABC):
    
    @config("Dataset")
    def __init__(self,  dataset_filename : str = "default", 
                        groups : Dict[str, Group] = {"default" : Group()},
                        patch : Patch = Patch(),
                        use_cache : bool = True,
                        subset : Optional[List[int]] = None,
                        num_workers : int = 4,
                        pin_memory : bool = True,
                        batch_size : int = 1,
                        shuffle : bool = True) -> None:
        self.dataset_filename = dataset_filename
        self.subset = subset
        self.groups = groups

        self.dataSet_args = dict(groups=self.groups, dataAugmentations = [], patch=patch, use_cache = use_cache)
        self.dataLoader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle)

    def __enter__(self):
        self.hdf5 = HDF5(self.dataset_filename)
        self.hdf5.__enter__()
        if not self.subset:
            self.subset = [0, len(self)]
        return self
    
    def __exit__(self, type, value, traceback):
        self.hdf5.__exit__(type, value, traceback)
    
    def __len__(self) -> int:
        sizes = []
        for group in self.groups:
            sizes.append(self.hdf5.getSize(group))
        assert len(np.unique(sizes)) == 1
        return sizes[0]

    @abstractmethod
    def getData(self) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
        for group in self.groups:
            assert self.device, "No device set"
            self.groups[group].load(group, self.device)

    @abstractmethod
    def load(self) -> None:
        pass
    
class DataTrain(Data):

    @config("Dataset")
    def __init__(self,  dataset_filename : str = "default", 
                        groups : Dict[str, Group] = {"default" : Group()},
                        augmentations : Dict[str, DataAugmentationLoader] = {"default:RandomGeometricTransform" : DataAugmentationLoader()},
                        patch : Patch = Patch(),
                        use_cache : bool = True,
                        subset : Optional[List[int]] = None,
                        num_workers : int = 4,
                        pin_memory : bool = True,
                        batch_size : int = 1,
                        shuffle : bool = True,
                        train_size : float = 0.8) -> None:
        super().__init__(dataset_filename, groups, patch, use_cache, subset, num_workers, pin_memory, batch_size, shuffle)
        self.augmentations = augmentations
        self.train_test_split_args = dict(train_size=train_size, shuffle=shuffle)

    def getData(self, random_state : Optional[int]) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
        super().getData()
        assert self.subset
        self.dataAugmentations : List[DataAugmentation] = []
        if self.augmentations is not None:
            for name, data_augmentation in self.augmentations.items():
                self.dataAugmentations.append(data_augmentation.getDataAugmentation(name))
        self.dataSet_args.update(dataAugmentations=self.dataAugmentations)
        
        for dataAugmentation in self.dataAugmentations:
            assert self.device, "No device set"
            dataAugmentation.setDevice(self.device)

        if self.train_test_split_args["train_size"] < 1:
            if self.train_test_split_args["shuffle"]:
                index_train, index_valid = train_test_split(range(0, len(self)), random_state=random_state, **self.train_test_split_args)
                nb = self.subset[1]-self.subset[0]
                index_train = index_train[:int(math.ceil(nb*self.train_test_split_args["train_size"]))]
                index_valid = index_valid[:int(math.ceil(nb*(1-self.train_test_split_args["train_size"])))]
            else:
                index_train, index_valid = train_test_split(range(self.subset[0], self.subset[1]), random_state=random_state, **self.train_test_split_args)

            self.dataset_train = DataSet(self.hdf5, index = index_train, **self.dataSet_args) # type: ignore
            self.dataset_validation = DataSet(self.hdf5, index = index_valid, **self.dataSet_args) # type: ignore
            return DataLoader(dataset=self.dataset_train, **self.dataLoader_args), DataLoader(dataset=self.dataset_validation, **self.dataLoader_args)
        else:
            if self.train_test_split_args["shuffle"]:
                self.dataset_train = DataSet(self.hdf5, index = random.choices(list(range(0, len(self))), k=self.subset[1]-self.subset[0]), **self.dataSet_args) # type: ignore
            else:
                self.dataset_train = DataSet(self.hdf5, index = list(range(self.subset[0], self.subset[1])), **self.dataSet_args) # type: ignore
            self.dataset_validation = None
            return DataLoader(dataset=self.dataset_train, **self.dataLoader_args)

    def load(self) -> None:
        if self.dataset_train is not None:
            self.dataset_train.load()
        if self.dataset_validation is not None:
            self.dataset_validation.load()

class DataPrediction(Data):

    @config("Dataset")
    def __init__(self,  dataset_filename : str = "Dataset.h5", 
                        groups : Dict[str, Group] = {"default" : Group()},
                        patch : Patch = Patch(),
                        use_cache : bool = True,
                        subset : Optional[List[int]] = None,
                        num_workers : int = 4,
                        pin_memory : bool = True,
                        batch_size : int = 1) -> None:

        super().__init__(dataset_filename, groups, patch, use_cache, subset, num_workers, pin_memory, batch_size, False)
        
    def getData(self) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
        super().getData()
        assert self.subset
        self.dataset_prediction = DataSet(self.hdf5, index = list(range(self.subset[0], self.subset[1])), **self.dataSet_args)  # type: ignore
        return DataLoader(dataset=self.dataset_prediction, **self.dataLoader_args)

    def load(self) -> None:
        self.dataset_prediction.load()