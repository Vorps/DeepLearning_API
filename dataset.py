import math
import os
import random
import torch
from torch.utils import data
from sklearn.model_selection import train_test_split
import tqdm
import numpy as np
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

from DeepLearning_API.HDF5 import HDF5, DatasetPatch, Dataset
from DeepLearning_API.config import config
from DeepLearning_API.utils import memoryInfo, cpuInfo, memoryForecast, getMemory, NeedDevice
from DeepLearning_API.transform import TransformLoader, Transform
from DeepLearning_API.augmentation import DataAugmentationsList


class GroupTransform:

    @config()
    def __init__(self,  pre_transforms : dict[str, TransformLoader] | list[Transform] = {"default:Normalize:Standardize:Unsqueeze:TensorCast:ResampleIsotropic:ResampleResize": TransformLoader()},
                        post_transforms : dict[str, TransformLoader]| list[Transform] = {"default:Normalize:Standardize:Unsqueeze:TensorCast:ResampleIsotropic:ResampleResize": TransformLoader()}) -> None:
        self._pre_transforms = pre_transforms
        self._post_transforms = post_transforms
        self.pre_transforms : list[Transform] = []
        self.post_transforms : list[Transform] = []
        
    def load(self, group_src : str, group_dest : str, device : torch.device):
        if self._pre_transforms is not None:
            if isinstance(self._pre_transforms, dict):
                for classpath, transform in self._pre_transforms.items():
                    transform = transform.getTransform(classpath, DL_args =  "{}.Dataset.groups_src.{}.groups_dest.{}.pre_transforms".format(os.environ["DEEP_LEARNING_API_ROOT"], group_src, group_dest))
                    transform.setDevice(device)
                    self.pre_transforms.append(transform)
            else:
                for transform in self._pre_transforms:
                    transform.setDevice(device)
                    self.pre_transforms.append(transform)

        if self._post_transforms is not None:
            if isinstance(self._post_transforms, dict):
                for classpath, transform in self._post_transforms.items():
                    transform = transform.getTransform(classpath, DL_args = "{}.Dataset.groups_src.{}.groups_dest.{}.post_transforms".format(os.environ["DEEP_LEARNING_API_ROOT"], group_src, group_dest))
                    transform.setDevice(device)
                    self.post_transforms.append(transform)
            else:
                for transform in self._post_transforms:
                    transform.setDevice(device)
                    self.post_transforms.append(transform)

class Group(dict):

    @config()
    def __init__(self, groups_dest: dict[str, GroupTransform] = {"default": GroupTransform()}):
        super().__init__(groups_dest)


class DataSet(data.Dataset):

    def __init__(self, hdf5 : HDF5, index : list[int], groups_src : dict[str, Group], dataAugmentationsList : list[DataAugmentationsList], patch : DatasetPatch | None, use_cache = True) -> None:
        self.data : dict[str, list[Dataset]] = {}
        self.cache : dict[str, set[int]] = {}

        self.dataAugmentationsList = dataAugmentationsList
        self.groups_src = groups_src

        nb_dataset_list = []
        nb_patch_list = []
        self.mapping: dict[int, list[int]]= {}
        for group_src in self.groups_src:
            datasets, mapping, _, _= hdf5.getDatasets(group_src, index)
            self.mapping = {i : l for i, l in enumerate(mapping)}
            nb_dataset_list.append(len(self.mapping))
            
            for group_dest in self.groups_src[group_src]:
                self.cache[group_dest] = set()
                self.data[group_dest] = [Dataset(group_dest, dataset, patch = patch, pre_transforms = self.groups_src[group_src][group_dest].pre_transforms) for dataset in datasets]
                nb_patch_list.append([len(dataset) for dataset in self.data[group_dest]])
        
        self.nb_dataset = nb_dataset_list[0]
        self.nb_patch = nb_patch_list[0]

        self.nb_augmentation = np.max([int(np.sum([data_augmentation.nb for data_augmentation in self.dataAugmentationsList])), 1])
        
        self.use_cache = use_cache

        self.map : dict[int, tuple[int, int, int]] = {}
        i = 0
        for x in range(self.nb_dataset):
            for y in range(self.nb_patch[x]):
                for z in range(self.nb_augmentation):
                    self.map[i] = (x,y,z)
                    i += 1

    def getDatasetsFromIndex(self, group_dest: str, indexs: list[int]) -> list[Dataset]:
        data = self.data[group_dest]
        result : list[Dataset] = []
        for index in indexs:
            result.append(data[index])
        return result

    def load(self):
        if self.use_cache:
            description = lambda memory_init, i : "Caching : {} | {} | {}".format(memoryInfo(), memoryForecast(memory_init, i, self.nb_dataset), cpuInfo())
            memory_init = getMemory()

            desc = description(memory_init, 0)
            with tqdm.tqdm(range(self.nb_dataset), desc=desc) as progressbar:
                for index in progressbar:
                    for group_src in self.groups_src:
                        for group_dest in self.groups_src[group_src]:
                            self.loadData(group_src, group_dest, index)

                    progressbar.set_description(description(memory_init, index))

    def loadData(self, group_src: str, group_dest : str, index : int) -> None:
        self.cache[group_dest].add(index)
        datasets = self.getDatasetsFromIndex(group_dest, self.mapping[index])
        for dataset in datasets:
            dataset.load(index, self.groups_src[group_src][group_dest].pre_transforms, self.dataAugmentationsList)

    def unloadData(self, group_dest : str, index : int) -> None:
        self.cache[group_dest]
        return self.data[group_dest][index].unload()

    def getMap(self, index) -> tuple[int, int, int]:
        return self.map[index]

    def __len__(self) -> int:
        return len(self.map)

    def __getitem__(self, index : int) -> dict[str, tuple[torch.Tensor, int, int, int]]:
        data = {}
        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                x, p, a = self.getMap(index)
                self.loadData(group_src, group_dest, x)
                if not self.use_cache and len(self.cache[group_dest]) > 50:
                    self.unloadData(group_dest, self.cache[group_dest].pop())

                datasets = self.getDatasetsFromIndex(group_dest, self.mapping[x])
                
                for i, dataset in enumerate(datasets):
                    data["{}".format(group_dest) if len(datasets) == 1 else "{}_{}".format(group_dest, i)] = (dataset.getData(p, a, self.groups_src[group_src][group_dest].post_transforms), x, p, a)
        return data

class Data(NeedDevice, ABC):
    
    @config("Dataset")
    def __init__(self,  dataset_filename : str = "default", 
                        groups_src : dict[str, Group] = {"default" : Group()},
                        patch : DatasetPatch | None = None,
                        use_cache : bool = True,
                        subset : list[int] | None = None,
                        num_workers : int = 4,
                        pin_memory : bool = True,
                        batch_size : int = 1,
                        shuffle : bool = True) -> None:
        self.dataset_filename = dataset_filename
        self.subset = subset
        self.groups_src = groups_src
        self.dataAugmentationsList: dict[str, DataAugmentationsList] = {}

        self.dataSet_args = dict(groups_src=self.groups_src, dataAugmentationsList = list(self.dataAugmentationsList.values()), patch=patch, use_cache = use_cache)
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
        for group in self.groups_src:
            sizes.append(self.hdf5.getSize(group))
        assert len(np.unique(sizes)) == 1
        return sizes[0]

    @abstractmethod
    def getData(self) -> tuple[DataLoader, DataLoader | None] | DataLoader:
        assert self.device, "No device set"
        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                self.groups_src[group_src][group_dest].load(group_src, group_dest, self.device)

        for key, dataAugmentations in self.dataAugmentationsList.items():
            dataAugmentations.load(key, self.device)

    @abstractmethod
    def load(self) -> None:
        pass
    
class DataTrain(Data):

    @config("Dataset")
    def __init__(self,  dataset_filename : str = "default", 
                        groups_src : dict[str, Group] = {"default" : Group()},
                        augmentations : dict[str, DataAugmentationsList] | None = {"DataAugmentation_0" : DataAugmentationsList()},
                        patch : DatasetPatch | None = DatasetPatch(),
                        use_cache : bool = True,
                        subset : list[int] | None = None,
                        num_workers : int = 4,
                        pin_memory : bool = True,
                        batch_size : int = 1,
                        shuffle : bool = True,
                        train_size : float = 0.8) -> None:
        super().__init__(dataset_filename, groups_src, patch, use_cache, subset, num_workers, pin_memory, batch_size, shuffle)
        self.dataAugmentationsList = augmentations if augmentations else {}
        self.train_test_split_args = dict(train_size=train_size, shuffle=shuffle)

    def getData(self, random_state : int | None) -> tuple[DataLoader, DataLoader | None] | DataLoader:
        super().getData()
        assert self.subset
        if len(self.subset) == 1:
            self.subset = [self.subset[0], self.subset[0]+1]
        nb = self.subset[1]-self.subset[0]+1

        self.dataSet_args.update(dataAugmentationsList=list(self.dataAugmentationsList.values()))
        
        if self.train_test_split_args["train_size"] < 1 and int(math.floor(nb*(1-self.train_test_split_args["train_size"]))) > 0:
            index_train, index_valid = train_test_split(range(self.subset[0], self.subset[1]), random_state=random_state, **self.train_test_split_args)
            self.dataset_train = DataSet(self.hdf5, index = index_train, **self.dataSet_args) # type: ignore
            self.dataset_validation = DataSet(self.hdf5, index = index_valid, **self.dataSet_args) # type: ignore
            return DataLoader(dataset=self.dataset_train, **self.dataLoader_args), DataLoader(dataset=self.dataset_validation, **self.dataLoader_args)
        else:
            if self.train_test_split_args["shuffle"]:
                self.dataset_train = DataSet(self.hdf5, index = random.sample(list(range(self.subset[0], self.subset[1])), self.subset[1]-self.subset[0]), **self.dataSet_args) # type: ignore
            else:
                self.dataset_train = DataSet(self.hdf5, index = list(range(self.subset[0], self.subset[1])), **self.dataSet_args) # type: ignore
            self.dataset_validation = None
            
            return DataLoader(dataset=self.dataset_train, **self.dataLoader_args), None

    def load(self) -> None:
        if self.dataset_train is not None:
            self.dataset_train.load()
        if self.dataset_validation is not None:
            self.dataset_validation.load()

class DataPrediction(Data):

    @config("Dataset")
    def __init__(self,  dataset_filename : str = "Dataset.h5", 
                        groups_src : dict[str, Group] = {"default" : Group()},
                        patch : DatasetPatch | None = DatasetPatch(),
                        use_cache : bool = True,
                        subset : list[int] | None = None,
                        num_workers : int = 4,
                        pin_memory : bool = True,
                        batch_size : int = 1) -> None:

        super().__init__(dataset_filename, groups_src, patch, use_cache, subset, num_workers, pin_memory, batch_size, False)
        
    def getData(self) -> tuple[DataLoader, DataLoader | None] | DataLoader:
        super().getData()
        assert self.subset
        if len(self.subset) == 1:
            self.subset = [self.subset[0], self.subset[0]+1]
        self.dataset_prediction = DataSet(self.hdf5, index = list(range(self.subset[0], self.subset[1])), **self.dataSet_args)  # type: ignore
        return DataLoader(dataset=self.dataset_prediction, **self.dataLoader_args)

    def load(self) -> None:
        self.dataset_prediction.load()