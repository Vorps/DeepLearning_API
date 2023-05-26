import math
import os
import random
import torch
from torch.utils import data
from sklearn.model_selection import train_test_split
import tqdm
import numpy as np
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Sampler
from typing import Union, Iterator

from DeepLearning_API.HDF5 import DatasetPatch, Dataset
from DeepLearning_API.config import config
from DeepLearning_API.utils import memoryInfo, cpuInfo, memoryForecast, getMemory, NeedDevice, DatasetUtils
from DeepLearning_API.transform import TransformLoader, Transform
from DeepLearning_API.augmentation import DataAugmentationsList


class GroupTransform:

    @config()
    def __init__(self,  pre_transforms : Union[dict[str, TransformLoader], list[Transform]] = {"default:Normalize:Standardize:Unsqueeze:TensorCast:ResampleIsotropic:ResampleResize": TransformLoader()},
                        post_transforms : Union[dict[str, TransformLoader], list[Transform]] = {"default:Normalize:Standardize:Unsqueeze:TensorCast:ResampleIsotropic:ResampleResize": TransformLoader()}) -> None:
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

class CustomSampler(Sampler[int]):

    def __init__(self, size: int, shuffle: bool = False) -> None:
        self.index = list()
        self.shuffle = shuffle
        self.size = size

    def init(self):
        indice = list(range(self.size))
        self.index = [indice[i] for i in torch.randperm(self.size)] if self.shuffle else indice

    def __iter__(self) -> Iterator[int]:
        for index in self.index:
            yield index

    def __len__(self) -> int:
        return self.size
    

class DataSet(data.Dataset):

    def __init__(self, dataset : DatasetUtils, index : list[int], patch : Union[DatasetPatch, None], groups_src : dict[str, Group], dataAugmentationsList : list[DataAugmentationsList], sampler: CustomSampler, use_cache = True, buffer_size: int = 10, nb_process: int = 4) -> None:
        self.use_cache = use_cache
        self.buffer_size = buffer_size
        self.nb_process = nb_process
        self.data : dict[str, list[Dataset]] = {}
        self.cache = list()
        self.dataAugmentationsList = dataAugmentationsList
        self.groups_src = groups_src
        self.map : dict[int, tuple[int, int, int]] = {}
        self.patch = patch
        nb_dataset_list = []
        nb_patch_list = []
        self.sampler = sampler
        self.dataset = dataset

        for group_src in self.groups_src:
            datasets_name = dataset.getInfos(group_src, index)
            nb_dataset_list.append(len(datasets_name))
            
            for group_dest in self.groups_src[group_src]:
                self.data[group_dest] = [Dataset(group_dest, name, dataset, patch = self.patch, pre_transforms = self.groups_src[group_src][group_dest].pre_transforms) for name in datasets_name]
                nb_patch_list.append([len(dataset) for dataset in self.data[group_dest]])
        
        self.nb_dataset = nb_dataset_list[0]
        self.nb_patch = nb_patch_list[0]

        self.nb_augmentation = np.max([int(np.sum([data_augmentation.nb for data_augmentation in self.dataAugmentationsList])), 1])
        
        i = 0
        for x in range(self.nb_dataset):
            for y in range(self.nb_patch[x]):
                for z in range(self.nb_augmentation):
                    self.map[i] = (x,y,z)
                    i += 1

    def _loadData(self, index):
        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                self.loadData(group_src, group_dest, index)
        self.cache.append(index)
        
    def load(self):
        if self.use_cache:
            memory_init = getMemory()
            description = lambda memory_init, i : "Caching : {} | {} | {}".format(memoryInfo(), memoryForecast(memory_init, i, self.nb_dataset), cpuInfo())
            with tqdm.tqdm(range(int(np.ceil(self.nb_dataset/self.nb_process))), desc=description(memory_init, 0)) as pbar:
                for i in pbar:
                    try:
                        from mpi4py.futures import MPICommExecutor
                        with MPICommExecutor(max_workers=self.nb_process) as executor:                        
                            for index in range((i)*self.nb_process, min((i+1)*self.nb_process, self.nb_dataset)):
                                executor.submit(self._loadData, index)
                    except KeyboardInterrupt:
                        pbar.close()
                        exit(0)
                    pbar.set_description(description(memory_init, index))
            
    def loadData(self, group_src: str, group_dest : str, index : int) -> None:
        dataset = self.data[group_dest][index]
        dataset.load(self.dataset, index, self.groups_src[group_src][group_dest].pre_transforms, self.dataAugmentationsList)

    def _unloadData(self, index : int) -> None:
        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                self.unloadData(group_dest, index)
        self.cache.remove(index)

    def unloadData(self, group_dest : str, index : int) -> None:
        return self.data[group_dest][index].unload()

    def __len__(self) -> int:
        return len(self.map)
        
    def __getitem__(self, index : int) -> dict[str, tuple[torch.Tensor, int, int, int]]:
        data = {}
        x, p, a = self.map[index]
        if x not in self.cache: 
            self._loadData(x)
        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                dataset = self.data[group_dest][x]
                data["{}".format(group_dest)] = (dataset.getData(p, a, self.groups_src[group_src][group_dest].post_transforms), x, p, a)
        if not self.use_cache and len(self.cache) > self.buffer_size:
            self._unloadData(self.cache[0])
        return data

class Data(NeedDevice, ABC):
    
    @config("Dataset")
    def __init__(self,  dataset_filename : str = "default", 
                        groups_src : dict[str, Group] = {"default" : Group()},
                        patch : Union[DatasetPatch, None] = None,
                        use_cache : bool = True,
                        buffer_size : int = 10,
                        nb_process : int = 4,
                        subset : Union[list[int], None] = None,
                        num_workers : int = 4,
                        pin_memory : bool = True,
                        batch_size : int = 1) -> None:
        self.subset = subset
        self.groups_src = groups_src
        self.dataAugmentationsList: dict[str, DataAugmentationsList] = {}
        self.dataset = DatasetUtils(dataset_filename)

        self.dataSet_args = dict(dataset=self.dataset, patch = patch, groups_src=self.groups_src, dataAugmentationsList = list(self.dataAugmentationsList.values()), use_cache = use_cache, buffer_size = buffer_size, nb_process = nb_process)
        self.dataLoader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    def __enter__(self):
        self.dataset.__enter__()
        if not self.subset:
            self.subset = [0, len(self)]
        return self
    
    def __exit__(self, type, value, traceback):
        self.dataset.__exit__(type, value, traceback)
    
    def __len__(self) -> int:
        sizes = []
        for group in self.groups_src:
            sizes.append(self.dataset.getSize(group))
        assert len(np.unique(sizes)) == 1
        return sizes[0]

    @abstractmethod
    def getData(self) -> Union[tuple[DataLoader, Union[DataLoader, None]], DataLoader]:
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
                        augmentations : Union[dict[str, DataAugmentationsList], None] = {"DataAugmentation_0" : DataAugmentationsList()},
                        patch : Union[DatasetPatch, None] = DatasetPatch(),
                        use_cache : bool = True,
                        buffer_size : int = 10,
                        nb_process: int = 4,
                        subset : Union[list[int], None] = None,
                        num_workers : int = 4,
                        pin_memory : bool = True,
                        batch_size : int = 1,
                        shuffle : bool = True,
                        train_size : float = 0.8) -> None:
        super().__init__(dataset_filename, groups_src, patch, use_cache, buffer_size, nb_process, subset, num_workers, pin_memory, batch_size)
        self.dataAugmentationsList = augmentations if augmentations else {}
        self.train_test_split_args = dict(train_size=train_size, shuffle=shuffle)
        self.shuffle = shuffle

    def getData(self, random_state : Union[int, None]) -> Union[tuple[DataLoader, Union[DataLoader, None]], DataLoader]:
        super().getData()
        assert self.subset
        if len(self.subset) == 1:
            self.subset = [self.subset[0], self.subset[0]+1]
        nb = self.subset[1]-self.subset[0]+1

        self.dataSet_args.update(dataAugmentationsList=list(self.dataAugmentationsList.values()))
        if self.train_test_split_args["train_size"] < 1 and int(math.floor(nb*(1-self.train_test_split_args["train_size"]))) > 0:
            index_train, index_valid = train_test_split(range(self.subset[0], self.subset[1]), random_state=random_state, **self.train_test_split_args)
            sampler_train = CustomSampler(len(index_train), self.shuffle)
            self.dataset_train = DataSet(index=index_train, sampler=sampler_train, **self.dataSet_args)
            sampler_test = CustomSampler(len(index_valid), shuffle=self.shuffle)
            self.dataset_validation = DataSet(index=index_valid, sampler=sampler_test, **self.dataSet_args)
            return DataLoader(dataset=self.dataset_train, sampler=sampler_train, **self.dataLoader_args), DataLoader(dataset=self.dataset_validation, sampler=sampler_test, **self.dataLoader_args)
        else:
            if self.train_test_split_args["shuffle"]:
                index_train =  random.sample(list(range(self.subset[0], self.subset[1])), self.subset[1]-self.subset[0])
            else:
                index_train = list(range(self.subset[0], self.subset[1]))
            sampler_train = CustomSampler(len(index_train), self.shuffle)
            self.dataset_train = DataSet(index=index_train, sampler=sampler_train, **self.dataSet_args)
            self.dataset_validation = None
            return DataLoader(dataset=self.dataset_train, sampler=sampler_train, **self.dataLoader_args), None

    def load(self) -> None:
        if self.dataset_train is not None:
            self.dataset_train.load()
        if self.dataset_validation is not None:
            self.dataset_validation.load()

class DataPrediction(Data):

    @config("Dataset")
    def __init__(self,  dataset_filename : str = "Dataset.h5", 
                        groups_src : dict[str, Group] = {"default" : Group()},
                        patch : Union[DatasetPatch, None] = DatasetPatch(),
                        use_cache : bool = True,
                        buffer_size : int = 10,
                        nb_process: int = 4,
                        subset : Union[list[int], None] = None,
                        num_workers : int = 4,
                        pin_memory : bool = True,
                        batch_size : int = 1) -> None:

        super().__init__(dataset_filename, groups_src, patch, use_cache, buffer_size, nb_process, subset, num_workers, pin_memory, batch_size)
        
    def getData(self) -> Union[tuple[DataLoader, Union[DataLoader, None]], DataLoader]:
        super().getData()
        assert self.subset
        if len(self.subset) == 1:
            self.subset = [self.subset[0], self.subset[0]+1]
        self.dataset_prediction = DataSet(self.dataset, index = list(range(self.subset[0], self.subset[1])), **self.dataSet_args)
        return DataLoader(dataset=self.dataset_prediction, **self.dataLoader_args)

    def load(self) -> None:
        self.dataset_prediction.load()