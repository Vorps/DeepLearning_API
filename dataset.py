import math
import os
import random
import torch
from torch.utils import data
import tqdm
import numpy as np
from abc import ABC
from torch.utils.data import DataLoader, Sampler
from typing import Union, Iterator

from DeepLearning_API.HDF5 import DatasetPatch, Dataset
from DeepLearning_API.config import config
from DeepLearning_API.utils import memoryInfo, cpuInfo, memoryForecast, getMemory, DatasetUtils
from DeepLearning_API.transform import TransformLoader, Transform
from DeepLearning_API.augmentation import DataAugmentationsList
import multiprocessing as mp

class GroupTransform:

    @config()
    def __init__(self,  pre_transforms : Union[dict[str, TransformLoader], list[Transform]] = {"default:Normalize:Standardize:Unsqueeze:TensorCast:ResampleIsotropic:ResampleResize": TransformLoader()},
                        post_transforms : Union[dict[str, TransformLoader], list[Transform]] = {"default:Normalize:Standardize:Unsqueeze:TensorCast:ResampleIsotropic:ResampleResize": TransformLoader()}) -> None:
        self._pre_transforms = pre_transforms
        self._post_transforms = post_transforms
        self.pre_transforms : list[Transform] = []
        self.post_transforms : list[Transform] = []
        
    def load(self, group_src : str, group_dest : str, datasetUtils: DatasetUtils):
        if self._pre_transforms is not None:
            if isinstance(self._pre_transforms, dict):
                for classpath, transform in self._pre_transforms.items():
                    transform = transform.getTransform(classpath, DL_args =  "{}.Dataset.groups_src.{}.groups_dest.{}.pre_transforms".format(os.environ["DEEP_LEARNING_API_ROOT"], group_src, group_dest))
                    transform.setDatasetUtils(datasetUtils)
                    self.pre_transforms.append(transform)
            else:
                for transform in self._pre_transforms:
                    transform.setDatasetUtils(datasetUtils)
                    self.pre_transforms.append(transform)

        if self._post_transforms is not None:
            if isinstance(self._post_transforms, dict):
                for classpath, transform in self._post_transforms.items():
                    transform = transform.getTransform(classpath, DL_args = "{}.Dataset.groups_src.{}.groups_dest.{}.post_transforms".format(os.environ["DEEP_LEARNING_API_ROOT"], group_src, group_dest))
                    transform.setDatasetUtils(datasetUtils)
                    self.post_transforms.append(transform)
            else:
                for transform in self._post_transforms:
                    transform.setDatasetUtils(datasetUtils)
                    self.post_transforms.append(transform)
    
    def to(self, device: torch.device):
        for transform in self.pre_transforms:
            transform.setDevice(device)
        for transform in self.post_transforms:
            transform.setDevice(device)

class Group(dict[str, GroupTransform]):

    @config()
    def __init__(self, groups_dest: dict[str, GroupTransform] = {"default": GroupTransform()}):
        super().__init__(groups_dest)

class CustomSampler(Sampler[int]):

    def __init__(self, size: int, shuffle: bool = False) -> None:
        self.size = size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        return iter(torch.randperm(len(self)).tolist() if self.shuffle else list(range(len(self))) )

    def __len__(self) -> int:
        return self.size

class DataSet(data.Dataset):

    def __init__(self, data : dict[str, list[Dataset]], map: dict[int, tuple[int, int, int]], groups_src : dict[str, Group], dataAugmentationsList : list[DataAugmentationsList], sampler: CustomSampler, num_workers: int, use_cache = True, buffer_size: float = 0.1, nb_process: int = 4) -> None:
        self.data = data
        self.map = map
        self.groups_src = groups_src
        self.dataAugmentationsList = dataAugmentationsList
        self.sampler = sampler

        self.use_cache = use_cache
        self.buffer_size = buffer_size if buffer_size < 1 else 1 
        self.nb_process = nb_process
        
        self.cache = list()
        self.cached = list()
        
        self.MPICommExecutor = mp.Pool(self.nb_process) if num_workers == 0 and not self.use_cache else None
        self.executor = None
        self.nb_dataset = len(data[list(data.keys())[0]])

    def to(self, device: torch.device):
        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                self.groups_src[group_src][group_dest].to(device)
        for dataAugmentations in self.dataAugmentationsList:
            dataAugmentations.to(device)

    def _loadData(self, index):
        self.cache.append(index)
        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                self.loadData(group_src, group_dest, index)
        self.cached.append(index)
        print(self.cached)

    def load(self):
        if self.use_cache:
            memory_init = getMemory()
            description = lambda i : "Caching : {} | {} | {}".format(memoryInfo(), memoryForecast(memory_init, i, int(np.ceil(self.nb_dataset/self.nb_process))), cpuInfo())

            with tqdm.tqdm(range(int(np.ceil(self.nb_dataset/self.nb_process))), desc=description(0)) as pbar:
                for i in pbar:
                    try:
                        for index in range((i)*self.nb_process, min((i+1)*self.nb_process, self.nb_dataset)):
                            self._loadData(index)
                        #mp.set_start_method('spawn')
                        #with mp.Pool(self.nb_process) as p:
                        #    p.map(self._loadData, list(range((i)*self.nb_process, min((i+1)*self.nb_process, self.nb_dataset))))
                    except KeyboardInterrupt:
                        pbar.close()
                        exit(0)
                    pbar.set_description(description(i))

    def init(self):
        self.sampler_iter = iter(self.sampler)
        if not self.use_cache:
            if self.MPICommExecutor:
                self.executor = self.MPICommExecutor.__enter__()
            memory_init = getMemory()
            description = lambda i : "Buffering : {} | {} | {}".format(memoryInfo(), memoryForecast(memory_init, i, int(np.ceil(self.buffer_size*self.nb_dataset/self.nb_process))), cpuInfo())
            
            with tqdm.tqdm(range(int(np.ceil(self.buffer_size*self.nb_dataset/self.nb_process))), desc=description(0)) as pbar:
                for i in pbar:
                    it = 0
                    try:
                        with Pool(self.nb_process) as p:
                            while it < self.nb_process and len(self.cache) < np.ceil(self.buffer_size*self.nb_dataset):
                                index, _, _ = self.map[next(self.sampler_iter)]
                                if index not in self.cache:
                                    self.cache.append(index)
                                    p.apply_async(self._loadData, args=(index))
                                    it+= 1
                    except KeyboardInterrupt:
                        pbar.close()
                        exit(0)
                    pbar.set_description(description(i))
    
    def getDatasetFromIndex(self, group_dest: str, index: int) -> Dataset:
        return self.data[group_dest][index]
    
    def close(self):
        if self.MPICommExecutor:
            self.MPICommExecutor.__exit__()

    def loadData(self, group_src: str, group_dest : str, index : int) -> None:
        dataset = self.data[group_dest][index]
        dataset.load(index, self.groups_src[group_src][group_dest].pre_transforms, self.dataAugmentationsList)

    def _unloadData(self, index : int) -> None:
        self.cache.remove(index)
        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                self.unloadData(group_dest, index)
        self.cached.remove(index)
    
    def unloadData(self, group_dest : str, index : int) -> None:
        return self.data[group_dest][index].unload()

    def __len__(self) -> int:
        return len(self.map)

    def __getitem__(self, index : int) -> dict[str, tuple[torch.Tensor, int, int, int]]:
        data = {}
        x, p, a = self.map[index]
        if not self.use_cache:
            if len(self.cache) - len(self.cached) >= self.nb_process:
                if self.MPICommExecutor:
                    self.MPICommExecutor.__exit__()
                    self.executor = self.MPICommExecutor.__enter__()
            if x not in self.cached:
                if self.MPICommExecutor:
                    self.MPICommExecutor.__exit__()
                    self.executor = self.MPICommExecutor.__enter__()
                if x not in self.cached:
                    self.cache.append(x)
                    self._loadData(x)

        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                dataset = self.data[group_dest][x]
                data["{}".format(group_dest)] = (dataset.getData(p, a, self.groups_src[group_src][group_dest].post_transforms), x, p, a)

        if not self.use_cache:
            index = next(self.sampler_iter, -1)
            if index != -1:
                x, _, _ = self.map[index]
                if x not in self.cache:
                    self.cache.append(x)
                    if self.MPICommExecutor:
                        self.executor.apply(self._loadData, args=(x))
                    else:
                        self._loadData(x)
                    if len(self.cache) > np.ceil(self.buffer_size*self.nb_dataset+10):
                        self._unloadData(self.cache[0])
        return data

class Data(ABC):
    
    @config("Dataset")
    def __init__(self,  dataset_filename : str = "default", 
                        groups_src : dict[str, Group] = {"default" : Group()},
                        patch : Union[DatasetPatch, None] = None,
                        use_cache : bool = True,
                        buffer_size : float = 0.1,
                        nb_process : int = 4,
                        subset : Union[list[int], None] = None,
                        num_workers : int = 4,
                        pin_memory : bool = True,
                        batch_size : int = 1,
                        shuffle : bool = False,
                        train_size: float = 1.0,
                        dataAugmentationsList: dict[str, DataAugmentationsList] = {}) -> None:
        self.dataset_filename = dataset_filename
        self.subset = subset
        self.groups_src = groups_src
        self.patch = patch
        self.shuffle = shuffle
        self.train_size = train_size
        self.dataAugmentationsList = dataAugmentationsList
        self.dataSet_args = dict(groups_src=self.groups_src, dataAugmentationsList = list(self.dataAugmentationsList.values()), use_cache = use_cache, buffer_size = buffer_size, nb_process = nb_process, num_workers=num_workers)
        self.dataLoader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        self.data : list[list[dict[str, list[Dataset]]], dict[str, list[Dataset]]] = []
        self.map : list[list[list[tuple[int, int, int]]], list[tuple[int, int, int]]] = []

    def _getDatasets(self, index: list[int]) -> tuple[dict[str, list[Dataset]], list[tuple[int, int, int]]]:
        nb_dataset = None
        nb_patch = None
        data = {}
        map = []
        for group_src in self.groups_src:
            datasets_name = self.dataset.getNames(group_src, index)
            nb_dataset = len(datasets_name)
        
            for group_dest in self.groups_src[group_src]:
                data[group_dest] = [Dataset(group_src, group_dest, name, self.dataset, patch = self.patch, pre_transforms = self.groups_src[group_src][group_dest].pre_transforms) for name in datasets_name]
                nb_patch = [len(dataset) for dataset in data[group_dest]]
        nb_augmentation = np.max([int(np.sum([data_augmentation.nb for data_augmentation in self.dataAugmentationsList])), 1])

        for x in range(nb_dataset):
            for y in range(nb_patch[x]):
                for z in range(nb_augmentation):
                    map.append((x, y, z))
        return data, map

    def getData(self, world_size: int = 1) -> list[list[DataLoader]]:
        self.dataset = DatasetUtils(self.dataset_filename)
        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                self.groups_src[group_src][group_dest].load(group_src, group_dest, self.dataset)

        for key, dataAugmentations in self.dataAugmentationsList.items():
            dataAugmentations.load(key)

        sizes = []
        
        for group in self.groups_src:
            sizes.append(self.dataset.getSize(group))
        assert len(np.unique(sizes)) == 1
        sizes = sizes[0]
        if self.subset is None:
            self.subset = [0, sizes]
        elif len(self.subset) == 1:
            self.subset = [self.subset[0], self.subset[0]+1]

        index = range(self.subset[0], self.subset[1])
        if self.shuffle:
            index = random.sample(list(range(self.subset[0], self.subset[1])), self.subset[1]-self.subset[0])

        data, map = self._getDatasets(index)
        offset = len(map)//world_size

        for i, itr in enumerate(range(0, len(map), offset)):
            if itr+offset > len(map):
                map_batched = map[itr-offset:]
            else:
                map_batched = map[itr:itr+offset]

            if self.train_size < 1.0 and int(math.floor(len(map_batched)*(1-self.train_size))) > 0:
                maps = [map_batched[:int(math.floor(len(map_batched)*self.train_size))], map_batched[int(math.floor(len(map_batched)*self.train_size)):]]
            else:
                maps = [map_batched]
            
            self.data.append([])
            self.map.append([])
            for map_tmp in maps:
                indexs = np.unique(np.asarray(map_tmp)[:, 0])
                self.data[i].append({k:[v[it] for it in indexs] for k, v in data.items()})
                map_tmp_array = np.asarray(map_tmp)
                for a, b in enumerate(indexs):
                    map_tmp_array[np.where(np.asarray(map_tmp_array)[:, 0] == b), 0] = a
                self.map[i].append([(a,b,c) for a,b,c in map_tmp_array])
                

        dataLoaders: list[list[DataLoader]] = []
        for i, (datas, maps) in enumerate(zip(self.data, self.map)):
            dataLoaders.append([])
            for data, map in zip(datas, maps):
                sampler_train = CustomSampler(len(map), self.shuffle)
                dataLoaders[i].append(DataLoader(dataset=DataSet(data=data, map=map, sampler=sampler_train, **self.dataSet_args), sampler=sampler_train, **self.dataLoader_args))
        return dataLoaders

class DataTrain(Data):

    @config("Dataset")
    def __init__(self,  dataset_filename : str = "default", 
                        groups_src : dict[str, Group] = {"default" : Group()},
                        augmentations : Union[dict[str, DataAugmentationsList], None] = {"DataAugmentation_0" : DataAugmentationsList()},
                        patch : Union[DatasetPatch, None] = DatasetPatch(),
                        use_cache : bool = True,
                        buffer_size : float = 0.1,
                        nb_process: int = 4,
                        subset : Union[list[int], None] = None,
                        num_workers : int = 4,
                        pin_memory : bool = True,
                        batch_size : int = 1,
                        shuffle : bool = True,
                        train_size : float = 0.8) -> None:
        super().__init__(dataset_filename, groups_src, patch, use_cache, buffer_size, nb_process, subset, num_workers, pin_memory, batch_size, shuffle, train_size, augmentations if augmentations else {})

class DataPrediction(Data):

    @config("Dataset")
    def __init__(self,  dataset_filename : str = "Dataset.h5", 
                        groups_src : dict[str, Group] = {"default" : Group()},
                        patch : Union[DatasetPatch, None] = DatasetPatch(),
                        use_cache : bool = True,
                        buffer_size : float = 0.1,
                        nb_process: int = 4,
                        subset : Union[list[int], None] = None,
                        num_workers : int = 4,
                        pin_memory : bool = True,
                        batch_size : int = 1) -> None:

        super().__init__(dataset_filename, groups_src, patch, use_cache, buffer_size, nb_process, subset, num_workers, pin_memory, batch_size)