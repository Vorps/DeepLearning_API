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

    def __init__(self, data : dict[str, list[Dataset]], map: dict[int, tuple[int, int, int]], groups_src : dict[str, Group], dataAugmentationsList : list[DataAugmentationsList], patch_size: Union[list[int], None], overlap: Union[list[int], None], use_cache = True) -> None:
        self.data = data
        self.map = map
        self.patch_size = patch_size
        self.overlap = overlap
        self.groups_src = groups_src
        self.dataAugmentationsList = dataAugmentationsList
        self.use_cache = use_cache
        self.nb_dataset = len(data[list(data.keys())[0]])

    def getPatchConfig(self) -> tuple[list[int], list[int]]:
        return self.patch_size, self.overlap
    
    def to(self, device: torch.device):
        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                self.groups_src[group_src][group_dest].to(device)
        for dataAugmentations in self.dataAugmentationsList:
            dataAugmentations.to(device)

    def getDatasetFromIndex(self, group_dest: str, index: int) -> Dataset:
        return self.data[group_dest][index]
    
    def load(self):
        if self.use_cache:
            memory_init = getMemory()
            description = lambda i : "Caching : {} | {} | {}".format(memoryInfo(), memoryForecast(memory_init, i, self.nb_dataset), cpuInfo())
            with tqdm.tqdm(range(self.nb_dataset), desc=description(0)) as pbar:
                for index in pbar:
                    self._loadData(index)
                    pbar.set_description(description(index))
    
    def _loadData(self, index):
        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                self.loadData(group_src, group_dest, index)

    def loadData(self, group_src: str, group_dest : str, index : int) -> None:
        self.data[group_dest][index].load(self.groups_src[group_src][group_dest].pre_transforms, self.dataAugmentationsList)

    def _unloadData(self, index : int) -> None:
        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                self.unloadData(group_dest, index)
    
    def unloadData(self, group_dest : str, index : int) -> None:
        return self.data[group_dest][index].unload()

    def __len__(self) -> int:
        return len(self.map)

    def __getitem__(self, index : int) -> dict[str, tuple[torch.Tensor, int, int, int]]:
        data = {}
        x, a, p = self.map[index]
        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                dataset = self.data[group_dest][x]
                self.loadData(group_src, group_dest, x)
                data["{}".format(group_dest)] = (dataset.getData(p, a, self.groups_src[group_src][group_dest].post_transforms), x, a, p)
        return data

class Data(ABC):
    
    def __init__(self,  dataset_filenames : list[str], 
                        groups_src : dict[str, Group],
                        patch : Union[DatasetPatch, None],
                        use_cache : bool,
                        subset : Union[str, list[int], list[str], None],
                        num_workers : int,
                        pin_memory : bool,
                        batch_size : int,
                        shuffle : bool = False,
                        train_size: float = 1,
                        dataAugmentationsList: dict[str, DataAugmentationsList]= {}) -> None:
        self.dataset_filenames = dataset_filenames
        self.subset = subset
        self.groups_src = groups_src
        self.patch = patch
        self.shuffle = shuffle
        self.train_size = train_size
        self.dataAugmentationsList = dataAugmentationsList
        self.dataSet_args = dict(groups_src=self.groups_src, dataAugmentationsList = list(self.dataAugmentationsList.values()), use_cache = use_cache, patch_size=self.patch.patch_size if self.patch is not None else None, overlap=self.patch.overlap if self.patch is not None else None)
        self.dataLoader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        self.data : list[list[dict[str, list[Dataset]]], dict[str, list[Dataset]]] = []
        self.map : list[list[list[tuple[int, int, int]]], list[tuple[int, int, int]]] = []
        self.datasets: dict[str, DatasetUtils] = {}

    def _getDatasets(self, names: list[str]) -> tuple[dict[str, list[Dataset]], list[tuple[int, int, int]]]:
        nb_dataset = len(names)
        nb_patch = None
        data = {}
        map = []
        nb_augmentation = np.max([int(np.sum([data_augmentation.nb for data_augmentation in self.dataAugmentationsList.values()])+1), 1])
        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                data[group_dest] = [Dataset(i, group_src, group_dest, name, self.datasets[group_src], patch = self.patch, pre_transforms = self.groups_src[group_src][group_dest].pre_transforms, dataAugmentationsList=list(self.dataAugmentationsList.values())) for i, name in enumerate(names)]
                nb_patch = [[dataset.getSize(a) for a in range(nb_augmentation)] for dataset in data[group_dest]]

        for x in range(nb_dataset):
            for y in range(nb_augmentation):
                for z in range(nb_patch[x][y]):
                    map.append((x, y, z))
        return data, map

    def getData(self, world_size: int = 1) -> list[list[DataLoader]]:
        for dataset_filename in self.dataset_filenames:
            filename, format = dataset_filename.split(":")
            datasetUtils = DatasetUtils(filename, format) 
            for group in self.groups_src:
                if datasetUtils.isGroupExist(group):
                    if group in self.datasets:
                        raise NameError("Error group source: {} is on two datasets".format(group))
                    self.datasets[group] = datasetUtils
        
        for group_src in self.groups_src:
            assert group_src in self.datasets, "Error group source {} not found".format(group_src)

            for group_dest in self.groups_src[group_src]:
                self.groups_src[group_src][group_dest].load(group_src, group_dest, self.datasets[group_src])

        for key, dataAugmentations in self.dataAugmentationsList.items():
            dataAugmentations.load(key)

        names = []
        for group in self.groups_src:
            names.append(set(self.datasets[group].getNames(group)))
        inter_name = names[0]
        for n in names[1:]:
            inter_name = inter_name.intersection(n)
        inter_name = list(inter_name)
        size = len(inter_name)

        if self.subset is None:
            index = list(range(0, size))
        else:
            if isinstance(self.subset, str):
                subset = [int(i) for i in self.subset.split(":")]
                index = list(range(subset[0], subset[1]))
            elif isinstance(self.subset, list):
                if len(self.subset) > 0:
                    if isinstance(self.subset[0], int):
                        if len(self.subset) == 1:
                            index = list(range(self.subset[0], min(size, self.subset[0]+1)))
                        else:
                            index = self.subset
                    if isinstance(self.subset[0], str):
                        index = []
                        for i, name in enumerate(inter_name):
                            if name in self.subset:
                                index.append(i)
        if self.shuffle:
            index = random.sample(index, self.subset[1]-self.subset[0])

        data, map = self._getDatasets([inter_name[i] for i in index])
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
                dataLoaders[i].append(DataLoader(dataset=DataSet(data=data, map=map, **self.dataSet_args), sampler=CustomSampler(len(map), self.shuffle), **self.dataLoader_args))
        return dataLoaders

class DataTrain(Data):

    @config("Dataset")
    def __init__(self,  dataset_filenames : list[str] = ["default:Dataset.h5"], 
                        groups_src : dict[str, Group] = {"default" : Group()},
                        augmentations : Union[dict[str, DataAugmentationsList], None] = {"DataAugmentation_0" : DataAugmentationsList()},
                        patch : Union[DatasetPatch, None] = DatasetPatch(),
                        use_cache : bool = True,
                        subset : Union[str, list[int], list[str], None] = None,
                        num_workers : int = 4,
                        pin_memory : bool = True,
                        batch_size : int = 1,
                        shuffle : bool = True,
                        train_size : float = 0.8) -> None:
        super().__init__(dataset_filenames, groups_src, patch, use_cache, subset, num_workers, pin_memory, batch_size, shuffle, train_size, augmentations if augmentations else {})

class DataPrediction(Data):

    @config("Dataset")
    def __init__(self,  dataset_filenames : list[str] = ["default:Dataset.h5"], 
                        groups_src : dict[str, Group] = {"default" : Group()},
                        augmentations : Union[dict[str, DataAugmentationsList], None] = {"DataAugmentation_0" : DataAugmentationsList()},
                        patch : Union[DatasetPatch, None] = DatasetPatch(),
                        use_cache : bool = True,
                        subset : Union[str, list[int], list[str], None] = None,
                        num_workers : int = 4,
                        pin_memory : bool = True,
                        batch_size : int = 1) -> None:

        super().__init__(dataset_filenames, groups_src, patch, use_cache,  subset, num_workers, pin_memory, batch_size, dataAugmentationsList=augmentations if augmentations else {})

class DataMetric(Data):

    @config("Dataset")
    def __init__(self,  dataset_filenames : list[str] = ["default:Dataset.h5"], 
                        groups_src : dict[str, Group] = {"default" : Group()},
                        patch : Union[DatasetPatch, None] = DatasetPatch(),
                        use_cache : bool = True,
                        subset : Union[str, list[int], list[str], None] = None,
                        num_workers : int = 4,
                        pin_memory : bool = True,
                        batch_size : int = 1) -> None:

        super().__init__(dataset_filenames, groups_src, patch, use_cache,  subset, num_workers, pin_memory, batch_size)