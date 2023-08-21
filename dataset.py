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
from DeepLearning_API.utils import memoryInfo, cpuInfo, memoryForecast, getMemory, DatasetUtils, State
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
        
    def load(self, group_src : str, group_dest : str, datasetsUtils: list[DatasetUtils]):
        if self._pre_transforms is not None:
            if isinstance(self._pre_transforms, dict):
                for classpath, transform in self._pre_transforms.items():
                    transform = transform.getTransform(classpath, DL_args =  "{}.Dataset.groups_src.{}.groups_dest.{}.pre_transforms".format(os.environ["DEEP_LEARNING_API_ROOT"], group_src, group_dest))
                    transform.setDatasetsUtils(datasetsUtils)
                    self.pre_transforms.append(transform)
            else:
                for transform in self._pre_transforms:
                    transform.setDatasetsUtils(datasetsUtils)
                    self.pre_transforms.append(transform)

        if self._post_transforms is not None:
            if isinstance(self._post_transforms, dict):
                for classpath, transform in self._post_transforms.items():
                    transform = transform.getTransform(classpath, DL_args = "{}.Dataset.groups_src.{}.groups_dest.{}.post_transforms".format(os.environ["DEEP_LEARNING_API_ROOT"], group_src, group_dest))
                    transform.setDatasetsUtils(datasetsUtils)
                    self.post_transforms.append(transform)
            else:
                for transform in self._post_transforms:
                    transform.setDatasetsUtils(datasetsUtils)
                    self.post_transforms.append(transform)
    
    def to(self, device: int):
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

    def __init__(self, rank: int, data : dict[str, list[Dataset]], map: dict[int, tuple[int, int, int]], groups_src : dict[str, Group], dataAugmentationsList : list[DataAugmentationsList], patch_size: Union[list[int], None], overlap: Union[list[int], None], buffer_size: int, use_cache = True) -> None:
        self.rank = rank
        self.data = data
        self.map = map
        self.patch_size = patch_size
        self.overlap = overlap
        self.groups_src = groups_src
        self.dataAugmentationsList = dataAugmentationsList
        self.use_cache = use_cache
        self.nb_dataset = len(data[list(data.keys())[0]])
        self.buffer_size = buffer_size
        self._index_cache: list[int] = []

    def getPatchConfig(self) -> tuple[list[int], list[int]]:
        return self.patch_size, self.overlap
    
    def to(self, device: int):
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
            with tqdm.tqdm(range(self.nb_dataset), desc=description(0), disable=self.rank != 0 and "DL_API_CLUSTER" not in os.environ) as pbar:
                for index in pbar:
                    self._loadData(index)
                    pbar.set_description(description(index))
    
    def _loadData(self, index):
        self._index_cache.append(index)
        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                self.loadData(group_src, group_dest, index)

    def loadData(self, group_src: str, group_dest : str, index : int) -> None:
        self.data[group_dest][index].load(self.groups_src[group_src][group_dest].pre_transforms, self.dataAugmentationsList)

    def _unloadData(self, index : int) -> None:
        self._index_cache.remove(index)
        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                self.unloadData(group_dest, index)
    
    def unloadData(self, group_dest : str, index : int) -> None:
        return self.data[group_dest][index].unload()

    def __len__(self) -> int:
        return len(self.map)

    def __getitem__(self, index : int) -> dict[str, tuple[torch.Tensor, int, int, int, str]]:
        data = {}
        x, a, p = self.map[index]
        if x not in self._index_cache:
            if len(self._index_cache) >= self.buffer_size and not self.use_cache:
                self._unloadData(self._index_cache[0])
            self._loadData(x)
            
        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                dataset = self.data[group_dest][x]
                data["{}".format(group_dest)] = (dataset.getData(p, a, self.groups_src[group_src][group_dest].post_transforms), x, a, p, dataset.name)
        return data

class Subset():
    
    def __init__(self, subset: Union[str, list[int], list[str], None] = None, shuffle: bool = True) -> None:
        self.subset = subset
        self.shuffle = shuffle

    def __call__(self, names: list[str]) -> set[str]:
        inter_name = set(names[0])
        for n in names[1:]:
            inter_name = inter_name.intersection(n)
        names = sorted(list(inter_name))
        size = len(names)
        index = []
        if self.subset is None:
            index = list(range(0, size))
        elif isinstance(self.subset, str):
            r = np.clip(np.asarray([int(self.subset.split(":")[0]), int(self.subset.split(":")[1])]), 0, size)
            index = list(range(r[0], r[1]))
        elif isinstance(self.subset, list):
            if len(self.subset) > 0:
                if isinstance(self.subset[0], int):
                    if len(self.subset) == 1:
                        index = list(range(self.subset[0], min(size, self.subset[0]+1)))
                    else:
                        index = self.subset
                if isinstance(self.subset[0], str):
                    index = []
                    for i, name in enumerate(names):
                        if name in self.subset:
                            index.append(i)
        if self.shuffle:
            index = random.sample(index, len(index))
        return set([names[i] for i in index])
    
class TrainSubset(Subset):

    @config()
    def __init__(self, subset: Union[str, list[int], list[str], None] = None, shuffle: bool = True) -> None:
        super().__init__(subset, shuffle)

class PredictionSubset(Subset):

    @config()
    def __init__(self, subset: Union[str, list[int], list[str], None] = None) -> None:
        super().__init__(subset, False)

class Data(ABC):
    
    def __init__(self,  dataset_filenames : list[str], 
                        groups_src : dict[str, Group],
                        patch : Union[DatasetPatch, None],
                        use_cache : bool,
                        subset : Union[Subset, dict[str, Subset]],
                        num_workers : int,
                        pin_memory : bool,
                        batch_size : int,
                        train_size: Union[float, str, list[int], list[str]] = 1,
                        dataAugmentationsList: dict[str, DataAugmentationsList]= {}) -> None:
        self.dataset_filenames = dataset_filenames
        self.subset = subset
        self.groups_src = groups_src
        self.patch = patch
        self.train_size = train_size
        self.dataAugmentationsList = dataAugmentationsList
        self.batch_size = batch_size
        self.dataSet_args = dict(groups_src=self.groups_src, dataAugmentationsList = list(self.dataAugmentationsList.values()), use_cache = use_cache, buffer_size=batch_size+1, patch_size=self.patch.patch_size if self.patch is not None else None, overlap=self.patch.overlap if self.patch is not None else None)
        self.dataLoader_args = dict(num_workers=num_workers, pin_memory=pin_memory)
        self.data : list[list[dict[str, list[Dataset]]], dict[str, list[Dataset]]] = []
        self.map : list[list[list[tuple[int, int, int]]], list[tuple[int, int, int]]] = []
        self.datasets: dict[str, DatasetUtils] = {}

    def _getDatasets(self, names: list[str], dataset_name: dict[str, dict[str, list[str]]]) -> tuple[dict[str, list[Dataset]], list[tuple[int, int, int]]]:
        nb_dataset = len(names)
        nb_patch = None
        data = {}
        map = []
        nb_augmentation = np.max([int(np.sum([data_augmentation.nb for data_augmentation in self.dataAugmentationsList.values()])+1), 1])
        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                data[group_dest] = [Dataset(i, group_src, group_dest, name, self.datasets[[filename for filename, names in dataset_name[group_src].items() if name in names][0]], patch = self.patch, pre_transforms = self.groups_src[group_src][group_dest].pre_transforms, dataAugmentationsList=list(self.dataAugmentationsList.values())) for i, name in enumerate(names)]
                nb_patch = [[dataset.getSize(a) for a in range(nb_augmentation)] for dataset in data[group_dest]]

        for x in range(nb_dataset):
            for y in range(nb_augmentation):
                for z in range(nb_patch[x][y]):
                    map.append((x, y, z))
        return data, map

    def _split(map: list[tuple[int, int, int]], world_size: int) -> list[list[tuple[int, int, int]]]:
        if len(map) == 0:
            return [[] for _ in range(world_size)]
        
        maps = []
        if os.environ["DL_API_STATE"] == str(State.PREDICTION) or os.environ["DL_API_STATE"] == str(State.METRIC):
            np_map = np.asarray(map)
            unique_index = np.unique(np_map[:, 0])
            offset = int(np.ceil(len(unique_index)/world_size))
            if offset == 0:
                offset = 1
            for itr in range(0, len(unique_index), offset):
                maps.append([tuple(v) for v in np_map[np.where(np.isin(np_map[:, 0], unique_index[itr:itr+offset]))[0], :]])
        else:
            offset = int(np.ceil(len(map)/world_size))
            if offset == 0:
                offset = 1
            for itr in range(0, len(map), offset):
                maps.append(list(map[-offset:]) if itr+offset > len(map) else map[itr:itr+offset])
        return maps

    def getData(self, world_size: int) -> list[list[DataLoader]]:
        datasets: dict[str, list[str]] = {}
        for dataset_filename in self.dataset_filenames:
            filename, format = dataset_filename.split(":")
            datasetUtils = DatasetUtils(filename, format) 
            self.datasets[filename] = datasetUtils

            for group in self.groups_src:
                if datasetUtils.isGroupExist(group):
                    if group in datasets:
                        datasets[group].append(filename) 
                    datasets[group] = [filename]

        for group_src in self.groups_src:
            assert group_src in datasets, "Error group source {} not found".format(group_src)

            for group_dest in self.groups_src[group_src]:
                self.groups_src[group_src][group_dest].load(group_src, group_dest, [self.datasets[filename] for filename in datasets[group_src]])

        for key, dataAugmentations in self.dataAugmentationsList.items():
            dataAugmentations.load(key)

        names : list[list[str]] = []
        dataset_name : dict[str, dict[str, list[str]]] = {}
        for group in self.groups_src:
            if group not in dataset_name:
                dataset_name[group] = {}
            for filename in datasets[group]:
                names.append(self.datasets[filename].getNames(group))
                dataset_name[group][filename] = self.datasets[filename].getNames(group)
        subset_names = set()
        if isinstance(self.subset, dict):
            for filename, subset in self.subset.items():
                subset_names.update(subset([dataset_name[group][filename] for group in dataset_name]))
        else:
            subset_names = self.subset(names)
        
        data, map = self._getDatasets(list(subset_names), dataset_name)
        
        train_map = map
        validate_map = []
        if isinstance(self.train_size, float):
            if self.train_size < 1.0 and int(math.floor(len(map)*(1-self.train_size))) > 0:
                train_map, validate_map = map[:int(math.floor(len(map)*self.train_size))], map[int(math.floor(len(map)*self.train_size)):]
        elif isinstance(self.train_size, str):
            if ":" in self.train_size:
                index = list(range(int(self.subset.split(":")[0]), int(self.subset.split(":")[1])))
                train_map = [m for m in map if m[0] not in index]
                validate_map = [m for m in map if m[0] in index]
            else:
                validate_map = train_map
        elif isinstance(self.train_size, list):
            if len(self.train_size) > 0:
                if isinstance(self.train_size[0], int):
                    train_map = [m for m in map if m[0] not in self.train_size]
                    validate_map = [m for m in map if m[0] in self.train_size]
                elif isinstance(self.train_size[0], str):
                    index = [i for i, n in enumerate(subset_names) if n in self.train_size]
                    train_map = [m for m in map if m[0] not in index]
                    validate_map = [m for m in map if m[0] in index]

        train_maps = Data._split(train_map, world_size)
        validate_maps = Data._split(validate_map, world_size)

        for i, (train_map, validate_map) in enumerate(zip(train_maps, validate_maps)):
            maps = [train_map]
            if len(validate_map):
                maps += [validate_map]
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
                dataLoaders[i].append(DataLoader(dataset=DataSet(rank=i, data=data, map=map, **self.dataSet_args), sampler=CustomSampler(len(map), True), batch_size=self.batch_size,**self.dataLoader_args))
        return dataLoaders

class DataTrain(Data):

    @config("Dataset")
    def __init__(self,  dataset_filenames : list[str] = ["default:Dataset.h5"], 
                        groups_src : dict[str, Group] = {"default" : Group()},
                        augmentations : Union[dict[str, DataAugmentationsList], None] = {"DataAugmentation_0" : DataAugmentationsList()},
                        patch : Union[DatasetPatch, None] = DatasetPatch(),
                        use_cache : bool = True,
                        subset : Union[TrainSubset, dict[str, TrainSubset]] = TrainSubset(),
                        num_workers : int = 4,
                        pin_memory : bool = True,
                        batch_size : int = 1,
                        train_size : Union[float, str, list[int], list[str]] = 0.8) -> None:
        super().__init__(dataset_filenames, groups_src, patch, use_cache, subset, num_workers, pin_memory, batch_size, train_size, augmentations if augmentations else {})

class DataPrediction(Data):

    @config("Dataset")
    def __init__(self,  dataset_filenames : list[str] = ["default:Dataset.h5"], 
                        groups_src : dict[str, Group] = {"default" : Group()},
                        augmentations : Union[dict[str, DataAugmentationsList], None] = {"DataAugmentation_0" : DataAugmentationsList()},
                        patch : Union[DatasetPatch, None] = DatasetPatch(),
                        use_cache : bool = True,
                        subset : Union[PredictionSubset, dict[str, PredictionSubset]] = PredictionSubset(),
                        num_workers : int = 4,
                        pin_memory : bool = True,
                        batch_size : int = 1) -> None:

        super().__init__(dataset_filenames, groups_src, patch, use_cache, subset, num_workers, pin_memory, batch_size, dataAugmentationsList=augmentations if augmentations else {})

class DataMetric(Data):

    @config("Dataset")
    def __init__(self,  dataset_filenames : list[str] = ["default:Dataset.h5"], 
                        groups_src : dict[str, Group] = {"default" : Group()},
                        subset : Union[PredictionSubset, dict[str, PredictionSubset]] = PredictionSubset(),
                        num_workers : int = 4) -> None:

        super().__init__(dataset_filenames, groups_src, None, False, subset, num_workers, False, 1)