from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, Iterator, List, Optional, Tuple
import SimpleITK as sitk
import h5py
import numpy as np
import torch
import os
import torch.nn.functional as F

from DeepLearning_API.config import config
from DeepLearning_API.utils import DatasetUtils, dataset_to_data
from DeepLearning_API.transform import Transform
from DeepLearning_API.augmentation import DataAugmentationsList


import itertools

class PathCombine(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        pass

class GausianImportance(PathCombine):

    def __init__(self, patch_size = [128,128,128], sigma : float = 0.2) -> None:
        super().__init__()
        gaussianSource = sitk.GaussianImageSource()
        gaussianSource.SetSize(tuple(patch_size))
        gaussianSource.SetMean([x // 2 for x in patch_size])
        gaussianSource.SetSigma([sigma * x for x in patch_size])
        gaussianSource.SetScale(1)
        gaussianSource.NormalizedOn()
        self.data = torch.from_numpy(sitk.GetArrayFromImage(gaussianSource.Execute())).unsqueeze(0)

    def __call__(self, input : torch.Tensor) -> torch.Tensor:
        return self.data.repeat([input.shape[0]]+[1]*(len(input.shape)-1))*input

class Accumulator():

    def __init__(self) -> None:
        self.layer_accumulator = []

    def addLayer(self, layer):
        self.layer_accumulator.append(layer)

class Patch(ABC):

    def __init__(self, patch_size: List[int], overlap: Optional[List[int]]) -> None:
        self.patch_size = patch_size
        self.overlap = overlap
        self.patch_slices : Optional[List[tuple[slice]]] = None
        self.shape: Optional[List[int]] = None
        self.dtype: Optional[torch.dtype] = None
        self.patchCombine = None
        
    def load(self, shape : List[int]) -> None:
        if not all(a >= b for a,b in zip(shape[-len(self.patch_size):], self.patch_size)) or not len(shape):
            return
        self.patch_slices = []
        self.shape = shape
        slices : List[List[slice]] = []
        if self.overlap is None:
            size = [a//b for a,b in zip(shape, self.patch_size)]
            overlap_tmp = np.zeros(len(size), dtype=np.int)
            for i, s in enumerate(size):
                if s > 1:
                    overlap_tmp[i] = np.mod(self.patch_size[i]-np.mod(shape[i], self.patch_size[i]), self.patch_size[i])//(size[i]-1)
        else:
            overlap_tmp = self.overlap

        for dim in range(len(shape)):
            slices.append([])
            index = 0
            while True:
                start = (self.patch_size[dim]-overlap_tmp[dim])*index
                end = start + self.patch_size[dim]
                if end >= shape[dim]:
                    end = shape[dim]
                    
                    if self.overlap is None and end-self.patch_size[dim] >= 0:
                        start = end-self.patch_size[dim]   
                    slices[dim].append(slice(start,end))
                    break
                slices[dim].append(slice(start,end))
                index += 1
        for chunk in itertools.product(*slices):
            self.patch_slices.append(tuple(chunk))

    def getData(self, data : torch.Tensor, index : int) -> torch.Tensor:
        if self.patch_slices is None or self.shape is None:
            return data
        self.dtype = data.dtype
        slices = []
        for max in data.shape[:-len(self.patch_slices[index])]:
            slices.append(slice(max))
        i = len(slices)

        slices += list(self.patch_slices[index])
        data = data[slices]
        padding = []
        for dim_it, _slice in enumerate(reversed(self.patch_slices[index])):
            dim = len(self.shape)-dim_it-1
            padding.append(0)
            padding.append(0 if _slice.start+self.patch_size[dim] <= self.shape[dim] else _slice.start+self.patch_size[dim]-_slice.stop)
    
        data = F.pad(data, tuple(padding), "constant", 0)
    
        for d in reversed(np.where(self.patch_size == 1)[0]+i):
            data = torch.squeeze(data, dim = d)
        return data

    def isFull(self):
        return self.layer_accumulator == len(self)

    def assemble(self, layer_accumulator: List[torch.Tensor], device: torch.device) -> torch.Tensor:
        if len(layer_accumulator) == 1 or self.patch_slices is None or self.shape is None or self.dtype is None:
            return layer_accumulator[0]

        patch_slice_shape = [slice.stop-slice.start for slice in self.patch_slices[0]]
        data_shape = list(layer_accumulator[0].shape[-len(patch_slice_shape):])
        isImage = patch_slice_shape == data_shape
        if isImage:
            result = torch.empty((list(layer_accumulator[0].shape[:-len(self.shape)])+list(self.shape)), dtype=self.dtype).to(device)
            for patch_slice, data in zip(self.patch_slices, layer_accumulator):
                slices = []
                for max in data.shape[:-len(patch_slice)]:
                    slices.append(slice(max))
                slices += list(patch_slice)
                result[slices] = data
        else:
            result = torch.cat([data for data in layer_accumulator], dim=0)
        layer_accumulator.clear()
        return result

    def __len__(self) -> int:
        return len(self.patch_slices) if self.patch_slices is not None else 1

class DatasetPatch(Patch):

    @config("Patch")
    def __init__(self, patch_size : List[int] = [128, 256, 256], overlap : Optional[List[int]] = None) -> None:
        super().__init__(patch_size, overlap)
    
class ModelPatch(Patch):

    @config("Patch")
    def __init__(self, patch_size : List[int] = [128, 256, 256], overlap : Optional[List[int]] = None, patchCombine:Optional[str] = None) -> None:
        super().__init__(patch_size, overlap)
        self.patchCombine = patchCombine

    def disassemble(self, *dataList: torch.Tensor) -> Iterator[List[torch.Tensor]]:
        for i in range(len(self)):
            yield [self.getData(data, i) for data in dataList]

class Dataset():

    def __init__(self, group : str, dataset : h5py.Dataset, patch : Optional[DatasetPatch], pre_transforms : List[Transform]) -> None:
        self._dataset = dataset
        self.group = group
        self.name = self._dataset.name    
        self.loaded = False

        self._shape = list(self._dataset.shape[1:])
        self.cache_attribute = {k : torch.tensor(v) for k, v in self._dataset.attrs.items()}
        
        self.data : List[torch.Tensor] = list()
        for transformFunction in pre_transforms:
            self._shape = transformFunction.transformMetaData(self._shape, self.cache_attribute)

        self.patch = DatasetPatch(patch_size=patch.patch_size, overlap=patch.overlap) if patch else DatasetPatch(self._shape)
        self.patch.load(self._shape)
    
    def load(self, index : int, pre_transform : List[Transform], dataAugmentationsList : List[DataAugmentationsList]) -> None:
        if self.loaded:
            return
        
        assert self.name
        i = len(pre_transform)
        data = None
        for transformFunction in reversed(pre_transform):
            if transformFunction.save is not None and os.path.exists(transformFunction.save):
                with DatasetUtils(transformFunction.save) as datasetUtils:
                    if self.group in datasetUtils.h5:
                        group = datasetUtils.h5[self.group]
                        if isinstance(group, h5py.Group) and self.name in group:
                            dataset = group[self.name]
                            if isinstance(dataset, h5py.Dataset):
                                data, attrib = dataset_to_data(dataset)
                                self.cache_attribute.update(attrib)
                                break
            i-=1

        if i==0:
            data = np.empty(self._dataset.shape, self._dataset.dtype)
            self._dataset.read_direct(data)
        
        data = torch.from_numpy(data)
        if len(pre_transform):
            for transformFunction in pre_transform[i:]:
                data = transformFunction(data, self.cache_attribute)
                if transformFunction.save is not None:
                    with DatasetUtils(transformFunction.save, read=False) as datasetUtils:
                        datasetUtils.writeData(self.group, self.name, data.numpy(), self.cache_attribute)

        self.data : List[torch.Tensor] = list()
        if not dataAugmentationsList:
            self.data.append(data)
            
        for dataAugmentations in dataAugmentationsList:
            for dataAugmentation in dataAugmentations.dataAugmentations:
                dataAugmentation.state_init(index, self._shape, self.cache_attribute)
                for d in dataAugmentation(data.unsqueeze(dim=0).repeat(dataAugmentations.nb, *[1 for _ in range(len(data.shape))])):
                    self.data.append(d)

        self.loaded = True
             
    def unload(self) -> None:
        del self.data
        self.loaded = False
    
    def getData(self, index : int, a : int, post_transforms : List[Transform]) -> torch.Tensor:
        data = self.patch.getData(self.data[a], index)
        for transformFunction in post_transforms:
            data = transformFunction(data, self.cache_attribute)
        return data

    def __len__(self) -> int:
        return len(self.patch)
  
class HDF5(DatasetUtils):

    def __init__(self, filename : str) -> None:
        super().__init__(filename, read=True)

    def getSize(self, group):
        result = 0
        h5_group = self.h5[group]
        assert isinstance(h5_group, h5py.Group)
        for dataset in [dataset for dataset in h5_group.values() if type(dataset) == h5py.Dataset]:
            group = dataset.name.split(".")[0]
            result += self.getSize(group) if group in self.h5 else 1
        return result
    
    def getDatasets(self, group: str, index: List[int], i: int = 0, it: int = 0) -> Tuple[List[h5py.Dataset], List[List[int]], int, int]:
        h5_group = self.h5[group]
        assert isinstance(h5_group, h5py.Group)
        datasets = []
        mapping: List[List[int]] = []
        for dataset in [dataset for dataset in h5_group.values() if type(dataset) == h5py.Dataset]:
            group = dataset.name.split(".")[0]
            if group in self.h5:
                sub_datasets, sub_mapping, i, it = self.getDatasets(group, index, i, it)
                for l in sub_mapping:
                    mapping.append([it]+l)
                datasets+=sub_datasets
                if len(sub_datasets):
                    datasets.append(dataset)
                    it += 1
            else:
                if i in index:
                    mapping.append([it])
                    datasets.append(dataset)
                    it += 1
                i += 1
        return datasets, mapping, i, it