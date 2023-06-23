from abc import ABC, abstractmethod
import SimpleITK as sitk
import h5py
import numpy as np
import torch
import os
import torch.nn.functional as F
from typing import Iterator

from DeepLearning_API.config import config
from DeepLearning_API.utils import DatasetUtils, dataset_to_data, Attribute, get_patch_slices_from_shape
from DeepLearning_API.transform import Transform, Save
from DeepLearning_API.augmentation import DataAugmentationsList
from typing import Union, Callable

class PathCombine(ABC):

    def __init__(self) -> None:
        self.model = None

    def setModel(self, model):
        self.model = model

    @abstractmethod
    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        pass


class Accumulator():

    def __init__(self, patch_slices: list[tuple[slice]], patchCombine: Union[str, None] = None, batch: bool = True) -> None:
        self._layer_accumulator: list[Union[torch.Tensor, None]] = [None for i in range(len(patch_slices))]
        self.patch_slices = patch_slices
        self.shape = max([[v.stop for v in patch] for patch in patch_slices])

        self.patchCombine = patchCombine
        self.batch = batch

    def addLayer(self, index: int, layer: torch.Tensor) -> None:
        self._layer_accumulator[index] = layer
    
    def isFull(self) -> bool:
        return len(self.patch_slices) == len([v for v in self._layer_accumulator if v is not None])

    def assemble(self, device: torch.device) -> torch.Tensor:
        patch_size = [sl.stop-sl.start for sl in self.patch_slices[0]]
        if all([self._layer_accumulator[0].shape[i-len(patch_size)] == size for i, size in enumerate(patch_size)]): 
            N = 2 if self.batch else 1
            result = torch.empty((list(self._layer_accumulator[0].shape[:N])+list(self.shape)), dtype=self._layer_accumulator[0].dtype).to(device)
            for patch_slice, data in zip(self.patch_slices, self._layer_accumulator):
                slices = tuple([slice(result.shape[i]) for i in range(N)] + list(patch_slice))
                for dim, s in enumerate(patch_slice):
                    if s.stop-s.start == 1:
                        data = data.unsqueeze(dim=dim+1)
                result[slices] = data
        else:
            result = torch.cat(tuple(self._layer_accumulator), dim=0)
        self._layer_accumulator.clear()
        return result

class Patch(ABC):

    def __init__(self, patch_size: list[int], overlap: Union[list[int], None], path_mask: Union[str, None] = None) -> None:
        self.patch_size = patch_size
        self.overlap = overlap
        self.patch_slices : list[tuple[slice]] = None
        self.nb_patch_per_dim: list[tuple[int, bool]] = None
        self.path_mask = path_mask
        self.mask = None
        if self.path_mask is not None:
            if os.path.exists(self.path_mask):
                self.mask = torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(self.path_mask)))
            else:
                raise NameError('Mask file not found')
            
    def load(self, shape : list[int]) -> None:
        self.patch_slices, self.nb_patch_per_dim = get_patch_slices_from_shape(self.patch_size, shape, self.overlap)

    def getData(self, data : torch.Tensor, index : int) -> torch.Tensor:
        if len(self.patch_slices) == 1:
            return data
        
        slices = []
        for max in data.shape[:-len(self.patch_slices[index])]:
            slices.append(slice(max))

        slices += list(self.patch_slices[index])
        data = data[slices]
        padding = []
        nb_dim = len(self.patch_size)
        for dim_it, _slice in enumerate(reversed(self.patch_slices[index])):
            dim = nb_dim-dim_it-1
            padding.append(0)
            padding.append(0 if _slice.start+self.patch_size[dim] <= data.shape[dim] else _slice.start+self.patch_size[dim]-_slice.stop)

        data = F.pad(data, tuple(padding), "constant", 0)
        if self.mask is not None:
            data = torch.where(self.mask == 0, torch.zeros((1), dtype=data.dtype), data)

        for d in [i for i, v in enumerate(reversed(self.patch_size)) if v == 1]:
            data = torch.squeeze(data, dim = len(data.shape)-d-1)
        return data

    def __len__(self) -> int:
        return len(self.patch_slices) if self.patch_slices is not None else 1
    
    def disassemble(self, *dataList: torch.Tensor) -> Iterator[list[torch.Tensor]]:
        for i in range(len(self)):
            yield [self.getData(data, i) for data in dataList]

class DatasetPatch(Patch):

    @config("Patch")
    def __init__(self, patch_size : list[int] = [128, 256, 256], overlap : Union[list[int], None] = None, mask: Union[str, None] = None) -> None:
        super().__init__(patch_size, overlap, mask)

class ModelPatch(Patch):

    @config("Patch")
    def __init__(self, patch_size : list[int] = [128, 256, 256], overlap : Union[list[int], None] = None, patchCombine: Union[str, None] = None, mask: Union[str, None] = None) -> None:
        super().__init__(patch_size, overlap, mask)
        self.patchCombine = patchCombine

class Dataset():

    def __init__(self, group_src, group_dest : str, name: str, datasetUtils : DatasetUtils, patch : Union[DatasetPatch, None], pre_transforms : list[Transform]) -> None:
        self.group_src = group_src
        self.group_dest = group_dest
        self.name = name
        self.datasetUtils = datasetUtils
        self.loaded = False

        self._shape, self.cache_attribute =  self.datasetUtils.getInfos(self.group_src, name)
        self._shape = list(self._shape[1:])
        
        self.data : list[torch.Tensor] = list()
        for transformFunction in pre_transforms:
            self._shape = transformFunction.transformShape(self._shape, self.cache_attribute)

        self.patch = DatasetPatch(patch_size=patch.patch_size, overlap=patch.overlap, mask=patch.path_mask) if patch else DatasetPatch(self._shape)
        self.patch.load(self._shape)
    
    def load(self, index : int, pre_transform : list[Transform], dataAugmentationsList : list[DataAugmentationsList]) -> None:
        if self.loaded:
            return
        assert self.name
        i = len(pre_transform)
        data = None
        for transformFunction in reversed(pre_transform):
            if isinstance(transformFunction, Save) and os.path.exists(transformFunction.save):
                datasetUtils = DatasetUtils(transformFunction.save if not transformFunction.save.endswith("/") else transformFunction.save[:-1])
                if datasetUtils.isExist(self.group_dest, self.name):
                    data, attrib = datasetUtils.readData(self.group_dest, self.name)
                    self.cache_attribute.update(attrib)
                    break
            i-=1
        
        if i==0:
            data, _ = self.datasetUtils.readData(self.group_src, self.name)

        data = torch.from_numpy(data)
        if len(pre_transform):
            for transformFunction in pre_transform[i:]:
                data = transformFunction(self.name, data, self.cache_attribute)
                if isinstance(transformFunction, Save):
                    datasetUtils = DatasetUtils(transformFunction.save if not transformFunction.save.endswith("/") else transformFunction.save[:-1], is_directory=transformFunction.save.endswith("/"))
                    datasetUtils.writeData(self.group_dest, self.name, data.numpy(), self.cache_attribute)
        self.data : list[torch.Tensor] = list()
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
    
    def getData(self, index : int, a : int, post_transforms : list[Transform]) -> torch.Tensor:
        data = self.patch.getData(self.data[a], index)
        for transformFunction in post_transforms:
            data = transformFunction(self.name, data, self.cache_attribute)
        return data

    def __len__(self) -> int:
        return len(self.patch)
  
    