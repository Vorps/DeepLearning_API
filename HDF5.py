from abc import ABC, abstractmethod
import SimpleITK as sitk
import h5py
import numpy as np
import torch
import os
import torch.nn.functional as F
from typing import Iterator

from DeepLearning_API.config import config
from DeepLearning_API.utils import DatasetUtils, NeedDevice, dataset_to_data, Attribute, get_patch_slices_from_shape
from DeepLearning_API.transform import Transform, Save
from DeepLearning_API.augmentation import DataAugmentationsList
from typing import Union, Callable

class PathCombine(NeedDevice, ABC):

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
        if self.path_mask is not None and os.path.exists(self.path_mask):
            self.mask = torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(self.path_mask)))

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

    def __init__(self, group : str, name: str, dataset : DatasetUtils, patch : Union[DatasetPatch, None], pre_transforms : list[Transform]) -> None:
        self.group = group
        self.name = name

        _dataset = dataset.getDataset(group, name)

        self.loaded = False
        self._shape = list(_dataset.shape[1:])

        self.cache_attribute = Attribute({k : torch.tensor(v) if isinstance(v, np.ndarray) else v for k, v in _dataset.attrs.items()})
        
        self.data : list[torch.Tensor] = list()
        for transformFunction in pre_transforms:
            self._shape = transformFunction.transformShape(self._shape, self.cache_attribute)

        self.patch = DatasetPatch(patch_size=patch.patch_size, overlap=patch.overlap, mask=patch.path_mask) if patch else DatasetPatch(self._shape, mask=patch.path_mask)
        self.patch.load(self._shape)
    
    def load(self, dataset: DatasetUtils, index : int, pre_transform : list[Transform], dataAugmentationsList : list[DataAugmentationsList]) -> None:
        if self.loaded:
            return
        
        assert self.name

        _dataset = dataset.getDataset(self.group, self.name)
        i = len(pre_transform)
        data = None
        for transformFunction in reversed(pre_transform):
            if isinstance(transformFunction, Save) and os.path.exists(transformFunction.save):
                with DatasetUtils(transformFunction.save if not transformFunction.save.endswith("/") else transformFunction.save[:-1]) as datasetUtils:
                    if datasetUtils.isExist(self.group, self.name):
                        _, dataset = datasetUtils.getDataset(self.group, self.name)
                        data, attrib = dataset_to_data(dataset)
                        self.cache_attribute.update(attrib)
                        break
            i-=1
        
        if i==0:
            data = np.empty(_dataset.shape, _dataset.dtype)
            _dataset.read_direct(data)

        data = torch.from_numpy(data)
        if len(pre_transform):
            for transformFunction in pre_transform[i:]:
                data = transformFunction(data, self.cache_attribute)
                if isinstance(transformFunction, Save):
                    with DatasetUtils(transformFunction.save if not transformFunction.save.endswith("/") else transformFunction.save[:-1], read=False, is_directory=transformFunction.save.endswith("/")) as datasetUtils:
                        datasetUtils.writeData(self.group, self.name, data.numpy(), self.cache_attribute)
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
            data = transformFunction(data, self.cache_attribute)
        return data

    def __len__(self) -> int:
        return len(self.patch)
  
    