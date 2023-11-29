from abc import ABC, abstractmethod
import SimpleITK as sitk
import h5py
import numpy as np
import torch
import os
import torch.nn.functional as F
from typing import Any, Iterator
from DeepLearning_API.config import config
from DeepLearning_API.utils import DatasetUtils, get_patch_slices_from_shape, Attribute
from DeepLearning_API.transform import Transform, Save
from DeepLearning_API.augmentation import DataAugmentationsList
from typing import Union
import itertools
import copy 

class PathCombine(ABC):

    def __init__(self) -> None:
        self.data: torch.Tensor = None
        self.dim: int = None

    def setPatchConfig(self, patch_size: list[int], overlap: list[int]):
        self.data = torch.ones([size-o*2 for size, o in zip(patch_size, overlap)])
        padding = []
        for o in reversed(overlap):
            padding.append(o)
            padding.append(o)
        self.dim = len([o for o in overlap if o > 0])
        self.data = self._setSides(self.data, padding)
        
        
        slices = [[slice(0, o), slice(-o, None)] if o > 0 else [slice(None, None)] for o in overlap]
        for i, s in enumerate(itertools.product(*slices)):
            self.data[s] = self._setCorners(self.data[s], i)
            
    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        return self.data.repeat([input.shape[0]]+[1]*(len(input.shape)-1)).to(input.device)*input

    @abstractmethod
    def _setCorners(self, data: torch.Tensor, i: int) -> torch.Tensor:
        pass
    
    @abstractmethod
    def _setSides(self, data: torch.Tensor, padding: list[int]):
        pass

class Mean(PathCombine):

    @config("Mean")
    def __init__(self) -> None:
        super().__init__()

    def _setCorners(self, data: torch.Tensor, i: int) -> torch.Tensor:
        return data/2
    
    def _setSides(self, data: torch.Tensor, padding: list[int]) -> torch.Tensor:
        return F.pad(data, padding, mode="constant", value=1/(2**(self.dim-1)))
    
class Cosinus(PathCombine):

    @config("Cosinus")
    def __init__(self) -> None:
        super().__init__()
        self.overlap: int = None 

    def setPatchConfig(self, patch_size: list[int], overlap: list[int]):
        assert len(np.unique(np.asarray([o for o in overlap if o > 0]))) == 1, "Overlap must be the same in each dimension"
        self.overlap = [o for o in overlap if o > 0][0]
        super().setPatchConfig(patch_size, overlap)
    
    def _setCorners(self, data: torch.Tensor, i: int) -> torch.Tensor:
        result = torch.zeros_like(data)
        suport_invert = list(reversed(range(self.overlap)))
        suport =  list(range(self.overlap))

        func = lambda l: torch.tensor(np.asarray([np.cos(np.pi*1/(2*(self.overlap-1))*(x))**2/2 for x in l]))

        if i == 0 or i == 2:
            for i in range(self.overlap):
                result[:, i, :] = func(suport_invert)   
        elif i == 1 or i == 3:
            for i in range(self.overlap):
                result[:, i, :] = func(suport)   

        return result
    
    def _function_sides(self, x):
        return np.clip(np.cos(np.pi*1/(2*(self.overlap-1))*((x-1) if x > 0 else 0))**2, 0, 1)

    def _setSides(self, data: torch.Tensor, padding: list[int]) -> torch.Tensor:
        data = F.pad(self.data, padding, mode="constant", value=0)
        image = sitk.GetImageFromArray(np.asarray(data, dtype=np.uint8))
        danielssonDistanceMapImageFilter = sitk.DanielssonDistanceMapImageFilter()
        distance = torch.tensor(sitk.GetArrayFromImage(danielssonDistanceMapImageFilter.Execute(image)))
        return distance.apply_(self._function_sides)
        
class Accumulator():

    def __init__(self, patch_slices: list[tuple[slice]], patch_size: list[int], patchCombine: Union[PathCombine, None] = None, batch: bool = True) -> None:
        self._layer_accumulator: list[Union[torch.Tensor, None]] = [None for i in range(len(patch_slices))]
        self.patch_slices = patch_slices
        self.shape = max([[v.stop for v in patch] for patch in patch_slices])
        self.patch_size = patch_size
        self.patchCombine = patchCombine
        self.batch = batch

    def addLayer(self, index: int, layer: torch.Tensor) -> None:
        self._layer_accumulator[index] = layer
    
    def isFull(self) -> bool:
        return len(self.patch_slices) == len([v for v in self._layer_accumulator if v is not None])

    def assemble(self) -> torch.Tensor:
        if all([self._layer_accumulator[0].shape[i-len(self.patch_size)] == size for i, size in enumerate(self.patch_size)]): 
            N = 2 if self.batch else 1
            result = torch.zeros((list(self._layer_accumulator[0].shape[:N])+list(self.shape)), dtype=self._layer_accumulator[0].dtype).to(self._layer_accumulator[0].device)
            for patch_slice, data in zip(self.patch_slices, self._layer_accumulator):
                slices_dest = tuple([slice(result.shape[i]) for i in range(N)] + list(patch_slice))
                slices_source = [slice(result.shape[i]) for i in range(N)]
                i = 0
                for s in patch_slice:
                    if s.stop-s.start == 1:
                        slices_source += [slice(0, s.stop-s.start)]
                    else:
                        slices_source += [slice((data.shape[i+N]-(s.stop-s.start))//2, (data.shape[i+N]+(s.stop-s.start))//2) ]
                        i += 1
                        
                slices_source = tuple(slices_source) 
                for dim, s in enumerate(patch_slice):
                    if s.stop-s.start == 1:
                        data = data.unsqueeze(dim=dim+N)
                if self.patchCombine is not None:
                    result[slices_dest] += self.patchCombine(data)[slices_source]
                else:
                    result[slices_dest] = data[slices_source]
        else:
            result = torch.cat(tuple(self._layer_accumulator), dim=0)
        self._layer_accumulator.clear()
        return result

class Patch(ABC):

    def __init__(self, patch_size: list[int], overlap: Union[list[int], None], path_mask: Union[str, None] = None, padValue: float = 0, extend_slice: int = 0) -> None:
        self.patch_size = patch_size
        self.overlap = overlap
        self._patch_slices : dict[int, list[tuple[slice]]] = {}
        self._nb_patch_per_dim: dict[int, list[tuple[int, bool]]] = {}
        self.path_mask = path_mask
        self.mask = None
        self.padValue = padValue
        self.extend_slice = extend_slice
        if self.path_mask is not None:
            if os.path.exists(self.path_mask):
                self.mask = torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(self.path_mask)))
            else:
                raise NameError('Mask file not found')
            
    def load(self, shape : dict[int, list[int]], a: int = 0) -> None:
        self._patch_slices[a], self._nb_patch_per_dim[a] = get_patch_slices_from_shape(self.patch_size, shape, self.overlap)

    def getPatch_slices(self, a: int = 0):
        return self._patch_slices[a]
    
    @abstractmethod
    def getData(self, data : torch.Tensor, index : int, a: int, isInput: bool) -> torch.Tensor:
        pass
    
    def _getData(self, data : torch.Tensor, index : int, a: int, isInput: bool) -> list[torch.Tensor]:        
        slices_pre = []
        for max in data.shape[:-len(self.patch_size)]:
            slices_pre.append(slice(max))
        datas = []
        extend_slice = self.extend_slice if isInput else 0
        for i in range(-extend_slice//2, int(np.ceil(extend_slice/2))+1):
            s = slice(self._patch_slices[a][index][0].start+i, self._patch_slices[a][index][0].stop+i)
            if s.start < 0 or s.stop > data.shape[len(slices_pre)]:
                data_sliced = torch.full(tuple(list(data.shape[:-len(self.patch_size)])+self.patch_size), 0 if data.dtype == torch.uint8 and self.padValue < 0 else self.padValue, dtype=data.dtype).to(data.device)
            else:
                slices =  [s] + list(self._patch_slices[a][index][1:])
                data_sliced = data[slices_pre+slices]
                padding = []
                for dim_it, _slice in enumerate(reversed(slices)):
                    p = 0 if _slice.start+self.patch_size[-dim_it-1] <= data.shape[-dim_it-1] else self.patch_size[-dim_it-1]-(data.shape[-dim_it-1]-_slice.start)
                    padding.append(p//2)
                    padding.append(int(np.ceil(p/2)))
                data_sliced = F.pad(data_sliced, tuple(padding), "constant", 0 if data_sliced.dtype == torch.uint8 and self.padValue < 0 else self.padValue)
                if self.mask is not None:
                    outside = torch.ones_like(data_sliced)*(0 if data_sliced.dtype == torch.uint8 and self.padValue < 0 else self.padValue)
                    data_sliced = torch.where(self.mask == 0, outside, data_sliced)
            
            for d in [i for i, v in enumerate(reversed(self.patch_size)) if v == 1]:
                data_sliced = torch.squeeze(data_sliced, dim = len(data_sliced.shape)-d-1)
            datas.append(data_sliced)
        return datas

    def getSize(self, a: int = 0) -> int:
        return len(self._patch_slices[a])

class DatasetPatch(Patch):

    @config("Patch")
    def __init__(self, patch_size : list[int] = [128, 256, 256], overlap : Union[list[int], None] = None, mask: Union[str, None] = None, padValue: float = 0, extend_slice: int = 0) -> None:
        super().__init__(patch_size, overlap, mask, padValue, extend_slice)

    def getData(self, data : torch.Tensor, index : int, a: int, isInput: bool) -> torch.Tensor:      
        data = torch.cat(super()._getData(data, index, a, isInput), dim=0)
        return data
    
class ModelPatch(Patch):

    @config("Patch")
    def __init__(self, patch_size : list[int] = [128, 256, 256], overlap : Union[list[int], None] = None, patchCombine: Union[str, None] = None, mask: Union[str, None] = None, padValue: float = 0, extend_slice: int = 0) -> None:
        super().__init__(patch_size, overlap, mask, padValue, extend_slice)
        self.patchCombine = patchCombine

    def disassemble(self, *dataList: torch.Tensor) -> Iterator[list[torch.Tensor]]:
        for i in range(self.getSize()):
            yield [self.getData(data, i, 0, True) for data in dataList]

    def getData(self, data : torch.Tensor, index : int, a: int, isInput: bool) -> torch.Tensor:      
        data = torch.cat(super()._getData(data, index, a, isInput), dim=1)
        return data

class Dataset():

    def __init__(self, index: int, group_src: str, group_dest : str, name: str, datasetUtils : DatasetUtils, patch : Union[DatasetPatch, None], pre_transforms : list[Transform], dataAugmentationsList : list[DataAugmentationsList]) -> None:
        self.group_src = group_src
        self.group_dest = group_dest
        self.name = name
        self.index = index
        self.datasetUtils = datasetUtils
        self.loaded = False
        self.cache_attributes: list[Attribute] = []
        _shape, cache_attribute =  self.datasetUtils.getInfos(self.group_src, name)
        self.cache_attributes.append(cache_attribute)
        _shape = list(_shape[1:])
        
        self.data : list[torch.Tensor] = list()
        for transformFunction in pre_transforms:
            _shape = transformFunction.transformShape(_shape, cache_attribute)
        
        self.patch = DatasetPatch(patch_size=patch.patch_size, overlap=patch.overlap, mask=patch.path_mask, padValue=patch.padValue, extend_slice=patch.extend_slice) if patch else DatasetPatch(_shape)
        self.patch.load(_shape, 0)
        self.s = _shape
        i = 1
        for dataAugmentations in dataAugmentationsList:
            shape = []
            caches_attribute = []
            for _ in range(dataAugmentations.nb):
                shape.append(_shape)
                caches_attribute.append(copy.deepcopy(cache_attribute))

            for dataAugmentation in dataAugmentations.dataAugmentations:
                shape = dataAugmentation.state_init(self.index, shape, caches_attribute)
            for it, s in enumerate(shape):
                self.cache_attributes.append(caches_attribute[it])
                self.patch.load(s, i)
                i+=1
        self.cache_attributes_bak = copy.deepcopy(self.cache_attributes)

    def load(self, pre_transform : list[Transform], dataAugmentationsList : list[DataAugmentationsList]) -> None:
        if self.loaded:
            return
        i = len(pre_transform)
        data = None
        for transformFunction in reversed(pre_transform):
            if isinstance(transformFunction, Save):
                filename, format = transformFunction.save.split(":")
                datasetUtils = DatasetUtils(filename, format)
                if datasetUtils.isDatasetExist(self.group_dest, self.name):
                    data, attrib = datasetUtils.readData(self.group_dest, self.name)
                    self.cache_attributes[0].update(attrib)
                    break
            i-=1
        
        if i==0:
            data, _ = self.datasetUtils.readData(self.group_src, self.name)

        data = torch.from_numpy(data)
        if len(pre_transform):
            for transformFunction in pre_transform[i:]:
                data = transformFunction(self.name, data, self.cache_attributes[0])
                if isinstance(transformFunction, Save):
                    filename, format = transformFunction.save.split(":")
                    datasetUtils = DatasetUtils(filename, format)
                    datasetUtils.write(self.group_dest, self.name, data.numpy(), self.cache_attributes[0])
        self.data : list[torch.Tensor] = list()
        
        self.data.append(data)
            
        for dataAugmentations in dataAugmentationsList:
            a_data = [data.clone() for _ in range(dataAugmentations.nb)]
            for dataAugmentation in dataAugmentations.dataAugmentations:
                a_data = dataAugmentation(self.index, a_data)
            
            for d in a_data:
                self.data.append(d)
        self.loaded = True
             
    def unload(self) -> None:
        del self.data
        self.cache_attributes = copy.deepcopy(self.cache_attributes_bak)
        self.loaded = False
    
    def getData(self, index : int, a : int, post_transforms : list[Transform], isInput: bool) -> torch.Tensor:
        data = self.patch.getData(self.data[a], index, a, isInput)
        for transformFunction in post_transforms:
            data = transformFunction(self.name, data, self.cache_attributes[a])
        return data

    def getSize(self, a: int) -> int:
        return self.patch.getSize(a)
  
    