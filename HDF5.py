from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import SimpleITK as sitk
import h5py
import numpy as np
import torch
import os
import torch.nn.functional as F

from DeepLearning_API.config import config
from DeepLearning_API.utils import DatasetUtils, dataset_to_data
from DeepLearning_API.transform import Transform
from DeepLearning_API.augmentation import DataAugmentation


import itertools

class Patch():

    @config("Patch")
    def __init__(self, patch_size : List[int] = [128, 256, 256], overlap : Optional[List[int]] = None) -> None:
        self.patch_size = np.asarray(patch_size)
        self.overlap = overlap
        self.patch_slices : Optional[List[tuple[slice]]] = None
        self.shape: Optional[np.ndarray] = None
        self.dtype: Optional[torch.dtype] = None

    def load(self, shape : torch.Size) -> None:
        assert all(shape[-len(self.patch_size):] >= self.patch_size)
        self.patch_slices = []
        self.shape = np.asarray(shape)
        slices : List[List[slice]] = []
        if self.overlap is None:
            size = np.asarray([int(x) for x in np.ceil(shape/self.patch_size)])
            overlap_tmp = np.zeros(len(size), dtype=np.int)
            for i, s in enumerate(size):
                if s > 1:
                    overlap_tmp[i] = np.floor((np.mod(self.patch_size[i]-np.mod(shape[i], self.patch_size[i]), self.patch_size[i]))/(size[i]-1))
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

    def disassemble(self, data: torch.Tensor) -> List[torch.Tensor]:
        return [self.getData(data, i) for i in range(len(self))]

    def assemble(self, data_list: List[torch.Tensor]) -> torch.Tensor:
        if len(data_list) == 1 or self.patch_slices is None or self.shape is None or self.dtype is None:
            return data_list[0]

        patch_slice_shape = [slice.stop-slice.start for slice in self.patch_slices[0]]
        data_shape = list(data_list[0].shape[-len(patch_slice_shape):])
        isImage = patch_slice_shape == data_shape
        if isImage:
            result = torch.empty((np.asarray(data_list[0].shape[:-len(self.shape)])+self.shape), dtype=self.dtype)
            for patch_slice, data in zip(self.patch_slices, data_list):
                slices = []
                for max in data.shape[:-len(patch_slice)]:
                    slices.append(slice(max))
                slices += list(patch_slice)
                result[slices] = data
        else:
            result = torch.cat([data for data in data_list], dim=0)
        return result

    def __len__(self) -> int:
        return len(self.patch_slices) if self.patch_slices is not None else 1
        
class Dataset(ABC):

    def __init__(self, group : str, dataset : h5py.Dataset) -> None:
        self._dataset = dataset
        self.group = group
        self.name = self._dataset.name    
        self.loaded = False
        self.data = None
    
    @abstractmethod
    def load(self, index : int, pre_transform : List[Transform], dataAugmentations : List[DataAugmentation]) -> None:
        pass

    def unload(self) -> None:
        del self.data
        self.loaded = False

    @abstractmethod
    def getData(self, index : int, a : int, post_transforms : List[Transform]) -> torch.Tensor:
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        pass

class Image_Dataset(Dataset):

    IMAGE_ATTRS = ["Origin", "Spacing", "Direction"]

    def __init__(self, group : str, dataset : h5py.Dataset, patch : Patch, pre_transforms : List[Transform]) -> None:
        super().__init__(group, dataset)
        self._origin = self.getOrigin()
        self._spacing = self.getSpacing() 
        self._direction = self.getDirection()
        self._shape = self.getShape()
        self.patch = Patch(patch_size=patch.patch_size, overlap=patch.overlap)
        self.cache_attribute = {k : v for k, v in self._dataset.attrs.items() if k not in Image_Dataset.IMAGE_ATTRS}
        
        self.data : List[torch.Tensor] = list()
        for transformFunction in pre_transforms:
            transformFunction.loadDataset(self)
            self._shape = transformFunction.transformShape(self._shape)
        self.patch.load(self._shape)

    def getOrigin(self) -> np.ndarray:
        return self._origin if self.loaded else np.asarray(self._dataset.attrs["Origin"])

    def getSpacing(self) -> np.ndarray:
        return self._spacing if self.loaded else np.asarray(self._dataset.attrs["Spacing"])
    
    def getDirection(self) -> np.ndarray:
        return self._direction if self.loaded else np.asarray(self._dataset.attrs["Direction"])
    
    def getShape(self) -> torch.Size:
        return self._shape if self.loaded else torch.Size(tuple(self._dataset.shape))

    def setMetaData(self, origin, spacing, direction) -> None:
        self._origin = np.asarray(origin)
        self._spacing = np.asarray(spacing)
        self._direction = np.asarray(direction)

    def load(self, index : int, pre_transform : List[Transform], dataAugmentations : List[DataAugmentation]) -> None:
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
                                self.setMetaData(attrib["Origin"], attrib["Spacing"], attrib["Direction"])
                                break
            i-=1

        if i==0:
            data = np.empty(self._dataset.shape, self._dataset.dtype)
            self._dataset.read_direct(data)
        
        data = torch.from_numpy(data)
        if len(pre_transform):
            for transformFunction in pre_transform[i:]:
                transformFunction.loadDataset(self)
                data = transformFunction(data)
                if transformFunction.save is not None:
                    with DatasetUtils(transformFunction.save, read=False) as datasetUtils:
                        datasetUtils.writeImage(self.group, self.name, self.to_image(data.numpy()))

        self.loaded = True

        assert (np.asarray(data.shape[-len(self.getShape()):]) == self.getShape()).all()                
        
        self.data : List[torch.Tensor] = list()
        if not dataAugmentations:
            self.data.append(data)
        for dataAugmentation in dataAugmentations:
            dataAugmentation.state_init(index, self)
            for d in dataAugmentation(data):
                self.data.append(d)

    def getData(self, index : int, a : int, post_transforms : List[Transform]) -> torch.Tensor:
        data = self.patch.getData(self.data[a], index)
        for transformFunction in post_transforms:
            transformFunction.loadDataset(self)
            data = transformFunction(data)
        return data
    
    def to_image(self, data : np.ndarray, dtype = None) -> sitk.Image:
        image = sitk.GetImageFromArray(data.astype(self._dataset.dtype if dtype is None else dtype))
        image.SetOrigin(self._origin)
        image.SetSpacing(self._spacing)
        image.SetDirection(self._direction)
        return image
    
    def __len__(self) -> int:
        return len(self.patch)

class Scalaire_Dataset(Dataset):
    
    def __init__(self, group : str, dataset : h5py.Dataset) -> None:
        super().__init__(group, dataset)

    def load(self, index : int, pre_transform : List[Transform], dataAugmentations : List[DataAugmentation]) -> None:
        self.data = np.empty(self._dataset.shape, self._dataset.dtype)
        self._dataset.read_direct(self.data)
        self.data = torch.from_numpy(self.data)
        self.loaded = True

    def getData(self, index : int, a : int, post_transforms : List[Transform]) -> torch.Tensor:
        
        return self.data
    
    def __len__(self) -> int:
        return 1
    
class HDF5():

    def __init__(self, filename : str) -> None:
        self.filename = filename

    def __enter__(self):
        self.h5 = h5py.File(self.filename, 'r')
        return self
    
    def getSize(self, group):
        h5_group = self.h5[group]
        assert isinstance(h5_group, h5py.Group)
        return len([dataset for dataset in h5_group.values() if type(dataset) == h5py.Dataset])
    
    def __exit__(self, type, value, traceback):
        self.h5.close()

    @staticmethod
    def _islabel(dataset : h5py.Dataset):
        return not all(k in dataset.attrs.keys() for k in Image_Dataset.IMAGE_ATTRS)

    def getDatasets(self, path : str, index : List[int], patch : Patch, pre_transform : List[Transform], index_map : List[int], level : int, mapping : Dict[int, List[int]]) -> Tuple[List[int], Dict[int, List[int]], List[Dataset]]:
        hdh5_datasets = self.h5[path]
        assert isinstance(hdh5_datasets, h5py.Group)
        list_of_hdh5_datasets = [dataset for dataset in hdh5_datasets.values() if type(dataset) == h5py.Dataset]
        datasets : List[Dataset] = []
        it = 0
        for i in index:
            dataset = Scalaire_Dataset(path, list_of_hdh5_datasets[i]) if HDF5._islabel(list_of_hdh5_datasets[i]) else Image_Dataset(path, list_of_hdh5_datasets[i], patch, pre_transform)
            datasets.append(dataset)
            
            index_map = index_map.copy()
            if len(index_map) == level:
                index_map += [it]
            else:
                index_map[level] = it
            it += 1
            assert dataset.name
            group = dataset.name.split(".")[0]
            if group in self.h5:
                h5_group = self.h5[group]
                assert isinstance(h5_group, h5py.Group)
                index_map, mapping, result = self.getDatasets(group, list(range(len(h5_group))), patch, pre_transform, index_map, level+1, mapping)
                datasets.append(*result)
                it += 1
            else:
                mapping[len(mapping)] = index_map
        return index_map, mapping, datasets