from typing import Dict, List, Tuple
import SimpleITK as sitk
import h5py
import numpy as np
import torch
import os

from DeepLearning_API import DatasetUtils, dataset_to_data, Transform

import itertools


class HDF5():

    class Dataset():

        def __init__(self, group : str, dataset : h5py.Dataset) -> None:
            self._dataset = dataset
            self.group = group
            
            self.name = self._dataset.name    
            self.loaded = False
            self._origin = self.getOrigin()
            self._spacing = self.getSpacing() 
            self._direction = self.getDirection()
            self._shape = self.getShape()
            self.cache_attribute = {}

            self.patch_slices : List[tuple[slice]] = None

        @staticmethod
        def _getPatch_slices(shape : np.ndarray, patch_size : np.ndarray, overlap : List[int]) -> List[Tuple[slice]]:
            patch_slices : List[tuple[slice]] = []
            slices : List[List[slice]] = []
            if overlap is None:
                size = np.asarray([int(x) for x in np.ceil(shape/patch_size)])
                overlap_tmp = np.zeros(len(size), dtype=np.int)
                for i, s in enumerate(size):
                    if s > 1:
                        overlap_tmp[i] = np.floor((np.mod(patch_size[i]-np.mod(shape[i], patch_size[i]), patch_size[i]))/(size[i]-1))
            else:
                overlap_tmp = overlap

            for dim in range(len(shape)):
                slices.append([])
                index = 0
                while True:
                    start = (patch_size[dim]-overlap_tmp[dim])*index
                    end = start + patch_size[dim]
                    if end >= shape[dim]:
                        end = shape[dim]
                        
                        if overlap is None and end-patch_size[dim] >= 0:
                            start = end-patch_size[dim]   
                        slices[dim].append(slice(start,end))
                        break
                    slices[dim].append(slice(start,end))
                    index += 1
            for chunk in itertools.product(*slices):
                patch_slices.append(tuple(chunk))
            return patch_slices


        def init(self, patch_size : np.ndarray, overlap : List[int], pre_transforms : List[Transform]):
            for transformFunction in pre_transforms:
                transformFunction.loadDataset(self)
                self._shape = transformFunction.transformShape(self._shape)
            self.patch_slices = HDF5.Dataset._getPatch_slices(self._shape, patch_size, overlap)
            return self

        def getOrigin(self) -> np.ndarray:
            return self._origin if self.loaded else np.asarray(self._dataset.attrs["Origin"])

        def getSpacing(self) -> np.ndarray:
            return self._spacing if self.loaded else np.asarray(self._dataset.attrs["Spacing"])
        
        def getDirection(self) -> np.ndarray:
            return self._direction if self.loaded else np.asarray(self._dataset.attrs["Direction"])
        
        def getShape(self) -> np.ndarray:
            return self._shape if self.loaded else np.asarray(self._dataset.shape)

        def setMetaData(self, origin, spacing, direction) -> None:
            self._origin = np.asarray(origin)
            self._spacing = np.asarray(spacing)
            self._direction = np.asarray(direction)
        
        
        def load(self, pre_transform : List[Transform]) -> None:
            if not self.loaded:
                i = len(pre_transform)
                for transformFunction in reversed(pre_transform):
                    if transformFunction.save is not None and os.path.exists(transformFunction.save):
                        with DatasetUtils(transformFunction.save) as datasetUtils:
                            if self.group in datasetUtils.h5:
                                if self.name in datasetUtils.h5[self.group]:
                                    dataset = datasetUtils.h5[self.group][self.name]
                                    self.data, origin, spacing, direction = dataset_to_data(dataset)
                                    self.setMetaData(origin, spacing, direction)
                                    break
                    i-=1

                if i==0:
                    self.data = np.empty(self._dataset.shape, self._dataset.dtype)
                    self._dataset.read_direct(self.data)

                self.data = torch.from_numpy(self.data)
                
                if len(pre_transform):
                    for transformFunction in pre_transform[i:]:
                        self.data = transformFunction.loadDataset(self)(self.data)
                        if transformFunction.save is not None:
                            with DatasetUtils(transformFunction.save, read=False) as datasetUtils:
                                datasetUtils.writeImage(self.group, self.name, self.to_image(self.data.numpy()))

                self.loaded = True

                assert (np.asarray(self.data.shape[-len(self.getShape()):]) == self.getShape()).all()                
                

        def unload(self) -> None:
            del self.data
            self.loaded = False

        def getData(self, patch_size : np.ndarray, index : int) -> torch.Tensor:
            slices = []
            for max in self.data.shape[:-len(self.patch_slices[index])]:
                slices.append(slice(max))
            i = len(slices)

            slices += list(self.patch_slices[index])
            data = self.data[slices]
            padding = []
            for dim_it, _slice in enumerate(reversed(self.patch_slices[index])):
                dim = len(self.getShape())-dim_it-1
                padding.append(0)
                padding.append(0 if _slice.start+patch_size[dim] <= self.getShape()[dim] else _slice.start+patch_size[dim]-_slice.stop)
            
            data = torch.nn.functional.pad(data, tuple(padding), "constant", 0)
            
            for d in reversed(np.where(patch_size == 1)[0]+i):
                data = torch.squeeze(data, dim = d)
            
            return data
        
        def setData(self, data, index : int) -> None:
            self.data[self.patch_slices[index]] = data
        
        def __len__(self) -> int:
            return len(self.patch_slices)

        def to_image(self, data : np.ndarray) -> sitk.Image:
            image = sitk.GetImageFromArray(data.astype(self._dataset.dtype))
            image.SetOrigin(self._origin)
            image.SetSpacing(self._spacing)
            image.SetDirection(self._direction)
            return image
    

    def __init__(self, filename : str) -> None:
        self.filename = filename

    def __enter__(self) -> None:
        self.h5 = h5py.File(self.filename, 'r')
        return self
    
    def getSize(self, group):
        return len([dataset for dataset in self.h5[group].values() if type(dataset) == h5py.Dataset])
    
    def __exit__(self, type, value, traceback) -> None:
        self.h5.close()

    def getDatasets(self, path : str, index : List[int], patch_size : np.ndarray, overlap : np.ndarray, pre_transform : Dict[str, Transform], index_map : List[int], level : int, mapping : Dict[int, List[int]]) -> List[Dataset]:
        hdh5_datasets = self.h5[path]
        list_of_hdh5_datasets = [dataset for dataset in hdh5_datasets.values() if type(dataset) == h5py.Dataset]
        datasets : List[HDF5.Dataset] = []
        it = 0
        for i in index:
            dataset = list_of_hdh5_datasets[i]
            datasets.append(HDF5.Dataset(path, dataset).init(patch_size, overlap, pre_transform))
            index_map = index_map.copy()
            if len(index_map) == level:
                index_map += [it]
            else:
                index_map[level] = it
            it += 1
            group = dataset.name.split(".")[0]
            if group in self.h5:
                index_map, mapping, result = self.getDatasets(group, range(len(self.h5[group])), patch_size, overlap, pre_transform, index_map, level+1, mapping)
                datasets.append(result)
                it += 1
            else:
                mapping[len(mapping)] = index_map
        return index_map, mapping, datasets