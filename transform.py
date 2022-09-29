import importlib
from operator import le
from typing import Any, Dict, List
import torch

import numpy as np
import SimpleITK as sitk
from DeepLearning_API.utils import _getModule, NeedDevice, Attribute
from DeepLearning_API.config import config
from abc import ABC, abstractmethod
import torch.nn.functional as F


class Transform(NeedDevice, ABC):
    
    def __init__(self, save = None) -> None:
        self.save = save
    
    def transformShape(self, shape: List[int]) -> List[int]:
        return shape

    @abstractmethod
    def __call__(self, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        pass

    @abstractmethod
    def inverse(self, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        pass

class TransformLoader:

    @config(None)
    def __init__(self) -> None:
        pass
    
    def getTransform(self, classpath : str, DL_args : str) -> Transform:
        module, name = _getModule(classpath, "transform")
        return getattr(importlib.import_module(module), name)(config = None, DL_args = DL_args)

class Clip(Transform):

    @config("Clip")
    def __init__(self, min_value : float = -1024, max_value : float = 1024, saveClip: bool = False) -> None:
        super().__init__()
        assert max_value > min_value
        self.min_value = min_value
        self.max_value = max_value
        self.saveClip = saveClip

    def __call__(self, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        input[torch.where(input < self.min_value)] = self.min_value
        input[torch.where(input > self.max_value)] = self.max_value
        if self.saveClip:
            cache_attribute["Min"] = self.min_value
            cache_attribute["Max"] = self.max_value
        return input

    def inverse(self, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return input

class Normalize(Transform):

    @config("Normalize")
    def __init__(self, lazy : bool = False, min_value : float = -1, max_value : float = 1) -> None:
        super().__init__()
        assert max_value > min_value
        self.lazy = lazy
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "Min" not in cache_attribute:
            cache_attribute["Min"] = torch.min(input)
        if "Max" not in cache_attribute:
            cache_attribute["Max"] = torch.max(input)

        if self.lazy:
            result = input
        else:
            input_min = cache_attribute["Min"]
            input_max = cache_attribute["Max"]

            norm = input_max-input_min
            assert norm != 0
            result = (self.max_value-self.min_value)*(input - input_min) / norm + self.min_value
        return result
    
    def inverse(self, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        min_value = torch.min(input)
        max_value = torch.max(input)
        input_min = cache_attribute.pop("Min")
        input_max = cache_attribute.pop("Max")
        return (input - min_value)*(input_max-input_min)/(max_value-min_value)+input_min

class Standardize(Transform):

    @config("Standardize")
    def __init__(self, lazy : bool = False) -> None:
        super().__init__()
        self.lazy = lazy

    def __call__(self, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "Mean" not in cache_attribute:
            cache_attribute["Mean"] = torch.mean(input.type(torch.float32))
        if "Std" not in cache_attribute:
            cache_attribute["Std"] = torch.std(input.type(torch.float32))

        if self.lazy:
            return input
        else:
            mean = cache_attribute["Mean"]
            std = cache_attribute["Std"]
            assert std > 0
            return (input - mean) / std
        
    def inverse(self, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        mean = cache_attribute.pop("Mean")
        std = cache_attribute.pop("Std")
        return input * std + mean

class TensorCast(Transform):

    @config("TensorCast")
    def __init__(self, dtype : str = "default:float32,int64,int16") -> None:
        super().__init__()
        self.dtype : torch.dtype = getattr(torch, dtype)

    def __call__(self, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        cache_attribute["dtype"] = input.dtype
        return input.type(self.dtype)
    
    def inverse(self, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return input.type(cache_attribute.pop("dtype"))

class Padding(Transform):

    @config("Padding")
    def __init__(self, padding : List[int] = [0,0], mode : str = "default:constant,reflect,replicate,circular", dim: int = 0) -> None:
        super().__init__()
        self.padding = padding
        self.mode = mode
        self.dim = dim

    def __call__(self, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "Origin" in cache_attribute and "Spacing" in cache_attribute and "Direction" in cache_attribute and self.dim > 0:
            origin = cache_attribute["Origin"]
            matrix = cache_attribute["Direction"].reshape((len(origin),len(origin)))
            origin = origin.dot(matrix)
            origin[-self.dim+1] -= self.padding[0]*cache_attribute["Spacing"][-self.dim+1]
            cache_attribute["Origin"] = origin.dot(np.linalg.inv(matrix))

        return F.pad(input, tuple([0, 0]*(len(input.shape)-self.dim-1)+self.padding+[0, 0]*(self.dim)), self.mode.split(":")[0], float(self.mode.split(":")[1]) if len(self.mode.split(":")) == 2 else 0)

    def transformShape(self, shape: List[int]) -> List[int]:
        if self.dim > 0:
            shape[self.dim-1] += sum(self.padding)
        return shape

    def inverse(self, input : torch.Tensor, cache_attribute: Dict[str, Any]) -> torch.Tensor:
        if "Origin" in cache_attribute and "Spacing" in cache_attribute and "Direction" in cache_attribute and self.dim > 0:
            cache_attribute.pop("Origin")
        slices = []
        for i, shape in enumerate(input.shape):
            if self.dim == i:
                slices.append(slice(self.padding[0], shape-self.padding[1]))
            else:
                slices.append(slice(0, shape))

        return input[slices]

class Squeeze(Transform):

    @config("Squeeze")
    def __init__(self, dim: int) -> None:
        super().__init__(None)
        self.dim = dim
    
    def __call__(self, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return input.squeeze(self.dim)

    def inverse(self, input : torch.Tensor, cache_attribute: Dict[str, Any]) -> torch.Tensor:
        return input.unsqueeze(self.dim)

class Resample(Transform, ABC):

    def __init__(self, save: str) -> None:
        super().__init__(save)

    def _resample(self, input: torch.Tensor, size: List[int]) -> torch.Tensor:
        if input.dtype == torch.uint8:
            mode = "nearest"
        elif len(input.shape) < 4:
            mode = "bilinear"
        else:
            mode = "trilinear"
        return F.interpolate(input.type(torch.float32).to(self.device).unsqueeze(0), size=tuple(size), mode=mode, align_corners=False).squeeze(0).type(input.dtype).cpu()

    @abstractmethod
    def __call__(self, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        pass
    
    @abstractmethod
    def transformShape(self, shape: List[int]) -> List[int]:
        pass
    
    def inverse(self, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        shape = cache_attribute.pop("Shape")
        if "Spacing" in cache_attribute:
            cache_attribute.pop("Spacing")
        return self._resample(input, shape)

class ResampleIsotropic(Resample):

    @config("ResampleIsotropic") 
    def __init__(self, spacing : List[float] = [1., 1., 1.], save : str = "name.h5") -> None:
        super().__init__(save)
        self.spacing = spacing
        
    def transformShape(self, shape: List[int]) -> List[int]:
        assert "Spacing" in cache_attribute, "Error no spacing"
        resize_factor = torch.tensor(self.spacing)/np.flip(cache_attribute["Spacing"])
        return  [int(x) for x in (shape * 1/resize_factor)]

    def __call__(self, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        assert "Spacing" in cache_attribute, "Error no spacing"
        resize_factor = torch.tensor(self.spacing)/np.flip(cache_attribute["Spacing"])
        cache_attribute["Spacing"] = np.flip(self.spacing)
        cache_attribute["Shape"] = list(input.shape[1:])
        self._resample(input, [int(x) for x in (input.shape[1:] * 1/resize_factor)])

class ResampleResize(Resample):

    @config("ResampleResize")
    def __init__(self, size : List[int] = [100,512,512], save : str = "name.h5") -> None:
        super().__init__(save)
        self.size = size

    def __call__(self, input: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "Spacing" in cache_attribute:
            cache_attribute["Spacing"] = torch.flip(torch.tensor(list(input.shape[1:]))/torch.tensor(self.size)*torch.flip(cache_attribute["Spacing"], dims=[0]), dims=[0])
        cache_attribute["Shape"] = list(input.shape[1:])
        return self._resample(input, self.size)

class Mask(Transform):
    
    @config("Mask")
    def __init__(self, path : str = "default:./default.mha") -> None:
        super().__init__()
        self.mask = torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(path))).unsqueeze(0)
        
    def __call__(self, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        input[torch.where(self.mask != 1)] = self.mask[torch.where(self.mask != 1)].type(input.dtype)
        return input

    def inverse(self, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return input

"""class NConnexeLabel(Transform):

    @config("NConnexeLabel") 
    def __init__(self, save=None) -> None:
        super().__init__(save)
    
    def __call__(self, input : torch.Tensor) -> torch.Tensor:
        nb_labels = len(torch.unique(input))-1
        result = torch.zeros_like(input)

        connectedComponentImageFilter = sitk.ConnectedComponentImageFilter()
        labelShapeStatisticsImageFilter = sitk.LabelShapeStatisticsImageFilter()
        for i in range(nb_labels):
            data = np.copy(input.numpy())
            data[np.where(data != i+1)] = 0
            connectedComponentImage = connectedComponentImageFilter.Execute(self.dataset.to_image(data))
            labelShapeStatisticsImageFilter.Execute(connectedComponentImage)
            stats = {label: labelShapeStatisticsImageFilter.GetNumberOfPixels(label) for label in labelShapeStatisticsImageFilter.GetLabels()}
            true_label = max(stats)

            tmp = sitk.GetArrayFromImage(connectedComponentImage)
            tmp[np.where(tmp != true_label)] = 0
            tmp[np.where(tmp == true_label)] = i+1
            result = result + torch.from_numpy(tmp.astype(np.uint8)).type(torch.uint8)
        return result

class ArgMax(Transform):

    @config("ArgMax") 
    def __init__(self, save=None) -> None:
        super().__init__(save)
    
    def __call__(self, input : torch.Tensor) -> torch.Tensor:
        return torch.argmax(input, dim=0).type(torch.uint8)



class MorphologicalClosing(Transform):

    @config("MorphologicalClosing")
    def __init__(self, radius : int = 2, save=None) -> None:
        super().__init__(save)
        self.binaryMorphologicalClosingImageFilter = sitk.BinaryMorphologicalClosingImageFilter()
        self.binaryMorphologicalClosingImageFilter.SetKernelRadius(radius)

    def __call__(self, input : torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(input)
        nb_labels = len(torch.unique(input))-1
        for i in range(nb_labels):
            data = np.copy(input.numpy())
            data[np.where(data != i+1)] = 0
            data[np.where(data == i+1)] = 1
            result_tmp = torch.from_numpy(sitk.GetArrayFromImage(self.binaryMorphologicalClosingImageFilter.Execute(self.dataset.to_image(data))))*(i+1)
            result[np.where(result_tmp == i+1)] = 0
            result += result_tmp
        return result"""
