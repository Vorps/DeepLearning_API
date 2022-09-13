import importlib
from typing import Dict, List, Optional, Tuple
import torch

import numpy as np
import SimpleITK as sitk
from DeepLearning_API.utils import _getModule, NeedDevice
from DeepLearning_API.config import config
from abc import ABC, abstractmethod
import torch.nn.functional as F

class Transform(NeedDevice, ABC):
    
    def __init__(self, save = None) -> None:
        self.save = save
    
    def transformMetaData(self, shape: List[int], cache_attribute: Dict[str, torch.Tensor]) -> List[int]:
        return shape

    @abstractmethod
    def __call__(self, input : torch.Tensor, cache_attribute: Dict[str, torch.Tensor]) -> torch.Tensor:
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
    def __init__(self, min_value : float = -1024, max_value : float = 1024) -> None:
        super().__init__()
        assert max_value > min_value
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, input : torch.Tensor, cache_attribute: Dict[str, torch.Tensor]) -> torch.Tensor:
        input[torch.where(input < self.min_value)] = self.min_value
        input[torch.where(input > self.max_value)] = self.max_value
        return input

class Normalize(Transform):

    @config("Normalize")
    def __init__(self, lazy : bool = False, min_value : float = -1, max_value : float = 1) -> None:
        super().__init__()
        assert max_value > min_value
        self.lazy = lazy
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, input : torch.Tensor, cache_attribute: Dict[str, torch.Tensor]) -> torch.Tensor:
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

class Standardize(Transform):

    @config("Standardize")
    def __init__(self, lazy : bool = False) -> None:
        super().__init__()
        self.lazy = lazy

    def __call__(self, input : torch.Tensor, cache_attribute: Dict[str, torch.Tensor]) -> torch.Tensor:
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

class DeNormalize(Transform):

    @config("DeNormalize")
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, input : torch.Tensor, cache_attribute: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_min = None
        input_max = None
        
        min_value = torch.min(input)
        max_value = torch.max(input)

        if "Min" in cache_attribute:
            input_min = cache_attribute["Min"]
        if "Max" in cache_attribute:
            input_max = cache_attribute["Max"]

        return (input - min_value)*(input_max-input_min)/(max_value-min_value)+input_min if input_min is not None and input_max is not None else input
         
class DeStandardize(Transform):

    @config("DeStandardize")
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, input : torch.Tensor, cache_attribute: Dict[str, torch.Tensor]) -> torch.Tensor:
        mean = None
        std = None
        if "Mean" in cache_attribute:
            mean = cache_attribute["Mean"]
        if "Std" in cache_attribute:
            std = cache_attribute["Std"]
        return input * std + mean if mean is not None and std is not None else input

class TensorCast(Transform):

    @config("TensorCast")
    def __init__(self, dtype : str = "default:float32,int64,int16") -> None:
        super().__init__()
        self.dtype : torch.dtype = getattr(torch, dtype)

    def __call__(self, input : torch.Tensor, cache_attribute: Dict[str, torch.Tensor]) -> torch.Tensor:
        return input.type(self.dtype)

class Padding(Transform):

    @config("Padding")
    def __init__(self, padding : List[int] = [0,0], mode : str = "default:constant,reflect,replicate,circular", dim: int = 0) -> None:
        super().__init__()
        self.padding = padding
        self.mode = mode
        self.dim = dim

    def __call__(self, input : torch.Tensor, cache_attribute: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.pad(input, tuple([0, 0]*(len(input.shape)-self.dim-1)+self.padding+[0, 0]*(self.dim)), self.mode.split(":")[0], float(self.mode.split(":")[1]) if len(self.mode.split(":")) == 2 else 0)

    def transformMetaData(self, shape: List[int], cache_attribute: Dict[str, torch.Tensor]) -> List[int]:
        if self.dim > 0:
            shape[self.dim-1] += sum(self.padding)
            if "Origin" in cache_attribute and "Spacing" in cache_attribute and "Direction" in cache_attribute:
                matrix = cache_attribute["Direction"].reshape((3,3))
                cache_attribute["Origin"] = cache_attribute["Origin"].dot(matrix)
                cache_attribute["Origin"][-self.dim] -= self.padding[0]*cache_attribute["Spacing"][-self.dim]
                cache_attribute["Origin"] = cache_attribute["Origin"].dot(np.linalg.inv(matrix))
        return shape

class Resample(Transform):

    def __init__(self, size: List[int], save: str) -> None:
        super().__init__(save)
        self.size = size

    def __call__(self, input : torch.Tensor, cache_attribute: Dict[str, torch.Tensor]) -> torch.Tensor:
        if input.dtype == torch.uint8:
            mode = "nearest"
        elif len(input.shape) < 4:
            mode = "bilinear"
        else:
            mode = "trilinear"
        return F.interpolate(input.type(torch.float32).to(self.device).unsqueeze(0), size=tuple(self.size), mode=mode, align_corners=False).squeeze(0).cpu()

    def transformMetaData(self, shape: List[int], cache_attribute: Dict[str, torch.Tensor]) -> List[int]:
        if "Spacing" in cache_attribute:
            cache_attribute["Spacing"] = torch.flip(torch.tensor(shape)/torch.tensor(self.size)*torch.flip(cache_attribute["Spacing"], dims=[0]), dims=[0])
        return self.size

class ResampleIsotropic(Resample):

    @config("ResampleIsotropic") 
    def __init__(self, spacing : List[float] = [1., 1., 1.], save : str = "name.h5") -> None:
        super().__init__([], save)
        self.spacing = spacing
        
    def transformMetaData(self, shape: List[int], cache_attribute: Dict[str, torch.Tensor]) -> List[int]:
        assert "Spacing" in cache_attribute, "Error no spacing"
        resize_factor = torch.tensor(self.spacing)/np.flip(cache_attribute["Spacing"])
        self.size =  [int(x) for x in (shape * 1/resize_factor)]
        cache_attribute["Spacing"] = np.flip(self.spacing)
        return self.size
        
class ResampleResize(Resample):

    @config("ResampleResize")
    def __init__(self, size : List[int] = [100,512,512], save : str = "name.h5") -> None:
        super().__init__(size, save)


class Mask(Transform):
    
    @config("Mask")
    def __init__(self, path : str = "default:./default.mha") -> None:
        super().__init__()
        self.mask = torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(path))).unsqueeze(0)
        
    def __call__(self, input : torch.Tensor, cache_attribute: Dict[str, torch.Tensor]) -> torch.Tensor:
        input[torch.where(self.mask != 1)] = self.mask[torch.where(self.mask != 1)].type(input.dtype)
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
