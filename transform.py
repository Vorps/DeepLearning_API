import importlib
from typing import List
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

    def loadDataset(self, dataset) -> None:
        self.dataset = dataset
    
    def transformShape(self, shape) -> torch.Size:
        return shape
    
    @abstractmethod
    def __call__(self, input : torch.Tensor) -> torch.Tensor:
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

    def __call__(self, input : torch.Tensor) -> torch.Tensor:
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

    def __call__(self, input : torch.Tensor) -> torch.Tensor:
        if "Min" not in self.dataset.cache_attribute:
            self.dataset.cache_attribute["Min"] = torch.min(input)
        if "Max" not in self.dataset.cache_attribute:
            self.dataset.cache_attribute["Max"] = torch.max(input)

        if self.lazy:
            return input
        else:
            input_min = self.dataset.cache_attribute["Min"]
            input_max = self.dataset.cache_attribute["Max"]
            norm = (input_max-input_min) + self.min_value
            assert norm != 0
            return (self.max_value-self.min_value)*(input - input_min) / norm

class Standardize(Transform):

    @config("Standardize")
    def __init__(self, lazy : bool = False) -> None:
        super().__init__()
        self.lazy = lazy

    def __call__(self, input : torch.Tensor) -> torch.Tensor:
        if "Mean" not in self.dataset.cache_attribute:
            self.dataset.cache_attribute["Mean"] = torch.mean(input.type(torch.float32))
        if "Std" not in self.dataset.cache_attribute:
            self.dataset.cache_attribute["Std"] = torch.std(input.type(torch.float32))

        if self.lazy:
            return input
        else:
            mean = self.dataset.cache_attribute["Mean"]
            std = self.dataset.cache_attribute["Std"]
            assert std > 0
            return (input - mean) / std

class DeNormalize(Transform):

    @config("DeNormalize")
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, input : torch.Tensor) -> torch.Tensor:
        input_min = None
        input_max = None
        
        min_value = torch.min(input)
        max_value = torch.max(input)

        if "Min" in self.dataset.cache_attribute:
            input_min = self.dataset.cache_attribute["Min"]
        if "Max" in self.dataset.cache_attribute:
            input_max = self.dataset.cache_attribute["Max"]

        return (input - min_value)*(input_max-input_min)/(max_value-min_value)+input_min if input_min is not None and input_max is not None else input
         
class DeStandardize(Transform):

    @config("DeStandardize")
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, input : torch.Tensor) -> torch.Tensor:
        mean = None
        std = None
        if "Mean" in self.dataset.cache_attribute:
            mean = self.dataset.cache_attribute["Mean"]
        if "Std" in self.dataset.cache_attribute:
            std = self.dataset.cache_attribute["Std"]
        return input * std + mean if mean is not None and std is not None else input

class TensorCast(Transform):

    @config("TensorCast")
    def __init__(self, dtype : str = "default:float32,int64,int16") -> None:
        super().__init__()
        self.dtype : torch.dtype = getattr(torch, dtype)

    def __call__(self, input : torch.Tensor) -> torch.Tensor:
        return input.type(self.dtype)

class ChannelPadding(Transform):

    @config("ChannelPadding")
    def __init__(self, pad : int, mode : str = "default:constant,reflect,replicate,circular") -> None:
        super().__init__()
        self.pad = pad
        self.mode = mode

    def __call__(self, input : torch.Tensor) -> torch.Tensor:
        return F.pad(input, (0, self.pad), self.mode)

class Resample(Transform):

    def __init__(self, reference_Size, save) -> None:
        super().__init__(save)
        self.reference_Size = np.asarray(reference_Size)

    def __call__(self, input : torch.Tensor) -> torch.Tensor:
        reference_Size = np.flip(self.reference_Size)
        image = self.dataset.to_image(input.numpy()) 
        resize_factor = np.asarray(image.GetSize())/reference_Size
        reference_Spacing = image.GetSpacing() * resize_factor
        dimension = image.GetDimension()
        reference_origin = image.GetOrigin()

        reference_image = sitk.Image([int(i) for i in reference_Size], image.GetPixelIDValue())
        reference_image.SetOrigin(image.GetOrigin())
        reference_image.SetSpacing(reference_Spacing)
        reference_image.SetDirection(image.GetDirection())

        reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))
    
        transform = sitk.AffineTransform(dimension)
        
        transform.SetMatrix(image.GetDirection())
        transform.SetTranslation(np.array(image.GetOrigin()) - reference_origin)

        centering_transform = sitk.TranslationTransform(dimension)
        img_center = np.array(image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize())/2.0))
        centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    
        result = sitk.Resample(image1=image, referenceImage = reference_image, transform=sitk.CompositeTransform([transform, centering_transform]), interpolator = sitk.sitkNearestNeighbor if input.dtype == torch.uint8 else sitk.sitkBSpline, defaultPixelValue = 0.0)
        self.dataset.setMetaData(result.GetOrigin(), result.GetSpacing(), np.asarray((1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0)))
        return torch.from_numpy(sitk.GetArrayFromImage(result))

    def transformShape(self, shape) -> np.ndarray:
        return self.reference_Size

class ResampleIsotropic(Resample):

    @config("ResampleIsotropic") 
    def __init__(self, spacing : float = 1, save : str = "name.h5") -> None:
        super().__init__(None, save)
        self.spacing = spacing
        
    def loadDataset(self, dataset) -> None:
        super().loadDataset(dataset)
        shape =  np.flip(dataset.getShape())
        resize_factor = ([self.spacing]*len(shape))/ np.flip(dataset.getSpacing())
        
        self.reference_Size =  np.asarray([int(x) for x in (shape * 1/resize_factor)])

class ResampleResize(Resample):

    @config("ResampleResize")
    def __init__(self, size : List[int] = [100,512,512], save : str = "name.h5") -> None:
        super().__init__(size, save)

class ResampleResizeForDataset(Resample):

    @config("ResampleResizeForDataset") 
    def __init__(self) -> None:
        super().__init__(None, None)

    def loadDataset(self, dataset):
        super().loadDataset(dataset)
        self.reference_Size =  np.roll(dataset._dataset.shape, -1)
        
class NConnexeLabel(Transform):

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

class GausianImportance(Transform):

    @config("GausianImportance") 
    def __init__(self, patch_size = [128,128,128], sigma : float = 0.2, save=None) -> None:
        super().__init__(save)
        gaussianSource = sitk.GaussianImageSource()
        gaussianSource.SetSize(tuple(patch_size))
        gaussianSource.SetMean([x // 2 for x in patch_size])
        gaussianSource.SetSigma([sigma * x for x in patch_size])
        gaussianSource.SetScale(1)
        gaussianSource.NormalizedOn()
        self.data = torch.from_numpy(sitk.GetArrayFromImage(gaussianSource.Execute())).unsqueeze(0)

    def __call__(self, input : torch.Tensor) -> torch.Tensor:
        return self.data.repeat([input.shape[0]]+[1]*(len(input.shape)-1))*input

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
        return result
