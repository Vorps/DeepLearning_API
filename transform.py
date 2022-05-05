import importlib
from typing import List
import torch

from . import config
import numpy as np
import SimpleITK as sitk
from DeepLearning_API import _getModule


class Transform:
    
    def __init__(self, save = None) -> None:
        self.save = save

    def loadDataset(self, dataset) -> None:
        self.dataset = dataset
        return self
    
    def transformShape(self, shape) -> np.ndarray:
        return shape

class TransformLoader:

    @config(None)
    def __init__(self) -> None:
        pass
    
    def getTransform(self, classpath : str, args : str) -> Transform:
        module, name = _getModule(classpath, "transform")
        return getattr(importlib.import_module(module), name)(config = None, args = args)

class Normalize(Transform):

    @config("Normalize")
    def __init__(self, min_value : float = 0, max_value : float = 100) -> None:
        super().__init__()
        assert max_value > min_value
        self.min_value = min_value
        self.max_value = max_value
        self.value_range = max_value - min_value

    def __call__(self, input : torch.Tensor) -> torch.Tensor:
        return (input - self.min_value) / self.value_range

class Standardize(Transform):

    @config("Standardize")
    def __init__(self, lazy : bool = False) -> None:
        super().__init__()
        self.lazy = lazy

    def __call__(self, input : torch.Tensor) -> torch.Tensor:
        mean = 0
        std = 0
        if self.lazy:
            self.dataset.cache_attribute["Mean"] = torch.mean(input.type(torch.float32))
            self.dataset.cache_attribute["Std"] = torch.std(input.type(torch.float32))
            return input
        else:
            mean = self.dataset.cache_attribute["Mean"] if "Mean" in self.dataset.cache_attribute else torch.mean(input.type(torch.float32))
            std = self.dataset.cache_attribute["Std"] if "Std" in self.dataset.cache_attribute else torch.std(input.type(torch.float32))
            return (input - mean) / std

class Unsqueeze(Transform):

    @config("Unsqueeze")
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, input : torch.Tensor) -> torch.Tensor:
        return torch.unsqueeze(input, dim=0)

class TensorCast(Transform):

    @config("TensorCast")
    def __init__(self, dtype : str = "default:float32,int64,int16") -> None:
        super().__init__()
        self.dtype = getattr(torch, dtype)

    def __call__(self, input : torch.Tensor) -> torch.Tensor:
        return input.type(self.dtype)


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
        return self

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
        return self
        
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
            true_label = max(stats, key=stats.get)

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
    def __init__(self, patch_size = [128,128,128], sigma : int = 0.2, save=None) -> None:
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
