import itertools
import pynvml
import psutil
import h5py
import SimpleITK as sitk
import numpy as np
import os
import torch
import datetime
from abc import ABC
from enum import Enum
from typing import Callable, Any

DATE = lambda : datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

class Attribute(dict[str, Any]):

    def __init__(self, attributes : dict[str, Any] = {}) -> None:
        super().__init__()
        for k, v in attributes.items():
            super().__setitem__(k, v)
    
    def __getitem__(self, key: str) -> Any:
        i = len([k for k in super().keys() if k.startswith(key)])
        if i > 0 and "{}_{}".format(key, i-1) in super().keys():   
            return super().__getitem__("{}_{}".format(key, i-1))
        else:
            raise NameError("{} not in cache_attribute".format(key))

    def __setitem__(self, key: str, value: Any) -> None:
        i = len([k for k in super().keys() if k.startswith(key)])
        super().__setitem__("{}_{}".format(key, i), value)

    def pop(self, key: str) -> Any:
        i = len([k for k in super().keys() if k.startswith(key)])
        if i > 0 and "{}_{}".format(key, i-1) in super().keys():   
            return super().pop("{}_{}".format(key, i-1))
        else:
            raise NameError("{} not in cache_attribute".format(key))


    def __contains__(self, key: str) -> bool:
        return len([k for k in super().keys() if k.startswith(key)]) > 0

def dataset_to_data(dataset : h5py.Dataset) -> tuple[np.ndarray, Attribute]:
    data = np.zeros(dataset.shape, dataset.dtype)
    dataset.read_direct(data)
    attrs = Attribute()
    attrs.update(dataset.attrs)
    return data, attrs

def data_to_image(data : np.ndarray, attributes: Attribute) -> sitk.Image:
    if data.shape[0] == 1:
        image = sitk.GetImageFromArray(data[0])
    else:
        data = data.transpose(tuple([i+1 for i in range(len(data.shape)-1)]+[0]))
        image = sitk.GetImageFromArray(data, isVector=True)
    if "Origin" in attributes:
        image.SetOrigin(attributes["Origin"])
    if "Spacing" in attributes:
        image.SetSpacing(attributes["Spacing"])
    if "Direction" in attributes:
        image.SetDirection(attributes["Direction"])
    return image

def dataset_to_image(dataset : h5py.Dataset) -> sitk.Image:
    data, attributes = dataset_to_data(dataset)
    return data_to_image(data, attributes)

def data_to_dataset(h5 : h5py.Group, name : str, data : np.ndarray, attributes : Attribute | None = None) -> None:
    if name in h5:
        del h5[name]
    if attributes is None:
        attributes = Attribute()
    dataset = h5.create_dataset(name, data=data, dtype=data.dtype, chunks=None)
    dataset.attrs.update({k : (v if isinstance(v.numpy() if isinstance(v, torch.Tensor) else v, np.ndarray) else str(v)) for k, v in attributes.items()})

def image_to_dataset(h5 : h5py.Group, name : str, image : sitk.Image, attributes : Attribute | None = None) -> None:
    if attributes is None:
        attributes = Attribute()
    attributes["Origin"] = image.GetOrigin()
    attributes["Spacing"] = image.GetSpacing()
    attributes["Direction"] = image.GetDirection()

    data = sitk.GetArrayFromImage(image)
    if image.GetNumberOfComponentsPerPixel() == 1:
        data = np.expand_dims(data, 0)
    else:
        data = np.transpose(data, (len(data.shape)-1, *[i for i in range(len(data.shape)-1)]))
    data_to_dataset(h5, name, data, attributes)

class DatasetUtils():

    def __init__(self, filename : str, read : bool = True) -> None:
        self.filename = filename
        self.data = {}
        self.read = read
        self.h5: h5py.File | None = None

    def __enter__(self):
        if self.read:
            self.h5 = h5py.File(self.filename, 'r')
        else:
            if not os.path.exists(self.filename):
                if len(self.filename.split("/")) > 1 and not os.path.exists("/".join(self.filename.split("/")[:-1])):
                    os.makedirs("/".join(self.filename.split("/")[:-1]))
                self.h5 = h5py.File(self.filename, 'w')
            else: 
                self.h5 = h5py.File(self.filename, 'r+')
            self.h5.attrs["Date"] = DATE()
        return self
    
    def __exit__(self, type, value, traceback):
        if self.h5 is not None:
            self.h5.close()

    def writeImage(self, group : str, name : str, image : sitk.Image, attributes : Attribute | None = None) -> None:
        assert self.h5
        if group not in self.h5:
            self.h5.create_group(group)
        h5_group = self.h5[group]
        if isinstance(h5_group, h5py.Group):
            image_to_dataset(h5_group, name, image, attributes)
    
    def writeData(self, group : str, name : str, data : np.ndarray, attributes : Attribute | None = None) -> None:
        assert self.h5
        if group not in self.h5:
            self.h5.create_group(group)
        h5_group = self.h5[group]
        if isinstance(h5_group, h5py.Group):
            data_to_dataset(h5_group, name, data, attributes)

    def readImages(self, path_dest : str, function: Callable[[np.ndarray, Attribute, str], np.ndarray] = lambda x, y, z: x) -> None:
        assert self.h5
        def write(name, obj):
            if isinstance(obj, h5py.Dataset):
                if len(name.split("/")) > 1 and not os.path.exists(path_dest+"/".join(name.split("/")[:-1])):
                    os.makedirs(path_dest+"/".join(name.split("/")[:-1]))
                if not os.path.exists(path_dest+name):
                    data, attrs = dataset_to_data(obj)
                    data = function(data, attrs, name)
                    im = data_to_image(data, attrs)
                    sitk.WriteImage(im, path_dest+name, True)
                    print("Write image : {}{}".format(path_dest, name))
    
        if not os.path.exists(path_dest):
            os.makedirs(path_dest)
            
        self.h5.visititems(write)

    def directory_to_dataset(self, src_path : str):
        assert self.h5
        for root, dirs, files in os.walk(src_path):
            path = root.replace(src_path, "")
            for i, file in enumerate(files):
                if file.endswith(".mha"):
                    if path not in self.h5:
                       self.h5.create_group(path)
                    h5_group = self.h5[path]
                    if isinstance(h5_group, h5py.Group):
                        image_to_dataset(h5_group, file, sitk.ReadImage("{}/{}".format(root, file)))
                print("Compute in progress : {:.2f} %".format((i+1)/len(files)*100))
        
        
def _getModule(classpath : str, type : str):
    if len(classpath.split("_")) > 1:
        module = ".".join(classpath.split("_")[:-1])
        name = classpath.split("_")[-1] 
    else:
        module = "DeepLearning_API."+type
        name = classpath
    return module, name

def cpuInfo() -> str:
    return "CPU ({:.2f} %)".format(psutil.cpu_percent(interval=0.5))

def memoryInfo() -> str:
    return "Memory ({:.2f}G ({:.2f} %))".format(psutil.virtual_memory()[3]/2**30, psutil.virtual_memory()[2])

def getMemory() -> float:
    return psutil.virtual_memory()[3]/2**30

def memoryForecast(memory_init : float, i : float, size : float) -> str:
    current_memory = getMemory()
    forecast = memory_init + ((current_memory-memory_init)*size/i) if i > 0 else 0
    return "Memory forecast ({:.2f}G ({:.2f} %))".format(forecast, forecast/(psutil.virtual_memory()[0]/2**30)*100)

def gpuInfo(device : int | torch.device) -> str:
    if isinstance(device, torch.device):
        if str(device).startswith("cuda:"):
            device = int(str(device).replace("cuda:", ""))
        else:
            return ""
    if device < pynvml.nvmlDeviceGetCount():
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    else:
        return ""
    return  "GPU({}) Memory GPU ({:.2f}G ({:.2f} %)) | {} | Power {}W | Temperature {}°C".format(device, float(memory.used)/(10**9), float(memory.used)/float(memory.total)*100, memoryInfo(), pynvml.nvmlDeviceGetPowerUsage(handle)//1000, pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))

def gpuMemory(device : int | torch.device) -> str:
    if isinstance(device, torch.device):
        if str(device).startswith("cuda:"):
            device = int(str(device).replace("cuda:", ""))
        else:
            return ""
    if device < pynvml.nvmlDeviceGetCount():
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    else:
        return ""
    return  "Memory GPU ({:.2f}G)".format(float(memory.used)/(10**9))

def getAvailableDevice() -> list[int]:
    pynvml.nvmlInit()
    available = []
    deviceCount = pynvml.nvmlDeviceGetCount()
    for id in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(id)
        use = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = int(use.memory)
        if memory == 0:
            available.append(id)
    pynvml.nvmlShutdown()
    return available

def getDevice(device : int | None) -> torch.device:
    if torch.cuda.is_available():
        if device is None:
            availableDevice = getAvailableDevice()
            if len(availableDevice):
               device = getAvailableDevice()[0]
            else:
                return torch.device("cpu")
        if device == "cpu":
            return torch.device("cpu")
        if device in getAvailableDevice():
            return torch.device("cuda:{}".format(device))
        else:
            raise Exception("GPU : {} is not available !".format(device))
    else:
        return torch.device("cpu")

def logImageFormat(input_torch : torch.Tensor):
    input = input_torch[0].detach().cpu().numpy()
    if input.shape[0] > 50:
        input = np.expand_dims(input, 0)
    nb_channel = input.shape[0]

    if len(input.shape)-1 == 3:
        input = input[:,input.shape[1]//2, ...]
    if nb_channel == 1:
        input = input[0]
    else:
        if nb_channel < 3:
            channel_split = [input[i] for i in range(input.shape[0])]
        else: 
            channel_split = np.split(input, 3, axis=0)
        input = np.zeros((3, *list(input.shape[1:])))
        for i, channels in enumerate(channel_split):
            input[i] = np.mean(channels, axis=0)
    b = -np.min(input)
    a = 1/(np.max(input)+b)
    return a*(input+b)

class NeedDevice(ABC):

    def __init__(self) -> None:
        super().__init__()
        self.device : torch.device
    
    def setDevice(self, device : torch.device):
        self.device = device

class State(Enum):
    TRAIN = "TRAIN"
    RESUME = "RESUME"
    TRANSFER_LEARNING = "TRANSFER_LEARNING"
    FINE_TUNNING = "FINE_TUNNING"
    PREDICTION = "PREDICTION"
    
    def __str__(self) -> str:
        return self.value

def get_patch_slices_from_nb_patch_per_dim(patch_size_tmp: list[int], nb_patch_per_dim : list[tuple[int, bool]], overlap: list[int] | None) -> list[tuple[slice]]:
    patch_slices = []
    slices : list[list[slice]] = []
    if overlap is None:
        overlap = [0 for _ in range(len(nb_patch_per_dim))]
    patch_size = []
    i = 0
    for nb in nb_patch_per_dim:
        if nb[1]:
            patch_size.append(1)
        else:
            patch_size.append(patch_size_tmp[i])
            i+=1

    for dim, nb in enumerate(nb_patch_per_dim):
        slices.append([])
        for index in range(nb[0]):
            start = (patch_size[dim]-overlap[dim])*index
            end = start + patch_size[dim]
            slices[dim].append(slice(start,end))
    for chunk in itertools.product(*slices):
        patch_slices.append(tuple(chunk))
    return patch_slices

def get_patch_slices_from_shape(patch_size: list[int], shape : list[int], overlap: list[int] | None) -> tuple[list[tuple[slice]], list[tuple[int, bool]]]:
    if len(shape) != len(patch_size) or not all(a >= b for a, b in zip(shape, patch_size)):
        return [tuple([slice(0, s) for s in shape])], [(1, True)]*len(shape)

    patch_slices = []
    nb_patch_per_dim = []
    slices : list[list[slice]] = []
    if overlap is None:
        size = [np.ceil(a/b) for a, b in zip(shape, patch_size)]
        overlap_tmp = np.zeros(len(size), dtype=np.int_)
        for i, s in enumerate(size):
            if s > 1:
                overlap_tmp[i] = np.mod(patch_size[i]-np.mod(shape[i], patch_size[i]), patch_size[i])//(size[i]-1)
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
        nb_patch_per_dim.append((index+1, patch_size[dim] == 1))

    for chunk in itertools.product(*slices):
        patch_slices.append(tuple(chunk))
    return patch_slices, nb_patch_per_dim


def resampleITK(path, image_reference : sitk.Image, image : sitk.Image, transforms_files : dict[str, bool], mask = True):
    transforms = []
    for transform_file, invert in transforms_files.items():
        transform = sitk.ReadTransform(path+transform_file+".itk.txt")
        type = None
        with open(path+transform_file+".itk.txt", "r") as f:
            for line in f:
                if line.startswith("Transform"):
                    type = line.split(": ")[1].strip("\n")
                    break
        if type == "Euler3DTransform_double_3_3":
            transform = sitk.Euler3DTransform(transform)
            if invert:
                transform = transform.GetInverse()
            transforms.append(transform)
        elif type == "AffineTransform_double_3_3":
            transform = sitk.AffineTransform(transform)
            if invert:
                transform = transform.GetInverse()
            transforms.append(transform)
        else:
            transform = sitk.BSplineTransform(transform)
            if invert:
                transformToDisplacementFieldFilter = sitk.TransformToDisplacementFieldFilter()
                transformToDisplacementFieldFilter.SetReferenceImage(image)
                displacementField = transformToDisplacementFieldFilter.Execute(transform)
                iterativeInverseDisplacementFieldImageFilter = sitk.IterativeInverseDisplacementFieldImageFilter()
                iterativeInverseDisplacementFieldImageFilter.SetNumberOfIterations(20)
                inverseDisplacementField = iterativeInverseDisplacementFieldImageFilter.Execute(displacementField)
                transform = sitk.DisplacementFieldTransform(inverseDisplacementField)
            transforms.append(transform)
    result_transform = sitk.CompositeTransform(transforms)
    return sitk.Resample(image, image_reference, result_transform, sitk.sitkNearestNeighbor if mask else sitk.sitkBSpline, *{"defaultPixelValue" : 0 if mask else -1024})

def formatMaskLabel(mask: sitk.Image, labels: list[tuple[int, int]]) -> sitk.Image:
    data = sitk.GetArrayFromImage(mask)
    result_data = np.zeros_like(data, np.uint8)

    for label_old, label_new in labels:
        result_data[np.where(data == label_old)] = label_new

    result = sitk.GetImageFromArray(result_data)
    result.CopyInformation(mask)
    return result
