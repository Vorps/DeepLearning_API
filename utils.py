import itertools
import pynvml
import psutil
import h5py
import SimpleITK as sitk
import numpy as np
import os
import torch
import datetime
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Any, Union, Optional, Iterator
from functools import partial
import copy

DATE = lambda : datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

class Attribute(dict[str, Any]):

    def __init__(self, attributes : dict[str, Any] = {}) -> None:
        super().__init__()
        for k, v in attributes.items():
            super().__setitem__(copy.deepcopy(k), copy.deepcopy(v))
    
    def __getitem__(self, key: str) -> Any:
        i = len([k for k in super().keys() if k.startswith(key)])
        if i > 0 and "{}_{}".format(key, i-1) in super().keys():   
            return super().__getitem__("{}_{}".format(key, i-1))
        else:
            raise NameError("{} not in cache_attribute".format(key))

    def __setitem__(self, key: str, value: Any) -> None:
        if "_" not in key:
            i = len([k for k in super().keys() if k.startswith(key)])
            super().__setitem__("{}_{}".format(key, i), value)
        else:
            super().__setitem__(key, value)
            
    def pop(self, key: str) -> Any:
        i = len([k for k in super().keys() if k.startswith(key)])
        if i > 0 and "{}_{}".format(key, i-1) in super().keys():   
            return super().pop("{}_{}".format(key, i-1))
        else:
            raise NameError("{} not in cache_attribute".format(key))

    def __contains__(self, key: str) -> bool:
        return len([k for k in super().keys() if k.startswith(key)]) > 0

def data_to_image(data : np.ndarray, attributes: Attribute) -> sitk.Image:
    if "Origin" not in attributes or "Spacing" not in attributes or "Direction" not in attributes:
        raise NameError("Data is not an image")
    if data.shape[0] == 1:
        image = sitk.GetImageFromArray(data[0])
    else:
        data = data.transpose(tuple([i+1 for i in range(len(data.shape)-1)]+[0]))
        image = sitk.GetImageFromArray(data, isVector=True)
    image.SetOrigin(attributes["Origin"].tolist())
    image.SetSpacing(attributes["Spacing"].tolist())
    image.SetDirection(attributes["Direction"].tolist())
    return image

class DatasetUtils():

    class AbstractFile(ABC):

        def __init__(self) -> None:
            pass
        
        def __enter__(self):
            pass

        def __exit__(self, type, value, traceback):
            pass

        @abstractmethod
        def file_to_data(self):
            pass

        @abstractmethod
        def data_to_file(self):
            pass

        @abstractmethod
        def getNames(self, group: Union[str, None] = None) -> list[str]:
            pass
        
        @abstractmethod
        def getInfos(self, group: Union[str, None], name: str) -> tuple[list[int], Attribute]:
            pass

    class H5File(AbstractFile):

        def __init__(self, filename: str, read: bool) -> None:
            self.h5: Union[h5py.File, None] = None
            self.filename = filename
            self.read = read

        def __enter__(self):
            args = {}
            if self.filename.endswith("/"):
                self.filename = self.filename[:-1]
            if self.read:
                self.h5 = h5py.File(self.filename, 'r', **args)
            else:
                if not os.path.exists(self.filename):
                    if len(self.filename.split("/")) > 1 and not os.path.exists("/".join(self.filename.split("/")[:-1])):
                        os.makedirs("/".join(self.filename.split("/")[:-1]))
                    self.h5 = h5py.File(self.filename, 'w', **args)
                else: 
                    self.h5 = h5py.File(self.filename, 'r+', **args)
                self.h5.attrs["Date"] = DATE()
            self.h5.__enter__()
            return self.h5
    
        def __exit__(self, type, value, traceback):
            if self.h5 is not None:
                self.h5.close()
        
        def file_to_data(self, name: str) -> tuple[np.ndarray, Attribute]:
            dataset = self.h5[name]
            data = np.zeros(dataset.shape, dataset.dtype)
            dataset.read_direct(data)
            attrs = Attribute()
            attrs.update(dataset.attrs)
            return data, attrs

        def data_to_file(self, name : str, data : Union[sitk.Image, sitk.Transform, np.ndarray], attributes : Union[Attribute, None] = None) -> None:
            if attributes is None:
                attributes = Attribute()
            if isinstance(data, sitk.Image):
                image = data
                attributes["Origin"] = torch.tensor(image.GetOrigin())
                attributes["Spacing"] = torch.tensor(image.GetSpacing())
                attributes["Direction"] = torch.tensor(image.GetDirection())
                data = sitk.GetArrayFromImage(image)

                if image.GetNumberOfComponentsPerPixel() == 1:
                    data = np.expand_dims(data, 0)
                else:
                    data = np.transpose(data, (len(data.shape)-1, *[i for i in range(len(data.shape)-1)]))
            elif isinstance(data, sitk.Transform):
                if isinstance(data, sitk.Euler3DTransform):
                    transform_type = "Euler3DTransform_double_3_3"
                if isinstance(data, sitk.AffineTransform):
                    transform_type = "AffineTransform_double_3_3"
                if isinstance(data, sitk.BSplineTransform):
                    transform_type = "BSplineTransform_double_3_3"
                attributes["Transform"] = transform_type
                attributes["FixedParameters"] = data.GetFixedParameters()
                data = np.asarray(data.GetParameters())

            h5_group = self.h5
            if len(name.split("/")) > 1:
                group = "/".join(name.split("/")[:-1])
                if group not in self.h5:
                    self.h5.create_group(group)
                h5_group = self.h5[group]

            name = name.split("/")[-1]
            if name in h5_group:
                del h5_group[name]

            dataset = h5_group.create_dataset(name, data=data, dtype=data.dtype, chunks=None)
            dataset.attrs.update({k : (v if isinstance(v.numpy() if isinstance(v, torch.Tensor) else v, np.ndarray) else str(v)) for k, v in attributes.items()})
        
        def getNames(self, group: Union[str, None] = None) -> list[str]:
            return [dataset.name for dataset in self.h5[group].values()] if group in self.h5[group] else []
        
        def getInfos(self, group: Union[str, None], name: str) -> tuple[list[int], Attribute]:
            h5Group = self.h5
            if group is not None:
                h5Group = self.h5[group]
            return h5Group[name].shape, Attribute({k : torch.tensor(v) if isinstance(v, np.ndarray) else v for k, v in h5Group[name].attrs.items()})

    class SitkFile(AbstractFile):

        def __init__(self, filename: str, read: bool) -> None:
            self.filename = filename
            if not os.path.exists(self.filename):
                os.makedirs(self.filename)
            self.read = read
            self.format = self.filename.split(".")[-1][:-1]

        def file_to_data(self, name: str) -> tuple[np.ndarray, Attribute]:
            image = sitk.ReadImage("{}{}".format(self.filename, name))
            if attributes is None:
                attributes = Attribute()
            attributes["Origin"] = torch.tensor(image.GetOrigin())
            attributes["Spacing"] = torch.tensor(image.GetSpacing())
            attributes["Direction"] = torch.tensor(image.GetDirection())

            data = sitk.GetArrayFromImage(image)
            if image.GetNumberOfComponentsPerPixel() == 1:
                data = np.expand_dims(data, 0)
            else:
                data = np.transpose(data, (len(data.shape)-1, *[i for i in range(len(data.shape)-1)]))
            return data, attributes
                
        def data_to_file(self, name : str, data : Union[sitk.Image, sitk.Transform, np.ndarray], attributes : Union[Attribute, None] = None) -> None:
            if isinstance(data, sitk.Image):
                sitk.WriteImage(data, "{}{}.{}".format(self.filename, name, self.format))
            elif isinstance(data, sitk.Transform):
                sitk.WriteTransform(data, "{}{}.itk.txt".format(self.filename, name))
            else:
                self.data_to_file(name, data_to_image(data, attributes), attributes)
        
        def getNames(self, group: Union[str, None] = None) -> list[str, ]:
            return sorted(os.listdir("{}{}/".format(self.filename, group))) if os.path.exists("{}{}/".format(self.filename, group)) else []

        def getInfos(self, group: Union[str, None], name: str) -> tuple[list[int], Attribute]:
            file_reader = sitk.ImageFileReader()
            file_reader.SetFileName("{}{}/{}".format(self.filename, group if group is not None else "", name))
            file_reader.ReadImageInformation()
            attributes = Attribute()
            attributes["Origin"] = torch.tensor(file_reader.GetOrigin())
            attributes["Spacing"] = torch.tensor(file_reader.GetSpacing())
            attributes["Direction"] = torch.tensor(file_reader.GetDirection())
            return list(file_reader.GetSize()), attributes

    class File(ABC):

        def __init__(self, filename: str, read: bool) -> None:
            self.filename = filename
            self.read = read
            self.file = None

        def __enter__(self):
            if ".h5" in self.filename:
                self.file = DatasetUtils.H5File(self.filename, self.read)
            else: 
                self.file = DatasetUtils.SitkFile(self.filename, self.read)
            self.file.__enter__()
            return self.file

        def __exit__(self, type, value, traceback):
            self.file.__exit__(type, value, traceback)

    def __init__(self, filename : str) -> None:
        if ".h5" not in filename and not filename.endswith("/"):
            filename = "{}/".format(filename)
        self.filename = filename
        self.is_directory = filename.endswith("/") 
        self.data = {}
    
    def write(self, group : str, name : str, data : Union[sitk.Image, sitk.Transform, np.ndarray], attributes : Union[Attribute, None] = None):
        if self.is_directory:
            if not os.path.exists(self.filename):
                os.makedirs(self.filename)
        if self.is_directory:
            s_group = group.split("/")
            if len(s_group) > 1:
                subDirectory = "/".join(s_group[:-1])
                name = "{}/{}".format(subDirectory, name)
                group = s_group[-1]
            with DatasetUtils.File("{}{}/".format(self.filename, name), False) as file:
                file.data_to_file(group, data, attributes)
        else:
            with DatasetUtils.File(self.filename, False) as file:
                file.data_to_file("{}/{}".format(group, name), data, attributes)
    
    def readData(self, group : str, name : str) -> tuple[np.ndarray, Attribute]:
        if self.is_directory:
            s_group = group.split("/")
            if len(s_group) > 1:
                subDirectory = "/".join(s_group[:-1])
                name = "{}/{}".format(subDirectory, name)
                group = s_group[-1]
            with DatasetUtils.File("{}{}".format(self.filename, name), False) as file:
                result = file.file_to_data(group)
        else:
            with DatasetUtils.File(self.filename, False) as file:
                result = file.file_to_data("{}/{}".format(group, name))
        return result
    
    def readTransform(self, group : str, name : str) -> sitk.Transform: 
        transformParameters, attribute = self.readData(group, name)
        transform_type = attribute["Transform"]
        if transform_type == "Euler3DTransform_double_3_3":
            transform = sitk.Euler3DTransform(3)
        if transform_type == "AffineTransform_double_3_3":
            transform = sitk.AffineTransform(3)
        if transform_type == "BSplineTransform_double_3_3":
            transform = sitk.BSplineTransform(3)
        transform.SetFixedParameters(eval(attribute["FixedParameters"]))
        transform.SetParameters(tuple(transformParameters))
        return transform
    
    def readImage(self, group : str, name : str):
         data, attribute = self.readData(group, name)
         return data_to_image(data, attribute)
            
    def getSize(self, group: str) -> int:
        if self.is_directory:
            name = self.filename
            s_group = group.split("/")
            if len(s_group) > 1:
                subDirectory = "/".join(s_group[:-1])
                name = "{}{}".format(name, subDirectory)
            result = len(os.listdir(name))  
        else:
            with DatasetUtils.File(self.filename, True) as file:
                result = len(file.getNames(group))
        return result

    def isExist(self, group: str, name: str) -> bool:
        if self.is_directory:
            if os.path.exists("{}/{}".format(self.filename, name)):
                s_group = group.split("/")
                if len(s_group) > 1:
                    subDirectory = "/".join(s_group[:-1])
                    name = "{}/{}".format(subDirectory, name)
                    group = s_group[-1]
                try:
                    with DatasetUtils.File("{}{}".format(self.filename, name), True) as file:
                        result = group in file.getNames()
                except:
                    os.remove("{}{}".format(self.filename, name))
                    result = False   
            else:
                result = False
        else:
            with DatasetUtils.File(self.filename, True) as file:
                result = name in file.getNames(group)
        return result
    
    def getNames(self, group: str, index: Optional[list[int]] = None) -> list[str]:
        if self.is_directory:
            name = self.filename
            s_group = group.split("/")
            if len(s_group) > 1:
                subDirectory = "/".join(s_group[:-1])
                name = "{}/{}".format(subDirectory, name)
            result = [name for i, name in enumerate(sorted(os.listdir(name))) if index is None or i in index]
        else:
            with DatasetUtils.File(self.filename, True) as file:
                result = [name for i, name in enumerate(file.getNames(group)) if index is None or i in index]
        return result
    
    def getInfos(self, group: str, name: str) -> tuple[list[int], Attribute]:
        if self.is_directory:
            s_group = group.split("/")
            if len(s_group) > 1:
                subDirectory = "/".join(s_group[:-1])
                name = "{}/{}".format(subDirectory, name)
                group = s_group[-1]
            with DatasetUtils.File("{}{}".format(self.filename, name), True) as file:
                result = file.getInfos(None, group)      
        else:
            with DatasetUtils.File(self.filename, True) as file:
                result = file.getInfos(group, name)
        return result

def _getModule(classpath : str, type : str) -> tuple[str, str]:
    if len(classpath.split("_")) > 1:
        if classpath.startswith("DeepLearning_API"):
            module = "DeepLearning_API."+".".join(classpath.split("_")[2:-1])
        else:
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

def gpuInfo(device : torch.device) -> str:
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


"""def gpuInfo(devices : torch.device) -> str:
    infos : list[tuple[int, float, float]] = []

    for device in devices:
        device_id = int(str(device).replace("cuda:", ""))
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        infos.append((device_id, float(memory.used)/(10**9), float(memory.used)/float(memory.total)*100))
    return "GPU({})".format("|".join([str(info[0]) for info in infos]))+" Memory GPU ({})".format("|".join(["{:.2f}G ({:.2f} %)".format(info[1], info[2]) for info in infos]))
    #float(memory.used)/(10**9), float(memory.used)/float(memory.total)*100, memoryInfo(), pynvml.nvmlDeviceGetPowerUsage(handle)//1000, pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
"""
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

def getDevice(devices : Union[list[int], None]) -> list[torch.device]:
    if torch.cuda.is_available():
        result_devices = []
        result_ids = []
        
        if devices is None:
            devices = "None"

        for device in devices:
            if device == "None":
                availableDevice = getAvailableDevice()
                if len(availableDevice):
                    device = "cpu"
                    for d in availableDevice:
                        if d not in result_ids:
                            device = d
                            break
                else:
                    result_devices.append(torch.device("cpu"))

            if device == "cpu":
                result_devices.append(torch.device("cpu"))
                result_ids.append(None)
            if device in getAvailableDevice():
                result_devices.append(torch.device("cuda:{}".format(device)))
                result_ids.append(device)
            else:
                raise Exception("GPU : {} is not available !".format(device))
        return result_devices
    else:
        return [torch.device("cpu")]

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
    METRIC = "METRIC"
    
    def __str__(self) -> str:
        return self.value

def get_patch_slices_from_nb_patch_per_dim(patch_size_tmp: list[int], nb_patch_per_dim : list[tuple[int, bool]], overlap: Union[list[int], None]) -> list[tuple[slice]]:
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

def get_patch_slices_from_shape(patch_size: list[int], shape : list[int], overlap: Union[list[int], None]) -> tuple[list[tuple[slice]], list[tuple[int, bool]]]:
    if len(shape) != len(patch_size):
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
        assert overlap_tmp[dim] < patch_size[dim],  "Overlap must be less than patch size"
            

    for dim in range(len(shape)):
        slices.append([])
        index = 0
        while True:
            start = (patch_size[dim]-overlap_tmp[dim])*index

            end = start + patch_size[dim]
            if end >= shape[dim]:
                end = shape[dim]
                slices[dim].append(slice(start, end))
                break
            slices[dim].append(slice(start, end))
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
    return sitk.Resample(image, image_reference, result_transform, sitk.sitkNearestNeighbor if mask else sitk.sitkBSpline, defaultPixelValue =  0 if mask else -1024)

def formatMaskLabel(mask: sitk.Image, labels: list[tuple[int, int]]) -> sitk.Image:
    data = sitk.GetArrayFromImage(mask)
    result_data = np.zeros_like(data, np.uint8)

    for label_old, label_new in labels:
        result_data[np.where(data == label_old)] = label_new

    result = sitk.GetImageFromArray(result_data)
    result.CopyInformation(mask)
    return result

def parameterMap_to_ITK_Euler3DTransform(path_src: str, path_dest: str) -> None:
    transform_rigid = sitk.ReadParameterFile("{}.0.txt".format(path_src))
    with open("{}.itk.txt".format(path_dest), "w") as f:
        f.write("#Insight Transform File V1.0\n")
        f.write("#Transform 0\n")
        f.write("Transform: Euler3DTransform_double_3_3\n")

        TransformParameters = np.array([float(i) for i in transform_rigid["TransformParameters"]])
        CenterOfRotationPoint = np.array([float(i) for i in transform_rigid["CenterOfRotationPoint"]])
            
        f.write("Parameters: "+" ".join([str(i) for i in TransformParameters])+""+"\n")
        f.write("FixedParameters: "+" ".join([str(i)+" " for i in CenterOfRotationPoint])+"\n")

def parameterMap_to_ITK_AffineTransform(path_src: str, path_dest: str) -> None:
    transform_affine = sitk.ReadParameterFile("{}.0.txt".format(path_src))
    with open("{}.itk.txt".format(path_dest), "w") as f:
        f.write("#Insight Transform File V1.0\n")
        f.write("#Transform 0\n")
        f.write("Transform: AffineTransform_double_3_3\n")

        TransformParameters = np.array([float(i) for i in transform_affine["TransformParameters"]])
        CenterOfRotationPoint = np.array([float(i) for i in transform_affine["CenterOfRotationPoint"]])
            
        f.write("Parameters: "+" ".join([str(i) for i in TransformParameters])+""+"\n")
        f.write("FixedParameters: "+" ".join([str(i)+" " for i in CenterOfRotationPoint])+"\n")


def parameterMap_to_ITK_BSplineTransform(path_src: str, path_dest: str) -> None:
    transform_rigid = sitk.ReadParameterFile("{}.0.txt".format(path_src))
    with open("{}.itk.txt".format(path_dest), "w") as f:
        f.write("#Insight Transform File V1.0\n")
        f.write("#Transform 0\n")
        f.write("Transform: BSplineTransform_double_3_3\n")
        TransformParameters = np.array([float(i) for i in transform_rigid["TransformParameters"]])

        GridSize = np.array([int(i) for i in transform_rigid["GridSize"]])
        GridOrigin = np.array([float(i) for i in transform_rigid["GridOrigin"]])
        GridSpacing = np.array([float(i) for i in transform_rigid["GridSpacing"]])
        GridDirection = np.array(np.array([float(i) for i in transform_rigid["GridDirection"]])).reshape((3,3)).T.flatten() 

        f.write("Parameters: "+" ".join([str(i) for i in TransformParameters])+"\n")
        f.write("FixedParameters: "+" ".join([str(i) for i in GridSize])+" "+" ".join([str(i) for i in GridOrigin])+" "+" ".join([str(i) for i in GridSpacing])+" "+" ".join([str(i) for i in GridDirection])+"\n")
