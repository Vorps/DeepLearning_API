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
from typing import Callable, Any, Union, Optional
from functools import partial

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
        image.SetOrigin(attributes["Origin"].tolist())
    if "Spacing" in attributes:
        image.SetSpacing(attributes["Spacing"].tolist())
    if "Direction" in attributes:
        image.SetDirection(attributes["Direction"].tolist())
    return image

def dataset_to_image(dataset : h5py.Dataset) -> sitk.Image:
    data, attributes = dataset_to_data(dataset)
    return data_to_image(data, attributes)

def data_to_dataset(h5 : Union[h5py.File, h5py.Group], name : str, data : np.ndarray, attributes : Union[Attribute, None] = None) -> None:
    if name in h5:
        del h5[name]
    if attributes is None:
        attributes = Attribute()
    dataset = h5.create_dataset(name, data=data, dtype=data.dtype, chunks=None)
    dataset.attrs.update({k : (v if isinstance(v.numpy() if isinstance(v, torch.Tensor) else v, np.ndarray) else str(v)) for k, v in attributes.items()})

def image_to_dataset(h5 : Union[h5py.File, h5py.Group], name : str, image : sitk.Image, attributes : Union[Attribute, None] = None) -> None:
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
    data_to_dataset(h5, name, data, attributes)

class DatasetUtils():

    class H5File():

        def __init__(self, filename: str, read: bool, parallel : bool = False) -> None:
            self.h5: Union[h5py.File, None] = None
            self.filename = filename
            self.read = read
            self.parallel = parallel

        def __enter__(self):
            args = {}
            if self.parallel:
                try:
                    from mpi4py import MPI
                    args.update(dict(driver='mpio', comm=MPI.COMM_WORLD))
                except:
                    print("Module importation error mpi4py")

            if self.read:
                self.h5 = h5py.File(self.filename, 'r', **args)
            else:
                if not os.path.exists(self.filename) or self.parallel:
                    if len(self.filename.split("/")) > 1 and not os.path.exists("/".join(self.filename.split("/")[:-1])):
                        os.makedirs("/".join(self.filename.split("/")[:-1]))
                    self.h5 = h5py.File(self.filename, 'w', **args)
                else: 
                    self.h5 = h5py.File(self.filename, 'r+', **args)
                self.h5.attrs["Date"] = DATE()
            self.h5.__enter__()
            return self
    
        def __exit__(self, type, value, traceback):
            if self.h5 is not None:
                self.h5.close()

    def __init__(self, filename : str, parallel : bool = False, is_directory: bool = False) -> None:
        self.filename = filename
        if os.path.exists(filename) and os.path.isdir(filename):
            self.is_directory = True
        else:
            self.is_directory = is_directory
        self.data = {}
        self.parallel = parallel
        if self.is_directory:
            if not os.path.exists(self.filename):
                os.makedirs(self.filename)
    
    def _write(self, group : str, name : str, func : Callable[[Union[h5py.File, h5py.Group], sitk.Image, Union[Attribute, None]], None], attributes : Union[Attribute, None] = None):
        if self.is_directory:
            s_group = group.split("/")
            if len(s_group) > 1:
                subDirectory = "/".join(s_group[:-1])
                name = "{}/{}".format(subDirectory, name)
                group = s_group[-1]
            with DatasetUtils.H5File("{}/{}".format(self.filename, name), False, self.parallel) as h5:
                func(h5 = h5.h5, name = group, attributes = attributes)
        else:
            with DatasetUtils.H5File(self.filename, False, self.parallel) as h5:
                if group not in h5.h5:
                    h5.h5.create_group(group)
                func(h5 = h5.h5[group], name = name, attributes = attributes)

    def writeImage(self, group : str, name : str, image : sitk.Image, attributes : Union[Attribute, None] = None) -> None:
        self._write(group, name, partial(image_to_dataset, image = image), attributes)

    def writeData(self, group : str, name : str, data : np.ndarray, attributes : Union[Attribute, None] = None) -> None:
        self._write(group, name, partial(data_to_dataset, data = data), attributes)
    
    def writeTransform(self, group : str, name : str, transform: sitk.Transform, attributes : Union[Attribute, None] = None) -> None:
        if isinstance(transform, sitk.Euler3DTransform):
            transform_type = "Euler3DTransform_double_3_3"
        if isinstance(transform, sitk.AffineTransform):
            transform_type = "AffineTransform_double_3_3"
        if isinstance(transform, sitk.BSplineTransform):
            transform_type = "BSplineTransform_double_3_3"
        if attributes is None:
            attribute = Attribute()
        attribute["Transform"] = transform_type
        attribute["FixedParameters"] = transform.GetFixedParameters()
        self._write(group, name, partial(data_to_dataset, data = np.asarray(transform.GetParameters())), attribute)

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
    
    def readData(self, group : str, name : str) -> tuple[np.ndarray, Attribute]:
        if self.is_directory:
            s_group = group.split("/")
            if len(s_group) > 1:
                subDirectory = "/".join(s_group[:-1])
                name = "{}/{}".format(subDirectory, name)
                group = s_group[-1]
            with DatasetUtils.H5File("{}/{}".format(self.filename, name), False, self.parallel) as h5:
                return dataset_to_data(h5.h5[group])
        else:
            with DatasetUtils.H5File(self.filename, False, self.parallel) as h5:
                return dataset_to_data(h5.h5[group][name])
    
    def readImages(self, path_dest : str, function: Callable[[np.ndarray, Attribute, str], np.ndarray] = lambda x, y, z: x) -> None:
        if not os.path.exists(path_dest):
            os.makedirs(path_dest)

        if self.is_directory:
            def write(name, group, obj):
                if isinstance(obj, h5py.Dataset):
                    if not os.path.exists("{}{}".format(path_dest, group)):
                        os.makedirs("{}{}".format(path_dest, group))
                    if not os.path.exists("{}{}/{}".format(path_dest, group, name)):
                        data, attrs = dataset_to_data(obj)
                        data = function(data, attrs, name)
                        im = data_to_image(data, attrs)
                        sitk.WriteImage(im, "{}{}/{}".format(path_dest, group, name), True)
                        print("Write image : {}{}/{}".format(path_dest, group, name))
            names = sorted(os.listdir(self.filename))
            for name in names:
                with DatasetUtils.H5File("{}/{}".format(self.filename, name), True, self.parallel) as h5:
                    h5.h5.visititems(partial(write, name))
        else:
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
            with DatasetUtils.H5File(self.filename, True, self.parallel) as h5:            
                h5.h5.visititems(partial(write, None))

    def directory_to_dataset(self, src_path : str):
        for root, dirs, files in os.walk(src_path):
            path = root.replace(src_path, "")
            for i, file in enumerate(files):
                if file.endswith(".mha"):
                    self.writeImage(path, file, sitk.ReadImage("{}/{}".format(root, file)))
                print("Compute in progress : {} {:.2f} %".format(path, (i+1)/len(files)*100))
    
    def getSize(self, group) -> int:
        if self.is_directory:
            name = self.filename
            s_group = group.split("/")
            if len(s_group) > 1:
                subDirectory = "/".join(s_group[:-1])
                name = "{}/{}".format(name, subDirectory)
            return len(os.listdir(name))  
        else:
            with DatasetUtils.H5File(self.filename, True, self.parallel) as h5:
                return len(h5.h5[group].keys())

    def isExist(self, group, name: str) -> bool:
        if self.is_directory:
            if os.path.exists("{}/{}".format(self.filename, name)):
                s_group = group.split("/")
                if len(s_group) > 1:
                    subDirectory = "/".join(s_group[:-1])
                    name = "{}/{}".format(subDirectory, name)
                    group = s_group[-1]
                try:
                    with DatasetUtils.H5File("{}/{}".format(self.filename, name), True, self.parallel) as h5:            
                        return group in h5.h5
                except:
                    os.remove("{}/{}".format(self.filename, name))
                    return False   
            else:
                return False
        else:
            with DatasetUtils.H5File(self.filename, True, self.parallel) as h5:
                return group in h5.h5 and name in h5.h5[group]
    
    def getNames(self, group, index: Optional[list[int]] = None) -> list[str]:
        if self.is_directory:
            name = self.filename
            s_group = group.split("/")
            if len(s_group) > 1:
                subDirectory = "/".join(s_group[:-1])
                name = "{}/{}".format(subDirectory, name)
            return [name for i, name in enumerate(sorted(os.listdir(name))) if index is None or i in index]
        else:
            with DatasetUtils.H5File(self.filename, True, self.parallel) as h5:
                return [dataset.name for i, dataset in enumerate(h5.h5[group].values()) if index is None or i in index]
    
    def getInfos(self, group, name) -> tuple[list[int], Attribute]:
        if self.is_directory:
            s_group = group.split("/")
            if len(s_group) > 1:
                subDirectory = "/".join(s_group[:-1])
                name = "{}/{}".format(subDirectory, name)
                group = s_group[-1]
            if os.path.exists("{}/{}".format(self.filename, name)):
                with DatasetUtils.H5File("{}/{}".format(self.filename, name), True, self.parallel) as h5:            
                    if group in h5.h5:
                        return h5.h5[group].shape, Attribute({k : torch.tensor(v) if isinstance(v, np.ndarray) else v for k, v in h5.h5[group].attrs.items()})
            else:
                assert ValueError()
        else:
            with DatasetUtils.H5File(self.filename, True, self.parallel) as h5:
                if group in h5.h5 and name in h5.h5[group]:
                    return h5.h5[group][name].shape, Attribute({k : torch.tensor(v) if isinstance(v, np.ndarray) else v for k, v in h5.h5[group][name].attrs.items()})
                else:
                    assert ValueError()

def _getModule(classpath : str, type : str) -> tuple[str, str]:
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

def gpuInfo(devices : list[torch.device]) -> str:
    infos : list[tuple[int, float, float]] = []

    for device in devices:
        device_id = int(str(device).replace("cuda:", ""))
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        infos.append((device_id, float(memory.used)/(10**9), float(memory.used)/float(memory.total)*100))
    return "GPU({})".format("|".join([str(info[0]) for info in infos]))+" Memory GPU ({})".format("|".join(["{:.2f}G ({:.2f} %)".format(info[1], info[2]) for info in infos]))
    #float(memory.used)/(10**9), float(memory.used)/float(memory.total)*100, memoryInfo(), pynvml.nvmlDeviceGetPowerUsage(handle)//1000, pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))

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

def getDevice(devices : Union[list[int], None]) -> Union[list[int], list[torch.device]]:
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
        return result_ids, result_devices
    else:
        return [None], [torch.device("cpu")]

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
