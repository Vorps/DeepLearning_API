from typing import Tuple
import pynvml
import psutil
import h5py
import SimpleITK as sitk
import numpy as np
import os
import torch
import datetime

DATE = lambda : datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

def dataset_to_data(dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.zeros(dataset.shape, dataset.dtype)
    dataset.read_direct(data)
    return data, dataset.attrs['Origin'], dataset.attrs['Spacing'], dataset.attrs['Direction']

def dataset_to_image(dataset) -> sitk.Image:
    data, origin, spacing, direction = dataset_to_data(dataset)
    image = sitk.GetImageFromArray(data)
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    return image

def data_to_dataset(h5 : h5py.File, name, data : np.ndarray, origin, spacing, direction) -> None:
    if name in h5:
        del h5[name]
    dataset = h5.create_dataset(name, data=data, dtype=data.dtype, chunks=None)
    dataset.attrs['Origin'] = origin
    dataset.attrs['Spacing'] = spacing
    dataset.attrs['Direction'] = direction

def image_to_dataset(h5, name, image) -> None:
    data_to_dataset(h5, name, sitk.GetArrayFromImage(image), origin=image.GetOrigin(), spacing=image.GetSpacing(), direction=image.GetDirection())


class DatasetUtils():

    def __init__(self, filename, read = True) -> None:
        self.filename = filename
        self.data = {}
        self.read = read

    def __enter__(self) -> None:
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
    
    def __exit__(self, type, value, traceback) -> None:
        self.h5.close()

    def writeImage(self, group, name, image) -> None:
        if group not in self.h5:
            self.h5.create_group(group)
        image_to_dataset(self.h5[group], name, image)

    def readImages(self, path_dest) -> None:
        def write(name, obj):
            if isinstance(obj, h5py._hl.dataset.Dataset):
                if len(name.split("/")) > 1 and not os.path.exists(path_dest+"/".join(name.split("/")[:-1])):
                    os.makedirs(path_dest+"/".join(name.split("/")[:-1]))
                if not os.path.exists(path_dest+name):
                    print("Write image : {}/{}".format(path_dest, name))
                    sitk.WriteImage(dataset_to_image(obj), path_dest+name)
    
        if not os.path.exists(path_dest):
            os.makedirs(path_dest)
                
        self.h5.visititems(write)

    def directory_to_dataset(self, src_path):
        for root, dirs, files in os.walk(src_path):
            path = root.replace(src_path, "")
            for i, file in enumerate(files):
                if file.endswith(".mha"):
                    if path not in self.h5:
                       self.h5.create_group(path)
                    image_to_dataset(self.h5[path], file, sitk.ReadImage("{}/{}".format(root, file)))
                print("Compute in progress : {:.2f} %".format((i+1)/len(files)*100))
        
        
def _getModule(classpath, type):
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

def getMemory() -> str:
    return psutil.virtual_memory()[3]/2**30

def memoryForecast(memory_init, i, size) -> str:
    current_memory = getMemory()
    forecast = memory_init + ((current_memory-memory_init)*size/i) if i > 0 else 0
    return "Memory forecast ({:.2f}G ({:.2f} %))".format(forecast, forecast/(psutil.virtual_memory()[0]/2**30)*100)

def gpuInfo(device) -> str:
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
    return  "Memory GPU ({:.2f}G ({:.2f} %)) | {} | Power {}W | Temperature {}°C".format(memory.used/(10**9), memory.used/memory.total*100, memoryInfo(), pynvml.nvmlDeviceGetPowerUsage(handle)//1000, pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))

def getAvailableDevice() -> int:
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

def getDevice(device : int) -> torch.device:
    if torch.cuda.is_available():
        if device is None:
            availableDevice = getAvailableDevice()
            if len(availableDevice):
               device = getAvailableDevice()[0]
            else:
                return torch.device("cpu")
        if device in getAvailableDevice():
            return torch.device("cuda:{}".format(device))
        else:
            raise Exception("GPU : {} is not available !".format(device))
    else:
        return torch.device("cpu")

def logImageNormalize(input : torch.Tensor):
    b = -np.min(input)
    a = 1/(np.max(input)+b)
    return a*(input+b)
