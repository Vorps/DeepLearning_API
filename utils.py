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
from typing import Any, Union, Optional
import copy
import csv
from DeepLearning_API import CONFIG_FILE, STATISTICS_DIRECTORY
import torch.distributed as dist
import argparse
import subprocess
import random
from torch.utils.data import DataLoader
import multiprocessing
import torch.nn.functional as F
from lxml import etree
import sys
import matplotlib.pyplot as pyplot
import pandas as pd
import vtk

DATE = lambda : datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")


def description(model, modelEMA = None, showMemory: bool = True) -> str:
    values_desc = lambda weights, values: " ".join(["{}({:.2f}) : {:.6f}".format(name.split(":")[-1], weight, value) for (name, value), weight in zip(values.items(), weights.values())])

    model_desc = lambda model : "("+" ".join(["{}({:.6f}) : {}".format(name, network.optimizer.param_groups[0]['lr'], values_desc(network.measure.getLastWeights(), network.measure.getLastValues())) for name, network in model.module.getNetworks().items() if network.measure is not None])+")"
    result = "Loss {}".format(model_desc(model))
    if modelEMA is not None:
        result += "Loss EMA {}".format(model_desc(modelEMA))
    result += " "+gpuInfo()
    if showMemory:
        result +=" | {}".format(memoryInfo())
    return result

class Plot():

    def __init__(self, root: etree.ElementTree) -> None:
        self.root = root

    def _explore(root, result, label):
        if len(root) == 0:
            for attribute in root.attrib:
                result["attrib:"+label+":"+attribute] = root.attrib[attribute]
            if root.text is not None:
                result[label] = np.fromstring(root.text, sep = ",").astype('double')
        else:
            for node in root:
                Plot._explore(node, result, label+":"+node.tag)

    def getNodes(root, path = None, id = None):
        nodes = []
        if path != None:
            path = path.split(":")
            for node_name in path:
                node = root.find(node_name)
                if node != None:
                    root = node
                else:
                    break
        if id != None:
            for node in root.findall(".//"+id):
                nodes.append(node)
        else:
            nodes.append(root)                
        return nodes

    def read(root, path = None, id = None):
        result = dict()
        for node in Plot.getNodes(root, path, id):
            Plot._explore(node, result, etree.ElementTree(root).getpath(node))    
        return result

    def _extract(self, ids = [], patients = []):
        result = dict()
        if len(patients) == 0:
            if len(ids) == 0:
                result.update(Plot.read(self.root,None, None))
            else:
                for id in ids:
                    result.update(Plot.read(self.root, None, id))
        else:
            for path in patients:
                if len(ids) == 0:
                    result.update(Plot.read(self.root, path, None))
                else:
                    for id in ids:
                        result.update(Plot.read(self.root, path, id))
        return result

    def plot(self, ids = [], patients = [], labels = [], colors = None):
        results = self._extract(ids=ids, patients=patients)

        attrs = {k: v for k, v in results.items() if k.startswith("attrib:")}
        errors = {k: v for k, v in results.items() if not k.startswith("attrib:")}

        patients = set()
        max = 0
        for key, error in errors.items():
            patients.add(key.replace("/",":").split(":")[2])
            if max < int(error.shape[0]/3): 
                max = int(error.shape[0]/3)
        patients = sorted(patients)

        norms = {patient : np.array([]) for patient in patients}
        markups = {patient : np.array([]) for patient in patients}
        series = list()
        for key, error in errors.items():
            patient = key.replace("/",":").split(":")[2]
            markup = np.full((max,3), np.nan)
            markup[0:int(error.shape[0]/3), :] = error.reshape(int(error.shape[0]/3),3)

            markups[patient] = np.append(markups[patient], markup)
            norms[patient] = np.append(norms[patient], np.linalg.norm(markup, ord=2, axis=1))

        if len(labels) == 0:
            labels = list(set([k.split("/")[-1] for k in errors.keys()]))

        for label in labels:
            series = series+[label]*max

        df = pd.DataFrame(dict([(k,pd.Series(v)) for k, v in norms.items()]))
        df['Categories'] = pd.Series(series)
        
        bp = df.boxplot(by='Categories', color="black", figsize=(12,8), notch=True,layout=(1,len(patients)), fontsize=18, rot=0, patch_artist = True, return_type='both',  widths=[0.5]*len(labels))

        color_pallet = {"b" : "paleturquoise", "g" : "lightgreen"}
        if colors == None:
            colors = ["b"] * len(patients)
        pyplot.suptitle('')
        it_1 = 0
        for index, (ax,row)  in bp.items():
            print(row)
            ax.set_xlabel('')
            ax.set_ylim(ymin=0)
            ax.set_ylabel("TRE (mm)", fontsize=18)
            ax.set_yticks([0,1,2,3,4,5,6,7,8,9,10,15,20,25])  # Set label locations.
            for i,object in enumerate(row["boxes"]):
                object.set_edgecolor("black")
                object.set_facecolor(color_pallet[colors[i]])
                object.set_alpha(0.7)
                object.set_linewidth(1.0)
            
            for i,object in enumerate(row["medians"]):
                object.set_color("indianred")
                xy = object.get_xydata()
                object.set_linewidth(2.0)
                it_1+=1
        return self
    
    def show(self):
        pyplot.show()


class Attribute(dict[str, Any]):

    def __init__(self, attributes : dict[str, Any] = {}) -> None:
        super().__init__()
        for k, v in attributes.items():
            super().__setitem__(copy.deepcopy(k), copy.deepcopy(v))
    
    def __getitem__(self, key: str) -> Any:
        i = len([k for k in super().keys() if k.startswith(key)])
        if i > 0 and "{}_{}".format(key, i-1) in super().keys():   
            return str(super().__getitem__("{}_{}".format(key, i-1)))
        else:
            raise NameError("{} not in cache_attribute".format(key))

    def __setitem__(self, key: str, value: Any) -> None:
        if "_" not in key:
            i = len([k for k in super().keys() if k.startswith(key)])
            result = None
            if isinstance(value, torch.Tensor):
                result = str(value.numpy())
            else:
                result = str(value)
            result = result.replace('\n', '')
            super().__setitem__("{}_{}".format(key, i), result)
        else:
            result = None
            if isinstance(value, torch.Tensor):
                result = str(value.numpy())
            else:
                result = str(value)
            result = result.replace('\n', '')
            super().__setitem__(key, result)

    def pop(self, key: str) -> Any:
        i = len([k for k in super().keys() if k.startswith(key)])
        if i > 0 and "{}_{}".format(key, i-1) in super().keys():   
            return super().pop("{}_{}".format(key, i-1))
        else:
            raise NameError("{} not in cache_attribute".format(key))

    def get_np_array(self, key) -> np.ndarray:
        return np.fromstring(self[key][1:-1], sep=" ", dtype=np.double)
    
    def get_tensor(self, key) -> torch.Tensor:
        return torch.tensor(self.get_np_array(key))
    
    def pop_np_array(self, key):
        return np.fromstring(self.pop(key)[1:-1], sep=" ", dtype=np.double)
    
    def pop_tensor(self, key) -> torch.Tensor:
        return torch.tensor(self.pop_np_array(key))
    
    def __contains__(self, key: str) -> bool:
        return len([k for k in super().keys() if k.startswith(key)]) > 0
    
def isAnImage(attributes: Attribute):
    return "Origin" in attributes and "Spacing" in attributes and "Direction" in attributes

def data_to_image(data : np.ndarray, attributes: Attribute) -> sitk.Image:
    if not isAnImage(attributes):
        raise NameError("Data is not an image")
    if data.shape[0] == 1:
        image = sitk.GetImageFromArray(data[0])
    else:
        data = data.transpose(tuple([i+1 for i in range(len(data.shape)-1)]+[0]))
        image = sitk.GetImageFromArray(data, isVector=True)
    image.SetOrigin(attributes.get_np_array("Origin").tolist())
    image.SetSpacing(attributes.get_np_array("Spacing").tolist())
    image.SetDirection(attributes.get_np_array("Direction").tolist())
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
        def getNames(self, group: str) -> list[str]:
            pass

        @abstractmethod
        def isExist(self, group: str, name: Union[str, None] = None) -> bool:
            pass
        
        @abstractmethod
        def getInfos(self, group: Union[str, None], name: str) -> tuple[list[int], Attribute]:
            pass

    class H5File(AbstractFile):

        def __init__(self, filename: str, read: bool) -> None:
            self.h5: Union[h5py.File, None] = None
            self.filename = filename
            if not self.filename.endswith(".h5"):
                self.filename += ".h5"
            self.read = read

        def __enter__(self):
            args = {}
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
        
        def file_to_data(self, groups: str, name: str) -> tuple[np.ndarray, Attribute]:
            dataset = self._getDataset(groups, name)
            data = np.zeros(dataset.shape, dataset.dtype)
            dataset.read_direct(data)
            return data, Attribute({k : str(v) for k, v in dataset.attrs.items()})

        def data_to_file(self, name : str, data : Union[sitk.Image, sitk.Transform, np.ndarray], attributes : Union[Attribute, None] = None) -> None:
            if attributes is None:
                attributes = Attribute()
            if isinstance(data, sitk.Image):
                image = data
                attributes["Origin"] = np.asarray(image.GetOrigin())
                attributes["Spacing"] = np.asarray(image.GetSpacing())
                attributes["Direction"] = np.asarray(image.GetDirection())
                data = sitk.GetArrayFromImage(image)

                if image.GetNumberOfComponentsPerPixel() == 1:
                    data = np.expand_dims(data, 0)
                else:
                    data = np.transpose(data, (len(data.shape)-1, *[i for i in range(len(data.shape)-1)]))
            elif isinstance(data, sitk.Transform):
                transforms = []
                if isinstance(data, sitk.CompositeTransform):
                    for i in range(data.GetNumberOfTransforms()):
                        transforms.append(data.GetNthTransform(i))    
                else:
                    transforms.append(data)
                datas = []
                for i, transform in enumerate(transforms):
                    if isinstance(transform, sitk.Euler3DTransform):
                        transform_type = "Euler3DTransform_double_3_3"
                    if isinstance(transform, sitk.AffineTransform):
                        transform_type = "AffineTransform_double_3_3"
                    if isinstance(transform, sitk.BSplineTransform):
                        transform_type = "BSplineTransform_double_3_3"
                    attributes["{}:Transform".format(i)] = transform_type
                    attributes["{}:FixedParameters".format(i)] = transform.GetFixedParameters()

                    datas.append(np.asarray(transform.GetParameters()))
                data = np.asarray(datas)

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
            dataset.attrs.update({k : str(v) for k, v in attributes.items()})
        
        def isExist(self, group: str, name: Union[str, None] = None) -> bool:
            if group in self.h5:
                if isinstance(self.h5[group], h5py.Dataset):
                    return True
                elif name is not None:
                    return name in self.h5[group]
                else:
                    return False
            return False

        def getNames(self, groups: str, h5_group: h5py.Group = None) -> list[str]:
            names = []
            if h5_group is None:
                h5_group = self.h5
            group = groups.split("/")[0]
            if group == "":
                names = [dataset.name.split("/")[-1] for dataset in h5_group.values() if isinstance(dataset, h5py.Dataset)]
            elif group == "*":
                for k in h5_group.keys():
                    if isinstance(h5_group[k], h5py.Group):
                        names.extend(self.getNames("/".join(groups.split("/")[1:]), h5_group[k]))
            else:
                if group in h5_group:
                    names.extend(self.getNames("/".join(groups.split("/")[1:]), h5_group[group]))
            return names
        
        def _getDataset(self, groups: str, name: str, h5_group: h5py.Group = None) -> h5py.Dataset:
            if h5_group is None:
                h5_group = self.h5
            if groups != "":
                group = groups.split("/")[0]
            else:
                group = ""
            result = None
            if group == "":
                if name in h5_group:
                    result = h5_group[name]
            elif group == "*":
                for k in h5_group.keys():
                    if isinstance(h5_group[k], h5py.Group):
                        result_tmp = self._getDataset("/".join(groups.split("/")[1:]), name, h5_group[k])
                        if result_tmp is not None:
                            result = result_tmp
            else:
                if group in h5_group:
                    result_tmp = self._getDataset("/".join(groups.split("/")[1:]), name, h5_group[group])
                    if result_tmp is not None:
                        result = result_tmp
            return result
        
        def getInfos(self, groups: str, name: str) -> tuple[list[int], Attribute]:
            dataset = self._getDataset(groups, name)
            return (dataset.shape, Attribute({k : str(v) for k, v in dataset.attrs.items()}))
        
    class SitkFile(AbstractFile):

        def __init__(self, filename: str, read: bool, format: str) -> None:
            self.filename = filename
            self.read = read
            self.format = format
     
        def file_to_data(self, group: str, name: str) -> tuple[np.ndarray, Attribute]:
            attributes = Attribute()
            if os.path.exists("{}{}.{}".format(self.filename, name, self.format)):
                image = sitk.ReadImage("{}{}.{}".format(self.filename, name, self.format))
                
                attributes["Origin"] = np.asarray(image.GetOrigin())
                attributes["Spacing"] = np.asarray(image.GetSpacing())
                attributes["Direction"] = np.asarray(image.GetDirection())
                
                for k in image.GetMetaDataKeys():
                    attributes[k] = image.GetMetaData(k)
                data = sitk.GetArrayFromImage(image)
                if image.GetNumberOfComponentsPerPixel() == 1:
                    data = np.expand_dims(data, 0)
                else:
                    data = np.transpose(data, (len(data.shape)-1, *[i for i in range(len(data.shape)-1)]))
            elif os.path.exists("{}{}.itk.txt".format(self.filename, name)): 
                data = sitk.ReadTransform("{}{}.itk.txt".format(self.filename, name))
                transforms = []
                if isinstance(data, sitk.CompositeTransform):
                    for i in range(data.GetNumberOfTransforms()):
                        transforms.append(data.GetNthTransform(i))    
                else:
                    transforms.append(data)
                datas = []
                for i, transform in enumerate(transforms):
                    if isinstance(transform, sitk.Euler3DTransform):
                        transform_type = "Euler3DTransform_double_3_3"
                    if isinstance(transform, sitk.AffineTransform):
                        transform_type = "AffineTransform_double_3_3"
                    if isinstance(transform, sitk.BSplineTransform):
                        transform_type = "BSplineTransform_double_3_3"
                    attributes["{}:Transform".format(i)] = transform_type
                    attributes["{}:FixedParameters".format(i)] = transform.GetFixedParameters()

                    datas.append(np.asarray(transform.GetParameters()))
                data = np.asarray(datas)
            elif os.path.exists("{}{}.fcsv".format(self.filename, name)):
                with open("{}{}.fcsv".format(self.filename, name), newline="") as csvfile:
                    reader = csv.reader(filter(lambda row: row[0]!='#', csvfile))
                    lines = list(reader)
                    data = np.zeros((len(list(lines)), 3), dtype=np.double)
                    for i, row in enumerate(lines):
                        data[i] = np.array(row[1:4], dtype=np.double)
                    csvfile.close()
            elif os.path.exists("{}{}.xml".format(self.filename, name)):
                with open("{}{}.xml".format(self.filename, name), 'rb') as xml_file:
                    result = etree.parse(xml_file, etree.XMLParser(remove_blank_text=True)).getroot()
                    xml_file.close()
                    return result
            return data, attributes
                
        def data_to_file(self, name : str, data : Union[sitk.Image, sitk.Transform, np.ndarray], attributes : Attribute = Attribute()) -> None:
            if not os.path.exists(self.filename) and (isinstance(data, sitk.Image) or isinstance(data, sitk.Transform) or isinstance(data, vtk.vtkPolyData) or isAnImage(attributes) or (len(data.shape) == 2 and data.shape[1] == 3 and data.shape[0] > 0)):
                os.makedirs(self.filename)
            if isinstance(data, sitk.Image):
                for k, v in attributes.items():
                    data.SetMetaData(k, v)
                sitk.WriteImage(data, "{}{}.{}".format(self.filename, name, self.format))
            elif isinstance(data, sitk.Transform):
                sitk.WriteTransform(data, "{}{}.itk.txt".format(self.filename, name))
            elif isinstance(data, vtk.vtkPolyData):
                vtkWriter = vtk.vtkPolyDataWriter()
                vtkWriter.SetFileName("{}{}.vtk".format(self.filename, name))
                vtkWriter.SetInputData(data)
                vtkWriter.Write()
            elif isAnImage(attributes):   
                self.data_to_file(name, data_to_image(data, attributes), attributes)
            elif (len(data.shape) == 2 and data.shape[1] == 3 and data.shape[0] > 0):
                data = np.round(data, 4)
                with open("{}{}.fcsv".format(self.filename, name), 'w') as f:
                    f.write("# Markups fiducial file version = 4.6\n# CoordinateSystem = 0\n# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
                    for i in range(data.shape[0]):
                        f.write("vtkMRMLMarkupsFiducialNode_"+str(i+1)+","+str(data[i, 0])+","+str(data[i, 1])+","+str(data[i, 2])+",0,0,0,1,1,1,0,F-"+str(i+1)+",,vtkMRMLScalarVolumeNode1\n")
                    f.close()
            elif "path" in attributes:
                if os.path.exists("{}{}.xml".format(self.filename, name)):
                    with open("{}{}.xml".format(self.filename, name), 'rb') as xml_file:
                        root = etree.parse(xml_file, etree.XMLParser(remove_blank_text=True)).getroot()
                        xml_file.close()
                else:
                    root = etree.Element(name)
                node = root
                path = attributes["path"].split(':')

                for node_name in path:
                    node_tmp = node.find(node_name)
                    if node_tmp == None:
                        node_tmp = etree.SubElement(node, node_name)
                        node.append(node_tmp)
                    node = node_tmp
                if attributes != None:
                    for attribute_tmp in attributes.keys():
                        attribute = "_".join(attribute_tmp.split("_")[:-1])
                        if attribute != "path":
                            node.set(attribute, attributes[attribute])
                if data.size > 0:
                    node.text = np.array2string(data, separator=',')[1:-1].replace('\n','')
                with open("{}{}.xml".format(self.filename, name), 'wb') as f:
                    f.write(etree.tostring(root, pretty_print=True, encoding='utf-8'))
                    f.close()
                 
        def isExist(self, group: str, name: Union[str, None] = None) -> bool:
            return os.path.exists("{}{}.{}".format(self.filename, group, self.format)) or os.path.exists("{}{}.itk.txt".format(self.filename, group)) or os.path.exists("{}{}.fcsv".format(self.filename, group))
        
        def getNames(self, group: str) -> list[str]:
            raise NotImplementedError()

        def getInfos(self, group: str, name: str) -> tuple[list[int], Attribute]:
            attributes = Attribute()
            if os.path.exists("{}{}{}.{}".format(self.filename, group if group is not None else "", name, self.format)):
                file_reader = sitk.ImageFileReader()
                file_reader.SetFileName("{}{}{}.{}".format(self.filename, group if group is not None else "", name, self.format))
                file_reader.ReadImageInformation()
                attributes["Origin"] = np.asarray(file_reader.GetOrigin())
                attributes["Spacing"] = np.asarray(file_reader.GetSpacing())
                attributes["Direction"] = np.asarray(file_reader.GetDirection())
                for k in file_reader.GetMetaDataKeys():
                    attributes[k] = file_reader.GetMetaData(k)
                size = list(file_reader.GetSize())
                if len(size) == 3:
                    size = list(reversed(size))
                size = [file_reader.GetNumberOfComponents()]+size
            else:
                data, attributes = self.file_to_data(group if group is not None else "", name)
                size = data.shape
            return tuple(size), attributes

    class File(ABC):

        def __init__(self, filename: str, read: bool, format: str) -> None:
            self.filename = filename
            self.read = read
            self.file = None
            self.format = format

        def __enter__(self):
            if self.format == "h5":
                self.file = DatasetUtils.H5File(self.filename, self.read)
            else:
                self.file = DatasetUtils.SitkFile(self.filename+"/", self.read, self.format)
            self.file.__enter__()
            return self.file

        def __exit__(self, type, value, traceback):
            self.file.__exit__(type, value, traceback)

    def __init__(self, filename : str, format: str) -> None:
        if format != "h5" and not filename.endswith("/"):
            filename = "{}/".format(filename)
        self.is_directory = filename.endswith("/") 
        self.filename = filename
        self.format = format
        
    def write(self, group : str, name : str, data : Union[sitk.Image, sitk.Transform, np.ndarray], attributes : Attribute = Attribute()):
        if self.is_directory:
            if not os.path.exists(self.filename):
                os.makedirs(self.filename)
        if self.is_directory:
            s_group = group.split("/")
            if len(s_group) > 1:
                subDirectory = "/".join(s_group[:-1])
                name = "{}/{}".format(subDirectory, name)
                group = s_group[-1]
            with DatasetUtils.File("{}{}".format(self.filename, name), False, self.format) as file:
                file.data_to_file(group, data, attributes)
        else:
            with DatasetUtils.File(self.filename, False, self.format) as file:
                file.data_to_file("{}/{}".format(group, name), data, attributes)
    
    def readData(self, groups : str, name : str) -> tuple[np.ndarray, Attribute]:
        if not os.path.exists(self.filename):
            raise NameError("Dataset {} not found".format(self.filename))
        if self.is_directory:
            for subDirectory in self._getSubDirectories(groups):
                group = groups.split("/")[-1]
                if os.path.exists("{}{}{}{}".format(self.filename, subDirectory, name, ".h5" if self.format == "h5" else "")):
                    with DatasetUtils.File("{}{}{}".format(self.filename, subDirectory, name), False, self.format) as file:
                        result = file.file_to_data("", group)
        else:
            with DatasetUtils.File(self.filename, False, self.format) as file:
                result = file.file_to_data(groups, name)
        return result
    
    def readTransform(self, group : str, name : str) -> sitk.Transform:
        if not os.path.exists(self.filename):
            raise NameError("Dataset {} not found".format(self.filename))
        transformParameters, attribute = self.readData(group, name)
        transforms_type = [v for k, v in attribute.items() if k.endswith(":Transform_0")]
        transforms = []
        for i, transform_type in enumerate(transforms_type):
            if transform_type == "Euler3DTransform_double_3_3":
                transform = sitk.Euler3DTransform()
            if transform_type == "AffineTransform_double_3_3":
                transform = sitk.AffineTransform(3)
            if transform_type == "BSplineTransform_double_3_3":
                transform = sitk.BSplineTransform(3)
            transform.SetFixedParameters(eval(attribute["{}:FixedParameters".format(i)]))
            transform.SetParameters(tuple(transformParameters[i]))
            transforms.append(transform)
        return sitk.CompositeTransform(transforms) if len(transforms) > 1 else transforms[0]

    def readImage(self, group : str, name : str):
         data, attribute = self.readData(group, name)
         return data_to_image(data, attribute)
            
    def getSize(self, group: str) -> int:
        return len(self.getNames(group))
    
    def isGroupExist(self, group: str) -> bool:
        return self.getSize(group) > 0
    
    def isDatasetExist(self, group: str, name: str) -> bool:
        return name in self.getNames(group)
    
    def _getSubDirectories(self, groups: str, subDirectory: str = ""):
        group = groups.split("/")[0]
        subDirectories = []
        if len(groups.split("/")) == 1:
            subDirectories.append(subDirectory)
        elif group == "*":
            for k in os.listdir("{}{}".format(self.filename, subDirectory)):
                if not os.path.isfile("{}{}{}".format(self.filename, subDirectory, k)):
                    subDirectories.extend(self._getSubDirectories("/".join(groups.split("/")[1:]), "{}{}/".format(subDirectory , k)))
        else:
            subDirectory = "{}{}/".format(subDirectory, group)
            if os.path.exists("{}{}".format(self.filename, subDirectory)):
                subDirectories.extend(self._getSubDirectories("/".join(groups.split("/")[1:]), subDirectory))
        return subDirectories

    def getNames(self, groups: str, index: Optional[list[int]] = None, subDirectory: str = "") -> list[str]:
        names = []
        if self.is_directory:
            for subDirectory in self._getSubDirectories(groups):
                group = groups.split("/")[-1]
                if os.path.exists("{}{}".format(self.filename, subDirectory)):
                    for name in sorted(os.listdir("{}{}".format(self.filename, subDirectory))):
                        if os.path.isfile("{}{}{}".format(self.filename, subDirectory, name)) or self.format != "h5":
                            with DatasetUtils.File("{}{}{}".format(self.filename, subDirectory, name), True, self.format) as file:
                                if file.isExist(group):
                                    names.append(name.replace(".h5", "") if self.format == "h5" else name)
        else:
            with DatasetUtils.File(self.filename, True, self.format) as file:
                names = file.getNames(groups)
        return [name for i, name in enumerate(names) if index is None or i in index]
    
    def getInfos(self, groups: str, name: str) -> tuple[list[int], Attribute]:
        if self.is_directory:
            for subDirectory in self._getSubDirectories(groups):
                group = groups.split("/")[-1]
                if os.path.exists("{}{}{}{}".format(self.filename, subDirectory, name, ".h5" if self.format == "h5" else "")):
                    with DatasetUtils.File("{}{}{}".format(self.filename, subDirectory, name), True, self.format) as file:
                        result = file.getInfos("", group)      
        else:
            with DatasetUtils.File(self.filename, True, self.format) as file:
                result = file.getInfos(groups, name)
        return result
    
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

def gpuInfo() -> str:
    if os.environ["CUDA_VISIBLE_DEVICES"] == "":
        return ""
    
    devices = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    device = devices[0]
    
    if device < pynvml.nvmlDeviceGetCount():
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    else:
        return ""
    node_name = "Node: {} " +os.environ["SLURMD_NODENAME"] if "SLURMD_NODENAME" in os.environ else ""
    return  "{}GPU({}) Memory GPU ({:.2f}G ({:.2f} %))".format(node_name, devices, float(memory.used)/(10**9), float(memory.used)/float(memory.total)*100)

def getMaxGPUMemory(device : Union[int, torch.device]) -> float:
    if isinstance(device, torch.device):
        if str(device).startswith("cuda:"):
            device = int(str(device).replace("cuda:", ""))
        else:
            return 0
    device = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")][device]
    if device < pynvml.nvmlDeviceGetCount():
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    else:
        return 0
    return float(memory.total)/(10**9)


def getGPUMemory(device : Union[int, torch.device]) -> float:
    if isinstance(device, torch.device):
        if str(device).startswith("cuda:"):
            device = int(str(device).replace("cuda:", ""))
        else:
            return 0
    device = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")][device]
    if device < pynvml.nvmlDeviceGetCount():
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    else:
        return 0
    return float(memory.used)/(10**9)


class NeedDevice(ABC):

    def __init__(self) -> None:
        super().__init__()
        self.device : torch.device
    
    def setDevice(self, device : int):
        self.device = getDevice(device)

def getDevice(device : int):
    return device if torch.cuda.is_available() and device >=0 else torch.device("cpu")

class State(Enum):
    TRAIN = "TRAIN"
    RESUME = "RESUME"
    TRANSFER_LEARNING = "TRANSFER_LEARNING"
    FINE_TUNNING = "FINE_TUNNING"
    PREDICTION = "PREDICTION"
    METRIC = "METRIC"
    HYPERPARAMETER = "HYPERPARAMETER"
    
    def __str__(self) -> str:
        return self.value

def get_patch_slices_from_nb_patch_per_dim(patch_size_tmp: list[int], nb_patch_per_dim : list[tuple[int, bool]], overlap: Union[int, None]) -> list[tuple[slice]]:
    patch_slices = []
    slices : list[list[slice]] = []
    if overlap is None:
        overlap = 0
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
            start = (patch_size[dim]-overlap)*index
            end = start + patch_size[dim]
            slices[dim].append(slice(start,end))
    for chunk in itertools.product(*slices):
        patch_slices.append(tuple(chunk))
    return patch_slices

def get_patch_slices_from_shape(patch_size: list[int], shape : list[int], overlap: Union[int, None]) -> tuple[list[tuple[slice]], list[tuple[int, bool]]]:
    if len(shape) != len(patch_size):
        return [tuple([slice(0, s) for s in shape])], [(1, True)]*len(shape)
    
    patch_slices = []
    nb_patch_per_dim = []
    slices : list[list[slice]] = []
    if overlap is None:
        size = [np.ceil(a/b) for a, b in zip(shape, patch_size)]
        tmp = np.zeros(len(size), dtype=np.int_)
        for i, s in enumerate(size):
            if s > 1:
                tmp[i] = np.mod(patch_size[i]-np.mod(shape[i], patch_size[i]), patch_size[i])//(size[i]-1)
        overlap = tmp
    else:
        overlap = [overlap if size > 1 else 0 for size in patch_size]
    
    for dim in range(len(shape)):
        assert overlap[dim] < patch_size[dim],  "Overlap must be less than patch size"

    for dim in range(len(shape)):
        slices.append([])
        index = 0
        while True:
            start = (patch_size[dim]-overlap[dim])*index

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

def formatMaskLabel(mask: sitk.Image, labels: list[tuple[int, int]]) -> sitk.Image:
    data = sitk.GetArrayFromImage(mask)
    result_data = np.zeros_like(data, np.uint8)

    for label_old, label_new in labels:
        result_data[np.where(data == label_old)] = label_new

    result = sitk.GetImageFromArray(result_data)
    result.CopyInformation(mask)
    return result

def _logSignalFormat(input : np.ndarray):
    return {str(i): channel for i, channel in enumerate(input)}

def _logImageFormat(input : np.ndarray):
    if len(input.shape) == 2:
        input = np.expand_dims(input, axis=0)

    if len(input.shape) == 3 and input.shape[0] != 1:
        input = np.expand_dims(input, axis=0)

    if len(input.shape) == 4:
        input = input[:, input.shape[1]//2]
    
    if input.dtype == np.uint8:
        return input
        
    input = input.astype(float)
    b = -np.min(input)
    if (np.max(input)+b) > 0:
        return (input+b)/(np.max(input)+b)
    else:
        return 0*input

def _logImagesFormat(input : np.ndarray):
    result = []
    for n in range(input.shape[0]):
        result.append(_logImageFormat(input[n]))
    result = np.stack(result, axis=0)
    return result

def _logVideoFormat(input : np.ndarray):
    result = []
    for t in range(input.shape[1]):
        result.append( _logImagesFormat(input[:, t,...]))
    result = np.stack(result, axis=1)

    nb_channel = result.shape[2]
    if nb_channel < 3:
        channel_split = [result[:, :, 0, ...] for i in range(3)]
    else:
        channel_split = np.split(result, 3, axis=0)
    input = np.zeros((result.shape[0], result.shape[1], 3, *list(result.shape[3:])))
    for i, channels in enumerate(channel_split):
        input[:,:,i] = np.mean(channels, axis=0)
    return input

class DataLog(Enum):
    SIGNAL   = lambda tb, name, layer, it : [tb.add_scalars(name, _logSignalFormat(layer[b, :, 0]), layer.shape[0]*it+b) for b in range(layer.shape[0])],
    IMAGE   = lambda tb, name, layer, it : tb.add_image(name, _logImageFormat(layer[0]), it),
    IMAGES  = lambda tb, name, layer, it : tb.add_images(name, _logImagesFormat(layer), it),
    VIDEO   = lambda tb, name, layer, it : tb.add_video(name, _logVideoFormat(layer), it),
    AUDIO   = lambda tb, name, layer, it : tb.add_audio(name, _logImageFormat(layer), it)

class Log():

    def __init__(self, name: str) -> None:
        if not os.path.exists("{}{}".format(os.environ["DL_API_STATISTICS_DIRECTORY"], name)):
            os.makedirs("{}{}".format(os.environ["DL_API_STATISTICS_DIRECTORY"], name))
        self.file = open("{}{}/log.txt".format(os.environ["DL_API_STATISTICS_DIRECTORY"], name), 'w')
        self.stdout_bak = sys.stdout
        self.stderr_bak = sys.stderr
        self.verbose = os.environ["DEEP_LEARNING_VERBOSE"] == "True"

    def __enter__(self):
        self.file.__enter__()
        sys.stdout = self
        sys.stderr = self
        return self
    
    def __exit__(self, type, value, traceback):
        self.file.__exit__(type, value, traceback)
        sys.stdout = self.stdout_bak
        sys.stderr = self.stderr_bak
         
    def write(self, msg):
        if msg.strip() != "":
            self.file.write(msg)
            if self.verbose:
                print(msg, file=sys.__stdout__)

    def flush(self):
        pass
    
class TensorBoard():
    from totalsegmentator.python_api import totalsegmentator
    def __init__(self, name: str) -> None:
        self.process = None
        self.name = name

    def __enter__(self):
        if "DEEP_LEARNING_TENSORBOARD_PORT" in os.environ:
            command = ["tensorboard", "--logdir", STATISTICS_DIRECTORY() + self.name + "/", "--port", os.environ["DEEP_LEARNING_TENSORBOARD_PORT"], "--bind_all"]
            self.process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(('10.255.255.255', 1))
                IP = s.getsockname()[0]
            except Exception:
                IP = '127.0.0.1'
            finally:
                s.close()
            print("Tensorboard : http://{}:{}/".format(IP, os.environ["DEEP_LEARNING_TENSORBOARD_PORT"]))
        return self
    
    def __exit__(self, type, value, traceback):
        if self.process is not None:
            self.process.terminate()
            self.process.wait()

class DistributedObject():

    def __init__(self, name: str) -> None:
        self.port = find_free_port()
        self.dataloader : list[list[DataLoader]]
        self.manual_seed: bool = None
        self.name = name
        self.size = 1
    
    @abstractmethod
    def setup(self, world_size: int):
        pass
    
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    @abstractmethod
    def run_process(self, world_size: int, global_rank: int, local_rank: int, dataloaders: list[DataLoader]):
        pass 
    
    def getMeasure(world_size: int, global_rank: int, gpu: int, models: dict[str, torch.nn.Module], n: int) -> dict[str, tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]]:
        data = {}
        for label, model in models.items():
            for name, network in model.getNetworks().items():
                if network.measure is not None:
                    data["{}{}".format(name, label)] = (network.measure.format(True, n), network.measure.format(False, n))
        outputs = synchronize_data(world_size, gpu, data)
        result = {}
        if global_rank == 0:
            for output in outputs:
                for k, v in output.items():
                    for t in range(len(v)):
                        for u, n in v[t].items():
                            if k not in result:
                                result[k] = ({}, {})
                            if u not in result[k][t]:
                                result[k][t][u] = (n[0], 0)
                            result[k][t][u] = (result[k][t][u][0], result[k][t][u][1]+n[1]/world_size)
        return result

    def __call__(self, rank: Union[int, None] = None) -> None:
        with Log(self.name) as log:
            world_size = len(self.dataloader)
            global_rank, local_rank = setupGPU(world_size, self.port, rank)
            if global_rank is None:
                return
            if torch.cuda.is_available():
                pynvml.nvmlInit()
            if self.manual_seed is not None:
                np.random.seed(self.manual_seed * world_size + global_rank)
                random.seed(self.manual_seed * world_size + global_rank)
                torch.manual_seed(self.manual_seed * world_size + global_rank)
            torch.backends.cudnn.benchmark = self.manual_seed is None
            torch.backends.cudnn.deterministic = self.manual_seed is not None
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            dataloaders = self.dataloader[global_rank]
            if torch.cuda.is_available():    
                torch.cuda.set_device(local_rank)
            for dataloader in dataloaders:
                dataloader.dataset.to(local_rank)
                dataloader.dataset.load()
        
            self.run_process(world_size, global_rank, local_rank, dataloaders)
            if torch.cuda.is_available():
                pynvml.nvmlShutdown()
            cleanup()

def setupAPI(parser: argparse.ArgumentParser) -> DistributedObject:
    # API arguments
    api_args = parser.add_argument_group('API arguments')
    api_args.add_argument("type", type=State, choices=list(State))
    api_args.add_argument('-y', action='store_true', help="Accept overwrite")
    api_args.add_argument('-tb', action='store_true', help='Start TensorBoard')
    api_args.add_argument("-c", "--config", type=str, default="None", help="Configuration file location")
    api_args.add_argument("-g", "--gpu", type=str, default=os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else "", help="List of GPU")
    api_args.add_argument('--num-workers', '--num_workers', default=4, type=int, help='No. of workers per DataLoader & GPU')
    api_args.add_argument("-models_dir", "--MODELS_DIRECTORY", type=str, default="./Models/", help="Models location")
    api_args.add_argument("-checkpoints_dir", "--CHECKPOINTS_DIRECTORY", type=str, default="./Checkpoints/", help="Checkpoints location")
    api_args.add_argument("-model", "--MODEL", type=str, default="", help="URL Model")
    api_args.add_argument("-predictions_dir", "--PREDICTIONS_DIRECTORY", type=str, default="./Predictions/", help="Predictions location")
    api_args.add_argument("-metrics_dir", "--METRICS_DIRECTORY", type=str, default="./Metrics/", help="Metrics location")
    api_args.add_argument("-statistics_dir", "--STATISTICS_DIRECTORY", type=str, default="./Statistics/", help="Statistics location")
    api_args.add_argument("-setups_dir", "--SETUPS_DIRECTORY", type=str, default="./Setups/", help="Setups location")
    api_args.add_argument('-log', action='store_true', help='Save log')
    api_args.add_argument('-quiet', action='store_false', help='')
    
    
    args = parser.parse_args()
    config = vars(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]
    os.environ["DL_API_MODELS_DIRECTORY"] = config["MODELS_DIRECTORY"]
    os.environ["DL_API_CHECKPOINTS_DIRECTORY"] = config["CHECKPOINTS_DIRECTORY"]
    os.environ["DL_API_PREDICTIONS_DIRECTORY"] = config["PREDICTIONS_DIRECTORY"]
    os.environ["DL_API_METRICS_DIRECTORY"] = config["METRICS_DIRECTORY"]
    os.environ["DL_API_STATISTICS_DIRECTORY"] = config["STATISTICS_DIRECTORY"]
    
    os.environ["DL_API_STATE"] = str(config["type"])
    
    os.environ["DL_API_MODEL"] = config["MODEL"]

    os.environ["DL_API_SETUPS_DIRECTORY"] = config["SETUPS_DIRECTORY"]

    os.environ["DL_API_OVERWRITE"] = "{}".format(config["y"])
    os.environ["DEEP_LEANING_API_CONFIG_MODE"] = "Done"
    if config["tb"]:
        os.environ["DEEP_LEARNING_TENSORBOARD_PORT"] = str(find_free_port())

    os.environ["DEEP_LEARNING_VERBOSE"] = str(config["quiet"])

    if config["config"] == "None":
        if config["type"] is State.PREDICTION:
             os.environ["DEEP_LEARNING_API_CONFIG_FILE"] = "Prediction.yml"
        elif config["type"] is State.METRIC:
            os.environ["DEEP_LEARNING_API_CONFIG_FILE"] = "Metric.yml"
        elif config["type"] is State.HYPERPARAMETER:
            os.environ["DEEP_LEARNING_API_CONFIG_FILE"] = "Hyperparameter.yml"
        else:
            os.environ["DEEP_LEARNING_API_CONFIG_FILE"] = "Config.yml"
    else:
        os.environ["DEEP_LEARNING_API_CONFIG_FILE"] = config["config"]
    torch.autograd.set_detect_anomaly(True)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    if config["type"] is State.PREDICTION:    
        from DeepLearning_API.predictor import Predictor
        os.environ["DEEP_LEARNING_API_ROOT"] = "Predictor"
        return Predictor(config=CONFIG_FILE())    
    elif config["type"] is State.METRIC:
        from DeepLearning_API.metric import Metric
        os.environ["DEEP_LEARNING_API_ROOT"] = "Metric"
        return Metric(config=CONFIG_FILE())
    elif config["type"] is State.HYPERPARAMETER:
        from DeepLearning_API.hyperparameter import Hyperparameter
        os.environ["DEEP_LEARNING_API_ROOT"] = "Hyperparameter"
        return Hyperparameter(config=CONFIG_FILE())
    else:
        from DeepLearning_API.trainer import Trainer
        os.environ["DEEP_LEARNING_API_ROOT"] = "Trainer"
        return Trainer(config=CONFIG_FILE())

import submitit

def setupGPU(world_size: int, port: int, rank: Union[int, None] = None) -> tuple[int , int]:
    try:
        host_name = subprocess.check_output("scontrol show hostnames {}".format(os.getenv('SLURM_JOB_NODELIST')).split()).decode().splitlines()[0]
    except:
        host_name = "localhost"
    if rank is None:
        job_env = submitit.JobEnvironment()
        global_rank = job_env.global_rank
        local_rank = job_env.local_rank
    else:
        global_rank = rank
        local_rank = rank
    if global_rank >= world_size:
        return None, None
    print("tcp://{}:{}".format(host_name, port))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        dist.init_process_group("nccl", rank=global_rank, init_method="tcp://{}:{}".format(host_name, port), world_size=world_size)
    return global_rank, local_rank

import socket
from contextlib import closing

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    
def cleanup():
    if torch.cuda.is_available():
        dist.destroy_process_group()

def synchronize_data(world_size: int, gpu: int, data: any) -> list[Any]:
    if torch.cuda.is_available():
        outputs: list[dict[str, tuple[dict[str, float], dict[str, float]]]] = [None for _ in range(world_size)]
        torch.cuda.set_device(gpu)
        dist.all_gather_object(outputs, data)
    else:
        outputs = [data]
    return outputs


def _resample(data: torch.Tensor, size: list[int]) -> torch.Tensor:
    if data.dtype == torch.uint8:
        mode = "nearest"
    elif len(data.shape) < 4:
        mode = "bilinear"
    else:
        mode = "trilinear"
    return F.interpolate(data.type(torch.float32).unsqueeze(0), size=tuple([s for s in reversed(size)]), mode=mode).squeeze(0).type(data.dtype)

def getFlatLabel(mask: sitk.Image, labels: list[int]) -> sitk.Image:
    data = sitk.GetArrayFromImage(mask)
    result_data = np.zeros_like(data, np.uint8)
    for label in labels:
        result_data[np.where(data == label)] = 1

    result = sitk.GetImageFromArray(result_data)
    result.CopyInformation(mask)        
    return result
