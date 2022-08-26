from abc import ABC
import importlib
from typing import Dict, List, Tuple
import torch
import numpy as np

import torch.nn.functional as F
import os

from DeepLearning_API.config import config
from DeepLearning_API.utils import NeedDevice, _getModule

class CriterionsAttr():

    @config()
    def __init__(self, l: float = 1.0) -> None:
        self.l = l

class CriterionsLoader():

    @config()
    def __init__(self, criterionsLoader: Dict[str, CriterionsAttr] = {"default:torch_nn_CrossEntropyLoss:Dice:NCC": CriterionsAttr()}) -> None:
        self.criterionsLoader = criterionsLoader

    def getCriterions(self, model_classname : str, output_group : str, target_group : str, train : bool) -> Dict[torch.nn.Module, float]:
        criterions = {}
        for module_classpath, criterionsAttr in self.criterionsLoader.items():
            module, name = _getModule(module_classpath, "measure")
            criterions[config("{}.Model.{}.outputsCriterions.{}.targetsCriterions.{}.criterionsLoader.{}".format("Trainer" if train else "Predictor", model_classname, output_group, target_group, module_classpath))(getattr(importlib.import_module(module), name))(config = None)] = criterionsAttr.l
        return criterions

class TargetCriterionsLoader():

    @config()
    def __init__(self, targetsCriterions : Dict[str, CriterionsLoader] = {"default" : CriterionsLoader()}) -> None:
        self.targetsCriterions = targetsCriterions
        
    def getTargetsCriterions(self, output_group : str, model_classname : str, train : bool) -> Dict[str, Dict[torch.nn.Module, float]]:
        targetsCriterions = {}
        for target_group, criterionsLoader in self.targetsCriterions.items():
            targetsCriterions[target_group] = criterionsLoader.getCriterions(model_classname, output_group, target_group, train)
        return targetsCriterions

class Criterion(NeedDevice, torch.nn.Module, ABC):

    def __init__(self) -> None:
        super().__init__()
        self.model = None

    def setModel(self, model):
        self.model = model

class Measure(NeedDevice):

    def __init__(self, model_classname : str, outputsCriterions: Dict[str, TargetCriterionsLoader], train : bool):
        super().__init__()
        self.outputsCriterions = {}
        for output_group, targetCriterionsLoader in outputsCriterions.items():
            self.outputsCriterions[output_group.replace("_", ".")] = targetCriterionsLoader.getTargetsCriterions(output_group, model_classname, train)
        self.value : List[float]= []
        self.values : Dict[str, List[float]] = dict()
        self.loss = torch.zeros((1))
        
        for output_group in self.outputsCriterions:
            for target_group in self.outputsCriterions[output_group]:
                for criterion in self.outputsCriterions[output_group][target_group]:
                    self.values["{}_{}_{}".format(output_group, target_group, criterion.__class__.__name__)] = []

    def setDevice(self, device: torch.device):
        super().setDevice(device)
        for output_group in self.outputsCriterions:
            for target_group in self.outputsCriterions[output_group]:
                for criterion in self.outputsCriterions[output_group][target_group]:
                    if isinstance(criterion, NeedDevice):
                        criterion.setDevice(device)

    def update(self, output_group, output : torch.Tensor, data_dict : Dict[str, torch.Tensor]):
        self.loss = torch.zeros((1), requires_grad = True).to(self.device, non_blocking=False)
        for target_group in self.outputsCriterions[output_group]:
            target = data_dict[target_group].to(self.device, non_blocking=False) if target_group in data_dict else None
            for criterion, l in self.outputsCriterions[output_group][target_group].items():
                result = criterion(output, target)
                self.loss += l*result
                self.values["{}_{}_{}".format(output_group, target_group, criterion.__class__.__name__)].append(result.item())
        self.value.append(self.loss.item())
        
    def getLastValue(self):
        return self.value[-1] if self.value else 0 
    
    def format(self) -> Dict[str, float]:
        result = dict()
        for name in self.values:
            result[name] = np.mean(self.values[name])
        return result

    def mean(self) -> float:
        return np.mean(self.value)
    
    def clear(self) -> None:
        self.value.clear()
        for name in self.values:
            self.values[name].clear()

class Dice(Criterion):
    
    def __init__(self, smooth : float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth
    
    def flatten(self, tensor : torch.Tensor) -> torch.Tensor:
        C = tensor.size(1)
        return tensor.permute((1, 0) + tuple(range(2, tensor.dim()))).contiguous().view(C, -1)

    def dice_per_channel(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = self.flatten(input)
        target = self.flatten(target)
        return (2.*(input * target).sum() + self.smooth)/(input.sum() + target.sum() + self.smooth)

    def forward(self, input: torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        input = F.one_hot(input.type(torch.int64)).permute(0, len(input.shape), *[i+1 for i in range(len(input.shape)-1)]).float()
        target = F.one_hot(target.type(torch.int64)).permute(0, len(target.shape), *[i+1 for i in range(len(target.shape)-1)]).float()
        return 1-torch.mean(self.dice_per_channel(input, target)) if self.training else torch.mean(self.dice_per_channel(input, target))

class GradientImages(Criterion):

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()
    
    @staticmethod
    def _image_gradient(image : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dx = image[:, :, 1:, :, :] - image[:, :, :-1, :, :]
        dy = image[:, :, :, 1:, :] - image[:, :, :, :-1, :]
        dz = image[:, :, :, :, 1:] - image[:, :, :, :, :-1]
       
        return dx, dy, dz

    def forward(self, input: torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        dx, dy, dz = GradientImages._image_gradient(input)
        if target is not None:
            dx_tmp, dy_tmp, dz_tmp = GradientImages._image_gradient(target)
            dx -= dx_tmp
            dy -= dy_tmp
            dz -= dz_tmp
    
        return dx.norm() + dy.norm() + dz.norm()

class BCE(Criterion):

    def __init__(self, target : float = 0) -> None:
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.register_buffer('target', torch.tensor(target).type(torch.float32))
    

    def forward(self, input: torch.Tensor, _ : torch.Tensor) -> torch.Tensor:
        target = self._buffers["target"]
        assert target
        return self.loss(input, target.to(self.device).expand_as(input))

class WGP(Criterion):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, gradient_norm: torch.Tensor, _ : torch.Tensor) -> torch.Tensor:
        return torch.mean((gradient_norm - 1)**2)

class ModelLoader():

        @config("Model")
        def __init__(self, classpath : str = "default:classificationV2.ConvNeXt") -> None:
            self.module, self.name = _getModule(classpath.split(".")[-1] if len(classpath.split(".")) > 1 else classpath, "networks" + "."+".".join(classpath.split(".")[:-1]) if len(classpath.split(".")) > 1 else "")
            
        def getModel(self):
            return getattr(importlib.import_module(self.module), self.name)(config = None, DL_args=os.environ["DEEP_LEARNING_API_CONFIG_VARIABLE"], DL_without = ["optimizer", "criterions", "scheduler", "nb_batch_per_step", "init_type", "init_gain"])

class MedPerceptualLoss(Criterion):
    
    
    def __init__(self, modelLoader : ModelLoader = ModelLoader(), path_model : str = "name", module_names : List[str] = ["ConvNextEncoder.ConvNexStage_2.BottleNeckBlock_0.Conv_2", "ConvNextEncoder.ConvNexStage_3.BottleNeckBlock_0.Conv_2"], shape: List[int] = [128, 256, 256]) -> None:
        super().__init__()
        self.model = modelLoader.getModel()
        state_dict = torch.load(path_model)
        self.model.load(state_dict)
        self.module_names = set(module_names)
        self.shape = shape
        self.mode = "trilinear" if  len(shape) == 3 else "bilinear"

    def setDevice(self, device: torch.device):
        super().setDevice(device)
        self.model.setDevice(self.device)
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        
    def forward(self, input : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        good_size = lambda shape : all([shape[-i-1] == size for i, size in enumerate(reversed(self.shape))])

        if not good_size(input.shape):
            input = F.interpolate(input, mode=self.mode, size=tuple(self.shape), align_corners=False)
        if not good_size(target.shape):
            target = F.interpolate(target, mode=self.mode, size=tuple(self.shape), align_corners=False)

        loss = torch.zeros((1), requires_grad = True).to(self.device, non_blocking=False)
        module_names = self.module_names.copy()
        for (name, input_layer), (_, target_layer) in zip(self.model.named_forward(input), self.model.named_forward(target)):
            if name in self.module_names:
                module_names.remove(name)
                loss += F.l1_loss(input_layer, target_layer)
                if not len(module_names):
                    break
        return loss

class KL_divergence(Criterion):

    class Latent():

        def __init__(self, dim : int = 100, mu : float = 0, sigma : float = 1) -> None:
            self.dim = dim
            self.mu = mu
            self.sigma = sigma 

    def __init__(self, module_names : Dict[str, Latent] = {"ConvNextEncoder.ConvNexStage_2.BottleNeckBlock_0.Linear_2": Latent()}) -> None:
        super().__init__()
        self.module_names = module_names

    def setModel(self, model):
        super().setModel(model)
        last_module = model
        for module_name, latent in self.module_names.items():
            for name in module_name.split(".")[:-1]:
                last_module = last_module[name]
            modules = last_module._modules.copy()
            last_module._modules.clear()

            for name, value in modules.items():
                last_module._modules[name] = value
                if name == module_name.split(".")[-1]:
                    module = last_module[module_name.split(".")[-1]]
                    last_module.add_module("mu", torch.nn.Linear(module.out_features, latent.dim), in_branch = last_module._modulesArgs[name].out_branch, out_branch = [-1])
                    last_module.add_module("sigma", torch.nn.Linear(module.out_features, latent.dim), in_branch = last_module._modulesArgs[name].out_branch, out_branch = [-2])

    def forward(self, input : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        raise NotImplemented

class Accuracy(Criterion):

    def __init__(self) -> None:
        super().__init__()
        self.n : int = 0
        self.corrects = torch.zeros((1))

    def forward(self, input : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        self.n += input.shape[0]
        self.corrects += (torch.argmax(torch.softmax(input, dim=1), dim=1) == target).sum().float()
        return self.corrects/self.n

