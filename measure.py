from abc import ABC
import importlib
import numpy as np
import torch

import torch.nn.functional as F
import os

from DeepLearning_API.config import config
from DeepLearning_API.utils import NeedDevice, _getModule
from DeepLearning_API.networks.blocks import LatentDistribution
from DeepLearning_API.networks.network import ModelLoader, Network
from typing import Callable

import torch.nn.functional as F
import itertools


modelsRegister = {}

class Criterion(NeedDevice, torch.nn.Module, ABC):

    def __init__(self) -> None:
        super().__init__()

    def init(self, model : torch.nn.Module, output_group : str, target_group : str) -> str:
        return output_group

class MaskedMSE(Criterion):

    def __init__(self) -> None:
        super().__init__()
        self.loss = torch.nn.MSELoss()

    def forward(self, input: torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0, dtype=torch.float32).to(self.device)
        for batch in range(input.shape[0]):
            mask = target[batch, -1, ...]
            if len(torch.where(mask == 1)[0]) > 0:
                loss += self.loss(torch.masked_select(input[batch, ...], mask == 1), torch.masked_select(target[batch, :-1, ...], mask == 1))/input.shape[0]
        return loss

class DistanceLoss(Criterion):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input1: torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        return torch.mean(input1[:,1:]*target)
        
class Dice(Criterion):
    
    def __init__(self, smooth : float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth
    
    def flatten(self, tensor : torch.Tensor) -> torch.Tensor:
        return tensor.permute((1, 0) + tuple(range(2, tensor.dim()))).contiguous().view(tensor.size(1), -1)

    def dice_per_channel(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = self.flatten(input)
        target = self.flatten(target)
        return (2.*(input * target).sum() + self.smooth)/(input.sum() + target.sum() + self.smooth)

    def forward(self, input1: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = F.one_hot(target.type(torch.int64), num_classes=input1.shape[1]).permute(0, len(target.shape), *[i+1 for i in range(len(target.shape)-1)]).float()
        return 1-torch.mean(self.dice_per_channel(input1, target))

class GradientImages(Criterion):

    def __init__(self):
        super().__init__()
    
    @staticmethod
    def _image_gradient2D(image : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dx = image[:, :, 1:, :] - image[:, :, :-1, :]
        dy = image[:, :, :, 1:] - image[:, :, :, :-1]
        return dx, dy

    @staticmethod
    def _image_gradient3D(image : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dx = image[:, :, 1:, :, :] - image[:, :, :-1, :, :]
        dy = image[:, :, :, 1:, :] - image[:, :, :, :-1, :]
        dz = image[:, :, :, :, 1:] - image[:, :, :, :, :-1]
        return dx, dy, dz
        
    def forward(self, input: torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        if len(input.shape) == 5:
            dx, dy, dz = GradientImages._image_gradient3D(input)
            if target is not None:
                dx_tmp, dy_tmp, dz_tmp = GradientImages._image_gradient3D(target)
            dx -= dx_tmp
            dy -= dy_tmp
            dz -= dz_tmp
            return dx.norm() + dy.norm() + dz.norm()
        else:
            dx, dy = GradientImages._image_gradient2D(input)
            if target is not None:
                dx_tmp, dy_tmp = GradientImages._image_gradient2D(target)
                dx -= dx_tmp
                dy -= dy_tmp
            return dx.norm() + dy.norm()
            
"""class GradientImages(Criterion):

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()

    def forward(self, input: torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        result = torch.zeros((input.shape[0])).to(self.device)
        for batch in range(input.shape[0]):
            X = input[batch]
            Y = target[batch]
            if len(X.shape) == 4:
                input_gradient = Gradient._image_gradient3D(X)
                target_gradient = Gradient._image_gradient3D(Y)
            else:
                input_gradient = Gradient._image_gradient2D(X)
                target_gradient = Gradient._image_gradient2D(Y)
            
            result[batch] = self.loss(input_gradient, target_gradient)
        return result.sum()"""
        
class BCE(Criterion):

    def __init__(self, target : float = 0) -> None:
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.register_buffer('target', torch.tensor(target).type(torch.float32))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        target = self._buffers["target"]
        return self.loss(input, target.to(self.device).expand_as(input))

class WGP(Criterion):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, gradient_norm: torch.Tensor, _ : torch.Tensor) -> torch.Tensor:
        return torch.mean((gradient_norm - 1)**2)


"""class Gram(Criterion):
    
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, input : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        return torch.nn.L1Loss(reduction='sum')(torch.mm(input, input.t()).div(np.prod(input.shape)), torch.mm(target, target.t()).div(np.prod(target.shape)))
"""

def computeGram(input : torch.Tensor):
    (b, ch, w) = input.size()
    features = input
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t).div(ch*w)
    return gram

class Gram(Criterion):
    
    def __init__(self) -> None:
        super().__init__()
        self.loss = torch.nn.L1Loss(reduction='mean')

    def forward(self, input : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        return self.loss(computeGram(input), computeGram(target))

class MedPerceptualLoss(Criterion):
    
    def __init__(self, modelLoader : ModelLoader = ModelLoader(), path_model : str = "name", module_names : list[str] = ["ConvNextEncoder.ConvNexStage_2.BottleNeckBlock_0.Linear_2", "ConvNextEncoder.ConvNexStage_3.BottleNeckBlock_0.Linear_2"], shape: list[int] = [128, 256, 256], losses: list[str] = ["Gram", "torch_nn_L1Loss"]) -> None:
        super().__init__()
        
        DEEP_LEARNING_API_CONFIG_PATH = ".".join(os.environ['DEEP_LEARNING_API_CONFIG_PATH'].split(".")[:-1])
        self.path_model = path_model
        if self.path_model not in modelsRegister:
            self.model = modelLoader.getModel(train=False, DL_args=os.environ['DEEP_LEARNING_API_CONFIG_PATH'], DL_without=["optimizer", "schedulers", "nb_batch_per_step", "init_type", "init_gain", "outputsCriterions", "drop_p"])
            if path_model.startswith("https"):
                state_dict = torch.hub.load_state_dict_from_url(path_model)
                state_dict = {"Model": {self.model.getName() : state_dict["model"]}}
            else:
                state_dict = torch.load(path_model)
            self.model.load(state_dict)
            modelsRegister[self.path_model] = self.model
        else:
            self.model = modelsRegister[self.path_model]

        self.module_names = list(set(module_names))
        self.shape = shape
        self.mode = "trilinear" if  len(shape) == 3 else "bilinear"
        self.losses = []
        for loss in losses:
            module, name = _getModule(loss, "measure")
            self.losses.append(config(DEEP_LEARNING_API_CONFIG_PATH)(getattr(importlib.import_module(module), name))(config = None))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def setDevice(self, device: torch.device):
        super().setDevice(device)
        self.model.setDevice(self.device)
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.mean = self.mean.to(self.device)
        self.std = self.std.to(self.device)

    def preprocessing(self, input: torch.Tensor) -> torch.Tensor:
        input = input.repeat(1, 3, *[1 for _ in range(len(input.shape)-2)])
        input = (input-torch.min(input))/(torch.max(input)-torch.min(input))
        input = (input-self.mean)/self.std
        return input

        #if not all([input.shape[-i-1] == size for i, size in enumerate(reversed(self.shape[2:]))]):
        #    input = F.interpolate(input, mode=self.mode, size=tuple(self.shape), align_corners=False).type(torch.float32)
        return input

    def forward(self, input : torch.Tensor, *targets_tmp : torch.Tensor) -> torch.Tensor:
        loss = torch.zeros((1), requires_grad = True).to(self.device, non_blocking=False).type(torch.float32)
        input = self.preprocessing(input)
        targets = [self.preprocessing(target) for target in targets_tmp]
        
        for zipped_input in zip([input], *[[target] for target in targets]):
            input = zipped_input[0]
            targets = zipped_input[1:]
            for zipped_layers in list(zip(self.model.get_layers([input], self.module_names.copy()), *[self.model.get_layers([target], self.module_names.copy()) for target in targets])):
                input_layer = zipped_layers[0][1].view(zipped_layers[0][1].shape[0], zipped_layers[0][1].shape[1], int(np.prod(zipped_layers[0][1].shape[2:])))
                for i, target_layer in enumerate(zipped_layers[1:]):
                    target_layer = target_layer[1].view(target_layer[1].shape[0], target_layer[1].shape[1], int(np.prod(target_layer[1].shape[2:])))
                    loss += self.losses[i](input_layer.float(), target_layer.float())/input_layer.shape[0]
        return loss

class KLDivergence(Criterion):
    
    def __init__(self, dim : int = 100, mu : float = 0, std : float = 1) -> None:
        super().__init__()
        self.latentDim = dim
        self.mu = torch.Tensor([mu])
        self.std = torch.Tensor([std])
        self.modelDim = 3
        
    def init(self, model : Network, output_group : str, target_group : str) -> str:
        super().init(model, output_group, target_group)
        model._compute_channels_trace(model, model.in_channels, None)
        self.modelDim = model.dim
        last_module = model
        for name in output_group.split(".")[:-1]:
            last_module = last_module[name]
        modules = last_module._modules.copy()
        last_module._modules.clear()

        for name, value in modules.items():
            last_module._modules[name] = value
            if name == output_group.split(".")[-1]:
                last_module.add_module("LatentDistribution", LatentDistribution(out_channels=last_module._modulesArgs[name].out_channels, out_is_channel=last_module._modulesArgs[name].out_is_channel , latentDim=self.latentDim, modelDim=self.modelDim, out_branch=last_module._modulesArgs[name].out_branch))
        return ".".join(output_group.split(".")[:-1])+".LatentDistribution.Concat"

    def setDevice(self, device: torch.device):
        super().setDevice(device)
        self.mu = self.mu.to(self.device)
        self.std = self.std.to(self.device)
    
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        mu = input[:, 0, :]
        std = torch.exp(input[:, 1, :]/2)
        z = input[:, 2, :]
        q = torch.distributions.Normal(mu, std)
        
        target_mu = torch.ones((self.latentDim)).to(self.device)*self.mu
        
        p = torch.distributions.Normal(target_mu, torch.ones((self.latentDim)).to(self.device)*self.std)
        log_pz = p.log_prob(z)

        log_qzx = q.log_prob(z)
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl.mean()

class Accuracy(Criterion):

    def __init__(self) -> None:
        super().__init__()
        self.n : int = 0
        self.corrects = torch.zeros((1))

    def forward(self, input : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        self.n += input.shape[0]
        self.corrects += (torch.argmax(torch.softmax(input, dim=1), dim=1) == target).sum().float().cpu()
        return self.corrects/self.n

class Contrastive(Criterion):

    def __init__(self, alpha: float = 1) -> None:
        super().__init__()
        self.loss = torch.nn.MSELoss(reduction="mean")
        self.alpha = alpha

    def J_ij(self, D: Callable[[int, int], torch.Tensor], i: int, j: int, negative_index: list[int]) -> torch.Tensor:
        loss = torch.tensor(0, dtype=torch.float, device=self.device)
        for k in negative_index:
            loss += torch.exp(self.alpha-D(i, k))
            loss += torch.exp(self.alpha-D(j, k))
        return torch.log(loss)+D(i,j)
        
    def forward(self, input : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        input = computeGram(input.view(input.shape[0], input.shape[1], int(np.prod(input.shape[2:]))))
        D = lambda i, j: self.loss(input[i], input[j])

        combinations = list(itertools.combinations(range(input.shape[0]), r=2))
        positive_pair = [(index_0, index_1) for index_0 ,index_1 in combinations if (target[index_0].item() == target[index_1].item())]
        loss = torch.tensor(0, dtype=torch.float, device=self.device)

        for i ,j in positive_pair:
            negative_index = [a for a in range(target.shape[0]) if target[a] != target[i]]
            loss += torch.pow(torch.relu(self.J_ij(D, i, j, negative_index)), 2)
        return loss*1/(2*len(positive_pair))