from abc import ABC
import importlib
import numpy as np
import torch

import torch.nn.functional as F
import os

from DeepLearning_API.config import config
from DeepLearning_API.utils import _getModule
from DeepLearning_API.networks.blocks import LatentDistribution
from DeepLearning_API.networks.network import ModelLoader, Network
from typing import Callable, Union
from functools import partial
import torch.nn.functional as F
import itertools
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

modelsRegister = {}

class Criterion(torch.nn.Module, ABC):

    def __init__(self) -> None:
        super().__init__()

    def init(self, model : torch.nn.Module, output_group : str, target_group : str) -> str:
        return output_group

class MaskedLoss(Criterion):

    def __init__(self, loss : Callable[[torch.Tensor, torch.Tensor], torch.Tensor], mode_image_masked: bool) -> None:
        super().__init__()
        self.loss = loss
        self.mode_image_masked = mode_image_masked

    def forward(self, input: torch.Tensor, *target : list[torch.Tensor]) -> torch.Tensor:
        loss = torch.tensor(0, dtype=torch.float32).to(input.device)
        for batch in range(input.shape[0]):
            if len(target) > 1:
                if self.mode_image_masked:
                    loss += self.loss(input[batch, ...]*target[1][batch, ...], target[0][batch, ...]*target[1][batch, ...])
                else:
                    if torch.count_nonzero(target[1][batch, ...]) > 0:
                        loss += self.loss(torch.masked_select(input[batch, ...], target[1][batch, ...] == 1), torch.masked_select(target[0][batch, ...], target[1][batch, ...] == 1))
            else:
                loss += self.loss(input[batch, ...], target[0][batch, ...])
        return loss/input.shape[0]
    
class MSE(MaskedLoss):

    def _loss(reduction: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.MSELoss(reduction=reduction)(x, y)

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(partial(MSE._loss, reduction), False)

class MAE(MaskedLoss):

    def _loss(reduction: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.L1Loss(reduction=reduction)(x, y)
    
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(partial(MAE._loss, reduction), False)

class PSNR(MaskedLoss):

    def _loss(dynamic_range: Union[float, None], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return peak_signal_noise_ratio(x[0].detach().cpu().numpy(), y[0].cpu().numpy(), data_range=dynamic_range if dynamic_range else (y.max()-y.min()).cpu().numpy())
    
    def __init__(self, dynamic_range: Union[float, None] = None) -> None:
        super().__init__(partial(PSNR._loss, dynamic_range), False)
    
class SSIM(MaskedLoss):
    
    def _loss(dynamic_range: Union[float, None], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return structural_similarity(x[0].detach().cpu().numpy(), y[0].cpu().numpy(), data_range=dynamic_range if dynamic_range else (y.max()-y.min()).cpu().numpy())
    
    def __init__(self, dynamic_range: Union[float, None] = None) -> None:
        super().__init__(partial(SSIM._loss, dynamic_range), True)

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
        
class BCE(Criterion):

    def __init__(self, target : float = 0) -> None:
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.register_buffer('target', torch.tensor(target).type(torch.float32))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        target = self._buffers["target"]
        return self.loss(input, target.to(input.device).expand_as(input))

class WGP(Criterion):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, gradient_norm: torch.Tensor, _ : torch.Tensor) -> torch.Tensor:
        return torch.mean((gradient_norm - 1)**2)

class Gram(Criterion):

    def computeGram(input : torch.Tensor):
        (b, ch, w) = input.size()
        features = input
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t).div(ch*w)
        return gram

    def __init__(self) -> None:
        super().__init__()
        self.loss = torch.nn.L1Loss(reduction='sum')

    def forward(self, input : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        return self.loss(Gram.computeGram(input), Gram.computeGram(target))

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
        self.model.eval()
        self.model.requires_grad_(False)
        self.initdevice = False

    def preprocessing(self, input: torch.Tensor) -> torch.Tensor:
        input = input.repeat(1, 3, *[1 for _ in range(len(input.shape)-2)])
        input = (input-torch.min(input))/(torch.max(input)-torch.min(input))
        input = (input-self.mean.to(input.device))/self.std.to(input.device)
        if not all([input.shape[-i-1] == size for i, size in enumerate(reversed(self.shape[2:]))]):
            input = F.interpolate(input, mode=self.mode, size=tuple(self.shape), align_corners=False).type(torch.float32)
        return input
    
    def _compute(self, input: torch.Tensor, targets: list[torch.Tensor]) -> torch.Tensor:
        loss = torch.zeros((1), requires_grad = True).to(input.device, non_blocking=False).type(torch.float32)
        input = self.preprocessing(input)
        targets = [self.preprocessing(target) for target in targets]
        
        for zipped_input in zip([input], *[[target] for target in targets]):
            input = zipped_input[0]
            targets = zipped_input[1:]
            for zipped_layers in list(zip(self.model.get_layers([input], self.module_names.copy()), *[self.model.get_layers([target], self.module_names.copy()) for target in targets])):
                input_layer = zipped_layers[0][1].view(zipped_layers[0][1].shape[0], zipped_layers[0][1].shape[1], int(np.prod(zipped_layers[0][1].shape[2:])))
                for i, target_layer in enumerate(zipped_layers[1:]):
                    target_layer = target_layer[1].view(target_layer[1].shape[0], target_layer[1].shape[1], int(np.prod(target_layer[1].shape[2:])))
                    loss += self.losses[i](input_layer.float(), target_layer.float())/input_layer.shape[0]
        return loss
    
    def forward(self, input : torch.Tensor, *targets : torch.Tensor) -> torch.Tensor:
        if not self.initdevice:
            self.model = Network.to(self.model, input.device.index)
            self.initdevice = True
        loss = torch.zeros((1), requires_grad = True).to(input.device, non_blocking=False).type(torch.float32)
        if len(input.shape) == 5:
            for i in range(input.shape[2]):
                loss += self._compute(input[:,:,i, ...], [t[:,:,i,...] for t in targets])/input.shape[2]
        else:
            loss = self._compute(input, targets)
        return loss

class KLDivergence(Criterion):
    
    def __init__(self, shape: list[int], dim : int = 100, mu : float = 0, std : float = 1) -> None:
        super().__init__()
        self.latentDim = dim
        self.mu = torch.Tensor([mu])
        self.std = torch.Tensor([std])
        self.modelDim = 3
        self.shape = shape
        
    def init(self, model : Network, output_group : str, target_group : str) -> str:
        super().init(model, output_group, target_group)
        model._compute_channels_trace(model, model.in_channels, None, None)
        self.modelDim = model.dim
        last_module = model
        for name in output_group.split(".")[:-1]:
            last_module = last_module[name]

        modules = last_module._modules.copy()
        modulesArgs = last_module._modulesArgs.copy()
        last_module._modules.clear()
        
        for name, value in modules.items():
            last_module._modules[name] = value
            if name == output_group.split(".")[-1]:
                last_module.add_module("LatentDistribution", LatentDistribution(in_channels=modulesArgs[name].out_channels, shape = self.shape, out_is_channel=modulesArgs[name].out_is_channel , latentDim=self.latentDim, modelDim=self.modelDim, out_branch=modulesArgs[name].out_branch))
        return ".".join(output_group.split(".")[:-1])+".LatentDistribution.Concat"
    
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        mu = input[:, 0, :]
        log_std = input[:, 1, :]
        """z = input[:, 2, :]

        q = torch.distributions.Normal(mu, std)
        
        target_mu = torch.ones((self.latentDim)).to(input.device)*self.mu.to(input.device)
        target_std = torch.ones((self.latentDim)).to(input.device)*self.std.to(input.device)

        p = torch.distributions.Normal(target_mu, target_std)
        log_pz = p.log_prob(z)

        log_qzx = q.log_prob(z)

        

        kl = (log_pz - log_qzx)
        kl = kl.sum(-1)"""
        return torch.mean(-0.5 * torch.sum(1 + log_std - mu**2 - torch.exp(log_std), dim= 0))

class Accuracy(Criterion):

    def __init__(self) -> None:
        super().__init__()
        self.n : int = 0
        self.corrects = torch.zeros((1))

    def forward(self, input : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        self.n += input.shape[0]
        self.corrects += (torch.argmax(torch.softmax(input, dim=1), dim=1) == target).sum().float().cpu()
        return self.corrects/self.n

