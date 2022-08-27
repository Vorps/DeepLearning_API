from abc import ABC
import importlib
from typing import List, Tuple
import torch

import torch.nn.functional as F
import os

from DeepLearning_API.config import config
from DeepLearning_API.utils import NeedDevice, _getModule
from DeepLearning_API.networks.blocks import LatentDistribution
from DeepLearning_API.networks.network import Network

class Criterion(NeedDevice, torch.nn.Module, ABC):

    def __init__(self) -> None:
        super().__init__()

    def init(self, model : torch.nn.Module, output_group : str, target_group : str) -> str:
        return output_group

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
    
    def __init__(self, modelLoader : ModelLoader = ModelLoader(), path_model : str = "name", module_names : List[str] = ["ConvNextEncoder.ConvNexStage_2.BottleNeckBlock_0.Linear_2", "ConvNextEncoder.ConvNexStage_3.BottleNeckBlock_0.Linear_2"], shape: List[int] = [128, 256, 256]) -> None:
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

class KLDivergence(Criterion):
    
    def __init__(self, dim : int = 100, mu : float = 0, std : float = 1) -> None:
        super().__init__()
        
        self.latentDim = dim
        self.mu = torch.Tensor([mu])
        self.std = torch.Tensor([std]) 
        self.modelDim = 3
        
    def init(self, model : Network, output_group : str, target_group : str) -> str:
        super().init(model, output_group, target_group)
        model._compute_channels_trace(model, model.in_channels)
        self.modelDim = model.dim
        last_module = model
        for name in output_group.split(".")[:-1]:
            last_module = last_module[name]
        modules = last_module._modules.copy()
        last_module._modules.clear()

        for name, value in modules.items():
            last_module._modules[name] = value
            if name == output_group.split(".")[-1]:
                last_module.add_module("LatentDistribution", LatentDistribution(out_channels=last_module._modulesArgs[name].out_channels, out_is_channel=last_module._modulesArgs[name].out_is_channel , latentDim=self.latentDim, modelDim=self.modelDim, out_branch=last_module._modulesArgs[name].out_branch), out_branch = [-1])
        return ".".join(output_group.split(".")[:-1])+".LatentDistribution.Concat"

    def setDevice(self, device: torch.device):
        super().setDevice(device)
        self.mu = self.mu.to(self.device)
        self.std = self.std.to(self.device)
    
    def forward(self, input : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        mu = input[:, 0, :]
        std = torch.exp(input[:, 1, :]/2)

        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        target_mu = torch.ones((self.latentDim)).to(self.device)*self.mu
        if target is not None:
            log_pz_list = []
            for i in range(target.shape[0]):
                p = torch.distributions.Normal(target_mu*target[i], torch.ones((self.latentDim)).to(self.device)*self.std)
                log_pz_list.append(torch.unsqueeze(p.log_prob(z[i]), dim=0))
            log_pz = torch.cat(log_pz_list, dim=0)
        else:
            p = torch.distributions.Normal(target_mu, torch.ones((self.latentDim)).to(self.device)*self.std)
            log_pz = p.log_prob(z)


        q = torch.distributions.Normal(mu, std)

        log_qzx = q.log_prob(z)
        

        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

class Accuracy(Criterion):

    def __init__(self) -> None:
        super().__init__()
        self.n : int = 0
        self.corrects = torch.zeros((1))

    def forward(self, input : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        self.n += input.shape[0]
        self.corrects += (torch.argmax(torch.softmax(input, dim=1), dim=1) == target).sum().float()
        return self.corrects/self.n

