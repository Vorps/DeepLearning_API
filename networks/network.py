from functools import partial
import importlib
from typing import Dict, List, Tuple
from typing_extensions import Self
import torch
from abc import ABC, abstractmethod
import numpy as np

from DeepLearning_API import config, _getModule, Loss

def set_requires_grad(nets : List[torch.nn.Module], requires_grad = False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad

class Criterion():

    @config(None)
    def __init__(self, group : str = "Default", l : float = 1) -> None:
        self.l = l
        self.group = group

    def getCriterion(self, classpath : str, group : str, key : str) -> torch.nn.Module:
        module, name = _getModule(classpath, "criterion")
        return config("Trainer.Model.{}.criterions.{}.criterion.{}".format(key, group,classpath))(getattr(importlib.import_module(module), name))(config = None)

class Criterions():

    @config(None)
    def __init__(self, criterion : Dict[str, Criterion] = {"default:torch_nn_CrossEntropyLoss:Dice:NCC" : Criterion()}) -> None:
        self.criterions = criterion

    def getCriterions(self, group : str, key : str) -> Dict[str, Tuple[float, torch.nn.Module]]:
        for classpath in self.criterions:
            self.criterions[classpath] = (self.criterions[classpath].group, self.criterions[classpath].l, self.criterions[classpath].getCriterion(classpath, group, key))
        return self.criterions

class Optimizer():
    
    @config("Optimizer")
    def __init__(self, name : str = "AdamW") -> None:
        self.name = name
    
    def getOptimizer(self, model : torch.nn.Module, key : str) -> torch.nn.Module:
        return config("Trainer.Model.{}.Optimizer".format(key))(getattr(importlib.import_module('torch.optim'), self.name))(model.parameters(), config = None)

class Scheduler():

    @config("Scheduler")
    def __init__(self, name : str = "ReduceLROnPlateau") -> None:
        self.name = name

    def getSheduler(self, optimizer : torch.nn.Module, key : str) -> torch.nn.Module:
        return config("Trainer.Model.{}.Scheduler".format(key))(getattr(importlib.import_module('torch.optim.lr_scheduler'), self.name))(optimizer, config = None)


class Network(torch.nn.Module, ABC):

    def __init__(self, optimizer : Optimizer = None, scheduler : Scheduler = None, criterions: Dict[str, Criterions] = None, init_type : str = "normal", init_gain : float = 0.02, dim : int = 3) -> None:
        super().__init__()
        self.dim = dim
        self.init_type  = init_type
        self.init_gain  = init_gain
        self.criterions = criterions
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        
    def init(self, key : str, loss : Loss) -> None:
        def init_func(m):
            classname = m.__class__.__name__
            
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if self.init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, self.init_gain)
                elif self.init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=self.init_gain)
                elif self.init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif self.init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=self.init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % self.init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm') != -1:
                torch.nn.init.normal_(m.weight.data, 1.0, std = self.init_gain)
                torch.nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        if self.optimizer is not None:
            self.optimizer      = self.optimizer.getOptimizer(self, key)
        if self.scheduler is not None:
            self.scheduler      = self.scheduler.getSheduler(self.optimizer, key)
        if self.criterions is not None:
            for classpath in self.criterions:
                self.criterions[classpath] = self.criterions[classpath].getCriterions(classpath, key)
                
        self.loss : Loss = loss(self.criterions)

    def getSubModels(self) -> List[Self]:
        return [self]
        
    def logImage(self, input : torch.Tensor, output : Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        return None

    def _logImageNormalize(self, input : torch.Tensor):
        b = -np.min(input)
        a = 1/(np.max(input)+b)
        return a*(input+b)

    def getName(self):
        return self.__class__.__name__

    def backward(self, input : torch.tensor):
        out = self.forward(input)
        self.loss.update(out[self.getName()] if len(self.getSubModels()) > 1 else out, input)

def getTorchModule(name_fonction : str, dim : int = None) -> torch.nn.Module:
    return getattr(importlib.import_module("torch.nn"), "{}".format(name_fonction) + ("{}d".format(dim) if dim is not None else ""))


class BlockConfig():

    @config("BlockConfig")
    def __init__(self, nb_conv_per_level : int = 1, kernel_size : int = 3, stride : int = 1, padding : int = 1, activation : str = "ReLU", batchNorm : bool = False) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.nb_conv_per_level = nb_conv_per_level
        self.activation = activation
        self.batchNorm = batchNorm

    def getConv(self, in_channels : int, out_channels : int, dim : int) -> torch.nn.Conv3d:
        return getTorchModule("Conv", dim = dim)(in_channels = in_channels, out_channels = out_channels, kernel_size = self.kernel_size, stride = self.stride, padding = self.padding)
    
    def getActivation(self) -> torch.nn.Module:
        return getTorchModule(self.activation)()
        
class ConvBlock(torch.nn.Module):
    
    def __init__(self, in_channels : int, out_channels : int, blockConfig : BlockConfig, dim : int) -> None:
        super().__init__()
        args = []
        for _ in range(blockConfig.nb_conv_per_level):
            args.append(blockConfig.getConv(in_channels, out_channels, dim))
            if blockConfig.batchNorm:
                args.append(getTorchModule("BatchNorm", dim = dim)(out_channels))
            args.append(blockConfig.getActivation())
            in_channels = out_channels
        self.block = torch.nn.Sequential(*args)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.block(x)

class ResBlock(ConvBlock):

    def __init__(self, in_channels : int, out_channels : int, blockConfig : BlockConfig, dim : int) -> None:
        super().__init__(in_channels, out_channels, blockConfig, dim)
        self.activation = self.block[-1]
        del self.block[-1]

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(x) + x)


class AttentionBlock(torch.nn.Module):

    def __init__(self, F_g : int, F_l : int, F_int : int, dim : int):
        super().__init__()
        self.W_g = getTorchModule("Conv", dim = dim)(in_channels = F_g, out_channels = F_int, kernel_size=1, stride=1, padding=0)
        self.W_x = getTorchModule("Conv", dim = dim)(in_channels = F_l, out_channels = F_int, kernel_size=1, stride=2, padding=0)
        self.relu = torch.nn.ReLU(inplace=True)
        self.psi = torch.nn.Sequential(getTorchModule("Conv", dim = dim)(in_channels = F_int, out_channels = 1, kernel_size=1,stride=1, padding=0), torch.nn.Sigmoid(), torch.nn.Upsample(scale_factor=2))
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi

