import ast
from enum import Enum
import importlib
from typing import List, Optional
import torch
from DeepLearning_API.config import config
from DeepLearning_API.networks import network

class NormMode(Enum):
    NONE = 0,
    BATCH = 1
    INSTANCE = 2
    GROUP = 3
    LAYER = 3

class UpSampleMode(Enum):
    CONV_TRANSPOSE = 0,
    UPSAMPLE_NEAREST = 1
    UPSAMPLE_LINEAR = 2
    UPSAMPLE_BILINEAR = 3
    UPSAMPLE_BICUBIC = 4
    UPSAMPLE_TRILINEAR = 5

class DownSampleMode(Enum):
    MAXPOOL = 0
    AVGPOOL = 1
    CONV_STRIDE = 2

def getTorchModule(name_fonction : str, dim : Optional[int] = None) -> torch.nn.Module:
    return getattr(importlib.import_module("torch.nn"), "{}".format(name_fonction) + ("{}d".format(dim) if dim is not None else ""))

class BlockConfig():

    @config("BlockConfig")
    def __init__(self, kernel_size : int = 3, stride : int = 1, padding : int = 1, bias = True, activation : str = "ReLU", normMode : str = "NONE") -> None:
        self.kernel_size = kernel_size
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.normMode = NormMode._member_map_[normMode]

    def getConv(self, in_channels : int, out_channels : int, dim : int) -> torch.nn.Conv3d:
        return getTorchModule("Conv", dim = dim)(in_channels = in_channels, out_channels = out_channels, kernel_size = self.kernel_size, stride = self.stride, padding = self.padding, bias=self.bias)
    
    def getNorm(self, channels : int, dim: int):
        if self.normMode == NormMode.BATCH:
            return getTorchModule("BatchNorm", dim = dim)(channels)
        if self.normMode == NormMode.INSTANCE:
            return getTorchModule("InstanceNorm", dim = dim)(channels)
        if self.normMode == NormMode.GROUP:
            return torch.nn.GroupNorm(num_groups=1, num_channels=channels)
        if self.normMode == NormMode.LAYER:
            return torch.nn.LayerNorm(channels)
        return torch.nn.Identity()

    def getActivation(self) -> torch.nn.Module:
        return getTorchModule(self.activation)(*[ast.literal_eval(value) for value in self.activation.split(";")[1:]], inplace=True) if self.activation != "None" else torch.nn.Identity()

class ConvBlock(network.ModuleArgsDict):
    
    def __init__(self, in_channels : int, out_channels : int, blockConfig : BlockConfig, dim : int, alias : List[List[str]]=[[], [], []]) -> None:
        super().__init__()
        self.add_module("Conv", blockConfig.getConv(in_channels, out_channels, dim), alias=alias[0])
        self.add_module("Norm", blockConfig.getNorm(out_channels, dim), alias=alias[1])
        self.add_module("Activation", blockConfig.getActivation(), alias=alias[2])

class Attention(network.ModuleArgsDict):

    def __init__(self, F_g : int, F_l : int, F_int : int, dim : int):
        super().__init__()
        self.add_module("W_x", getTorchModule("Conv", dim = dim)(in_channels = F_l, out_channels = F_int, kernel_size=1, stride=2, padding=0), in_branch=[0], out_branch=[0])
        self.add_module("W_g", getTorchModule("Conv", dim = dim)(in_channels = F_g, out_channels = F_int, kernel_size=1, stride=1, padding=0), in_branch=[1], out_branch=[1])
        self.add_module("Add", Add(), in_branch=[0,1])
        self.add_module("ReLU", torch.nn.ReLU(inplace=True))
        self.add_module("Conv", getTorchModule("Conv", dim = dim)(in_channels = F_int, out_channels = 1, kernel_size=1,stride=1, padding=0))
        self.add_module("Sigmoid", torch.nn.Sigmoid())
        self.add_module("Upsample", torch.nn.Upsample(scale_factor=2))
        self.add_module("Multiply", Multiply(), in_branch=[2,0])
    
def downSample(in_channels: int, out_channels: int, downSampleMode: DownSampleMode, dim: int):
    if downSampleMode == DownSampleMode.MAXPOOL:
        return getTorchModule("MaxPool", dim = dim)(2)
    if downSampleMode == DownSampleMode.AVGPOOL:
        return getTorchModule("AvgPool", dim = dim)(2)
    if downSampleMode == DownSampleMode.CONV_STRIDE:
        return getTorchModule("Conv", dim)(in_channels, out_channels, kernel_size=2, stride=2, padding=0)

def upSample(in_channels: int, out_channels: int, upSampleMode: UpSampleMode, dim: int):
    if upSampleMode == UpSampleMode.CONV_TRANSPOSE:
        return getTorchModule("ConvTranspose", dim = dim)(in_channels = in_channels, out_channels = out_channels, kernel_size = 2, stride = 2, padding = 0)
    else:
        return torch.nn.Upsample(scale_factor=2, mode=upSampleMode.name.replace("UPSAMPLE_", "").lower(), align_corners=False)

class Unsqueeze(torch.nn.Module):

    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return torch.unsqueeze(x, self.dim)
    
    def extra_repr(self):
        return "dim={}".format(self.dim)

class Permute(torch.nn.Module):

    def __init__(self, dims : List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return torch.permute(x, self.dims)

    def extra_repr(self):
        return "dims={}".format(self.dims)

class ToChannels(Permute):

    def __init__(self, dim):
        super().__init__([0, dim+1, *[i+1 for i in range(dim)]])
        
class ToFeatures(Permute):

    def __init__(self, dim):
        super().__init__([0, *[i+2 for i in range(dim)], 1])        


class Add(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input : torch.Tensor, output : torch.Tensor) -> torch.Tensor:
        return input+output
        

class Multiply(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input : torch.Tensor, output : torch.Tensor) -> torch.Tensor:
        return input*output

class Concat(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input : torch.Tensor, output : torch.Tensor) -> torch.Tensor:
        return torch.cat([input, output], dim=1)

class LatentDistribution(network.ModuleArgsDict):

    def __init__(self, out_channels: int, out_is_channel : bool, latentDim: int, modelDim: int, out_branch : List[int]) -> None:
        super().__init__()
        if not out_is_channel:
            self.add_module("ToChannels", ToChannels(modelDim))
        
        self.add_module("AdaptiveAvgPool", getTorchModule("AdaptiveAvgPool", modelDim)(1))
        self.add_module("Flatten", torch.nn.Flatten(1))
        self.add_module("mu", torch.nn.Linear(out_channels, latentDim), in_branch = out_branch, out_branch = [1])
        self.add_module("log_std", torch.nn.Linear(out_channels, latentDim), in_branch = out_branch, out_branch = [2])
        self.add_module("Unsqueeze_mu", Unsqueeze(1), in_branch = [1], out_branch = [1])
        self.add_module("Unsqueeze_log_std", Unsqueeze(1), in_branch = [2], out_branch = [2])
        self.add_module("Concat", Concat(), in_branch=[1,2])