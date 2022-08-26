from ast import alias
from typing import Dict, List
import torch
from torchvision.ops import StochasticDepth

from DeepLearning_API.networks import network, blocks
from DeepLearning_API.config import config
from DeepLearning_API.measure import TargetCriterionsLoader
from DeepLearning_API.dataset import Patch

class LayerScaler(torch.nn.Module):
    
    def __init__(self, init_value: float, dimensions: int):
        super().__init__()
        self.gamma = torch.nn.parameter.Parameter(init_value * torch.ones((dimensions)), requires_grad=True)
        
    def forward(self, x):
        if len(x.shape) == 5:
            return self.gamma[None,...,None,None,None] * x
        else:
            return self.gamma[None,...,None,None] * x

class BottleNeckBlock(network.ModuleArgsDict):

    def __init__(self, in_features: int, out_features: int, drop_p: float, dim: int):
        super().__init__()
        expanded_features = out_features * 4
        self.add_module("Conv_0", blocks.getTorchModule("Conv", dim)(in_features, in_features, kernel_size=7, padding=3, bias=False, groups=in_features), alias=["block.0"])
        self.add_module("GroupNorm", torch.nn.GroupNorm(num_groups=1, num_channels=in_features), alias=["block.1"])
        self.add_module("Conv_1", blocks.getTorchModule("Conv", dim)(in_features, expanded_features, kernel_size=1), alias=["block.2"])
        self.add_module("GELU", torch.nn.GELU())
        self.add_module("Conv_2", blocks.getTorchModule("Conv", dim)(expanded_features, out_features, kernel_size=1), alias=["block.4"])
        self.add_module("LayerScaler", LayerScaler(1e-6, out_features), alias=["layer_scaler"])
        self.add_module("StochasticDepth", StochasticDepth(drop_p, mode="batch"))
        self.add_module("Residual", blocks.Concat(), in_branch=[0,1])

class ConvNexStage(network.ModuleArgsDict):
    def __init__(self, in_features: int, out_features: int, depth: int, drop_p: float, dim: int):
        super().__init__()
        self.add_module("GroupNorm", torch.nn.GroupNorm(num_groups=1, num_channels=in_features), alias=["0.0"])
        self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_features, out_features, kernel_size=2, stride=2), alias=["0.1"])        

        for i in range(depth):
            self.add_module("BottleNeckBlock_{}".format(i), BottleNeckBlock(out_features, out_features, drop_p, dim), alias=["{}".format(i+1)])

class ConvNextStem(network.ModuleArgsDict):
    def __init__(self, in_features: int, out_features: int, dim: int):
        super().__init__()
        self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_features, out_features, kernel_size=4, stride=4), alias=["0"])
        self.add_module("BatchNorm", blocks.getTorchModule("BatchNorm", dim)(out_features), alias=["1"])

class ConvNextEncoder(network.ModuleArgsDict):

    def __init__(   self,
                    in_channels: int,
                    stem_features: int,
                    depths: List[int],
                    widths: List[int],
                    drop_p: float,
                    dim : int):
        super().__init__()
        self.add_module("ConvNextStem", ConvNextStem(in_channels, stem_features, dim=dim), alias=["0.0"])
    
        drop_probs = [x.item() for x in torch.linspace(0, drop_p, sum(depths))] 
        
        self.add_module("ConvNexStage_0", ConvNexStage(stem_features, widths[0], depths[0], drop_p=drop_probs[0], dim=dim), alias=["0.1"])
        
        for i, ((in_features, out_features), depth, drop_p) in enumerate(zip(list(zip(widths, widths[1:])), depths[1:], drop_probs[1:])):
            self.add_module("ConvNexStage_{}".format(i+1), ConvNexStage(in_features, out_features, depth, drop_p, dim), alias=["{}".format(i+1)])

class Head(network.ModuleArgsDict):

    def __init__(self, in_features : int, num_classes : int, dim : int) -> None:
        super().__init__()
        self.add_module("AdaptiveAvgPool", blocks.getTorchModule("AdaptiveAvgPool", dim)(tuple([1]*dim)))
        self.add_module("Flatten", torch.nn.Flatten(1))
        self.add_module("LayerNorm", torch.nn.LayerNorm(in_features), alias=["2"])
        self.add_module("Linear", torch.nn.Linear(in_features, num_classes), alias=["3"])
        self.add_module("Unsqueeze", blocks.Unsqueeze(2))

class ConvNeXt(network.Network):

    @config("ConvNeXt")
    def __init__(   self,
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.SchedulersLoader = network.SchedulersLoader(),
                    outputsCriterions: Dict[str, TargetCriterionsLoader] = {"default" : TargetCriterionsLoader()},
                    patch : Patch = Patch(),
                    padding : int = 0,
                    paddingMode : str = "default:constant:reflect:replicate:circular",
                    dim : int = 3,
                    in_channels: int = 1,
                    stem_features: int = 64,
                    depths: List[int] = [3,4,6,4],
                    widths: List[int] = [256, 512, 1024, 2048],
                    drop_p: float = 0.1,
                    num_classes: int = 10):
        super().__init__(optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, dim = dim, patch=patch, init_type = "trunc_normal", init_gain=0.02, padding=padding, paddingMode=paddingMode)
        self.add_module("ConvNextEncoder", ConvNextEncoder(in_channels=in_channels, stem_features=stem_features, depths=depths, widths=widths, drop_p=drop_p, dim=dim), alias=["stages"])
        self.add_module("Head", Head(in_features=widths[-1], num_classes=num_classes, dim=dim), alias=["head"])