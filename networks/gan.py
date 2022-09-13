import copy
from typing import Dict
import torch
from DeepLearning_API.config import config
from DeepLearning_API.networks import network, blocks
from DeepLearning_API.HDF5 import ModelPatch
from networks.segmentation import UNet

class Discriminator(network.Network):

    class DiscriminatorHead(network.ModuleArgsDict):

        def __init__(self, channels: int, dim: int) -> None:
            super().__init__()
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels=channels, out_channels=1, kernel_size=4, stride=1, padding=1))
            self.add_module("AdaptiveAvgPool", blocks.getTorchModule("AdaptiveAvgPool", dim)(tuple([1]*dim)))
            self.add_module("Flatten", torch.nn.Flatten(1))
            
    @config("Discriminator")
    def __init__(self,
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.SchedulersLoader = network.SchedulersLoader(),
                    outputsCriterions: Dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    dim : int = 3,
                    in_channels : int = 1) -> None:
        super().__init__(in_channels = in_channels, optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, dim=dim)
        
        ndf = 64
        n_layers=3
        norm_layer = blocks.getTorchModule("BatchNorm", dim)
        kernel_size = 4
        padding = 1
        bias = False
        
        self.add_module("Conv_0", blocks.getTorchModule("Conv", dim)(in_channels=in_channels, out_channels=ndf, kernel_size=kernel_size, stride=2, padding=padding))
        self.add_module("LeakyReLU_0", torch.nn.LeakyReLU(0.2, True))
        
        channels = [ndf*min(2 ** i, 8) for i in range(n_layers+1)]
        for i, (in_channels, out_channels, stride) in enumerate(zip(channels, channels[1:], [2]*(len(channels)-2)+[1])):
            self.add_module("Conv_{}".format(i+1), blocks.getTorchModule("Conv", dim)(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
            self.add_module("Norm_{}".format(i+i), norm_layer(out_channels))
            self.add_module("LeakyReLU_{}".format(i+i), torch.nn.LeakyReLU(0.2, True))
        
        self.add_module("Head", Discriminator.DiscriminatorHead(channels=channels[-1], dim=dim))
        
class Generator(network.Network):

    class GeneratorStem(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, norm_layer : torch.nn.Module, bias: bool, dim: int) -> None:
            super().__init__()
            self.add_module("Reflection", blocks.getTorchModule("ReflectionPad", dim)(3))
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels, out_channels, kernel_size=7, padding=0, bias=bias))
            self.add_module("Norm", norm_layer(out_channels))
            self.add_module("Relu", torch.nn.ReLU(True)) 

    class GeneratorHead(network.ModuleArgsDict):

        def __init__(self, channels: int, dim: int) -> None:
            super().__init__()
            self.add_module("Reflection", blocks.getTorchModule("ReflectionPad", dim)(3))
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(channels, 1, kernel_size=7, padding=0))
            self.add_module("Tanh", torch.nn.Tanh())

    class GeneratorDownSample(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, norm_layer: torch.nn.Module, bias: bool, dim: int) -> None:
            super().__init__()
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=bias))
            self.add_module("Norm", norm_layer(out_channels))
            self.add_module("Relu", torch.nn.ReLU(True))

    class GeneratorResnetBlock(network.ModuleArgsDict):

        def __init__(self, channels : int, padding_type : str, norm_layer : torch.nn.Module, dropout : bool, bias : bool, dim : int):
            super().__init__()
            padding = 0 if padding_type != 'zero' else 1
            if padding == 0:
                self.add_module("Padding_0", blocks.getTorchModule("{}Pad".format(padding_type), dim)(1))

            self.add_module("Conv_0", blocks.getTorchModule("Conv", dim)(channels, channels, kernel_size=3, padding=padding, bias=bias))
            self.add_module("Norm_0", norm_layer(channels))
            self.add_module("Relu_0", torch.nn.ReLU(True))
            if dropout:
                self.add_module("Dropout", torch.nn.Dropout(0.5, inplace=True))
            
            if padding == 0:
                self.add_module("Padding_1", blocks.getTorchModule("{}Pad".format(padding_type), dim)(1))

            self.add_module("Conv_1", blocks.getTorchModule("Conv", dim)(channels, channels, kernel_size=3, padding=padding, bias=bias))
            self.add_module("Norm_1", norm_layer(channels))
            self.add_module("Residual", blocks.Add(), in_branch=[0,1])


    class GeneratorUpSample(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, norm_layer: torch.nn.Module, bias: bool, dim: int) -> None:
            super().__init__()
            self.add_module("ConvTranspose", blocks.getTorchModule("ConvTranspose", dim)(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias))
            self.add_module("Norm", norm_layer(out_channels))
            self.add_module("Relu", torch.nn.ReLU(True))

    @config("Generator")
    def __init__(self, 
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.SchedulersLoader = network.SchedulersLoader(),
                    patch : ModelPatch = ModelPatch(),
                    outputsCriterions: Dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    in_channels : int = 1,
                    dim : int = 3) -> None:
        #super().__init__(optimizer=optimizer, schedulers=schedulers, outputsCriterions=outputsCriterions, dim=dim, channels=[1,64,128], nb_conv_per_stage=2)
        #self.add_module("Tanh", torch.nn.Tanh())
        super().__init__(optimizer=optimizer, in_channels=in_channels, schedulers=schedulers, patch=patch, outputsCriterions=outputsCriterions, dim=dim)
        ngf=64
        norm_layer = blocks.getTorchModule("BatchNorm", dim)
        dropout = False
        n_blocks=6
        padding_type="Reflection"
        bias = False
        channels = [ngf*2**i for i in range(3)]
        
        self.add_module("Stem", Generator.GeneratorStem(in_channels=in_channels, out_channels=ngf, norm_layer=norm_layer, bias=bias, dim=dim))

        for i, (in_channels, out_channels) in enumerate(zip(channels, channels[1:])):
            self.add_module("DownSample_{}".format(i), Generator.GeneratorDownSample(in_channels=in_channels, out_channels=out_channels, norm_layer=norm_layer, bias=bias, dim=dim))

        for i in range(n_blocks):
            self.add_module("ResnetBlock_{}".format(i), Generator.GeneratorResnetBlock(channels=channels[-1], padding_type=padding_type, norm_layer=norm_layer, dropout=dropout, bias=bias, dim=dim))

        for i, (in_channels, out_channels) in enumerate(zip(reversed(channels), reversed(channels[:-1]))):
            self.add_module("UpSample_{}".format(i), Generator.GeneratorUpSample(in_channels=in_channels, out_channels=out_channels, norm_layer=norm_layer, bias=bias, dim=dim))

        self.add_module("Head", Generator.GeneratorHead(channels=ngf, dim=dim))

class Gan(network.Network):

    @config("Gan")
    def __init__(self, generator : Generator = Generator(), discriminator : Discriminator = Discriminator()) -> None:
        super().__init__()
        self.add_module("Discriminator_B", discriminator, in_branch=[1], out_branch=[-1], requires_grad=True)
        self.add_module("Generator_A_to_B", generator, in_branch=[0], out_branch=["pB"])
        
        self.add_module("detach", blocks.Detach(), in_branch=["pB"], out_branch=["pB_detach"])
        self.add_module("Discriminator_pB_detach", discriminator, in_branch=["pB_detach"], out_branch=[-1])
          
        self.add_module("Discriminator_pB", discriminator, in_branch=["pB"], out_branch=[-1], requires_grad=False)
        

class CycleGan(network.Network):

    @config("CycleGan")
    def __init__(self, generator : Generator = Generator(), discriminator : Discriminator = Discriminator()) -> None:
        super().__init__()
        self.add_module("Discriminator_A", copy.deepcopy(discriminator), in_branch=[1], out_branch=[-1], requires_grad=True)
        self.add_module("Discriminator_B", copy.deepcopy(discriminator), in_branch=[0], out_branch=[-1], requires_grad=True)

        self.add_module("Generator_A_to_B", copy.deepcopy(generator), in_branch=[0], out_branch=["pB"])
        self.add_module("Generator_B_to_A", copy.deepcopy(generator), in_branch=[1], out_branch=["pA"])
        
        self.add_module("detach", blocks.Detach(), in_branch=["pB"], out_branch=["pB_detach"])
        self.add_module("Discriminator_pB_detach", copy.deepcopy(discriminator), in_branch=["pA"], out_branch=[-1])
        
        self.add_module("detach", blocks.Detach(), in_branch=["pA"], out_branch=["pB_detach"])
        self.add_module("Discriminator_pA_detach", copy.deepcopy(discriminator), in_branch=["pB"], out_branch=[-1])

        self.add_module("Generator_pA_to_B", self["Generator_A_to_B"], in_branch=["pA"], out_branch=["cyc_A"])
        self.add_module("Generator_pB_to_A", self["Generator_B_to_A"], in_branch=["pB"], out_branch=["cyc_B"])
        
        self.add_module("Discriminator_pA", discriminator, in_branch=["pA"], out_branch=[-1], requires_grad=False)
        self.add_module("Discriminator_pB", discriminator, in_branch=["pB"], out_branch=[-1], requires_grad=False)