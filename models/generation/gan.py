import copy
from functools import partial
from typing import Callable
import torch
from DeepLearning_API.config import config
from DeepLearning_API.networks import network, blocks
from DeepLearning_API.HDF5 import ModelPatch

from typing import Dict, List

class DiscriminatorV1(network.Network):

    class DiscriminatorStem(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, dim: int) -> None:
            super().__init__()
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1))
            self.add_module("LeakyReLU", torch.nn.LeakyReLU(0.2, True))

    class DiscriminatorHead(network.ModuleArgsDict):

        def __init__(self, channels: int, dim: int) -> None:
            super().__init__()
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels=channels, out_channels=1, kernel_size=4, stride=1, padding=1))
            self.add_module("AdaptiveAvgPool", blocks.getTorchModule("AdaptiveAvgPool", dim)(tuple([1]*dim)))
            self.add_module("Flatten", torch.nn.Flatten(1))
    
    class DiscriminatorLayers(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, stride: int, norm_layer: Callable[[int], torch.nn.Module], bias: bool, dim: int) -> None:
            super().__init__()
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=stride, padding=1, bias=bias))
            self.add_module("Norm", norm_layer(out_channels))
            self.add_module("LeakyReLU", torch.nn.LeakyReLU(0.2, True))

    class DiscriminatorNLayers(network.ModuleArgsDict):

        def __init__(self, channels: List[int], norm_layer: Callable[[int], torch.nn.Module], bias: bool, dim: int) -> None:
            super().__init__()
            for i, (in_channels, out_channels, stride) in enumerate(zip(channels, channels[1:], [2]*(len(channels)-2)+[1])):
                self.add_module("Layer_{}".format(i), DiscriminatorV1.DiscriminatorLayers(in_channels, out_channels, stride, norm_layer, bias, dim))

    @config("Discriminator")
    def __init__(self,
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.SchedulersLoader = network.SchedulersLoader(),
                    outputsCriterions: Dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    dim : int = 3,
                    nb_batch_per_step: int = 64,
                    normMode: str = "INSTANCE",
                    in_channels : int = 1) -> None:
        super().__init__(in_channels = in_channels, optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, dim=dim, nb_batch_per_step=nb_batch_per_step)
        
        ndf = 16
        n_layers=3
        norm_layer = partial(blocks.getNorm, blocks.NormMode._member_map_[normMode], dim=dim)
        bias = False
        channels = [ndf*min(2 ** i, 8) for i in range(n_layers+1)]
        self.add_module("Stem", DiscriminatorV1.DiscriminatorStem(in_channels, ndf, dim))
        self.add_module("Layers", DiscriminatorV1.DiscriminatorNLayers(channels, norm_layer, bias, dim))
        self.add_module("Head", DiscriminatorV1.DiscriminatorHead(channels=channels[-1], dim=dim))
    
    def getName(self):
        return "Discriminator"

class GeneratorV1(network.Network):

    class GeneratorStem(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, norm_layer : Callable[[int], torch.nn.Module], bias: bool, dim: int) -> None:
            super().__init__()
            self.add_module("Reflection", blocks.getTorchModule("ReflectionPad", dim)(1))
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels, out_channels, kernel_size=3, padding=0, bias=bias))
            self.add_module("Norm", norm_layer(out_channels))
            self.add_module("Relu", torch.nn.ReLU(True)) 

    class GeneratorHead(network.ModuleArgsDict):

        def __init__(self, channels: int, dim: int) -> None:
            super().__init__()
            self.add_module("Reflection", blocks.getTorchModule("ReflectionPad", dim)(1))
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(channels, 1, kernel_size=3, padding=0))

    class GeneratorDownSample(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, norm_layer: Callable[[int], torch.nn.Module], bias: bool, dim: int) -> None:
            super().__init__()
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=bias))
            self.add_module("Norm", norm_layer(out_channels))
            self.add_module("Relu", torch.nn.ReLU(True))

    class GeneratorResnetBlock(network.ModuleArgsDict):

        def __init__(self, channels : int, padding_type : str, norm_layer : Callable[[int], torch.nn.Module], dropout : bool, bias : bool, dim : int):
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

        def __init__(self, in_channels: int, out_channels: int, norm_layer: Callable[[int], torch.nn.Module], bias: bool, dim: int) -> None:
            super().__init__()
            self.add_module("ConvTranspose", blocks.getTorchModule("ConvTranspose", dim)(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias))
            self.add_module("Norm", norm_layer(out_channels))
            self.add_module("Relu", torch.nn.ReLU(True))
    
    class GeneratorEncoder(network.ModuleArgsDict):
        def __init__(self, channels: List[int], norm_layer: Callable[[int], torch.nn.Module], bias: bool, dim: int) -> None:
            super().__init__()
            for i, (in_channels, out_channels) in enumerate(zip(channels, channels[1:])):
                self.add_module("DownSample_{}".format(i), GeneratorV1.GeneratorDownSample(in_channels=in_channels, out_channels=out_channels, norm_layer=norm_layer, bias=bias, dim=dim))
    
    class GeneratorDecoder(network.ModuleArgsDict):
        def __init__(self, channels: List[int], norm_layer: Callable[[int], torch.nn.Module], bias: bool, dim: int) -> None:
            super().__init__()
            for i, (in_channels, out_channels) in enumerate(zip(reversed(channels), reversed(channels[:-1]))):
                self.add_module("UpSample_{}".format(i), GeneratorV1.GeneratorUpSample(in_channels=in_channels, out_channels=out_channels, norm_layer=norm_layer, bias=bias, dim=dim))
    
    class GeneratorNResnetBlock(network.ModuleArgsDict):

        def __init__(self, channels: int, norm_layer: Callable[[int], torch.nn.Module], bias: bool, dim: int) -> None:
            super().__init__()
            dropout = False
            n_blocks=6
            padding_type="Reflection"
            for i in range(n_blocks):
                self.add_module("ResnetBlock_{}".format(i), GeneratorV1.GeneratorResnetBlock(channels=channels, padding_type=padding_type, norm_layer=norm_layer, dropout=dropout, bias=bias, dim=dim))

    class GeneratorAutoEncoder(network.ModuleArgsDict):

        def __init__(self, ngf: int, norm_layer: Callable[[int], torch.nn.Module], bias: bool, dim: int) -> None:
            super().__init__()
            channels = [ngf*2**i for i in range(3)]
            
            self.add_module("Encoder", GeneratorV1.GeneratorEncoder(channels, norm_layer, bias, dim))
            self.add_module("NResnetBlock", GeneratorV1.GeneratorNResnetBlock(channels[-1], norm_layer, bias, dim))
            self.add_module("Decoder", GeneratorV1.GeneratorDecoder(channels, norm_layer, bias, dim))

    class GeneratorConv(network.ModuleArgsDict):

        def __init__(self, norm_layer: Callable[[int], torch.nn.Module], bias: bool, dim: int) -> None:
            super().__init__()
            self.add_module("Conv_0", blocks.getTorchModule("Conv", dim)(1, 32, kernel_size=3, padding=1, bias=bias))
            self.add_module("Norm_0", norm_layer(32))
            self.add_module("Relu_0", torch.nn.ReLU(True))
            self.add_module("Conv_1", blocks.getTorchModule("Conv", dim)(32, 32, kernel_size=3, padding=1, bias=bias))
            self.add_module("Norm_1", norm_layer(32))
            self.add_module("Relu_1", torch.nn.ReLU(True))

            self.add_module("Conv_2", blocks.getTorchModule("Conv", dim)(32, 1, kernel_size=3, padding=1, bias=bias))
            
            
    @config("Generator")
    def __init__(self, 
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.SchedulersLoader = network.SchedulersLoader(),
                    patch : ModelPatch = ModelPatch(),
                    outputsCriterions: Dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    in_channels : int = 1,
                    nb_batch_per_step: int = 64,
                    normMode: str = "INSTANCE",
                    dim : int = 3) -> None:
        super().__init__(optimizer=optimizer, in_channels=in_channels, schedulers=schedulers, patch=patch, outputsCriterions=outputsCriterions, dim=dim, nb_batch_per_step=nb_batch_per_step)
        channels: list[int]=[1, 64, 128]
        #self.add_module("Identity", torch.nn.Identity())
        #self.add_module("UNetBlock_0", UNetBlock(channels, nb_conv_per_stage=1, blockConfig=blocks.BlockConfig(3, 1,bias=True, activation="ReLU", normMode=normMode), downSampleMode=blocks.DownSampleMode.CONV_STRIDE, upSampleMode=blocks.UpSampleMode.CONV_TRANSPOSE, attention=False, dim=dim))
        #self.add_module("Head", GeneratorV1.GeneratorHead(channels=64, dim=dim))

        ngf=32
        norm_layer = partial(blocks.getNorm, blocks.NormMode._member_map_[normMode], dim=dim)
        bias = False
            
        self.add_module("Stem", GeneratorV1.GeneratorStem(in_channels=in_channels, out_channels=ngf, norm_layer=norm_layer, bias=bias, dim=dim))
        
        self.add_module("AutoEncoder", GeneratorV1.GeneratorAutoEncoder(ngf, norm_layer, bias, dim))

        self.add_module("Head", GeneratorV1.GeneratorConv(norm_layer, bias, dim=dim))

        self.add_module("Add", blocks.Add(), in_branch=[0,1])

        self.add_module("Tanh", torch.nn.Tanh())

        
    def getName(self):
        return "Generator"

class DiscriminatorV2(network.Network):

    class DiscriminatorStem(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, dim: int) -> None:
            super().__init__()
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1))
            self.add_module("LeakyReLU", torch.nn.LeakyReLU(0.2, True))

    class DiscriminatorHead(network.ModuleArgsDict):

        def __init__(self, channels: int, dim: int) -> None:
            super().__init__()
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels=channels, out_channels=1, kernel_size=4, stride=1, padding=1))
            self.add_module("AdaptiveAvgPool", blocks.getTorchModule("AdaptiveAvgPool", dim)(tuple([1]*dim)))
            self.add_module("Flatten", torch.nn.Flatten(1))
    
    class DiscriminatorLayers(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, stride: int, norm_layer: Callable[[int], torch.nn.Module], bias: bool, dim: int) -> None:
            super().__init__()
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=stride, padding=1, bias=bias))
            self.add_module("Norm", norm_layer(out_channels))
            self.add_module("LeakyReLU", torch.nn.LeakyReLU(0.2, True))

    class DiscriminatorNLayers(network.ModuleArgsDict):

        def __init__(self, channels: List[int], norm_layer: Callable[[int], torch.nn.Module], bias: bool, dim: int) -> None:
            super().__init__()
            for i, (in_channels, out_channels, stride) in enumerate(zip(channels, channels[1:], [2]*(len(channels)-2)+[1])):
                self.add_module("Layer_{}".format(i), DiscriminatorV2.DiscriminatorLayers(in_channels, out_channels, stride, norm_layer, bias, dim))


    @config("Discriminator")
    def __init__(self,
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.SchedulersLoader = network.SchedulersLoader(),
                    outputsCriterions: Dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    dim : int = 3,
                    nb_batch_per_step: int = 64,
                    normMode: str = "INSTANCE",
                    in_channels : int = 1) -> None:
        super().__init__(in_channels = in_channels, optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, dim=dim, nb_batch_per_step=nb_batch_per_step)
        
        ndf = 32
        n_layers=3
        norm_layer = partial(blocks.getNorm, blocks.NormMode._member_map_[normMode], dim=dim)
        bias = True
        channels = [ndf*min(2 ** i, 8) for i in range(n_layers+1)]
        self.add_module("Stem", DiscriminatorV2.DiscriminatorStem(in_channels, ndf, dim))
        self.add_module("Layers", DiscriminatorV2.DiscriminatorNLayers(channels, norm_layer, bias, dim))
        self.add_module("Head", DiscriminatorV2.DiscriminatorHead(channels=channels[-1], dim=dim))


    def getName(self):
        return "Discriminator"

class GeneratorV2(network.Network):

    class GeneratorStem(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, norm_layer : Callable[[int], torch.nn.Module], bias: bool, dim: int) -> None:
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

        def __init__(self, in_channels: int, out_channels: int, norm_layer: Callable[[int], torch.nn.Module], bias: bool, dim: int) -> None:
            super().__init__()
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=bias))
            self.add_module("Norm", norm_layer(out_channels))
            self.add_module("Relu", torch.nn.ReLU(True))

    class GeneratorResnetBlock(network.ModuleArgsDict):

        def __init__(self, channels : int, padding_type : str, norm_layer : Callable[[int], torch.nn.Module], dropout : bool, bias : bool, dim : int):
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

        def __init__(self, in_channels: int, out_channels: int, norm_layer: Callable[[int], torch.nn.Module], bias: bool, dim: int) -> None:
            super().__init__()
            self.add_module("ConvTranspose", blocks.getTorchModule("ConvTranspose", dim)(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias))
            self.add_module("Norm", norm_layer(out_channels))
            self.add_module("Relu", torch.nn.ReLU(True))
    
    class GeneratorEncoder(network.ModuleArgsDict):
        def __init__(self, channels: List[int], norm_layer: Callable[[int], torch.nn.Module], bias: bool, dim: int) -> None:
            super().__init__()
            for i, (in_channels, out_channels) in enumerate(zip(channels, channels[1:])):
                self.add_module("DownSample_{}".format(i), GeneratorV2.GeneratorDownSample(in_channels=in_channels, out_channels=out_channels, norm_layer=norm_layer, bias=bias, dim=dim))
    
    class GeneratorDecoder(network.ModuleArgsDict):
        def __init__(self, channels: List[int], norm_layer: Callable[[int], torch.nn.Module], bias: bool, dim: int) -> None:
            super().__init__()
            for i, (in_channels, out_channels) in enumerate(zip(reversed(channels), reversed(channels[:-1]))):
                self.add_module("UpSample_{}".format(i), GeneratorV2.GeneratorUpSample(in_channels=in_channels, out_channels=out_channels, norm_layer=norm_layer, bias=bias, dim=dim))
    
    class GeneratorNResnetBlock(network.ModuleArgsDict):

        def __init__(self, channels: int, norm_layer: Callable[[int], torch.nn.Module], bias: bool, dim: int) -> None:
            super().__init__()
            dropout = False
            n_blocks=6
            padding_type="Reflection"
            for i in range(n_blocks):
                self.add_module("ResnetBlock_{}".format(i), GeneratorV2.GeneratorResnetBlock(channels=channels, padding_type=padding_type, norm_layer=norm_layer, dropout=dropout, bias=bias, dim=dim))

    class GeneratorAutoEncoder(network.ModuleArgsDict):

        def __init__(self, ngf: int, norm_layer: Callable[[int], torch.nn.Module], bias: bool, dim: int) -> None:
            super().__init__()
            channels = [ngf*2**i for i in range(3)]
            
            self.add_module("Identity", torch.nn.Identity())
            
            self.add_module("Encoder", GeneratorV2.GeneratorEncoder(channels, norm_layer, bias, dim))
            self.add_module("NResnetBlock", GeneratorV2.GeneratorNResnetBlock(channels[-1], norm_layer, bias, dim))
            self.add_module("Decoder", GeneratorV2.GeneratorDecoder(channels, norm_layer, bias, dim))
            
            
    @config("Generator")
    def __init__(self, 
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.SchedulersLoader = network.SchedulersLoader(),
                    patch : ModelPatch = ModelPatch(),
                    outputsCriterions: Dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    in_channels : int = 1,
                    nb_batch_per_step: int = 64,
                    normMode: str = "INSTANCE",
                    dim : int = 3) -> None:
        super().__init__(optimizer=optimizer, in_channels=in_channels, schedulers=schedulers, patch=patch, outputsCriterions=outputsCriterions, dim=dim, nb_batch_per_step=nb_batch_per_step)
        ngf=32
        norm_layer = partial(blocks.getNorm, blocks.NormMode._member_map_[normMode], dim=dim)
        bias = True
        
        self.add_module("Stem", GeneratorV2.GeneratorStem(in_channels=in_channels, out_channels=ngf, norm_layer=norm_layer, bias=bias, dim=dim))
        
        self.add_module("AutoEncoder", GeneratorV2.GeneratorAutoEncoder(ngf, norm_layer, bias, dim))
        
        self.add_module("Head", GeneratorV2.GeneratorHead(channels=ngf, dim=dim))

    def getName(self):
        return "Generator"


class Gan(network.Network):

    @config("Gan")
    def __init__(self, generator : GeneratorV2 = GeneratorV2(), discriminator : DiscriminatorV2 = DiscriminatorV2()) -> None:
        super().__init__()
        self.add_module("Discriminator_B", discriminator, in_branch=[1], out_branch=[-1], requires_grad=True)
        self.add_module("Generator_A_to_B", generator, in_branch=[0], out_branch=["pB"])
        
        self.add_module("detach", blocks.Detach(), in_branch=["pB"], out_branch=["pB_detach"])
        self.add_module("Discriminator_pB_detach", discriminator, in_branch=["pB_detach"], out_branch=[-1])
          
        self.add_module("Discriminator_pB", discriminator, in_branch=["pB"], out_branch=[-1], requires_grad=False)

class CycleGan(network.Network):

    @config("CycleGan")
    def __init__(self, generator : GeneratorV1 = GeneratorV1(), discriminator : DiscriminatorV1 = DiscriminatorV1()) -> None:
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
