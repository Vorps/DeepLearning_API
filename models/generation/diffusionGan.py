import copy
from functools import partial
from typing import Union
import torch
from DeepLearning_API.config import config
from DeepLearning_API.networks import network, blocks
from DeepLearning_API.HDF5 import ModelPatch
from DeepLearning_API.models.generation.ddpm import DDPM
from DeepLearning_API.models.segmentation import UNet, NestedUNet
import numpy as np

class Discriminator(network.Network):

    class DDPM_TE(torch.nn.Module):

        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.linear_0 = torch.nn.Linear(in_channels, out_channels)
            self.siLU = torch.nn.SiLU()
            self.linear_1 = torch.nn.Linear(out_channels, out_channels)
        
        def forward(self, input: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return input + self.linear_1(self.siLU(self.linear_0(t))).reshape(input.shape[0], -1, *[1 for _ in range(len(input.shape)-2)])
        
    class Discriminator_SampleT(torch.nn.Module):

        def __init__(self, noise_step: int) -> None:
            super().__init__()
            self.noise_step_C = noise_step
            self.noise_step = 0.1
            self.measure = None
            self.C = 1
            self._it = 0
            self.n = 4
            self.d = 0.5

        def setMeasure(self, measure: network.Measure, names: list[str]):
            self.measure = measure
            self.names = names

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            if self.measure is not None and self._it % self.n == 0:
                value = sum([v for k, v in self.measure.getLastMetrics_hist(self.n).items() if k in self.names])
                self.noise_step += np.sign(self.d - value)*self.C
                self.noise_step = np.clip(self.noise_step, 1, self.noise_step_C)
            result = torch.randint(0, int(np.clip(self.noise_step, 1, self.noise_step_C)), (input.shape[0],)).to(input.device)
            self._it += 1
            return result
        
    class DiscriminatorNLayers(network.ModuleArgsDict):

        def __init__(self, channels: list[int], strides: list[int], time_embedding_dim: int, dim: int) -> None:
            super().__init__()
            blockConfig = partial(blocks.BlockConfig, kernel_size=4, padding=1, bias=False, activation=partial(torch.nn.LeakyReLU, negative_slope = 0.2, inplace=True), normMode=blocks.NormMode.SYNCBATCH)
            for i, (in_channels, out_channels, stride) in enumerate(zip(channels, channels[1:], strides)):
                self.add_module("Te_{}".format(i), Discriminator.DDPM_TE(time_embedding_dim, in_channels), in_branch=[0, 1])
                self.add_module("Layer_{}".format(i), blocks.ConvBlock(in_channels, out_channels, [blockConfig(stride=stride)], dim))
    
    class DiscriminatorHead(network.ModuleArgsDict):

        def __init__(self, channels: int, dim: int) -> None:
            super().__init__()
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels=channels, out_channels=1, kernel_size=4, stride=1, padding=1))
            #self.add_module("AdaptiveAvgPool", blocks.getTorchModule("AdaptiveAvgPool", dim)(tuple([1]*dim)))
            #self.add_module("Flatten", torch.nn.Flatten(1))

    class DiscriminatorBlock(network.ModuleArgsDict):

        def __init__(self,  channels: list[int] = [1, 16, 32, 64, 64],
                            strides: list[int] = [2,2,2,1],
                            dim : int = 3) -> None:
            super().__init__()
            noise_step = 1000
            beta_start = 1e-4
            beta_end = 0.02
            time_embedding_dim = 100
            self.add_module("Noise", blocks.NormalNoise(), out_branch=["eta"])
            self.add_module("Sample", Discriminator.Discriminator_SampleT(noise_step), out_branch=["t"])
            self.add_module("Forward", DDPM.DDPM_ForwardProcess(noise_step, beta_start, beta_end), in_branch=[0, "t", "eta"])
            self.add_module("t", DDPM.DDPM_TimeEmbedding(noise_step, time_embedding_dim), in_branch=["t"], out_branch=["te"])
            self.add_module("Layers", Discriminator.DiscriminatorNLayers(channels, strides, time_embedding_dim, dim), in_branch=[0, "te"])
            self.add_module("Head", Discriminator.DiscriminatorHead(channels[-1], dim))

    @config("Discriminator")
    def __init__(self,
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.SchedulersLoader = network.SchedulersLoader(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    channels: list[int] = [1, 16, 32, 64, 64],
                    strides: list[int] = [2,2,2,1],
                    dim : int = 3) -> None:
        super().__init__(in_channels = 1, optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, dim=dim)
        self.add_module("DiscriminatorModel", Discriminator.DiscriminatorBlock(channels, strides, dim))

    def initialized(self):
        self["DiscriminatorModel"]["Sample"].setMeasure(self.measure, ["Discriminator_B.DiscriminatorModel.Head.Conv:None:PatchGanLoss"])

class Generator_V1(network.Network):

    class GeneratorStem(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, dim: int) -> None:
            super().__init__()
            self.add_module("ConvBlock", blocks.ConvBlock(in_channels, out_channels, blockConfigs=[blocks.BlockConfig(bias=False, activation="ReLU", normMode="SYNCBATCH")], dim=dim))

    class GeneratorHead(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, dim: int) -> None:
            super().__init__()
            self.add_module("ConvBlock", blocks.ConvBlock(in_channels, in_channels, blockConfigs=[blocks.BlockConfig(bias=False, activation="ReLU", normMode="SYNCBATCH")], dim=dim))
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels, out_channels, kernel_size=1, bias=False))
            self.add_module("Tanh", torch.nn.Tanh())

    class GeneratorDownSample(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, dim: int) -> None:
            super().__init__()
            self.add_module("ConvBlock", blocks.ConvBlock(in_channels, out_channels, blockConfigs=[blocks.BlockConfig(stride=2, bias=False, activation="ReLU", normMode="SYNCBATCH")], dim=dim))
    
    class GeneratorUpSample(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, dim: int) -> None:
            super().__init__()
            self.add_module("ConvBlock", blocks.ConvBlock(in_channels, out_channels, blockConfigs=[blocks.BlockConfig(bias=False, activation="ReLU", normMode="SYNCBATCH")], dim=dim))
            self.add_module("Upsample", torch.nn.Upsample(scale_factor=2, mode="bilinear" if dim < 3 else "trilinear"))
    
    class GeneratorEncoder(network.ModuleArgsDict):
        def __init__(self, channels: list[int], dim: int) -> None:
            super().__init__()
            for i, (in_channels, out_channels) in enumerate(zip(channels, channels[1:])):
                self.add_module("DownSample_{}".format(i), Generator_V1.GeneratorDownSample(in_channels=in_channels, out_channels=out_channels, dim=dim))
    
    class GeneratorResnetBlock(network.ModuleArgsDict):

        def __init__(self, channels : int, dim : int):
            super().__init__()
            self.add_module("Conv_0", blocks.getTorchModule("Conv", dim)(channels, channels, kernel_size=3, padding=1, bias=False))
            self.add_module("Norm", torch.nn.LeakyReLU(0.2, inplace=True))
            self.add_module("Conv_1", blocks.getTorchModule("Conv", dim)(channels, channels, kernel_size=3, padding=1, bias=False))
            self.add_module("Residual", blocks.Add(), in_branch=[0,1])

    class GeneratorNResnetBlock(network.ModuleArgsDict):

        def __init__(self, channels: int, nb_conv: int, dim: int) -> None:
            super().__init__()
            for i in range(nb_conv):
                self.add_module("ResnetBlock_{}".format(i), Generator_V1.GeneratorResnetBlock(channels=channels, dim=dim))

    class GeneratorDecoder(network.ModuleArgsDict):
        def __init__(self, channels: list[int], dim: int) -> None:
            super().__init__()
            for i, (in_channels, out_channels) in enumerate(zip(reversed(channels), reversed(channels[:-1]))):
                self.add_module("UpSample_{}".format(i), Generator_V1.GeneratorUpSample(in_channels=in_channels, out_channels=out_channels, dim=dim))
    
    class GeneratorAutoEncoder(network.ModuleArgsDict):

        def __init__(self, ngf: int, dim: int) -> None:
            super().__init__()
            channels = [ngf, ngf*2]
            self.add_module("Encoder", Generator_V1.GeneratorEncoder(channels, dim))
            self.add_module("NResBlock", Generator_V1.GeneratorNResnetBlock(channels=channels[-1], nb_conv=6, dim=dim))
            self.add_module("Decoder", Generator_V1.GeneratorDecoder(channels, dim))

    class GeneratorBlock(network.ModuleArgsDict):

        def __init__(self, ngf: int, dim: int) -> None:
            super().__init__()
            self.add_module("Stem", Generator_V1.GeneratorStem(3, ngf, dim))
            self.add_module("AutoEncoder", Generator_V1.GeneratorAutoEncoder(ngf, dim))
            self.add_module("Head", Generator_V1.GeneratorHead(in_channels=ngf, out_channels=1, dim=dim))

    @config("Generator_V1")
    def __init__(self, 
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.SchedulersLoader = network.SchedulersLoader(),
                    patch : ModelPatch = ModelPatch(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    dim : int = 3) -> None:
        super().__init__(optimizer=optimizer, in_channels=3, schedulers=schedulers, patch=patch, outputsCriterions=outputsCriterions, dim=dim)
        self.add_module("GeneratorModel", Generator_V1.GeneratorBlock(32, dim))

class Generator_V2(network.Network):

    class NestedUNetHead(network.ModuleArgsDict):

        def __init__(self, in_channels: list[int], dim: int) -> None:
            super().__init__()
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels = in_channels[1], out_channels = 1, kernel_size = 1, stride = 1, padding = 0))
            self.add_module("Tanh", torch.nn.Tanh())
    
    class GeneratorBlock(network.ModuleArgsDict):

        def __init__(self, 
                    channels: list[int],
                    blockConfig: blocks.BlockConfig,
                    nb_conv_per_stage: int,
                    downSampleMode: str,
                    upSampleMode: str,
                    attention : bool,
                    blockType: str,
                    dim : int,) -> None:
            super().__init__()
            self.add_module("UNetBlock_0", NestedUNet.NestedUNetBlock(channels, nb_conv_per_stage, blockConfig, downSampleMode=blocks.DownSampleMode._member_map_[downSampleMode], upSampleMode=blocks.UpSampleMode._member_map_[upSampleMode], attention=attention, block = blocks.ConvBlock if blockType == "Conv" else blocks.ResBlock, dim=dim), out_branch=["X_0_{}".format(j+1) for j in range(len(channels)-2)])    
            self.add_module("Head", Generator_V2.NestedUNetHead(channels[:2], dim=dim), in_branch=["X_0_{}".format(len(channels)-2)])

    @config("Generator_V2")
    def __init__(   self,
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.SchedulersLoader = network.SchedulersLoader(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    patch : Union[ModelPatch, None] = None,
                    channels: list[int]=[1, 64, 128, 256, 512, 1024],
                    blockConfig: blocks.BlockConfig = blocks.BlockConfig(),
                    nb_conv_per_stage: int = 2,
                    downSampleMode: str = "MAXPOOL",
                    upSampleMode: str = "CONV_TRANSPOSE",
                    attention : bool = False,
                    blockType: str = "Conv",
                    dim : int = 3) -> None:
        super().__init__(in_channels = channels[0], optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, patch=patch, dim = dim)
        self.add_module("GeneratorModel", Generator_V2.GeneratorBlock(channels, blockConfig, nb_conv_per_stage, downSampleMode, upSampleMode, attention, blockType, dim))

class Generator_V3(network.Network):

    class NestedUNetHead(network.ModuleArgsDict):

        def __init__(self, in_channels: list[int], dim: int) -> None:
            super().__init__()
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels = in_channels[1], out_channels = 1, kernel_size = 1, stride = 1, padding = 0))
            self.add_module("Tanh", torch.nn.Tanh())
    
    class GeneratorBlock(network.ModuleArgsDict):

        def __init__(self, 
                    channels: list[int],
                    blockConfig: blocks.BlockConfig,
                    nb_conv_per_stage: int,
                    downSampleMode: str,
                    upSampleMode: str,
                    attention : bool,
                    blockType: str,
                    dim : int,) -> None:
            super().__init__()
            self.add_module("UNetBlock_0", UNet.UNetBlock(channels, nb_conv_per_stage, blockConfig, downSampleMode=blocks.DownSampleMode._member_map_[downSampleMode], upSampleMode=blocks.UpSampleMode._member_map_[upSampleMode], attention=attention, block = blocks.ConvBlock if blockType == "Conv" else blocks.ResBlock, nb_class=1, dim=dim), out_branch=["X_0_{}".format(j+1) for j in range(len(channels)-2)])    
            self.add_module("Head", Generator_V3.NestedUNetHead(channels[:2], dim=dim), in_branch=["X_0_{}".format(len(channels)-2)])

    @config("Generator_V3")
    def __init__(   self,
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.SchedulersLoader = network.SchedulersLoader(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    patch : Union[ModelPatch, None] = None,
                    channels: list[int]=[1, 64, 128, 256, 512, 1024],
                    blockConfig: blocks.BlockConfig = blocks.BlockConfig(),
                    nb_conv_per_stage: int = 2,
                    downSampleMode: str = "MAXPOOL",
                    upSampleMode: str = "CONV_TRANSPOSE",
                    attention : bool = False,
                    blockType: str = "Conv",
                    dim : int = 3) -> None:
        super().__init__(in_channels = channels[0], optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, patch=patch, dim = dim)
        self.add_module("GeneratorModel", Generator_V3.GeneratorBlock(channels, blockConfig, nb_conv_per_stage, downSampleMode, upSampleMode, attention, blockType, dim), out_branch=["pB"])

class DiffusionGan(network.Network):

    @config("DiffusionGan")
    def __init__(self, generator : Generator_V1 = Generator_V1(), discriminator : Discriminator = Discriminator()) -> None:
        super().__init__()
        self.add_module("Generator_A_to_B", generator, in_branch=[0], out_branch=["pB"])
        self.add_module("Discriminator_B", discriminator, in_branch=[1], out_branch=[-1], requires_grad=True)
        self.add_module("detach", blocks.Detach(), in_branch=["pB"], out_branch=["pB_detach"])
        self.add_module("Discriminator_pB_detach", discriminator, in_branch=["pB_detach"], out_branch=[-1])
        self.add_module("Discriminator_pB", discriminator, in_branch=["pB"], out_branch=[-1], requires_grad=False)

class CycleGanDiscriminator(network.Network):

    @config("CycleGanDiscriminator")
    def __init__(self,
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.SchedulersLoader = network.SchedulersLoader(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    patch : Union[ModelPatch, None] = None,
                    channels: list[int] = [1, 16, 32, 64, 64],
                    strides: list[int] = [2,2,2,1],
                    dim : int = 3) -> None:
        super().__init__(in_channels = 1, optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, patch=patch, dim=dim)
        self.add_module("Discriminator_A", Discriminator.DiscriminatorBlock(channels, strides, dim), in_branch=[0], out_branch=[0])
        self.add_module("Discriminator_B", Discriminator.DiscriminatorBlock(channels, strides, dim), in_branch=[1], out_branch=[1])
        
    def initialized(self):
        self["Discriminator_A"]["Sample"].setMeasure(self.measure, ["Discriminator.Discriminator_A.Head.Flatten:None:PatchGanLoss"])
        self["Discriminator_B"]["Sample"].setMeasure(self.measure, ["Discriminator.Discriminator_B.Head.Flatten:None:PatchGanLoss"])

class CycleGanGenerator_V1(network.Network):

    @config("CycleGanGenerator_V1")
    def __init__(self, 
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.SchedulersLoader = network.SchedulersLoader(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    patch : Union[ModelPatch, None] = None,
                    dim : int = 3) -> None:
        super().__init__(in_channels = 1, optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions,  patch=patch, dim=dim)
        self.add_module("Generator_A_to_B", Generator_V1.GeneratorBlock(32, dim), in_branch=[0], out_branch=["pB"])
        self.add_module("Generator_B_to_A", Generator_V1.GeneratorBlock(32, dim), in_branch=[1], out_branch=["pA"])

class CycleGanGenerator_V2(network.Network):

    @config("CycleGanGenerator_V2")
    def __init__(self, 
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.SchedulersLoader = network.SchedulersLoader(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    patch : Union[ModelPatch, None] = None,
                    channels: list[int]=[1, 64, 128, 256, 512, 1024],
                    blockConfig: blocks.BlockConfig = blocks.BlockConfig(),
                    nb_conv_per_stage: int = 2,
                    downSampleMode: str = "MAXPOOL",
                    upSampleMode: str = "CONV_TRANSPOSE",
                    attention : bool = False,
                    blockType: str = "Conv",
                    dim : int = 3) -> None:
        super().__init__(in_channels = 1, optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, patch=patch, dim=dim)
        self.add_module("Generator_A_to_B", Generator_V2.GeneratorBlock(channels, blockConfig, nb_conv_per_stage, downSampleMode, upSampleMode, attention, blockType, dim), in_branch=[0], out_branch=["pB"])
        self.add_module("Generator_B_to_A", Generator_V2.GeneratorBlock(channels, blockConfig, nb_conv_per_stage, downSampleMode, upSampleMode, attention, blockType, dim), in_branch=[1], out_branch=["pA"])

class CycleGanGenerator_V3(network.Network):

    @config("CycleGanGenerator_V3")
    def __init__(self, 
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.SchedulersLoader = network.SchedulersLoader(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    patch : Union[ModelPatch, None] = None,
                    channels: list[int]=[1, 64, 128, 256, 512, 1024],
                    blockConfig: blocks.BlockConfig = blocks.BlockConfig(),
                    nb_conv_per_stage: int = 2,
                    downSampleMode: str = "MAXPOOL",
                    upSampleMode: str = "CONV_TRANSPOSE",
                    attention : bool = False,
                    blockType: str = "Conv",
                    dim : int = 3) -> None:
        super().__init__(in_channels = 1, optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, patch=patch, dim=dim)
        self.add_module("Generator_A_to_B", Generator_V3.GeneratorBlock(channels, blockConfig, nb_conv_per_stage, downSampleMode, upSampleMode, attention, blockType, dim), in_branch=[0], out_branch=["pB"])
        self.add_module("Generator_B_to_A", Generator_V3.GeneratorBlock(channels, blockConfig, nb_conv_per_stage, downSampleMode, upSampleMode, attention, blockType, dim), in_branch=[1], out_branch=["pA"])

class DiffusionCycleGan(network.Network):

    @config("DiffusionCycleGan")
    def __init__(self, generators : CycleGanGenerator_V3 = CycleGanGenerator_V3(), discriminators : CycleGanDiscriminator = CycleGanDiscriminator()) -> None:
        super().__init__()
        self.add_module("Generator", generators, in_branch=[0, 1], out_branch=["pB", "pA"])
        self.add_module("Discriminator", discriminators, in_branch=[0, 1], out_branch=[-1], requires_grad=True)
        
        self.add_module("Generator_identity", generators, in_branch=[1, 0], out_branch=[-1])
        
        self.add_module("Generator_p", generators, in_branch=["pA", "pB"], out_branch=[-1])
    
        self.add_module("detach_pA", blocks.Detach(), in_branch=["pA"], out_branch=["pA_detach"])
        self.add_module("detach_pB", blocks.Detach(), in_branch=["pB"], out_branch=["pB_detach"])

        self.add_module("Discriminator_p_detach", discriminators, in_branch=["pA_detach", "pB_detach"], out_branch=[-1])
        self.add_module("Discriminator_p", discriminators, in_branch=["pA", "pB"], out_branch=[-1], requires_grad=False)
        
        