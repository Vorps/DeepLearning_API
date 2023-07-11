from DeepLearning_API.config import config
from DeepLearning_API.networks import network, blocks
import torch
from DeepLearning_API.HDF5 import ModelPatch
from typing import Union

class UNetBlock(network.ModuleArgsDict):

    def __init__(self, channels: list[int], nb_conv_per_stage: int, blockConfig: blocks.BlockConfig, downSampleMode: blocks.DownSampleMode, upSampleMode: blocks.UpSampleMode, attention : bool, block: type, dim: int, i : int = 0) -> None:
        super().__init__()
        if i > 0:
            self.add_module(downSampleMode.name, blocks.downSample(in_channels=channels[0], out_channels=channels[1], downSampleMode=downSampleMode, dim=dim))
        self.add_module("DownConvBlock", block(in_channels=channels[1 if downSampleMode == blocks.DownSampleMode.CONV_STRIDE and i > 0 else 0], out_channels=channels[1], nb_conv=nb_conv_per_stage, blockConfig=blockConfig, dim=dim))
        if len(channels) > 2:
            self.add_module("UNetBlock_{}".format(i+1), UNetBlock(channels[1:], nb_conv_per_stage, blockConfig, downSampleMode, upSampleMode, attention, block, dim, i+1))
            self.add_module("UpConvBlock", block(in_channels=(channels[1]+channels[2]) if upSampleMode != blocks.UpSampleMode.CONV_TRANSPOSE else channels[1]*2, out_channels=channels[1], nb_conv=nb_conv_per_stage, blockConfig=blockConfig, dim=dim))
        if i > 0:
            if attention:
                self.add_module("Attention", blocks.Attention(F_g=channels[1], F_l=channels[0], F_int=channels[0], dim=dim), in_branch=[1, 0], out_branch=[1])
            self.add_module(upSampleMode.name, blocks.upSample(in_channels=channels[1], out_channels=channels[0], upSampleMode=upSampleMode, dim=dim))
            self.add_module("SkipConnection", blocks.Concat(), in_branch=[0, 1])

class UNet(network.Network):

    class UNetHead(network.ModuleArgsDict):

        def __init__(self, in_channels: int, nb_class: int, dim: int) -> None:
            super().__init__()
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels = in_channels, out_channels = nb_class, kernel_size = 3, stride = 1, padding = 1))
            self.add_module("Softmax", torch.nn.Softmax(dim=1))
            self.add_module("Argmax", blocks.ArgMax(dim=1))

    @config("UNet")
    def __init__(   self,
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.SchedulersLoader = network.SchedulersLoader(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    patch : Union[ModelPatch, None] = None,
                    dim : int = 3,
                    channels: list[int]=[1, 64, 128, 256, 512, 1024],
                    nb_class: int = 2,
                    blockConfig: blocks.BlockConfig = blocks.BlockConfig(),
                    nb_conv_per_stage: int = 2,
                    downSampleMode: str = "MAXPOOL",
                    upSampleMode: str = "CONV_TRANSPOSE",
                    attention : bool = False,
                    blockType: str = "Conv") -> None:
        super().__init__(in_channels = channels[0], optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, patch=patch, dim = dim)
        self.add_module("UNetBlock_0", UNetBlock(channels, nb_conv_per_stage, blockConfig, downSampleMode=blocks.DownSampleMode._member_map_[downSampleMode], upSampleMode=blocks.UpSampleMode._member_map_[upSampleMode], attention=attention, block = blocks.ConvBlock if blockType == "Conv" else blocks.ResBlock, dim=dim))    
        self.add_module("Head", UNet.UNetHead(in_channels=channels[1], nb_class=nb_class, dim=dim))