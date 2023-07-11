from DeepLearning_API.networks import network, blocks
from DeepLearning_API.config import config
import torch
from functools import partial

class VAE(network.Network):

    class AutoEncoderBlock(network.ModuleArgsDict):

        def __init__(self, channels: list[int], nb_conv_per_stage: int, blockConfig: blocks.BlockConfig, downSampleMode: blocks.DownSampleMode, upSampleMode: blocks.UpSampleMode, dim: int, block: type, i : int = 0) -> None:
            super().__init__()
            if i > 0:
                self.add_module(downSampleMode.name, blocks.downSample(in_channels=channels[0], out_channels=channels[1], downSampleMode=downSampleMode, dim=dim))
            self.add_module("DownBlock", block(in_channels=channels[1 if downSampleMode == blocks.DownSampleMode.CONV_STRIDE and i > 0 else 0], out_channels=channels[1], nb_conv=nb_conv_per_stage, blockConfig=blockConfig, dim=dim))
            if len(channels) > 2:
                self.add_module("AutoEncoder_{}".format(i+1), VAE.AutoEncoderBlock(channels[1:], nb_conv_per_stage, blockConfig, downSampleMode, upSampleMode, dim, block, i+1))
                self.add_module("UpBlock", block(in_channels=channels[2] if upSampleMode != blocks.UpSampleMode.CONV_TRANSPOSE else channels[1], out_channels=channels[1], nb_conv=nb_conv_per_stage, blockConfig=blockConfig, dim=dim))
            if i > 0:
               self.add_module(upSampleMode.name, blocks.upSample(in_channels=channels[1], out_channels=channels[0], upSampleMode=upSampleMode, dim=dim))
    
    class VAE_Head(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, dim: int) -> None:
            super().__init__()
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1))
            self.add_module("Tanh", torch.nn.ReLU()) 

    @config("VAE")
    def __init__(self,
                    optimizer: network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers: network.SchedulersLoader = network.SchedulersLoader(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    dim : int = 3,
                    channels: list[int]=[1, 64, 128, 256, 512, 1024],
                    blockConfig: blocks.BlockConfig = blocks.BlockConfig(),
                    nb_conv_per_stage: int = 2,
                    downSampleMode: str = "MAXPOOL",
                    upSampleMode: str = "CONV_TRANSPOSE",
                    blockType: str = "Conv") -> None:
        
        super().__init__(in_channels = channels[0], init_type="normal", optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, dim=dim, nb_batch_per_step=1)
        
        self.add_module("AutoEncoder_0", VAE.AutoEncoderBlock(channels, nb_conv_per_stage, blockConfig, downSampleMode=blocks.DownSampleMode._member_map_[downSampleMode], upSampleMode=blocks.UpSampleMode._member_map_[upSampleMode], dim=dim, block = blocks.ConvBlock if blockType == "Conv" else blocks.ResBlock))
        self.add_module("Head", VAE.VAE_Head(channels[1], channels[0], dim))