from DeepLearning_API.networks import network
from DeepLearning_API.config import config
from DeepLearning_API.networks import blocks
from typing import Dict, List
from DeepLearning_API.models.segmentation.uNet import UNetBlock
import torch

class Generator(network.Network):

    class UNetHead(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, dim: int) -> None:
            super().__init__()
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1))

    @config("Generator")
    def __init__(   self,
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.SchedulersLoader = network.SchedulersLoader(),
                    outputsCriterions: Dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    dim : int = 3,
                    channels: List[int]=[1, 64, 128, 256, 512, 1024],
                    blockConfig: blocks.BlockConfig = blocks.BlockConfig(),
                    nb_conv_per_stage: int = 2,
                    downSampleMode: str = "MAXPOOL",
                    upSampleMode: str = "CONV_TRANSPOSE",
                    attention : bool = False,
                    nb_batch_per_step: int = 1) -> None:
        super().__init__(in_channels = channels[0], optimizer = optimizer, nb_batch_per_step=nb_batch_per_step, schedulers = schedulers, outputsCriterions = outputsCriterions, dim = dim)
        self.add_module("UNetBlock_0", UNetBlock(channels, nb_conv_per_stage, blockConfig, downSampleMode=blocks.DownSampleMode._member_map_[downSampleMode], upSampleMode=blocks.UpSampleMode._member_map_[upSampleMode], attention=attention, dim=dim))
        self.add_module("Head", Generator.UNetHead(in_channels=channels[1], out_channels=channels[0], dim=dim))
        self.add_module("Add", blocks.Add(), in_branch=[0,1])
        self.add_module("Tanh", torch.nn.Tanh())