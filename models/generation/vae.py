from DeepLearning_API.networks import network
from DeepLearning_API.config import config

import torch
from networks import blocks
from typing import Dict

class VAE(network.Network):

    class VAE_Encoder(network.ModuleArgsDict):

        def __init__(self, dim: int) -> None:
            super().__init__()                                          #1 256 256
            self.add_module("Conv_0", torch.nn.Conv2d(1, 32, 4, 2, 1))  #32 128 128
            self.add_module("Activation_0", torch.nn.ReLU(True))

            self.add_module("Conv_1", torch.nn.Conv2d(32, 64, 4, 2, 1)) #64 64 64
            self.add_module("Activation_1", torch.nn.ReLU(True))
            
            self.add_module("Conv_2", torch.nn.Conv2d(64, 128, 4, 2, 1)) #128 32 32
            self.add_module("Activation_2", torch.nn.ReLU(True))
            
            self.add_module("Conv_3", torch.nn.Conv2d(128, 256, 4, 2, 1)) #256 16 16
            self.add_module("Activation_3", torch.nn.ReLU(True))

            self.add_module("Conv_4", torch.nn.Conv2d(256, 512, 4, 2, 1)) #512 8 8
            self.add_module("Activation_4", torch.nn.ReLU(True))

            self.add_module("Conv_5", torch.nn.Conv2d(512, 1024, 4, 2, 1)) #1024 4 4
            self.add_module("Activation_5", torch.nn.ReLU(True))

            self.add_module("Conv_6", torch.nn.Conv2d(1024, 2048, 4, 1)) #2048 1 1
            self.add_module("Activation_6", torch.nn.ReLU(True))

    class VAE_Decoder(network.ModuleArgsDict):

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.add_module("ConvTranspose_0", torch.nn.ConvTranspose2d(2048, 1024, 4)) #1024 4 4
            self.add_module("Activation_0", torch.nn.ReLU(True))
            
            self.add_module("ConvTranspose_1", torch.nn.ConvTranspose2d(1024, 512, 4, 2, 1)) #512 8 8
            self.add_module("Activation_1", torch.nn.ReLU(True))
            
            self.add_module("ConvTranspose_2", torch.nn.ConvTranspose2d(512, 256, 4, 2, 1)) #256 16 16
            self.add_module("Activation_2", torch.nn.ReLU(True))
            
            self.add_module("ConvTranspose_3", torch.nn.ConvTranspose2d(256, 128, 4, 2, 1)) #128 32 32
            self.add_module("Activation_3", torch.nn.ReLU(True))

    class Downsample(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.add_module("Conv", torch.nn.Conv2d(in_channels, out_channels, 4, 2, 1))
            self.add_module("Activation", torch.nn.ReLU(True))
    
    class Upsample(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.add_module("Conv_0", torch.nn.Conv2d(in_channels*2, in_channels, 3, 1, 1))
            self.add_module("Activation_0", torch.nn.LeakyReLU(0.02, True))
            self.add_module("Conv_1", torch.nn.Conv2d(in_channels, in_channels, 3, 1, 1))
            self.add_module("Activation_1", torch.nn.LeakyReLU(0.02, True))
            self.add_module("ConvTranspose", torch.nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1))
            self.add_module("Activation_2", torch.nn.LeakyReLU(0.02, True))

    class VAE_UNet(network.ModuleArgsDict):

        def __init__(self, dim: int) -> None:
            super().__init__()

            self.add_module("Downsample_0", VAE.Downsample(1, 32), out_branch=[0, "Stage_0"])

            self.add_module("Downsample_1", VAE.Downsample(32, 64), out_branch=[0, "Stage_1"])
            
            self.add_module("Downsample_2", VAE.Downsample(64, 128))
            
            self.add_module("SkipConnection_0", blocks.Concat(), in_branch=[0, 1])
            self.add_module("Upsample_0", VAE.Upsample(128, 64))
            
            self.add_module("SkipConnection_1", blocks.Concat(), in_branch=[0, "Stage_1"])
            self.add_module("Upsample_1", VAE.Upsample(64, 32))

            self.add_module("SkipConnection_2", blocks.Concat(), in_branch=[0, "Stage_0"])
            self.add_module("Upsample_2", VAE.Upsample(32, 1))

    class VAE_Head(network.ModuleArgsDict):

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.add_module("Conv", torch.nn.Conv2d(1, 1, 3, 1, 1)) #128 32 32
            self.add_module("Tanh", torch.nn.ReLU()) 

    @config("VAE")
    def __init__(self,
                    optimizer: network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers: network.SchedulersLoader = network.SchedulersLoader(),
                    outputsCriterions: Dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    dim: int = 3) -> None:
        super().__init__(in_channels = 1, init_type="kaiming", optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, dim=dim, nb_batch_per_step=1)
        
        self.add_module("Encoder", VAE.VAE_Encoder(dim))
        self.add_module("Decoder", VAE.VAE_Decoder(dim))
        
        self.add_module("Noise", blocks.HistogramNoise(10, 0.3), in_branch=[1], out_branch=[1])
        self.add_module("UNet_Gradient",  VAE.VAE_UNet(dim), in_branch=[1,0])

        self.add_module("Head", VAE.VAE_Head(dim))
