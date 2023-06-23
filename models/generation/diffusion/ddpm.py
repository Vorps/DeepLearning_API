from DeepLearning_API.config import config
from DeepLearning_API.networks import network
from DeepLearning_API.HDF5 import ModelPatch
from typing import Union
from DeepLearning_API.networks import blocks
from DeepLearning_API.models.segmentation.uNet import UNetBlock
import torch

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class DDPM(network.Network):
    
    class UNetHead(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, dim: int) -> None:
            super().__init__()
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1))

    class ForwardProcess(network.ModuleArgsDict):

        def __init__(self, noise_step: int=1000, beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
            super().__init__()
            betas = torch.linspace(beta_start, beta_end, noise_step)
            self.alpha_hat = torch.cumprod(1.-betas, dim=0)

        def forward(self, input: torch.Tensor, t: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
            return self.alpha_hat[t].sqrt() * input + (1 - self.alpha_hat[t]).sqrt() * eta
    
    class SampleT(torch.nn.Module):

        def __init__(self, noise_step: int) -> None:
            super().__init__()
            self.noise_step = noise_step
    
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.randint(0, self.noise_step, (input.shape[0],))
    
    @config("DDPM")
    def __init__(   self,
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.SchedulersLoader = network.SchedulersLoader(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    patch : Union[ModelPatch, None] = None,
                    dim : int = 3,
                    noise_step: int = 1000,
                    beta_start: float = 1e-4, 
                    beta_end: float = 0.02,
                    channels: list[int]=[1, 64, 128, 256, 512, 1024],
                    blockConfig: blocks.BlockConfig = blocks.BlockConfig(),
                    nb_conv_per_stage: int = 2,
                    downSampleMode: str = "MAXPOOL",
                    upSampleMode: str = "CONV_TRANSPOSE",
                    attention : bool = False) -> None:
        super().__init__(in_channels=1, optimizer=optimizer, schedulers=schedulers, outputsCriterions=outputsCriterions, patch=patch, dim=dim)
        self.add_module("Identity", torch.nn.Identity())
        self.add_module("Noise", blocks.NormalNoise(), out_branch=["eta"])
        self.add_module("Sample", DDPM.SampleT(noise_step), out_branch=["t"])
        self.add_module("Forward", DDPM.ForwardProcess(noise_step, beta_start, beta_end), in_branch=[0, "t", "eta"])


        self.add_module("UNetBlock_0", UNetBlock(channels, nb_conv_per_stage, blockConfig, downSampleMode=blocks.DownSampleMode._member_map_[downSampleMode], upSampleMode=blocks.UpSampleMode._member_map_[upSampleMode], attention=attention, dim=dim))
        self.add_module("Head", DDPM.UNetHead(in_channels=channels[1], out_channels=channels[0], dim=dim))
        

        