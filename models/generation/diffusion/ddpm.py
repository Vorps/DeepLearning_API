from DeepLearning_API.config import config
from DeepLearning_API.networks import network
from DeepLearning_API.HDF5 import ModelPatch
from typing import Union
from DeepLearning_API.networks import blocks
import torch


class Alpha_tilde(torch.nn.Module):

    def __init__(self, T: int, min_beta=10 ** -4, max_beta=0.02) -> None:
        super().__init__()
        self.T = T
        self.min_beta = min_beta
        self.max_beta = max_beta

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        betas = torch.linspace(self.min_beta, self.max_beta, self.T)
        alphas = 1 - betas
        alpha_bars = torch.tensor([torch.prod(alphas[:i + 1]) for i in range(len(alphas))])
        print(alpha_bars)

        eta = torch.randn_like(input)

        #noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * input + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return eta
        
class DDPM(network.Network):

    @config("DDPM")
    def __init__(   self,
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.SchedulersLoader = network.SchedulersLoader(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    patch : Union[ModelPatch, None] = None,
                    dim : int = 3,
                    T: int = 1000) -> None:
        super().__init__(in_channels=1, optimizer=optimizer, schedulers=schedulers, outputsCriterions=outputsCriterions, patch=patch, dim=dim)
        self.add_module("Alpha_tilde", Alpha_tilde(T))
        self.add_module("Conv", torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3))
        

        