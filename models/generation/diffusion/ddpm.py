from DeepLearning_API.config import config
from DeepLearning_API.networks import network
from DeepLearning_API.HDF5 import ModelPatch
from typing import Union
from DeepLearning_API.networks import blocks
import torch



class Alpha_tilde(torch.nn.Module):


    def __init__(self, T: int) -> None:
        super().__init__()
        self.T = T

    def forward(self) -> torch.Tensor:
        self.betas = torch.linspace(10**-4, 0.02, self.T)
        
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))])
        print(self.betas)
        print(self.alphas)
        print(self.alpha_bars)
        return None#torch.cat((torch.sqrt(alpha), torch.sqrt(1-alpha)))

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
        self.add_module("Alpha_tilde", Alpha_tilde(T), in_branch=[], out_branch=[1])
        self.add_module("Multiply_std", blocks.Multiply(), in_branch=[0,1])
        self.add_module("Noise", blocks.NormalNoise(), in_branch=[], out_branch=[2])
        self.add_module("Add_mu", blocks.Add(), in_branch=[1,2])

        self.add_module("Conv", torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3))
        

        