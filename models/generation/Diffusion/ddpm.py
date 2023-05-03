from DeepLearning_API.config import config
from DeepLearning_API.networks import network
from DeepLearning_API.HDF5 import ModelPatch

class DDPM(network.Network):

    @config("DDPM")
    def __init__(   self,
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.SchedulersLoader = network.SchedulersLoader(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    patch : ModelPatch | None = None,
                    dim : int = 3,
                    noiseStep: int = 1000,
                    beta_schedulers : network.SchedulersLoader = network.SchedulersLoader(),) -> None:
        super().__init__(in_channels=1, optimizer=optimizer, schedulers=schedulers, outputsCriterions=outputsCriterions, patch=patch, dim=dim)
        self.noiseStep = noiseStep
        self.beta_schedulers = beta_schedulers.getShedulers()