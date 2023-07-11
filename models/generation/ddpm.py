from DeepLearning_API.config import config
from DeepLearning_API.networks import network
from DeepLearning_API.HDF5 import ModelPatch
from DeepLearning_API.utils import gpuInfo
from typing import Union, Callable
from DeepLearning_API.networks import blocks
from DeepLearning_API.measure import Criterion
import torch
import tqdm

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class DDPM(network.Network):
    
    class DDPM_TE(torch.nn.Module):

        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.linear_0 = torch.nn.Linear(in_channels, out_channels)
            self.siLU = torch.nn.SiLU()
            self.linear_1 = torch.nn.Linear(out_channels, out_channels)
        
        def forward(self, input: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return input + self.linear_1(self.siLU(self.linear_0(t))).reshape(input.shape[0], -1, *[1 for _ in range(len(input.shape)-2)])
            
    class DDPM_UNetBlock(network.ModuleArgsDict):
        
        def __init__(self, channels: list[int], nb_conv_per_stage: int, blockConfig: blocks.BlockConfig, downSampleMode: blocks.DownSampleMode, upSampleMode: blocks.UpSampleMode, attention : bool, time_embedding_dim: int, dim: int, i : int = 0) -> None:
            super().__init__()
            if i > 0:
                self.add_module(downSampleMode.name, blocks.downSample(in_channels=channels[0], out_channels=channels[1], downSampleMode=downSampleMode, dim=dim))
            self.add_module("Te_down", DDPM.DDPM_TE(time_embedding_dim, channels[1 if downSampleMode == blocks.DownSampleMode.CONV_STRIDE and i > 0 else 0]), in_branch=[0, 1])
            self.add_module("DownConvBlock", blocks.ResBlock(in_channels=channels[1 if downSampleMode == blocks.DownSampleMode.CONV_STRIDE and i > 0 else 0], out_channels=channels[1], nb_conv=nb_conv_per_stage, blockConfig=blockConfig, dim=dim))
            if len(channels) > 2:
                self.add_module("UNetBlock_{}".format(i+1), DDPM.DDPM_UNetBlock(channels[1:], nb_conv_per_stage, blockConfig, downSampleMode, upSampleMode, attention, time_embedding_dim, dim, i+1), in_branch=[0,1])
                self.add_module("Te_up", DDPM.DDPM_TE(time_embedding_dim, (channels[1]+channels[2]) if upSampleMode != blocks.UpSampleMode.CONV_TRANSPOSE else channels[1]*2), in_branch=[0, 1])
                self.add_module("UpConvBlock", blocks.ResBlock(in_channels=(channels[1]+channels[2]) if upSampleMode != blocks.UpSampleMode.CONV_TRANSPOSE else channels[1]*2, out_channels=channels[1], nb_conv=nb_conv_per_stage, blockConfig=blockConfig, dim=dim))
            if i > 0:
                if attention:
                    self.add_module("Attention", blocks.Attention(F_g=channels[1], F_l=channels[0], F_int=channels[0], dim=dim), in_branch=[2, 0], out_branch=[2])
                self.add_module(upSampleMode.name, blocks.upSample(in_channels=channels[1], out_channels=channels[0], upSampleMode=upSampleMode, dim=dim))
                self.add_module("SkipConnection", blocks.Concat(), in_branch=[0, 2])

    class DDPM_UNetHead(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, dim: int) -> None:
            super().__init__()
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1))

    class DDPM_ForwardProcess(torch.nn.Module):

        def __init__(self, noise_step: int=1000, beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
            super().__init__()
            self.betas = torch.linspace(beta_start, beta_end, noise_step)
            self.alphas = 1 - self.betas
            self.alpha_hat = torch.cumprod(1.-self.betas, dim=0)

        def forward(self, input: torch.Tensor, t: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
            alpha_hat_t = self.alpha_hat[t.cpu()].to(input.device).reshape(input.shape[0], *[1 for _ in range(len(input.shape)-1)])
            return alpha_hat_t.sqrt() * input + (1 - alpha_hat_t).sqrt() * eta
    
    class DDPM_SampleT(torch.nn.Module):

        def __init__(self, noise_step: int) -> None:
            super().__init__()
            self.noise_step = noise_step
    
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.randint(0, self.noise_step, (input.shape[0],)).to(input.device)
    
    class DDPM_TimeEmbedding(torch.nn.Module):
        
        def sinusoidal_embedding(noise_step: int, time_embedding_dim: int):
            embedding = torch.zeros(noise_step, time_embedding_dim)
            wk = torch.tensor([1 / 10_000 ** (2 * j / time_embedding_dim) for j in range(time_embedding_dim)])
            wk = wk.reshape((1, time_embedding_dim))
            t = torch.arange(noise_step).reshape((noise_step, 1))
            embedding[:,::2] = torch.sin(t * wk[:,::2])
            embedding[:,1::2] = torch.cos(t * wk[:,::2])
            return embedding

        def __init__(self, noise_step: int=1000, time_embedding_dim: int=100) -> None:
            super().__init__()
            self.time_embed = torch.nn.Embedding(noise_step, time_embedding_dim)
            self.time_embed.weight.data = DDPM.DDPM_TimeEmbedding.sinusoidal_embedding(noise_step, time_embedding_dim)
            self.time_embed.requires_grad_(False)
        
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.time_embed(input)

    class DDPM_UNet(network.ModuleArgsDict):

        def __init__(self, noise_step: int, channels: list[int], blockConfig: blocks.BlockConfig, nb_conv_per_stage: int, downSampleMode: str, upSampleMode: str, attention : bool, time_embedding_dim: int, dim : int) -> None:
            super().__init__()
            self.add_module("t", DDPM.DDPM_TimeEmbedding(noise_step, time_embedding_dim), in_branch=[1], out_branch=["te"])
            self.add_module("UNetBlock_0", DDPM.DDPM_UNetBlock(channels, nb_conv_per_stage, blockConfig, downSampleMode=blocks.DownSampleMode._member_map_[downSampleMode], upSampleMode=blocks.UpSampleMode._member_map_[upSampleMode], attention=attention, time_embedding_dim=time_embedding_dim, dim=dim), in_branch=[0,"te"])
            self.add_module("Head", DDPM.DDPM_UNetHead(in_channels=channels[1], out_channels=1, dim=dim))

    class DDPM_Inference(torch.nn.Module):

        def __init__(self, model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], noise_step: int, beta_start: float, beta_end: float) -> None:
            super().__init__()
            self.model = model
            self.noise_step = noise_step
            self.forwardProcess = DDPM.DDPM_ForwardProcess(noise_step, beta_start, beta_end)
        
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            x = torch.randn_like(input).to(input.device)

            description = lambda : "Inference : "+gpuInfo([input.device])+gpuInfo(input.device)

            t_list = list(range(self.noise_step))[::-1]
            with tqdm.tqdm(iterable = enumerate(t_list), desc = description(), total=len(t_list), leave=False) as batch_iter:
                for _, t in batch_iter:
                    # Estimating noise to be removed
                    time_tensor = (torch.ones(input.shape[0], 1) * t).to(input.device).long()
                    
                    eta_theta = self.model(torch.concat((x, input), dim=1), time_tensor)

                    alpha_t = self.forwardProcess.alphas[t]
                    alpha_t_bar = self.forwardProcess.alpha_hat[t]

                    # Partially denoising the image
                    x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

                    if t > 0:
                        z = torch.randn_like(input).to(input.device)

                        # Option 1: sigma_t squared = beta_t
                        beta_t = self.forwardProcess.betas[t]
                        sigma_t = beta_t.sqrt()

                        # Option 2: sigma_t squared = beta_tilda_t
                        # prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                        # beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                        # sigma_t = beta_tilda_t.sqrt()

                        # Adding some more noise like in Langevin Dynamics fashion
                        x = x + sigma_t * z
                    batch_iter.set_description(description()) 
            return x

    @config("DDPM")
    def __init__(   self,
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.SchedulersLoader = network.SchedulersLoader(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    patch : Union[ModelPatch, None] = None,
                    noise_step: int = 1000,
                    beta_start: float = 1e-4, 
                    beta_end: float = 0.02,
                    time_embedding_dim: int = 100,
                    channels: list[int]=[1, 64, 128, 256, 512, 1024],
                    blockConfig: blocks.BlockConfig = blocks.BlockConfig(),
                    nb_conv_per_stage: int = 2,
                    downSampleMode: str = "MAXPOOL",
                    upSampleMode: str = "CONV_TRANSPOSE",
                    attention : bool = False,
                    dim : int = 3,
                    trainning: bool = False) -> None:
        super().__init__(in_channels=1, optimizer=optimizer, schedulers=schedulers, outputsCriterions=outputsCriterions, patch=patch, dim=dim)
        if trainning:
            self.add_module("Identity", torch.nn.Identity())
            self.add_module("Noise", blocks.NormalNoise(), out_branch=["eta"])
            self.add_module("Sample", DDPM.DDPM_SampleT(noise_step), out_branch=["t"])
            self.add_module("Forward", DDPM.DDPM_ForwardProcess(noise_step, beta_start, beta_end), in_branch=[0, "t", "eta"])
            self.add_module("Concat", blocks.Concat(), in_branch=[0,1])
            self.add_module("UNet", DDPM.DDPM_UNet(noise_step, channels, blockConfig, nb_conv_per_stage, downSampleMode, upSampleMode, attention, time_embedding_dim, dim), in_branch=[0, "t"])
            self.add_module("Noise_optim", blocks.Concat(), in_branch=[0, "eta"])
        else:
            self.add_module("UNet", DDPM.DDPM_UNet(noise_step, channels, blockConfig, nb_conv_per_stage, downSampleMode, upSampleMode, attention, time_embedding_dim, dim), in_branch=[])
            self.add_module("Inference", DDPM.DDPM_Inference(lambda x, t: self._modules["UNet"](x, t), noise_step, beta_start, beta_end), in_branch=[0])

class MSE(Criterion):

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.loss(input[:, 0, ...], input[:, 1, ...])


#model = DDPM()
#model._compute_channels_trace(model, model.in_channels, None)
#print(model)