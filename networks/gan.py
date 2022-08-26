from typing import Dict, List
import torch
from DeepLearning_API.config import config
from DeepLearning_API.networks import network, blocks
from DeepLearning_API.measure import TargetCriterionsLoader
from DeepLearning_API.dataset import Patch

import numpy as np

class Discriminator(network.Network):

    @config("Discriminator")
    def __init__(self,
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.SchedulersLoader = network.SchedulersLoader(),
                    outputsCriterions: Dict[str, TargetCriterionsLoader] = {"default" : TargetCriterionsLoader()},
                    dim : int = 3,
                    in_channels : int = 1) -> None:
        super().__init__(optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, dim=dim)
        
        ndf = 64
        n_layers=3
        norm_layer = blocks.getTorchModule("BatchNorm", dim)
        kernel_size = 4
        padding = 1
        bias = False
        
        self.add_module("Conv_0", blocks.getTorchModule("Conv", dim)(in_channels=in_channels, out_channels=ndf, kernel_size=kernel_size, stride=2, padding=padding))
        self.add_module("LeakyReLU_0", torch.nn.LeakyReLU(0.2, True))
        
        channels = [ndf*min(2 ** i, 8) for i in range(n_layers+1)]
        for i, (in_channels, out_channels, stride) in enumerate(zip(channels, channels[1:], [2]*(len(channels)-2)+[1])):
            self.add_module("Conv_{}".format(i+1), blocks.getTorchModule("Conv", dim)(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
            self.add_module("Norm_{}".format(i+i), norm_layer(out_channels))
            self.add_module("LeakyReLU_{}".format(i+i), torch.nn.LeakyReLU(0.2, True))
        
        self.add_module("Head", blocks.getTorchModule("Conv", dim)(in_channels=channels[-1], out_channels=1, kernel_size=4, stride=1, padding=1))

class Generator(network.Network):

    class GeneratorStem(network.ModuleArgsDict):

        def __init__(self, in_channel: int, out_channel: int, norm_layer : torch.nn.Module, bias: bool, dim: int) -> None:
            super().__init__()
            self.add_module("Reflection", blocks.getTorchModule("ReflectionPad", dim)(3))
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channel, out_channel, kernel_size=7, padding=0, bias=bias))
            self.add_module("Norm", norm_layer(out_channel))
            self.add_module("Relu", torch.nn.ReLU(True)) 

    class GeneratorHead(network.ModuleArgsDict):

        def __init__(self, channel: int, dim: int) -> None:
            super().__init__()
            self.add_module("Reflection", blocks.getTorchModule("ReflectionPad", dim)(3))
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(channel, 1, kernel_size=7, padding=0))
            self.add_module("Tanh", torch.nn.Tanh())

    class GeneratorDownSample(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, norm_layer: torch.nn.Module, bias: bool, dim: int) -> None:
            super().__init__()
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=bias))
            self.add_module("Norm", norm_layer(out_channels))
            self.add_module("Relu", torch.nn.ReLU(True))

    class GeneratorResnetBlock(network.ModuleArgsDict):

        def __init__(self, channel : int, padding_type : str, norm_layer : torch.nn.Module, dropout : bool, bias : bool, dim : int):
            super().__init__()
            padding = 0 if padding_type != 'zero' else 1
            if padding == 0:
                self.add_module("Padding_0", blocks.getTorchModule("{}Pad".format(padding_type), dim)(1))

            self.add_module("Conv_0", blocks.getTorchModule("Conv", dim)(channel, channel, kernel_size=3, padding=padding, bias=bias))
            self.add_module("Norm_0", norm_layer(channel))
            self.add_module("Relu_0", torch.nn.ReLU(True))
            if dropout:
                self.add_module("Dropout", torch.nn.Dropout(0.5))
            
            if padding == 0:
                self.add_module("Padding_1", blocks.getTorchModule("{}Pad".format(padding_type), dim)(1))

            self.add_module("Conv_1", blocks.getTorchModule("Conv", dim)(channel, channel, kernel_size=3, padding=padding, bias=bias))
            self.add_module("Norm_1", norm_layer(channel))
            self.add_module("Residual", blocks.Concat(), in_branch=[0,1])


    class GeneratorUpSample(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, norm_layer: torch.nn.Module, bias: bool, dim: int) -> None:
            super().__init__()
            self.add_module("ConvTranspose", blocks.getTorchModule("ConvTranspose", dim)(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias))
            self.add_module("Norm", norm_layer(out_channels))
            self.add_module("Relu", torch.nn.ReLU(True))

    @config("Generator")
    def __init__(self, 
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.SchedulersLoader = network.SchedulersLoader(),
                    patch : Patch = Patch(),
                    outputsCriterions: Dict[str, TargetCriterionsLoader] = {"default" : TargetCriterionsLoader()},
                    in_channel : int = 1,
                    dim : int = 3) -> None:
        super().__init__(optimizer=optimizer, schedulers=schedulers, patch=patch, outputsCriterions=outputsCriterions)
        ngf=64
        norm_layer = blocks.getTorchModule("BatchNorm", dim)
        dropout = False
        n_blocks=6
        padding_type="Reflection"
        bias = False
        channels = [ngf*2**i for i in range(3)]
        
        self.add_module("Stem", Generator.GeneratorStem(in_channel=in_channel, out_channel=ngf, norm_layer=norm_layer, bias=bias, dim=dim))

        for i, (in_channels, out_channels) in enumerate(zip(channels, channels[1:])):
            self.add_module("DownSample_{}".format(i), Generator.GeneratorDownSample(in_channels=in_channels, out_channels=out_channels, norm_layer=norm_layer, bias=bias, dim=dim))

        for i in range(n_blocks):
            self.add_module("ResnetBlock_{}".format(i), Generator.GeneratorResnetBlock(channel=channels[-1], padding_type=padding_type, norm_layer=norm_layer, dropout=dropout, bias=bias, dim=dim))

        for i, (in_channels, out_channels) in enumerate(zip(reversed(channels), reversed(channels[:-1]))):
            self.add_module("UpSample_{}".format(i), Generator.GeneratorUpSample(in_channels=in_channels, out_channels=out_channels, norm_layer=norm_layer, bias=bias, dim=dim))

        self.add_module("Head", Generator.GeneratorHead(channel=ngf, dim=dim))

class Gan(network.Network):

    @config("Gan")
    def __init__(self, generator : Generator = Generator(), discriminator : Discriminator = Discriminator()) -> None:
        super().__init__()
        self.add_module("Generator", generator)

        self.add_module("Discriminator", discriminator)
        
    def forward(self, data_dict : Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        A, B = list(data_dict.values())
        fake_B = self["Generator"]({"A" : A})["out"]

        D_B = self["Discriminator"]({"B" : B})["out"]
        D_D_fake_B = self["Discriminator"]({"fake_B" : fake_B.detach()})["out"]
        
        self["Discriminator"].requires_grad_(False)
        G_D_fake_B = self["Discriminator"]({"fake_B" : fake_B})["out"]
        self["Discriminator"].requires_grad_(True)
        
        
        
        return {"fake_B" : fake_B, "G_D_fake_B" : G_D_fake_B, "D_B" : D_B, "D_D_fake_B" : D_D_fake_B}

    def logImage(self, data_dict : Dict[str, torch.Tensor], output : Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        fake_B = output["fake_B"][0, 0, output["fake_B"].shape[2]//2, ...].detach().cpu().numpy()
        A, B = list(data_dict.values())
        A = A[0, 0, A.shape[2]//2, ...].cpu().numpy()
        B = B[0, 0, B.shape[2]//2, ...].cpu().numpy()
        return {"A" : A, "B" : B, "fake_B" : fake_B, "Diff" : A-fake_B}

"""class CycleGanDiscriminator(network.Network):

    @config("CycleGanDiscriminator")
    def __init__(   self, 
                    optimizer : network.Optimizer = network.Optimizer(),
                    scheduler : network.Scheduler = network.Scheduler(),
                    criterions: Dict[str, CriterionsLoader] = {"default" : CriterionsLoader()},
                    dim : int = 3,
                    channels : List[int] = [1, 16, 32, 64, 128, 1],
                    blockConfig : network.BlockConfig = network.BlockConfig()) -> None:
        super().__init__(optimizer = optimizer, scheduler = scheduler, criterions = criterions, dim = dim)
        self.discriminator_1, self.discriminator_2 = [Discriminator(optimizer, scheduler, criterions, dim, channels, blockConfig) for _ in range(2)]

    def forward(self, _ : torch.Tensor) -> Dict[str, torch.Tensor]:
        return {}

    def backward(self, input : torch.Tensor) -> Dict[str, torch.Tensor]:
        network.set_requires_grad([self], True)
        
        A = torch.unsqueeze(input[:,0,...], 1)
        B = torch.unsqueeze(input[:,1,...], 1)
        fake_A = torch.unsqueeze(input[:,2,...], 1)
        fake_B = torch.unsqueeze(input[:,3,...], 1)

        D_fake_A = self.discriminator_2.backward(fake_A)["is_real"]
        D_A = self.discriminator_2.backward(A)["is_real"]

        D_fake_B = self.discriminator_1.backward(fake_B)["is_real"]
        D_B = self.discriminator_1.backward(B)["is_real"]

        out = {"D_A" : D_A, "D_B" : D_B, "D_fake_A" : D_fake_A, "D_A" : D_A, "D_fake_B" : D_fake_B, "D_B" : D_B}

        self.loss.update(out, input)

        return {}

class CycleGanGenerator(network.Network):

    @config("CycleGanGenerator")
    def __init__(self, 
                    optimizer : network.Optimizer = network.Optimizer(),
                    scheduler : network.Scheduler = network.Scheduler(),
                    criterions: Dict[str, CriterionsLoader] = {"default" : CriterionsLoader()},
                    dim : int = 3,
                    encoder_channels : List[int] = [1, 16, 32, 64],
                    decoder_channels : List[int] = [64, 32, 16],
                    final_channels : List[int] = [],
                    attention : bool = False,
                    blockConfig : network.BlockConfig = network.BlockConfig()) -> None:
        super().__init__(optimizer = optimizer, scheduler = scheduler, criterions = criterions, dim = dim)

        self.generator_1, self.generator_2 = [Generator(optimizer=None,
                                                        scheduler=None,
                                                        criterions=None,
                                                        dim=dim,
                                                        encoder_channels=encoder_channels,
                                                        decoder_channels=decoder_channels,
                                                        final_channels=final_channels,
                                                        attention=attention,
                                                        blockConfig=blockConfig) for _ in range(2)]

    def forward(self, x : torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"fake_A" : self.generator_2(torch.unsqueeze(x[:,0,...], 1))["fake"], "fake_B" : self.generator_1(torch.unsqueeze(x[:,1,...], 1))["fake"]}

    def backward(self, input : torch.Tensor, cycleGanDiscriminator : CycleGanDiscriminator) -> Dict[str, torch.Tensor]:
        network.set_requires_grad([cycleGanDiscriminator], False)
        network.set_requires_grad([self], True)

        A = torch.unsqueeze(input[:,0,...], 1)
        B = torch.unsqueeze(input[:,1,...], 1)

        generator_1 = self.generator_1.backward(B, cycleGanDiscriminator.discriminator_1)
        generator_2 = self.generator_2.backward(A, cycleGanDiscriminator.discriminator_2)

        fake_B = generator_1["fake"]
        fake_A = generator_2["fake"]
        
        D_fake_B = generator_1["is_real"]
        D_fake_A = generator_2["is_real"]

        cyc_A = self.generator_1.backward(fake_A)["fake"]
        cyc_B = self.generator_2.backward(fake_B)["fake"]
        
        out = {"fake_A" : fake_A, "D_fake_A" : D_fake_A, "cyc_A" : cyc_A, "fake_B" : fake_B, "D_fake_B" : D_fake_B, "cyc_B" : cyc_B}
        
        self.loss.update(out[self.getName()] if len(self.getSubModels()) > 1 else out, input)
        return {"fake_A" : fake_A, "cyc_A" : cyc_A, "fake_B" : fake_B, "cyc_B" : cyc_B}

    def logImage(self, input : torch.Tensor, output : Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        A = input[0,0, input.shape[2]//2, ...].cpu().numpy()
        B = input[0,1, input.shape[2]//2, ...].cpu().numpy()

        cyc_A = output["cyc_A"][0,0, output["cyc_A"].shape[2]//2, ...].detach().cpu().numpy()
        cyc_B = output["cyc_B"][0,0, output["cyc_B"].shape[2]//2, ...].detach().cpu().numpy()

        A_to_B = self.generator_2.logImage(input[:,0, ...], {"fake" : output["fake_B"]}).values()["fake"]
        B_to_A = self.generator_1.logImage(input[:,1, ...], {"fake" : output["fake_A"]}).values()["fake"]

        diff_A = A-cyc_A
        diff_B = B-cyc_B

        return {"B/real_B" : B, "B/fake_B" : A_to_B, "A/real_A" : A, "A/fake_A" : B_to_A, "A/cyc_A" : diff_A, "B/cyc_B" : diff_B}

class CycleGan(network.Network):

    @config("CycleGan")
    def __init__(self, generator : CycleGanGenerator = CycleGanGenerator(), discriminator : CycleGanDiscriminator = CycleGanDiscriminator()) -> None:
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        
    def init(self, key : str, loss : Loss, device : torch.device) -> None:
        self.generator.init("{}.CycleGanGenerator".format(key), loss, device)
        self.discriminator.init("{}.CycleGanDiscriminator".format(key), loss, device)

    def getSubModels(self) -> List[network.Network]:
        return [self.generator, self.discriminator]
        
    def forward(self, x : torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.generator(x)

    def backward(self, data_dict : Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input = self.getInput(data_dict)
        result : Dict[str, Dict[str, torch.Tensor]] = {}          
        result[self.generator.getName()] = self.generator.backward(input, self.discriminator)
        result[self.discriminator.getName()] = self.discriminator.backward(torch.cat((input, result[self.generator.getName()]["fake_A"].detach(), result[self.generator.getName()]["fake_B"].detach()), dim=1))
        return result

    def logImage(self, input : torch.Tensor, output : Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        return self.generator.logImage(input, output[self.generator.getName()])
"""