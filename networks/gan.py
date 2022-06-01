from audioop import bias
from typing import Dict, List
import torch
from DeepLearning_API import config, Loss
from DeepLearning_API.networks import network, segmentation
import numpy as np

class Discriminator(network.Network):

    @config("Discriminator")
    def __init__(self,
                    optimizer : network.Optimizer = network.Optimizer(),
                    scheduler : network.Scheduler = network.Scheduler(),
                    criterions: Dict[str, network.Criterions] = {"default" : network.Criterions()},
                    dim : int = 3,
                    channels : List[int] = [1, 16, 32, 64, 128, 1],
                    blockConfig : network.BlockConfig = network.BlockConfig()) -> None:
        super().__init__(optimizer = optimizer, scheduler = scheduler, criterions = criterions, dim=dim)
        self.model = torch.nn.Sequential(*[network.ConvBlock(channels[i], channels[i+1], blockConfig, dim=dim) for i in range(len(channels)-1)][:-1])
        #self.head = torch.nn.Linear(channels[-1], 1)

    def forward(self, x : torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"is_real" : self.model(x)}

class Generator(segmentation.UNet):

    @config("Generator")
    def __init__(self, 
                    optimizer : network.Optimizer = network.Optimizer(),
                    scheduler : network.Scheduler = network.Scheduler(),
                    criterions: Dict[str, network.Criterions] = {"default" : network.Criterions()},
                    dim : int = 3,
                    encoder_channels : List[int] = [1, 16, 32, 64],
                    decoder_channels : List[int] = [64, 32, 16],
                    final_channels : List[int] = [16, 1],
                    attention : bool = False,
                    blockConfig : network.BlockConfig = network.BlockConfig()) -> None:

        super().__init__(   optimizer = optimizer, 
                            scheduler = scheduler,
                            criterions = criterions,
                            dim = dim,
                            encoder_channels=encoder_channels,
                            decoder_channels=decoder_channels,
                            final_channels=final_channels,
                            downSampleMode="MAXPOOL",
                            upSampleMode="CONV_TRANSPOSE",
                            attention=attention,
                            blockConfig=blockConfig)
        
    def forward(self, input : torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"fake_B" : super().forward(input)["output"]}
        
    def logImage(self, input : torch.Tensor, output : Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        result = output["fake_B"][0, 0, output["fake_B"].shape[2]//2, ...].detach().cpu().numpy()
        return {"A" : input[0, 0, input.shape[2]//2, ...].cpu().numpy(), "B" : input[0, 1, input.shape[2]//2, ...].cpu().numpy(), "fake_B" : result}

class Gan(network.Network):

    @config("Gan")
    def __init__(self, generator : Generator = Generator(), discriminator : Discriminator = Discriminator()) -> None:
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
    
    def init(self, key : str, loss : Loss) -> None:
        self.generator.init("{}.Generator".format(key), loss)
        self.discriminator.init("{}.Discriminator".format(key), loss)

    def getSubModels(self) -> List[network.Network]:
        return [self.generator, self.discriminator]

    def forward(self, x : torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.generator(x)

    def backward(self, input : torch.Tensor) -> Dict[str, torch.Tensor]:
        output : Dict[str, Dict[str, torch.Tensor]] = {}
        
        A, B = torch.unsqueeze(input[:, 0, ...], 1), torch.unsqueeze(input[:, 1, ...], 1)
        
        D_B = self.discriminator.forward(B)["is_real"]
        fake_B = self.generator.forward(A)["fake_B"]
        D_fake_B = self.discriminator.forward(fake_B.detach())["is_real"]
        
        output[self.discriminator.getName()] = {"D_B" : D_B, "D_fake_B" : D_fake_B}
        self.discriminator.loss.update(output[self.discriminator.getName()], input)
        
        D_fake_B = self.discriminator.forward(fake_B)["is_real"]
        output[self.generator.getName()] = {"fake_B" : fake_B, "D_fake_B" : D_fake_B}
        self.generator.loss.update(output[self.generator.getName()], input)
        
        return output

    def logImage(self, input : torch.Tensor, output : Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        return self.generator.logImage(input, output[self.generator.getName()])

class CycleGanDiscriminator(network.Network):

    @config("CycleGanDiscriminator")
    def __init__(   self, 
                    optimizer : network.Optimizer = network.Optimizer(),
                    scheduler : network.Scheduler = network.Scheduler(),
                    criterions: Dict[str, network.Criterions] = {"default" : network.Criterions()},
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
                    criterions: Dict[str, network.Criterions] = {"default" : network.Criterions()},
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
        
    def init(self, key : str, loss : Loss) -> None:
        self.generator.init("{}.CycleGanGenerator".format(key), loss)
        self.discriminator.init("{}.CycleGanDiscriminator".format(key), loss)

    def getSubModels(self) -> List[network.Network]:
        return [self.generator, self.discriminator]
        
    def forward(self, x : torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.generator(x)

    def backward(self, x : torch.Tensor) -> Dict[str, torch.Tensor]:
        result : Dict[str, Dict[str, torch.Tensor]] = {}          
        result[self.generator.getName()] = self.generator.backward(x, self.discriminator)
        result[self.discriminator.getName()] = self.discriminator.backward(torch.cat((x, result[self.generator.getName()]["fake_A"].detach(), result[self.generator.getName()]["fake_B"].detach()), dim=1))
        return result

    def logImage(self, input : torch.Tensor, output : Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        return self.generator.logImage(input, output[self.generator.getName()])
