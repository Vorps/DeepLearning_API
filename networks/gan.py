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
                    dim : int = 3) -> None:

        super().__init__(optimizer = optimizer, scheduler = scheduler, criterions = criterions, dim=dim)
        
        conv = network.getTorchModule("Conv", dim)

        convs : List[torch.nn.Conv3d]= [
            conv(in_channels = 1, out_channels = 16, kernel_size = 4, stride = 2, padding = 1),
            conv(in_channels = 16, out_channels = 32, kernel_size = 4, stride = 2, padding = 1),
            conv(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2, padding = 1),
            conv(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1),
            conv(in_channels = 128, out_channels = 1, kernel_size = 4, stride = 2, padding = 1)]

        args = []
        for conv in convs[:-1]:
            args.append(conv)
            args.append(torch.nn.BatchNorm3d(conv.out_channels))
            args.append(torch.nn.LeakyReLU(0.02))
        
        args.append(convs[-1])
        #args.append(torch.nn.BatchNorm3d(convs[-1].out_channels))
        #args.append(torch.nn.Sigmoid())
        
        self.conv = torch.nn.Sequential(*args)

    def forward(self, _ : torch.Tensor) -> Dict[str, torch.Tensor]:
        return {}

    def backward(self, x : torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"is_real" : self.conv(x)}

class Generator(network.Network):

    @config("Generator")
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
        self.unet = segmentation.UNet(  optimizer=None,
                                        scheduler=None,
                                        criterions=None,
                                        dim=dim,
                                        encoder_channels=encoder_channels,
                                        decoder_channels=decoder_channels,
                                        final_channels=final_channels,
                                        downSampleMode="MAXPOOL",
                                        upSampleMode="CONV_TRANSPOSE",
                                        attention=attention,
                                        blockConfig=blockConfig)
        self.model = torch.nn.Sequential(*[network.ResBlock(decoder_channels[-1], decoder_channels[-1], blockConfig, dim) for _ in range(2)], 
            network.getTorchModule("Conv", dim)(in_channels = decoder_channels[-1], out_channels = 1, kernel_size = 3, stride = 1, padding = 1))

    def forward(self, x : torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"fake" : self.model(self.unet(x))}

    def backward(self, x : torch.Tensor, discriminator : Discriminator = None):
        fake_image = self.model(self.unet(x))
        result = {"fake" : fake_image}
        if discriminator is not None:
            result.update(discriminator.backward(fake_image))
        return result

    def logImage(self, _ : torch.Tensor, output : Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        result = output["fake"][0, 0, output["fake"].shape[2]//2, ...].detach().cpu().numpy()
        return {"fake" : self._logImageNormalize(result)}



class CycleGanDiscriminator(network.Network):

    @config("CycleGanDiscriminator")
    def __init__(self, optimizer : network.Optimizer = network.Optimizer(), scheduler : network.Scheduler = network.Scheduler(), criterions: Dict[str, network.Criterions] = {"default" : network.Criterions()}, dim : int = 3) -> None:
        super().__init__(optimizer = optimizer, scheduler = scheduler, criterions = criterions, dim = dim)
        self.discriminator_1, self.discriminator_2 = [Discriminator(optimizer, scheduler, criterions, dim) for _ in range(2)]

    def forward(self, _ : torch.Tensor) -> Dict[str, torch.Tensor]:
        return {}

    def backward(self, input : torch.Tensor):
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

        self.loss.update(out[self.getName()] if len(self.getSubModels()) > 1 else out, input)

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
        return {"fake_A" : self.generator_2(torch.unsqueeze(x[:,0,...], 1))["fake"]}

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

        A_to_B = list(self.generator_2.logImage(input[:,0, ...], {"fake" : output["fake_B"]}).values())[0]
        B_to_A = list(self.generator_1.logImage(input[:,1, ...], {"fake" : output["fake_A"]}).values())[0]

        diff_A = A-cyc_A
        diff_B = B-cyc_B
        return {"B/real_B" : self._logImageNormalize(B), "B/fake_B" : A_to_B, "A/real_A" : self._logImageNormalize(A), "A/fake_A" : B_to_A, "A/cyc_A" : self._logImageNormalize(diff_A), "B/cyc_B" : self._logImageNormalize(diff_B)}
                        
    
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
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.generator(x)

    def backward(self, x : torch.Tensor) -> torch.Tensor:
        result = {}          
        result[self.generator.getName()] = self.generator.backward(x, self.discriminator)
        result[self.discriminator.getName()] = self.discriminator.backward(torch.cat((x, result[self.generator.getName()]["fake_A"].detach(), result[self.generator.getName()]["fake_B"].detach()), dim=1))
        return result

    def logImage(self, input : torch.Tensor, output : Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        return self.generator.logImage(input, output[self.generator.getName()])

"""   def backward_G(self):
        # GAN loss D_A(G_A(A))
        D_fake_A = self.netD_A(self.fake_B)
        # GAN loss D_B(G_B(B))
        D_fake_B = self.netD_B(self.fake_A)
        
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def backward_D_basic(self, netD, real, fake):

        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)"""