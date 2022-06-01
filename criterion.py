from typing import Dict, List, Tuple
import torch
import numpy as np
import math
import torchvision

class Loss:

    def __init__(self, criterions, device : str, nb_batch_per_step : int, groupsInput : List[str]) -> None:
        self.criterions = criterions
        if not self.criterions:
            self.criterions = {}
        self.device = device
        self.nb_batch_per_step = nb_batch_per_step
        self.groupsInput = groupsInput

        self.value : List[float]= []
        self.values : Dict[str, List[float]] = dict()
        self.loss = 0
        for group in self.criterions:
            for _, _, criterion in self.criterions[group].values():
                self.values[group+":"+criterion.__class__.__name__] = []
          
    def update(self, out_dict : torch.Tensor, input : torch.Tensor):
        self.loss = torch.zeros((1), requires_grad = True).to(self.device, non_blocking=False)
        data_dict = {group : input[:, i,...] for i, group in enumerate(self.groupsInput)}
        for group in self.criterions:
            output = out_dict[group] if group in out_dict else None
            for true_group, l, criterion in self.criterions[group].values():
                target = torch.unsqueeze(data_dict[true_group].to(self.device, non_blocking=False), 1) if true_group in data_dict else None
                criterion.setDevice(self.device)
                result = criterion(output, target)
                self.loss = self.loss + l*result
                self.values[group+":"+criterion.__class__.__name__].append(result.item())
                
        self.value.append(self.loss.item())
        self.loss = self.loss/self.nb_batch_per_step

    def getLastValue(self):
        return self.value[-1] if self.value else 0 

    def format(self) -> Dict[str, float]:
        result = dict()
        for name in self.values:
            result[name] = np.mean(self.values[name])
        return result

    def mean(self) -> float:
        return np.mean(self.value)
    
    def clear(self) -> None:
        self.value.clear()
        for name in self.values:
            self.values[name].clear()

class Criterion(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.device = None
    
    def setDevice(self, device):
        self.device = device

class Dice(Criterion):
    
    def __init__(self, smooth : float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth
    
    def flatten(self, tensor : torch.Tensor) -> torch.Tensor:
        C = tensor.size(1)
        return tensor.permute((1, 0) + tuple(range(2, tensor.dim()))).contiguous().view(C, -1)

    def dice_per_channel(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = self.flatten(input)
        target = self.flatten(target)
        return (2.*(input * target).sum() + self.smooth)/(input.sum() + target.sum() + self.smooth)

    def forward(self, input: torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        return 1-torch.mean(self.dice_per_channel(torch.nn.functional.softmax(input, dim=1).float(), torch.nn.functional.one_hot(target, input.shape[1]).permute(0, len(target.shape), *[i+1 for i in range(len(target.shape)-1)]).float()))

class GradientImages(Criterion):

    def __init__(self):
        super().__init__()

    @staticmethod
    def _image_gradient(image : torch.Tensor):
        dx = image[:, :, 1:, :, :] - image[:, :, :-1, :, :]
        dy = image[:, :, :, 1:, :] - image[:, :, :, :-1, :]
        dz = image[:, :, :, :, 1:] - image[:, :, :, :, :-1] 
        return dx.pow(2), dy.pow(2), dz.pow(2)

    def forward(self, input: torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        dx, dy, dz = GradientImages._image_gradient(input)
        if target is not None:
            dx_tmp, dy_tmp, dz_tmp = GradientImages._image_gradient(target)
            dx -= dx_tmp
            dy -= dy_tmp
            dz -= dz_tmp
    
        return dx.norm(p=2) + dy.norm(p=2) + dz.norm(p=2)


class BCE(Criterion):

    def __init__(self, target : float = 0) -> None:
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.register_buffer('target', torch.tensor(target).type(torch.float32))
    

    def forward(self, input: torch.Tensor, _ : torch.Tensor) -> torch.Tensor:
        return self.loss(input, self.target.to(self.device).expand_as(input))

# TODO Fix for UNet GAN
class VGGPerceptualLoss(Criterion):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval().to(self.device))
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval().to(self.device))
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval().to(self.device))
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval().to(self.device))
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)).to('cuda')
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)).to('cuda')
        self.resize = resize

    def forward(self, input : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss