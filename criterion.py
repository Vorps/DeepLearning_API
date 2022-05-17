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
                result = criterion(output, target)
                self.values[group+":"+criterion.__class__.__name__].append(result.item())
                self.loss = self.loss + l*result
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
        #self.device = device

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

class NCC(Criterion):

    def __init__(self, win=None) -> None:
        super().__init__()
        self.win = win

    def forward(self, input: torch.Tensor, target : torch.Tensor) -> torch.Tensor:

        Ii = input
        Ji = target

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(torch.nn, 'Conv%dd' % ndims)
        torch.nn.Conv2d()
        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class MSE(Criterion):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        return torch.mean((input - target) ** 2)

class L1(Criterion):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.l1_loss(input, target)


class GradientImages(Criterion):

    def __init__(self):
        super().__init__()

    @staticmethod
    def _image_gradient(image):
        dx = torch.abs(image[:, :, 1:, :, :] - image[:, :, :-1, :, :])
        dy = torch.abs(image[:, :, :, 1:, :] - image[:, :, :, :-1, :])
        dz = torch.abs(image[:, :, :, :, 1:] - image[:, :, :, :, :-1]) 
        return dx, dy, dz

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
        return self.loss(input, self.target.to("cuda:{}".format(input.get_device())).expand_as(input))


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval().to('cuda'))
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval().to('cuda'))
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval().to('cuda'))
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval().to('cuda'))
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)).to('cuda')
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)).to('cuda')
        self.resize = resize

    def forward(self, input, target):
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