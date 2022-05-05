import torch
import numpy as np
import math

class Criterion(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

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

class DiceVoxelMorph(Dice):

    def __init__(self, smooth : float = 1e-6) -> None:
        super().__init__(smooth)

    def forward(self, input : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        input = torch.nn.functional.one_hot(input.type(torch.int64)).permute(0, len(input.shape), *[i+1 for i in range(len(input.shape)-1)]).float()
        target = torch.nn.functional.one_hot(target.type(torch.int64)).permute(0, len(target.shape), *[i+1 for i in range(len(target.shape)-1)]).float()
        return 1-torch.mean(self.dice_per_channel(input, target))



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


class Grad(Criterion):

    def __init__(self, penalty : str ="l1"):
        super().__init__()
        self.penalty = penalty

    def forward(self, input: torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        dy = torch.abs(target[:, :, 1:, :, :] - target[:, :, :-1, :, :])
        dx = torch.abs(target[:, :, :, 1:, :] - target[:, :, :, :-1, :])
        dz = torch.abs(target[:, :, :, :, 1:] - target[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0
        return grad