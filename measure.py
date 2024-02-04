from abc import ABC
import importlib
import numpy as np
import torch

import torch.nn.functional as F
import os

from DeepLearning_API.config import config
from DeepLearning_API.utils import _getModule
from DeepLearning_API.networks.blocks import LatentDistribution
from DeepLearning_API.networks.network import ModelLoader, Network
from typing import Callable, Union
from functools import partial
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import copy

modelsRegister = {}

class Criterion(torch.nn.Module, ABC):

    def __init__(self) -> None:
        super().__init__()

    def init(self, model : torch.nn.Module, output_group : str, target_group : str) -> str:
        return output_group

class MaskedLoss(Criterion):

    def __init__(self, loss : Callable[[torch.Tensor, torch.Tensor], torch.Tensor], mode_image_masked: bool) -> None:
        super().__init__()
        self.loss = loss
        self.mode_image_masked = mode_image_masked

    def getMask(self, targets: list[torch.Tensor]) -> torch.Tensor:
        result = None
        if len(targets) > 0:
            result = targets[0]
            for mask in targets[1:]:
                result = result*mask
        return result

    def forward(self, input1: torch.Tensor, *target : list[torch.Tensor]) -> torch.Tensor:
        loss = torch.tensor(0, dtype=torch.float32).to(input1.device)
        for batch in range(input1.shape[0]):
            mask = self.getMask(target[1:])
            if mask is not None:
                if self.mode_image_masked:
                    for i in torch.unique(mask):
                        if i != 0:
                            loss += self.loss(input1[batch, ...]*torch.where(mask == i, 1, 0), target[0][batch, ...]*torch.where(mask == i, 1, 0))
                else:
                    for i in torch.unique(mask):
                        if i != 0:
                            loss += self.loss(torch.masked_select(input1[batch, ...], mask[batch, ...] == i), torch.masked_select(target[0][batch, ...], mask[batch, ...] == i))
            else:
                loss += self.loss(input1[batch, ...], target[0][batch, ...])
        return loss/input1.shape[0]
    
class MSE(MaskedLoss):

    def _loss(reduction: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.MSELoss(reduction=reduction)(x, y)

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(partial(MSE._loss, reduction), False)

class MAE(MaskedLoss):

    def _loss(reduction: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.L1Loss(reduction=reduction)(x, y)
    
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(partial(MAE._loss, reduction), False)

class PSNR(MaskedLoss):

    def _loss(dynamic_range: Union[float, None], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return peak_signal_noise_ratio(x[0].detach().cpu().numpy(), y[0].cpu().numpy(), data_range=dynamic_range if dynamic_range else (y.max()-y.min()).cpu().numpy())
    
    def __init__(self, dynamic_range: Union[float, None] = None) -> None:
        super().__init__(partial(PSNR._loss, dynamic_range), False)
    
class SSIM(MaskedLoss):
    
    def _loss(dynamic_range: Union[float, None], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return structural_similarity(x[0].detach().cpu().numpy(), y[0].cpu().numpy(), data_range=dynamic_range if dynamic_range else (y.max()-y.min()).cpu().numpy())
    
    def __init__(self, dynamic_range: Union[float, None] = None) -> None:
        super().__init__(partial(SSIM._loss, dynamic_range), True)

class DistanceLoss(Criterion):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input1: torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        return torch.mean(input1[:,1:]*target)
        
class Dice(Criterion):
    
    def __init__(self, smooth : float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth
    
    def flatten(self, tensor : torch.Tensor) -> torch.Tensor:
        return tensor.permute((1, 0) + tuple(range(2, tensor.dim()))).contiguous().view(tensor.size(1), -1)

    def dice_per_channel(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = self.flatten(input)
        target = self.flatten(target)
        return (2.*(input * target).sum() + self.smooth)/(input.sum() + target.sum() + self.smooth)

    def forward(self, input1: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = F.one_hot(target.type(torch.int64), num_classes=input1.shape[1]).permute(0, len(target.shape), *[i+1 for i in range(len(target.shape)-1)]).float().squeeze(2)
        #input1 = F.one_hot(input1.type(torch.int64)).permute(0, len(input1.shape), *[i+1 for i in range(len(input1.shape)-1)]).float().squeeze(2)
        return 1-torch.mean(self.dice_per_channel(input1, target))

class GradientImages(Criterion):

    def __init__(self):
        super().__init__()
    
    @staticmethod
    def _image_gradient2D(image : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dx = image[:, :, 1:, :] - image[:, :, :-1, :]
        dy = image[:, :, :, 1:] - image[:, :, :, :-1]
        return dx, dy

    @staticmethod
    def _image_gradient3D(image : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dx = image[:, :, 1:, :, :] - image[:, :, :-1, :, :]
        dy = image[:, :, :, 1:, :] - image[:, :, :, :-1, :]
        dz = image[:, :, :, :, 1:] - image[:, :, :, :, :-1]
        return dx, dy, dz
        
    def forward(self, input: torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        if len(input.shape) == 5:
            dx, dy, dz = GradientImages._image_gradient3D(input)
            if target is not None:
                dx_tmp, dy_tmp, dz_tmp = GradientImages._image_gradient3D(target)
                dx -= dx_tmp
                dy -= dy_tmp
                dz -= dz_tmp
            return dx.norm() + dy.norm() + dz.norm()
        else:
            dx, dy = GradientImages._image_gradient2D(input)
            if target is not None:
                dx_tmp, dy_tmp = GradientImages._image_gradient2D(target)
                dx -= dx_tmp
                dy -= dy_tmp
            return dx.norm() + dy.norm()
        
class BCE(Criterion):

    def __init__(self, target : float = 0) -> None:
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.register_buffer('target', torch.tensor(target).type(torch.float32))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        target = self._buffers["target"]
        return self.loss(input, target.to(input.device).expand_as(input))

class PatchGanLoss(Criterion):

    def __init__(self, target : float = 0) -> None:
        super().__init__()
        self.loss = torch.nn.MSELoss()
        self.register_buffer('target', torch.tensor(target).type(torch.float32))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        target = self._buffers["target"]
        return self.loss(input, (torch.ones_like(input)*target).to(input.device))

class WGP(Criterion):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, gradient_norm: torch.Tensor, _ : torch.Tensor) -> torch.Tensor:
        return torch.mean((gradient_norm - 1)**2)

class Gram(Criterion):

    def computeGram(input : torch.Tensor):
        (b, ch, w) = input.size()
        features = input
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t).div(ch*w)
        return gram

    def __init__(self) -> None:
        super().__init__()
        self.loss = torch.nn.L1Loss(reduction='sum')

    def forward(self, input : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        return self.loss(Gram.computeGram(input), Gram.computeGram(target))

class MedPerceptualLoss(Criterion):
    
    class Module():
        
        @config(None)
        def __init__(self, losses: dict[str, float] = {"Gram": 1, "torch_nn_L1Loss": 1}) -> None:
            self.losses = losses
            self.DL_args = os.environ['DEEP_LEARNING_API_CONFIG_PATH'] if "DEEP_LEARNING_API_CONFIG_PATH" in os.environ else ""

        def getLoss(self) -> dict[torch.nn.Module, float]:
            result: dict[torch.nn.Module, float] = {}
            for loss, l in self.losses.items():
                module, name = _getModule(loss, "measure")
                result[config(self.DL_args)(getattr(importlib.import_module(module), name))(config=None)] = l   
            return result
        
    def __init__(self, modelLoader : ModelLoader = ModelLoader(), path_model : str = "name", modules : dict[str, Module] = {"UNetBlock_0.DownConvBlock.Activation_1": Module({"Gram": 1, "torch_nn_L1Loss": 1})}, shape: list[int] = [128, 256, 256]) -> None:
        super().__init__()
        self.path_model = path_model
        if self.path_model not in modelsRegister:
            self.model = modelLoader.getModel(train=False, DL_args=os.environ['DEEP_LEARNING_API_CONFIG_PATH'].split("MedPerceptualLoss")[0]+"MedPerceptualLoss.Model", DL_without=["optimizer", "schedulers", "nb_batch_per_step", "init_type", "init_gain", "outputsCriterions", "drop_p"])
            if path_model.startswith("https"):
                state_dict = torch.hub.load_state_dict_from_url(path_model)
                state_dict = {"Model": {self.model.getName() : state_dict["model"]}}
            else:
                state_dict = torch.load(path_model)
            self.model.load(state_dict)
            modelsRegister[self.path_model] = self.model
        else:
            self.model = modelsRegister[self.path_model]

        self.shape = shape
        self.mode = "trilinear" if  len(shape) == 3 else "bilinear"
        self.modules_loss: dict[str, dict[torch.nn.Module, float]] = {}
        for name, losses in modules.items():
            self.modules_loss[name.replace(":", ".")] = losses.getLoss()

        self.model.eval()
        self.model.requires_grad_(False)
        self.models: dict[int, torch.nn.Module] = {}

    def preprocessing(self, input: torch.Tensor) -> torch.Tensor:
        if not all([input.shape[-i-1] == size for i, size in enumerate(reversed(self.shape[2:]))]):
            input = F.interpolate(input, mode=self.mode, size=tuple(self.shape), align_corners=False).type(torch.float32)
        return input
    
    def _compute(self, input: torch.Tensor, targets: list[torch.Tensor]) -> torch.Tensor:
        loss = torch.zeros((1), requires_grad = True).to(input.device, non_blocking=False).type(torch.float32)
        input = self.preprocessing(input)
        targets = [self.preprocessing(target) for target in targets]
        for zipped_input in zip([input], *[[target] for target in targets]):
            input = zipped_input[0]
            targets = zipped_input[1:]
           
            for zipped_layers in list(zip(self.models[input.device.index].get_layers([input], set(self.modules_loss.keys()).copy()), *[self.models[input.device.index].get_layers([target], set(self.modules_loss.keys()).copy()) for target in targets])):
                input_layer = zipped_layers[0][1].view(zipped_layers[0][1].shape[0], zipped_layers[0][1].shape[1], int(np.prod(zipped_layers[0][1].shape[2:])))
                for (loss_function, l), target_layer in zip(self.modules_loss[zipped_layers[0][0]].items() , zipped_layers[1:]):
                    target_layer = target_layer[1].view(target_layer[1].shape[0], target_layer[1].shape[1], int(np.prod(target_layer[1].shape[2:])))
                    loss = loss+l*loss_function(input_layer.float(), target_layer.float())/input_layer.shape[0]
        return loss
    
    def forward(self, input : torch.Tensor, *targets : torch.Tensor) -> torch.Tensor:
        if input.device.index not in self.models:
            del os.environ["device"]
            self.models[input.device.index] = Network.to(copy.deepcopy(self.model).eval(), input.device.index)

        loss = torch.zeros((1), requires_grad = True).to(input.device, non_blocking=False).type(torch.float32)
        if len(input.shape) == 5 and len(self.shape) == 2:
            for i in range(input.shape[2]):
                loss = loss + self._compute(input[:, :, i, ...], [t[:, :, i, ...] for t in targets])/input.shape[2]
        else:
            loss = self._compute(input, targets)
        return loss.to(input)

class KLDivergence(Criterion):
    
    def __init__(self, shape: list[int], dim : int = 100, mu : float = 0, std : float = 1) -> None:
        super().__init__()
        self.latentDim = dim
        self.mu = torch.Tensor([mu])
        self.std = torch.Tensor([std])
        self.modelDim = 3
        self.shape = shape
        self.loss = torch.nn.KLDivLoss()
        
    def init(self, model : Network, output_group : str, target_group : str) -> str:
        super().init(model, output_group, target_group)
        model._compute_channels_trace(model, model.in_channels, None, None)

        last_module = model
        for name in output_group.split(".")[:-1]:
            last_module = last_module[name]

        modules = last_module._modules.copy()
        last_module._modules.clear()
        
        for name, value in modules.items():
            last_module._modules[name] = value
            if name == output_group.split(".")[-1]:
                last_module.add_module("LatentDistribution", LatentDistribution(shape = self.shape, latentDim=self.latentDim))
        return ".".join(output_group.split(".")[:-1])+".LatentDistribution.Concat"

    def forward(self, input : torch.Tensor) -> torch.Tensor:
        mu = input[:, 0, :]
        log_std = input[:, 1, :]
        return torch.mean(-0.5 * torch.sum(1 + log_std - mu**2 - torch.exp(log_std), dim = 1), dim = 0)

    """
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        mu = input[:, 0, :]
        log_std = input[:, 1, :]

        z = input[:, 2, :]

        q = torch.distributions.Normal(mu, log_std)

        target_mu = torch.ones((self.latentDim)).to(input.device)*self.mu.to(input.device)
        target_std = torch.ones((self.latentDim)).to(input.device)*self.std.to(input.device)

        p = torch.distributions.Normal(target_mu, target_std)
        
        log_pz = p.log_prob(z)
        log_qzx = q.log_prob(z)

        kl = (log_pz - log_qzx)
        kl = kl.sum(-1)
        return kl
    """
    
class Accuracy(Criterion):

    def __init__(self) -> None:
        super().__init__()
        self.n : int = 0
        self.corrects = torch.zeros((1))

    def forward(self, input : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        self.n += input.shape[0]
        self.corrects += (torch.argmax(torch.softmax(input, dim=1), dim=1) == target).sum().float().cpu()
        return self.corrects/self.n

class NCC(Criterion):

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

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
        conv_fn = getattr(F, 'conv%dd' % ndims)

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

class GradientPenalty(Criterion):

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()