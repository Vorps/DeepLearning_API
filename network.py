import importlib
import torch
from abc import ABC, abstractmethod
import numpy as np
from typing import List
from DeepLearning_API import config

class Network(torch.nn.Module, ABC):

    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def logImage(self, out : torch.Tensor) -> np.ndarray:
        return None

    @abstractmethod
    def getOutputModel(self, group : str, out : torch.Tensor) -> np.ndarray:
        return None

def getTorchModule(name_fonction : str, dim : int = None) -> torch.nn.Module:
    return getattr(importlib.import_module("torch.nn"), "{}".format(name_fonction) + ("{}d".format(dim) if dim is not None else ""))

class BlockConfig():

    @config("BlockConfig")
    def __init__(self, nb_conv_per_level : int = 1, kernel_size : int = 3, stride : int = 1, padding : int = 1, activation : str = "ReLU") -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.nb_conv_per_level = nb_conv_per_level
        self.activation = activation

    def getConv(self, in_channels : int, out_channels : int, dim : int):
        return getTorchModule("Conv", dim = dim)(in_channels = in_channels, out_channels = out_channels, kernel_size = self.kernel_size, stride = self.stride, padding = self.padding)
    
    def getActivation(self):
        return getTorchModule(self.activation)()
        
class Block(torch.nn.Module):
    
    def __init__(self, in_channels : int, out_channels : int, blockConfig : BlockConfig, dim : int) -> None:
        super().__init__()
        args = []
        for _ in range(blockConfig.nb_conv_per_level):
            args+= [blockConfig.getConv(in_channels, out_channels, dim), blockConfig.getActivation()]
            in_channels = out_channels
        self.block = torch.nn.Sequential(*args)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.block(x)

class Encoder(torch.nn.Module):
    def __init__(self, channels : List[int], blockConfig : BlockConfig, dim : int) -> None:
        super().__init__()
        self.encoder_blocks = torch.nn.ModuleList([Block(channels[i], channels[i+1], blockConfig, dim=dim) for i in range(len(channels)-1)])
        self.pool           = getTorchModule("MaxPool", dim = dim)(2)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        filters = []
        for block in self.encoder_blocks:
            x = block(x)
            filters.append(x)
            x = self.pool(x)
        return filters

class Decoder(torch.nn.Module):
    def __init__(self, channels : List[int], encoder_channels : List[int], blockConfig : BlockConfig, upsample : bool, dim : int) -> None:
        super().__init__()
        self.encoder_channels = encoder_channels
        self.upsampling = torch.nn.ModuleList([torch.nn.Upsample(scale_factor=2, mode='nearest') if upsample else getTorchModule("ConvTranspose", dim = dim)(in_channels = channels[i], out_channels = channels[i+1], kernel_size = 2, stride = 2, padding = 0) for i in range(len(channels)-1)])
        self.decoder_blocks = torch.nn.ModuleList([Block(channels[i] + (0 if i == 0 else encoder_channels[-(i+1)]), channels[i+1], blockConfig, dim=dim) for i in range(len(channels)-1)]) 
    
    @staticmethod
    def _centerCrop(size : List[int], x : torch.Tensor) -> torch.Tensor:
        x_shape = x.shape[2:]
        slices = []
        for max in x.shape[:2]:
            slices.append(slice(max))
        for i, shape in enumerate(x_shape):
            slices += [slice((shape - size[i]) // 2, (shape + size[i]) // 2)]
        return x[slices]

    def forward(self, x : torch.Tensor, encoder_filters : torch.nn.ModuleList) -> torch.Tensor:
        for i in range(len(encoder_filters)):
            x        = self.decoder_blocks[i](x)
            x        = self.upsampling[i](x)
            x        = torch.cat([x, Decoder._centerCrop(x.shape[2:], encoder_filters[i])], dim=1)
        return x

class UNet(Network):

    @config("UNet")
    def __init__(self,
                    encoder_channels : List[int] = [1, 32, 64, 128],
                    decoder_channels : List[int] = [128, 64, 32],
                    final_channels : List[int] = [32, 2],
                    upsample : bool = True,
                    blockConfig : BlockConfig = BlockConfig(),
                    dim : int = 3) -> None:
        super().__init__()
        self.encoder     = Encoder(channels = encoder_channels, blockConfig = blockConfig, dim=dim)
        self.decoder     = Decoder(channels = decoder_channels, encoder_channels = encoder_channels, blockConfig = blockConfig, upsample=upsample, dim=dim)
        
        self.final_conv =  torch.nn.Sequential(*[Block(final_channels[i] +(encoder_channels[1] if i == 0 else 0), final_channels[i+1], blockConfig, dim=dim) for i in range(len(final_channels)-1)])
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        encoder_filters = self.encoder(x)
        out      = self.decoder(encoder_filters[-1], encoder_filters[::-1][1:])
        out      = self.final_conv(out)
        return out
    
    def logImage(self, out : torch.Tensor) -> np.ndarray:
        result = torch.argmax(torch.softmax(out[0, ...], dim = 0), dim=0).cpu().numpy()
        if len(result.shape) == 3:
            result = np.add.reduce(result, axis = 2)
        return result

    def getOutputModel(self, group : str, out : torch.Tensor) -> np.ndarray:
        return torch.softmax(out, dim = 0)


class UNetVoxelMorph(UNet):

    @config("UNet")
    def __init__(self, encoder_channels: List[int] = [4, 16,32,32,32], decoder_channels: List[int] = [32,32,32,32,32], final_channels: List[int] = [32, 16, 16], upsample : bool = True, blockConfig: BlockConfig = BlockConfig(), dim: int = 3) -> None:
        super().__init__(encoder_channels, decoder_channels, final_channels, upsample, blockConfig, dim)
        self.final_nf = final_channels[-1]

class VoxelMorph(Network):

    @config("VoxelMorph")
    def __init__(   self,
                    shape : List[int] = [192, 192, 192],
                    unet : UNetVoxelMorph = UNetVoxelMorph(),
                    int_steps : int = 7,
                    int_downsize : int = 2):

        super().__init__()
        self.unet = unet
        ndims = len(shape)

        self.flow = getTorchModule("Conv", dim=ndims)(in_channels = self.unet.final_nf, out_channels = ndims, kernel_size=3, stride = 1, padding=1)
        self.flow.weight = torch.nn.Parameter(torch.distributions.Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = torch.nn.Parameter(torch.zeros(self.flow.bias.shape))

        if int_steps > 0 and int_downsize > 1:
            self.resize = ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        if int_steps > 0 and int_downsize > 1:
            self.fullsize = ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        down_shape = [int(dim / int_downsize) for dim in shape]

        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None
    
        self.transformer = SpatialTransformer(shape)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        out = self.unet(x)
        flow_field = self.flow(out)

        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        neg_flow = -pos_flow

        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow)
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow)

        images_0 = self.transformer(torch.unsqueeze(x[:, 0, ...], 1), pos_flow, "bilinear")
        if x.shape[1] > 2:
            masks_0 = self.transformer(torch.unsqueeze(x[:, 2,...], 1), pos_flow, "nearest")

        images_1 = self.transformer(torch.unsqueeze(x[:, 1,...], 1), neg_flow, "bilinear")
        if x.shape[1] > 3:
            masks_1 = self.transformer(torch.unsqueeze(x[:, 3,...], 1), neg_flow, "nearest")
        
        result = {}
        result.update({"Images_0" : images_0})
        if masks_0 is not None:
            result.update({"Masks_0" : masks_0})

        result.update({"Images_1" : images_1})
        if masks_1 is not None:
            result.update({"Masks_1" : masks_1})

        result.update({"DeformationField" : pos_flow})
        return result 

    def logImage(self, out : torch.Tensor) -> np.ndarray:
        return None

    def getOutputModel(self, group : str, out : torch.Tensor) -> np.ndarray:
        return out


class SpatialTransformer(torch.nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size : List[int]):
        super().__init__()

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        
        self.register_buffer('grid', grid)

    def forward(self, src, flow, mode):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]

        return torch.nn.functional.grid_sample(src, new_locs, align_corners=True, mode=mode)


class VecInt(torch.nn.Module):

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec, "bilinear")
        return vec


class ResizeTransform(torch.nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize

    def forward(self, x):
        if self.factor < 1:
            x = torch.nn.functional.interpolate(x, align_corners=True, scale_factor=self.factor, mode="trilinear")
            x = self.factor * x

        elif self.factor > 1:
            x = self.factor * x
            x = torch.nn.functional.interpolate(x, align_corners=True, scale_factor=self.factor, mode="trilinear")
        return x

