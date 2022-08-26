"""from functools import partial
import importlib
import torch
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List
from DeepLearning_API import config
from DeepLearning_API.networks import network, segmentation

class VoxelMorph(segmentation.UNet):

    @config("VoxelMorph")
    def __init__(   self,
                    optimizer : network.Optimizer = network.Optimizer(),
                    scheduler : network.Scheduler = network.Scheduler(),
                    criterions: Dict[str, network.Criterions] = {"default" : network.Criterions()},
                    dim : int = 3,
                    encoder_channels : List[int] = [4, 16,32,32,32],
                    decoder_channels : List[int] = [32,32,32,32,32],
                    final_channels : List[int] = [32, 16, 16],
                    blockConfig : network.BlockConfig = network.BlockConfig(),
                    shape : List[int] = [192, 192, 192],
                    int_steps : int = 7,
                    int_downsize : int = 2):

        super().__init__(   optimizer = optimizer, 
                            scheduler = scheduler,
                            criterions = criterions,
                            dim = dim,
                            encoder_channels=encoder_channels,
                            decoder_channels=decoder_channels,
                            final_channels=final_channels,
                            downSampleMode="MAXPOOL",
                            upSampleMode="CONV_TRANSPOSE",
                            attention=False,
                            blockConfig=blockConfig)

        ndims = len(shape)
        self.flow = network.getTorchModule("Conv", dim=ndims)(in_channels = self.final_channel, out_channels = ndims, kernel_size=3, stride=1, padding=1)
        self.flow.weight = torch.nn.Parameter(torch.distributions.Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = torch.nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.resize = ResizeTransform(int_downsize, ndims) if int_steps > 0 and int_downsize > 1 else None
        self.fullsize = ResizeTransform(1 / int_downsize, ndims) if int_steps > 0 and int_downsize > 1 else None
        down_shape = [int(dim / int_downsize) for dim in shape]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None
        self.transformer = SpatialTransformer(shape)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        out = super().forward(x)["output"]
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
            masks_0 = self.transformer(torch.unsqueeze(x[:, 2,...], 1), pos_flow, "bilinear")

        images_1 = self.transformer(torch.unsqueeze(x[:, 1,...], 1), neg_flow, "bilinear")
        if x.shape[1] > 3:
            masks_1 = self.transformer(torch.unsqueeze(x[:, 3,...], 1), neg_flow, "bilinear")
        
        result = {}
        result.update({"Images_0" : images_0})
        if masks_0 is not None:
            result.update({"Masks_0" : masks_0})

        result.update({"Images_1" : images_1})
        if masks_1 is not None:
            result.update({"Masks_1" : masks_1})

        result.update({"DeformationField" : pos_flow})
        return result 

    def _logImage(self, input : torch.Tensor, output : torch.Tensor, i, j):
        return input[0, i,input.shape[2]//2, ...]+output[0, j, input.shape[2]//2, ...]

    def _logImageGradient(self, input : torch.Tensor, output : torch.Tensor, i, j):
        gradient_magnitude = lambda x : np.linalg.norm(np.asarray(np.gradient(x)), ord=2, axis=0)
        result = gradient_magnitude(input[0, i,input.shape[2]//2, ...])+gradient_magnitude(output[0, j, input.shape[2]//2, ...])
        return result

    def logImage(self, input : torch.Tensor, output : Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        result = dict()
        input = input.cpu().numpy()
        prediction_images = partial(self._logImage, input, torch.cat(tuple([v.detach().cpu() for k, v in output.items() if k.startswith("Images_")]), dim=1).numpy())
        prediction_images_gradient = partial(self._logImageGradient, input, torch.cat(tuple([v.detach().cpu() for k, v in output.items() if k.startswith("Images_")]), dim=1).numpy())
        images = partial(self._logImage, input, input)

        result.update({"Image_0 -> Image_1" : images(0,1)})
        result.update({"Prediction/Image_0 -> Image_1" : prediction_images(0,1)})
        result.update({"Gradient/Image_0 -> Image_1" : prediction_images_gradient(0,1)})

        if input.shape[1] > 2:
            prediction_masks = partial(self._logImage, input, torch.cat(tuple([v.detach().cpu() for k, v in output.items() if k.startswith("Masks_")]), dim=1).numpy())
            prediction_masks_gradient = partial(self._logImageGradient, input, torch.cat(tuple([v.detach().cpu() for k, v in output.items() if k.startswith("Masks_")]), dim=1).numpy())
            result.update({"Mask_0 -> Mask_1" : prediction_masks(2,1)})
            result.update({"Gradient/Mask_0 -> Mask_1" : prediction_masks_gradient(2,1)})        

        result.update({"DeformationField" : output["DeformationField"][0, :, input.shape[2]//2, ...].detach().cpu().numpy()})
        return result


class SpatialTransformer(torch.nn.Module):
    

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
   
    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize

    def forward(self, x):
        if self.factor < 1:
            x = torch.nn.functional.interpolate(x, align_corners=True, scale_factor=self.factor, mode="trilinear", recompute_scale_factor = True)
            x = self.factor * x

        elif self.factor > 1:
            x = self.factor * x
            x = torch.nn.functional.interpolate(x, align_corners=True, scale_factor=self.factor, mode="trilinear", recompute_scale_factor = True)
        return x"""