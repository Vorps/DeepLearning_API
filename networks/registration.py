from functools import partial
import torch
from torch.nn.parameter import Parameter
import numpy as np
from typing import Dict, List
from DeepLearning_API.config import config
from DeepLearning_API.networks import network, segmentation
from networks import blocks
import torch.nn.functional as F

class VoxelMorph(network.Network):

    @config("VoxelMorph")
    def __init__(   self,
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.SchedulersLoader = network.SchedulersLoader(),
                    outputsCriterions: Dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    dim : int = 3,
                    channels : List[int] = [4, 16,32,32,32],
                    blockConfig: blocks.BlockConfig = blocks.BlockConfig(),
                    nb_conv_per_stage: int = 2,
                    downSampleMode: str = "MAXPOOL",
                    upSampleMode: str = "CONV_TRANSPOSE",
                    attention : bool = False,
                    shape : List[int] = [192, 192, 192],
                    int_steps : int = 7,
                    int_downsize : int = 2):
        super().__init__(in_channels = channels[0], optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, dim = dim)
        self.add_module("Concat", blocks.Concat(), in_branch=[0,1,2,3])
        self.add_module("UNetBlock_0", segmentation.UNetBlock(channels, nb_conv_per_stage, blockConfig, downSampleMode=blocks.DownSampleMode._member_map_[downSampleMode], upSampleMode=blocks.UpSampleMode._member_map_[upSampleMode], attention=attention, dim=dim))
        self.add_module("Head", blocks.getTorchModule("Conv", dim)(in_channels = channels[1], out_channels = channels[0], kernel_size = 3, stride = 1, padding = 1))
        
        self["Head"].weight = Parameter(torch.distributions.Normal(0, 1e-5).sample(self["Head"].weight.shape))
        self["Head"].bias = Parameter(torch.zeros(self["Head"].bias.shape))

        if int_steps > 0 and int_downsize > 1:
            self.add_module("DownSample", ResizeTransform(int_downsize))

        self.add_module("NegativeFlow", blocks.ApplyFunction(lambda input : -input), out_branch=["neg_flow"])
        if int_steps > 0:
            self.add_module("Integrate_pos_flow", VecInt([int(dim / int_downsize) for dim in shape], int_steps))
            self.add_module("Integrate_neg_flow", VecInt([int(dim / int_downsize) for dim in shape], int_steps), in_branch=["neg_flow"], out_branch=["neg_flow"])

        if int_steps > 0 and int_downsize > 1:
            self.add_module("Upsample_pos_flow", ResizeTransform(1 / int_downsize))
            self.add_module("Upsample_neg_flow", ResizeTransform(1 / int_downsize), in_branch=["neg_flow"], out_branch=["neg_flow"])

        self.add_module("SpatialTransformer_pos_flow", SpatialTransformer(shape))
        self.add_module("SpatialTransformer_pos_flow", SpatialTransformer(shape), in_branch=["neg_flow"], out_branch=["neg_flow"])

class SpatialTransformer(torch.nn.Module):
    
    def __init__(self, size : List[int]):
        super().__init__()
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.float)
        self.register_buffer('grid', grid)

    def forward(self, src, flow, mode):
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]
        return F.grid_sample(src, new_locs, align_corners=True, mode=mode)

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
   
    def __init__(self, vel_resize):
        super().__init__()
        self.factor = 1.0 / vel_resize

    def forward(self, x):
        if self.factor < 1:
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode="trilinear", recompute_scale_factor = True)
            x = self.factor * x

        elif self.factor > 1:
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode="trilinear", recompute_scale_factor = True)
        return x