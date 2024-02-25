import torch
from torch.nn.parameter import Parameter
from DeepLearning_API.config import config
from models.segmentation import UNet
from DeepLearning_API.networks import blocks, network
import torch.nn.functional as F

class VoxelMorph(network.Network):

    class Flow(network.ModuleArgsDict):

        def __init__(self, in_channels: int, int_steps: int, int_downsize: int, shape: list[int], dim: int) -> None:
            super().__init__()
            self.add_module("Head", blocks.getTorchModule("Conv", dim)(in_channels = in_channels, out_channels = dim, kernel_size = 3, stride = 1, padding = 1))
            self["Head"].weight = Parameter(torch.distributions.Normal(0, 1e-5).sample(self["Head"].weight.shape))
            self["Head"].bias = Parameter(torch.zeros(self["Head"].bias.shape))
        
            if int_steps > 0 and int_downsize > 1:
                self.add_module("DownSample", ResizeTransform(int_downsize))

            if int_steps > 0:
                self.add_module("Integrate_pos_flow", VecInt([int(dim / int_downsize) for dim in shape], int_steps))
            
            if int_steps > 0 and int_downsize > 1:
                self.add_module("Upsample_pos_flow", ResizeTransform(1 / int_downsize))
                
    @config("VoxelMorph")
    def __init__(   self,
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.LRSchedulersLoader = network.LRSchedulersLoader(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    dim : int = 3,
                    channels : list[int] = [4, 16,32,32,32],
                    blockConfig: blocks.BlockConfig = blocks.BlockConfig(),
                    nb_conv_per_stage: int = 2,
                    downSampleMode: str = "MAXPOOL",
                    upSampleMode: str = "CONV_TRANSPOSE",
                    attention : bool = False,
                    shape : list[int] = [192, 192, 192],
                    int_steps : int = 7,
                    int_downsize : int = 2,
                    nb_field: int = 1, 
                    nb_batch_per_step : int = 1):
        super().__init__(in_channels = channels[0], optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, dim = dim, nb_batch_per_step=nb_batch_per_step)
        self.add_module("Concat", blocks.Concat(), in_branch=[0,1,2,3], out_branch=["input_concat"])
        self.add_module("UNetBlock_0", UNet.UNetBlock(channels, nb_conv_per_stage, blockConfig, downSampleMode=blocks.DownSampleMode._member_map_[downSampleMode], upSampleMode=blocks.UpSampleMode._member_map_[upSampleMode], attention=attention, block = blocks.ConvBlock, nb_class=0, dim=dim), in_branch=["input_concat"], out_branch=["unet"])
        for i in range(nb_field):
            self.add_module("Flow_{}".format(i), VoxelMorph.Flow(channels[1], int_steps, int_downsize, shape, dim), in_branch=["unet"], out_branch=["pos_flow_{}".format(i)])

        self.add_module("MaskFlow", MaskFlow(), in_branch=[2]+["pos_flow_{}".format(i) for i in range(nb_field)], out_branch=["pos_flow"])
        self.add_module("Negative", blocks.Negative(), in_branch=["pos_flow"], out_branch=["neg_flow"])

        self.add_module("FixedImageResample", SpatialTransformer(shape), in_branch=[0, "neg_flow"], out_branch=["fixed_image_resample"])
        self.add_module("MovingImageResample", SpatialTransformer(shape), in_branch=[1, "pos_flow"], out_branch=["moving_image_resample"])

        self.add_module("FixedMaskResample", SpatialTransformer(shape, True), in_branch=[2, "neg_flow"], out_branch=["fixed_mask_resample"])
        self.add_module("MovingMaskResample", SpatialTransformer(shape, True), in_branch=[3, "pos_flow"], out_branch=["moving_mask_resample"])

class MaskFlow(torch.nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, mask: torch.Tensor, *flows: torch.Tensor):
        result = torch.zeros_like(flows[0])
        for i, flow in enumerate(flows):
            result = result+torch.where(mask == i+1, flow, torch.tensor(0))
        return result

class SpatialTransformer(torch.nn.Module):
    
    def __init__(self, size : list[int], mask: bool = False):
        super().__init__()
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.float)
        self.register_buffer('grid', grid)
        self.mask = mask

    def forward(self, src: torch.Tensor, flow: torch.Tensor):
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]
        result = F.grid_sample(src, new_locs, align_corners=True, mode="bilinear")
        return result
    
class VecInt(torch.nn.Module):

    def __init__(self, inshape, nsteps):
        super().__init__()
        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec: torch.Tensor):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

class ResizeTransform(torch.nn.Module):
   
    def __init__(self, size):
        super().__init__()
        self.factor = 1.0 / size

    def forward(self, x: torch.Tensor):
        if self.factor < 1:
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode="trilinear", recompute_scale_factor = True)
            x = self.factor * x

        elif self.factor > 1:
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode="trilinear", recompute_scale_factor = True)
        return x