from monai.networks.nets.swin_unetr import PatchEmbed, ensure_tuple_rep, UnetrBasicBlock, look_up_option, MERGING_MODE, UnetrUpBlock, UnetOutBlock, BasicLayer
from DeepLearning_API.config import config
from DeepLearning_API.networks import network, blocks
import torch
from typing import Union
from DeepLearning_API.HDF5 import ModelPatch

class SwinTransformer(network.ModuleArgsDict):

    class Proj(torch.nn.Module):

        def __init__(self, normalize: bool) -> None:
            super().__init__()
            self.normalize = normalize
        
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            if self.normalize:
                x_shape = input.size()
                ch = x_shape[1]
                input = blocks.ToFeatures(3 if len(x_shape) == 5 else 2)(input)
                input = torch.nn.functional.layer_norm(input, [ch])
                input = blocks.ToChannels(3 if len(x_shape) == 5 else 2)(input)
            return input
    
    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        window_size: list[int],
        patch_size: list[int],
        depths: list[int],
        num_heads: list[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[torch.nn.LayerNorm] = torch.nn.LayerNorm,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        use_v2=False,
        normalize: bool = False
    ) -> None:
        super().__init__()
        num_layers = len(depths)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        layers = torch.nn.ModuleList()
        if use_v2:
            layersc = torch.nn.ModuleList()
        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample

        for i_layer in range(num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=down_sample_mod,
                use_checkpoint=use_checkpoint,
            )
            layers.append(layer)
            if use_v2:
                layerc = UnetrBasicBlock(
                    spatial_dims=3,
                    in_channels=embed_dim * 2**i_layer,
                    out_channels=embed_dim * 2**i_layer,
                    kernel_size=3,
                    stride=1,
                    norm_name="instance",
                    res_block=True,
                )
                layersc.append(layerc)


        self.add_module("PatchEmbed", PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
            spatial_dims=spatial_dims
        ), out_branch=["x0"])

        self.add_module("Pos_drop", torch.nn.Dropout(drop_rate), in_branch=["x0"], out_branch=["x0"])
        self.add_module("Proj_x0", SwinTransformer.Proj(normalize), in_branch=["x0"], out_branch=["x0_out"])
            
        for i in range(4):
            if use_v2:
                self.add_module("Layers_{}c".format(i), layersc[i], in_branch=["x{}".format(i)], out_branch=["x{}".format(i)])
            
            self.add_module("Layers_{}".format(i), layers[i], in_branch=["x{}".format(i)], out_branch=["x{}".format(i+1)])
            self.add_module("Proj_x{}".format(i+1), SwinTransformer.Proj(normalize), in_branch=["x{}".format(i+1)], out_branch=["x{}_out".format(i+1)])
    
class SwinUNETR(network.ModuleArgsDict):

    def __init__(
        self,
        img_size: list[int],
        in_channels: int,
        out_channels: int,
        depths: list[int] = [2, 2, 2, 2],
        num_heads: list[int] = [3, 6, 12, 24],
        feature_size: int = 24,
        norm_name: Union[tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        downsample="merging",
        use_v2=False,
    ) -> None:
        super().__init__()
        spatial_dims = len(img_size)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        self.add_module("SwinViT", SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
            use_v2=use_v2,
            normalize=normalize
        ), in_branch=[0], out_branch=["x0", "x1", "x2", "x3", "x4"])

        self.add_module("Encoder_0", UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        ), in_branch=[0], out_branch=["encoder_0"])

        self.add_module("Encoder_1", UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        ), in_branch=["x0"], out_branch=["encoder_1"])
       
        self.add_module("Encoder_2", UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        ), in_branch=["x1"], out_branch=["encoder_2"])

        self.add_module("Encoder_3", UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        ), in_branch=["x2"], out_branch=["encoder_3"])

        self.add_module("Encoder_4", UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        ), in_branch=["x4"], out_branch=["encoder_4"])

        self.add_module("Decoder_4", UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        ), in_branch=["encoder_4", "x3"], out_branch=["decoder_4"])

        self.add_module("Decoder_3", UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        ), in_branch=["decoder_4", "encoder_3"], out_branch=["decoder_3"])

        self.add_module("Decoder_2", UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        ), in_branch=["decoder_3", "encoder_2"], out_branch=["decoder_2"])

        self.add_module("Decoder_1", UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        ), in_branch=["decoder_2", "encoder_1"], out_branch=["decoder_1"])

        self.add_module("Decoder_0", UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        ), in_branch=["decoder_1", "encoder_0"])

        self.add_module("Out", UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels))
    
class SwinUnet(network.Network):

    class SwinUnetHead(network.ModuleArgsDict):

        def __init__(self) -> None:
            super().__init__()
            self.add_module("Softmax", torch.nn.Softmax(dim=1))
            self.add_module("Argmax", blocks.ArgMax(dim=1))

    @config("SwinUnet")
    def __init__(self,
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.SchedulersLoader = network.SchedulersLoader(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    patch : Union[ModelPatch, None] = None,
                    dim : int = 3,
                    img_size: list[int] = [256,512,512],
                    in_channel: int = 1,
                    nb_class: int = 2):
        super().__init__(in_channels = in_channel, optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, patch=patch, dim = dim)
        self.add_module("SwinUnet", SwinUNETR(  img_size=img_size,
                                                in_channels=in_channel,
                                                out_channels=nb_class, use_checkpoint=True))
        self.add_module("Head", SwinUnet.SwinUnetHead())
