import torch
import numpy as np

class MLP(torch.nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()

        self.mlp_layers = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, mlp_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(mlp_dim, embedding_dim),
            torch.nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.mlp_layers(x)

        return x


class TransformerEncoderBlock(torch.nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()

        self.multi_head_attention = torch.nn.MultiheadAttention(embedding_dim, head_num)
        self.mlp = MLP(embedding_dim, mlp_dim)

        self.layer_norm1 = torch.nn.LayerNorm(embedding_dim)
        self.layer_norm2 = torch.nn.LayerNorm(embedding_dim)

        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        _x = self.multi_head_attention(x)
        _x = self.dropout(_x)
        x = x + _x
        x = self.layer_norm1(x)

        _x = self.mlp(x)
        x = x + _x
        x = self.layer_norm2(x)

        return x


class TransformerEncoder(torch.nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num, z_idx_list):
        super().__init__()

        self.z_idx_list = z_idx_list

        self.layer_blocks = torch.nn.ModuleList(
            [TransformerEncoderBlock(embedding_dim, head_num, mlp_dim) for _ in range(block_num)])

    def forward(self, x):
        z_outputs = []
        for idx, layer_block in enumerate(self.layer_blocks, start=1):
            x = layer_block(x)
            if idx in self.z_idx_list:
                z_outputs.append(x)

        return z_outputs


class AbsPositionalEncoding1D(torch.nn.Module):
    def __init__(self, tokens, dim):
        super().__init__()
        self.abs_pos_enc = torch.nn.Parameter(torch.randn(1, tokens, dim))

    def forward(self, input: torch.Tensor):
        return input + self.abs_pos_enc.repeat(*([input.shape[0]]+[1]*(len(input.shape)-1)))


class Transformer3D(torch.nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim, block_num, patch_size, z_idx_list):
        super().__init__()

        self.patch_size = patch_size
        self.n_patches = int((img_dim[0] * img_dim[1] * img_dim[2]) / (patch_size ** 3))

        self.patch_embeddings = nn.Conv3d(in_channels, embedding_dim,
                                          kernel_size=patch_size, stride=patch_size, bias=False)

        self.position_embeddings = AbsPositionalEncoding1D(self.n_patches, embedding_dim)
        self.dropout = nn.Dropout(0.1)

        self.transformer = TransformerEncoder(embedding_dim, head_num, mlp_dim, block_num, z_idx_list)

    def forward(self, x):
        embeddings = rearrange(self.patch_embeddings(x), 'b d x y z -> b (x y z) d')
        embeddings = self.position_embeddings(embeddings)
        embeddings = self.dropout(embeddings)

        z_outputs = self.transformer(embeddings)

        return z_outputs


if __name__ == '__main__':
    trans = Transformer3D(img_dim=(128, 128, 128),
                          in_channels=4,
                          patch_size=16,
                          embedding_dim=768,
                          block_num=12,
                          head_num=12,
                          mlp_dim=3072,
                          z_idx_list=[3, 6, 9, 12])
    z3, z6, z9, z12 = trans(torch.rand(1, 4, 128, 128, 128))
    print(z3.shape)
    print(z6.shape)
    print(z9.shape)
    print(z12.shape)

class YellowBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.downsample = in_channels != out_channels

        self.conv_block = nn.Sequential(nn.Conv3d(in_channels, out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding),
                                        normalization(out_channels),
                                        nn.LeakyReLU(negative_slope=.01, inplace=True),
                                        nn.Conv3d(out_channels, out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding),
                                        normalization(out_channels))

        if self.downsample:
            self.conv_block2 = nn.Sequential(nn.Conv3d(in_channels, out_channels,
                                                       kernel_size=1, stride=1, padding=0),
                                             normalization(out_channels))

        self.leaky_relu = nn.LeakyReLU(negative_slope=.01, inplace=True)

    def forward(self, x):
        res = x

        conv_output = self.conv_block(x)

        if self.downsample:
            res = self.conv_block2(res)

        conv_output += res
        x = self.leaky_relu(conv_output)
        return x


class SingleBlueBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization):
        super().__init__()

        self.conv_block = nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels,
                                                           kernel_size=2, stride=2, padding=0, bias=False),

                                        # Not exactly yellow block but it is
                                        YellowBlock(in_channels=out_channels,
                                                    out_channels=out_channels,
                                                    normalization=normalization))

    def forward(self, x):
        x = self.conv_block(x)
        return x


class BlueBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization, layer_num):
        super().__init__()

        self.transpose_conv = nn.ConvTranspose3d(in_channels, out_channels,
                                                 kernel_size=2, stride=2, padding=0, bias=False)

        layers = []
        for _ in range(layer_num - 1):
            layers.append(SingleBlueBlock(in_channels=out_channels,
                                          out_channels=out_channels,
                                          normalization=normalization))

        self.blocks = nn.ModuleList(layers)

    def forward(self, x):
        x = self.transpose_conv(x)
        for block in self.blocks:
            x = block(x)

        return x


class GreenBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.deconv_block = nn.ConvTranspose3d(in_channels, out_channels,
                                               kernel_size=2, stride=2, padding=0, bias=False)

    def forward(self, x):
        x = self.deconv_block(x)
        return x


class UneTR(nn.Module):
    def __init__(self, img_dim, in_channels, base_filter, class_num,
                 patch_size, embedding_dim, block_num, head_num, mlp_dim, z_idx_list):
        super().__init__()

        self.patch_dim = [int(x / patch_size) for x in img_dim]

        self.transformer = Transformer3D(img_dim=img_dim,
                                         in_channels=in_channels,
                                         patch_size=patch_size,
                                         embedding_dim=embedding_dim,
                                         block_num=block_num,
                                         head_num=head_num,
                                         mlp_dim=mlp_dim,
                                         z_idx_list=z_idx_list)

        self.z0_yellow_block = YellowBlock(in_channels=in_channels,
                                           out_channels=base_filter,
                                           normalization=nn.InstanceNorm3d)

        self.z3_blue_block = BlueBlock(in_channels=embedding_dim,
                                       out_channels=base_filter * 2,
                                       normalization=nn.InstanceNorm3d,
                                       layer_num=3)

        self.z6_blue_block = BlueBlock(in_channels=embedding_dim,
                                       out_channels=base_filter * 4,
                                       normalization=nn.InstanceNorm3d,
                                       layer_num=2)

        self.z9_blue_block = BlueBlock(in_channels=embedding_dim,
                                       out_channels=base_filter * 8,
                                       normalization=nn.InstanceNorm3d,
                                       layer_num=1)

        self.z3_green_block = GreenBlock(in_channels=base_filter * 2,
                                         out_channels=base_filter)

        self.z6_green_block = GreenBlock(in_channels=base_filter * 4,
                                         out_channels=base_filter * 2)

        self.z9_green_block = GreenBlock(in_channels=base_filter * 8,
                                         out_channels=base_filter * 4)

        self.z12_green_block = GreenBlock(in_channels=embedding_dim,
                                          out_channels=base_filter * 8)

        self.z3_yellow_block = YellowBlock(in_channels=base_filter * 2 * 2,
                                           out_channels=base_filter * 2,
                                           normalization=nn.InstanceNorm3d)

        self.z6_yellow_block = YellowBlock(in_channels=base_filter * 4 * 2,
                                           out_channels=base_filter * 4,
                                           normalization=nn.InstanceNorm3d)

        self.z9_yellow_block = YellowBlock(in_channels=base_filter * 8 * 2,
                                           out_channels=base_filter * 8,
                                           normalization=nn.InstanceNorm3d)

        self.output_block = nn.Sequential(YellowBlock(in_channels=base_filter * 2,
                                                      out_channels=base_filter,
                                                      normalization=nn.InstanceNorm3d),
                                          nn.Conv3d(base_filter, class_num, kernel_size=1, stride=1))

    def forward(self, x):
        z_embedding = self.transformer(x)

        # arrange z values
        arranger = lambda z_emb: rearrange(z_emb, 'b (x y z) d -> b d x y z',
                                           x=self.patch_dim[0], y=self.patch_dim[1], z=self.patch_dim[2])
        z3, z6, z9, z12 = [arranger(z) for z in z_embedding]

        # init yellow block operation
        z0 = self.z0_yellow_block(x)

        # blue block operations
        z3 = self.z3_blue_block(z3)
        z6 = self.z6_blue_block(z6)
        z9 = self.z9_blue_block(z9)

        # green and yellow blocks operations and their concatenations
        z12 = self.z12_green_block(z12)
        y = torch.cat([z12, z9], dim=1)
        y = self.z9_yellow_block(y)

        y = self.z9_green_block(y)
        y = torch.cat([y, z6], dim=1)
        y = self.z6_yellow_block(y)

        y = self.z6_green_block(y)
        y = torch.cat([y, z3], dim=1)
        y = self.z3_yellow_block(y)

        y = self.z3_green_block(y)
        y = torch.cat([y, z0], dim=1)

        y = self.output_block(y)

        return y


if __name__ == '__main__':
    nn = UneTR(img_dim=(128, 128, 128),
               in_channels=4,
               base_filter=16,
               class_num=3,
               patch_size=16,
               embedding_dim=768,
               block_num=12,
               head_num=12,
               mlp_dim=3072,
               z_idx_list=[3, 6, 9, 12])

    r = nn(torch.rand(1, 4, 128, 128, 128))
    print(r.shape)


torch.nn.MultiheadAttention(embed_dim, num_heads)
