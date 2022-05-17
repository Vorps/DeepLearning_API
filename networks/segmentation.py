from enum import Enum
import torch
import numpy as np
from typing import Dict, List
from DeepLearning_API import config
from DeepLearning_API.networks import network 

class UpSampleMode(Enum):
    CONV_TRANSPOSE = 0,
    UPSAMPLE = 1

class DownSampleMode(Enum):
    MAXPOOL = 0
    AVGPOOL = 1
    CONV_STRIDE = 2

class Encoder(torch.nn.Module):
    
    @staticmethod
    def _upsample_function(in_channels : int, blockConfig : network.BlockConfig, downSampleMode : DownSampleMode, dim : int) -> torch.nn.Module:
        if downSampleMode == DownSampleMode.MAXPOOL:
            pool           = network.getTorchModule("MaxPool", dim = dim)(2)
        if downSampleMode == DownSampleMode.AVGPOOL:
            pool           = network.getTorchModule("AvgPool", dim = dim)(2)
        if downSampleMode == DownSampleMode.CONV_STRIDE:
            args = []
            conv = blockConfig.getConv(in_channels = in_channels, out_channels = in_channels, dim = dim)
            conv.stride = 2
            args.append(conv)
            if blockConfig.batchNorm:
                args.append(network.getTorchModule("BatchNorm", dim = dim))
            args.append(blockConfig.getActivation())
            pool           = torch.nn.Sequential(*args)
        return pool

    def __init__(self, channels : List[int], blockConfig : network.BlockConfig, downSampleMode : DownSampleMode, dim : int) -> None:
        super().__init__()
        self.encoder_blocks = torch.nn.ModuleList([network.ConvBlock(channels[i], channels[i+1], blockConfig, dim=dim) for i in range(len(channels)-1)])
        self.pool = torch.nn.ModuleList([Encoder._upsample_function(channels[i+1], blockConfig, downSampleMode, dim=dim) for i in range(len(channels)-2)])
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        filters = []
        x = self.encoder_blocks[0](x)
        for block, pool in zip(self.encoder_blocks[1:], self.pool):
            filters.append(x)
            x = pool(x)
            x = block(x)
        return x, filters

class Decoder(torch.nn.Module):
    @staticmethod
    def _upsample_function(in_channels : int, out_channels : int, upSampleMode : UpSampleMode, dim : int) -> torch.nn.Module:
        if upSampleMode == UpSampleMode.UPSAMPLE:
            upSample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        if upSampleMode == UpSampleMode.CONV_TRANSPOSE:
            upSample = network.getTorchModule("ConvTranspose", dim = dim)(in_channels = in_channels, out_channels = in_channels, kernel_size = 2, stride = 2, padding = 0)
        return upSample

    def __init__(self, channels : List[int], encoder_channels : List[int], blockConfig : network.BlockConfig, upSampleMode : UpSampleMode, attention : bool, dim : int) -> None:
        super().__init__()
        self.encoder_channels = encoder_channels
        self.upsampling = torch.nn.ModuleList([Decoder._upsample_function(channels[i], channels[i+1], upSampleMode, dim = dim) for i in range(len(channels)-1)])
        
        self.decoder_blocks = torch.nn.ModuleList([network.ConvBlock(channels[i] + encoder_channels[-i-2], channels[i+1], blockConfig, dim=dim) for i in range(len(channels)-1)]) 
        self.attentionBlocks = torch.nn.ModuleList([network.AttentionBlock(channels[i], encoder_channels[-i-2], channels[i+1], dim=dim) for i in range(len(channels)-1)]) if attention else None

    def forward(self, x : torch.Tensor, encoder_filters : torch.nn.ModuleList) -> torch.Tensor:
        for i in range(len(encoder_filters)):
            if self.attentionBlocks is not None:
                encoder_filters[i] = self.attentionBlocks[i](x, encoder_filters[i])
            x        = self.upsampling[i](x)
            x        = torch.cat([x,  encoder_filters[i]], dim=1)
            x        = self.decoder_blocks[i](x)
        return x

class UNet(network.Network):

    @config("UNet")
    def __init__(self,
                    optimizer : network.Optimizer = network.Optimizer(),
                    scheduler : network.Scheduler = network.Scheduler(),
                    criterions: Dict[str, network.Criterions] = {"default" : network.Criterions()},
                    dim : int = 3,
                    encoder_channels : List[int] = [1, 32, 64, 128],
                    decoder_channels : List[int] = [128, 64, 32],
                    final_channels : List[int] = [32, 2],
                    downSampleMode : str = "MAXPOOL",
                    upSampleMode : str = "UPSAMPLE",
                    attention : bool = False,
                    blockConfig : network.BlockConfig = network.BlockConfig()) -> None:
        super().__init__(optimizer = optimizer, scheduler = scheduler, criterions = criterions, dim = dim)
        
        self.encoder     = Encoder(channels = encoder_channels, blockConfig = blockConfig, downSampleMode = DownSampleMode._member_map_[downSampleMode], dim=dim)
        self.decoder     = Decoder(channels = decoder_channels, encoder_channels = encoder_channels, blockConfig = blockConfig, upSampleMode=UpSampleMode._member_map_[upSampleMode], attention = attention, dim=dim)
        self.final_conv =  torch.nn.Sequential(*[network.ConvBlock(final_channels[i], final_channels[i+1], blockConfig, dim=dim) for i in range(len(final_channels)-1)])
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x ,encoder_filters = self.encoder(x)
        out      = self.decoder(x, encoder_filters[::-1])
        out      = self.final_conv(out)
        return out
    
    def logImage(self, input : torch.Tensor, output : Dict[str, torch.Tensor]) -> List[np.ndarray]:
        result = torch.argmax(torch.softmax(output[0, ...], dim = 0), dim=0).cpu().numpy()
        if len(result.shape) == 3:
            result = np.add.reduce(result, axis = 2)
        return result