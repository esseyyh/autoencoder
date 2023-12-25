import torch
import torch.nn as nn
from builder import block,AttentionHead



class encoder(nn.Module):
    def __init__(
        self,
        fan_in: int=3,
        fan_out: int=64,
        channels: list  = [4],
        resnet_layers_per_block: int = 1,
        latent_dim=64
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(
            fan_in,channels[0], kernel_size=3, stride=1, padding=1
        )





        self.down_blocks = [
            block(
                in_channels,
                out_channels,
                resnet_layers_per_block,
                True
            )
            for  in_channels, out_channels in zip(channels, channels[1:])
        ]
        self.attn= AttentionHead(latent_dim*latenet_dim)


        



    def __call__(self, x):
        x=self.conv_in(x)
        for down in self.down_blocks:
            x=down(x)
       # x=self.attn(x)  


        return x

