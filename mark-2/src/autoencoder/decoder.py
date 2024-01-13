import torch
import torch.nn as nn
from builder import block #,AttentionHead



class decoder(nn.Module):
    def __init__(
        self,
        fan_in: int=1,
        fan_out: int=64,
        channels: list  = [4,8],
        resnet_layers_per_block: int = 1,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(
            fan_in,channels[0], kernel_size=3, stride=1, padding=1
        )





        self.up_blocks = [
            block(
                in_channels,
                out_channels,
                resnet_layers_per_block,
                False,
                True
            )
            for  in_channels, out_channels in zip(channels, channels[1:])
        ]
        #self.attn= attentionhead(144*144)


        



    def __call__(self, x):
        x=self.conv_in(x)
        for down in self.up_blocks:
            x=down(x)
       # x=self.attn(x)  


        return x

x=torch.randn(1,1,10,10)
model=decoder()
print(model(x).shape)
