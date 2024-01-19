import torch
import torch.nn as nn
from torchsummary import summary
from .builder import block,Attentionblock
from typing import List
from torchinfo import  summary



class Encoder(nn.Module):
    """Implements the encoder side of the Autoencoder."""

    def __init__(
        self,
        fan_in: int=3,
        fan_out: int=8,
        channels: List[int] = [128,256,512],
        layers_per_block: int = 1,
        resnet_groups: int = 32,
    ):
        super().__init__()

    


        self.conv_in = nn.Conv2d(
             fan_in, channels[0], kernel_size=3, stride=1, padding=1
         )
        
        channels = [channels[0]] + channels

        
        
        self.down_blocks = nn.ModuleList([ 
            
            nn.Sequential(

            block(in_channels,out_channels,resnetlayer=layers_per_block,resnet_groups=resnet_groups,down=False,up=False),
            block(out_channels,out_channels,resnetlayer=layers_per_block,resnet_groups=resnet_groups,down=True,up=False) 
            )
            
            for (in_channels, out_channels) in zip(channels, channels[1:])
        ])

        
        self.mid_blocks = nn.ModuleList([
            block(
                channels[-1],
                channels[-1],
                resnetlayer=layers_per_block,
            ),
            
            block(
                channels[-1],
                channels[-1],
                resnetlayer=layers_per_block,
            ),
            block(
                channels[-1],
                channels[-1],
                resnetlayer=layers_per_block,
            ),
            #Attentionblock(512),
        ])



        self.conv_out = nn.Conv2d(channels[-1], fan_out, 3, padding=1)
        self.conv = nn.Conv2d(fan_out, fan_out, 1, padding=0)
        



        self.act=nn.SiLU()

    def forward(self, x,noise=0.01):
        x = self.conv_in(x)

        for l in self.down_blocks:
             x = l(x)
             

        for l in self.mid_blocks:
             x = l(x)
        
        x=self.conv(self.conv_out(self.act(x)))

        mean,l_var=torch.chunk(x,2,dim=1)
        var=l_var.exp()
        std=var.sqrt()
        z= mean + std *noise


        

        return z*0.18215
             
