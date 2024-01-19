import torch
import torch.nn as nn
from torchsummary import summary

from .builder import block
from typing import List
from torchinfo import summary



class Decoder(nn.Module):
    """Implements the encoder side of the Autoencoder."""

    def __init__(
        self,
        fan_in: int=4,
        fan_out: int=3,
        block_out_channels: List[int] = [128,256,512],
        layers_per_block: int = 1,
      
    ):
        super().__init__()

    


       

        self.conv_in = nn.Conv2d(fan_in,block_out_channels[-1],  3, padding=1)



        self.mid_blocks = nn.ModuleList([
            block(
                block_out_channels[-1],
                block_out_channels[-1],
                resnetlayer=layers_per_block,
            ),
            
            block(
                block_out_channels[-1],
                block_out_channels[-1],
                resnetlayer=layers_per_block,
            ),
            block(
                block_out_channels[-1],
                block_out_channels[-1],
                resnetlayer=layers_per_block,
            ),
        ])


        channels = list(reversed(block_out_channels))
        channels = [channels[0]] + channels

       
        
        self.up_blocks = nn.ModuleList([

             nn.Sequential(


            block(in_channels,out_channels,resnetlayer=layers_per_block,down=False,up=False),
            block(out_channels,out_channels,resnetlayer=layers_per_block,down=False,up=True)  

             )
             
            for i, (in_channels, out_channels) in enumerate(zip(channels, channels[1:]))
        ])



        self.conv_out = nn.Conv2d(
              channels[-1],fan_out, kernel_size=3, stride=1, padding=1
        )
        

        
        

        
        self.act=nn.ReLU()

    def forward(self, x):
        x = self.conv_in(x)
        
        for l in self.mid_blocks:
             x = l(x)
             

        for l in self.up_blocks:
             x = l(x)


        
        

        
        
        return self.conv_out(self.act(x))
        



