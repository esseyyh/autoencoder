import torch
import torch.nn as nn
from torchsummary import summary
#from builder import block,AttentionHead
from builder import block
from typing import List




class Encoder(nn.Module):
    """Implements the encoder side of the Autoencoder."""

    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        block_out_channels: List[int] = [4,8,16,32,64],
        layers_per_block: int = 16,
        resnet_groups: int = 32,
    ):
        super().__init__()

    


        self.conv_in = nn.Conv2d(
             fan_in, block_out_channels[0], kernel_size=3, stride=1, padding=1
         )

        channels = [block_out_channels[0]] + list(block_out_channels)
        
        self.down_blocks = nn.ModuleList([
            block(in_channels,out_channels,resnetlayer=layers_per_block,down=i < len(block_out_channels) - 1,up=False)  
            for i, (in_channels, out_channels) in enumerate(zip(channels, channels[1:]))
        ])

        
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



        self.conv_out = nn.Conv2d(block_out_channels[-1], fan_out, 3, padding=1)
        # self.fc_mean = nn.Linear(4 * 32 * 32, 4 * 32 * 32)
        # self.fc_logvar = nn.Linear(4 * 32 * 32, 4 * 32 * 32)




        self.act=nn.ReLU()

    def forward(self, x):
        x = self.conv_in(x)

        for l in self.down_blocks:
             x = l(x)


        for l in self.mid_blocks:
             x = l(x)
        # x= self.conv_out(self.act(x))
        
        # x = x.view(x.size(0), -1)

        # mean = self.fc_mean(x)
        # logvar = self.fc_logvar(x)
        # return mean,logvar

        return self.conv_out(self.act(x))
             
        

