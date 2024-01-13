import torch 
import torch.nn as nn
from decoder import Decoder
from encoder import Encoder
from torchsummary import summary
from typing import List

class AE(nn.Module):
    


    def __init__(
        self,
        fan_in: int,
        bottle_neck: int,
        block_out_channels: List[int] = [4,8,16,32,64],
        layers_per_block: int = 16
        
    ):
        super().__init__()

    
        self.encoder=Encoder(fan_in,bottle_neck,block_out_channels,layers_per_block)
        self.decoder=Decoder(bottle_neck,fan_in,block_out_channels,layers_per_block)

    
    def decode(self, x):
        return self.decode(x)
    def encode(self, x):
        return self.encode(x)
    

    def forward(self, x):

        
        
        return self.decoder(self.encoder(x))
    
