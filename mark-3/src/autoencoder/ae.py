import torch 
import torch.nn as nn
from decoder import Decoder
from encoder import Encoder
from torchinfo import summary 
from typing import List

class VAE(nn.Module):
    


    def __init__(
        self,
        fan_in: int=3,
        bottle_neck: int=4,
        block_out_channels: List[int] = [4,8,16,32,64],
        layers_per_block: int = 16
        
    ):
        super().__init__()

    
        self.encoder=Encoder()
        self.decoder=Decoder()

    
    # def decode(self, x):
    #     return self.decode(x)
    # def encode(self, x):
    #     return self.encode(x)
    

    def forward(self, x):

        
        
        return self.decoder(self.encoder(x))
    
x=torch.randn(1,3,512,512)
blok=VAE()
summary(blok,input_data=x)