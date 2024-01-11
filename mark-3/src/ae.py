import torch 
import torch.nn as nn
from decoder import Decoder
from encoder import Encoder
from torchsummary import summary

class AE(nn.Module):
    


    def __init__(
        self,
        fan_in: int,
        bottle_neck: int,
        # block_out_channels: List[int] = [4,8,16,32,64],
        # layers_per_block: int = 16,
        # resnet_groups: int = 32,
    ):
        super().__init__()

    


        self.encoder=Encoder(fan_in,bottle_neck)
        self.decoder=Decoder(bottle_neck,fan_in)

    def forward(self, x):

        return self.decoder(x)
        #return self.encoder(x)
               
        #return self.decoder(self.encoder(x))
    
    
    







#x=torch.randn(1,3,512,512)
x=torch.randn(1,4,32,32)
blok=AE(3,4)

print(summary(blok,(4,32,32)))
