import torch
import torch.nn as  nn
from layers.conv import ConvBlock
from layers.dconv import DconvBlock
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder=nn.Sequential(
            ConvBlock( 3, 1, 1),
            ConvBlock(3, 1, 1),
            ConvBlock(3, 1, 1)
            )
        

    def forward(self, x):

        x = self.encoder(x)

        return x
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder=nn.Sequential(
            DconvBlock( 3, 1, 1),
            DconvBlock( 3, 1, 1),
            DconvBlock(3, 1, 1)
            ) 

    def forward(self,x):
        x=self.decoder(x)
        return x

class AE(nn.Module):
       def __init__(self):
           super().__init__()
           self.encoder=Encoder()
           self.decoder=Decoder()
       def forward(self,x,train,encode):
           if train:
               x=self.encoder(x)
               x=self.decoder(x)
               return x
           else:
               if encode:
                   x=self.encoder(x)
               else:
                   x=x.decode(x)

           return x

