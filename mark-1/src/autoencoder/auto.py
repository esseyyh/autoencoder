import torch
import torch.nn as  nn
from src.autoencoder.layers.conv import ConvBlock
from src.autoencoder.layers.dconv import DconvBlock
class Encoder(nn.Module):
    def __init__(self,encoder):
        super(Encoder, self).__init__()
        self.encoder=nn.Sequential(
            ConvBlock(encoder.first),
            ConvBlock(encoder.second),
            ConvBlock(encoder.third),
            )
        

    def forward(self, x):

        x = self.encoder(x)

        return x
class Decoder(nn.Module):
    def __init__(self,decoder):
        super(Decoder, self).__init__()
        self.decoder=nn.Sequential(
            DconvBlock( decoder.first),         
            DconvBlock( decoder.second),
            DconvBlock( decoder.third),
            ) 

    def forward(self,x):
        x=self.decoder(x)
        return x

class AE(nn.Module):
       def __init__(self,model_params):
           super().__init__()
           self.encoder=Encoder(model_params.encoder)
           self.decoder=Decoder(model_params.decoder)
       def forward(self,x,train,encode):
           if train:
               x=self.encoder(x)
               x=self.decoder(x)
               return x
           else:
               if encode:
                   x=self.encoder(x)
               else:
                   x=self.decoder(x)

           return x

