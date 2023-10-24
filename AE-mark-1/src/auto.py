import torch
import torch.nn as  nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 3, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 3, 3, padding=1)
        self.conv3 = nn.Conv2d(3, 3, 3, padding=1)
        self.conv4 = nn.Conv2d(3, 3, 3, padding=1)
        
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.maxpool(x)

        x = self.conv4(x)
       

        return x
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv1 = nn.ConvTranspose2d(3, 3, 3, padding=1)
        self.conv2 = nn.ConvTranspose2d(3, 3, 3, padding=1)
        self.conv3 = nn.ConvTranspose2d(3, 3 ,3, padding=1)
        self.conv4 = nn.ConvTranspose2d(3, 3, 3, padding=1)
        

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.upsample(x)
       
        x = self.conv2(x)
        x = self.upsample(x)
        
        x = self.conv3(x)
        x = self.upsample(x)
        
        x = self.conv4(x)

        return x

class AE(nn.Module):
       def __init__(self):
           super(AE,self).__init__()
           self.encoder=Encoder()
           self.decoder=Decoder()
       def forward(self,x,train,encode):
           if train:
               x=self.encoder(x)
               x=self.decoder(x)
               return x
           else:
               if encode:
                   x=self.encode(x)
               else:
                   x=x.decode(x)

           return x
