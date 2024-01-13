import torch.nn as nn
import torch
from torchsummary import summary 
import math

class resnet(nn.Module):
    """ResNet block """
   
    def __init__(self,
                 fan_in : int,
                 fan_out: int
                 ):
        super().__init__()
        self.layer1=nn.Conv2d(fan_in, fan_out,kernel_size=3, stride=1, padding=1)
        self.bn1=nn.BatchNorm2d(fan_in)
        self.layer2=nn.Conv2d(fan_out,fan_out,kernel_size=3,stride=1,padding= 1) 
        self.bn2=nn.BatchNorm2d(fan_out)
        self.act=nn.ReLU()
        
        self.proj_layer = nn.Conv2d(fan_in,fan_out, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
    
        out=self.act(self.layer1(self.bn1(x)))
        out=self.act(self.layer2(self.bn2(out)))
        

        if self.proj_layer is not None:
             return (out + (self.proj_layer(x)) )
        else:
            return (out +  x )
            

        
            

class block(nn.Module):
    """ a stack of resnet blocks for VAE """
    def __init__(self,
                 fan_in:int,    
                 fan_out:int,
                 resnetlayer :int,
                 down:bool = False,
                 up:bool=False):

        super().__init__()
        self.down=down
        self.up=up
        self.start=nn.Conv2d(fan_in,fan_out,3,1,1)

        self.res_layers = nn.ModuleList([resnet(fan_out,
                       fan_out) 
                       for i in range (resnetlayer)
                       ])
        if down :
            self.down_step = nn.Conv2d(fan_out,fan_out,3,2,1)
        if up:
            self.up_step_c= nn.Conv2d(fan_out,fan_out,3,1,1)
            self.up_step = nn.Upsample(scale_factor=2,mode='bilinear')
    def forward(self,x):

        x=self.start(x)

        for res in self.res_layers:
            x=res(x)
          
        

        if self.down:
            x=self.down_step(x)
        if self.up:
            x=self.up_step(self.up_step_c(x))
        return x






