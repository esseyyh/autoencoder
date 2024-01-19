import torch.nn as nn
import torch
from torchsummary import summary 
import math
import torch.nn.functional as F

class resnet(nn.Module):
    """ResNet block """
   
    def __init__(self,
                 fan_in : int,
                 fan_out: int,
                 resnet_groups:int =32
                 ):
        super().__init__()
        self.layer1=nn.Conv2d(fan_in, fan_out,kernel_size=3, stride=1, padding=1)
        self.gn1=nn.GroupNorm(resnet_groups,fan_in)
        self.layer2=nn.Conv2d(fan_out,fan_out,kernel_size=3,stride=1,padding= 1) 
        self.gn2=nn.GroupNorm(resnet_groups,fan_out)
        self.act=nn.SiLU()
        
        if fan_in==fan_out:
            self.proj_layer=nn.Identity()
        else:
            self.proj_layer = nn.Conv2d(fan_in,fan_out, kernel_size=1, stride=1, padding=0)

            
        
    def forward(self,x):    
    
        out=self.layer1(self.act(self.gn1(x)))
        out=self.layer2(self.act(self.gn2(out)))
        

        
        return (out +  self.proj_layer(x) )
            

        
            

class block(nn.Module):
    """ a stack of resnet blocks for VAE """
    def __init__(self,
                 fan_in:int,    
                 fan_out:int,
                 resnetlayer :int,
                 resnet_groups:int =32,
                 down:bool = False,
                 up:bool=False):

        super().__init__()
        self.down=down
        self.up=up
        

        self.res_layers = nn.ModuleList([resnet(fan_in,
                       fan_out,resnet_groups=resnet_groups) 
                       for i in range (resnetlayer)
                       ])
        if down :
            self.down_step = nn.Conv2d(fan_out,fan_out,3,2,0)
        if up:
            self.up_step_c= nn.Conv2d(fan_out,fan_out,3,1,1)
            self.up_step = nn.Upsample(scale_factor=2,mode='bilinear')
    def forward(self,x):

        

        for res in self.res_layers:
            x=res(x)
          
        

        if self.down:
            x=F.pad(x,(0,1,0,1))
            x=self.down_step(x)
        if self.up:
            x=self.up_step(self.up_step_c(x))
        return x

