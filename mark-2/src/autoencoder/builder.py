import torch.nn as nn
import torch


class resnet(nn.Module):
    """creates a resnet block with optional dimension reduction """
   
    def __init__(self,in_channels: int,out_channels: int):
       
        super().__init__()
        

        
        self.layer1=nn.Conv2d(in_channels, out_channels,kernel_size=3, stride=1, padding=1)

        self.bn=nn.BatchNorm2d(in_channels)

        self.layer2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding= 1)
        self.act=nn.ReLU()


    def forward(self,x):
        out=self.layer1(x)

        out=self.layer2(out)
        
        out=self.layer1(x)

        out=self.layer2(out)

        out=out+x
        
        return out

class block(nn.Module):
    """ a simple block for convolutional block fro upsampling and downsampling """
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

        self.res_layers = [resnet(fan_out,
                       fan_out) 
                       for i in range (resnetlayer)
                       ]
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
            out=self.down_step(x)
        if self.up:
            out=self.up_step(self.up_step_c(x))
        return out



