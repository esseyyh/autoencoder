import torch.nn as nn
from torch import Tensor

class DconvBlock(nn.Module):
    """Simple deconvolutional or transpose convolution block: ConvTranspose 2D -> BatchNorm -> Activation."""

   
    def __init__(self,Kernel,Stride,Padding,fan_in=3, fan_out=3): 
        """Constructs the Conv Transpose Block.

        Args:
            in_size (int): Size of input feature map.
            out_size (int): Size of output feature map.
            activation (Callable, optional): Activation function. Defaults to nn.ReLU.
            padding:int
            stride:int
        """
        super().__init__()

        self.conv = nn.ConvTranspose2d(fan_in, fan_out, kernel_size=Kernel, padding=Padding,stride=Stride)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')


        self.bn = nn.BatchNorm2d(fan_out)
        
        self.act = nn.ReLU()


    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.upsample(x)
        x = self.bn(x)
        
        x = self.act(x)
        

        return x
