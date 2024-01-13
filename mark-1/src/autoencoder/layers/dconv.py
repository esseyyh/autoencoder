import torch.nn as nn
from torch import Tensor

class DconvBlock(nn.Module):
    """Simple deconvolutional or transpose convolution block: ConvTranspose 2D -> BatchNorm -> Activation."""

   
    def __init__(self,decoder): 
        """Constructs the Conv Transpose Block.

        Args:
            in_size (int): Size of input feature map.
            out_size (int): Size of output feature map.
            activation (Callable, optional): Activation function. Defaults to nn.ReLU.
            padding:int
            stride:int
        """
        super().__init__()

        self.conv = nn.ConvTranspose2d(decoder.fan_in, decoder.fan_out, kernel_size=decoder.kernel, padding=decoder.padding,stride=decoder.stride)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1=nn.Conv2d(decoder.fan_out,decoder.fan_out,kernel_size=decoder.kernel,padding=decoder.padding,stride=decoder.stride)
        self.conv2=nn.Conv2d(decoder.fan_out,decoder.fan_out,kernel_size=decoder.kernel,padding=decoder.padding,stride=decoder.stride)


        self.bn = nn.BatchNorm2d(decoder.fan_out)
        
        self.act = nn.ReLU()


    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        x = self.bn(x)
        
        x = self.act(x)
        

        return x
