import torch.nn as nn
from torch import Tensor


class ConvBlock(nn.Module):
    """Simple convolutional block: Conv2D -> BatchNorm -> Activation."""

    def __init__(self,Kernel,Stride,Padding,fan_in=3, fan_out=3): 
        """Constructs the ConvBlock.

        Args:
            in_size (int): Size of input feature map.
            out_size (int): Size of output feature map.
            activation (Callable, optional): Activation function. Defaults to nn.ReLU.
            padding:int
            stride:int
        """
        super().__init__()
                                                                                                                                                        
        self.conv = nn.Conv2d(fan_in, fan_out, kernel_size=Kernel, padding=Padding,stride=Stride)
        self.bn = nn.BatchNorm2d(fan_out)
        
        self.act = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.bn(x)
        x = self.act(x)
      

        return x
