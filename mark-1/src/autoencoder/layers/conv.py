import torch.nn as nn
from torch import Tensor


class ConvBlock(nn.Module):
    """Simple convolutional block: Conv2D -> BatchNorm -> Activation -> Max pooling(2)."""

    def __init__(self,encoder): 
        """Constructs the ConvBlock.

        Args:
            in_size (int): Size of input feature map. 
            out_size (int): Size of output feature map.
            activation (Callable, optional): Activation function. Defaults to nn.ReLU.
            padding:int
            stride:int

        """
        super().__init__()
                                                                                                                                                        
        self.conv = nn.Conv2d(encoder.fan_in,encoder.fan_out, kernel_size=encoder.kernel, padding=encoder.padding,stride=encoder.stride)
        self.bn = nn.BatchNorm2d(encoder.fan_out)
        self.conv1 = nn.Conv2d(encoder.fan_out,encoder.fan_out, kernel_size=encoder.kernel, padding=encoder.padding,stride=encoder.stride)
        self.conv2 = nn.Conv2d(encoder.fan_in,encoder.fan_out, kernel_size=encoder.kernel, padding=encoder.padding,stride=encoder.stride)
        self.act = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.bn(x)
        x = self.act(x)
      

        return x
