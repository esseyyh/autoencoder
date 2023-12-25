import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


def reverse_transform(image):
    
    image = (image + 1) / 2

    image = image.permute(1, 2, 0)

    
    image = image * 255.

    
    image = image.cpu().numpy().astype(np.uint8)

    
    image = transforms.ToPILImage()(image)

    return image


class ten2img(nn.Module):
    def __call__(self, x):
        return reverse_transform(x)
