from collections import namedtuple

from torchvision import models
import torch 
import torch.nn as nn

from robustness import datasets
from robustness.tools import helpers

class vgg16(nn.Module):
    def __init__(self, requires_grad = False):
        super(vgg16, self).__init__()
        vgg = models.vgg16(pretrained= True).features
        self.model = vgg
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    def forward(self, x, select_layer):
        layers = {}
        #vgg_outputs = namedtuple("VggOutputs", select_layer)
        i = 0 
        for layer in self.model.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                layer = nn.ReLU(inplace = False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
                #layer = nn.AvgPool2d(kernel_size = 2, stride = 1)
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'
            else:
                raise(f"Unrecognized layer: {layer.__class__.__name__}")
            
            x = layer(x)
            
            if name in select_layer:
                layers[name] = x
                
            if len(layers) == len(select_layer) :
                break
        
        return layers

