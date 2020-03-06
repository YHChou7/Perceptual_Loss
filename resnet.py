import torch
import torch.nn as nn
from robustness import datasets, model_utils

class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        dataset = datasets.RestrictedImageNet('')
        model_kwargs = {
            "arch":'resnet50',
            'dataset' : dataset,
            'resume_path' : './RestrictedImageNet.pt',
            'parallel' : False
        } 
        model, ckpt = model_utils.make_and_restore_model(**model_kwargs)
        self.model = model.model
        for param in self.parameters():
            param.requires_grad = False


    def forward(self, x, select_layer):
        layers = {}
        #x = self.normalize(x)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x, fake_relu = True)
        
        for name in layers:
            if name not in select_layer:
                del layers[name]

        return layers
