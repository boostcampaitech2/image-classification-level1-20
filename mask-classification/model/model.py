import torch.nn as nn
import torch
import torchvision
import timm


class EfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNet, self).__init__()
        self.model = timm.create_model('efficientnet_b4', num_classes=num_classes, pretrained=True)
    
    def forward(self, x):
        output = self.model(x)
        return output

## resnet은 수정
class resnet50(nn.Module):
    def __init__(self, num_classes):
        super(resnet50).__init__()
        self.ft_model=torchvision.models.resnet50(pretrained=True)


        num_ftrs = self.ft_model.fc.in_features
        self.ft_model.fc=nn.Linear(num_ftrs,num_classes) 

    def forward(self, x):
        output = self.ft_model(x)
        return output

class VIT(nn.Module):


    def __init__(self, num_classes):
        super(VIT, self).__init__()
        self.model = timm.create_model('vit_tiny_patch16_224', num_classes=num_classes, pretrained=True)
    
    def forward(self, x):
        output = self.model(x)
        return output


