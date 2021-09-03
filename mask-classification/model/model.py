import torch.nn as nn
import torch
import torchvision
import timm
from torchvision.models import resnet50

class EfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNet, self).__init__()
        self.model = timm.create_model('efficientnet_b4', num_classes=num_classes, pretrained=True)
    
    def forward(self, x):
        output = self.model(x)
        return output

## resnet은 수정
class ResNet_Mask(nn.Module):
    def __init__(self, num_classes):
        super(ResNet_Mask,self).__init__()
        self.ft_model=resnet50(pretrained=True)
        self.num_classes=num_classes


        num_ftrs = self.ft_model.fc.in_features
        self.ft_model.fc=nn.Linear(num_ftrs, self.num_classes) 

    def forward(self, x):
        output = self.ft_model(x)
        return output

class ResNet_Gender(nn.Module):
    def __init__(self, num_classes):
        super(ResNet_Gender,self).__init__()
        self.ft_model=torchvision.models.resnet50(pretrained=True)


        num_ftrs = self.ft_model.fc.in_features
        self.ft_model.fc=nn.Linear(num_ftrs,num_classes) 

    def forward(self, x):
        output = self.ft_model(x)
        return output

class ResNet_Age(nn.Module):
    def __init__(self, num_classes):
        super(ResNet_Age,self).__init__()
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


