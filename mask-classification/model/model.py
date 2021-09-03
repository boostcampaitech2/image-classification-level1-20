import torch.nn as nn
import torch
import torchvision
import timm


class EfficientNet(nn.Module):


    def __init__(self, num_classes):
        super(EfficientNet, self).__init__()
        self.model = timm.create_model('tf_efficientnet_b4', num_classes=num_classes, pretrained=True)
    
    def forward(self, x):
        output = self.model(x)
        return output

## resnet은 수정
class resnet34(nn.Module):
    def __init__(self, num_classes):
        super(resnet34).__init__()
        self.ft_model=torchvision.models.resnet34(pretrained=True)


        num_ftrs = self.ft_model.fc.in_features
        self.ft_model.fc=nn.Linear(num_ftrs,num_classes)  

    def forward(self, x):
        output = self.ft_model(x)
        return output