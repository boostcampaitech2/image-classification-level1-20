import torch.nn as nn
import torchvision.models

class Resnet18:
    def __init__(self):
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(512, 18)
    
    def gen(self):
        return self.model
