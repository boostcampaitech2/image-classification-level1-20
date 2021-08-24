import torch
import torch.nn as nn

class block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(block, self).__init__()
        # in_channels = out_channels
        self.expansion = 4
        # 1*1 zero-padding
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.relu = nn.ReLU()
        self.identity = nn.Identity()

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x = self.relu(x + identity)

        return x
# 제대로 동작되려면 이미지 사이즈 처음에 일단 무조건 224 * 224해야할듯
class ResNet(nn.Module): # [3, 4, 6, 3]
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, self.in_channels, kernel_size=7, stride=2)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
    
    def _make_layer(self, block, num_residual_blocks):
        layers = []

        for num in num_residual_blocks:
            layer = nn.Sequential()
            for i in range(1, num + 1):
                layer.add_module("block"+i, block(self.in_channels, 64, stride=1)) 