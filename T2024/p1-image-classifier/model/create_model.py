import torch.nn as nn
import torch.nn.functional as F
import timm

class ImgClassifierModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            model_name=model_name, 
            num_classes=num_classes, 
            pretrained=pretrained)
        
    
    def forward(self, X):
        X = self.model(X)
        return X
