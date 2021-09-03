import torchvision
import torch.nn as nn
import torch.nn.init as init




def initialize_weights(model):
    """
    Xavier uniform 분포로 모든 weight 를 초기화합니다.
    더 많은 weight 초기화 방법은 다음 문서에서 참고해주세요. https://pytorch.org/docs/stable/nn.init.html
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()



class Resnext50(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnext50_32x4d(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 18)
        )
        initialize_weights(self.model.fc)
    
    def __call__(self):
        return self.model
    
    def forward(self, x):
        return self.model(x)



class Resnext50_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnext50_32x4d(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 3)
        )
        initialize_weights(self.model.fc)
    
    def __call__(self):
        return self.model
    
    def forward(self, x):
        return self.model(x)
    


class Resnext50_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnext50_32x4d(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
        initialize_weights(self.model.fc)
    
    def __call__(self):
        return self.model
    
    def forward(self, x):
        return self.model(x)

