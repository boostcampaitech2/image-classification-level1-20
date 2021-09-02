import numpy as np
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset

class Custom_Dataset(Dataset):
    """[summary]

    Description:
        Dataset을 생성하는 class

    Args:
        Dataset ([type]): [description]
        path ([string]) : labeling이 되어있는 csv 파일 경로
        target ([string]) : target이 될 feature name
        transform ([transforms.Compose]) : image의 transform 
        train ([boolean]) : train 목적의 bool 값
    """
    
    def __init__(self, df, target, transforms, train=True):
        super().__init__()
        self.train = train
        self.data = df
        self.transforms = transforms
        self.target = target

        self.img_paths = self.data['path']
        self.labels = list(self.data[target])
        
        for i in range(len(self.labels)): # label들의 형태 변환
            self.labels[i] = int(self.labels[i])

    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, idx):
        X, y = None, None
        im = Image.open(self.img_paths[idx])

        if self.transforms is not None:
            y = self.labels[idx]
            X = self.transforms(im)

        if self.train:
            return X, y
        
        return X
