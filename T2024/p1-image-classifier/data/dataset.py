import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

TRAIN_PATH = './input/data/train/data.csv'

class Custom_Dataset(Dataset):
    def __init__(self, path, target, transform, train=True):
        self.train = train
        self.path = path
        self.transform = transform

        self.data = pd.read_csv(self.path, index_col=0)
        self.classes = np.sort(self.data[target].unique())

        self.y = self.data[target]
        self.X = []
        for path in self.data['path']:
            im = Image.open(path)
            self.X.append(im)
    

    def __len__(self):
        return len(self.y)
    

    def __getitem__(self, index):
        X, y = self.X[index], self.y[index]
        if self.transform:
            X = self.transform(X)
        return X, torch.tensor(y, dtype=torch.long)
