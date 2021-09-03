from PIL import Image
import torch
from torch.utils.data.dataloader import Dataset
import torchvision
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor,RandomErasing, Normalize,RandomHorizontalFlip, ColorJitter
from sklearn.model_selection import train_test_split
import pandas as pd
import os.path
import glob


def labeling(path):
    split_path= path.split('/')
    directory = split_path[-2].split('_')
    gender= 0 if (directory[1])[0]=='m' else 1
    age = (int(directory[-1]))
    if age<30:
        age=0
    elif age<50:
        age=1
    else: age=2
    image_name = split_path[-1]
    mask=[0 for _ in range(3)]
    if image_name[0]=='m':
        mask=0
    elif image_name[0]=='i':
        mask=1
    else: mask=2
    return mask, gender, age

def encoding(mask,gender,age):
    return mask*6 + gender*3 +age



class MaskDataset(Dataset):
    def __init__(self,im_paths):
        self.im_paths=im_paths
        self.shape =(3,512,384)

        self.train_transform = transforms.Compose([
        Resize((512,384), Image.BILINEAR),
        ToTensor(),
        RandomHorizontalFlip(0.5),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
])
    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self,idx):
        
        X = Image.open(self.im_paths[idx])
        X= self.train_transform(X)
        
        mask ,gender,age = labeling(self.im_paths[idx])
        label = encoding(age,gender,age)
        return X, label

class TestMaskDataset(Dataset):
    def __init__(self,im_paths):
        self.im_paths=im_paths
        self.shape =(3,512,384)

        self.train_transform = transforms.Compose([
        Resize((512,384), Image.BILINEAR),
        ToTensor(),
        RandomHorizontalFlip(0.5),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
])
    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self,idx):
        X = Image.open(self.im_paths[idx])
        X= self.train_transform(X)
        return X





