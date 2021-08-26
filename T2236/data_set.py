import torch
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class TrainDataset(Dataset):
    def __init__(self, data_df,transform):
        self.data_df = data_df
        self.img_paths = list(data_df["path"])
        self.label = list(data_df["label"])
        self.transform = transform
            
    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        
                
        return image, self.label[index]

    def __len__(self):
        return len(self.img_paths)

class TrainDataset_Gender(Dataset):
    def __init__(self, data_df,transform):
        self.data_df = data_df
        self.img_paths = list(data_df["path"])
        self.label = list(data_df["gender_code"])
        self.transform = transform
            
    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        
                
        return image, self.label[index]

    def __len__(self):
        return len(self.img_paths)


class TrainDataset_Age(Dataset):
    def __init__(self, data_df,transform):
        self.data_df = data_df
        self.img_paths = list(data_df["path"])
        self.label = list(data_df["age_code"])
        self.transform = transform
            
    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        
                
        return image, self.label[index]

    def __len__(self):
        return len(self.img_paths)