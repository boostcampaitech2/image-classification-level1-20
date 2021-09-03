# import library

import torch
import torchvision
from PIL import Image
import json

import os
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize

# Labeling Class

'''
input = path to root folder
output = csv file whose columns are (1) path to image and (2)label

'''

root = "input/data/train/images"
dirs_in_root = [x for x in os.listdir(root) if "._" not in x]

total = []

for dir_in_root in dirs_in_root:
    attr = dir_in_root.split("_") # sex : 1, age = -1
    path = os.path.join(root, dir_in_root)
    img_list = [x for x in os.listdir(path) if "._" not in x and "check" not in x]
        
    for img in img_list:
        path_ = os.path.join(path,img)
        if "mask" in img and "incorrect" not in img:
            mask = "Wear"
        elif "incorrect" in img:
            mask = "Incorrect"
        else:
            mask = "Not Wear"
        temp = [path_, attr[1], int(attr[-1]), mask]
        
        total.append(temp)
        
df = pd.DataFrame(total, columns = ["path", "sex", "age", "mask"])
df["age"] = pd.cut(df["age"], bins = [0, 30, 60, 90], right = False, labels = ["young","mid","old"])  
df["sex"] = df["sex"].map({"male":0, "female": 1})
df["mask"] = df["mask"].map({"Wear":0, "Incorrect": 1, "Not Wear" : 2})
df["age"] = df["age"].map({"young":0, "mid":1, "old" : 2}).astype(int)

df["class"] = df["mask"]*6 + df["sex"]*3 + df["age"]
df = df.drop(["sex","age","mask"], axis = 1)

group = df.groupby(df["class"])

# Dataset Customizing

'''
Data Imbalance is the most urgent problem to be addressed.

For every batch, the dataset makes a tensor made up of shuffled 18 images, each of which are from each class.

'''

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, group, transforms_ = None):
        super().__init__()
        self.group = group
        self.values_by_class = [self.group.get_group(i).values[:648,:] for i in range(18)] 
        #최대 648개로 맞추자. (왜? 중간값 and 18의 배수)
        self.lens = torch.FloatTensor([len(x) for x in self.values_by_class])
        self.transforms_ = transforms_
                
    def __len__(self):
        return int(self.lens[0])
    
    def __getitem__(self, idx):
        indices = torch.FloatTensor([idx]).expand(18)%self.lens
        X = []
        y = []
        
        for i in range(len(indices)):
            index_ = int(indices[i].item())
            img_path = self.values_by_class[i][index_]
            
            img = Image.open(img_path[0]).convert("RGB")
            
            if self.transforms_:
                img = self.transforms_(img)
            else:
                img = torchvision.transforms.ToTensor()(img)
                
            X.append(img)
            y.append(img_path[1])
            
        rand_ind = torch.randperm(18)
        
        X = torch.stack(X)[rand_ind]
        y = torch.LongTensor(y)[rand_ind]
        
        return X, y

# DataLoader

transforms_ = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(p = 0.5),
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize([.5,.5,.5],[.5,.5,.5])
])

dataset = Dataset(group, transforms_ = transforms_)

def collate_fn(samples):
    X = [sample[0] for sample in samples]
    y = [sample[1] for sample in samples]
    
    return torch.cat(X), torch.cat(y)

data_loader = torch.utils.data.DataLoader(dataset, batch_size = 4, shuffle = True, collate_fn = collate_fn)