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
    elif age<60:
        age=1
    else: age=2
    image_name = split_path[-1]
    if image_name[0]=='m':
        masked=0
    elif image_name[0]=='i':
        masked=1
    else: masked=2
    return masked, gender, age

def encoding(mask,gender,age):
    return mask*6 + gender*3 +age

    
class MaskDataset(Dataset):
    def __init__(self, im_dir_path, im_paths,transform, y_type='label'):
        self.im_paths=im_paths
        self.im_dir_path =im_dir_path
        self.shape =(3,512,384)
        self.train_transform = transform
        self.y_type=y_type
    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self,idx):
        X = Image.open(os.path.join(self.im_dir_path,self.im_paths[idx]))
        X= self.train_transform(X)
        
        mask ,gender,age = labeling(self.im_paths[idx])
        if self.y_type=='label':
            y=encoding(mask,gender,age)
        elif self.y_type=='age_code':
            y=age
        elif self.y_type=='mask_code':
            y=mask
        elif self.y_type=='gender_code':
            y=gender

        return X, y

class TestMaskDataset(Dataset):
    def __init__(self,im_dir_path, im_paths, transform):
        self.im_paths=self.im_paths=os.path.join(im_dir_path,im_paths)
        self.shape =(3,512,384)

        self.test_transform = transform


    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self,idx):
        X = Image.open(self.im_paths[idx])
        X= self.test_transform(X)
        return X

