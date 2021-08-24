import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from skimage import io
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize

config = OmegaConf.load("config.json")

class CatsAndDogsDataset(Dataset):
    def __init__(self):
        transform = transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.Resize((256, 256)),
                                            transforms.RandomCrop((224, 224)),
                                            transforms.RandomRotation(degrees=45),
                                            transforms.ColorJitter(brightness=0.5),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.ToTensor()
                                        ])
        self.annotations = pd.read_csv("catanddog.csv")
        self.root_dir = "C:\\DeepLearning\\data\\catanddog"
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.loc[idx, 'image'])
        # Load an image from file -> ndarray
        image = io.imread(img_path)
        label = torch.tensor(int(self.annotations.loc[idx, 'label']))

        if self.transform:
            image = self.transform(image)
        return (image, label)



class MNISTDataset(Dataset):
    def __init__(self):
        self.data = torchvision.datasets.MNIST('data/MNIST/', # 다운로드 경로 지정
                                                    train=True, # True를 지정하면 훈련 데이터로 다운로드
                                                    transform=transforms.ToTensor(), # 텐서로 변환
                                                    download=True, 
                                                    )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0]/255, self.data[index][1]


class CIFAR100Dataset(Dataset):
    def __init__(self):
        self.data = torchvision.datasets.CIFAR100('data/CIFAR100/', 
                                                    train=True, 
                                                    transform=transforms.ToTensor(), 
                                                    download=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0]/255, self.data[index][1]  




class MaskDataset(Dataset):
    def __init__(self):
        transform = transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.Resize((256, 256)),
                                            transforms.RandomCrop((224, 224)),
                                            transforms.RandomRotation(degrees=45),
                                            transforms.ColorJitter(brightness=0.5),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.ToTensor()
                                        ])
        self.annotations = pd.read_csv("C:\\DeepLearning\\data\MASK\\train\\mask_train_data.csv")
        self.root_dir = "C:\\DeepLearning\\data\\MASK\\train\\images\\"
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.loc[idx, 'file'])
        # Load an image from file -> ndarray
        image = io.imread(img_path)
        label = torch.tensor(int(self.annotations.loc[idx, 'class']))

        if self.transform:
            image = self.transform(image)
        return (image, label)


class MaskAugmentationDataset(Dataset):
    def __init__(self, transforms, annotations, root_dir):
        self.transform = transforms
        self.annotations = annotations
        self.root_dir = root_dir
        
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.loc[idx, 'file'])
        # Load an image from file -> ndarray
        image = io.imread(img_path)
        label = torch.tensor(int(self.annotations.loc[idx, 'class']))

        if self.transform:
            image = self.transform(image)
        return (image, label)




class TestDataset(Dataset):
    def __init__(self, root_dir, annotations, transform):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotations)
        self.transform = transform
        

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.loc[index, 'ImageID'])
        image = io.imread(img_path)

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.annotations)