import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
import torchvision.transforms as transforms
import cv2

from skimage import io
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from PIL import ImageFile
from torchvision.transforms.transforms import ToPILImage



config = OmegaConf.load("config.json")


class MaskDataset(Dataset):
    def __init__(self):
        transform = transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.Resize((256, 256)),
                                            transforms.ToTensor(),
                                            transforms.ColorJitter(brightness=0.5),
                                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
                                        ])
        self.annotations = pd.read_csv("/opt/ml/code/custom/data/shuffle_augmented_train_data.csv")
        self.root_dir = "/opt/ml/code/custom/data/shuff_aug_mask_train_data"
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.loc[idx, 'file'])
        # Load an image from file -> ndarray
        image = io.imread(img_path)
        label = torch.tensor(int(self.annotations.loc[idx, 'label']))

        if self.transform:
            image = self.transform(image)
        return (image, label)



class MaskFaceCropDataset(Dataset):
    # OSError: image file is truncated
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    def __init__(self):
        transform = transforms.Compose([
                                            
                                            transforms.Resize((512, 384)),
                                            transforms.RandomHorizontalFlip(0.5),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
                                        ])
        self.annotations = pd.read_csv("/opt/ml/code/custom/data/shuffle_augmented_train_data.csv")
        self.root_dir = "/opt/ml/code/custom/data/face_crop"
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.loc[idx, 'file'])
        # Load an image from file -> ndarray
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(int(self.annotations.loc[idx, 'label']))

        if self.transform:
            image = self.transform(image)
        return (image, label)


class MaskAgeDataset(Dataset):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    def __init__(self):
        transform = transforms.Compose([
                                            transforms.Resize((512, 384)),
                                            transforms.RandomHorizontalFlip(0.5),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
                                        ])
        self.annotations = pd.read_csv("/opt/ml/code/custom/data/labeled_by_age.csv")
        self.root_dir = "/opt/ml/code/custom/data/face_crop"
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.loc[idx, 'file'])
        # Load an image from file -> ndarray
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(int(self.annotations.loc[idx, 'label']))

        if self.transform:
            image = self.transform(image)
        return (image, label)

class MaskGenderDataset(Dataset):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    def __init__(self):
        transform = transforms.Compose([
                                            transforms.Resize((512, 384)),
                                            transforms.RandomHorizontalFlip(0.5),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
                                        ])
        self.annotations = pd.read_csv("/opt/ml/code/custom/data/labeled_by_gender.csv")
        self.root_dir = "/opt/ml/code/custom/data/face_crop"
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.loc[idx, 'file'])
        # Load an image from file -> ndarray
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(int(self.annotations.loc[idx, 'label']))

        if self.transform:
            image = self.transform(image)
        return (image, label)



class MaskMaskDataset(Dataset):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    def __init__(self):
        transform = transforms.Compose([
                                            transforms.Resize((512, 384)),
                                            transforms.RandomHorizontalFlip(0.5),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
                                        ])
        self.annotations = pd.read_csv("/opt/ml/code/custom/data/labeled_by_mask.csv")
        self.root_dir = "/opt/ml/code/custom/data/face_crop"
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.loc[idx, 'file'])
        # Load an image from file -> ndarray
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(int(self.annotations.loc[idx, 'label']))

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
        label = torch.tensor(int(self.annotations.loc[idx, 'label']))

        if self.transform:
            image = self.transform(image)
        return (image, label)


class OldDataset(Dataset):
    def __init__(self, transforms, annotations, annotations_old, root_dir):
        self.transform = transforms
        self.annotations = annotations
        self.annotations_old = annotations_old
        self.root_dir = root_dir
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.loc[idx, 'file'])
        img_path_old = os.path.join(self.root_dir, self.annotations_old.loc[idx, 'file'])
        
        image = io.imread(img_path)
        image_old = io.imread(img_path_old)

        label = torch.tensor(int(self.annotations.loc[idx, 'label']))
        label_old = torch.tensor(int(self.annotations_old.loc[idx, 'label']))

        if self.transform:
            image = self.transform(image)
            image_old = self.transform(image_old)
        return ((image, label), (image_old, label_old))



class ScarfAugDataset(Dataset):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    def __init__(self):
        transform = transforms.Compose([
                                            transforms.Resize((512, 384)),
                                            transforms.RandomHorizontalFlip(0.5),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
                                        ])
        self.annotations = pd.read_csv("/opt/ml/code/custom/data/shuffle_augmented_train_data.csv")
        self.annotations_2 = pd.read_csv("/opt/ml/code/custom/data/scarf.csv")
        self.root_dir = "/opt/ml/code/custom/data/face_crop"
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.loc[idx, 'file'])
        img_path_2 = os.path.join("/opt/ml/code/custom/data/scarf", self.annotations_2.loc[idx, 'file'])
        # Load an image from file -> ndarray
        image = Image.open(img_path).convert('RGB')
        image_2 = Image.open(img_path_2).convert('RGB')
        label = torch.tensor(int(self.annotations.loc[idx, 'label']))

        if self.transform:
            image = self.transform(image)
            image_2 = self.transform(image_2)
        return (image, image_2, label)