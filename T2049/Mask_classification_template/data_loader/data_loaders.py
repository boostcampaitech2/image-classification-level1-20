import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from skimage import io
from omegaconf import OmegaConf
import data_loader.dataset as dataset

config = OmegaConf.load("config.json")




class CustomDataLoader():
    def __init__(self):
        data = getattr(dataset, config["data_loader"]["type"])()
        trainset, validset = random_split(data, [int(len(data)*0.8), len(data)-int(len(data)*0.8)])
        self.trainset = trainset
        self.validset = validset
        self.train_loader = DataLoader(dataset=trainset, **config["data_loader"]["args"])
        self.valid_loader = DataLoader(dataset=validset, **config["data_loader"]["args"])

    def gen_data_loader(self):
        return self.train_loader, self.valid_loader

    def num_of_data(self):
        return len(self.trainset), len(self.validset)
    

