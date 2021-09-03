from torch.utils.data import DataLoader, random_split
from omegaconf import OmegaConf
import data_loader.dataset as dataset
import utils
from utils import face_net_collate_fn
import models.face_crop
from custom.data_loader.dataset import ScarfAugDataset
config = OmegaConf.load("config.json")






class CustomDataLoader():
    def __init__(self):
        data = getattr(dataset, config["data_loader"]["type"])()
        
        trainset, validset = random_split(data, [int(len(data)*0.9), len(data)-int(len(data)*0.9)])
        self.trainset = trainset
        self.validset = validset
        if config["data_loader"]["collate_fn"]:
            my_collate_fn = getattr(utils, config["data_loader"]["collate_fn"])
            self.train_loader = DataLoader(dataset=trainset, collate_fn=my_collate_fn, **config["data_loader"]["args"])
            self.valid_loader = DataLoader(dataset=validset, collate_fn=my_collate_fn, **config["data_loader"]["args"])
        else:
            self.train_loader = DataLoader(dataset=trainset, **config["data_loader"]["args"])
            self.valid_loader = DataLoader(dataset=validset, **config["data_loader"]["args"])

    def gen_data_loader(self):
        return self.train_loader, self.valid_loader

    def num_of_data(self):
        return len(self.trainset), len(self.validset)
    



class AugmentationDataLoader():
    def __init__(self):
        self.dataset = ScarfAugDataset()
        self.train_loader = DataLoader(dataset=self.dataset, batch_size=1)

    def gen_data_loader(self):
        return self.train_loader

    def num_of_data(self):
        return len(self.dataset)