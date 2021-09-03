import random
import numpy as np

import torch
import torchvision

import os


from importlib import import_module
##torch==1.7.1

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  
seed_everything()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_EPOCH = 30
BATCH_SIZE = 16
LEARNING_RATE =  0.002
num_classes= 18
validation_ratio = 0.1
model_name = 'EfficientNet'




def get_model(model_name, device):
    module = getattr(import_module('model'),model_name)
    model = module(num_classes)
    model.to(device)

    return model



def train (model):
    pass


