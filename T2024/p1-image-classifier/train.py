# Library
import numpy as np
import pandas as pd
import random
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize

from sklearn.metrics import f1_score

# user library
import dataset.custom_dataset
import model.loss.f1_loss

# Warning ignore
import warnings
warnings.filterwarnings("ignore")

# Image warning ignore
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

################################################################################

# seed 고정
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
set_seed(42)

# PyTorch version & Check using cuda
print ("PyTorch version:[%s]."%(torch.__version__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #'cuda:0'
print ("device:[%s]."%(device))

################################################################################

# data.csv, info.csv path
TRAIN_PATH = '/opt/ml/input/data/train/data.csv'
EVAL_PATH = '/opt/ml/input/data/eval'

# target
TARGET = 'class'

# transforms
transform = transforms.Compose([
    transforms.Resize((512, 384), Image.BILINEAR),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
])

BATCH_SIZE = 32

# model name
MODEL_NAME = 'vit_tiny_patch16_224'

# epochs
NUM_EPOCHS = 100

# lr_rate
LEARNING_RATE = 0.002

# loss weight
ALPHA = 0.8

NUM_CLASSES = 18

PRETRAINED = True

################################################################################

train_dataset = dataset.custom_dataset.Custom_Dataset(path=TRAIN_PATH, target=TARGET, transforms=transform, train=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = model(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=PRETRAINED)
model.to(device)

criterion = model.loss.f1_loss.F1_loss().to(device)
optimizer = torch.optim.Adam()

torch.cuda.empty_cache() # empty cache

model.train()

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    running_acc = 0.0
    n_iter = 0
    epoch_f1 = 0.0

    for index, (images, labels) in enumerate(train_dataloader):
        images = torch.stack(list(images), dim=0).to(device)
        labels = torch.tensor(list(labels)).to(device)

        optimizer.zero_grad()
        
        logits = model(images)
        _, preds = torch.max(logits, 1)

        loss1 = criterion(logits, labels)
        loss2 = F.cross_entropy(logits, labels)
        loss = loss1 * (1 - ALPHA) + loss2 * ALPHA

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_acc += torch.sum(preds == labels.data)
        epoch_f1 += f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
        n_iter += 1
    
    epoch_loss = running_loss / len(train_dataloader.dataset)
    epoch_acc = running_acc / len(train_dataloader.dataset)
    epoch_f1 = epoch_f1/n_iter

    print(f"Epoch: {epoch} -  Loss : {epoch_loss:.3f},  Accuracy : {epoch_acc:.3f},  F1-Score : {epoch_f1:.4f}")
    if epoch_loss < best_loss:
        PATH = './checkpoint/' +"model_saved.pt"
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, PATH)
        best_loss = epoch_loss

################################################################################

