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
import model.create_model
import dataset.custom_dataset
import model.loss.f1_loss

# Warning ignore
import warnings
warnings.filterwarnings("ignore")

# Image warning ignore
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# weight and bias
import wandb

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

# wandb init
wandb.login()
config = {
    'ALPHA' : None,
    'batch_size' : 32,
    'learning_rate' : 0.002,
}
wandb.init(project='test-model',config=config)
wandb.run.name = 'vit_tiny_patch16_224'

# model name
MODEL_NAME = 'vit_tiny_patch16_224'

# epochs
NUM_EPOCHS = 1

NUM_CLASSES = 18

PRETRAINED = True

################################################################################

dataset = dataset.custom_dataset.Custom_Dataset(path=TRAIN_PATH, target=TARGET, transforms=transform, train=True)

n_train = int(len(dataset) * 0.8)
n_val = int(len(dataset) * 0.2)
train_set, val_set = random_split(dataset, [n_train, n_val])
print(len(train_set), len(val_set))

train_loader = DataLoader(dataset=train_set, batch_size=config['batch_size'], shuffle=True, num_workers=4)

model = model.create_model.ImgClassifierModel(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=PRETRAINED)
model.to(device)

# criterion = model.loss.f1_loss.F1_loss().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

torch.cuda.empty_cache() # empty cache

model.train()

print('Train')
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    running_acc = 0.0
    n_iter = 0
    epoch_f1 = 0.0

    for index, (images, labels) in enumerate(train_loader):
        images = torch.stack(list(images), dim=0).to(device)
        labels = torch.tensor(list(labels)).to(device)

        optimizer.zero_grad()
        
        logits = model(images)
        _, preds = torch.max(logits, 1)

        # loss1 = criterion(logits, labels)
        # loss2 = F.cross_entropy(logits, labels)
        # loss = loss1 * (1 - config['ALPHA']) + loss2 * config['ALPHA']
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_acc += torch.sum(preds == labels.data)
        epoch_f1 += f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
        n_iter += 1
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_acc / len(train_loader.dataset)
    epoch_f1 = epoch_f1/n_iter

    print(f"Epoch: {epoch} -  Loss : {epoch_loss:.3f},  Accuracy : {epoch_acc:.3f},  F1-Score : {epoch_f1:.4f}")
    
    wandb.log({'acc' : epoch_acc, 'f1 score' : epoch_f1, 'loss' : epoch_loss,})
    
    # if epoch_loss < best_loss:
    #     PATH = '.input/data/checkpoint/' +"model_saved.pt"
    #     torch.save({'epoch': epoch,
    #                 'model_state_dict': model.state_dict(),
    #                 'optimizer_state_dict': optimizer.state_dict(),
    #                 'loss': loss,
    #                 }, PATH)
    #     best_loss = epoch_loss

################################################################################

val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=True, num_workers=4, drop_last=True)

with torch.no_grad():
    print("Calculating validation results...")
    model.eval()
    val_loss_items = []
    val_acc_items = []
    for val_batch in val_loader:
        inputs, labels = val_batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        outs = model(inputs)
        preds = torch.argmax(outs, dim=-1)

        loss_item = criterion(outs, labels).item()
        acc_item = (labels == preds).sum().item()
        val_loss_items.append(loss_item)
        val_acc_items.append(acc_item)

        wandb.log({'Image':
            [wandb.Image(inputs[idx], caption='pred : {pred}, truth : {label}'.format(pred=preds[idx].item(), label=labels[idx].item())) for idx in range(config['batch_size']) if preds[idx].item() != labels[idx].item()]
            })

    val_loss = np.sum(val_loss_items) / len(val_loader)
    val_acc = np.sum(val_acc_items) / len(val_set)
    
    print(f'val_acc : {val_acc}, val loss : {val_loss}')
