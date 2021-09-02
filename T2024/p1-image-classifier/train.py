# Library
from ast import Str
from itertools import accumulate
import os
from logging import critical
import numpy as np
import pandas as pd
import random
from PIL import Image
from pandas.core.frame import DataFrame
from scipy.sparse.construct import rand

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast_mode, grad_scaler

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from adamp import AdamP

from tqdm import tqdm

# user library
import models.create_model
import dataset.custom_dataset
import losses.f1_loss
import losses.Focal_loss
import losses.Label_smoothing

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
    
set_seed(42)

# PyTorch version & Check using cuda
print ("PyTorch version:[%s]."%(torch.__version__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #'cuda:0'
print ("device:[%s]."%(device))

################################################################################

# data.csv, info.csv path
TRAIN_PATH = '/opt/ml/input/data/train/data.csv'
EVAL_PATH = '/opt/ml/input/data/eval'
PATH = '/opt/ml/input/data/checkpoint'


# target
TARGET = 'class'

# transforms
transform = transforms.Compose([
    transforms.Resize((512,384), Image.BILINEAR),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(0.5),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
])

# wandb init
# wandb.login()
config = {
    'ALPHA' : None,
    'batch_size' : 32,
    'learning_rate' : 0.002,
}
# wandb.init(project='test-model',config=config)
# wandb.run.name = 'mlp_mixer'

# model name
MODEL_NAME = 'efficientnet_b4'

# epochs
NUM_EPOCHS = 20

NUM_CLASSES = 18

PRETRAINED = True

# 중앙을 중심으로 지킬앤 하이드 처럼 좌우에 컷믹스
def rand_bbox(size, lam):
    H = size[2]
    W = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)


    cx = np.random.randn() + W//2
    cy = np.random.randn() + H//2

    # 패치의 4점
    bbx1 = np.clip(cx - cut_w // 2, 0, W//2)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W//2)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return int(bbx1), int(bby1), int(bbx2), int(bby2)

################################################################################

df = pd.read_csv(TRAIN_PATH, index_col=0)


# model
from mlp_mixer_pytorch import MLPMixer

# train
torch.cuda.empty_cache() # empty cache

stratifiedKFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

print('train start')

for index, (train_idx, val_idx) in enumerate(stratifiedKFold.split(df['path'],df['class'])):
    if index == 0: continue
    print('{idx}th stratifiedKFold'.format(idx=index))

    df_train = pd.DataFrame(columns=['path','class'])
    df_val = pd.DataFrame(columns=['path','class'])

    for i in train_idx:
        df_train = df_train.append({'path':df['path'].loc[i], 'class':df['class'].loc[i]}, ignore_index=True)
    
    for i in val_idx:    
        df_val = df_val.append({'path':df['path'].loc[i], 'class':df['class'].loc[i]}, ignore_index=True)

    train_set = dataset.custom_dataset.Custom_Dataset(df_train, target=TARGET, transforms=transform, train=True)
    val_set = dataset.custom_dataset.Custom_Dataset(df_val, target=TARGET, transforms=transform, train=True)

    train_loader = DataLoader(train_set, batch_size=32, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=32, num_workers=4)


    model = models.create_model.ImgClassifierModel(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=PRETRAINED)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    for s in ['train', 'validation']:
        if s == 'train':
            model.train()

            scaler = grad_scaler.GradScaler()
            
            best_f1, best_epoch = 0., 0.
            stop_count = 0

            for epoch in range(NUM_EPOCHS):
                running_loss = 0.0
                running_acc = 0.0
                n_iter = 0
                epoch_f1 = 0.0
                
                for index, (images, labels) in enumerate(tqdm(train_loader)):
                    images = images.to(device)
                    target = labels.to(device)

                    optimizer.zero_grad()
                    
                    if np.random.random() > 0.5: # Cutmix
                        random_index = torch.randperm(images.size()[0])
                        target_a = target
                        targeb_b = target[random_index]

                        lam = np.random.beta(1.0, 1.0)
                        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)

                        images[:, :, bbx1:bbx2, bby1:bby2] = images[random_index, :, bbx1:bbx2, bby1:bby2]
                        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))

                        with autocast_mode.autocast():
                            logits = model(images.float())
                            loss = criterion(logits, target_a) * lam + criterion(logits, targeb_b) * (1. - lam)

                            _, preds = torch.max(logits, 1)
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                    else:
                        with autocast_mode.autocast():
                            logits = model(images.float())
                            loss = criterion(logits, target)

                            _, preds = torch.max(logits, 1)
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()


                    running_loss += loss.item() * images.size(0)
                    running_acc += torch.sum(preds == target.data)
                    epoch_f1 += f1_score(target.cpu().numpy(), preds.cpu().numpy(), average='macro')
                    n_iter += 1
                
                epoch_loss = running_loss / len(train_loader.dataset)
                epoch_acc = running_acc / len(train_loader.dataset)
                epoch_f1 = epoch_f1 / n_iter

                print(f"epoch: {epoch} -  Loss : {epoch_loss:.3f},  Accuracy : {epoch_acc:.3f},  F1-Score : {epoch_f1:.4f}")

                # wandb.log({'acc' : epoch_acc, 'f1 score' : epoch_f1, 'loss' : epoch_loss,})

                if epoch_f1 > best_f1:
                    stop_count = 0
                    dir = f'/opt/ml/input/data/checkpoint'
                    torch.save(model.state_dict(), f'{dir}/model_saved_{epoch}.pt')

                    best_f1 = epoch_f1
                    best_epoch = epoch
                else:
                    stop_count += 1
                    if stop_count > 2:
                        break

        else:
            load_dir = os.path.join(dir, 'model_saved_{best_epoch}.pt')
            model.load_state_dict(torch.load(load_dir))
            
            model.eval()
            with torch.no_grad():
                val_loss_items = []
                val_acc_items = []
                val_f1_score = 0.0
                v_iter = 0

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
                    val_f1_score += f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
                    v_iter += 1

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_set)
                val_f1 = np.sum(val_f1_score) / v_iter

            print(f'val_acc : {val_acc:.3f}, val loss : {val_loss:.3f}, val f1 : {val_f1:.3f}')
            # wandb.log({'val_acc' : val_acc, 'val_loss' : val_loss, 'val_f1' : val_f1})

    del model, optimizer, val_loader, scaler
    torch.cuda.empty_cache()
    
    

################################################################################

