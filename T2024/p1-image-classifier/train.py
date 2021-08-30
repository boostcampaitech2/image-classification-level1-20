# Library
import os
from logging import critical
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
import loss.f1_loss
import loss.Focal_loss
import loss.Label_smoothing

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
PATH = '/opt/ml/input/data/checkpoint/model_saved.pt'


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
# wandb.login()
config = {
    'ALPHA' : None,
    'batch_size' : 32,
    'learning_rate' : 0.002,
}
# wandb.init(project='test-model',config=config)
# wandb.run.name = 'vit_tiny_patch16_224'

# model name
MODEL_NAME = 'vit_tiny_patch16_224'

# epochs
NUM_EPOCHS = 30

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

# Dataset, Dataloader
dataset = dataset.custom_dataset.Custom_Dataset(path=TRAIN_PATH, target=TARGET, transforms=transform, train=True)

n_train = int(len(dataset) * 0.8)
n_val = int(len(dataset) * 0.2)
train_set, val_set = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(dataset=train_set, batch_size=config['batch_size'], shuffle=True, num_workers=4)


# model
model = model.create_model.ImgClassifierModel(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=PRETRAINED)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])


# train
torch.cuda.empty_cache() # empty cache
model.train()
best_acc = 0.0

print('Train')
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    running_acc = 0.0
    n_iter = 0
    epoch_f1 = 0.0
    accumulate_step = 4

    for index, (images, labels) in enumerate(train_loader):
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

            logits = model(images)
            loss = criterion(logits, target_a) * lam + criterion(logits, targeb_b) * (1. - lam)

        else:
            logits = model(images)
            loss = criterion(logits, target)

        _, preds = torch.max(logits, 1)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_acc += torch.sum(preds == target.data)
        epoch_f1 += f1_score(target.cpu().numpy(), preds.cpu().numpy(), average='macro')
        n_iter += 1
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_acc / len(train_loader.dataset)
    epoch_f1 = epoch_f1/n_iter

    print(f"Epoch: {epoch} -  Loss : {epoch_loss:.3f},  Accuracy : {epoch_acc:.3f},  F1-Score : {epoch_f1:.4f}")
    
    # wandb.log({'acc' : epoch_acc, 'f1 score' : epoch_f1, 'loss' : epoch_loss,})
    
    if epoch_acc > best_acc:
        torch.save({'model_state_dict': model.state_dict()}, PATH)
        best_acc = epoch_acc

################################################################################

val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=True, num_workers=4, drop_last=True)

df = pd.DataFrame(columns=['pred', 'label'])

with torch.no_grad():
    print("Calculating validation results...")
    model.load_state_dict(torch.load(PATH)['model_state_dict'])
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

        # wandb.log({'Image':
        #     [wandb.Image(inputs[idx], caption='pred : {pred}, truth : {label}'.format(pred=preds[idx].item(), label=labels[idx].item())) for idx in range(config['batch_size']) if preds[idx].item() != labels[idx].item()]
        #     })
        
        for idx in range(config['batch_size']):
            df = df.append({'pred':preds[idx].item(), 'label':labels[idx].item()}, ignore_index=True)
            # if preds[idx].item() != labels[idx].item():
            #     df = df.append({'pred':preds[idx].item(), 'label':labels[idx].item()}, ignore_index=True)

    val_loss = np.sum(val_loss_items) / len(val_loader)
    val_acc = np.sum(val_acc_items) / len(val_set)
    
    print(f'val_acc : {val_acc:.3f}, val loss : {val_loss:.3f}')

    df.to_csv('/opt/ml/input/data/eval/difff1ce.csv',sep=',')


################################################################################

# Inference
class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

test_dir = '/opt/ml/input/data/eval'
submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
image_dir = os.path.join(test_dir, 'images')

# Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
transform = transforms.Compose([
    transforms.Resize((512, 384), Image.BILINEAR),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
])
dataset = TestDataset(image_paths, transform)

loader = DataLoader(
    dataset,
    shuffle=False
)

# 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
model.load_state_dict(torch.load(PATH)['model_state_dict'])
model.eval()

# 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
all_predictions = []
for images in loader:
    with torch.no_grad():
        images = images.to(device)
        pred = model(images)
        pred = pred.argmax(dim=-1)
        all_predictions.extend(pred.cpu().numpy())
submission['ans'] = all_predictions

# 제출할 파일을 저장합니다.
submission.to_csv(os.path.join(test_dir, 'submission.csv'), index=False)
print('test inference is done!')
