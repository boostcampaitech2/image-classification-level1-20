import os
import pandas as pd
import numpy as np
import random
from PIL import Image
import tqdm
import timm

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from itertools import cycle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

from torchvision import transforms
from torchvision import models
from torchvision.transforms import Resize, ToTensor,RandomErasing, Normalize,RandomHorizontalFlip, ColorJitter
import matplotlib.pyplot as plt

from data_set import TrainDataset

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  
seed_everything()



# Parameter
NUM_EPOCH = 30
BATCH_SIZE = 32
LEARNING_RATE =  0.002
num_classes= 18
validation_ratio = 0.1




# Dataset
data_df = pd.read_csv("/opt/ml/new_train_data_path_and_class.csv")
train_transform = transforms.Compose([
    Resize((300,300), Image.BILINEAR),
    ToTensor(),
    RandomHorizontalFlip(0.5),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
])   # ColorJitter(contrast=(0.1),brightness=(0.1),hue = 0.1),
# valid_transform = transforms.Compose([
#     Resize((300,300), Image.BILINEAR),
#     ToTensor(),
#     Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
# ])
train_dataset = TrainDataset(data_df,train_transform)
old_dataset = TrainDataset(data_df,train_transform)



# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size = BATCH_SIZE,
    shuffle= True,
    drop_last=True
)
old_loader = DataLoader(
    old_dataset,
    batch_size = BATCH_SIZE,
    shuffle= True,
    drop_last=True
)



# Model
model_arch = 'efficientnet_b4'
model =  timm.create_model(model_arch, num_classes = num_classes, pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)




# Train
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
kfold = StratifiedKFold(n_splits=5, shuffle=True)
for epoch in range(NUM_EPOCH):
    epoch_f1 = 0.0
    n_iter = 0.0
    running_loss = 0.0
    running_acc = 0.0


    model.train()
	
	
    for (i, data), (j, data_60) in tqdm.tqdm(zip(enumerate(train_loader, 0),cycle(enumerate(old_loader, 0))),total=len(train_loader)):

        images, labels = data
        images = torch.stack(list(images), dim=0).to(device)
        labels = torch.tensor(list(labels)).to(device)
            
        inputs_60, labels_60 = data_60
        inputs_60 = inputs_60.to(device)
        labels_60 = labels_60.to(device)

        with torch.cuda.amp.autocast():

            if np.random.random() >= 0.75 :
                rand_index = torch.randperm(inputs_60.size()[0]) # ????????? ????????? label
                target_a = labels # ?????? ????????? label
                target_b = labels_60[rand_index] # ?????? ????????? label  
                lam = np.random.beta(1.0, 1.0)     
                # bbx1, bby1, bbx2, bby2 = rand_bbox(inputs_60.size(), lam)

                    # ?????? ???????????? ????????? ??????
                images[:, :, :, :150] = inputs_60[rand_index, :, :, :150]

                    # ?????? ???????????? ?????? ???????????? ?????? ??????
                lam = 0.5

                    # ????????? ????????? 1???
                logits = model(images)
                _, preds = torch.max(logits, 1)
                    # ?????? ???????????? ???????????? ?????? ???????????? ???????????? ?????? loss ?????????
                loss = criterion(logits, target_a) * lam + criterion(logits, target_b) * (1. - lam)
            else:
                logits = model(images)
                _, preds = torch.max(logits, 1)
                loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_acc += torch.sum(preds == labels.data)
        epoch_f1 += f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
        n_iter += 1


	# ??? epoch??? ?????? ??????????????? ???,
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_acc / len(train_loader.dataset)
    epoch_f1 = epoch_f1/n_iter

    print(f"Epoch: {epoch} -  Loss : {epoch_loss:.3f},  Accuracy : {epoch_acc:.3f},  F1-Score : {epoch_f1:.4f}")

    # correct = 0
    # total = 0
    # for i, data in enumerate(valid_loader, 0):
    #     inputs, labels = data
    #     inputs, labels = inputs.to(device), labels.to(device)
    #     outputs = model(inputs)
        
    #     _, predicted = torch.max(outputs.data, 1)
    #     total += labels.size(0)
    #     correct += (predicted == labels).sum().item()

    # print(f"Epoch: {epoch} -  Loss : {epoch_loss:.3f},  Accuracy : {epoch_acc:.3f},  F1-Score : {epoch_f1:.4f}, Validation ACC:{correct/total:.3f}")


# Model Save
PATH = model_arch +"model_saved_eff_best_b4_epoch_30_with_cutmix7.pt"
torch.save({'epoch': NUM_EPOCH,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, PATH)


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
image_dir = os.path.join(test_dir, 'new_imgs')

# Test Dataset ????????? ????????? ???????????? DataLoader??? ????????????.
image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
transform = transforms.Compose([
    Resize((300,300), Image.BILINEAR),
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
])
dataset = TestDataset(image_paths, transform)

loader = DataLoader(
    dataset,
    shuffle=False
)

# ????????? ???????????????. (????????? ????????? ????????? torch.load??? ????????? ???????????????!)
model.eval()

# ????????? ????????? ??????????????? ???????????? ????????? ???????????????.
all_predictions = []
for images in loader:
    with torch.no_grad():
        images = images.to(device)
        pred = model(images)
        pred = pred.argmax(dim=-1)
        all_predictions.extend(pred.cpu().numpy())
submission['ans'] = all_predictions

# ????????? ????????? ???????????????.
submission.to_csv(os.path.join(test_dir, 'submission_new_eff_b4_8_31_eph30_with_cutmix7.csv'), index=False)
print('test inference is done!')



