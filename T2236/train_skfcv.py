import os
import pandas as pd
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
import timm

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

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

from data_set import TrainDataset, Dataset_kfold

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
BATCH_SIZE = 16
LEARNING_RATE =  0.002
num_classes= 18


# Transforms for Dataset
train_transform = transforms.Compose([
    Resize((512,384), Image.BILINEAR),
    ToTensor(),
    RandomHorizontalFlip(0.5),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
])   
valid_transform = transforms.Compose([
    Resize((512,384), Image.BILINEAR),
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
])   


# Model
model_arch = 'efficientnet_b4'
model =  timm.create_model(model_arch, num_classes = num_classes, pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


kfold = StratifiedKFold(n_splits=5, shuffle=True)
train_df = pd.read_csv("/opt/ml/canny_new_train_data_path_and_class.csv")
x_train = train_df['path'].to_numpy()
y_train = train_df['label'].to_numpy()


criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

total_acc = 0
for fold, (train_index, test_index) in enumerate(kfold.split(x_train, y_train)):
        
    x_train_fold = x_train[train_index]
    x_test_fold = x_train[test_index]
    y_train_fold = y_train[train_index]
    y_test_fold = y_train[test_index]

    train = Dataset_kfold(x_train_fold, y_train_fold,train_transform)
    test = Dataset_kfold(x_test_fold, y_test_fold,valid_transform)
    train_loader = DataLoader(train, batch_size = BATCH_SIZE, shuffle = False)
    test_loader = DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)

    for epoch in range(NUM_EPOCH):
        print('\nEpoch {} / {} \nFold number {} / {}'.format(epoch + 1, NUM_EPOCH, fold + 1 , kfold.get_n_splits()))
        correct = 0
        model.train()
        for batch_index, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = torch.stack(list(x_batch), dim=0).to(device)
            y_batch = torch.tensor(list(y_batch)).to(device)

            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            pred = torch.max(out.data, dim=1)[1]
            correct += (pred == y_batch).sum()
            if (batch_index + 1) % 16 == 0:
                print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                    (batch_index + 1)*len(x_batch), len(train_loader.dataset),
                    100.*batch_index / len(train_loader), loss.data, float(correct*100) / float(BATCH_SIZE*(batch_index+1))))
    total_acc += float(correct*100) / float(BATCH_SIZE*(batch_index+1))
total_acc = (total_acc / kfold.get_n_splits())
print('\n\nTotal accuracy cross validation: {:.3f}%'.format(total_acc))





# Model Save
PATH = model_arch +"model_saved_eff_best_b4_epoch_30_with_canny.pt"
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
image_dir = os.path.join(test_dir, 'canny_new_imgs')

# Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
transform = transforms.Compose([
    Resize((512,384), Image.BILINEAR),
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
])
dataset = TestDataset(image_paths, transform)

loader = DataLoader(
    dataset,
    shuffle=False
)

# 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
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
submission.to_csv(os.path.join(test_dir, 'submission_new_eff_b4_8_31_eph30_with_canny.csv'), index=False)
print('test inference is done!')



