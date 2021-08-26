import os
import pandas as pd
import numpy as np
import random
from PIL import Image
import tqdm
import timm

from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from torchvision import transforms
from torchvision import models
from torchvision.transforms import Resize, ToTensor, Normalize
import matplotlib.pyplot as plt

from data_set import TrainDataset_Gender

NUM_EPOCH = 10
BATCH_SIZE = 32
LEARNING_RATE =  0.002
phase = "train"
num_classes= 18

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  




# Dataset
data_df = pd.read_csv("/opt/ml/train_data_path_and_class.csv")
transform = transforms.Compose([
    Resize((512, 384), Image.BILINEAR),
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
])
dataset = TrainDataset_Gender(data_df,transform)

# DataLoader
loader = DataLoader(
    dataset,
    batch_size = BATCH_SIZE,
    shuffle=False
)


# Model
# model = models.densenet121(pretrained=True)
# num_ftrs = model.classifier.in_features
# model.classifier = nn.Linear(num_ftrs, num_classes)

model_arch = 'efficientnet_b3'
model =  timm.create_model(model_arch, num_classes = num_classes, pretrained=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# Train
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

for epoch in range(NUM_EPOCH):
    epoch_f1 = 0.0
    n_iter = 0.0
    running_loss = 0.0
    running_acc = 0.0


    model.train()
	
		
    for ind, (images, labels) in enumerate(tqdm.tqdm(loader, leave=False)):
        
        images = torch.stack(list(images), dim=0).to(device)
        labels = torch.tensor(list(labels)).to(device)

        optimizer.zero_grad()  


        logits = model(images)
        _, preds = torch.max(logits, 1)
        loss = criterion(logits, labels)

			
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_acc += torch.sum(preds == labels.data)
        epoch_f1 += f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
        n_iter += 1


		# 한 epoch이 모두 종료되었을 때,
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)
    epoch_f1 = epoch_f1/n_iter

    print(f"Epoch: {epoch} -  Loss : {epoch_loss:.3f},  Accuracy : {epoch_acc:.3f},  F1-Score : {epoch_f1:.4f}")
print("Train Finished!")


# # print("Model's state_dict:")
# # for param_tensor in model.state_dict():
# #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# # ### 옵티마이저의 state_dict 출력
# # print("Optimizer's state_dict:")
# # for var_name in optimizer.state_dict():
# #     print(var_name, "\t", optimizer.state_dict()[var_name])

# # PATH = model_arch +"model_saved.pt"
# # torch.save({'epoch': NUM_EPOCH,
# #             'model_state_dict': model.state_dict(),
# #             'optimizer_state_dict': optimizer.state_dict(),
# #             'loss': loss,
# #             }, PATH)


# class TestDataset(Dataset):
#     def __init__(self, img_paths, transform):
#         self.img_paths = img_paths
#         self.transform = transform

#     def __getitem__(self, index):
#         image = Image.open(self.img_paths[index])

#         if self.transform:
#             image = self.transform(image)
#         return image

#     def __len__(self):
#         return len(self.img_paths)

# test_dir = '/opt/ml/input/data/eval'
# submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
# image_dir = os.path.join(test_dir, 'images')

# # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
# image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
# transform = transforms.Compose([
#     Resize((512, 384), Image.BILINEAR),
#     ToTensor(),
#     Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
# ])
# dataset = TestDataset(image_paths, transform)

# loader = DataLoader(
#     dataset,
#     shuffle=False
# )

# # 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
# model.eval()

# # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
# all_predictions = []
# for images in loader:
#     with torch.no_grad():
#         images = images.to(device)
#         pred = model(images)
#         pred = pred.argmax(dim=-1)
#         all_predictions.extend(pred.cpu().numpy())
# submission['ans'] = all_predictions

# # 제출할 파일을 저장합니다.
# submission.to_csv(os.path.join(test_dir, 'submission_eff.csv'), index=False)
# print('test inference is done!')



