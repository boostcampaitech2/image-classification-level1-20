import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import models.create_model

print ("PyTorch version:[%s]."%(torch.__version__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #'cuda:0'
print ("device:[%s]."%(device))

transform = transforms.Compose([
    transforms.Resize((512,384), Image.BILINEAR),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(0.5),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
])

MODEL_NAME = 'efficientnet_b4'

NUM_EPOCHS = 30

NUM_CLASSES = 18

PRETRAINED = True

config = {
    'ALPHA' : None,
    'batch_size' : 32,
    'learning_rate' : 0.002,
}


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
dataset = TestDataset(image_paths, transform)

loader = DataLoader(
    dataset,
    shuffle=False
)

model = models.create_model.ImgClassifierModel(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=PRETRAINED)
model.load_state_dict(torch.load('/opt/ml/input/data/checkpoint/model_saved_472.pt'))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

model.eval()
        
all_predictions = []
for images in loader:
    with torch.no_grad():
        images = images.to(device)
        pred = model(images)
        pred = pred.argmax(dim=-1)
        all_predictions.extend(pred.cpu().numpy())
submission['ans'] = all_predictions

# 제출할 파일을 저장합니다.
submission.to_csv(os.path.join(test_dir, 'submission2.csv'), index=False)
print('test inference is done!')
