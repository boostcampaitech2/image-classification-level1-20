# BoostCamp P-stage 1 Image Classification - Team 20
## Install Requirements
`pip install -r requirements.txt` 

## Data Description 
- train images: 2700*7 = 18900
- test images: 1800*7 = 12600
- image size: 384 x 512
- classes: 18

## Performance
- Public LB: 0.732
- Private LB: 0.717

## Train
- Train을 실행하기 전 utils의 facenet.py를 먼저 실행해야 한다. 
- 이는 사진의 얼굴부분만 자른 사진을 new Images라는 폴더에 저장하도록 하였다.
`python train.py --model [model_name]` 

### models
- EfficientNet_b4 (model_name:  EfficientNet)
- vit_tiny_patch16_224 (model_name:VIT)
- resnet34 (mask, gender, age) (model_name:  ResNet_Mask, ResNet_Age, ResNet_Gedner)

### train augmentations
개별 모델마다 input image에 대해 다른 augmentation을 적용함으로써 모델을 다양하게 사용하고자 하였다. 또한 상대적으로 성능이 잘 나오지 않은 ViT에 Cutmix를 이용해 데이터를 늘림으로써 모델의 성능을 높이고자 하였다.
EfficientNet : Resize, RandomHorizontalFlip, Normalize
ResNet : RandomHorizontalFlip, Normalize
ViT : Resize, CenterCrop, RandomHorizontalFlip, Normalize
```python
class Efficient_transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((300,300), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),

        ])
    def __call__(self,X):
        return self.transform(X)
```
```python
class ResNet_transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),

        ])
    def __call__(self,X):
        return self.transform(X)
```
```python
class VIT_transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((512,384), Image.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
        ])
    def __call__(self,X):
        return self.transform(X)
```
```python
- Custom CutMix(All models)
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
if cutmix:
                if np.random.random() > 0.5: # Cutmix
                    random_index = torch.randperm(x_batch.size()[0])
                    target_a = y_batch
                    targeb_b = y_batch[random_index]

                    lam = np.random.beta(1.0, 1.0)
                    bbx1, bby1, bbx2, bby2 = rand_bbox(x_batch.size(), lam)

                    x_batch[:, :, bbx1:bbx2, bby1:bby2] = x_batch[random_index, :, bbx1:bbx2, bby1:bby2]
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x_batch.size()[-1] * x_batch.size()[-2]))

                    logits = model(x_batch.float())
                    loss = criterion(logits, target_a) * lam + criterion(logits, targeb_b) * (1. - lam)

                    _, preds = torch.max(logits, 1)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                else:
                    logits = model(x_batch.float())
                    loss = criterion(logits, y_batch)

                    _, preds = torch.max(logits, 1)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

```
## Inference
`python inference.py --mdoel [model_name] --fold [fold]`

### Ensemble
학습된 EfficientNet, Resnet, VisionTransformer에 대해 각각 K-fold Ensemble을 진행한 뒤, 

```python
p = [
        [0.2, 0.2, 0.2], # mask
        [0.4, 0.15, 0.15], # gender
        [0.3, 0.3, 0.3] # age
    ]
    
```
로 각 모델에서 mask, gender, age를 개별 가중치로 합산.

