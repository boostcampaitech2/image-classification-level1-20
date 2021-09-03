import torch
import torch.nn as nn
import numpy as np
from facenet_pytorch import MTCNN
import cv2
import torchvision.transforms as transforms
import torch.nn.functional as F



class F1_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.classes = 18
        self.epsilon = 1e-7
    
    def forward(self, y_pred, y_true):
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean(), f1.mean()



def wrinkle(img, label):
    g_y=np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_horizon = cv2.filter2D(img,-1,g_y)


    img= cv2.add(img, img_horizon)

    img = torch.from_numpy(img)

    return img, label



def face_net_collate_fn(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(keep_all=True, device=device) 
    
    img_list, label_list = [], []
    
    for img, label in batch:
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 픽셀값의 최댓값이 200이하이면 밝게해주기
        if max(img.reshape(-1)) < 200:
            img += 50


        #mtcnn 적용
        img = img.numpy()
        img = np.transpose(img, [1, 2, 0])
        boxes,probs = mtcnn.detect(img)

        # boxes 확인
        if len(probs) > 1: 
            print(boxes)
        if not isinstance(boxes, np.ndarray):


            # 직접 crop
            img = img.astype(np.uint8)
            # img = transforms.ToPILImage()(img)
            # img = transforms.CenterCrop(200)(img)
            # img = transforms.ToTensor()(img)
            # img = img.numpy()

            # imsave가 채널이 끝에 오길바람
            # img = np.transpose(img, [1, 2, 0])
            # print(f'my crop img shape: {img.shape}')

        
        # boexes size 확인
        else:
            xmin = int(boxes[0, 0])-30
            ymin = int(boxes[0, 1])-30
            xmax = int(boxes[0, 2])+30
            ymax = int(boxes[0, 3])+30
            
            if xmin < 0: xmin = 0
            if ymin < 0: ymin = 0
            if xmax > 384: xmax = 384
            if ymax > 512: ymax = 512
            
            img = img[ymin:ymax, xmin:xmax, :]
            

        # plt.imsave(os.path.join("/opt/ml/code/custom/data/face_crop/", f"abc_{cnt}.png"), img)

        
        img = torch.from_numpy(img)
        img = img.to(device)
        label = label.to(device)


        img_list.append(img)
        label_list.append(label)

    img_list = torch.stack(img_list)
    img_list = img_list.to(device)

    label_list = torch.LongTensor(label_list)

    label_list = label_list.to(device)
    
    

    return img_list, label_list


# 중앙을 중심으로 지킬앤 하이드 처럼 좌우에 컷믹스
def rand_bbox(size, lam):
    H = size[2]
    W = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randn() + W//2
    cy = np.random.randn() + H//2

    bbx1 = np.clip(cx - cut_w // 2, 0, W//2)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W//2)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return int(bbx1), int(bby1), int(bbx2), int(bby2)

