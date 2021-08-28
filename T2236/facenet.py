import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import os, cv2
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
new_img_dir = '/opt/ml/input/data/train/new_imgs'
img_path = '/opt/ml/input/data/train/images'

cnt = 0
crop_range = 70

    
for paths in tqdm(os.listdir(img_path)):
    if paths[0] == '.': continue
    
    sub_dir = os.path.join(img_path, paths)
    
    for imgs in os.listdir(sub_dir):
        if imgs[0] == '.': continue
        
        img_dir = os.path.join(sub_dir, imgs)
        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        #mtcnn 적용
        boxes,probs = mtcnn.detect(img)
        
        # boxes 확인
        # if len(probs) > 1: 
            # print(boxes)
        if not isinstance(boxes, np.ndarray):
            # print('Nope!')
            # 직접 crop
            img=img[100:400, 50:350, :]
        
        # boexes size 확인
        else:
            xmin = int(boxes[0, 0])-crop_range
            ymin = int(boxes[0, 1])-crop_range
            xmax = int(boxes[0, 2])+crop_range-10
            ymax = int(boxes[0, 3])+crop_range-10
            
            if xmin < 0: xmin = 0
            if ymin < 0: ymin = 0
            if xmax > 384: xmax = 384
            if ymax > 512: ymax = 512
            
            img = img[ymin:ymax, xmin:xmax, :]
            
        img = cv2.resize(img,(300,300))

        for i in range(300):
            if i >195 and i <225:
                for j in range(300):
                    img[i][j] = np.array([0,0,0])
        sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, sharpening)
        tmp = os.path.join(new_img_dir, paths)
        cnt += 1
        plt.imsave(os.path.join(tmp, imgs), img)
        


new_img_dir = '/opt/ml/input/data/eval/new_imgs'
img_path = '/opt/ml/input/data/eval/images'

cnt = 0

print("not yet")
    

for imgs in tqdm(os.listdir(img_path)):
    # print(imgs)
    if imgs[0] == '.': continue
        
    img_dir = os.path.join(img_path, imgs)
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        #mtcnn 적용
    boxes,probs = mtcnn.detect(img)
        
        # boxes 확인
        # if len(probs) > 1: 
            # print(boxes)
    if not isinstance(boxes, np.ndarray):
            # print('Nope!')
            # 직접 crop
        img=img[100:400, 50:350, :]
        
        # boexes size 확인
    else:
        xmin = int(boxes[0, 0])-crop_range
        ymin = int(boxes[0, 1])-crop_range
        xmax = int(boxes[0, 2])+crop_range-10
        ymax = int(boxes[0, 3])+crop_range-10
            
        if xmin < 0: xmin = 0
        if ymin < 0: ymin = 0
        if xmax > 384: xmax = 384
        if ymax > 512: ymax = 512
            
        img = img[ymin:ymax, xmin:xmax, :]
        
            
    img = cv2.resize(img,(300,300))

    for i in range(300):
        if i >195 and i <225:
            for j in range(300):
                img[i][j] = np.array([0,0,0])
    sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = cv2.filter2D(img, -1, sharpening)
    
            
    # tmp = os.path.join(new_img_dir, paths)
    cnt += 1
    plt.imsave(os.path.join(new_img_dir, imgs), img)
        
print("done")