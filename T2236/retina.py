import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import os, cv2
from tqdm import tqdm
from retinaface import RetinaFace
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# mtcnn = MTCNN(keep_all=True, device=device)
# new_img_dir = '/opt/ml/input/data/train/new_imgs'
# img_path = '/opt/ml/input/data/train/images'

cnt = 0
crop_range = 70

    
# for paths in tqdm(os.listdir(img_path)):
#     if paths[0] == '.': continue
    
#     sub_dir = os.path.join(img_path, paths)
    
#     for imgs in os.listdir(sub_dir):
#         if imgs[0] == '.': continue
        
#         img_dir = os.path.join(sub_dir, imgs)
#         img = cv2.imread(img_dir)
#         img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
#         #mtcnn 적용
#         result_detected = RetinaFace.detect_faces(os.path.join(sub_dir, imgs))
#         print(result_detected)
#         break
#         tmp = os.path.join(new_img_dir, paths)
#         cnt += 1
#         plt.imsave(os.path.join(tmp, imgs), img)


result_detected = RetinaFace.detect_faces("/opt/ml/input/data/eval/images/14801c4fe0acac4ae0784068c7b11eb3c16c8700.jpg")
print(result_detected)   
# {'face_1': {'score': 0.9997658133506775, 'facial_area': [105, 138, 274, 348], 'landmarks': {'right_eye': [152.70848, 218.68748], 'left_eye': [232.8995, 225.29901], 'nose': [191.15756, 257.10056], 'mouth_right': [156.16864, 294.4676], 'mouth_left': [219.0949, 299.94876]}}}
img = cv2.imread("/opt/ml/input/data/eval/images/14801c4fe0acac4ae0784068c7b11eb3c16c8700.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
ymin=result_detected['face_1']['facial_area'][1]
ymax=result_detected['face_1']['facial_area'][3]
xmin=result_detected['face_1']['facial_area'][0]
xmax=result_detected['face_1']['facial_area'][2]
img = img[ymin:ymax, xmin:xmax, :]
plt.imsave("retina_img.jpg", img)
# new_img_dir = '/opt/ml/input/data/eval/new_imgs'
# img_path = '/opt/ml/input/data/eval/images'

# cnt = 0

# print("not yet")
    

# for imgs in tqdm(os.listdir(img_path)):
#     # print(imgs)
#     if imgs[0] == '.': continue
        
#     img_dir = os.path.join(img_path, imgs)
#     img = cv2.imread(img_dir)
#     img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
#         #mtcnn 적용
#     boxes,probs = mtcnn.detect(img)
        
#         # boxes 확인
#         # if len(probs) > 1: 
#             # print(boxes)
#     if not isinstance(boxes, np.ndarray):
#             # print('Nope!')
#             # 직접 crop
#         img=img[100:400, 50:350, :]
        
#         # boexes size 확인
#     else:
#         xmin = int(boxes[0, 0])-crop_range
#         ymin = int(boxes[0, 1])-crop_range
#         xmax = int(boxes[0, 2])+crop_range-10
#         ymax = int(boxes[0, 3])+crop_range-10
            
#         if xmin < 0: xmin = 0
#         if ymin < 0: ymin = 0
#         if xmax > 384: xmax = 384
#         if ymax > 512: ymax = 512
            
#         img = img[ymin:ymax, xmin:xmax, :]
        
            
#     img = cv2.resize(img,(300,300))

#     # for i in range(300):
#     #     if i >195 and i <225:
#     #         for j in range(300):
#     #             img[i][j] = np.array([0,0,0])
    
    
            
#     # tmp = os.path.join(new_img_dir, paths)
#     cnt += 1
#     plt.imsave(os.path.join(new_img_dir, imgs), img)
        
# print("done")