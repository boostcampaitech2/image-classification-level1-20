


# # 모델 자체에 이식시켜야함 -> 그래야 eval에서 할 때도 적용되니까
# # facenet
# import torch
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from facenet_pytorch import MTCNN
# import os
# import cv2
# import torchvision.transforms as transforms
# from skimage import io

# def run():
# # eval에서는 for문 빼고 하나 들어올 때마다 하나씩 반환
#     def face_detection(img_dir, img_csv, new_img_dir):
#         device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

#         mtcnn = MTCNN(keep_all=True, device=device)

#         # '/opt/ml/code/custom/data/shuff_aug_mask_train_data/'
#         img_dir = img_dir

#         # image1623_label_12.png, 12
#         img_df = pd.read_csv(img_csv)

#         # '/opt/ml/code/custom/data/face_crop/'
#         new_img_dir = new_img_dir
        
#         cnt = 0

#         for i in range(len(img_df)):
#             img_name = img_df.loc[i, 'file']
#             img_label = img_df.loc[i, 'class']
#             img_path = os.path.join(img_dir, img_name)
#             img = cv2.imread(img_path)
#             img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#             # 픽셀값의 최댓값이 200이하이면 밝게해주기
#             if max(img.reshape(-1)) < 200:
#                 img += 50


                

#             #mtcnn 적용            
#             boxes, props = mtcnn.detect(img)

#             # boxes 확인
#             # if len(probs) > 1: 
#             #     print(f"boxes: {boxes}")
#             if not isinstance(boxes, np.ndarray):
#                 cnt += 1
#                 print(f'I can\'t detect face {img_name}!')
#                 print(type(img))
#                 print(img.shape)
#                 print(img)

#                 # 직접 crop
#                 img = transforms.ToPILImage()(img)
#                 img = transforms.ToTensor()(img)
#                 img = img.numpy()

#                 # imsave가 채널이 끝에 오길바람
#                 img = np.transpose(img , [1, 2, 0])
#                 # print(f'my crop img shape: {img.shape}')
#                 print(img.shape)

            
#             # boexes size 확인
#             else:
#                 xmin = int(boxes[0, 0])-30
#                 ymin = int(boxes[0, 1])-30
#                 xmax = int(boxes[0, 2])+30
#                 ymax = int(boxes[0, 3])+30
                
#                 if xmin < 0: xmin = 0
#                 if ymin < 0: ymin = 0
#                 if xmax > 384: xmax = 384
#                 if ymax > 512: ymax = 512
                
#                 img = img[ymin:ymax, xmin:xmax, :]
#                 print(f"detected img shape: {img.shape}")
                
            
#             plt.imsave(os.path.join(new_img_dir, img_name), img)
        
#         print(cnt)

#             # return img, img_label


#     face_detection("/opt/ml/code/custom/data/shuff_aug_mask_train_data/", "/opt/ml/code/custom/data/shuffle_augmented_train_data.csv", "/opt/ml/code/custom/data/face_crop")

# run()