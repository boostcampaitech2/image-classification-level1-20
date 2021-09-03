import cv2
import numpy as np
import torch


def wrinkle(img, label):
    g_y=np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_horizon = cv2.filter2D(img,-1,g_y)


    img= cv2.add(img, img_horizon)

    img = torch.from_numpy(img)

    return img, label
    # ret, thresh = cv2.threshold(img1,120, 200, cv2.THRESH_TOZERO_INV)
#     print(thresh.shape)
#     print(thresh)
#     pil_image=Image.fromarray(thresh)
#     pil_image.show("horizon", thresh)
#     return img1, thresh