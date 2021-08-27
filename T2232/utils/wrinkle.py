import cv2
import numpy as np

def wrinkle(img_path):
    g_y=np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    img = cv2.imread(img_path)
    img_horizon = cv2.filter2D(img,-1,g_y)
    ret, thresh = cv2.threshold(img_horizon,20,255,cv2.THRESH_BINARY)
    img1= cv2.add(img,img_horizon)
    img1= cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    return img1