import torch
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import os, cv2

mtcnn = MTCNN(keep_all=True, device = torch.device("cuda:0"))

def get_img(img_path: str):
    """
    Args:
        img_path: train image의 디렉토리
    
    Returns:
        np.ndarray, (512, 384, 3), RGB 이미지
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def img_resize(img: np.ndarray, x: int, y: int):
    """
    img를 (x, y) 크기로 resize (3차원 이상이라면 뒤의 차원은 유지)

    Args:
        img: np.ndarray 타입의 이미지
        x: int
        y: int
    
    Returns:
        np.ndarray, (x, y, ...) 타입의 이미지
    """
    img = cv2.resize(img, dsize=(x, y), interpolation=cv2.INTER_AREA)
    return img

def face_crop(img: np.ndarray):
    """
    (512, 384, 3), RGB image를 매개변수로 받아서 얼굴 부분만 crop한 뒤 resize해서 리턴하는 함수

    Args:
        img: np.ndarray 타입의 (512, 384, 3), RGB 이미지 (get_img 이용 추천)
    
    Returns:
        np.ndarray, (224, 224, 3), RGB 이미지
    """
    box, prob = mtcnn.detect(img)
    if not isinstance(box, np.ndarray):
        img = img[100:400, 50:350, :]
    else:
        x_min = max(int(box[0, 0]) - 30, 0)
        y_min = max(int(box[0, 1]) - 30, 0)
        x_max = min(int(box[0, 2]) + 30, 384)
        y_max = min(int(box[0, 3]) + 30, 512)
        img = img[y_min:y_max, x_min:x_max, :]
    
    img = img_resize(img, 224, 224)
    return img

def face_crop_and_save(path_in, path_out):
    for img_name in os.listdir(path_in):
        img = get_img(os.path.join(path_in, img_name))
        img = face_crop(img)
        plt.imsave(os.path.join(path_out, img_name), img)

if __name__ == "__main__":
    face_crop_and_save(r'/opt/ml/mask-classification/data/data',
                       r'/opt/ml/mask-classification/data/face_cropped')
