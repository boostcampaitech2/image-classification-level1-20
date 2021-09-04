import os
import pandas as pd
import numpy as np
import argparse
from PIL import Image
from importlib import import_module
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from omegaconf import OmegaConf
from sklearn.metrics import f1_score
from tqdm import tqdm

from data_loader.dataset import TestMaskDataset
from model.model import VIT, EfficientNet, resnet50
import data_transform


def get_model(model_name, num_classes, device): # import model
    module = getattr(import_module('model.model'), model_name)
    model = module(num_classes)
    model.to(device)

    return model


def main(config, model_name, fold):
    # Check torch version & cuda
    print ("PyTorch version:[%s]."%(torch.__version__))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #'cuda:0'
    print ("device:[%s]."%(device))


    # path
    test_dir = '/opt/ml/input/data/eval'
    submission = pd.read_csv(os.path.join(test_dir,'info.csv'))
    image_dir = os.path.join(test_dir, 'images')


    # load transforms
    transform_module = getattr(data_transform, config[model_name]["transform"])
    transform = transform_module()


    # dataset & dataloader
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    dataset = TestMaskDataset(image_paths, transform)
    loader = DataLoader(dataset, shuffle=False)


    # load model
    num_classes = config[model_name]['num_classes']
    model = get_model(model_name=model_name, num_classes=num_classes, device=device)
    model_saved = model_name + "model_saved" + str(fold) + ".pt" # fold는 어디서?
    pt_dir = os.path.join('/opt/ml/code/image-classification-level1-20/mask-classification/model_saved',model_saved)
    model.load_state_dict(torch.load(pt_dir))
    model.to(device)


    # inference
    model.eval()
    all_predictions = []
    for index, images in enumerate(tqdm(loader)):
        with torch.no_grad():
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())
    submission['ans'] = all_predictions


    # 제출할 파일을 저장합니다.
    submission_saved = f'submission_{model_name}.csv'
    submission.to_csv(os.path.join(test_dir, submission_saved), index=False)
    print(f'{model_name} test inference is done!')



if __name__ == '__main__':
    # argparser
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--fold', type=int, required=True)

    args = parser.parse_args()
    model_name = args.model
    fold = args.fold

    config = OmegaConf.load("/opt/ml/code/image-classification-level1-20/mask-classification/config.json")
    main(config, model_name, fold)