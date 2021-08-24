import time
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from omegaconf import OmegaConf
from trainer import Trainer
from data_loader.data_loaders import CustomDataLoader
from model.Resnet18 import Resnet18


SEED = 123
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)



def main(config, checkpoint=False):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    
    model = Resnet18()
    model = model.model
    optimizer = getattr(optim, config["optimizer"]["type"])(model.parameters(), **(config["optimizer"]["args"]))
    print(f'모델 로드 성공여부: {bool(checkpoint)}')
    if checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    
    
    lr_scheduler = getattr(optim.lr_scheduler, config["lr_scheduler"]["type"])(optimizer, **(config["lr_scheduler"]["args"]))
    criterion = getattr(nn, config["crit"])()
    loader = CustomDataLoader()
    data_loader, valid_data_loader = loader.gen_data_loader()
    epochs = config["argument"]["epochs"]

    num_trainset, num_validset = loader.num_of_data()


    trainer = Trainer(model, criterion, optimizer, lr_scheduler, device, data_loader, valid_data_loader, num_trainset, num_validset, epochs)

    trainer.train()

    now = time.localtime()
    filename = str('best_model/{}-{}-{}-{}-{}_bs-{}_lr-{}.pt'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, config["data_loader"]["args"]["batch_size"], config["optimizer"]["args"]["lr"]))
    torch.save(trainer.best_model, filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-bs', '--batch_size', type=int,
                      help='Batch size for input/valid data(ex.1, 2, 4, ..,16, 32, ..)', required=True)
    parser.add_argument('-lr', '--learning_rate', type=float,
                      help='Learning rate for training(ex.0.001)', required=True)
    parser.add_argument('-ep', '--epochs', type=int,
                      help='Number of iteration for total dataset(ex. 5, 10, ..)', required=True)
    
    parser.add_argument('-lm', '--load_model', type=str,
                      help='Load model by file name (ex. 2021-08-23-4-23_bs-16_lr-0.0001', default=False)
    
    args = parser.parse_args()


    with open('config.json', 'r') as jsonfile:
        data = json.load(jsonfile)
    
    data["data_loader"]["args"]["batch_size"] = args.batch_size
    data["optimizer"]["args"]["lr"] = args.learning_rate
    data["argument"]["epochs"] = args.epochs

    with open("config.json", "w") as jsonfile:
        json.dump(data, jsonfile)
    
    config = OmegaConf.load("config.json")
    print(f'checkpoint불러오기 성공여부: {bool(args.load_model)}')
    if args.load_model:
        checkpoint = torch.load(f".\\best_model\\{args.load_model}")
        main(config, checkpoint)
    main(config)

