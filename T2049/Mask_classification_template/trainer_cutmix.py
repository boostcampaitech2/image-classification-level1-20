import time
import torch
import numpy as np
import wandb
import torchvision
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import rand_bbox
from sklearn.metrics import f1_score, accuracy_score
from pytorchtools import EarlyStopping


class TrainerCutmix():
    def __init__(self, model, config, criterion_1, criterion_2, optimizer, lr_scheduler, 
                device, data_loader, valid_data_loader, num_trainset, num_validset, epochs):
        self.device = device
        self.model = model.to(self.device)
        self.criterion_1 = criterion_1
        self.criterion_2 = criterion_2
        self.optimizer = optimizer
        self.scaler = torch.cuda.amp.GradScaler()
        self.lr_scheduler = lr_scheduler
        self.data_loader = data_loader
        self.valid_loader = valid_data_loader
        self.epochs = epochs

        self.train_loss = []
        self.valid_loss = []
        self.valid_predict_list = np.array([])
        self.valid_target_list = np.array([])
        self.train_total_loss = 0
        self.valid_total_loss = 0
        self.train_correct = 0
        self.valid_correct = 0
        self.num_trainset = num_trainset
        self.num_validset = num_validset
        self.best_model = {'state_dict': None, 'optimizer': None}

        now = time.localtime()
        filename = str('/opt/ml/code/custom/best_model/final_linear_model-{}-{}-{}-{}-{}_bs-{}_lr-{}.pt'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min,  config["data_loader"]["args"]["batch_size"], config["optimizer"]["args"]["lr"]))
        self.early_stopping = EarlyStopping(patience=2, verbose=True, path=filename)


        
    
    def train_per_epoch(self, epoch):
        self.model.train()
        self.train_total_loss = 0
        self.train_correct = 0
        print(self.device)
        for data, target in tqdm(self.data_loader, leave=False):

            data = data.to(self.device)
            target = target.to(self.device)
            with torch.cuda.amp.autocast():
                if np.random.random() > 0.5:

                    rand_index = torch.randperm(data.size()[0]) # 패치에 사용할 label
                    target_a = target # 원본 이미지 label
                    target_b = target[rand_index] # 패치 이미지 label  
                    lam = np.random.beta(1.0, 1.0)     
                    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
                    data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))


                    pred = self.model(data)
                    loss_1 = self.criterion_1(pred, target_a) * lam + self.criterion_1(pred, target_b) * (1. - lam)
                    loss_2 = self.criterion_2(pred, target_a)[0] * lam + self.criterion_2(pred, target_b)[0] * (1. - lam)
                    
                else:
                    pred = self.model(data)
                    loss_1 = self.criterion_1(pred, target)
                    loss_2 = self.criterion_2(pred, target)[0]
            
                loss = 0.2 * loss_1 + 0.8 * loss_2
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            # AssertionError: No inf checks were recorded for this optimizer
            # Scaler (optimizer) looks for parameters used in the graph which is empty, hence, the error.
            try:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            except:
                pass
            

            self.train_correct += sum(pred.argmax(dim=1) == target)
            self.train_total_loss += loss.item()
        self.train_total_loss /= self.num_trainset
        self.train_loss.append(self.train_total_loss)
        self.lr_scheduler.step()
        print(f"{epoch}/{self.epochs}Epochs {self.train_correct}/{self.num_trainset} train accuracy: {self.train_correct/self.num_trainset:.4f}, loss: {self.train_total_loss:.4f}")
 
    
    def valid_per_epoch(self, epoch):
        self.model.eval()
        with torch.no_grad():
            self.valid_ce_loss = self.valid_f1_loss = epoch_f1 = 0.0
            self.valid_correct = 0
            loop = tqdm(self.valid_loader, total=len(self.valid_loader))

            for data, target in loop:
                data = data.to(self.device)
                target = target.to(self.device)
                if np.random.random() > 0.5:

                    rand_index = torch.randperm(data.size()[0])
                    target_a = target # 원본 이미지 label
                    target_b = target[rand_index] # 패치 이미지 label  
                    lam = np.random.beta(1.0, 1.0)     
                    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
                    data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

                    pred = self.model(data)
                    loss_1 = self.criterion_1(pred, target_a) * lam + self.criterion_1(pred, target_b) * (1. - lam)
                    loss_2 = self.criterion_2(pred, target_a)[0] * lam + self.criterion_2(pred, target_b)[0] * (1. - lam)

                else:
                    pred = self.model(data)
                    loss_1 = self.criterion_1(pred, target)
                    loss_2 = self.criterion_2(pred, target)[0]


                print([pred.argmax(dim=1), target])



                self.valid_predict_list = np.concatenate((self.valid_predict_list, pred.argmax(dim=1).cpu().numpy()))
                self.valid_target_list = np.concatenate((self.valid_target_list, target.cpu().numpy()))


                # metric도 모듈화 하고 싶은데 아직 감이 안온다
                self.valid_correct += sum(pred.argmax(dim=1) == target)
                self.valid_ce_loss += loss_1.item()
                self.valid_f1_loss += loss_2.item()
                self.valid_total_loss += (0.8 * loss_2.item()) + (0.2 * loss_1.item())
                epoch_f1 += self.criterion_2(pred, target)[1]
            # example_images.append(wandb.Image(data[0], caption=f"Pred: {pred[0].item()}, Truth: {target[0]}"))
            self.valid_ce_loss /= self.num_validset
            self.valid_f1_loss /= self.num_validset
            self.valid_total_loss /= self.num_validset
            self.valid_loss.append(self.valid_total_loss)
            accuracy = accuracy_score(y_true=self.valid_target_list, y_pred=self.valid_predict_list)
            f1 = f1_score(y_true=self.valid_target_list, y_pred=self.valid_predict_list, average="macro")
            print(f"{epoch}/{self.epochs}Epochs {self.valid_correct}/{self.num_validset} valid accuracy: {accuracy:.4f}, f1 score: {f1}, CrossEntropy loss: {self.valid_ce_loss:.4f}, F1 loss: {self.valid_f1_loss}")
            
            
            return self.valid_total_loss, accuracy, f1


    def train(self):

        writer = SummaryWriter("logs/CIFAR100")

        for epoch_index in range(1, self.epochs + 1):
            self.train_per_epoch(epoch_index)
            valid_loss, accuracy, f1 = self.valid_per_epoch(epoch_index)
            wandb.log({'Valid accuracy': accuracy, 'Valid f1 score': f1, 'Valid loss': valid_loss})

            # early_stop
            self.early_stopping(self.valid_total_loss, self.model, self.optimizer)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

            # if valid_loss < lowest_loss:
            #     lowest_loss = valid_loss
            #     self.best_model['state_dict'] = self.model.state_dict()
            #     self.best_model['optimizer'] = self.optimizer.state_dict()

