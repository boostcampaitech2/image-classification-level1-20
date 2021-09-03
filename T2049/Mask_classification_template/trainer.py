import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer():
    def __init__(self, config, model, criterion, optimizer, lr_scheduler, 
                device, data_loader, valid_data_loader, num_trainset, num_validset, epochs):
        self.device = device
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.data_loader = data_loader
        self.valid_loader = valid_data_loader
        self.epochs = epochs

        self.train_loss = []
        self.valid_loss = []
        self.train_total_loss = 0
        self.valid_total_loss = 0
        self.train_correct = 0
        self.valid_correct = 0
        self.num_trainset = num_trainset
        self.num_validset = num_validset
        self.best_model = {'state_dict': None, 'optimizer': None}

        
    
    def train_per_epoch(self, epoch):
        self.model.train()
        self.train_total_loss = 0
        self.train_correct = 0
        print(self.device)
        for data, target in tqdm(self.data_loader, leave=False):
            
            data = data.to(self.device)
            target = target.to(self.device)
            pred = self.model(data)
            loss = self.criterion(pred, target)
            # print([pred.argmax(dim=1), target])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # self.lr_scheduler.step()

            self.train_correct += sum(pred.argmax(dim=1) == target)
            self.train_total_loss += loss.item()
        self.train_total_loss /= self.num_trainset
        self.train_loss.append(self.train_total_loss)
        print(f"{epoch}/{self.epochs}Epochs {self.train_correct}/{self.num_trainset} train accuracy: {self.train_correct/self.num_trainset:.4f}, loss: {self.train_total_loss:.4f}")
 
    
    def valid_per_epoch(self, epoch):
        self.model.eval()

        with torch.no_grad():
            self.valid_total_loss = 0
            self.valid_correct = 0
            loop = tqdm(self.valid_loader, total=len(self.valid_loader))

            for data, target in loop:
                data = data.to(self.device)
                target = target.to(self.device)
                pred = self.model(data)
                loss = self.criterion(pred, target)

                print([pred.argmax(dim=1), target])


                # metric도 모듈화 하고 싶은데 아직 감이 안온다
                self.valid_correct += sum(pred.argmax(dim=1) == target)
                self.valid_total_loss += loss.item()
            self.valid_total_loss /= self.num_validset
            self.valid_loss.append(self.valid_total_loss)
            print(f"{epoch}/{self.epochs}Epochs {self.valid_correct}/{self.num_validset} valid accuracy: {self.valid_correct/self.num_validset:.4f}, loss: {self.valid_total_loss:.4f}")
            loop.set_postfix(loss = self.valid_total_loss)
            # valid에서는 best_model을 저장하기 위해 loss값을 리턴한다
            return self.valid_total_loss


    def train(self):
        lowest_loss = np.inf

        writer = SummaryWriter("logs/CIFAR100")

        for epoch_index in range(1, self.epochs + 1):
            self.train_per_epoch(epoch_index)
            valid_loss = self.valid_per_epoch(epoch_index)

            if valid_loss < lowest_loss:
                lowest_loss = valid_loss
                self.best_model['state_dict'] = self.model.state_dict()
                self.best_model['optimizer'] = self.optimizer.state_dict()
            
            writer.add_scalar('Loss/train', self.train_total_loss, epoch_index)
            writer.add_scalar('Loss/valid', self.valid_total_loss, epoch_index)
            writer.add_scalar('Accuracy/train', self.train_correct/self.num_trainset, epoch_index)
            writer.add_scalar('Accuracy/valid', self.valid_correct/self.num_validset, epoch_index)
