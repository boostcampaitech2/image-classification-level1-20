

import torch
from torch import optim , nn
from model import MLP, func_eval
from dataset import train_iter, test_iter
from config import device
from utils import *



M = MLP(xdim=784, hdim=256, ydim=10).to(device)
loss = nn.CrossEntropyLoss()
optm = optim.Adam(M.parameters(), lr=1e-3)

M.init_param()
train_accr = func_eval(M, train_iter, device)
test_accr = func_eval(M, test_iter, device)
print("train_accr:{} test_accr:{}".format(train_accr, test_accr))

print("Start training")
M.init_param()
M.train()
EPOCHS, print_every = 1, 1
for epoch in range(EPOCHS):
    loss_val_sum = 0
    for batch_in, batch_out in train_iter:
        # Forward path
        y_pred = M.forward(batch_in.view(-1,28*28).to(device))
        loss_out = loss(y_pred,batch_out.to(device))
        # Update
        optm.zero_grad()
        loss_out.backward()
        optm.step()
        loss_val_sum += loss_out
    loss_val_avg = loss_val_sum / len(train_iter)

    # Print
    if (epoch % print_every) == 0 or (epoch==(EPOCHS-1)):
        train_accr = func_eval(M, train_iter, device)
        test_accr = func_eval(M, test_iter, device)
        print("epoch: {} loss: {} train_accr: {} test_accr: {}".format(epoch, loss_val_avg, train_accr, test_accr))

print("Done")

# torch.save(M,'MNIST_MLP') / save model


