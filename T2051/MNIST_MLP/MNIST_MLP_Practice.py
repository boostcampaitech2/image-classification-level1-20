

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device %s" % device)

from torchvision import datasets
from torchvision import transforms

mnist_train = datasets.MNIST(root='./data/',train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root='./data/',train=False, transform=transforms.ToTensor(), download=True)
print("mnist_train:\n",mnist_train,"\n")
print("mnist_test:\n",mnist_test,"\n")
print("Done")

BATCH_SIZE = 32
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
print("Done")

class MLP(nn.Module):
    def __init__(self, xdim=784, hdim=256, ydim=10):
        super(MLP, self).__init__()
        self.xdim = xdim
        self.hdim = hdim
        self.ydim = ydim

        self.Linear1 = nn.Linear(
            self.xdim,
            self.hdim
        )

        self.Linear2 = nn.Linear(
            self.hdim,
            self.ydim
        )
        self.init_param()
    def init_param(self):
            nn.init.kaiming_normal_(self.Linear1.weight)
            nn.init.zeros_(self.Linear1.bias)
            nn.init.kaiming_normal_(self.Linear2.weight)
            nn.init.zeros_(self.Linear2.bias)

    def forward(self, input):
            net = input
            net = self.Linear1(net)
            net = F.relu(net)
            net = self.Linear2(net)
            return net

M = MLP(xdim=784, hdim=256, ydim=10).to(device)
loss = nn.CrossEntropyLoss()
optm = optim.Adam(M.parameters(), lr=1e-3)
print("Done")


x_numpy = np.random.randn(2, 784)
x_torch = torch.from_numpy(x_numpy).float().to(device)
y_torch = M.forward(x_torch) # forward path
y_numpy = y_torch.detach().cpu().numpy()


print("x_numpy:\n",x_numpy)
print("x_torch:\n",x_torch)
print("y_torch:\n",y_torch)
print("y_numpy:\n",y_numpy)

np.set_printoptions(precision=3)
n_param=0
for p_idx, (param_name,param) in enumerate(M.named_parameters()):
    param_numpy = param.detach().cpu().numpy()
    n_param += len(param_numpy.reshape(-1))
    print("{} name: {} shape: {}".format(p_idx, param_name,param_numpy.shape))
    print(" val: {}".format(param_numpy.reshape(-1)[:5]))
print("Total number of parameters:{}".format(n_param,',d'))

def func_eval(model, data_iter, device):
    with torch.no_grad():
        model.eval() # evaluate (affects DropOut and BN)
        n_total, n_correct = 0, 0
        for batch_in, batch_out in data_iter:
            y_target = batch_out.to(device)
            model_pred = model(batch_in.view(-1,28*28).to(device))
            _, y_pred = torch.max(model_pred.data, 1)
            n_correct += (y_pred==y_target).sum().item()

            n_total += batch_in.size(0)
        val_accr = (n_correct/n_total)
        model.train() # batck to train mode
    return val_accr
print("Done")

M.init_param()
train_accr = func_eval(M, train_iter, device)
test_accr = func_eval(M, test_iter, device)
print("train_accr:{} test_accr:{}".format(train_accr, test_accr))


print("Start training")
M.init_param()
M.train()
EPOCHS, print_every = 10, 1
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



n_samples = 25
sample_indices = np.random.choice(len(mnist_test.targets), n_samples, replace=False)
test_x = mnist_test.data[sample_indices]
test_y = mnist_test.targets[sample_indices]
with torch.no_grad():
    y_pred = M.forward(test_x.view(-1,28*28).type(torch.float).to(device)/255.)
y_pred = y_pred.argmax(axis=1)
plt.figure(figsize=(10,10))
for idx in range(n_samples):
    plt.subplot(5, 5, idx+1)
    plt.imshow(test_x[idx], cmap='gray')
    plt.axis('off')
    plt.title("Pred: {}, Label: {}".format(y_pred[idx], test_y[idx]))
plt.show()
print("Done")
