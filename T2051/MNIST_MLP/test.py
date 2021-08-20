

import numpy as np
import torch
from torch import nn
from torch import optim
from model import MLP
from config import device

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