
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



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