


import torch

BATCH_SIZE = 32
ROOT = 3


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device %s" % device)