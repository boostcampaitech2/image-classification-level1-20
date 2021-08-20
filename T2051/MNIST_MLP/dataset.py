

import torch
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