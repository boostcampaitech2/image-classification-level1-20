

import numpy as np
import matplotlib.pyplot as plt
import torch
from dataset import mnist_test
from config import *


M = torch.load('MNIST_MLP')




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