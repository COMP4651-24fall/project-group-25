import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F 
import sys
import numpy as np



NUM_TRAIN = 49000 # The number of training images from the CIFAR-10 dataset
print_every = 100
USE_GPU = False
dtype = torch.float32 
device = torch.device('cpu') # default to CPU

# Load dataset
transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

cifar10_train = dset.CIFAR10('/home/app/function/core/dataset', train=True, download=True,
                            transform=transform)

loader_train = DataLoader(cifar10_train, batch_size=64,
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

cifar10_val = dset.CIFAR10('/home/app/function/core/dataset', train=True, download=True,
                        transform=transform)

loader_val = DataLoader(cifar10_val, batch_size=64,
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

cifar10_test = dset.CIFAR10('/home/app/function/core/dataset', train=False, download=True,
                            transform=transform)

loader_test = DataLoader(cifar10_test, batch_size=64)


# Flatten the data

def flatten(x):
    N = x.shape[0]
    return x.view(N, -1)
class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)
        
# Check accuracy
def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval() 
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc

# Train model
def train(model, optimizer, epochs=1):
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy(loader_val, model)
                print()
                
# Model  
def mymodel():
  
    channel_1 = 32
    channel_2 = 64
    learning_rate = 1e-2
    
    model = nn.Sequential(
        nn.Conv2d(3, channel_1, 3, padding=2),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Conv2d(channel_1, channel_1, 3, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(channel_1, channel_2, 3, padding=2),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Conv2d(channel_2, channel_2, 3, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        Flatten(),
        nn.Linear(7744, 10)
    )

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    train(model, optimizer, epochs=3)
    
    return check_accuracy(loader_test, model)
