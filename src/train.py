# Training script
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

def set_Params(network, rate):
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(),lr=rate)
    return loss_function, optimizer

def train(network, epoch, dataset, rate):
    loss_function, optimizer = set_Params(network, rate)
    network.train()
    for e in range(epoch):
        running_loss = 0
        for images, labels in dataset:
            optimizer.zero_grad()
            output = network(images)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(e, running_loss/len(dataset)))