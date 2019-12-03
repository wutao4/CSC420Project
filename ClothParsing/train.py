import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import math
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Function
from torchsummary import summary
from tqdm import tqdm

import header
from data.fashiondata import FashionData
from models.cnn import Shallow
from models.resnet import ResNet, LargeResNet


##################################################################
#                    Train different models                      #
##################################################################

train_mode = True  # True for training, False for simply testing
batch_size = 20  # batch size for training
learn_rate = 1e-4  # learning rate
epochs = 100  # number of epochs to train
weight_path = "checkpoints/"  # directory to save or load trained weights

# Data loader
train_dset = FashionData("datasets/myntradataset/", "lists/train_list.txt")
train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=False)
test_dset = FashionData("datasets/myntradataset/", "lists/test_list.txt")
test_loader = DataLoader(test_dset, batch_size=1, shuffle=False)

# Gpu or cpu device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
# model = Shallow()
# model = LargeResNet()
model = ResNet()
model = model.to(device)

# Load saved model parameters (if pre-trained)
if not train_mode:
    map_loc = "cuda:0" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(os.path.join(weight_path, "resnet_lr4_ep80"),
                            map_location=map_loc)
    model.load_state_dict(state_dict)

# Loss function
criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
# Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


# Train one epoch
def train(epoch):
    model.train()
    train_loss = 0
    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)
        pred = model(data)
        optimizer.zero_grad()
        loss = criterion(pred, label.long())
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        # print(pred.cpu(), loss.item())
    else:
        train_loss /= len(train_loader)
        print('====> Epoch: %d, Avg training loss: %.4f' % (epoch, train_loss))
    scheduler.step()

# Validation
def validate():
    model.eval()
    eval_loss = 0
    accuracy = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data = data.to(device)
            label = label.to(device)
            pred = model(data)
            loss = criterion(pred, label.long())
            eval_loss += loss.item()
            pred, label = pred.cpu().squeeze(), label.cpu().squeeze()
            # print(batch_idx, pred, label)
            if torch.argmax(pred) == label:
                accuracy += 1
    eval_loss /= len(test_loader)
    accuracy /= len(test_loader)
    print('====> Avg validation loss: %.4f, accuracy: %.2f%%' % (eval_loss, accuracy*100))
    return accuracy


# Complete training
if train_mode:
    accuracy = []
    for epoch in range(1, epochs + 1):
        train(epoch)
        accuracy.append(validate())
        if epoch % 10 == 0:
            # Save optimized model parameters
            torch.save(model.state_dict(), os.path.join(weight_path, "resnet2_lr4_ep%d" % epoch))
    print(accuracy)
else:
    validate()
