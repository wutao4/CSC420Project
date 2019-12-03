import os
import numpy as np
import cv2 as cv
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


##################################################################
# Scratching file, please ignore.                                #
##################################################################

def fashionista():
    img1 = cv.imread('000040.jpg')
    map1 = cv.imread('000040_map.png')
    print(map1)
    print(np.max(map1[:, :, 2]))
    concat = np.concatenate((img1, map1), axis=1)
    cv.imshow('img - map', concat)
    cv.waitKey()
    a = map1[:, :, 1]
    cv.imshow('a', a.astype(float))
    cv.waitKey()
    b = map1[:, :, 2]
    cv.imshow('b', b)
    cv.waitKey()

def fashion_product():
    # img = cv.imread('datasets/myntradataset/images/45579.jpg')
    # print(img.shape)
    # cv.imshow('img', img)
    # cv.waitKey()
    img = Image.open('datasets/myntradataset/images/45579.jpg')
    img = img.convert('RGB')
    im = np.asarray(img)
    print(im.shape)
    img.show()

def test_tensor():
    resizeToTensor = transforms.Compose([
        transforms.Resize((60, 80)),
        transforms.ToTensor()
    ])
    img = Image.open('datasets/myntradataset/images/33021.jpg')
    img = resizeToTensor(img)
    print(img.shape)

def softmax():
    soft = nn.Softmax(dim=1)
    a = torch.Tensor([[1, 2, 3, 4, 5]])
    b = soft(a)
    print(b)

def numpy():
    a = np.array([1., 2., 0., .6])
    # b = torch.from_numpy(a).long()
    # c = torch.Tensor([5]).long()
    b = np.repeat(a, 3, axis=0)
    c = [a, a*2, a*0]
    c = np.square(np.array(c))
    d = np.sum(c, axis=1)
    print(c, d)


if __name__ == '__main__':
    print("----------- start -------------")
    # fashionista()
    # fashion_product()
    # test_tensor()
    # softmax()
    numpy()

    print("------------ end --------------")
