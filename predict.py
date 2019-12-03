import os
import argparse
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

from ClothParsing import header
from ClothParsing.models.cnn import Shallow
from ClothParsing.models.resnet import ResNet, LargeResNet


###############################################################################
#   Estimate human poses and predict the clothing types in the input images   #
###############################################################################

parser = argparse.ArgumentParser(description='Pose Estimation and Cloth Parsing')
parser.add_argument('--inputpath', default='test_images/test1.jpg',
                    help='path to input image')
parser.add_argument('--outpath', default='pred_results/',
                    help='path for saving predicted results')
parser.add_argument('--pweight', default='PoseEstimation/some_weight',  # TODO
                    help='path to pre-trained weight for pose estimation model')
parser.add_argument('--cweight', default='ClothParsing/checkpoints/resnet_lr4_ep80',
                    help='path to pre-trained weight for clothing parsing model')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()

# Gpu or cpu device
device = torch.device("cuda" if args.cuda else "cpu")

# Input image
image = Image.open(args.inputpath).convert('RGB')
width, height = image.size
# image.show()


#            Pose Estimation part
###########################################################################
# pmodel = SomeNet()  # TODO
# pmodel = pmodel.to(device)
# p_state_dict = torch.load(args.pweight,
#                           map_location="cuda:0" if args.cuda else "cpu")
# pmodel.load_state_dict(p_state_dict)
#
# # Resize and convert PIL image to tensor
# p_resize = transforms.Compose([
#     transforms.Resize((220, 220)),
#     transforms.ToTensor()
# ])
# p_image = p_resize(image).unsqueeze(0).to(device)
#
# # Predict joints with the pre-trained model
# with torch.no_grad():
#     p_pred = pmodel(p_image)
# p_pred = p_pred.cpu()
#
# # The joint coordinates
# head = (p_pred[0], p_pred[1])  # TODO
#
# # Rescale the coordinates to match the original image
# head = (int(head[0] * height/220), int(head[1] * width/220))  # TODO
#
# # Draw poses
# # TODO


#            Clothing Parsing part
###########################################################################
cmodel = ResNet()
cmodel = cmodel.to(device)
c_state_dict = torch.load(args.cweight,
                          map_location="cuda:0" if args.cuda else "cpu")
cmodel.load_state_dict(c_state_dict)

# Resize and convert PIL image to tensor
c_resize = transforms.Compose([
    transforms.Resize((80, 60)),
    transforms.ToTensor()
])

# Cut patches: [headwear, topwear, bottomwear, footwear1, footwear2]
im = np.asarray(image)
h_min = [15, 50, 230, 420, 420]  # min height for each patch
h_max = [105, 300, 490, 500, 500]  # max height for each patch
w_min = [125, 70, 70, 85, 155]  # min width for each patch
w_max = [230, 270, 270, 160, 230]  # max width for each patch
headwear = im[h_min[0]:h_max[0], w_min[0]:w_max[0], :]  # TODO
topwear = im[h_min[1]:h_max[1], w_min[1]:w_max[1], :]
bottomwear = im[h_min[2]:h_max[2], w_min[2]:w_max[2], :]
footwear1 = im[h_min[3]:h_max[3], w_min[3]:w_max[3], :]
footwear2 = im[h_min[4]:h_max[4], w_min[4]:w_max[4], :]
patches = [headwear, topwear, bottomwear, footwear1, footwear2]
colors = [(255, 255, 0), (255, 0, 0), (0, 0, 255), (255, 0, 255), (0, 255, 0)]

# Predict each patch
for i in range(len(patches)):
    patch = Image.fromarray(patches[i])
    c_image = c_resize(patch).unsqueeze(0).to(device)

    # Predict clothing types with the pre-trained model
    with torch.no_grad():
        c_pred = cmodel(c_image)
    c_pred = c_pred.cpu().squeeze()
    c_idx = np.argmax(c_pred)
    print(c_pred, "====>", header.TYPES[c_idx])

    # Set threshold for prediction confidence, and draw predictions
    threshold = 0.5
    if c_pred[c_idx] > threshold:  # TODO
        cv.rectangle(im, (w_min[i], h_min[i]), (w_max[i], h_max[i]), colors[i], thickness=2)
        cv.putText(im, header.TYPES[c_idx], (w_min[i], h_min[i] + 20),
                   fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=colors[i])

# Save results
plt.imshow(im)
plt.show()
