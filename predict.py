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

from PoseEstimation.models.residualNet import ResidualNet
from ClothParsing import header
from ClothParsing.models.cnn import Shallow
from ClothParsing.models.resnet import ResNet, LargeResNet


###############################################################################
#   Estimate human poses and predict the clothing types in the input images   #
###############################################################################

parser = argparse.ArgumentParser(description='Pose Estimation and Cloth Parsing')
parser.add_argument('--inputpath', default='test_images/test1.jpg',
                    help='path to input image')
parser.add_argument('--outpath', default='pred_results/pred1.jpg',
                    help='path for saving predicted results')
parser.add_argument('--pweight', default='PoseEstimation/weights/weight',
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
pmodel = ResidualNet(9)
pmodel = pmodel.to(device)
p_state_dict = torch.load(args.pweight,
                          map_location="cuda:0" if args.cuda else "cpu")
pmodel.load_state_dict(p_state_dict)

# Resize and convert PIL image to tensor
p_resize = transforms.Compose([
    transforms.Resize((220, 220)),
    transforms.ToTensor()
])
p_image = p_resize(image).unsqueeze(0).to(device)

# Predict joints with the pre-trained model
with torch.no_grad():
    p_pred = pmodel(p_image)
p_pred = p_pred.cpu().squeeze()

# The joint coordinates (height, width)
head = (p_pred[1], p_pred[0])
l_ankle = (p_pred[3], p_pred[2])
r_ankle = (p_pred[5], p_pred[4])
l_elbow = (p_pred[7], p_pred[6])
r_elbow = (p_pred[9], p_pred[8])
l_shoulder = (p_pred[11], p_pred[10])
r_shoulder = (p_pred[13], p_pred[12])
l_hip = (p_pred[15], p_pred[14])
r_hip = (p_pred[17], p_pred[16])

# Rescale the coordinates to match the original image
head = (int(head[0] * height/220), int(head[1] * width/220))
l_ankle = (int(l_ankle[0] * height/220), int(l_ankle[1] * width/220))
r_ankle = (int(r_ankle[0] * height/220), int(r_ankle[1] * width/220))
l_elbow = (int(l_elbow[0] * height/220), int(l_elbow[1] * width/220))
r_elbow = (int(r_elbow[0] * height/220), int(r_elbow[1] * width/220))
l_shoulder = (int(l_shoulder[0] * height/220), int(l_shoulder[1] * width/220))
r_shoulder = (int(r_shoulder[0] * height/220), int(r_shoulder[1] * width/220))
l_hip = (int(l_hip[0] * height/220), int(l_hip[1] * width/220))
r_hip = (int(r_hip[0] * height/220), int(r_hip[1] * width/220))

# Draw poses
pose_im = np.asarray(image)
# draw line from head to shoulder
pose_im = cv.line(pose_im, (head[1], head[0]), (l_shoulder[1], l_shoulder[0]), (0, 255, 0), 2)
pose_im = cv.line(pose_im, (head[1], head[0]), (r_shoulder[1], r_shoulder[0]), (0, 255, 0), 2)
# draw line from shoulder to elbow
pose_im = cv.line(pose_im, (l_elbow[1], l_elbow[0]), (l_shoulder[1], l_shoulder[0]), (0, 255, 0), 2)
pose_im = cv.line(pose_im, (r_elbow[1], r_elbow[0]), (r_shoulder[1], r_shoulder[0]), (0, 255, 0), 2)
# draw line from elbow to wrist
pose_im = cv.line(pose_im, (l_hip[1], l_hip[0]), (l_ankle[1], l_ankle[0]), (0, 255, 0), 2)
pose_im = cv.line(pose_im, (r_hip[1], r_hip[0]), (r_ankle[1], r_ankle[0]), (0, 255, 0), 2)
# draw line from shoulder to hip
pose_im = cv.line(pose_im, (l_hip[1], l_hip[0]), (l_shoulder[1], l_shoulder[0]), (0, 255, 0), 2)
pose_im = cv.line(pose_im, (r_hip[1], r_hip[0]), (r_shoulder[1], r_shoulder[0]), (0, 255, 0), 2)

plt.imshow(pose_im)
plt.show()


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
# Each patch is cut based on a ratio of the person's shape (unit_h, unit_w below)
unit_h = l_hip[0] - l_shoulder[0]
unit_w = l_shoulder[1] - r_shoulder[1]
im = np.asarray(image)
h_min = [                                      # min height for each patch
    max(0, int(head[0] - unit_h*0.4)),         # headwear
    max(0, int(l_shoulder[0] - unit_h*0.45)),  # topwear
    max(0, int(l_hip[0] - unit_h*0.25)),       # bottomwear
    max(0, int(l_ankle[0] - unit_h*0.12)),     # footwear (left)
    max(0, int(r_ankle[0] - unit_h*0.12))      # footwear (right)
]
h_max = [                                      # max height for each patch
    min(height, int(head[0] + unit_h*0.3)),
    min(height, int(l_hip[0] + unit_h*0.45)),
    min(height, int(l_ankle[0] + unit_h*0.45)),
    min(height, int(l_ankle[0] + unit_h*0.45)),
    min(height, int(r_ankle[0] + unit_h*0.45))
]
w_min = [                                      # min width for each patch
    max(0, int(head[1] - unit_w*0.4)),
    max(0, int(r_elbow[1] - unit_w*0.45)),
    max(0, int(r_ankle[1] - unit_w*0.5)),
    max(0, int(l_ankle[1] - unit_w*0.4)),
    max(0, int(r_ankle[1] - unit_w*0.4))
]
w_max = [                                      # max width for each patch
    min(width, int(head[1] + unit_w*0.4)),
    min(width, int(l_elbow[1] + unit_w*0.45)),
    min(width, int(l_ankle[1] + unit_w*0.5)),
    min(width, int(l_ankle[1] + unit_w*0.4)),
    min(width, int(r_ankle[1] + unit_w*0.4))
]
headwear = im[h_min[0]:h_max[0], w_min[0]:w_max[0], :]
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
    if c_pred[c_idx] > threshold:
        cv.rectangle(pose_im, (w_min[i], h_min[i]), (w_max[i], h_max[i]), colors[i], thickness=2)
        cv.putText(pose_im, header.TYPES[c_idx], (w_min[i], h_min[i] + 20),
                   fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=colors[i])

# Save results
plt.imshow(pose_im)
plt.show()
cv.imwrite(args.outpath, cv.cvtColor(pose_im, cv.COLOR_RGB2BGR))
