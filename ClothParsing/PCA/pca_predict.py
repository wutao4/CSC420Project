import os
import numpy as np
import cv2 as cv
import pandas as pd
import sys

sys.path.insert(0, '..')
import header


##################################################################
#          Predict clothing types with PCA on test sets          #
##################################################################

datapath = "../datasets/myntradataset/"
listpath = "../lists/test_list.txt"
targetpath = "results/"


# Read the average clothing image for each type
targets = []
for cloth_type in header.TYPES:
    tgt = cv.imread(os.path.join(targetpath, cloth_type + ".jpg")) / 255
    targets.append(tgt.flatten())
targets = np.array(targets)

# Read data in 'styles.csv' file. Neglect lines with error
df = pd.read_csv(os.path.join(datapath, 'styles.csv'), error_bad_lines=False)
df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
df = df.set_index('id', drop=False)

# Read image list
f = open(listpath, 'r')
lines = f.readlines()
f.close()


def compute_ssd(input, target):
    return np.sum(np.square(input - target), axis=1)

def norm_correlation(input, target):
    nominator = np.sum(input * target, axis=1)
    denominator = np.sqrt(np.sum(np.square(input), axis=1)) \
                  * np.sqrt(np.sum(np.square(target), axis=1))
    return nominator / denominator


# Predict the classes of test images and compute accuracy
accuracy_ssd = 0
accuracy_nc = 0

for row_id in lines:
    row_id = int(row_id.rstrip())
    row = df.loc[row_id]
    label = header.TYPE_INDEX[row['articleType']]
    # read test image
    img = cv.imread(os.path.join(datapath, 'images', row['image'])) / 255
    img = cv.resize(img, (60, 80)).flatten()
    # compute differences to each clothing type
    img_patch = np.array([img]).repeat(len(targets), axis=0)
    ssd = compute_ssd(img_patch, targets)
    nc = norm_correlation(img_patch, targets)
    # make prediction and compare to test label
    pred_ssd = np.argmin(ssd)
    pred_nc = np.argmax(nc)
    accuracy_ssd += pred_ssd == label
    accuracy_nc += pred_nc == label

accuracy_ssd /= len(lines)
accuracy_nc /= len(lines)
print("PCA test accuracy with SSD: %.2f%%" % (accuracy_ssd * 100))
print("PCA test accuracy with NC : %.2f%%" % (accuracy_nc * 100))
