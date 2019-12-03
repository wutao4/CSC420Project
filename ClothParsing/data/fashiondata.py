import os
import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import sys

sys.path.insert(0, '..')
import header


##################################################################
#                      PyTorch Dataset                           #
##################################################################

# resize and convert PIL image to tensor
resizeToTensor = transforms.Compose([
    transforms.Resize((80, 60)),
    transforms.ToTensor()
])


class FashionData(Dataset):
    """ Dataset of fashion images and labels of the types """
    def __init__(self, datapath, listpath):
        self.img_dir = os.path.join(datapath, "images")
        self.data = []

        # Read data in 'styles.csv' file. Neglect lines with error
        df = pd.read_csv(os.path.join(datapath, 'styles.csv'), error_bad_lines=False)
        df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
        df = df.set_index('id', drop=False)
        # Read image list
        f = open(listpath, 'r')
        lines = f.readlines()
        f.close()

        for row_id in lines:
            row_id = int(row_id.rstrip())
            row = df.loc[row_id]
            # label = torch.from_numpy(header.TYPE_LABEL[row['articleType']])
            label = header.TYPE_INDEX[row['articleType']]
            self.data.append((row['image'], label))

    def __getitem__(self, i):
        img_name, label = self.data[i]
        image = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        image = resizeToTensor(image)
        return image, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    print("----------- start -------------")

    train_dset = FashionData("../datasets/myntradataset/", "../lists/train_list.txt")
    train_loader = DataLoader(train_dset, batch_size=10, shuffle=False)
    test_dset = FashionData("../datasets/myntradataset/", "../lists/test_list.txt")
    test_loader = DataLoader(test_dset, batch_size=1, shuffle=False)
    for batch_idx, (data, label) in enumerate(train_loader):
        if data.size(1) != 3:
            print(batch_idx, data.shape, label)
    # for batch_idx, (data, label) in enumerate(test_loader):
    #     print(batch_idx, data.shape, label)
    #     data_img = data.view(3, 80, 60).permute(1, 2, 0)
    #     plt.imshow(data_img), plt.axis('off')
    #     plt.show()

    print("------------ end --------------")
