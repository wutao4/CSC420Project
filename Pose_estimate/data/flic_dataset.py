import csv
import json
import os
from scipy.io import loadmat
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv


def get_joint_list(joints):
    head = np.asarray(joints['reye']) + \
        np.asarray(joints['leye']) + \
        np.asarray(joints['nose'])
    head /= 3
    del joints['reye']
    del joints['leye']
    del joints['nose']
    joints['head'] = head.tolist()
    joint_pos = [joints['head']]
    joint_pos.append(joints['lwri'])
    joint_pos.append(joints['rwri'])
    joint_pos.append(joints['lelb'])
    joint_pos.append(joints['relb'])
    joint_pos.append(joints['lsho'])
    joint_pos.append(joints['rsho'])
    joint_pos.append(joints['lhip'])
    joint_pos.append(joints['rhip'])
    return np.array(joint_pos).flatten()


def save_images():
    examples = loadmat('./FLIC/examples.mat')
    examples = examples['examples'][0]
    N_test = int(len(examples) * 0.1)
    testing_indices = np.random.permutation(int(len(examples)))[:N_test].tolist()
    joint_ids = ['lsho', 'lelb', 'lwri', 'rsho', 'relb', 'rwri', 'lhip',
                 'lkne', 'lank', 'rhip', 'rkne', 'rank', 'leye', 'reye',
                 'lear', 'rear', 'nose', 'msho', 'mhip', 'mear', 'mtorso',
                 'mluarm', 'mruarm', 'mllarm', 'mrlarm', 'mluleg', 'mruleg',
                 'mllleg', 'mrlleg']
    fp_train = open('./FLIC/train_joints.csv', 'w')
    fp_test = open('./FLIC/test_joints.csv', 'w')
    for i, example in enumerate(examples):
        joint = example[2].T
        joint = dict(zip(joint_ids, joint))
        fname = example[3][0]
        # print(joint)
        joint = get_joint_list(joint)

        msg = '{},{}'.format(fname, ','.join([str(j) for j in joint.tolist()]))
        if i not in testing_indices:
            print(msg, file=fp_train)
        else:
            print(msg, file=fp_test)

class mydataset(Dataset):
    def __init__(self, csv_fn, img_dir, symmetric_joints):
        self.im_size = 220
        self.csv_fn = csv_fn
        self.img_dir = img_dir
        self.symmetric_joints = json.loads(symmetric_joints)
        self.load_image()

    def __len__(self):
        return len(self.joints)

    def load_image(self):
        self.images = {}
        self.joints = []
        for line in csv.reader(open(self.csv_fn)):
            image_id = line[0]
            if image_id in self.images:
                image = self.images[image_id]
            else:
                img_fn = '{}/{}'.format(self.img_dir, image_id)
                assert os.path.exists(img_fn), \
                    'File not found: {}'.format(img_fn)
                image = cv.imread(img_fn)
                self.images[image_id] = image
            coords = [float(c) for c in line[1:]]
            joints = np.array(list(zip(coords[0::2], coords[1::2])))
            self.joints.append((image_id, joints))


    def calc_joint_center(self, joints):
        x_center = (np.min(joints[:, 0]) + np.max(joints[:, 0])) / 2
        y_center = (np.min(joints[:, 1]) + np.max(joints[:, 1])) / 2
        return [x_center, y_center]

    def calc_joint_bbox_size(self, joints):
        lt = np.min(joints, axis=0)
        rb = np.max(joints, axis=0)
        return rb[0] - lt[0], rb[1] - lt[1]

    def crop_reshape(self, image, joints, bw, bh, cx, cy):
        y_min = int(np.clip(cy - bh / 2, 0, image.shape[0]))
        y_max = int(np.clip(cy + bh / 2, 0, image.shape[0]))
        x_min = int(np.clip(cx - bw / 2, 0, image.shape[1]))
        x_max = int(np.clip(cx + bw / 2, 0, image.shape[1]))
        image = image[y_min:y_max, x_min:x_max]
        joints -= np.array([x_min, y_min])
        fx, fy = self.im_size / image.shape[1], self.im_size / image.shape[0]
        cx, cy = image.shape[1] // 2, image.shape[0] // 2
        image, joints = self.apply_zoom(image, joints, cx, cy, fx, fy)[:2]
        return image, joints

    def apply_zoom(self, image, joints, center_x, center_y, fx, fy):
        joint_vecs = joints - np.array([center_x, center_y])
        image = cv.resize(image, None, fx=fx, fy=fy)
        joint_vecs *= np.array([fx, fy])
        center_x, center_y = center_x * fx, center_y * fy
        joints = joint_vecs + np.array([center_x, center_y])
        return image, joints, center_x, center_y

    def __getitem__(self, index):
        img_id, joints = self.joints[index]
        image = self.images[img_id]
        bbox_w, bbox_h = self.calc_joint_bbox_size(joints)
        center_x, center_y = self.calc_joint_center(joints)
        image, joints = self.crop_reshape(
            image, joints, bbox_w, bbox_h, center_x, center_y)
        image = image.astype(np.float32).transpose(2, 0, 1)
        joints = joints.astype(np.float32).flatten()
        return image, joints

if __name__ == '__main__':
    save_images()