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
    examples = loadmat('FLIC/examples.mat')
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
    def __init__(self, csv_fn, img_dir):
        self.im_size = 220
        self.csv_fn = csv_fn
        self.img_dir = img_dir
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
def save_joints():
    joint_data_fn = 'data.json'
    mat = loadmat('mpii_human_pose_v1_u12_1.mat')

    fp = open(joint_data_fn, 'w')

    for i, (anno, train_flag) in enumerate(
        zip(mat['RELEASE']['annolist'][0, 0][0],
            mat['RELEASE']['img_train'][0, 0][0])):

        img_fn = anno['image']['name'][0, 0][0]
        train_flag = int(train_flag)

        head_rect = []
        if 'x1' in str(anno['annorect'].dtype):
            head_rect = zip(
                [x1[0, 0] for x1 in anno['annorect']['x1'][0]],
                [y1[0, 0] for y1 in anno['annorect']['y1'][0]],
                [x2[0, 0] for x2 in anno['annorect']['x2'][0]],
                [y2[0, 0] for y2 in anno['annorect']['y2'][0]])

        if 'annopoints' in str(anno['annorect'].dtype):
            annopoints = anno['annorect']['annopoints'][0]
            head_x1s = anno['annorect']['x1'][0]
            head_y1s = anno['annorect']['y1'][0]
            head_x2s = anno['annorect']['x2'][0]
            head_y2s = anno['annorect']['y2'][0]
            for annopoint, head_x1, head_y1, head_x2, head_y2 in zip(
                    annopoints, head_x1s, head_y1s, head_x2s, head_y2s):
                if annopoint != []:
                    head_rect = [float(head_x1[0, 0]),
                                 float(head_y1[0, 0]),
                                 float(head_x2[0, 0]),
                                 float(head_y2[0, 0])]

                    # joint coordinates
                    annopoint = annopoint['point'][0, 0]
                    j_id = [str(j_i[0, 0]) for j_i in annopoint['id'][0]]
                    x = [x[0, 0] for x in annopoint['x'][0]]
                    y = [y[0, 0] for y in annopoint['y'][0]]
                    joint_pos = {}
                    for _j_id, (_x, _y) in zip(j_id, zip(x, y)):
                        joint_pos[str(_j_id)] = [float(_x), float(_y)]
                    # joint_pos = fix_wrong_joints(joint_pos)

                    # visiblity list
                    if 'is_visible' in str(annopoint.dtype):
                        vis = [v[0] if v else [0]
                               for v in annopoint['is_visible'][0]]
                        vis = dict([(k, int(v[0])) if len(v) > 0 else v
                                    for k, v in zip(j_id, vis)])
                    else:
                        vis = None

                    if len(joint_pos) == 16:
                        data = {
                            'filename': img_fn,
                            'train': train_flag,
                            'head_rect': head_rect,
                            'is_visible': vis,
                            'joint_pos': joint_pos
                        }

                        print(json.dumps(data), file=fp)


def write_line(datum, fp):
    joints = sorted([[int(k), v] for k, v in datum['joint_pos'].items()])
    joints = np.array([j for i, j in joints]).flatten()

    out = [datum['filename']]
    out.extend(joints)
    out = [str(o) for o in out]
    out = ','.join(out)

    print(out, file=fp)


def split_train_test():
    fp_test = open('mpiitest_joints.csv', 'w')
    fp_train = open('mpiitrain_joints.csv', 'w')
    all_data = open('data.json').readlines()
    N = len(all_data)
    N_test = int(N * 0.1)
    N_train = N - N_test

    print('N:{}'.format(N))
    print('N_train:{}'.format(N_train))
    print('N_test:{}'.format(N_test))

    np.random.seed(1701)
    perm = np.random.permutation(N)
    test_indices = perm[:N_test]
    train_indices = perm[N_test:]

    print('train_indices:{}'.format(len(train_indices)))
    print('test_indices:{}'.format(len(test_indices)))

    for i in train_indices:
        datum = json.loads(all_data[i].strip())
        write_line(datum, fp_train)

    for i in test_indices:
        datum = json.loads(all_data[i].strip())
        write_line(datum, fp_test)


if __name__ == '__main__':
    save_images()
    save_joints()
    split_train_test()