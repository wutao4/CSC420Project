from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import torch
import models.DeepNet as deepnet
import models.simplenet as simplenet
import models.residualNet as resnet
import data.flic_dataset as flic_dataset


def plt_image(image, label, model):
    ax1 = plt.subplot(1, 2, 1)
    ax1.title.set_text("image with label joints")
    ax2 = plt.subplot(1, 2, 2)
    ax2.title.set_text("image with predict joints")
    with torch.no_grad():
        pred = model(image)
    pred = pred.cpu()
    image = image.squeeze().permute(1, 2, 0).cpu()
    image = np.array(image).astype(np.uint8)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image_label = image.copy()
    image_pred = image.copy()
    joint_dict = {}
    joint_name = ['head', 'lank', 'rank', 'lelb', 'relb', 'lsho', 'rsho', 'lhip', 'rhip']
    joint_dict['head'] = [(label[0][18], label[0][19]), (pred[0][18], pred[0][19])]
    joint_dict['lelb'] = [(label[0][28], label[0][29]), (pred[0][28], pred[0][29])]
    joint_dict['relb'] = [(label[0][22], label[0][23]), (pred[0][22], pred[0][23])]
    joint_dict['lsho'] = [(label[0][26], label[0][27]), (pred[0][26], pred[0][27])]
    joint_dict['rsho'] = [(label[0][24], label[0][25]), (pred[0][24], pred[0][25])]
    joint_dict['lhip'] = [(label[0][6], label[0][7]), (pred[0][6], pred[0][7])]
    joint_dict['rhip'] = [(label[0][4], label[0][5]), (pred[0][4], pred[0][5])]
    joint_dict['lank'] = [(label[0][10], label[0][11]), (pred[0][10], pred[0][11])]
    joint_dict['rank'] = [(label[0][0], label[0][1]), (pred[0][0], pred[0][1])]
    # draw line from head to shoulder
    image_pred = cv.line(image_pred, joint_dict['head'][1], joint_dict['lsho'][1], (255, 0, 0), 2)
    image_label = cv.line(image_label, joint_dict['head'][0], joint_dict['lsho'][0], (255, 0, 0), 2)
    image_pred = cv.line(image_pred, joint_dict['head'][1], joint_dict['rsho'][1], (255, 0, 0), 2)
    image_label = cv.line(image_label, joint_dict['head'][0], joint_dict['rsho'][0], (255, 0, 0), 2)
    # draw line from shoulder to elbow
    image_pred = cv.line(image_pred, joint_dict['lelb'][1], joint_dict['lsho'][1], (0, 255, 0), 2)
    image_label = cv.line(image_label, joint_dict['lelb'][0], joint_dict['lsho'][0], (0, 255, 0), 2)
    image_pred = cv.line(image_pred, joint_dict['relb'][1], joint_dict['rsho'][1], (0, 255, 0), 2)
    image_label = cv.line(image_label, joint_dict['relb'][0], joint_dict['rsho'][0], (0, 255, 0), 2)
    # draw line from elbow to wrist
    image_pred = cv.line(image_pred, joint_dict['lhip'][1], joint_dict['lank'][1], (0, 0, 255), 2)
    image_label = cv.line(image_label, joint_dict['lhip'][0], joint_dict['lank'][0], (0, 0, 255), 2)
    image_pred = cv.line(image_pred, joint_dict['rhip'][1], joint_dict['rank'][1], (0, 0, 255), 2)
    image_label = cv.line(image_label, joint_dict['rhip'][0], joint_dict['rank'][0], (0, 0, 255), 2)
    # draw line from shoulder to hip
    image_pred = cv.line(image_pred, joint_dict['lhip'][1], joint_dict['lsho'][1], (255, 0, 255), 2)
    image_label = cv.line(image_label, joint_dict['lhip'][0], joint_dict['lsho'][0], (255, 0, 255), 2)
    image_pred = cv.line(image_pred, joint_dict['rhip'][1], joint_dict['rsho'][1], (255, 0, 255), 2)
    image_label = cv.line(image_label, joint_dict['rhip'][0], joint_dict['rsho'][0], (255, 0, 255), 2)
    ax1.imshow(image_label)
    ax2.imshow(image_pred)
    plt.show()


if __name__ == '__main__':
    model = resnet.ResidualNet(16)
    weight_path = "./mpii_weight/weight_epoch50"
    print("init model")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    model = model.to(device)
    model.eval()
    model.load_state_dict(torch.load(weight_path))
    print("initial test dataset")
    testset = flic_dataset.mydataset("./data/mpiitrain_joints.csv", "./data/images")
    testloader = torch.utils.data.DataLoader(testset,shuffle=True)
    image, label = next(iter(testloader))
    image = image.to(device)
    plt_image(image, label, model)