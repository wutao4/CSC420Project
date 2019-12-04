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
    joint_name = ['head', 'lwri', 'rwri', 'lelb', 'relb', 'lsho', 'rsho', 'lhip', 'rhip']
    for i in range(9):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x_index = i * 2
        y_index = 2 * i + 1
        joint_dict[joint_name[i]]=[(label[0][x_index], label[0][y_index]), (pred[0][x_index], pred[0][y_index])]
        image_pred = cv.circle(image_pred, (pred[0][x_index], pred[0][y_index]), 2, color, 2)
        image_label = cv.circle(image_label, (label[0][x_index], label[0][y_index]), 2, color, 2)
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
    image_pred = cv.line(image_pred, joint_dict['lelb'][1], joint_dict['lwri'][1], (0, 0, 255), 2)
    image_label = cv.line(image_label, joint_dict['lelb'][0], joint_dict['lwri'][0], (0, 0, 255), 2)
    image_pred = cv.line(image_pred, joint_dict['relb'][1], joint_dict['rwri'][1], (0, 0, 255), 2)
    image_label = cv.line(image_label, joint_dict['relb'][0], joint_dict['rwri'][0], (0, 0, 255), 2)
    # draw line from shoulder to hip
    image_pred = cv.line(image_pred, joint_dict['lhip'][1], joint_dict['lsho'][1], (255, 0, 255), 2)
    image_label = cv.line(image_label, joint_dict['lhip'][0], joint_dict['lsho'][0], (255, 0, 255), 2)
    image_pred = cv.line(image_pred, joint_dict['rhip'][1], joint_dict['rsho'][1], (255, 0, 255), 2)
    image_label = cv.line(image_label, joint_dict['rhip'][0], joint_dict['rsho'][0], (255, 0, 255), 2)
    ax1.imshow(image_label)
    ax2.imshow(image_pred)
    plt.show()


if __name__ == '__main__':
    model_select = "resnet"
    if model_select == "deepnet":
        model = deepnet.DeepNet(9)
        weight_path = "./deepnet_weight/weight_epoch100"
    elif model_select == "resnet":
        model = resnet.ResidualNet(9)
        weight_path = "./residuenet_weight/weight_epoch50"
    elif model_select == "simplenet":
        model = simplenet.simpleNet(9)
        weight_path = "./simple_weight/weight_epoch10"
    print("init model")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    model = model.to(device)
    model.eval()
    model.load_state_dict(torch.load(weight_path))
    print("initial test dataset")
    testset = flic_dataset.mydataset("./data/FLIC/train_joints.csv", "./data/FLIC/images")
    testloader = torch.utils.data.DataLoader(testset,shuffle=True)
    image, label = next(iter(testloader))
    image = image.to(device)
    plt_image(image, label, model)