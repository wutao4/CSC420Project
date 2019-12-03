from torch.nn import functional as F
from torch import nn


class DeepNet(nn.Module):
    def __init__(self, n_joints):
        super(DeepNet, self).__init__()
        self.c1 = nn.Sequential(nn.Conv2d(3, 96, 11, stride=4, padding=4),
                                nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(nn.MaxPool2d(3, stride=2),
                                nn.Conv2d(96, 256, 5, padding=2),
                                nn.ReLU(inplace=True)
                                )
        self.c3 = nn.Sequential(nn.MaxPool2d(3, stride=2),
                                nn.Conv2d(256, 384, 3, padding=1),
                                nn.ReLU(inplace=True))
        self.c4 = nn.Sequential(nn.Conv2d(384, 384, 3, padding=1),
                                nn.ReLU(inplace=True))
        self.c5 = nn.Sequential(nn.Conv2d(384, 256, 3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(3, stride=2))
        self.fc6 = nn.Sequential(nn.Linear(9216, 4096),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(0.1))
        self.fc7 = nn.Sequential(nn.Linear(4096, 4096),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(0.1))
        self.fc8 = nn.Linear(4096, n_joints * 2)

    def forward(self, x):
        conv1 = self.c1(x)
        conv2 = self.c2(F.local_response_norm(conv1, 5))
        conv3 = self.c3(F.local_response_norm(conv2, 5))
        conv4 = self.c4(conv3)
        conv5 = self.c5(conv4)
        fc6 = self.fc6(conv5.view(-1, 9216))
        fc7 = self.fc7(fc6)
        fc8 = self.fc8(fc7)
        return fc8