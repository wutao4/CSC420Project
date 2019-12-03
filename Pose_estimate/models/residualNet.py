from torch.nn import functional as F
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding = 1):
        super(ResidualBlock, self).__init__()
        self.origin = nn.Sequential()
        if in_ch != out_ch:
            self.origin = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride),
                                  nn.ReLU(inplace=True),
                                  nn.BatchNorm2d(out_ch),
                                  nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
                                  nn.ReLU(inplace=True),
                                  nn.BatchNorm2d(out_ch))
    def forward(self, x):
        return self.origin(x) + self.conv(x)


class ResidualNet(nn.Module):
    def __init__(self, n_joints):
        super(ResidualNet, self).__init__()
        self.c1 = nn.Sequential(nn.Conv2d(3, 96, 11, stride=4, padding=4),
                                nn.ReLU(inplace=True),
                                nn.BatchNorm2d(96)
                                )
        self.c2 = nn.Sequential(nn.MaxPool2d(3, stride=2),
                                ResidualBlock(96, 256, 5, padding=2),
                                )
        self.c3 = nn.Sequential(nn.MaxPool2d(3, stride=2),
                                ResidualBlock(256, 384, 3, padding=1))
        self.c4 = ResidualBlock(384, 384, 3, padding=1)
        self.c5 = nn.Sequential(ResidualBlock(384, 256, 3, padding=1),
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