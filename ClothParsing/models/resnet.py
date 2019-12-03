import torch
import torchvision
import torch.nn as nn
from torchsummary import summary


##################################################################
#                      Residual Networks                         #
##################################################################

class ResBlock(nn.Module):
    """ A residual block. Double convolution, ReLU """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.shortcut(x) + self.conv(x)


class LargeResNet(nn.Module):
    """ My residual network simplified and adapted from ResNet18 """
    def __init__(self):
        super(LargeResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)  # 80x60 => 40x30
        self.conv2_x = ResBlock(64, 128)
        self.conv3_x = ResBlock(128, 256, stride=(2, 1))  # => 20x30
        self.conv4_x = ResBlock(256, 512, stride=2)  # => 10x15
        self.conv5_x = ResBlock(512, 1024, stride=(2, 3))  # => 5x5
        self.avgpool = nn.AvgPool2d(kernel_size=5)
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 1000),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1000, 15),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ResNet(nn.Module):
    """ Improved residual network with reduced layers and parameters """
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)  # 80x60 => 40x30
        self.conv2_x = ResBlock(64, 128, stride=2)  # => 20x15
        self.conv3_x = ResBlock(128, 256, stride=(2, 3))  # => 10x5
        self.conv4_x = ResBlock(256, 512, stride=(2, 1))  # => 5x5
        self.avgpool = nn.AvgPool2d(kernel_size=5)
        self.fc = nn.Sequential(
            nn.Linear(512, 15),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    print("----------- start -------------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = ResNet().to(device)
    model = LargeResNet().to(device)
    # dummy = torch.randn(10, 3, 80, 60).to(device)
    # out = model(dummy)
    # print("Out size:", out.size())
    summary(model, input_size=(3, 80, 60))

    print("------------ end --------------")
