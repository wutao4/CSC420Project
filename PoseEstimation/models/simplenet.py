import torch
from torchsummary import summary
from torch import nn


class simpleNet(nn.Module):
    def __init__(self, n_joints):
        super(simpleNet, self).__init__()
        self.c1 = nn.Sequential(nn.Conv2d(3, 64, stride=4, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(2))
        self.c2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(2))
        self.c3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(2))
        self.fc1 = nn.Sequential(nn.Linear(9216, 4096),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(0.2))
        self.fc2 = nn.Linear(4096, n_joints * 2)

    def forward(self, x):
        conv1 = self.c1(x)
        conv2 = self.c2(conv1)
        conv3 = self.c3(conv2)
        fc1 = self.fc1(conv3.view(-1, 9216))
        fc2 = self.fc2(fc1)
        return fc2

if __name__ == '__main__':
    model = simpleNet(9)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    summary(model, input_size=(3, 220, 220))
