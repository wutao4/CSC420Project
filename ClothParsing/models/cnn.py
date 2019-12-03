import torch
import torch.nn as nn
from torchsummary import summary


##################################################################
#                          CNN Models                            #
##################################################################

class Shallow(nn.Module):
    """ A simple and shallow cnn model """
    def __init__(self):
        super(Shallow, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Sequential(
            nn.Linear(20 * 15 * 128, 1024),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 15),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = output.reshape(output.size(0), -1)
        output = self.dropout(output)
        output = self.fc1(output)
        output = self.fc2(output)
        return output


if __name__ == '__main__':
    print("----------- start -------------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Shallow().to(device)
    summary(model, input_size=(3, 80, 60))
    dummy = torch.randn(10, 3, 80, 60).to(device)
    out = model(dummy)
    print("Out size:", out.size())

    print("------------ end --------------")
