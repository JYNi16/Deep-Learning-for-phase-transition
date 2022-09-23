# CNN model for KT transition in 2DXY model
import torch
from torch import nn

class xy(nn.Module):

    def __init__(self, lattice):
        super(xy, self).__init__()
        self.lattice = lattice
        self.c1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.r1 = nn.ReLU(True)
        self.m1 = nn.MaxPool2d(kernel_size=(2,2))
        self.c2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.r2 = nn.ReLU(True)
        self.m2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.l1 = nn.Linear(576, 32)
        self.l2 = nn.Linear(32, 1)
        self.s = nn.Sigmoid()

    def forward(self, spin_data):
        x = self.c1(spin_data)
        x = self.r1(x)
        x = self.m1(x)
        x = self.c2(x)
        x = self.r2(x)
        x = self.m2(x)
        x = x.view(x.shape[0], -1)
        x = self.l1(x)
        x = self.l2(x)
        x = self.s(x)

        return x


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.C1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.R1 = nn.ReLU()
        self.S2 = nn.MaxPool2d(kernel_size=2)
        self.C3 = nn.Conv2d(6, 16, 5, 1, 0)
        self.R2 = nn.ReLU()
        self.S4 = nn.MaxPool2d(2)
        self.C5 = nn.Conv2d(16, 120, 5, 1, 0)
        self.R3 = nn.ReLU()
        self.F6 = nn.Linear(in_features=120, out_features=84)
        self.R4 = nn.ReLU()
        self.OUT = nn.Linear(84, 10)

    def forward(self, x):
        x = self.C1(x)
        x = self.R1(x)
        x = self.S2(x)
        x = self.C3(x)
        x = self.R2(x)
        x = self.S4(x)
        x = self.C5(x)
        x = self.R3(x)
        x = x.view(x.size(0), -1)
        x = self.F6(x)
        x = self.R4(x)
        x = self.OUT(x)
        return x


if __name__ == "__main__":
    model = xy(32)
    a = torch.randn(1, 1, 32, 32)
    b = model(a)
    print(b.shape)