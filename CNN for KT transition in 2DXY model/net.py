# CNN model for KT transition in 2DXY model
import torch
from torch import nn

class xynn(nn.Module):
    def __init__(self, lattice):
        self.lattice = lattice
        self.c1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.r1 = nn.ReLU(True)
        self.m1 = nn.MaxPool2d(kernel_size=(2,2))
        self.c2 = nn.Conv2d(in_channels=8, out_channels=16)
        self.r2 = nn.ReLU(True)
        self.f1 = nn.Linear()
        self.s = nn.Sigmoid()

    def forward(self, spin_data):
        x = self.c1(spin_data)
        x = self.r1(x)
        x = self.m1(x)
        x = self.c2(x)
        x = self.r2(x)
        x = self.f1(x)
        x = self.s(x)

        return x