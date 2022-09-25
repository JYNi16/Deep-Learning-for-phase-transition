import torch
from torchvision import transforms
import os, glob
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
from PIL import Image


class train_data(nn.Module):
    def __init__(self, data_path, lattice):
        self.datapath = []
        self.datapath.extend(glob.glob(os.path.join(data_path, "*.npy")))
        self.L = lattice

    def __getitem__(self, index):
        img_path = self.datapath[index]
        data = np.load(img_path)
        spin_data = np.reshape(data, [-1, self.L, self.L])
        if np.abs(np.sum(spin_data)) < 0.8:
            label = 0
        else:
            label = 1

        return spin_data, label

    def __len__(self):
        return len(self.datapath)

if __name__=="__main__":
    path = "E:/deeplearning/depp learning for Phys/PCA for 2DIsing model/40_L"
    datatrain = train_data(path, 40)
    dataloader = DataLoader(datatrain, batch_size=8)

    for x, y in dataloader:
        print("x.shape is:", x.shape)