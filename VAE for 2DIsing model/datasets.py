import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os, glob
import config as cf

class MCdata(Dataset):
    def __init__(self, paths):
        self.path = []
        self.path.extend(glob.glob(os.path.join(paths, '*.npy')))
        self.path

    def __getitem__(self, index):
        spin_data = np.load(self.path[index])
        spin_data = torch.tensor(np.reshape(spin_data, [1,-1]), dtype=torch.float)
        t_data = float(self.path[index].split("/")[-1].split("_")[0])

        return spin_data, t_data

    def __len__(self):
        return len(self.path)

if __name__=="__main__":
    paths = "45_L"
    dataset = MCdata(paths)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for spin_data, t_data in dataloader:
        print("spin_data is:", spin_data.shape)
        print("t_data is:", t_data)
