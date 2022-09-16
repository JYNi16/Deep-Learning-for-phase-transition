import torch
from torch import nn
from model import Autoencoder
from datasets import MCdata
from torch.utils.data import DataLoader

epochs = 100
bz = 8
model = Autoencoder(40)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
torch.set_grad_enabled(True)
model.to(device)
loss_func = nn.MSELoss()

def train(dataloader):
    for epoch in range(epochs):
        iter = 0
        loss_all = 0
        for spin_data, t_data in dataloader:
            iter += 1
            optimizer.zero_grad()
            spin_data = torch.reshape(spin_data, [bz,-1])
            spin_data.to(device)
            out, _ = model(spin_data)
            loss = loss_func(out, spin_data)
            loss.backward()
            optimizer.step()
            loss_all += loss.data.cpu()
        print("epoch is:", epoch, "loss is:", loss_all/iter)

if __name__=="__main__":
    paths = "E:/deeplearning/Unsupervised learning for Phys/PCA for 2DIsing model/40_L"
    dataset = MCdata(paths)
    dataloader = DataLoader(dataset, batch_size=bz, shuffle=True)
    train(dataloader)
