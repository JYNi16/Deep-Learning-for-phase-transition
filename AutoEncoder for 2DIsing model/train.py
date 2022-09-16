import torch
from torch import nn
from model import Autoencoder
from datasets import MCdata
from torch.utils.data import DataLoader
from tqdm import tqdm

epochs = 100
bz = 1048
model = Autoencoder(45)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
torch.set_grad_enabled(True)
model.to(device)
loss_func = nn.MSELoss()

def train(dataloader):
    for epoch in range(epochs):
        #iter = 0
        #loss_all = 0
        for spin_data, t_data in tqdm(dataloader):
            #iter += 1
            optimizer.zero_grad()
            #print("spin_data:", spin_data.shape)
            spin_data = torch.reshape(spin_data, (-1,45*45))
            #print("spin_data is:", spin_data.shape)
            spin_data.to(device)
            out, _ = model(spin_data)
            loss = loss_func(out, spin_data)
            loss.backward()
            optimizer.step()
            #loss_all += loss.data.cpu()
            #print("loss is:", loss.data.cpu())
        if epoch % 10 == 1:
            print("epoch is:", epoch, "loss is:", loss.item())
            torch.save(model.state_dict(),"save_model/ae_45_L_{:d}.pth".format(epoch))

if __name__=="__main__":
    paths = "E:/deeplearning/Unsupervised learning for Phys/AutoEncoder for 2DIsing model/45_L_val"
    dataset = MCdata(paths)
    dataloader = DataLoader(dataset, batch_size=bz, shuffle=True)
    train(dataloader)
