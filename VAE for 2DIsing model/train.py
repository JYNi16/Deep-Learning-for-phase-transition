import torch
from torch import nn
from model import VAE
from datasets import MCdata
from torch.utils.data import DataLoader
from tqdm import tqdm

epochs = 100
bz = 1048
model = VAE(45, 1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
torch.set_grad_enabled(True)
model.to(device)

def train(dataloader):
    for epoch in range(epochs):
        #iter = 0
        #loss_all = 0
        for spin_data, t_data in tqdm(dataloader):
            #iter += 1
            optimizer.zero_grad()
            #print("spin_data:", spin_data.shape)
            batch_size = spin_data.shape[0]
            spin_data = spin_data.view(batch_size,-1)
            spin_data.to(device)
            mu_prime, mu, log_var, _ = model(spin_data)
            loss = model.loss(spin_data, mu_prime, mu, log_var)
            loss.backward()
            optimizer.step()
            #loss_all += loss.data.cpu()
            #print("loss is:", loss.data.cpu())
        if epoch % 10 == 1:
            print("epoch is:", epoch, "loss is:", loss.item())
            torch.save(model.state_dict(),"save_model/vae_45_L_{:d}.pth".format(epoch))

if __name__=="__main__":
    paths = "E:/deeplearning/Unsupervised learning for Phys/AutoEncoder for 2DIsing model/45_L_val"
    dataset = MCdata(paths)
    dataloader = DataLoader(dataset, batch_size=bz, shuffle=True)
    train(dataloader)
