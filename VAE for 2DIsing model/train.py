import torch
from torch import nn
from model import VAE
from datasets import MCdata
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import config as cf

epochs = 250
bz = 1048
model = VAE(cf.L, 1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size= 10, gamma = 0.98)
torch.set_grad_enabled(True)
model.to(device)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(dataloader):
    for epoch in range(epochs):
        iter = 0
        loss_all = 0
        for spin_data, t_data in dataloader:
            iter += 1
            optimizer.zero_grad()
            #print("spin_data:", spin_data.shape)
            spin_data = spin_data.view(-1,cf.L*cf.L)
            x = spin_data.to(device)
            #print("input.shape is:", spin_data)
            mu_prime, mu, log_var, _ = model(x)
            loss = model.loss(x, mu_prime, mu, log_var)
            loss.backward()
            optimizer.step()
            loss_all += loss.data.cpu()
            #print("loss is:", loss.data.cpu())
        #if epoch % 10 == 1:
        lr = get_lr(optimizer)
        print("learning rate is:", lr)
        print("epoch is:", epoch, "loss is:", loss_all/(iter*bz))
        torch.save(model.state_dict(),"{}_L_save_model/vae_{:d}.pth".format(cf.L, epoch))

        scheduler.step()

if __name__=="__main__":
    paths = "{}_L".format(cf.L)
    dataset = MCdata(paths)
    dataloader = DataLoader(dataset, batch_size=bz, shuffle=True)
    train(dataloader)
