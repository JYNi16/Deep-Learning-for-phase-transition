import matplotlib.pyplot as plt
import torch
from torch import nn
from model import Autoencoder
from datasets import MCdata
from torch.utils.data import DataLoader
from tqdm import tqdm

path = "E:/deeplearning/Unsupervised learning for Phys/AutoEncoder for 2DIsing model"

torch.set_grad_enabled(False)
model = Autoencoder(45)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load(path + "/save_model/ae_45_L_91.pth"))
dataset = MCdata(path + "/45_L_val")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

def inference():
    tem, latent = [], []
    for spin_m, t in tqdm(dataloader):
        spin_m = torch.reshape(spin_m, (-1, 45*45))
        _, out = model(spin_m)
        tem.append(t.numpy()[0])
        latent.append(out.numpy()[0][0])

    return tem, latent

def plot(tem, latent):
    plt.scatter(tem, latent, c=tem)
    plt.colorbar()
    plt.savefig("figure/45_ae.jpg")
    plt.show()

if __name__=="__main__":
    tem, latent = inference()
    plot(tem, latent)