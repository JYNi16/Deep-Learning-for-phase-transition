import matplotlib.pyplot as plt
import torch
from torch import nn
from model import VAE
from datasets import MCdata
from torch.utils.data import DataLoader
from tqdm import tqdm
import config as cf 
from matplotlib.colors import LinearSegmentedColormap

modelpath = "{}_L_save_model".format(cf.L)
datapath = "{}_L".format(cf.L)

torch.set_grad_enabled(False)
model = VAE(cf.L, 1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load(modelpath + "/vae_200.pth"))
dataset = MCdata(datapath)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

def inference():
    tem, latent = [], []
    for spin_m, t in tqdm(dataloader):
        spin_m = torch.reshape(spin_m, (1, cf.L*cf.L)).to(device)
        _, out, _, _ = model(spin_m)
        tem.append(t.numpy()[0])
        latent.append(out.data.cpu().numpy()[0][0])

    return tem, latent

def plot(tem, latent):
    #print("temperature range is:", tem)
    plt.figure(1, figsize=(9, 8))

    font = {"weight":"normal", "size":32,}
    plt.scatter(tem, latent, c=tem)
    #plt.colorbar()
    colors1 = [(0, 0, 1), (0.9, 0.9, 0.9), (1, 0.0, 0.0)]  # 蓝色到乳灰色到红色
    cmap_name = 'custom_blue_red'
    cm1 = LinearSegmentedColormap.from_list(cmap_name, colors1, N=256)
    cb = plt.colorbar(aspect=25, pad=0.02, shrink=0.9, fraction=0.05, cmap = cm1)

    cb.ax.tick_params(labelsize=20)
    #for l in cb.ax.yaxis.get_ticklabels():
    #    l.set_family("Times New Roman") 
    
    plt.xlabel("temperature", font)
    plt.ylabel("order", font)
    plt.xticks(fontsize = 24)
    plt.yticks(fontsize = 24)

    #plt.subplots_adjust(left=0.15, bottom=0.1, right=0.98, top=0.98, wspace=0.16, hspace=0.15)
    plt.savefig("figure/{}_vae.jpg".format(cf.L))
    #plt.show()

if __name__=="__main__":
    tem, latent = inference()
    plot(tem, latent)
