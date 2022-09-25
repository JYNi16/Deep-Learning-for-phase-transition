from torch.utils.data import DataLoader
import torch
from torch import nn, optim
from model import LeNet
from tqdm import tqdm
from dataset import train_data

model = LeNet()
BZ = 16
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = "E:/deeplearning/depp learning for Phys/PCA for 2DIsing model/40_L"
data = train_data(path, 40)
Data = DataLoader(data, batch_size=BZ)
model.to(device)
lossfunction = nn.CrossEntropyLoss()

def Acc(y_pred, y_truth):
    pred = y_pred.argmax(dim=1)
    return (pred.data.cpu() == y_truth.data).sum()

def train():
    for epoch in range(10):
        torch.set_grad_enabled(True)
        acc = 0
        loss_all = 0
        step = 0
        for spin_data, label in tqdm(Data):
            step += 1
            optimizer.zero_grad()
            spin_data = spin_data.to(device, torch.float)
            out = model(spin_data)
            loss = lossfunction(out, label.to(device, torch.long))
            loss_all += loss
            acc += Acc(out, label)
            loss.backward()
            optimizer.step()
        print("epoch is:", epoch, "loss is %.4f" % (loss_all / step), "accuracy is %.4f" % (acc / (step * BZ)))

if __name__=="__main__":
    train()