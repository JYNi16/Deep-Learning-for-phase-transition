import torch
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, lattice_size, p_dropout=0.5):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(lattice_size ** 2, 625),
            nn.ReLU(True),
            nn.Dropout(p_dropout),
            nn.Linear(625, 256),
            nn.ReLU(True),
            nn.Dropout(p_dropout),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Dropout(p_dropout),
            nn.Linear(64, 1),
            nn.Tanh())

        self.decoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(True),
            nn.Dropout(p_dropout),
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Dropout(p_dropout),
            nn.Linear(256, 625),
            nn.ReLU(True),
            nn.Dropout(p_dropout),
            nn.Linear(625, lattice_size ** 2),
            nn.Tanh()
        )

    def forward(self, x):
        encoded_values = self.encoder(x)
        x = self.decoder(encoded_values)
        return x, encoded_values

if __name__=="__main__":
    model = Autoencoder(40, p_dropout=0.5)
    x = torch.ones(1, 40*40)
    out, _ = model(x)
    print("out.shape is:", out.shape)