import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self, lattice, latent_size):
        super(VAE, self).__init__()
        self.infetures = lattice * lattice
        self.latent_size = latent_size
        self.Encoder = nn.Sequential(
            nn.Linear(self.infetures, 625),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(625, 256),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(64, self.latent_size*2),
            nn.Tanh()
    )

        self.Decoder = nn.Sequential(
            nn.Linear(self.latent_size, 64),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(64, 256),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(256, 625),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(625, self.infetures),
            nn.Tanh()
        )

    def encoder(self, x):
        out = self.Encoder(x)
        mu = out[:, :self.latent_size]
        log_var = out[:, self.latent_size:]

        return mu, log_var

    def decoder(self, z):
        mu_prime = self.Decoder(z)
        return mu_prime

    def reparameterization(self, mu, log_var):
        epsilon = torch.randn_like(log_var)
        z = mu + epsilon * torch.sqrt(log_var.exp())
        return z

    def loss(self, X, mu_prime, mu, log_var):
        # reconstruction_loss = F.mse_loss(mu_prime, X, reduction='mean') is wrong!
        reconstruction_loss = torch.mean(torch.square(X - mu_prime).sum(dim=1))

        latent_loss = torch.mean(0.5 * (log_var.exp() + torch.square(mu) - log_var).sum(dim=1))
        return reconstruction_loss + latent_loss

    def forward(self, X):
        mu, log_var = self.encoder(X)
        z = self.reparameterization(mu, log_var)
        mu_prime = self.decoder(z)
        return mu_prime, mu, log_var, z

if __name__=="__main__":
    model = VAE(45, 1)
    spin_data = torch.ones(1, 45*45)
    mu_prime, mu, log_var = model(spin_data)
    print("mu_prime is:", mu_prime.shape, "mu is:", mu.shape, "log_var is:", log_var.shape)
    loss = model.loss(spin_data, mu_prime, mu, log_var)

    print("loss is:", loss)