import torch
from torch import nn
from torch.nn import functional as F

class Autoencoder(nn.Module):
    """
    Standard Autoencoder for dimensionality reduction using PyTorch.
    """
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class VAE(nn.Module):
    """
    Variational Autoencoder for generating new data points using PyTorch.
    """
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder layers
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_mean = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)

        # Decoder layers
        self.decode_fc1 = nn.Linear(latent_dim, 128)
        self.decode_fc2 = nn.Linear(128, 256)
        self.decode_fc3 = nn.Linear(256, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc_mean(h2), self.fc_log_var(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.decode_fc1(z))
        h4 = F.relu(self.decode_fc2(h3))
        return self.decode_fc3(h4)

    def forward(self, x):
        # The view is to ensure the input is flattened
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def generate(self, n_samples=1):
        """
        Generate new samples from the latent space.
        """
        # Set model to evaluation mode
        self.eval()
        with torch.no_grad():
            z_sample = torch.randn(n_samples, self.latent_dim)
            samples = self.decode(z_sample)
        return samples


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss function = Reconstruction loss + beta * KL-Divergence
    """
    # Using Mean Squared Error for reconstruction loss
    recon_loss = F.mse_loss(recon_x, x.view(-1, x.shape[1]), reduction='sum')
    
    # KL-Divergence
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kld_loss 