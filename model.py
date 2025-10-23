import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import LATENT_DIM

class BetaCVAE(nn.Module):
    def __init__(self, input_channels=1, seq_len=23, latent_dim=LATENT_DIM, cond_dim=1):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels + cond_dim, 32, 3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Hidden length chosen as ceil(seq_len/4) to have room for upsampling
        hidden_len = max(1, math.ceil(seq_len / 4))
        self.decoder_input = nn.Linear(latent_dim + cond_dim, 128 * hidden_len)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, hidden_len)),
            nn.ConvTranspose1d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, input_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x, c):
        if c.dim() == 1:
            c = c.unsqueeze(1).unsqueeze(2)
        elif c.dim() == 2:
            c = c.unsqueeze(2)
        c = c.repeat(1, 1, x.size(-1))
        h = self.encoder(torch.cat([x, c], dim=1)).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        zc = torch.cat([z, c], dim=1)
        h = self.decoder_input(zc)
        x_recon = self.decoder(h)
        return F.interpolate(x_recon, size=self.seq_len, mode="linear", align_corners=False)

    def forward(self, x, c):
        # Normalize and ensure c shape
        if c.dim() == 1:
            c = c.unsqueeze(1)  # [B, 1]
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

def beta_vae_loss(x, x_recon, mu, logvar, beta=4.0):
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld, recon_loss, kld
