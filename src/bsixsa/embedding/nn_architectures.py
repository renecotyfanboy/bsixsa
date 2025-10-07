from typing import Any

import torch
from torch import nn


class Scaler(nn.Module):

    def __init__(self, data):
        super(Scaler, self).__init__()


class Autoencoder(nn.Module):
    def __init__(self, n_bins, latent_dim=32):
        super(Autoencoder, self).__init__()

        # Encoder
        """
        self.encoder_module = nn.Sequential(
            nn.Linear(n_bins, n_bins//2),
            nn.LayerNorm(n_bins//2),
            nn.GELU(),
            nn.Linear(n_bins//2, n_bins // 4),
            nn.LayerNorm(n_bins // 4),
            nn.GELU(),
            nn.Linear(n_bins//4, latent_dim)
        )

        # Decoder
        self.decoder_module = nn.Sequential(
            nn.Linear(latent_dim, n_bins//4),
            nn.LayerNorm(n_bins//4),
            nn.GELU(),
            nn.Linear(n_bins//4, n_bins//2),
            nn.LayerNorm(n_bins // 2),
            nn.GELU(),
            nn.Linear(n_bins // 2, n_bins),
        )
        """

        self.encoder_module = nn.Sequential(
            nn.Linear(n_bins, n_bins//2),
            nn.BatchNorm1d(n_bins//2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(n_bins //2, latent_dim),
        )

        # Decoder
        self.decoder_module = nn.Sequential(
            nn.Linear(latent_dim, n_bins//2),
            nn.BatchNorm1d(n_bins//2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(n_bins // 2, n_bins),
        )

        self.latent_dim = latent_dim
        self.input_dim = n_bins
        self.is_scaled = False

    def set_scaler(self, mean, std) -> None:

        self.register_buffer("mean", mean, persistent=True)
        self.register_buffer("std", std, persistent=True)
        self.is_scaled = True

    def encoder(self, x):

        x = torch.log1p(x)
        x = (x - self.mean) / self.std
        x = self.encoder_module(x)

        return x

    def decoder(self, x):

        x = self.decoder_module(x)
        x = x * self.std + self.mean
        x = torch.exp(x)

        return x

    def forward(self, x):

        return self.decoder(self.encoder(x))


class VariationaAutoencoder(nn.Module):
    def __init__(self, n_bins, latent_dim=32):
        super(VariationaAutoencoder, self).__init__()

        # Encoder
        """
        self.encoder_module = nn.Sequential(
            nn.Linear(n_bins, n_bins//2),
            nn.LayerNorm(n_bins//2),
            nn.GELU(),
            nn.Linear(n_bins//2, n_bins // 4),
            nn.LayerNorm(n_bins // 4),
            nn.GELU(),
            nn.Linear(n_bins//4, latent_dim)
        )

        # Decoder
        self.decoder_module = nn.Sequential(
            nn.Linear(latent_dim, n_bins//4),
            nn.LayerNorm(n_bins//4),
            nn.GELU(),
            nn.Linear(n_bins//4, n_bins//2),
            nn.LayerNorm(n_bins // 2),
            nn.GELU(),
            nn.Linear(n_bins // 2, n_bins),
        )
        """

        self.encoder_module = nn.Sequential(
            nn.Linear(n_bins, n_bins // 2),
            nn.BatchNorm1d(n_bins // 2),
            nn.GELU(),
        )

        # Decoder
        self.decoder_module = nn.Sequential(
            nn.Linear(latent_dim, n_bins // 2),
            nn.BatchNorm1d(n_bins // 2),
            nn.GELU(),
            nn.Linear(n_bins // 2, n_bins),
        )

        self.mu_layer = nn.Linear(n_bins // 2, latent_dim)
        self.logvar_layer = nn.Linear(n_bins // 2, latent_dim)

        self.latent_dim = latent_dim
        self.input_dim = n_bins
        self.is_scaled = False

    def set_scaler(self, mean, std) -> None:

        self.register_buffer("mean", mean, persistent=True)
        self.register_buffer("std", std, persistent=True)
        self.is_scaled = True

    def encoder(self, x):
        x = torch.log1p(x)
        x = (x - self.mean) / self.std
        x = self.encoder_module(x)
        mu, logvar = self.mu_layer(x), self.logvar_layer(x)
        return mu, logvar

    def sampler(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decoder(self, x):

        x = self.decoder_module(x)
        x = x * self.std + self.mean
        x = torch.exp(x)
        return x

    def forward(self, x):

        mu, logvar = self.encoder(x)
        z = self.sampler(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z