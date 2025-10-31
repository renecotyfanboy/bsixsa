import torch
from torch import nn


class LogTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus(beta=10.0)

    def forward(self, x: torch.Tensor):
        x = torch.log1p(x)
        return x

    def inverse(self, x: torch.Tensor):
        x = self.softplus(x)
        x = torch.expm1(x)

        return x


class StandardScaler(nn.Module):
    def __init__(self):
        super(StandardScaler, self).__init__()

    def fit(self, x: torch.Tensor):
        self.register_buffer("mean_", torch.mean(x, dim=0), persistent=True)
        self.register_buffer("scale_", torch.std(x, dim=0), persistent=True)

    def forward(self, x: torch.Tensor):
        return (x - self.mean_) / (self.scale_ + 1e-3)

    def inverse(self, x: torch.Tensor):
        return (x * (self.scale_ + 1e-3)) + self.mean_


class Autoencoder(nn.Module):
    def __init__(self, n_bins, latent_dim=32, hidden_dims: list[int] | None = None):
        super(Autoencoder, self).__init__()

        self.transform = LogTransform()
        self.scaler = StandardScaler()

        if hidden_dims is None:
            raise ValueError("hidden_dims cannot be None")

        self.hidden_dims = list(hidden_dims)

        encoder_layers: list[nn.Module] = []
        prev_dim = n_bins
        for hidden_dim in self.hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.01),
                ]
            )
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder_module = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        prev_dim = latent_dim
        for hidden_dim in reversed(self.hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.01),
                ]
            )
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, n_bins))
        self.decoder_module = nn.Sequential(*decoder_layers)

        self.latent_dim = latent_dim
        self.input_dim = n_bins

    def encoder(self, x):
        x = self.transform.forward(x)
        x = self.scaler.forward(x)
        x = self.encoder_module(x)

        return x

    def decoder(self, x):
        x = self.decoder_module(x)
        x = self.scaler.inverse(x)
        x = self.transform.inverse(x)

        return x

    def forward(self, x):
        return self.decoder(self.encoder(x))


class VariationalAutoencoder(nn.Module):
    def __init__(self, n_bins, latent_dim=32, hidden_dims: list[int] | None = None):
        super(VariationalAutoencoder, self).__init__()

        default_hidden_dims = [
            max(1, n_bins // 2),
            max(1, n_bins // 4),
            max(1, n_bins // 8),
        ]
        self.hidden_dims = (
            list(hidden_dims) if hidden_dims is not None else default_hidden_dims
        )

        encoder_layers: list[nn.Module] = []
        prev_dim = n_bins
        for hidden_dim in self.hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.01),
                ]
            )
            prev_dim = hidden_dim
        self.encoder_module = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers: list[nn.Module] = []
        prev_decoder_dim = latent_dim
        for hidden_dim in reversed(self.hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_decoder_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.01),
                ]
            )
            prev_decoder_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_decoder_dim, n_bins))
        self.decoder_module = nn.Sequential(*decoder_layers)

        self.mu_layer = nn.Linear(prev_dim, latent_dim)
        self.logvar_layer = nn.Linear(prev_dim, latent_dim)

        self.latent_dim = latent_dim
        self.input_dim = n_bins
        self.is_scaled = False

    def set_scaler(self, mean, std) -> None:
        self.register_buffer("mean", mean, persistent=True)
        self.register_buffer("std", std, persistent=True)
        self.is_scaled = True

    def encoder(self, x):
        x = torch.log1p(x)

        if torch.isnan(x).any():
            raise ValueError("NaN in encoder")

        x = (x - self.mean) / self.std

        if torch.isnan(x).any():
            raise ValueError("NaN in encoder")

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
        x = torch.expm1(x)
        return x

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.sampler(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z
