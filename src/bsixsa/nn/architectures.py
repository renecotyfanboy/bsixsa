import torch
from pyknos.nflows.nn import nets
from torch import nn


def cstat_loss(y_true, y_pred, eps=1e-6):

    term = torch.where(
        (y_true > 0.0) & (y_pred > 0.0),
        torch.log(y_true.clamp(min=eps, max=1e13)) - torch.log(y_pred.clamp(min=eps, max=1e13)),
        0.0
    )

    cstat = 2 * torch.sum(y_pred - y_true + y_true * term, dim=1)

    return cstat.mean()


class LogTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x = torch.log1p(x)
        return x

    def inverse(self, x: torch.Tensor):
        x = torch.expm1(x.clamp(min=0., max=30.))

        return x


class StandardScaler(nn.Module):
    def __init__(self):
        super(StandardScaler, self).__init__()
        self.eps = 1e-6
        self.register_buffer("mean_", torch.Tensor([0.]), persistent=True)
        self.register_buffer("scale_", torch.Tensor([1.]), persistent=True)

    def fit(self, x: torch.Tensor):
        self.register_buffer("mean_", torch.mean(x, dim=0), persistent=True)
        self.register_buffer("scale_", torch.std(x, dim=0), persistent=True)

    def forward(self, x: torch.Tensor):
        return (x - self.mean_) / (self.scale_ + self.eps)

    def inverse(self, x: torch.Tensor):
        return (x * (self.scale_ + self.eps)) + self.mean_


class MlpMapper(nn.Module):
    def __init__(self, input_dim=16, output_dim=5, hidden_dims=[128, 64, 32]):
        super().__init__()
        self.scaler_inputs = StandardScaler()
        self.scaler_outputs = StandardScaler()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], output_dim)
        )

    def forward_unscaled(self, x):
        x = self.scaler_inputs.forward(x)
        return self.model(x)

    def forward(self, x):
        x = self.forward_unscaled(x)
        return self.scaler_outputs.inverse(x)

    def loss(self, batch):
        x, theta = batch

        theta = self.scaler_outputs.forward(theta)
        pred = self.forward_unscaled(x)
        loss = torch.mean((pred - theta)**2)

        return loss, {"loss": loss.detach()}


class ResnetMapper(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_features=128, num_blocks=3, dropout_rate=0.2, use_batch_norm=True):
        super().__init__()
        self.scaler_inputs = StandardScaler()
        self.scaler_outputs = StandardScaler()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.resnet = nets.ResidualNet(
            in_features=self.input_dim,
            out_features=self.output_dim,
            hidden_features=hidden_features,
            num_blocks=num_blocks,
            dropout_probability=dropout_rate,
            use_batch_norm=use_batch_norm
        )

    def forward(self, x):
        x = self.forward_unscaled(x)
        x = self.scaler_outputs.inverse(x)

        return x

    def forward_unscaled(self, x):
        x = self.scaler_inputs.forward(x)
        x = self.resnet(x)

        return x

    def loss(self, batch: torch.Tensor):
        x, theta = batch

        theta = self.scaler_outputs.forward(theta)
        pred = self.forward_unscaled(x)
        loss = torch.mean((pred - theta)**2)

        return loss, {"loss": loss.detach()}


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

    def loss(self, batch: torch.Tensor):
        spectrum, = batch
        pred = self.forward(spectrum)
        loss = cstat_loss(spectrum, pred)

        return loss, {
            "loss": loss,
            "reconstruction": loss,
        }


class ResnetAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32, hidden_features=128, num_blocks=2):
        super(ResnetAutoencoder, self).__init__()

        self.transform = LogTransform()
        self.scaler = StandardScaler()
        self.encoder_module = nets.ResidualNet(
            in_features=input_dim,
            out_features=latent_dim,
            hidden_features=hidden_features,
            num_blocks=num_blocks,
            use_batch_norm=True
        )
        self.decoder_module = nets.ResidualNet(
            in_features=latent_dim,
            out_features=input_dim,
            hidden_features=hidden_features,
            num_blocks=num_blocks,
            use_batch_norm=True
        )


        self.latent_dim = latent_dim
        self.input_dim = input_dim

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

    def loss(self, batch: torch.Tensor):
        spectrum, = batch
        pred = self.forward(spectrum)
        loss = cstat_loss(spectrum, pred)

        return loss, {
            "loss": loss,
            "reconstruction": loss,
        }