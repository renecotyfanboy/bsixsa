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


def _pairwise_squared_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = torch.sum(x * x, dim=1, keepdim=True)
    y_norm = torch.sum(y * y, dim=1).unsqueeze(0)
    return (x_norm + y_norm - 2.0 * (x @ y.T)).clamp_min(0.0)


def mmd_loss(
    z_sampled: torch.Tensor,
    z_prior: torch.Tensor,
    bandwidths: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0),
) -> torch.Tensor:
    dist_xx = _pairwise_squared_distances(z_sampled, z_sampled)
    dist_yy = _pairwise_squared_distances(z_prior, z_prior)
    dist_xy = _pairwise_squared_distances(z_sampled, z_prior)

    xx = 0.0
    yy = 0.0
    xy = 0.0
    for bandwidth in bandwidths:
        gamma = 1.0 / (2.0 * bandwidth * bandwidth)
        xx = xx + torch.exp(-gamma * dist_xx)
        yy = yy + torch.exp(-gamma * dist_yy)
        xy = xy + torch.exp(-gamma * dist_xy)

    return (xx.mean() + yy.mean() - 2.0 * xy.mean()).clamp_min(0.0)


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


class MMDVAEEncoderPipeline(nn.Module):
    """Full MMD-VAE encoder pipeline, suitable as an external embedding net."""

    def __init__(
        self,
        transform: nn.Module,
        scaler: nn.Module,
        backbone: nn.Module,
        mean_head: nn.Module,
        log_var_head: nn.Module,
        *,
        log_var_min: float = -30.0,
        log_var_max: float = 20.0,
    ):
        super().__init__()
        self.transform = transform
        self.scaler = scaler
        self.backbone = backbone
        self.mean_head = mean_head
        self.log_var_head = log_var_head
        self.log_var_min = float(log_var_min)
        self.log_var_max = float(log_var_max)

    @staticmethod
    def reparameterize(mean, log_var):
        std = torch.exp(0.5 * log_var)
        return mean + std * torch.randn_like(std)

    def encode_stats(self, x):
        x = self.transform.forward(x)
        x = self.scaler.forward(x)
        x = self.backbone(x)
        mean = self.mean_head(x)
        log_var = self.log_var_head(x).clamp(
            min=self.log_var_min, max=self.log_var_max
        )
        return mean, log_var

    def sample(self, x):
        mean, log_var = self.encode_stats(x)
        return self.reparameterize(mean, log_var)

    def forward(self, x):
        mean, _ = self.encode_stats(x)
        return mean


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


class MMDVariationalAutoencoder(nn.Module):
    def __init__(
        self,
        n_bins,
        latent_dim=32,
        hidden_dims: list[int] | None = None,
        mmd_weight: float = 10.0,
    ):
        super(MMDVariationalAutoencoder, self).__init__()

        transform = LogTransform()
        scaler = StandardScaler()

        if hidden_dims is None:
            raise ValueError("hidden_dims cannot be None")

        self.hidden_dims = list(hidden_dims)
        self.mmd_weight = float(mmd_weight)

        encoder_layers: list[nn.Module] = []
        prev_dim = n_bins
        for hidden_dim in self.hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                ]
            )
            prev_dim = hidden_dim
        encoder_backbone = nn.Sequential(*encoder_layers)
        encoder_mean = nn.Linear(prev_dim, latent_dim)
        encoder_log_var = nn.Linear(prev_dim, latent_dim)
        self.encoder_module = MMDVAEEncoderPipeline(
            transform=transform,
            scaler=scaler,
            backbone=encoder_backbone,
            mean_head=encoder_mean,
            log_var_head=encoder_log_var,
        )

        decoder_layers: list[nn.Module] = []
        prev_dim = latent_dim
        for hidden_dim in reversed(self.hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                ]
            )
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, n_bins))
        self.decoder_module = nn.Sequential(*decoder_layers)

        self.latent_dim = latent_dim
        self.input_dim = n_bins

    @property
    def transform(self):
        return self.encoder_module.transform

    @property
    def scaler(self):
        return self.encoder_module.scaler

    @property
    def embedding_network(self) -> nn.Module:
        """External embedding module, ready to pass where an `nn.Module` is expected."""
        return self.encoder_module

    @property
    def encoder_backbone(self):
        return self.encoder_module.backbone

    @property
    def encoder_mean(self):
        return self.encoder_module.mean_head

    @property
    def encoder_log_var(self):
        return self.encoder_module.log_var_head

    def _encode_inputs(self, x):
        return self.encoder_module.encode_stats(x)

    @staticmethod
    def _reparameterize(mean, log_var):
        return MMDVAEEncoderPipeline.reparameterize(mean, log_var)

    def encoder(self, x):
        mean, _ = self._encode_inputs(x)
        return mean

    def decoder(self, x):
        x = self.decoder_module(x)
        x = self.scaler.inverse(x)
        x = self.transform.inverse(x)
        return x

    def forward(self, x):
        mean, log_var = self._encode_inputs(x)
        z = self._reparameterize(mean, log_var)
        return self.decoder(z)

    def _forward_with_latent(self, x):
        mean, log_var = self._encode_inputs(x)
        z = self._reparameterize(mean, log_var)
        pred = self.decoder(z)
        return pred, z, mean, log_var

    def loss(self, batch: torch.Tensor):
        spectrum, = batch
        pred, z, _, _ = self._forward_with_latent(spectrum)

        reconstruction = cstat_loss(spectrum, pred)
        mmd = mmd_loss(z, torch.randn_like(z))
        loss = reconstruction + self.mmd_weight * mmd

        return loss, {
            "loss": loss,
            "reconstruction": reconstruction,
            "mmd": mmd,
        }


class ResnetAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32, hidden_features=128, num_blocks=2, dropout_probability=0.1):
        super(ResnetAutoencoder, self).__init__()

        self.transform = LogTransform()
        self.scaler = StandardScaler()
        self.encoder_module = nets.ResidualNet(
            in_features=input_dim,
            out_features=latent_dim,
            hidden_features=hidden_features,
            num_blocks=num_blocks,
            dropout_probability=dropout_probability,
            use_batch_norm=True
        )

        self.decoder_module = nets.ResidualNet(
            in_features=latent_dim,
            out_features=input_dim,
            hidden_features=hidden_features,
            num_blocks=num_blocks,
            dropout_probability=dropout_probability,
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
