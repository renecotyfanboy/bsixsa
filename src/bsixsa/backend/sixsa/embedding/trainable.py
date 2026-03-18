from .abc import Embedding
from abc import abstractmethod, ABC
from torch import nn
from torch.utils.data import TensorDataset
import numpy as np
import torch
import xspec
from .nn import Autoencoder, MMDVariationalAutoencoder, ResnetAutoencoder
from .nn.training import training_loop


class TrainableEmbedding(Embedding, ABC):
    model: nn.Module
    trainable = True

    def __init__(self):
        super().__init__()
        self._is_fitted = False

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass


def _as_tensor_2d(data) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data.to(dtype=torch.float32)
    else:
        tensor = torch.from_numpy(np.asarray(data, dtype=np.float32))

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)

    return tensor


class TorchModuleEmbedding(TrainableEmbedding, ABC):
    def __init__(
        self,
        retrain_from_scratch: bool = True,
        auto_fit_on_transform: bool = True,
        **model_kwargs,
    ):
        super().__init__()
        self.retrain_from_scratch = retrain_from_scratch
        self.auto_fit_on_transform = auto_fit_on_transform

        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model_kwargs = model_kwargs
        self.model = self.build_model(**self.model_kwargs).to(self.device)

    @property
    def input_dim(self):
        data = np.asarray(xspec.AllData(1).values)
        return len(data)

    @property
    def embedding_dim(self):
        data = np.asarray(xspec.AllData(1).values)
        return int(
            self.model(
                torch.from_numpy(data.astype(np.float32)).to(self.device).unsqueeze(0)
            ).shape[1]
        )

    @abstractmethod
    def build_model(self, **kwargs):
        pass

    @property
    def names(self) -> list[str]:
        return [f"latent {i}" for i in range(1, self.embedding_dim + 1)]

    def fit(
        self,
        data,
        *,
        metrics_path=None,
        max_epochs: int = 1_000,
        prefix="Training | ",
        force=False,
        **kwargs,
    ):
        if self.is_fitted and not force:
            return self

        data_tensor = _as_tensor_2d(data).detach().cpu()

        # Reset the weights
        if self.retrain_from_scratch:
            self.model = self.build_model(**self.model_kwargs).to(self.device)

        # Fit the scaler if it exists
        scaled_data = self.model.transform.forward(
            data_tensor.to(self.device)
        )
        self.model.scaler.fit(scaled_data)

        # Train the model
        self.model = training_loop(
            self.model, TensorDataset(data_tensor),
            device=self.device,
            max_epochs=max_epochs,
            metrics_path=metrics_path,
            prefix=prefix,
            **kwargs
        )
        self._is_fitted = True
        return self

    def transform(self, spectra):
        if self.auto_fit_on_transform and not self.is_fitted:
            self.fit(spectra)

        if not self.is_fitted:
            raise RuntimeError(
                "Embedding is not fitted. Call `fit` before `transform`."
            )

        spectra_tensor = _as_tensor_2d(spectra).to(self.device)
        with torch.no_grad():
            embedded = self.model.encoder(spectra_tensor)
        return embedded.detach().cpu().numpy()

    # Backward-compatible alias
    def train(self, data, **kwargs):
        return self.fit(data, **kwargs)


class AutoencoderEmbedding(TorchModuleEmbedding):
    def __init__(self, latent_dim=32, hidden=(2, 4), **kwargs):
        self.latent_dim = latent_dim
        model_kwargs = dict(hidden_dims=[self.input_dim // h for h in hidden])

        super().__init__(**(model_kwargs | kwargs))

    @property
    def embedding_dim(self):
        return self.latent_dim

    def build_model(self, **kwargs):
        return Autoencoder(self.input_dim, self.latent_dim, **kwargs)

    def fit(self, data, **kwargs):
        kwargs.setdefault("max_epochs", 1_000)
        kwargs.setdefault("prefix", "Autoencoder | ")
        return super().fit(data, **kwargs)


class ResnetEmbedding(TorchModuleEmbedding):
    def __init__(self, latent_dim=32, hidden_features=128, num_blocks=3, **kwargs):
        self.latent_dim = latent_dim
        model_kwargs = dict(hidden_features=hidden_features, num_blocks=num_blocks)
        super().__init__(**(model_kwargs | kwargs))

    @property
    def embedding_dim(self):
        return self.latent_dim

    def build_model(self, **kwargs):
        return ResnetAutoencoder(self.input_dim, self.latent_dim, **kwargs)

    def fit(self, data, **kwargs):
        kwargs.setdefault("max_epochs", 1_000)
        kwargs.setdefault("prefix", "Autoencoder | ")
        return super().fit(data, **kwargs)


class MMDVariationalAutoencoderEmbedding(TorchModuleEmbedding):
    def __init__(
        self,
        latent_dim=32,
        hidden=(2, 4),
        mmd_weight: float = 10.0,
        **kwargs,
    ):
        self.latent_dim = latent_dim
        model_kwargs = dict(
            hidden_dims=[self.input_dim // h for h in hidden],
            mmd_weight=mmd_weight,
        )
        super().__init__(**(model_kwargs | kwargs))

    @property
    def embedding_dim(self):
        return self.latent_dim

    def build_model(self, **kwargs):
        return MMDVariationalAutoencoder(self.input_dim, self.latent_dim, **kwargs)

    def fit(self, data, **kwargs):
        kwargs.setdefault("max_epochs", 1_000)
        kwargs.setdefault("prefix", "MMD-VAE | ")
        return super().fit(data, **kwargs)

    def transform(self, spectra):
        if self.auto_fit_on_transform and not self.is_fitted:
            self.fit(spectra)

        if not self.is_fitted:
            raise RuntimeError(
                "Embedding is not fitted. Call `fit` before `transform`."
            )

        spectra_tensor = _as_tensor_2d(spectra).to(self.device)
        with torch.no_grad():
            embedded = self.model.encoder_module.sample(spectra_tensor)
        return embedded.detach().cpu().numpy()
