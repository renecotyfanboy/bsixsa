from .abc import Embedding
from abc import abstractmethod, ABC
from torch import nn
from torch.utils.data import TensorDataset
import numpy as np
import torch
import xspec
from .nn import Autoencoder, ResnetAutoencoder
from .nn.training import training_loop


class TrainableEmbedding(Embedding, ABC):
    model: nn.Module
    trainable = True

    @abstractmethod
    def train(self, *args, **kwargs):
        pass


class TorchModuleEmbedding(TrainableEmbedding, ABC):
    def __init__(self, retrain_from_scratch: bool = True, **model_kwargs):
        self.retrain_from_scratch = retrain_from_scratch

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

    def train(
        self,
        data,
        *,
        metrics_path,
        max_epochs: int = 1_000,
        prefix="Training | ",
        **kwargs,
    ):
        # Reset the weights
        if self.retrain_from_scratch:
            self.model = self.build_model().to(self.device)

        # Fit the scaler if it exists
        scaled_data = self.model.transform.forward(
            data.to(self.device)
        )
        self.model.scaler.fit(scaled_data)

        # Train the model
        self.model = training_loop(
            self.model, TensorDataset(data),
            device=self.device,
            max_epochs=max_epochs,
            metrics_path=metrics_path,
            prefix=prefix,
            **kwargs
        )


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

    def __call__(self, spectra):
        if not isinstance(spectra, torch.Tensor):
            spectra = torch.from_numpy(spectra.astype(np.float32)).to(self.device)

        return self.model.encoder(spectra)

    def train(self, data, **kwargs):
        metrics_path = kwargs.pop("metrics_path", None)

        super().train(
            data,
            max_epochs=1_000,
            prefix="Autoencoder | ",
            metrics_path=metrics_path,
            **kwargs,
        )


class ResnetEmbedding(TorchModuleEmbedding):
    def __init__(self, latent_dim=32, hidden_features=128, num_blocks=3, **kwargs):
        self.latent_dim = latent_dim
        self.model_kwargs = dict(hidden_dims=hidden_features, num_blocks=num_blocks)
        super().__init__(**kwargs)

    @property
    def embedding_dim(self):
        return self.latent_dim

    def build_model(self, **kwargs):
        return ResnetAutoencoder(self.input_dim, self.latent_dim, **kwargs)

    def __call__(self, spectra):
        if not isinstance(spectra, torch.Tensor):
            spectra = torch.from_numpy(spectra.astype(np.float32)).to(self.device)

        return self.model.encoder(spectra)

    def train(self, data, **kwargs):
        metrics_path = kwargs.pop("metrics_path", None)

        super().train(
            data,
            max_epochs=1_000,
            prefix="Autoencoder | ",
            metrics_path=metrics_path,
            **kwargs,
        )