import numpy as np
import torch

from .abc import Embedding, MultipleEmbedding


class IdentityEmbedding(Embedding):
    def __init__(self):
        super().__init__()
        self._n_features = 0

    @property
    def names(self) -> list[str]:
        return [f"channel {i}" for i in range(1, self._n_features + 1)]

    def transform(self, spectra):
        if isinstance(spectra, torch.Tensor):
            spectra = spectra.detach().cpu().numpy()
        embedded = np.atleast_2d(np.asarray(spectra, dtype=np.float32))
        self._n_features = embedded.shape[1]
        return embedded


__all__ = ["Embedding", "MultipleEmbedding", "IdentityEmbedding"]
