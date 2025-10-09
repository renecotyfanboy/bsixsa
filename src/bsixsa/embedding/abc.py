import numpy as np
from abc import ABC, abstractmethod


class Embedding(ABC):
    trainable: bool = False

    @abstractmethod
    def __call__(self, spectra):
        """Compress spectra into the embedding space.

        Parameters:
            spectra (numpy.ndarray): Input spectra to transform. The array can
                be 1D (single spectrum) or 2D (batch of spectra).

        Returns:
            (numpy.ndarray): Embedded representation for each spectrum.
        """
        pass

    @property
    @abstractmethod
    def names(self) -> list[str]:
        """Return human-readable labels for each embedding dimension.

        Returns:
            (list[str]): Names aligned with the embedding output order.
        """
        return []


class MultipleEmbedding(Embedding):
    def __init__(self, embeddings: list[Embedding]):
        self.embeddings = embeddings

    @property
    def names(self) -> list[str]:
        return [name for embedding in self.embeddings for name in embedding.names]

    def __call__(self, spectra):
        reduced_spectra_list = []

        if spectra.ndim == 1:
            spectra = spectra[None, :]

        for embedding in self.embeddings:
            reduced_spectra_list.append(embedding(spectra))

        return np.hstack(reduced_spectra_list)
