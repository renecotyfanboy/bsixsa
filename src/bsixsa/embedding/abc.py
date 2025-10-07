import numpy as np
from abc import ABC, abstractmethod


class Embedding(ABC):

    trainable: bool = False

    @abstractmethod

    @abstractmethod
    def __call__(self, spectra):
        """
        Implements the compression scheme related to the embedding
        """
        pass

    @property
    @abstractmethod
    def names(self) -> list[str]:
        """
        Returns the names of the embedding dimensions
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