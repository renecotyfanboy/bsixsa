import numpy as np
import matplotlib.pyplot as plt
import torch
from abc import ABC, abstractmethod


class Embedding(ABC):
    trainable: bool = False
    device: torch.device = torch.device("cpu")

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

    def plot_coverage(
        self,
        x_train: torch.Tensor,
        x_o: torch.Tensor,
        round_number: int = 0,
        save_to_path: str = "",
    ) -> plt.Figure:
        """Plot the coverage of the training set compared to the actual observed value in the embedding space.

        Parameters:
            x_train (numpy.ndarray): Training set for each embedding dimension.
            x_o (numpy.ndarray): Actual observed value for the embedding dimension.
            round_number (int, optional): Current round number. Using zero will ignore this argument.
            save_to_path (str, optional): Path to save the figure. Using `""` will not save the figure.

        Returns:
            (matplotlib.figure.Figure): Figure containing the plot.
        """

        cols = int(np.ceil(np.sqrt(len(self.names))))
        rows = int(np.ceil(len(self.names) / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

        axes = axes.flatten()

        for i, feature_name in enumerate(self.names):
            axes[i].hist(
                x_train[:, i].detach().numpy(),
                bins=30,
                color="skyblue",
                edgecolor="black",
                log=True,
            )
            axes[i].axvline(
                x_o[i].detach().numpy(), color="red", linestyle="dashed", linewidth=2
            )
            axes[i].set_title(feature_name)

        plt.suptitle(
            f"Summary stats" + (f"- Round {round_number}" if round_number else "")
        )
        plt.tight_layout()

        if save_to_path:
            plt.savefig(
                save_to_path,
                bbox_inches="tight",
            )

        plt.close()
        return fig


class MultipleEmbedding(Embedding):
    """
    Merges multiple embeddings into a single embedding.
    """

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
