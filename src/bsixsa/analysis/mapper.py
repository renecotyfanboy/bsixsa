"""Utilities for analysing parameter mappers between embeddings and posteriors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import cmasher as cmr
from torch.utils.data import TensorDataset

from ..nn.architectures import ResnetMapper
from ..nn.training import training_loop

if TYPE_CHECKING:
    from ..solver import SIXSASolver
    from ..embedding.abc import Embedding


def fit_parameter_mapper(
    solver: "SIXSASolver",
    embedding: "Embedding"=None,
    sampler_name: str = "prior",
    n_samples: int = 50_000,
) -> ResnetMapper:
    """Train a deterministic mapper from embedding space to parameters.

    The routine mirrors the ad-hoc diagnostics previously embedded in
    :class:`bsixsa.solver.SIXSASolver`. Given a ``solver`` instance, an
    ``embedding`` and a sampler name, it simulates spectra, fits a
    :class:`ResnetMapper` to predict parameters from embedding outputs, and
    renders scatter plots plus a correlation heatmap for quick inspection.

    Parameters:
        solver (SIXSASolver): Configured solver providing ``simulate`` and
            ``unit_cube_to_bxa`` helpers.
        embedding (Embedding): Embedding used to compress spectra prior to
            regression. For torch-based embeddings the associated device is
            honoured.
        sampler_name (str, optional): Sampler identifier registered on the
            solver. Defaults to ``"prior"``.
        n_samples (int, optional): Number of simulated spectra to draw.
            Defaults to ``50_000``.

    Returns:
        ResnetMapper: The trained mapper instance. The diagnostic figures are
        displayed inline via Matplotlib.
    """

    if embedding is None:
        raise ValueError("embedding must be provided to fit a parameter mapper")

    theta_unit, spectra, _ = solver.simulate(n_samples, sampler=sampler_name)
    theta_bxa = solver.unit_cube_to_bxa(theta_unit.numpy().T)
    theta = torch.from_numpy(theta_bxa.astype(np.float32)).T

    # Move spectra to the embedding device when available
    target_device = getattr(embedding, "device", None)
    spectra_for_embedding = (
        spectra.to(target_device)
        if target_device is not None
        else spectra
    )

    embedding_output = embedding(spectra_for_embedding)

    if isinstance(embedding_output, torch.Tensor):
        x = embedding_output.detach().cpu()
    else:
        x = torch.as_tensor(np.asarray(embedding_output), dtype=torch.float32)

    theta = theta.detach().cpu()

    mapper = ResnetMapper(
        input_dim=x.shape[1],
        output_dim=theta.shape[1],
        hidden_features=128,
        num_blocks=3,
        dropout_rate=0.2,
    )

    mapper.scaler_inputs.fit(x)
    mapper.scaler_outputs.fit(theta)

    mapper = training_loop(
        mapper,
        TensorDataset(x, theta),
        learning_rate=1e-4,
        min_delta=1e-3,
        max_epochs=1000,
        patience=30,
    )

    theta_pred = mapper(x)
    parameter_names = solver.parameter_names

    for i, parameter_name in enumerate(parameter_names):
        y = np.linspace(theta[:, i].min().item(), theta[:, i].max().item(), 2)
        plt.plot(y, y, color="black", linestyle="--", linewidth=2)
        plt.scatter(
            theta.detach()[-10_000:, i],
            theta_pred.detach()[-10_000:, i],
        )
        plt.xlabel(f"{parameter_name} [True]")
        plt.ylabel(f"{parameter_name} [Pred]")
        plt.show()

    X = theta.numpy()
    Y = theta_pred.detach().cpu().numpy()

    Xc = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=0) + 1e-12)
    Yc = (Y - Y.mean(axis=0)) / (Y.std(axis=0, ddof=0) + 1e-12)
    XY_corr = Xc.T @ Yc / X.shape[0]

    ax = plt.axes()
    sns.heatmap(
        XY_corr,
        annot=True,
        cmap=cmr.prinsenvlag_r,
        vmin=-1,
        vmax=1,
        xticklabels=[par.replace(" ", "\n") for par in parameter_names],
        yticklabels=[par.replace(" ", "\n") for par in parameter_names],
        ax=ax,
    )
    ax.set(xlabel="Input", ylabel="Predicted")
    plt.show()

    return mapper
