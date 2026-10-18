"""Diagnostic figures for SIXSA."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from ._sixsa import SIXSABackend


def plot_diagnostics(backend: SIXSABackend) -> list[str]:
    """Save the three diagnostic figures of a finished run under `backend.outdir`.

    - `convergence.pdf`: k-hat and efficiency per round against the gate
      thresholds (dashed).
    - `compressor.pdf`: the observed counts and their reconstruction (a
      systematic misfit means the code misses features of the data), and the
      validation C-stat of each compressor fit.
    - `weights.pdf`: the final importance weights sorted by size; a steep head
      means a few draws dominate the posterior.

    Parameters:
        backend: The SIXSA backend after `run()`.

    Returns:
        The paths of the saved figures.
    """
    out = backend.outdir
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(fig, name):
        FigureCanvasAgg(fig)
        path = out / name
        fig.savefig(path, bbox_inches="tight")
        paths.append(str(path))

    fig = Figure(figsize=(9, 3.5), layout="constrained")
    axes = fig.subplots(1, 2)
    rounds = backend.history
    round_numbers = [record["round"] for record in rounds]
    axes[0].plot(round_numbers, [record["k_hat"] for record in rounds], "o-")
    axes[0].axhline(backend.config.khat_threshold, color="C3", linestyle="--")
    axes[0].set(xlabel="Round", ylabel="Pareto k-hat")
    axes[1].plot(round_numbers, [record["efficiency"] for record in rounds], "o-")
    axes[1].axhline(backend.config.min_efficiency, color="C3", linestyle="--")
    axes[1].set(xlabel="Round", ylabel="ESS / exact folds (%)")
    save(fig, "convergence.pdf")

    fig = Figure(figsize=(9, 3.5), layout="constrained")
    axes = fig.subplots(1, 2)
    observed = backend.solver.observed_counts
    reconstructed = backend.compressor.reconstruct(observed)[0]
    axes[0].plot(observed, color="0.5", lw=0.7, label="Observed counts")
    axes[0].plot(reconstructed, lw=0.8, label="Reconstruction")
    axes[0].set(xlabel="Noticed channel", ylabel="Counts", yscale="symlog")
    axes[0].legend()
    compressor_fits = backend.compressor.history
    axes[1].plot([fit["generation"] for fit in compressor_fits], [fit["val_cstat"] for fit in compressor_fits], "o-")
    axes[1].set(xlabel="Compressor generation", ylabel="Validation C-stat")
    save(fig, "compressor.pdf")

    fig = Figure(figsize=(7, 3.5), layout="constrained")
    ax = fig.subplots()
    log_w = backend.posterior.log_w_smoothed
    weights = np.sort(np.exp(log_w[np.isfinite(log_w)]))[::-1]
    ax.plot(np.arange(1, len(weights) + 1), weights)
    ax.set(xscale="log", yscale="log", xlabel="Draw rank",
           ylabel="Normalised PSIS weight",
           title=f"Final k-hat={backend.posterior.k_hat:.3f}; ESS={backend.posterior.ess:.0f}")
    save(fig, "weights.pdf")
    return paths
