"""Diagnostic figures for SIXSA, using Matplotlib's object API."""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def plot_diagnostics(backend):
    """Save convergence, compression and importance-weight plots; return paths."""
    out = Path(backend.solver.outputfiles_basename) / "sixsa"
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(fig, name):
        FigureCanvasAgg(fig)
        path = out / name
        fig.savefig(path, bbox_inches="tight")
        paths.append(str(path))

    fig = Figure(figsize=(9, 3.5), layout="constrained")
    axes = fig.subplots(1, 2)
    history = backend.history
    rounds = [h["round"] for h in history]
    axes[0].plot(rounds, [h["k_hat"] for h in history], "o-")
    axes[0].axhline(backend.config.khat_threshold, color="C3", linestyle="--")
    axes[0].set(xlabel="Round", ylabel="Pareto k-hat")
    axes[1].plot(rounds, [h["efficiency"] for h in history], "o-")
    axes[1].axhline(backend.config.min_efficiency, color="C3", linestyle="--")
    axes[1].set(xlabel="Round", ylabel="ESS / exact folds (%)")
    save(fig, "convergence.pdf")

    fig = Figure(figsize=(9, 3.5), layout="constrained")
    axes = fig.subplots(1, 2)
    x = backend.solver.observed_counts
    reconstructed = backend.compressor.reconstruct(x)[0]
    axes[0].plot(x, color="0.5", lw=0.7, label="Observed counts")
    axes[0].plot(reconstructed, lw=0.8, label="Reconstruction")
    axes[0].set(xlabel="Noticed channel", ylabel="Counts", yscale="symlog")
    axes[0].legend()
    hist = backend.compressor.history
    axes[1].plot([h["generation"] for h in hist],
                 [h.get("val_cstat", np.nan) for h in hist], "o-")
    axes[1].set(xlabel="Compressor generation", ylabel="Validation C-stat")
    save(fig, "compressor.pdf")

    fig = Figure(figsize=(7, 3.5), layout="constrained")
    ax = fig.subplots()
    lw = backend.posterior.log_w_smoothed
    weights = np.sort(np.exp(lw[np.isfinite(lw)]))[::-1]
    ax.plot(np.arange(1, len(weights) + 1), weights)
    ax.set(xscale="log", yscale="log", xlabel="Draw rank",
           ylabel="Normalised PSIS weight",
           title=f"Final k-hat={backend.posterior.k_hat:.3f}; ESS={backend.posterior.ess:.0f}")
    save(fig, "weights.pdf")
    return paths
