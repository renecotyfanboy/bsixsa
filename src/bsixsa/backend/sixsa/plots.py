"""Diagnostic plots for the SIXSA backend, following each training round.

Ported from ``_sixsa_dev/utils/plot_utils.py``. These are pure-matplotlib and do
**not** force ``text.usetex`` (mathtext labels only), so they render without a
LaTeX toolchain. Every entry point fails soft: a plotting error returns ``None``
rather than aborting an inference run.

Three plots:

- :func:`plot_training_history` — six-panel run summary from the per-round
  ``history`` (validation loss, training time, cumulative simulations,
  ``Delta logL``, PSIS k-hat, IS efficiency).
- :func:`plot_round_coverage` — per-round predictive band over the observation,
  with the importance-sampling-selected and rejected pools drawn separately.
- :func:`plot_ensemble_posteriors` — overlay of each NDE posterior and the
  ensemble posterior for one round (shows ensemble stability).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _to_numpy(a):
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)


def plot_training_history(history, khat_threshold=None, outdir=None,
                          filename="training_history.pdf", save=True):
    """Six-panel run summary from SIXSA's per-round ``history``.

    Panels: (1) per-NDE best validation loss with the per-round median;
    (2) ensemble training time per round; (3) cumulative simulations;
    (4) ``Delta logL = -0.5 * cstat`` of the IS-selected samples (min-max band +
    median); (5) PSIS k-hat with the convergence threshold and the 0.7 line;
    (6) IS efficiency (%). Round axes use integer ticks and the function degrades
    gracefully when optional keys are missing.

    Args:
        history: List of per-round dicts produced by the backend.
        khat_threshold: Convergence threshold to mark on the k-hat panel.
        outdir: Directory to save the figure into (``None`` -> do not save).
        filename: Output filename when saving.
        save: Whether to write the figure to ``outdir/filename``.

    Returns:
        ``(fig, axes)`` or ``None`` on failure / empty history.
    """
    if not history:
        return None
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        rounds = [h["round"] for h in history]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        def _int_x(ax):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # (1) per-NDE validation loss: one connected line per member + median
        ax = axes[0, 0]
        n_members = max((len(h.get("nde_stats", [])) for h in history), default=0)
        cmap = plt.cm.viridis(np.linspace(0.15, 0.85, max(n_members, 1)))
        for j in range(n_members):
            xs, ys = [], []
            for h in history:
                nde = h.get("nde_stats", [])
                if j < len(nde):
                    xs.append(h["round"])
                    ys.append(nde[j].get("best_val_loss", np.nan))
            ax.plot(xs, ys, "-o", ms=4, color=cmap[j], alpha=0.9,
                    label=f"NDE {j + 1}")
        med_val = [np.nanmedian([s.get("best_val_loss", np.nan)
                                 for s in h.get("nde_stats", [])])
                   if h.get("nde_stats") else np.nan for h in history]
        ax.plot(rounds, med_val, "--", color="crimson", lw=1.4, label="median")
        ax.set_xlabel("round"); ax.set_ylabel("best val loss")
        ax.set_title("NDE validation loss")
        ax.legend(frameon=False, fontsize=8, ncol=2); _int_x(ax)

        # (2) ensemble training time + total in legend
        ax = axes[0, 1]
        tt = [sum(s.get("training_time_s", 0.0) for s in h.get("nde_stats", []))
              for h in history]
        total_tt = float(np.nansum(tt))
        ax.bar(rounds, tt, color="steelblue", alpha=0.8,
               label=f"total = {total_tt:.0f} s")
        ax.set_xlabel("round"); ax.set_ylabel("seconds")
        ax.set_title("Ensemble training time")
        ax.legend(frameon=False, fontsize=8); _int_x(ax)

        # (3) cumulative simulations
        ax = axes[0, 2]
        nsim = [h.get("n_simulations_total", np.nan) for h in history]
        total_sims = next((int(v) for v in reversed(nsim) if np.isfinite(v)), None)
        ax.plot(rounds, nsim, "-o", color="steelblue",
                label=(f"total = {total_sims:,}" if total_sims is not None else None))
        ax.set_xlabel("round"); ax.set_ylabel("simulations")
        ax.set_title("Total number of simulations")
        if total_sims is not None:
            ax.legend(frameon=False, fontsize=8)
        _int_x(ax)

        # (4) Delta log-likelihood (= -0.5 * cstat) of IS-selected samples
        ax = axes[1, 0]
        if all("cstat_min" in h for h in history):
            dl_lo = [-0.5 * h["cstat_max"] for h in history]
            dl_hi = [-0.5 * h["cstat_min"] for h in history]
            dl_med = [-0.5 * h["cstat_median"] for h in history]
            ax.fill_between(rounds, dl_lo, dl_hi, color="steelblue", alpha=0.3,
                            label="min-max")
            ax.plot(rounds, dl_med, "-o", color="crimson", label="median")
        else:
            ax.plot(rounds, [-0.5 * h.get("min_cstat_new", np.nan) for h in history],
                    "-o", color="steelblue", label=r"max $\Delta\log L$")
        ax.set_xlabel("round")
        ax.set_ylabel(r"$\Delta\log L = -0.5\,\mathrm{cstat}$")
        ax.set_title(r"$\Delta\log L$ of IS-selected samples")
        ax.legend(frameon=False, fontsize=8); _int_x(ax)

        # (5) k_hat with threshold + reliability line
        ax = axes[1, 1]
        ax.plot(rounds, [h.get("k_hat", np.nan) for h in history], "-o",
                color="steelblue", label=r"$\hat{k}$")
        if khat_threshold is not None:
            ax.axhline(khat_threshold, ls="--", color="crimson",
                       label=f"threshold = {khat_threshold}")
        ax.axhline(0.7, ls=":", color="0.5", label="0.7 (reliability)")
        ax.set_xlabel("round"); ax.set_ylabel(r"$\hat{k}$")
        ax.set_title("PSIS k-hat"); ax.legend(frameon=False, fontsize=8); _int_x(ax)

        # (6) IS efficiency, on a percentage axis
        ax = axes[1, 2]
        ax.plot(rounds, [h.get("efficiency", np.nan) for h in history], "-o",
                color="steelblue")
        ax.set_xlabel("round")
        ax.set_ylabel(r"efficiency (\%)" if plt.rcParams.get("text.usetex", False)
                      else "efficiency (%)")
        ax.set_ylim(0, 100)
        ax.set_title("IS efficiency"); _int_x(ax)

        fig.tight_layout()
        if save and outdir is not None:
            outpath = Path(outdir)
            outpath.mkdir(parents=True, exist_ok=True)
            fig.savefig(outpath / filename, format="pdf", bbox_inches="tight")
        return fig, axes
    except Exception:
        return None


def plot_round_coverage(x_sim, observed_counts, bin_edges_1d, selected_mask=None,
                        round_index=None, quantile=0.68, outdir=None,
                        filename=None, title=None, save=True):
    """Per-round predictive coverage of the simulated spectra over the observation.

    When ``selected_mask`` is given (importance-sampling filtering on), the
    IS-selected and rejected pools are drawn as separate bands so the shrinking
    coverage across rounds is visible.

    Args:
        x_sim: ``(n_pool, n_bins)`` simulated count spectra for the round.
        observed_counts: ``(n_bins,)`` observed counts.
        bin_edges_1d: ``(n_bins + 1,)`` energy bin edges (keV).
        selected_mask: ``(n_pool,)`` bool, ``True`` = IS-selected; ``None`` ->
            a single band.
        round_index: Round number, used in the default filename and title.
        quantile: Central predictive interval width (0.68 -> 16th..84th band).
        outdir: Directory to save into (``None`` -> do not save).
        filename: Output filename (defaults to ``round_<r>_coverage.pdf``).
        title: Optional plot title.
        save: Whether to write the figure.

    Returns:
        ``(fig, ax)`` or ``None`` on failure.
    """
    try:
        import matplotlib.pyplot as plt

        x_sim = _to_numpy(x_sim).astype(float)
        x_obs = _to_numpy(observed_counts).astype(float).reshape(-1)
        edges = np.asarray(bin_edges_1d, dtype=float)
        e_cen = np.sqrt(edges[:-1] * edges[1:])

        plo = 100.0 * (1.0 - quantile) / 2.0
        phi = 100.0 * (1.0 + quantile) / 2.0
        pct = int(round(quantile * 100))

        fig, ax = plt.subplots(figsize=(8, 5))

        def _band(data, color, label, alpha):
            if data.shape[0] == 0:
                return
            lo = np.percentile(data, plo, axis=0)
            hi = np.percentile(data, phi, axis=0)
            med = np.percentile(data, 50.0, axis=0)
            ax.fill_between(e_cen, lo, hi, step="mid", color=color, alpha=alpha,
                            linewidth=0, label=label)
            ax.step(e_cen, med, where="mid", color=color, lw=1.0, alpha=0.9)

        if selected_mask is None:
            _band(x_sim, "steelblue", f"Simulated {pct}%", 0.40)
        else:
            m = np.asarray(selected_mask, dtype=bool)
            _band(x_sim[~m], "0.6", f"Rejected {pct}%", 0.30)
            _band(x_sim[m], "seagreen", f"IS-selected {pct}%", 0.45)

        ax.step(e_cen, x_obs, where="mid", color="black", lw=1.2, label="Observed")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(e_cen.min(), e_cen.max())
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Counts")
        ax.set_title(title or (f"Round {round_index} coverage"
                               if round_index is not None else "Coverage"))
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()

        if save and outdir is not None:
            if filename is None:
                filename = (f"round_{round_index}_coverage.pdf"
                            if round_index is not None else "coverage.pdf")
            outpath = Path(outdir)
            outpath.mkdir(parents=True, exist_ok=True)
            fig.savefig(outpath / filename, format="pdf", bbox_inches="tight")
        return fig, ax
    except Exception:
        return None


def plot_ensemble_posteriors(posteriors, ensemble, observation, prior,
                             parameter_names, n_samples=2000, outdir=None,
                             round_index=None, filename=None, save=True):
    """Overlay each NDE posterior and the ensemble posterior for one round.

    Samples are drawn in the unit cube (conditioned on the observed spectrum) and
    mapped to physical space via ``prior.from_unit_cube`` before plotting with
    ``chainconsumer``.

    Args:
        posteriors: List of per-NDE posteriors (sbi ``DirectPosterior``).
        ensemble: The combined ``EnsemblePosterior``.
        observation: Observed spectrum used to condition the posteriors.
        prior: Object exposing ``from_unit_cube`` (the solver prior).
        parameter_names: Column names for the corner plot.
        n_samples: Samples drawn from each posterior.
        outdir: Directory to save into (``None`` -> do not save).
        round_index: Round number, used in the default filename.
        filename: Output filename (defaults to ``round_<r>_ensemble_posteriors.pdf``).
        save: Whether to write the figure.

    Returns:
        The figure, or ``None`` on failure (e.g. chainconsumer missing).
    """
    try:
        import pandas as pd
        import torch
        from chainconsumer import Chain, ChainConsumer

        obs_t = torch.as_tensor(np.asarray(observation), dtype=torch.float32)
        names = list(parameter_names)
        cc = ChainConsumer()

        def _add(sampler, name):
            samples = sampler.sample((n_samples,), x=obs_t)
            samples = _to_numpy(samples)
            physical = np.asarray(prior.from_unit_cube(samples))
            cc.add_chain(Chain(samples=pd.DataFrame(physical, columns=names),
                               name=name))

        for k, post in enumerate(posteriors):
            _add(post, f"NDE {k + 1}")
        _add(ensemble, "Ensemble")

        fig = cc.plotter.plot()

        if save and outdir is not None:
            if filename is None:
                filename = (f"round_{round_index}_ensemble_posteriors.pdf"
                            if round_index is not None
                            else "ensemble_posteriors.pdf")
            outpath = Path(outdir)
            outpath.mkdir(parents=True, exist_ok=True)
            fig.savefig(outpath / filename, format="pdf", bbox_inches="tight")
        return fig
    except Exception:
        return None
