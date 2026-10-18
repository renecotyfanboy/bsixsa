"""Diagnostic figure of one history-matching wave."""

from __future__ import annotations

import os

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

__all__ = ["plot_wave"]

_KEPT = "#2a78d6"
_LOST = "#eb6834"
_WASTED = "#1baf7a"
_NEUTRAL = "#b8b7b0"
_INK = "#52514e"
_MAX_PROBE_POINTS = 20_000


def _finite_positive(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    return finite.size > 0 and bool(np.all(finite > 0))


def _history_series(history: list[dict], key: str) -> np.ndarray:
    return np.array(
        [np.nan if row.get(key) is None else float(row[key]) for row in history], dtype=np.float64
    )


def plot_wave(
    history: list[dict],
    u: np.ndarray,
    cstat: np.ndarray,
    pred_cstat: np.ndarray,
    probe: np.ndarray,
    pred_probe: np.ndarray,
    span: float,
    outdir: str,
    param_names: list[str] | None = None,
) -> str:
    """Write the four-panel diagnostic PDF of the last wave in *history*.

    Panels: predicted against exact C-stat of every fold (with precision and
    recall), the region on its two narrowest axes, the histogram of every
    exact C-stat, and the region width and safety margin per wave.

    Parameters:
        history: Per-wave history dicts up to the wave to plot.
        u: Folded unit-cube points, shape ``(N, d)``.
        cstat: Their exact C-stat, shape ``(N,)``.
        pred_cstat: Surrogate predictions on ``u``, shape ``(N,)``.
        probe: Uniform probe points, shape ``(M, d)``.
        pred_probe: Surrogate predictions on ``probe``, shape ``(M,)``.
        span: Posterior span in C-stat units.
        outdir: Output directory, created if needed.
        param_names: Axis labels; ``u0, u1, ...`` when omitted.

    Returns:
        Path of the written ``restriction_wave_{wave}.pdf``.
    """
    u = np.asarray(u, dtype=np.float64)
    cs = np.asarray(cstat, dtype=np.float64).reshape(-1)
    pr = np.asarray(pred_cstat, dtype=np.float64).reshape(-1)
    probe = np.asarray(probe, dtype=np.float64)[:_MAX_PROBE_POINTS]
    probe_pred = np.asarray(pred_probe, dtype=np.float64).reshape(-1)[:_MAX_PROBE_POINTS]
    last = history[-1]
    wave = int(last["wave"])
    thr = float(last["threshold"])
    best = float(last["best"])
    d = u.shape[1]
    names = [str(n) for n in param_names] if param_names else [f"u{j}" for j in range(d)]

    fig = Figure(figsize=(11.0, 8.5))
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(2, 2)
    ax_scatter = fig.add_subplot(grid[0, 0])
    ax_probe = fig.add_subplot(grid[0, 1])
    ax_hist = fig.add_subplot(grid[1, 0])
    inner = grid[1, 1].subgridspec(2, 1, hspace=0.08)
    ax_spans = fig.add_subplot(inner[0])
    ax_margin = fig.add_subplot(inner[1], sharex=ax_spans)

    # 1. predicted vs exact
    finite = np.isfinite(cs) & np.isfinite(pr)
    keep, good = pr <= thr, cs <= thr
    precision = float(np.mean(good[keep])) if keep.any() else float("nan")
    recall = float(np.mean(keep[good])) if good.any() else float("nan")
    groups = (
        (~keep & ~good, _NEUTRAL, 2.0, 0.35, "rejected, above"),
        (keep & ~good, _WASTED, 6.0, 0.8, "kept, above: wasted"),
        (keep & good, _KEPT, 6.0, 0.7, "kept, belongs"),
        (~keep & good, _LOST, 12.0, 0.95, "rejected, belongs: lost"),
    )
    for mask, colour, size, alpha, label in groups:
        sel = mask & finite
        ax_scatter.scatter(
            cs[sel], pr[sel], s=size, c=colour, alpha=alpha, linewidths=0, label=f"{label} ({int(sel.sum())})",
            rasterized=True,
        )
    both = np.concatenate([cs[finite], pr[finite]])
    if both.size:
        if _finite_positive(both):
            ax_scatter.set_xscale("log")
            ax_scatter.set_yscale("log")
        lim = [float(both.min()), float(both.max())]
        ax_scatter.plot(lim, lim, color=_INK, lw=0.8)
    ax_scatter.axhline(thr, color="k", ls="--", lw=0.9)
    ax_scatter.axvline(thr, color="k", ls=":", lw=0.9)
    ax_scatter.set_xlabel("exact C-stat")
    ax_scatter.set_ylabel("predicted C-stat")
    ax_scatter.set_title(
        f"precision {100 * precision:.1f}% (cost), recall {100 * recall:.1f}% (risk)", fontsize="medium"
    )
    ax_scatter.legend(frameon=False, fontsize="small", loc="upper left", markerscale=2.0)

    # 2. the region on the probe
    accepted = probe_pred <= thr
    if d >= 2 and accepted.sum() > 5:
        rel = (probe[accepted].max(axis=0) - probe[accepted].min(axis=0)) / np.maximum(
            probe.max(axis=0) - probe.min(axis=0), 1e-12
        )
        j1, j2 = sorted(int(j) for j in np.argsort(rel)[:2])
        ax_probe.scatter(probe[~accepted, j1], probe[~accepted, j2], s=1.5, c=_NEUTRAL, linewidths=0, rasterized=True)
        ax_probe.scatter(
            probe[accepted, j1], probe[accepted, j2], s=4.0, c=_KEPT, linewidths=0, alpha=0.85,
            label="accepted probe points", rasterized=True,
        )
        ax_probe.set_xlabel(names[j1])
        ax_probe.set_ylabel(names[j2])
        ax_probe.legend(frameon=False, fontsize="small", markerscale=3.0)
        ax_probe.set_title("the region on its two narrowest axes", fontsize="medium")
    elif d == 1 and probe.size:
        ax_probe.scatter(probe[:, 0], probe_pred, s=2.0, c=np.where(accepted, _KEPT, _NEUTRAL), linewidths=0,
                         rasterized=True)
        ax_probe.axhline(thr, color="k", ls="--", lw=0.9)
        if _finite_positive(probe_pred):
            ax_probe.set_yscale("log")
        ax_probe.set_xlabel(names[0])
        ax_probe.set_ylabel("predicted C-stat")
        ax_probe.set_title("the surrogate along the axis", fontsize="medium")
    else:
        ax_probe.text(0.5, 0.5, "fewer than 6 probe points accepted", ha="center", va="center",
                      transform=ax_probe.transAxes, color=_INK)
        ax_probe.set_axis_off()

    # 3. histogram of the exact C-stat
    values = cs[np.isfinite(cs)]
    if values.size:
        if _finite_positive(values) and values.max() > values.min():
            bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 45)
            ax_hist.set_xscale("log")
        else:
            bins = 45
        ax_hist.hist(values, bins=bins, histtype="step", lw=1.3, color=_KEPT)
    ax_hist.axvline(thr, color="k", ls="--", lw=0.9, label="threshold")
    ax_hist.set_xlabel("exact C-stat")
    ax_hist.set_ylabel("folds")
    ax_hist.legend(frameon=False, fontsize="small")
    ax_hist.set_title(f"every fold so far ({values.size:,} finite)", fontsize="medium")

    # 4. width and safety per wave
    waves = np.array([int(row["wave"]) for row in history])
    spans = _history_series(history, "spans")
    margin = _history_series(history, "margin_sigma")
    ax_spans.plot(waves, spans, "-o", ms=6, mfc="none", color=_KEPT, lw=2)
    if _finite_positive(spans):
        ax_spans.set_yscale("log")
    ax_spans.set_ylabel("width (spans)")
    ax_spans.set_title("region width against safety margin", fontsize="medium")
    ax_spans.tick_params(labelbottom=False)
    ax_margin.plot(waves, margin, "-s", ms=6, mfc="none", color=_LOST, lw=2)
    ax_margin.set_ylabel("margin (RMSE)")
    ax_margin.set_xlabel("wave")
    ax_margin.set_xticks(waves)

    for ax in (ax_scatter, ax_probe, ax_hist, ax_spans, ax_margin):
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"Restriction, wave {wave}: C-stat <= {thr:,.1f} = best + {(thr - best) / float(span):.1f} spans, "
        f"{100 * float(last.get('prior_mass') or 0.0):.4g}% of the prior"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"restriction_wave_{wave}.pdf")
    fig.savefig(path, format="pdf", bbox_inches="tight")
    return path
