"""Record of every likelihood evaluation of a run, with a progress plot."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure


class EvaluationTracer:
    """Keeps every likelihood evaluation of a run and plots how the search progresses.

    While the run goes on, `progress.png` is redrawn and `evaluation_trace.npz`
    (arrays `params`, `cstats` and `timestamps` in seconds) is saved, so a
    crashed run keeps a partial trace. A trace from a previous run in
    `output_dir` is deleted. When disabled, only `n_evals` is counted.

    Parameters:
        output_dir: Directory of `progress.png` and `evaluation_trace.npz`.
        parameter_names: Names of the fitted parameters, used as axis labels.
        enabled: Whether to keep and plot the evaluations.
        plot_every: Evaluations before the first progress plot.
        plot_step_percent: Spacing of the later plots, in percent of the next power of ten.

    Attributes:
        n_evals: Number of evaluations so far.
    """

    def __init__(
        self,
        output_dir: str | Path,
        parameter_names: list[str],
        *,
        enabled: bool,
        plot_every: int,
        plot_step_percent: int,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.parameter_names = list(parameter_names)
        self.enabled = enabled
        self.plot_every = plot_every
        self.plot_step_percent = plot_step_percent
        self.n_evals = 0

        self._params: list[np.ndarray] = []
        self._cstats: list[np.ndarray] = []
        self._timestamps: list[np.ndarray] = []
        self._t0 = perf_counter()
        self._next_plot = plot_every
        self._trace_path = self.output_dir / "evaluation_trace.npz"
        self._trace_path.unlink(missing_ok=True)

    def record(self, params: np.ndarray, cstat: np.ndarray | float) -> None:
        """Record one evaluation or a batch of them.

        Parameters:
            params: Physical parameters, shape `(d,)` or `(n, d)`.
            cstat: The matching C-statistic value(s).
        """
        params = np.atleast_2d(params)
        self.n_evals += len(params)
        if not self.enabled:
            return
        self._params.append(params)
        self._cstats.append(np.atleast_1d(np.asarray(cstat, dtype=np.float64)).ravel())
        self._timestamps.append(np.full(len(params), perf_counter() - self._t0))

        if self.n_evals >= self._next_plot:
            self.plot_progress()
            self.save()
            self._next_plot = self._next_plot_after(self.n_evals)

    def _next_plot_after(self, n: int) -> int:
        """Next evaluation count at which to plot."""
        step = max(1, int(10 ** int(np.floor(np.log10(n))) * self.plot_step_percent / 10))
        return (n // step + 1) * step

    @property
    def all_params(self) -> np.ndarray:
        """All recorded parameters, shape `(n_recorded, d)`."""
        if not self._params:
            return np.empty((0, len(self.parameter_names)))
        return np.concatenate(self._params, axis=0)

    @property
    def all_cstats(self) -> np.ndarray:
        """All recorded C-statistics, shape `(n_recorded,)`."""
        return np.concatenate(self._cstats) if self._cstats else np.empty(0)

    @property
    def all_timestamps(self) -> np.ndarray:
        """Time of each recorded evaluation in seconds, shape `(n_recorded,)`."""
        return np.concatenate(self._timestamps) if self._timestamps else np.empty(0)

    def save(self) -> Path:
        """Write everything recorded so far to `evaluation_trace.npz` and return its path."""
        if self._params:
            np.savez_compressed(
                self._trace_path,
                params=self.all_params,
                cstats=self.all_cstats,
                timestamps=self.all_timestamps,
            )
        return self._trace_path

    def plot_progress(self, path: str | Path | None = None) -> Path:
        """Draw each parameter against the likelihood rank, coloured by C-stat, and return the figure path.

        Parameters:
            path: Output file; defaults to `output_dir / "progress.png"`.
        """
        path = self.output_dir / "progress.png" if path is None else Path(path)
        cstats, params = self.all_cstats, self.all_params
        finite = np.isfinite(cstats) & np.all(np.isfinite(params), axis=1)
        cstats, params = cstats[finite], params[finite]
        n = len(cstats)
        if n == 0:
            return path

        # Rank 1 is the highest C-stat, so the best evaluations sit on the right.
        rank = np.empty(n, dtype=np.int64)
        rank[np.argsort(cstats)[::-1]] = np.arange(1, n + 1)
        # A log colour scale needs positive values.
        color = np.clip(cstats, np.finfo(np.float64).tiny, None)
        vmin, vmax = np.percentile(color, [1, 99])
        if vmin == vmax:
            vmin, vmax = color.min(), color.max()
        norm = LogNorm(vmin=vmin, vmax=vmax) if vmin < vmax else None
        best = params[np.argmin(cstats)]

        ndim = params.shape[1]
        fig = Figure(figsize=(6, 2.5 * ndim), layout="compressed")
        axs = fig.subplots(ndim, 1, squeeze=False, sharex=True).ravel()
        for ax, name, values, best_value in zip(axs, self.parameter_names, params.T, best):
            mappable = ax.scatter(
                rank, values, s=1, alpha=0.3, c=color, cmap="viridis_r", norm=norm, rasterized=True
            )
            ax.axhline(best_value, color="black", lw=1.5, ls="--", label="best fit")
            ax.set_ylabel(name)
            ax.legend(loc="upper right", fontsize="small")
        axs[0].set_title(f"Evaluation progress — {n} evals")
        axs[-1].set_xlim(1, n)
        axs[-1].set_xlabel("likelihood rank")
        fig.colorbar(mappable, ax=axs, location="bottom", label="C-stat", pad=0.05, aspect=20)
        fig.savefig(path, dpi=100)
        return path
