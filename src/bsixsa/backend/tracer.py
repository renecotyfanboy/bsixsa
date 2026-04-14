"""Unified evaluation tracer for all backends."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import contextlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


@contextlib.contextmanager
def _agg_backend():
    """Temporarily switch to the non-interactive Agg backend, then restore."""
    prev = matplotlib.get_backend()
    matplotlib.use("Agg")
    try:
        yield
    finally:
        matplotlib.use(prev)

# TODO : set individual backend output to on or off along with backend

class EvaluationTracer:
    """Records ``(parameters, cstat)`` for every likelihood evaluation.

    All backends call :meth:`record` after each evaluation (or batch of
    evaluations). A progress PNG is written to *output_dir* at adaptive
    evaluation thresholds starting at *plot_every* and then advancing with
    a configurable log-scaled cadence. The trace is flushed to disk
    incrementally so that memory usage stays bounded even for
    billion-evaluation runs.

    Parameters:
        output_dir:
            Directory where ``progress.png`` and ``evaluation_trace.npz``
            are saved.
        parameter_names:
            Names of the fitted parameters (used for axis labels).
        plot_every:
            Write the first progress PNG after this many evaluations.
            Later plots are scheduled adaptively based on
            ``plot_step_percent``. Set to ``0`` to disable periodic
            plotting.
        plot_step_percent:
            Relative cadence for later plots. ``10`` reproduces the current
            10%-per-decade schedule (e.g. 50, 60, ..., 100, 200, ...).
    """

    def __init__(
        self,
        output_dir: str | Path,
        parameter_names: list[str],
        plot_every: int = 20_000,
        plot_step_percent: int = 10,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.parameter_names = parameter_names
        self.plot_every = plot_every
        self.plot_step_percent = plot_step_percent
        self._flush_record_every = 200

        # In-memory buffer (flushed periodically)
        self._params_buf: list[np.ndarray] = []
        self._cstats_buf: list[np.ndarray] = []
        self._timestamps_buf: list[np.ndarray] = []

        self._n_record_calls: int = 0
        self._n_evals: int = 0
        self._t0: float = perf_counter()
        self._trace_path: Path = self.output_dir / "evaluation_trace.npz"
        self._next_plot_eval: int | None = (
            max(1, int(plot_every)) if plot_every > 0 else None
        )

        # Clear any leftover trace from a previous run
        if self._trace_path.exists():
            self._trace_path.unlink()

    def record(self, params: np.ndarray, cstat: np.ndarray) -> None:
        """Record one or more evaluations.

        Parameters:
            params:
                Parameter values in *physical* space.  Shape ``(ndim,)`` for a
                single evaluation or ``(batch, ndim)`` for a batch.
            cstat:
                Corresponding C-statistic value(s).  Scalar or ``(batch,)``.
        """
        params = np.atleast_2d(params)
        cstat = np.atleast_1d(np.asarray(cstat, dtype=np.float64)).ravel()
        batch_size = params.shape[0]
        t = perf_counter() - self._t0
        timestamps = np.full(batch_size, t)

        self._params_buf.append(params)
        self._cstats_buf.append(cstat)
        self._timestamps_buf.append(timestamps)
        self._n_evals += batch_size
        self._n_record_calls += 1

        if self._n_record_calls % self._flush_record_every == 0:
            self._flush_to_disk()

        if self._next_plot_eval is not None and self._n_evals >= self._next_plot_eval:
            self.plot_progress()
            self._next_plot_eval = self._compute_next_plot_eval(self._n_evals)

    def _compute_next_plot_eval(self, n_evals: int) -> int | None:
        """Return the next evaluation count that should trigger a plot."""
        if self.plot_every <= 0:
            return None

        n_evals = max(n_evals, self.plot_every)
        step_base = 10 ** int(np.floor(np.log10(max(n_evals, 1))))
        step = max(1, int(step_base * self.plot_step_percent / 10))
        return ((n_evals // step) + 1) * step

    def _flush_to_disk(self) -> None:
        """Append buffered data to the on-disk npz and free memory."""
        if not self._params_buf:
            return

        new_params = np.concatenate(self._params_buf, axis=0)
        new_cstats = np.concatenate(self._cstats_buf, axis=0)
        new_timestamps = np.concatenate(self._timestamps_buf, axis=0)

        if self._trace_path.exists():
            existing = np.load(self._trace_path)
            new_params = np.concatenate([existing["params"], new_params], axis=0)
            new_cstats = np.concatenate([existing["cstats"], new_cstats], axis=0)
            new_timestamps = np.concatenate([existing["timestamps"], new_timestamps], axis=0)

        np.savez_compressed(
            self._trace_path,
            params=new_params,
            cstats=new_cstats,
            timestamps=new_timestamps,
        )

        self._params_buf.clear()
        self._cstats_buf.clear()
        self._timestamps_buf.clear()

    @property
    def all_params(self) -> np.ndarray:
        """All recorded parameters, shape ``(total_evals, ndim)``."""
        parts = []
        if self._trace_path.exists():
            parts.append(np.load(self._trace_path)["params"])
        if self._params_buf:
            parts.append(np.concatenate(self._params_buf, axis=0))
        if not parts:
            return np.empty((0, 0))
        return np.concatenate(parts, axis=0)

    @property
    def all_cstats(self) -> np.ndarray:
        """All recorded C-statistics, shape ``(total_evals,)``."""
        parts = []
        if self._trace_path.exists():
            parts.append(np.load(self._trace_path)["cstats"])
        if self._cstats_buf:
            parts.append(np.concatenate(self._cstats_buf, axis=0))
        if not parts:
            return np.empty(0)
        return np.concatenate(parts, axis=0)

    @property
    def all_timestamps(self) -> np.ndarray:
        """All recorded timestamps (seconds since start), shape ``(total_evals,)``."""
        parts = []
        if self._trace_path.exists():
            parts.append(np.load(self._trace_path)["timestamps"])
        if self._timestamps_buf:
            parts.append(np.concatenate(self._timestamps_buf, axis=0))
        if not parts:
            return np.empty(0)
        return np.concatenate(parts, axis=0)

    @property
    def n_evals(self) -> int:
        """Total number of individual evaluations recorded."""
        return self._n_evals

    def save(self) -> Path:
        """Flush remaining buffer and return path to the ``.npz`` file."""
        self._flush_to_disk()
        return self._trace_path

    def plot_progress(self, path: str | Path | None = None) -> Path:
        """Write a progress PNG.

        Layout: one row per parameter with parameter value against
        likelihood rank on the x-axis.

        Parameters:
            path:
                File path.  Defaults to ``{output_dir}/progress.png``.
        """
        if path is None:
            path = self.output_dir / "progress.png"
        path = Path(path)

        cstats = self.all_cstats
        params = self.all_params
        # Drop entries with non-finite cstat or parameter values (nan / inf)
        finite_mask = np.isfinite(cstats) & np.all(np.isfinite(params), axis=1)
        cstats = cstats[finite_mask]
        params = params[finite_mask]

        n = len(cstats)
        if n == 0:
            return path

        ndim = params.shape[1]
        if ndim == 0:
            return path
        n_rows = ndim

        stat_for_color = np.clip(cstats, np.finfo(np.float64).tiny, None)
        order = np.argsort(cstats)[::-1]
        likelihood_rank = np.empty(n, dtype=np.int64)
        likelihood_rank[order] = np.arange(1, n + 1)
        vmin = float(np.percentile(stat_for_color, 1))
        vmax = float(np.percentile(stat_for_color, 99))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin <= 0 or vmin == vmax:
            vmin = float(np.min(stat_for_color))
            vmax = float(np.max(stat_for_color))
        norm = None if vmin == vmax else LogNorm(vmin=vmin, vmax=vmax)

        with _agg_backend():
            fig, axs = plt.subplots(
                n_rows, 1, figsize=(6, 2.5 * n_rows), squeeze=False, sharex=True, layout='compressed'
            )
            axs = axs.ravel()

            best_idx = int(np.argmin(cstats))
            best_params = params[best_idx]

            for i in range(ndim):
                ax = axs[i]
                name = self.parameter_names[i] if i < len(self.parameter_names) else f"p{i}"
                mappable = ax.scatter(
                    likelihood_rank,
                    params[:, i],
                    s=1,
                    alpha=0.3,
                    c=stat_for_color,
                    cmap="viridis_r",
                    norm=norm,
                    rasterized=True,
                )
                ax.axhline(best_params[i], color="black", lw=1.5, ls="--", label="best fit")
                ax.set_ylabel(name)
                ax.legend(loc="upper right", fontsize="small")

            axs[0].set_title(f"Evaluation progress — {n} evals")
            axs[-1].set_xlim(1, n)
            axs[-1].set_xlabel("likelihood rank")
            fig.colorbar(mappable, ax=axs, location="bottom", label="C-stat", pad=0.05, aspect=20)
            fig.savefig(path, dpi=100)
            plt.close(fig)

        return path


class DummyTracer(EvaluationTracer):
    """Dummy tracer that does nothing."""

    def record(self, *args, **kwargs):
        pass
