"""Unified evaluation tracer for all backends."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import contextlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cmasher as cmr
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

# TODO : skip n first
# TODO : set individual backend output to on or off along with backend

class EvaluationTracer:
    """Records ``(parameters, cstat)`` for every likelihood evaluation.

    All backends call :meth:`record` after each evaluation (or batch of
    evaluations).  Every *plot_every* calls a progress PNG is written to
    *output_dir*.  The trace is flushed to disk incrementally so that
    memory usage stays bounded even for billion-evaluation runs.

    Parameters:
        output_dir:
            Directory where ``progress.png`` and ``evaluation_trace.npz``
            are saved.
        parameter_names:
            Names of the fitted parameters (used for axis labels).
        plot_every:
            Write a new progress PNG after this many :meth:`record` calls.
            Set to ``0`` to disable periodic plotting.
        flush_every:
            Flush accumulated data to the ``.npz`` file after this many
            :meth:`record` calls and free memory.  Set to ``0`` to keep
            everything in memory (not recommended for large runs).
    """

    def __init__(
        self,
        output_dir: str | Path,
        parameter_names: list[str],
        plot_every: int = 20_000, # TODO : plot every log-iter
        flush_every: int = 200,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.parameter_names = parameter_names
        self.plot_every = plot_every
        self.flush_every = flush_every

        # In-memory buffer (flushed periodically)
        self._params_buf: list[np.ndarray] = []
        self._cstats_buf: list[np.ndarray] = []
        self._timestamps_buf: list[np.ndarray] = []

        self._n_record_calls: int = 0
        self._n_evals: int = 0
        self._t0: float = perf_counter()
        self._trace_path: Path = self.output_dir / "evaluation_trace.npz"

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

        if self.flush_every > 0 and self._n_record_calls % self.flush_every == 0:
            self._flush_to_disk()

        if self.plot_every > 0 and self._n_evals % self.plot_every == 0:
            self.plot_progress()

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

        Layout: one row for C-stat vs eval#, one row for C-stat vs
        timestamp, then one row per parameter (value vs timestamp).

        Parameters:
            path:
                File path.  Defaults to ``{output_dir}/progress.png``.
        """
        if path is None:
            path = self.output_dir / "progress.png"
        path = Path(path)

        cstats = self.all_cstats
        params = self.all_params
        timestamps = self.all_timestamps

        # Drop entries with non-finite cstat or parameter values (nan / inf)
        finite_mask = np.isfinite(cstats) & np.all(np.isfinite(params), axis=1)
        cstats = cstats[finite_mask]
        params = params[finite_mask]
        timestamps = timestamps[finite_mask]

        n = len(cstats)
        if n == 0:
            return path

        ndim = params.shape[1]
        n_rows = 1 + ndim  # cstat vs eval, then one per param
        evals = np.arange(1, n + 1)
        running_best = np.minimum.accumulate(cstats)

        with _agg_backend():
            fig, axs = plt.subplots(
                n_rows, 1, figsize=(6, 2.5 * n_rows), squeeze=False, sharex=True, layout='compressed'
            )
            axs = axs.ravel()

            ax = axs[0]
            ax.scatter(evals, cstats, s=1, alpha=0.3, color="C0", rasterized=True)
            ax.plot(evals, running_best, color="C3", lw=1.5, label="running best")
            ax.set_ylabel("C-stat")
            ax.set_yscale("log")
            ax.legend(loc="upper right", fontsize="small")
            ax.set_title(f"Evaluation progress — {n} evals")

            best_idx = int(np.argmin(cstats))
            best_params = params[best_idx]

            for i in range(ndim):
                ax = axs[1 + i]
                name = self.parameter_names[i] if i < len(self.parameter_names) else f"p{i}"
                norm = LogNorm(vmin=np.percentile(cstats, 1), vmax=np.percentile(cstats, 99))
                mappable = ax.scatter(evals, params[:, i], s=1, alpha=0.3, c=cstats, rasterized=True, cmap=cmr.ember_r, norm=norm)
                ax.axhline(best_params[i], color="black", lw=1.5, ls="--", label="best fit")
                ax.set_ylabel(name)
                ax.legend(loc="upper right", fontsize="small")

            ax.set_xlabel("Evaluation #")
            ax.set_xscale('log')
            fig.colorbar(mappable, ax=axs, location='bottom', label="C-stat", pad=0.05, aspect=20)
            fig.savefig(path, dpi=100)
            plt.close(fig)

        return path


class DummyTracer(EvaluationTracer):
    """Dummy tracer that does nothing."""

    def record(self, *args, **kwargs):
        pass