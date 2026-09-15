"""Training of one neural posterior estimator, runnable in a bare worker process.

This module is shipped to loky workers *by value* (see
[`train_members`][bsixsa.backend.sixsa.flows.train_members]), so a worker never
imports ``bsixsa`` and therefore never imports XSPEC. It must stay
self-contained: numpy, torch, sbi and the standard library only, and no import
from ``bsixsa``, not even a relative one.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import torch

__all__ = ["NullTracker", "train_member"]


class NullTracker:
    """A no-op implementation of the sbi 0.27 ``Tracker`` protocol.

    sbi builds a TensorBoard tracker when none is given, which creates
    ``./sbi-logs`` in the current directory of every process that trains.
    """

    @property
    def log_dir(self) -> str | None:
        """Directory of the logs: there is none."""
        return None

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Discard one metric."""

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Discard a dictionary of metrics."""

    def log_params(self, params: dict[str, Any]) -> None:
        """Discard the hyper-parameters."""

    def add_figure(self, name: str, figure: Any, step: int | None = None) -> None:
        """Discard a figure."""

    def flush(self) -> None:
        """Nothing to flush."""


def train_member(
    theta_u: np.ndarray,
    z: np.ndarray,
    nde: dict,
    training: dict,
    seed: int,
    n_threads: int = 1,
) -> tuple[Any, dict]:
    r"""Train one NPE member $q(u \mid z)$ on unit-cube parameters.

    The estimator is an sbi ``NPE_C`` trained with the maximum-likelihood loss
    under a ``BoxUniform(0, 1)`` prior. The torch global random state is forked
    and seeded with *seed*, so the result is a pure function of the inputs and
    of the thread count, and the caller's random state is left untouched.

    Parameters:
        theta_u:
            Training parameters in the unit cube, shape ``(n, d)``.
        z:
            Matching compressed data, shape ``(n, n_features)``.
        nde:
            Keyword arguments of ``sbi.neural_nets.posterior_nn``. ``None``
            values are dropped, and ``num_bins`` is dropped unless the model
            name contains ``"nsf"`` (it only configures splines).
        training:
            Keyword arguments of ``NPE_C.train``, passed through.
        seed:
            Seed of the torch global generator for the whole training
            (initialisation, validation split and batch order).
        n_threads:
            Intra-op torch threads during training; the previous value is
            restored on exit.

    Returns:
        ``(estimator, info)``. *estimator* is the trained sbi density estimator
        (a ``ZukoFlow`` for the ``zuko_*`` models) on the CPU in eval mode.
        *info* holds ``best_validation_loss`` (float or ``None``),
        ``epochs_trained`` (int or ``None``), ``seconds``,
        ``validation_curve`` and ``training_curve`` (lists of floats, lower is
        better) and ``warnings`` (the messages of the warnings raised during
        training, which a worker process would otherwise print to its own
        stderr).

    Raises:
        ValueError: If *theta_u* is not two-dimensional, *z* does not have one
            row per parameter vector, or *n_threads* is smaller than 1.
    """
    from sbi.inference import NPE_C
    from sbi.neural_nets import posterior_nn
    from sbi.utils import BoxUniform

    # Copies, never views: joblib hands large arrays to workers as read-only
    # memmaps, and torch warns on (and would share) non-writable buffers.
    theta = np.array(theta_u, dtype=np.float32, copy=True)
    if theta.ndim != 2:
        raise ValueError(f"theta_u must have shape (n, d), got {theta.shape}.")
    x = np.array(z, dtype=np.float32, copy=True)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] != theta.shape[0]:
        raise ValueError(
            f"z must have shape (n, n_features) with n = {theta.shape[0]}, got {x.shape}."
        )
    n_threads = int(n_threads)
    if n_threads < 1:
        raise ValueError(f"n_threads must be >= 1, got {n_threads}.")
    d = theta.shape[1]

    kw = {k: v for k, v in dict(nde).items() if v is not None}
    if "nsf" not in str(kw.get("model", "")).lower():
        kw.pop("num_bins", None)
    training_kw = dict(training)
    # Use the reference first-round MLE objective. Exact truncation is flat
    # on its region; reference SIR is only approximate. In either case the
    # flow is a proposal, checked later against the full exact target.
    training_kw.setdefault("force_first_round_loss", True)

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(n_threads)
    t0 = time.perf_counter()
    try:
        with torch.random.fork_rng(devices=[]), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.default_generator.manual_seed(int(seed))
            prior = BoxUniform(torch.zeros(d), torch.ones(d))
            net = posterior_nn(**kw)
            inference = NPE_C(
                prior=prior,
                density_estimator=net,
                show_progress_bars=False,
                tracker=NullTracker(),
            )
            estimator = inference.append_simulations(
                torch.from_numpy(theta), torch.from_numpy(x)
            ).train(**training_kw)
    finally:
        torch.set_num_threads(previous_threads)
    seconds = time.perf_counter() - t0

    summary = dict(getattr(inference, "summary", None) or {})
    best = list(summary.get("best_validation_loss") or [])
    epochs = list(summary.get("epochs_trained") or [])
    info = {
        "best_validation_loss": float(best[-1]) if best else None,
        "epochs_trained": int(epochs[-1]) if epochs else None,
        "seconds": float(seconds),
        "validation_curve": [float(v) for v in summary.get("validation_loss") or []],
        "training_curve": [float(v) for v in summary.get("training_loss") or []],
        "warnings": [str(w.message) for w in caught],
    }
    return estimator.cpu().eval(), info
