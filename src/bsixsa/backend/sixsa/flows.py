r"""Neural posterior estimators and the proposals they give at the observation.

A neural posterior estimator (NPE) is a normalising flow trained to
approximate $p(u \mid z)$, the posterior of the cube parameters $u$ given the
code $z$ of a spectrum. Evaluated at the observed code, it becomes a
[`Proposal`][bsixsa.backend.sixsa.certify.Proposal].
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import logsumexp

from .certify import Proposal

__all__ = ["FlowProposal", "MixtureProposal", "train_flow"]

# Training settings of every flow; the configurable ones are in `FlowConfig`.
_LEARNING_RATE = 1e-3
_VALIDATION_FRACTION = 0.2
_CLIP_MAX_NORM = 1.0


class FlowProposal:
    """A trained conditional estimator evaluated at the observation's code."""

    def __init__(self, estimator: Any, z_obs: np.ndarray, *, name: str, batch: int = 50_000) -> None:
        self.estimator = estimator
        self.z_obs = np.array(z_obs, dtype=np.float64, copy=True).reshape(-1)
        self.name = str(name)
        self.batch = int(batch)
        self._condition = torch.as_tensor(self.z_obs, dtype=torch.float32).reshape(1, -1)
        self.d = int(np.prod(tuple(estimator.input_shape)))

    def sample_raw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``(n, d)`` points, seeded from *rng*."""
        n = int(n)
        # sbi samples with torch's global generator.
        torch.manual_seed(int(rng.integers(2**31 - 1)))
        parts = []
        with torch.no_grad():
            for start in range(0, n, self.batch):
                m = min(self.batch, n - start)
                draws = self.estimator.sample((m,), condition=self._condition)
                parts.append(draws.reshape(m, -1).to(torch.float64).numpy())
        return np.concatenate(parts, axis=0) if parts else np.empty((0, self.d), dtype=np.float64)

    def log_prob_raw(self, u: np.ndarray) -> np.ndarray:
        """Log density at *u*; ``-inf`` on rows with non-finite entries."""
        with np.errstate(over="ignore", invalid="ignore"):
            points = np.asarray(u, dtype=np.float64).reshape(-1, self.d).astype(np.float32)
        out = np.full(points.shape[0], -np.inf, dtype=np.float64)
        rows = np.flatnonzero(np.isfinite(points).all(axis=1))
        with torch.no_grad():
            for start in range(0, rows.size, self.batch):
                idx = rows[start : start + self.batch]
                # sbi expects inputs shaped (sample, batch, d).
                inputs = torch.from_numpy(points[idx]).reshape(-1, 1, self.d)
                log_q = self.estimator.log_prob(inputs, condition=self._condition)
                out[idx] = log_q.reshape(-1).to(torch.float64).numpy()
        return out


class MixtureProposal:
    """Equal-weight mixture of proposals."""

    def __init__(self, members: Sequence[Proposal], name: str = "mixture") -> None:
        self.members = list(members)
        self.name = str(name)
        self.d = int(self.members[0].d)

    def sample_raw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw *n* points, a multinomial share from each member."""
        n = int(n)
        counts = rng.multinomial(n, np.full(len(self.members), 1.0 / len(self.members)))
        parts = [m.sample_raw(int(c), rng) for m, c in zip(self.members, counts) if c > 0]
        u = np.concatenate(parts, axis=0) if parts else np.empty((0, self.d), dtype=np.float64)
        # Shuffled so that any prefix of the draws is itself a sample of the mixture.
        return u[rng.permutation(n)]

    def log_prob_raw(self, u: np.ndarray) -> np.ndarray:
        r"""Mixture log density, $\operatorname{logsumexp}_k \log q_k(u) - \log K$."""
        points = np.asarray(u, dtype=np.float64).reshape(-1, self.d)
        log_q = np.stack([np.asarray(m.log_prob_raw(points), dtype=np.float64).reshape(-1) for m in self.members])
        return logsumexp(log_q, axis=0) - np.log(len(self.members))


def train_flow(
    u: np.ndarray,
    z: np.ndarray,
    *,
    hidden_features: int,
    num_transforms: int,
    training_batch_size: int,
    max_num_epochs: int,
    stop_after_epochs: int,
    seed: int,
    log_dir: str | Path,
) -> tuple[Any, dict]:
    """Train one masked autoregressive flow on ``(u, z)`` pairs; return the estimator and its training record.

    Training curves are written to *log_dir* for TensorBoard.
    """
    from sbi.inference import NPE_C
    from sbi.inference.trainers.base import TensorBoardTracker
    from sbi.neural_nets import posterior_nn
    from sbi.utils import BoxUniform
    from torch.utils.tensorboard import SummaryWriter

    theta = torch.as_tensor(np.asarray(u, dtype=np.float32))
    x = torch.as_tensor(np.asarray(z, dtype=np.float32)).reshape(theta.shape[0], -1)
    d = theta.shape[1]

    torch.manual_seed(int(seed))
    writer = SummaryWriter(log_dir=str(log_dir))
    try:
        estimator_builder = posterior_nn("zuko_maf", hidden_features=hidden_features, num_transforms=num_transforms)
        inference = NPE_C(
            prior=BoxUniform(torch.zeros(d), torch.ones(d)), density_estimator=estimator_builder,
            show_progress_bars=False, tracker=TensorBoardTracker(writer),
        )
        # Training draws are flat on their region, so the plain maximum-likelihood loss applies.
        estimator = inference.append_simulations(theta, x).train(
            force_first_round_loss=True, training_batch_size=training_batch_size, max_num_epochs=max_num_epochs,
            stop_after_epochs=stop_after_epochs, learning_rate=_LEARNING_RATE,
            validation_fraction=_VALIDATION_FRACTION, clip_max_norm=_CLIP_MAX_NORM,
        )
    finally:
        writer.close()
    summary = inference.summary
    info = {
        "seed": int(seed),
        "log_dir": str(log_dir),
        "epochs_trained": int(summary["epochs_trained"][-1]),
        "best_validation_loss": float(summary["best_validation_loss"][-1]),
    }
    return estimator.eval(), info
