"""Persistent replace-worst training set ``T`` for the SIXSA round loop.

The SIXSA specification refines a single persistent training set ``T`` of
``(theta_latent, x_poisson, cstat)`` rows. Each new sub-batch of exactly
simulated rows is folded in by a *nested-sampling move*: the union of the
current buffer and the new rows is truncated back to the held size, keeping the
best (lowest-cstat) rows. Because we keep the best of the union rather than
blindly overwriting the current worst rows, the buffer's cstat ceiling ``c*``
(``= max cstat in T``) is **monotone non-increasing** even if some new rows are
worse than rows already in ``T``.

``x`` is the single Poisson realization drawn once per theta at simulation time
(the noisy count spectrum the NDE conditions on); ``cstat`` is the deterministic
Cash statistic of the noiseless folded model against the real observation
(``logL = -0.5 * cstat``) and is what the nested move and the importance
sampling select on.
"""

from __future__ import annotations

import numpy as np


class TrainingBuffer:
    """Aligned ``(theta_latent, x_poisson, cstat)`` rows with a monotone ceiling.

    Args:
        theta: ``(N, d)`` latent parameters (unit-Gaussian space).
        x: ``(N, n_bins)`` the one Poisson realization per row (NDE input).
        cstat: ``(N,)`` deterministic Cash statistics (``logL = -0.5*cstat``).
        capacity: Held buffer size. ``None`` -> ``N`` at construction.
    """

    def __init__(self, theta, x, cstat, capacity: int | None = None):
        self.theta = np.asarray(theta, dtype=np.float32)
        self.x = np.asarray(x, dtype=np.float32)
        self.cstat = np.asarray(cstat, dtype=np.float64).reshape(-1)
        if not (len(self.theta) == len(self.x) == len(self.cstat)):
            raise ValueError(
                "theta, x and cstat must have the same number of rows "
                f"({len(self.theta)}, {len(self.x)}, {len(self.cstat)})."
            )
        if self.theta.ndim != 2 or self.x.ndim != 2:
            raise ValueError("theta and x must be 2D (n_rows, n_features).")
        self.capacity = int(capacity) if capacity is not None else len(self.theta)

    def __len__(self) -> int:
        return len(self.cstat)

    @property
    def ceiling(self) -> float:
        """Current cstat ceiling ``c*`` (worst kept row). Monotone non-increasing."""
        return float(self.cstat.max()) if len(self) else float("inf")

    @property
    def d(self) -> int:
        return self.theta.shape[1]

    def training_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(theta, x)`` for NPE_C ``append_simulations`` (copies)."""
        return self.theta.copy(), self.x.copy()

    def replace_worst(self, new_theta, new_x, new_cstat) -> int:
        """Merge new rows and keep the ``capacity`` best (lowest-cstat) of the union.

        Returns the number of NEW rows that survived into the kept set (the
        effective "replaced" count). Implemented as keep-best-of-union so the
        ceiling ``c*`` never rises, regardless of the new rows' quality.
        """
        new_theta = np.asarray(new_theta, dtype=np.float32)
        new_x = np.asarray(new_x, dtype=np.float32)
        new_cstat = np.asarray(new_cstat, dtype=np.float64).reshape(-1)
        if not (len(new_theta) == len(new_x) == len(new_cstat)):
            raise ValueError("new theta, x and cstat must share their row count.")
        if len(new_theta) == 0:
            return 0

        n_old = len(self.cstat)
        theta = np.concatenate([self.theta, new_theta], axis=0)
        x = np.concatenate([self.x, new_x], axis=0)
        cstat = np.concatenate([self.cstat, new_cstat], axis=0)

        keep = min(self.capacity, len(cstat))
        # Stable argsort so ties keep the older (already-vetted) rows first;
        # `kind="stable"` guarantees old-before-new ordering within equal cstat.
        order = np.argsort(cstat, kind="stable")[:keep]
        n_new_kept = int(np.count_nonzero(order >= n_old))

        self.theta = theta[order]
        self.x = x[order]
        self.cstat = cstat[order]
        return n_new_kept
