r"""Training draws of the next round: the prior restricted to the proposal's highest-density region.

The next training set is drawn approximately uniformly on

    A = [0, 1]^d ∩ {u : log q(u) > t} ∩ region,

where q is the round's proposal, t the ``quantile`` of its own log density
(so that A holds ``1 - quantile`` of its in-cube mass) and ``region`` the
optional history-matching region. Flat draws keep the plain NPE loss valid,
and the flows learn the posterior restricted to A (truncated sequential NPE,
Deistler et al. 2022).

The draws come from sampling importance resampling: ``oversampling * n``
proposal candidates, weight 1/q inside A and zero outside, resampled to ``n``
rows. sbi's ``RestrictedPrior(sample_with="sir")`` is not equivalent: it keeps
candidates outside A with a non-zero weight.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..._cube import in_cube
from .certify import Proposal

if TYPE_CHECKING:
    from ...restriction.region import Region

__all__ = ["density_threshold", "sample_truncated"]

_MIN_SUPPORT_IN_CUBE = 10


def density_threshold(proposal: Proposal, *, quantile: float, n_support: int, rng: np.random.Generator) -> float:
    """The *quantile* of the log density over ``n_support`` in-cube proposal draws.

    Raises:
        ValueError: If too few draws fall in the cube.
    """
    u = np.asarray(proposal.sample_raw(int(n_support), rng), dtype=np.float64)
    kept = u[in_cube(u)]
    if kept.shape[0] < _MIN_SUPPORT_IN_CUBE:
        raise ValueError(
            f"Only {kept.shape[0]} of {int(n_support):,} proposal draws fall in the unit cube, but "
            f"at least {_MIN_SUPPORT_IN_CUBE} are needed to place the density threshold; the proposal "
            "puts almost no mass inside the prior bounds."
        )
    log_q = np.asarray(proposal.log_prob_raw(kept), dtype=np.float64)
    if not np.all(np.isfinite(log_q)):
        raise ValueError("The proposal returned a non-finite log density on its own in-cube draws.")
    return float(np.quantile(log_q, quantile))


def sample_truncated(
    proposal: Proposal,
    *,
    n: int,
    quantile: float,
    n_support: int,
    oversampling: int,
    rng: np.random.Generator,
    region: Region | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return ``n`` approximately flat draws on A and the SIR diagnostics.

    Raises:
        RuntimeError: If no candidate falls in A.
    """
    n = int(n)
    threshold = density_threshold(proposal, quantile=quantile, n_support=n_support, rng=rng)
    n_candidates = int(oversampling) * n
    u = np.asarray(proposal.sample_raw(n_candidates, rng), dtype=np.float64)

    accepted = np.flatnonzero(in_cube(u))
    log_q = np.full(n_candidates, -np.inf)
    if accepted.size:
        log_q[accepted] = np.asarray(proposal.log_prob_raw(u[accepted]), dtype=np.float64)
        accepted = accepted[log_q[accepted] > threshold]
    if accepted.size and region is not None:
        accepted = accepted[np.asarray(region.accepts(u[accepted]), dtype=bool)]
    if accepted.size == 0:
        raise RuntimeError(
            f"None of {n_candidates:,} proposal draws lies inside the proposal's highest-density region"
            + (" and the history-matching region; the proposal and the region do not overlap."
               if region is not None else " within the prior bounds.")
        )

    log_w = -log_q[accepted]
    weights = np.exp(log_w - log_w.max())
    weights /= weights.sum()
    rows = accepted[rng.choice(accepted.size, size=n, replace=True, p=weights)]
    info = {
        "threshold": threshold,
        "n_candidates": n_candidates,
        "acceptance": accepted.size / n_candidates,
        "sir_ess": float(1.0 / np.sum(weights**2)),
        "n_unique": int(np.unique(rows).size),
    }
    return u[rows], info
