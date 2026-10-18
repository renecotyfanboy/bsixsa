"""The final posterior: an importance sample from the best proposal, and its equal-weight resample."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .certify import Certificate, CstatFn, Proposal, _checked_log_weights, build_certificate

__all__ = ["importance_sample", "resample"]


def importance_sample(
    proposal: Proposal,
    cstat_fn: CstatFn,
    *,
    target_ess: float,
    max_draws: int,
    batch: int,
    rng: np.random.Generator,
    khat_threshold: float,
    min_efficiency: float,
) -> Certificate:
    """Draw from *proposal* in batches until the PSIS ESS reaches *target_ess* or *max_draws*."""
    max_draws, batch = int(max_draws), int(batch)
    draws: list[np.ndarray] = []
    log_qs: list[np.ndarray] = []
    cstats: list[np.ndarray] = []
    n_raw = 0
    while True:
        m = min(batch, max_draws - n_raw)
        u = np.asarray(proposal.sample_raw(m, rng), dtype=np.float64)
        draws.append(u)
        log_qs.append(np.asarray(proposal.log_prob_raw(u), dtype=np.float64).ravel())
        cstats.append(np.asarray(cstat_fn(u), dtype=np.float64).ravel())
        n_raw += m
        cert = build_certificate(
            proposal.name, np.concatenate(draws), np.concatenate(cstats), np.concatenate(log_qs),
            round_index=None, khat_threshold=khat_threshold, min_efficiency=min_efficiency, proposal=proposal,
        )
        if cert.ess >= float(target_ess) or n_raw >= max_draws:
            return cert


def resample(u: np.ndarray, log_w: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Return *n* equal-weight rows drawn with probability `exp(log_w)`, and the number of distinct rows.

    A low number of distinct rows means a few heavy draws dominate.
    """
    lw = _checked_log_weights(log_w).ravel()
    finite = np.isfinite(lw)
    if not finite.any():
        raise ValueError("cannot resample: no draw has a positive weight.")
    p = np.zeros(lw.size)
    p[finite] = np.exp(lw[finite] - logsumexp(lw[finite]))
    index = rng.choice(lw.size, size=int(n), replace=True, p=p / p.sum())
    return np.asarray(u)[index], int(np.unique(index).size)
