r"""Check a proposal against the exact likelihood by importance sampling.

Each draw $u$ of a proposal $q$ is folded exactly and weighted by

$$ w = \frac{e^{-C/2}}{q(u)} $$

(the prior density is 1 on the unit cube; draws outside it get zero weight but
still count in $N$). The evidence estimate
$\log Z = \operatorname{logsumexp}(\log w) - \log N$ is unbiased whenever the
proposal covers the posterior.

Pareto-smoothed importance sampling (PSIS, Vehtari et al. 2024) fits the tail
of the largest weights and reads its shape $\hat k$ ("k-hat"): below 0.5 the
weight variance is finite, up to 0.7 the estimates are reliable, and above 0.7
a few draws dominate. The efficiency is the effective sample size (ESS) per
exact fold, in percent. The evidence uses the raw weights; the ESS and the
efficiency use the smoothed ones.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from scipy.special import logsumexp

from ..._cube import in_cube

__all__ = [
    "Certificate",
    "CstatFn",
    "Proposal",
    "build_certificate",
    "certify",
    "ess_from_log_weights",
    "log_evidence",
    "psis",
    "rank",
]

# Below this k-hat the weight variance is finite and efficiencies compare.
KHAT_SAFE = 0.5
# k-hat differences within this band are noise at the gate's sample size.
KHAT_TOLERANCE = 0.05
# PSIS needs a handful of tail points to fit a generalised Pareto at all.
_MIN_PSIS_WEIGHTS = 5


class Proposal(Protocol):
    r"""A density on $\mathbb{R}^d$ used as an importance-sampling proposal.

    "Raw" means not renormalised to the unit cube: draws may fall outside it,
    where they get zero weight.

    Attributes:
        name: Label used in logs and certificates.
        d: Number of parameters.
    """

    name: str
    d: int

    def sample_raw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw `n` points, shape `(n, d)`."""
        ...

    def log_prob_raw(self, u: np.ndarray) -> np.ndarray:
        """Log density at the rows of `u`, shape `(n,)`."""
        ...


# Exact C-stat of (n, d) cube points, +inf outside the cube (never folded).
CstatFn = Callable[[np.ndarray], np.ndarray]


@dataclass(eq=False)
class Certificate:
    """Draws, importance weights and diagnostics of one proposal.

    `log_w` are the raw log weights, `log_w_smoothed` the normalised PSIS
    ones; `n_in` counts the draws inside the cube and
    `efficiency = 100 * ESS / n_in`.
    """

    name: str
    round_index: int | None
    u: np.ndarray = field(repr=False)
    cstat: np.ndarray = field(repr=False)
    log_w: np.ndarray = field(repr=False)
    log_w_smoothed: np.ndarray = field(repr=False)
    n_raw: int
    n_in: int
    k_hat: float
    ess: float
    efficiency: float
    log_z: float
    log_z_err: float
    passed: bool
    proposal: Proposal | None = field(default=None, repr=False)

    @property
    def khat_reliable(self) -> bool:
        """True when k-hat is finite and the ~3√N-weight Pareto tail had enough positive weights."""
        positive = int(np.isfinite(self.log_w).sum())
        return math.isfinite(self.k_hat) and positive > math.ceil(3.0 * math.sqrt(self.n_raw)) + 4

    def summary(self) -> dict[str, Any]:
        """Scalar diagnostics, without arrays or the proposal."""
        return {
            "name": self.name, "round_index": self.round_index, "n_raw": self.n_raw, "n_in": self.n_in,
            "k_hat": self.k_hat, "ess": self.ess, "efficiency": self.efficiency,
            "log_z": self.log_z, "log_z_err": self.log_z_err, "passed": self.passed,
        }


def _checked_log_weights(log_w: Any) -> np.ndarray:
    """`log_w` as float64, rejecting NaN and +inf."""
    lw = np.asarray(log_w, dtype=np.float64)
    if np.any(np.isnan(lw)) or np.any(np.isposinf(lw)):
        raise ValueError("log weights must not contain NaN or +inf.")
    return lw


def _normalised(lw: np.ndarray) -> np.ndarray:
    out = np.full_like(lw, -np.inf)
    finite = np.isfinite(lw)
    if finite.any():
        out[finite] = lw[finite] - logsumexp(lw[finite])
    return out


def psis(log_w: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalised PSIS log weights and k-hat.

    Without a valid tail fit, the raw normalised weights are returned with
    `k_hat = inf`, which never passes.
    """
    lw = _checked_log_weights(log_w)
    finite = np.isfinite(lw)
    if int(finite.sum()) < _MIN_PSIS_WEIGHTS:
        return _normalised(lw), math.inf

    from arviz.stats import psislw

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")  # ties at the tail cutoff make the GPD fit warn
        # psislw works in place; the caller still needs the raw weights.
        smoothed, k_hat = psislw(lw.copy(), reff=1.0, normalize=True)
    smoothed = np.asarray(smoothed, dtype=np.float64)
    k_hat = float(k_hat)
    if not math.isfinite(k_hat) or np.any(np.isnan(smoothed[finite])):
        return _normalised(lw), math.inf
    smoothed[~finite] = -np.inf
    return smoothed, k_hat


def ess_from_log_weights(log_w: np.ndarray) -> float:
    r"""Kish effective sample size $1 / \sum_i \bar w_i^2$ of the normalised weights."""
    lw = _checked_log_weights(log_w)
    finite = np.isfinite(lw)
    if not finite.any():
        return 0.0
    w = np.exp(lw[finite] - logsumexp(lw[finite]))
    return float(1.0 / np.sum(w * w))


def log_evidence(log_w: np.ndarray, n_raw: int) -> tuple[float, float]:
    """Raw log evidence `logsumexp(w) - log N` and its delta-method error `sqrt(1/ESS - 1/N)`."""
    lw = _checked_log_weights(log_w)
    n_raw = int(n_raw)
    if n_raw < max(1, lw.size):
        raise ValueError(f"n_raw must be >= max(1, len(log_w)) = {max(1, lw.size)}, got {n_raw}.")
    if not np.isfinite(lw).any():
        return -math.inf, math.inf
    log_z = float(logsumexp(lw[np.isfinite(lw)]) - math.log(n_raw))
    return log_z, math.sqrt(max(1.0 / ess_from_log_weights(lw) - 1.0 / n_raw, 0.0))


def build_certificate(
    name: str,
    u: np.ndarray,
    cstat: np.ndarray,
    log_q: np.ndarray,
    *,
    round_index: int | None,
    khat_threshold: float,
    min_efficiency: float,
    proposal: Proposal | None = None,
) -> Certificate:
    """Weight draws against their exact C-stat and compute the diagnostics.

    Raises:
        ValueError: On inconsistent shapes, or a NaN/`-inf` C-stat or
            non-finite `log_q` inside the cube.
    """
    u = np.asarray(u, dtype=np.float64)
    n_raw = int(u.shape[0])
    if u.ndim != 2 or n_raw < 1:
        raise ValueError(f"u must have shape (n_raw, d) with n_raw >= 1, got {u.shape}.")
    cstat = np.array(cstat, dtype=np.float64, copy=True).ravel()
    log_q = np.asarray(log_q, dtype=np.float64).ravel()
    if cstat.shape != (n_raw,) or log_q.shape != (n_raw,):
        raise ValueError(f"cstat and log_q must have shape ({n_raw},), got {cstat.shape} and {log_q.shape}.")

    inside = in_cube(u)
    cstat[~inside] = np.inf
    invalid = np.isnan(cstat) | np.isneginf(cstat)
    if invalid.any():
        raise ValueError(
            f"certificate {name!r}: the C-stat is NaN or -inf for {int(invalid.sum())} parameter vector(s) "
            "inside the prior; check the model at these parameters."
        )
    zero = np.isposinf(cstat)
    if not np.all(np.isfinite(log_q[~zero])):
        raise ValueError(f"certificate {name!r}: log_q is not finite on a row with a finite statistic.")
    with np.errstate(invalid="ignore", over="ignore"):
        log_w = np.where(zero, -np.inf, -0.5 * cstat - log_q)

    smoothed, k_hat = psis(log_w)
    ess = ess_from_log_weights(smoothed)
    n_in = int(inside.sum())
    efficiency = 100.0 * ess / max(n_in, 1)
    log_z, log_z_err = log_evidence(log_w, n_raw)
    cert = Certificate(
        name=str(name), round_index=round_index, u=u, cstat=cstat, log_w=log_w,
        log_w_smoothed=smoothed, n_raw=n_raw, n_in=n_in, k_hat=k_hat, ess=ess, efficiency=efficiency,
        log_z=log_z, log_z_err=log_z_err, passed=False, proposal=proposal,
    )
    cert.passed = bool(cert.khat_reliable and k_hat < float(khat_threshold) and efficiency >= float(min_efficiency))
    return cert


def certify(
    proposal: Proposal,
    n: int,
    cstat_fn: CstatFn,
    *,
    rng: np.random.Generator,
    round_index: int | None,
    khat_threshold: float,
    min_efficiency: float,
) -> Certificate:
    """Certify a proposal on `n` of its own draws."""
    u = np.asarray(proposal.sample_raw(int(n), rng), dtype=np.float64)
    return build_certificate(
        proposal.name, u, cstat_fn(u), proposal.log_prob_raw(u), proposal=proposal, round_index=round_index,
        khat_threshold=khat_threshold, min_efficiency=min_efficiency,
    )


def _tier(cert: Certificate) -> int:
    """0: passed; 1: reliable k-hat below `KHAT_SAFE`; 2: reliable k-hat above; 3: unreliable."""
    if cert.passed:
        return 0
    if not cert.khat_reliable:
        return 3
    return 1 if cert.k_hat < KHAT_SAFE else 2


def rank(certs: Sequence[Certificate]) -> int:
    """Index of the best certificate: k-hat gates, efficiency ranks.

    Passing certificates come first, then those with a finite weight variance
    (k-hat below `KHAT_SAFE`), compared on efficiency; otherwise the
    efficiency decides among the k-hats within `KHAT_TOLERANCE` of the lowest.
    """
    certs = list(certs)
    tiers = [_tier(c) for c in certs]
    best_tier = min(tiers)
    pool = [i for i, t in enumerate(tiers) if t == best_tier]
    if best_tier == 3:
        return pool[0]
    if best_tier == 2:
        k_min = min(certs[i].k_hat for i in pool)
        pool = [i for i in pool if certs[i].k_hat <= k_min + KHAT_TOLERANCE + 1e-12]  # float rounding
    return max(pool, key=lambda i: certs[i].efficiency)
