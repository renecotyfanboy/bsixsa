r"""Importance-sampling certificates of SIXSA proposals against the exact C-stat.

Every number SIXSA publishes goes through a certificate: a proposal $q$ only
proposes, and the exact fold decides. For $N_\mathrm{raw}$ raw draws
$u_i \sim q$ (not rejected to the cube), the log weights are

$$
\log w_i = -\tfrac12 C(u_i) - \log q_\mathrm{raw}(u_i),
\qquad \log w_i = -\infty \text{ when } u_i \notin [0, 1]^d \text{ or } C(u_i) = +\infty,
$$

with the uniform cube prior $\pi_u = 1$. Because $q_\mathrm{raw}$ is the
estimator's un-truncated density and the draws that leak out of the cube stay
in the count, the evidence estimate

$$
\log \hat Z = \operatorname{logsumexp}_i \log w_i - \log N_\mathrm{raw}
$$

is unbiased in $Z$ whatever the leakage: there is no normaliser of $q$ to
measure. Dividing by the number of in-cube rows instead would estimate
$Z / P_q(\text{cube})$.

A mixture candidate (`MixtureProposal`) is certified on the pooled draws of its
members, with the stratified density
$q_\mathrm{mix} = \sum_k (n_k / N)\, q_k$; with deterministic counts $n_k$
this is again unbiased and costs no extra fold.

Pareto smoothing (arviz `psislw`, `reff=1`) runs on the full $N_\mathrm{raw}$
array, zeros included, and gives $\hat k$ and the ESS used for efficiency,
gating and ranking; the raw ESS only enters the delta-method error
$\sigma_{\log Z} = \sqrt{1/\mathrm{ESS}_\mathrm{raw} - 1/N_\mathrm{raw}}$.

This module imports numpy, scipy and arviz only (no torch, no XSPEC).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import numpy as np
from scipy.special import logsumexp

from ..._cube import in_cube
from ...exceptions import FoldError

__all__ = [
    "Certificate",
    "MixtureLike",
    "ProposalLike",
    "SimulatorLike",
    "StratifiedLike",
    "build_certificate",
    "certify_fresh",
    "certify_many",
    "draw_raw",
    "ess_from_log_weights",
    "is_alive",
    "log_evidence",
    "psis",
    "rank_candidates",
    "tier",
]

# PSIS needs a handful of tail points to fit a generalised Pareto at all;
# arviz itself gives up below five.
_MIN_PSIS_WEIGHTS = 5
# Absorbs float noise in `k_min + tolerance` so a candidate exactly at the edge
# of the tolerance band is not dropped by rounding.
_TOLERANCE_SLACK = 1e-12

_KINDS = ("screen", "fresh", "final")


class ProposalLike(Protocol):
    """Raw proposal protocol: draws are not rejected to the cube."""

    name: str

    def sample_raw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw `(n, d)` float64 points, possibly outside the cube."""
        ...

    def log_prob_raw(self, u: np.ndarray) -> np.ndarray:
        """Un-truncated log density `(n,)`, defined outside the cube too."""
        ...


class MixtureLike(ProposalLike, Protocol):
    """Mixture of raw proposals that can be evaluated on stratified draws."""

    members: list

    def log_prob_stratified(self, u: np.ndarray, counts: Sequence[int]) -> np.ndarray:
        """`logsumexp_k(log q_k(u) + log(counts[k] / sum(counts)))`, shape `(n,)`."""
        ...


class StratifiedLike(ProposalLike, Protocol):
    """Raw proposal that draws in deterministic strata (e.g. `flows.MixtureProposal`)."""

    def allocation(self, n: int) -> list[int]:
        """Deterministic number of draws per stratum, summing to *n*."""
        ...

    def sample_stratified(self, counts: Sequence[int], rng: np.random.Generator) -> np.ndarray:
        """`counts[k]` raw draws of stratum `k`, shape `(sum(counts), d)`."""
        ...

    def log_prob_stratified(self, u: np.ndarray, counts: Sequence[int]) -> np.ndarray:
        """Density of the design that drew `counts[k]` points per stratum, shape `(n,)`."""
        ...


class SimulatorLike(Protocol):
    """The part of the cube simulator a certificate needs."""

    def cstat(self, u: np.ndarray, *, phase: str) -> np.ndarray:
        """Exact C-stat `(n,)`, `+inf` outside the closed cube."""
        ...


@dataclass(eq=False)
class Certificate:
    r"""Exact importance-sampling diagnostics of one proposal on one set of raw draws.

    Attributes:
        name: Label of the certified proposal.
        kind: `"screen"`, `"fresh"` or `"final"`.
        round_index: SIXSA round, `None` outside the loop.
        u: Raw draws, shape `(n_raw, d)`.
        cstat: Exact statistic, shape `(n_raw,)`, `+inf` outside the cube.
        log_q: Raw log density of the proposal at *u*, shape `(n_raw,)`.
        log_w: $-C/2 - \log q$, `-inf` where *cstat* is `+inf`.
        log_w_smoothed: PSIS-smoothed normalised log weights (`-inf` kept).
        n_raw: Number of raw draws, the denominator of the evidence.
        n_in: Rows inside the closed cube, i.e. the rows that were folded.
        k_hat: Pareto shape estimate (`inf` when it could not be fitted).
        ess: ESS of the smoothed weights.
        ess_raw: ESS of the raw weights.
        efficiency: `100 * ess / max(n_in, 1)`, in percent.
        log_z: `logsumexp(log_w) - log(n_raw)`.
        log_z_err: `sqrt(max(1/ess_raw - 1/n_raw, 0))`.
        alive: Whether *k_hat* is interpretable (see `is_alive`), counted on the finite weights.
        passed: `alive and k_hat < khat_threshold and efficiency >= min_efficiency`.
        tier: Selection tier (see `tier`).
        proposal: The certified proposal object. It is not serialised: pickling
            or copying a certificate drops it, so no estimator travels with it.
        extra: Free-form metadata.
    """

    name: str
    kind: str
    round_index: int | None
    u: np.ndarray = field(repr=False)
    cstat: np.ndarray = field(repr=False)
    log_q: np.ndarray = field(repr=False)
    log_w: np.ndarray = field(repr=False)
    log_w_smoothed: np.ndarray = field(repr=False)
    n_raw: int
    n_in: int
    k_hat: float
    ess: float
    ess_raw: float
    efficiency: float
    log_z: float
    log_z_err: float
    alive: bool
    passed: bool
    tier: int
    proposal: object = field(default=None, repr=False)
    extra: dict = field(default_factory=dict, repr=False)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["proposal"] = None
        return state

    def summary(self) -> dict[str, Any]:
        """Scalar diagnostics, without arrays or the proposal.

        Returns:
            A dict with `name`, `kind`, `round_index`, `n_raw`, `n_in`, `k_hat`,
            `ess`, `ess_raw`, `efficiency`, `log_z`, `log_z_err`, `alive`,
            `passed` and `tier` as plain Python values.
        """
        return {
            "name": self.name,
            "kind": self.kind,
            "round_index": self.round_index,
            "n_raw": int(self.n_raw),
            "n_in": int(self.n_in),
            "k_hat": float(self.k_hat),
            "ess": float(self.ess),
            "ess_raw": float(self.ess_raw),
            "efficiency": float(self.efficiency),
            "log_z": float(self.log_z),
            "log_z_err": float(self.log_z_err),
            "alive": bool(self.alive),
            "passed": bool(self.passed),
            "tier": int(self.tier),
        }

    def without_proposal(self) -> Certificate:
        """Return a shallow copy whose `proposal` is `None`.

        Returns:
            A new certificate sharing the arrays of this one.
        """
        return replace(self, proposal=None)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def _as_log_weights(log_w: Any) -> np.ndarray:
    """Validate log weights: a 1-D float64 copy with no `NaN` and no `+inf`.

    Raises:
        ValueError: If *log_w* is not 1-D or holds `NaN` or `+inf`.
    """
    lw = np.array(log_w, dtype=np.float64, copy=True)
    if lw.ndim != 1:
        raise ValueError(f"log weights must be 1-D, got shape {lw.shape}.")
    if np.any(np.isnan(lw)) or np.any(np.isposinf(lw)):
        raise ValueError("log weights must not contain NaN or +inf.")
    return lw


def _normalised(lw: np.ndarray) -> np.ndarray:
    """Normalise finite log weights to sum to one; `-inf` stays `-inf`."""
    out = np.full_like(lw, -np.inf)
    finite = np.isfinite(lw)
    if finite.any():
        out[finite] = lw[finite] - logsumexp(lw[finite])
    return out


def psis(log_w: np.ndarray) -> tuple[np.ndarray, float]:
    r"""Pareto-smoothed importance sampling on the full array of raw log weights.

    Runs arviz `psislw` with `reff=1` and `normalize=True` on a float64 copy.
    Zero weights (`-inf`) stay in the array, so the tail size is set by
    $N_\mathrm{raw}$ and not by the number of positive weights.

    Parameters:
        log_w: Raw log weights, shape `(n,)`; `-inf` entries are allowed.

    Returns:
        `(smoothed, k_hat)`: the smoothed log weights normalised to
        `logsumexp == 0` with every `-inf` preserved, and the Pareto shape.
        With fewer than five finite weights, or when the generalised Pareto
        fit fails, the normalised raw weights and `k_hat = inf` are returned
        (a `NaN` shape is reported as `inf`, so it can never pass a gate).

    Raises:
        ValueError: If *log_w* is not 1-D or contains `NaN` or `+inf`.
    """
    lw = _as_log_weights(log_w)
    finite = np.isfinite(lw)
    if int(finite.sum()) < _MIN_PSIS_WEIGHTS:
        return _normalised(lw), math.inf

    from arviz.stats import psislw

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        # Degenerate tails (ties at the cutoff) make the GPD fit divide by
        # zero; the outcome is a non-finite k, handled below.
        warnings.simplefilter("ignore")
        smoothed, k_hat = psislw(lw.copy(), reff=1.0, normalize=True)
    smoothed = np.asarray(smoothed, dtype=np.float64)
    k_hat = float(k_hat)
    if not math.isfinite(k_hat) or np.any(np.isnan(smoothed[finite])):
        return _normalised(lw), math.inf
    smoothed[~finite] = -np.inf
    return smoothed, k_hat


def ess_from_log_weights(log_w: np.ndarray) -> float:
    r"""Kish effective sample size $1 / \sum_i \bar w_i^2$ of normalised weights.

    Parameters:
        log_w: Log weights, shape `(n,)`, normalised or not; `-inf` entries
            count as zero weights.

    Returns:
        The ESS, `0.0` when no weight is finite.

    Raises:
        ValueError: If *log_w* is not 1-D or contains `NaN` or `+inf`.
    """
    lw = _as_log_weights(log_w)
    finite = np.isfinite(lw)
    if not finite.any():
        return 0.0
    w = np.exp(lw[finite] - logsumexp(lw[finite]))
    return float(1.0 / np.sum(w * w))


def log_evidence(log_w: np.ndarray, n_raw: int) -> tuple[float, float]:
    r"""Evidence estimate and its delta-method error from raw log weights.

    Parameters:
        log_w: Raw (not smoothed) log weights, shape `(m,)` with `m <= n_raw`;
            draws absent from the array count as zero weights.
        n_raw: Total number of raw draws, the denominator of the estimate.

    Returns:
        `(log_z, log_z_err)` with `log_z = logsumexp(log_w) - log(n_raw)` and
        `log_z_err = sqrt(max(1/ess_raw - 1/n_raw, 0))`; `(-inf, inf)` when no
        weight is finite.

    Raises:
        ValueError: If *n_raw* is smaller than one or than the array, or if
            *log_w* is invalid.
    """
    lw = _as_log_weights(log_w)
    n_raw = int(n_raw)
    if n_raw < 1 or n_raw < lw.size:
        raise ValueError(f"n_raw must be >= max(1, len(log_w)) = {max(1, lw.size)}, got {n_raw}.")
    finite = np.isfinite(lw)
    if not finite.any():
        return -math.inf, math.inf
    log_z = float(logsumexp(lw[finite]) - math.log(n_raw))
    ess_raw = ess_from_log_weights(lw)
    err = math.sqrt(max(1.0 / ess_raw - 1.0 / n_raw, 0.0))
    return log_z, err


def is_alive(k_hat: float, n_in: int, n_raw: int) -> bool:
    r"""Whether a Pareto shape estimate can be read at all.

    arviz fits the tail to the $\lceil 3\sqrt{N_\mathrm{raw}}\rceil$ largest
    of all $N_\mathrm{raw}$ weights. With that few positive weights or fewer,
    the "tail" is the whole body and $\hat k$ says nothing.

    Parameters:
        k_hat: Pareto shape.
        n_in: Number of usable rows.
        n_raw: Number of raw draws.

    Returns:
        `isfinite(k_hat) and n_in > ceil(3 * sqrt(n_raw)) + 4`.
    """
    return bool(math.isfinite(float(k_hat)) and int(n_in) > math.ceil(3.0 * math.sqrt(int(n_raw))) + 4)


# ---------------------------------------------------------------------------
# Certificates
# ---------------------------------------------------------------------------


def tier(cert: Certificate, *, khat_safe: float) -> int:
    """Selection tier of a certificate.

    Parameters:
        cert: The certificate.
        khat_safe: Pareto shape below which efficiencies are comparable.

    Returns:
        0 if it passed the gate; 1 if alive with `k_hat < khat_safe`; 2 if
        otherwise alive; 3 if not alive.
    """
    if cert.passed:
        return 0
    if not cert.alive:
        return 3
    return 1 if cert.k_hat < float(khat_safe) else 2


def build_certificate(
    name: str,
    u: np.ndarray,
    cstat: np.ndarray,
    log_q: np.ndarray,
    *,
    n_raw: int,
    kind: str,
    round_index: int | None,
    khat_threshold: float,
    khat_safe: float,
    min_efficiency: float,
    proposal: object = None,
    extra: dict | None = None,
) -> Certificate:
    r"""Weight raw draws against the exact statistic and fill every diagnostic.

    Rows outside the closed cube get zero weight even if *cstat* is finite
    there, so the certificate always targets the full cube prior. `n_in`
    counts the in-cube (folded) rows, while `alive` is decided on the finite
    weights, which also exclude folds that returned `+inf`.

    Parameters:
        name: Label of the proposal.
        u: Raw draws, shape `(n_raw, d)`.
        cstat: Exact statistic at *u*, shape `(n_raw,)`; `+inf` is a zero weight.
        log_q: Raw log density of the proposal at *u*, shape `(n_raw,)`.
        n_raw: Number of raw draws; must equal `len(u)`.
        kind: `"screen"`, `"fresh"` or `"final"`.
        round_index: SIXSA round, or `None`.
        khat_threshold: Gate on the Pareto shape (strict `<`).
        khat_safe: Tier-1 bound on the Pareto shape (strict `<`).
        min_efficiency: Gate on the efficiency in percent (`>=`).
        proposal: The certified proposal, kept by reference.
        extra: Metadata stored on the certificate.

    Returns:
        The `Certificate`.

    Raises:
        ValueError: On inconsistent shapes, an unknown *kind*, or a non-finite
            *log_q* on a row with a finite statistic (a proposal cannot have
            zero or infinite density at its own draw).
        FoldError: If *cstat* contains `NaN` or `-inf`.
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}.")
    u = np.asarray(u, dtype=np.float64)
    n_raw = int(n_raw)
    if u.ndim != 2 or u.shape[0] != n_raw or n_raw < 1:
        raise ValueError(f"u must have shape (n_raw={n_raw}, d) with n_raw >= 1, got {u.shape}.")
    cstat = np.array(cstat, dtype=np.float64, copy=True).ravel()
    log_q = np.asarray(log_q, dtype=np.float64).ravel()
    if cstat.shape != (n_raw,) or log_q.shape != (n_raw,):
        raise ValueError(f"cstat and log_q must have shape ({n_raw},), got {cstat.shape} and {log_q.shape}.")
    if np.any(np.isnan(cstat)) or np.any(np.isneginf(cstat)):
        raise FoldError(f"certificate {name!r}: the statistic contains NaN or -inf.")

    inside = in_cube(u)
    cstat[~inside] = np.inf
    zero = np.isposinf(cstat)
    if not np.all(np.isfinite(log_q[~zero])):
        raise ValueError(f"certificate {name!r}: log_q is not finite on a row with a finite statistic.")
    with np.errstate(invalid="ignore", over="ignore"):
        log_w = np.where(zero, -np.inf, -0.5 * cstat - log_q)

    smoothed, k_hat = psis(log_w)
    ess = ess_from_log_weights(smoothed)
    ess_raw = ess_from_log_weights(log_w)
    n_in = int(inside.sum())
    efficiency = 100.0 * ess / max(n_in, 1)
    log_z, log_z_err = log_evidence(log_w, n_raw)
    # The tail fit needs positive weights; folded rows at C = +inf are zeros
    # too, so the stricter count of finite weights decides.
    alive = is_alive(k_hat, int((~zero).sum()), n_raw)
    passed = bool(alive and k_hat < float(khat_threshold) and efficiency >= float(min_efficiency))
    cert = Certificate(
        name=str(name),
        kind=kind,
        round_index=None if round_index is None else int(round_index),
        u=u,
        cstat=cstat,
        log_q=log_q,
        log_w=log_w,
        log_w_smoothed=smoothed,
        n_raw=n_raw,
        n_in=n_in,
        k_hat=float(k_hat),
        ess=float(ess),
        ess_raw=float(ess_raw),
        efficiency=float(efficiency),
        log_z=float(log_z),
        log_z_err=float(log_z_err),
        alive=alive,
        passed=passed,
        tier=3,
        proposal=proposal,
        extra=dict(extra) if extra is not None else {},
    )
    cert.tier = tier(cert, khat_safe=khat_safe)
    return cert


def _is_stratified(proposal: object) -> bool:
    return all(callable(getattr(proposal, attr, None)) for attr in ("allocation", "sample_stratified", "log_prob_stratified"))


def draw_raw(proposal: ProposalLike, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    r"""Raw draws of a proposal and the log density of the design that produced them.

    A plain proposal gives `(sample_raw(n, rng), log_prob_raw(u))`. A
    stratified proposal (see `StratifiedLike`, e.g. `flows.MixtureProposal`)
    is drawn with its deterministic `allocation(n)` and weighed with
    `log_prob_stratified` of that allocation: its equal-weight `log_prob_raw`
    is the density of the design only when the number of strata $K$ divides
    *n*, and would otherwise bias the evidence by up to $K/n$ in relative
    terms. The draws are the same as `sample_raw(n, rng)` for a
    `MixtureProposal`.

    Parameters:
        proposal: Raw proposal.
        n: Number of draws, at least 1.
        rng: Generator consumed by the draws.

    Returns:
        `(u, log_q)`: float64 arrays of shapes `(n, d)` and `(n,)`.

    Raises:
        ValueError: If the proposal returns inconsistent shapes or an
            allocation that does not sum to *n*.
    """
    n = int(n)
    label = getattr(proposal, "name", proposal)
    if _is_stratified(proposal):
        counts = [int(c) for c in proposal.allocation(n)]
        if sum(counts) != n:
            raise ValueError(f"proposal {label!r} allocated {sum(counts)} draws for {n}.")
        u = np.asarray(proposal.sample_stratified(counts, rng), dtype=np.float64)
    else:
        counts = None
        u = np.asarray(proposal.sample_raw(n, rng), dtype=np.float64)
    d = getattr(proposal, "d", None)
    if u.ndim != 2 or u.shape[0] != n or (d is not None and u.shape[1] != int(d)):
        raise ValueError(f"proposal {label!r} returned shape {u.shape}, expected ({n}, {d if d is not None else 'd'}).")
    if counts is None:
        log_q = proposal.log_prob_raw(u)
    else:
        log_q = proposal.log_prob_stratified(u, counts)
    log_q = np.asarray(log_q, dtype=np.float64).ravel()
    if log_q.shape != (n,):
        raise ValueError(f"proposal {label!r} returned {log_q.shape[0]} log densities for {n} draws.")
    return u, log_q


def _fold(sim: SimulatorLike, u: np.ndarray, phase: str) -> np.ndarray:
    """One `sim.cstat` call, validated to `(len(u),)` float64."""
    cstat = np.asarray(sim.cstat(u, phase=phase), dtype=np.float64).ravel()
    if cstat.shape != (u.shape[0],):
        raise ValueError(f"the simulator returned shape {cstat.shape} for {u.shape[0]} rows.")
    return cstat


def certify_many(
    proposals: Sequence[ProposalLike],
    n_raw: int,
    sim: SimulatorLike,
    *,
    rng: np.random.Generator,
    phase: str,
    round_index: int | None,
    khat_threshold: float,
    khat_safe: float,
    min_efficiency: float,
    mixture: MixtureLike | None = None,
    kind: str = "screen",
) -> list[Certificate]:
    r"""Certify several proposals with a single simulator call.

    Each proposal draws *n_raw* raw points from its own child stream of *rng*
    (`rng.spawn`) through `draw_raw`, so a stratified proposal is weighed with
    the density of its realised allocation; all draws are folded in one
    `sim.cstat` call and split back.
    With a *mixture* whose members are exactly *proposals*, its certificate is
    appended, computed on the pooled draws with the stratified density
    $\log q = \operatorname{logsumexp}_k(\log q_k + \log(1/K))$: unbiased for
    these deterministic counts, and no extra fold.

    Parameters:
        proposals: Raw proposals (see `ProposalLike`).
        n_raw: Raw draws per proposal.
        sim: Simulator with a `cstat(u, *, phase)` method.
        rng: Parent generator; `rng.spawn` advances its spawn counter.
        phase: Fold-accounting phase passed to the simulator.
        round_index: SIXSA round stored on the certificates.
        khat_threshold: Gate on the Pareto shape.
        khat_safe: Tier-1 bound on the Pareto shape.
        min_efficiency: Gate on the efficiency in percent.
        mixture: Optional mixture of the proposals, certified on their draws.
        kind: Kind stored on every certificate.

    Returns:
        One certificate per proposal in order, then the mixture's if given.

    Raises:
        ValueError: If *proposals* is empty, *n_raw* is not positive, the draws
            have inconsistent shapes, or the mixture members are not
            *proposals* (by identity, in order).
        FoldError: Propagated from the simulator or on an invalid statistic.
    """
    proposals = list(proposals)
    if not proposals:
        raise ValueError("certify_many needs at least one proposal.")
    n_raw = int(n_raw)
    if n_raw < 1:
        raise ValueError(f"n_raw must be positive, got {n_raw}.")
    if mixture is not None:
        members = list(mixture.members)
        if len(members) != len(proposals) or any(m is not p for m, p in zip(members, proposals)):
            raise ValueError("the mixture members must be exactly the certified proposals, in order.")

    streams = rng.spawn(len(proposals))
    drawn = [draw_raw(proposal, n_raw, stream) for proposal, stream in zip(proposals, streams)]
    draws = [u for u, _ in drawn]
    if len({u.shape[1] for u in draws}) != 1:
        raise ValueError("all proposals must draw in the same dimension.")
    pooled = np.concatenate(draws, axis=0)
    cstat = _fold(sim, pooled, phase)

    common = {
        "kind": kind,
        "round_index": round_index,
        "khat_threshold": khat_threshold,
        "khat_safe": khat_safe,
        "min_efficiency": min_efficiency,
    }
    certificates = []
    for k, (proposal, (u, log_q)) in enumerate(zip(proposals, drawn)):
        certificates.append(
            build_certificate(
                proposal.name,
                u,
                cstat[k * n_raw : (k + 1) * n_raw],
                log_q,
                n_raw=n_raw,
                proposal=proposal,
                **common,
            )
        )
    if mixture is not None:
        counts = [n_raw] * len(proposals)
        log_q = np.asarray(mixture.log_prob_stratified(pooled, counts), dtype=np.float64)
        certificates.append(
            build_certificate(
                mixture.name,
                pooled,
                cstat,
                log_q,
                n_raw=pooled.shape[0],
                proposal=mixture,
                extra={"counts": counts},
                **common,
            )
        )
    return certificates


def certify_fresh(
    proposal: ProposalLike,
    n_raw: int,
    sim: SimulatorLike,
    *,
    rng: np.random.Generator,
    phase: str,
    round_index: int | None,
    khat_threshold: float,
    khat_safe: float,
    min_efficiency: float,
) -> Certificate:
    """Certify one proposal on new draws (the gate certificate).

    Parameters:
        proposal: Raw proposal.
        n_raw: Raw draws.
        sim: Simulator with a `cstat(u, *, phase)` method.
        rng: Parent generator (one child stream is spawned).
        phase: Fold-accounting phase.
        round_index: SIXSA round.
        khat_threshold: Gate on the Pareto shape.
        khat_safe: Tier-1 bound on the Pareto shape.
        min_efficiency: Gate on the efficiency in percent.

    Returns:
        A certificate of kind `"fresh"`.

    Raises:
        ValueError: As in `certify_many`.
        FoldError: As in `certify_many`.
    """
    return certify_many(
        [proposal],
        n_raw,
        sim,
        rng=rng,
        phase=phase,
        round_index=round_index,
        khat_threshold=khat_threshold,
        khat_safe=khat_safe,
        min_efficiency=min_efficiency,
        kind="fresh",
    )[0]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def rank_candidates(certs: Sequence[Certificate], *, khat_safe: float, tolerance: float) -> int:
    r"""Index of the best certificate by tiers; $\hat k$ gates and efficiency ranks.

    The lowest tier present wins. Within tiers 0 and 1 the weight variances
    are finite, so the highest efficiency wins. Within tier 2 efficiencies do
    not transfer between sample sizes: only candidates with
    `k_hat <= min(k_hat) + tolerance` are kept, then the highest efficiency
    wins. Within tier 3 nothing is readable and the lowest index wins. Ties go
    to the lower index.

    The choice depends on the whole set, never on a chain of pairwise
    comparisons, so tracking the best round as
    `rank_candidates(all fresh certificates so far)` cannot drift: a
    pairwise tolerance rule would walk r1 -> r2 -> r3 -> r4 along a ramp of
    k_hat values each within tolerance of the previous one.

    Parameters:
        certs: Candidates.
        khat_safe: Tier-1 bound on the Pareto shape; tiers are recomputed with it.
        tolerance: Tier-2 band on the Pareto shape.

    Returns:
        The index of the selected certificate in *certs*.

    Raises:
        ValueError: If *certs* is empty.
    """
    certs = list(certs)
    if not certs:
        raise ValueError("rank_candidates needs at least one certificate.")
    tiers = [tier(cert, khat_safe=khat_safe) for cert in certs]
    best_tier = min(tiers)
    pool = [i for i, t in enumerate(tiers) if t == best_tier]
    if best_tier == 3:
        return pool[0]
    if best_tier == 2:
        k_min = min(certs[i].k_hat for i in pool)
        pool = [i for i in pool if certs[i].k_hat <= k_min + float(tolerance) + _TOLERANCE_SLACK]
    # max() keeps the first maximum, i.e. the lower index on a tie.
    return max(pool, key=lambda i: certs[i].efficiency)
