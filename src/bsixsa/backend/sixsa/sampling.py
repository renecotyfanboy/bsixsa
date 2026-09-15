r"""Final posterior sampling: planned importance sampling from one proposal.

The evidence uses only the number of draws fixed in advance. An optional
ESS-driven top-up is used for posterior resampling, but never for the evidence:
even a single adaptive top-up would bias an estimate pooling both batches.
Posterior weights are smoothed once over all available draws. At most one
top-up is allowed when the planned draws fall short of the target ESS; any remaining shortfall is reported, not chased.

Two ESS conventions, never mixed: the PSIS ESS (smoothed weights) sizes the
top-up and drives resampling; the raw ESS only enters the delta-method error
$\sigma_{\log Z} = \sqrt{1/\mathrm{ESS}_\mathrm{raw} - 1/N_\mathrm{raw}}$,
which is calibrated for $\hat k < 0.5$. Above that the error is doubled and
flagged unreliable.

This module imports numpy, scipy and arviz only (no torch, no XSPEC).
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from .certify import ProposalLike, SimulatorLike, build_certificate, draw_raw

__all__ = ["FinalPosterior", "resample", "sample_final"]

# Below this Pareto shape the delta-method error of log Z is calibrated.
_KHAT_RELIABLE = 0.5


@dataclass(eq=False)
class FinalPosterior:
    r"""Weighted posterior draws and evidence from the final certificate.

    Attributes:
        u: Raw draws in cube coordinates, shape `(n_raw, d)`.
        cstat: Exact statistic, shape `(n_raw,)`, `+inf` outside the cube.
        log_w: Raw log weights, `-inf` for zero weights.
        log_w_smoothed: PSIS-smoothed normalised log weights.
        n_raw: Number of raw draws (planned plus top-up).
        n_in: Rows inside the closed cube.
        k_hat: Pareto shape of the final weights.
        ess: PSIS ESS.
        ess_raw: Raw ESS.
        efficiency: `100 * ess / max(n_in, 1)`.
        log_z: Raw evidence on the first `evidence_n_raw` planned draws only.
        log_z_err: Planned-draw delta-method error, doubled at evidence k-hat >= 0.5.
        log_z_err_reliable: `False` when `evidence_k_hat >= 0.5`, when that diagnostic cannot be
            read, or when the proposal was not certified.
        certified: Whether the proposal passed the gate.
        n_planned: Raw draws planned before any top-up.
        n_top_up: Raw draws added by the single top-up (0 if none).
        shortfall: `ceil(target_ess - ess)` floored at 0.
        warnings: Human-readable warnings for the caller to emit.
    """

    u: np.ndarray = field(repr=False)
    cstat: np.ndarray = field(repr=False)
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
    log_z_err_reliable: bool
    certified: bool
    n_planned: int
    n_top_up: int
    shortfall: int
    warnings: list[str] = field(default_factory=list)
    n_unique: int = 0
    evidence_n_raw: int = 0
    evidence_k_hat: float = float("inf")


def _positive_int(value: int, name: str) -> int:
    try:
        out = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.") from None
    if out < 1:
        raise ValueError(f"{name} must be positive, got {out}.")
    return int(out)


def _top_up_size(ess: float, n_raw: int, target_ess: float, max_draws: int, chunk: int) -> int:
    """Raw draws of the single top-up.

    Extrapolates the measured PSIS ESS rate to the target, rounds up to whole
    chunks, and caps at the remaining draw budget. With no positive ESS there
    is no rate to extrapolate, so the whole remaining budget is spent.
    """
    room = max_draws - n_raw
    rate = ess / n_raw
    if rate <= 0.0:
        return room
    wanted = max(math.ceil(target_ess / rate - n_raw), 1)
    wanted = math.ceil(wanted / chunk) * chunk
    return min(room, wanted)


def sample_final(
    proposal: ProposalLike,
    sim: SimulatorLike,
    *,
    n_planned: int,
    target_ess: float,
    max_draws: int,
    rng: np.random.Generator,
    certified: bool,
    khat_threshold: float = 0.7,
    chunk: int = 5000,
    phase: str = "sampling",
) -> FinalPosterior:
    r"""Draw a planned number of raw points from *proposal* and weight them once.

    The *n_planned* raw draws are made in chunks of *chunk* (each chunk one
    `certify.draw_raw` and one `sim.cstat` call), then weighted in one
    certificate.
    If the PSIS ESS is below *target_ess* and the budget allows, exactly one
    top-up of `min(max_draws - n_raw, ceil(target_ess / ess_rate - n_raw))`
    draws, rounded up to whole chunks, is made and the certificate is
    recomputed once on everything.

    Parameters:
        proposal: Raw proposal (see `certify.ProposalLike`).
        sim: Simulator with a `cstat(u, *, phase)` method.
        n_planned: Raw draws planned in advance.
        target_ess: PSIS ESS aimed for.
        max_draws: Budget on the total number of raw draws.
        rng: Generator consumed by the proposal draws, chunk after chunk.
        certified: Whether the proposal passed the gate; `False` marks the
            evidence error unreliable.
        khat_threshold: Pareto shape above which the posterior itself is
            flagged unreliable.
        chunk: Raw draws per simulator call.
        phase: Fold-accounting phase.

    Returns:
        The `FinalPosterior`.

    Raises:
        TypeError: If a count is not an integer.
        ValueError: If a count is not positive, *target_ess* is not positive,
            or *n_planned* exceeds *max_draws*.
        FoldError: Propagated from the simulator.
    """
    n_planned = _positive_int(n_planned, "n_planned")
    max_draws = _positive_int(max_draws, "max_draws")
    chunk = _positive_int(chunk, "chunk")
    target_ess = float(target_ess)
    if not target_ess > 0.0:
        raise ValueError(f"target_ess must be positive, got {target_ess}.")
    if n_planned > max_draws:
        raise ValueError(f"n_planned ({n_planned}) exceeds max_draws ({max_draws}).")

    draws: list[np.ndarray] = []
    folds: list[np.ndarray] = []
    log_qs: list[np.ndarray] = []

    def draw(n: int) -> None:
        for start in range(0, n, chunk):
            m = min(chunk, n - start)
            # Per chunk, a stratified proposal is weighed with the density of
            # that chunk's allocation; each chunk's weights are then unbiased
            # for its own draws, hence so is their pooled sum.
            u, log_q = draw_raw(proposal, m, rng)
            cstat = np.asarray(sim.cstat(u, phase=phase), dtype=np.float64).ravel()
            if cstat.shape != (m,):
                raise ValueError(f"the simulator returned shape {cstat.shape} for {m} rows.")
            draws.append(u)
            folds.append(cstat)
            log_qs.append(log_q)

    def certificate():
        u = np.concatenate(draws, axis=0)
        return build_certificate(
            proposal.name,
            u,
            np.concatenate(folds),
            np.concatenate(log_qs),
            n_raw=u.shape[0],
            kind="final",
            round_index=None,
            khat_threshold=khat_threshold,
            khat_safe=_KHAT_RELIABLE,
            min_efficiency=0.0,
        )

    draw(n_planned)
    cert = certificate()
    evidence = cert
    n_top_up = 0
    if cert.ess < target_ess and cert.n_raw < max_draws:
        n_top_up = _top_up_size(cert.ess, cert.n_raw, target_ess, max_draws, chunk)
        draw(n_top_up)
        cert = certificate()

    messages: list[str] = []
    shortfall = max(math.ceil(target_ess - cert.ess), 0)
    if shortfall > 0:
        messages.append(
            f"final sampling reached a PSIS ESS of {cert.ess:,.1f} with {cert.n_raw:,} raw draws, "
            f"{shortfall:,} short of the target {target_ess:,.0f}; the resampled rows repeat accordingly."
        )
    log_z_err = evidence.log_z_err
    reliable = True
    if not evidence.k_hat < _KHAT_RELIABLE:
        log_z_err *= 2.0
        reliable = False
        messages.append(
            f"evidence k_hat {evidence.k_hat:.3f} >= {_KHAT_RELIABLE}: the log Z error is not calibrated "
            f"there, so it is doubled and marked unreliable."
        )
    if not cert.k_hat < float(khat_threshold):
        messages.append(
            f"final k_hat {cert.k_hat:.3f} >= {float(khat_threshold)}: the tail diagnostic exceeds the "
            f"configured reliability threshold; the posterior itself is unreliable."
        )
    if not evidence.alive:
        reliable = False
    if not cert.alive:
        reliable = False
        messages.append(
            f"only {int(np.isfinite(cert.log_w).sum()):,} of {cert.n_raw:,} final draws have a "
            f"positive weight, too few for k_hat to be read."
        )
    if not certified:
        reliable = False
        messages.append("the final proposal did not pass the gate: the posterior and log Z are not certified.")

    return FinalPosterior(
        u=cert.u,
        cstat=cert.cstat,
        log_w=cert.log_w,
        log_w_smoothed=cert.log_w_smoothed,
        n_raw=cert.n_raw,
        n_in=cert.n_in,
        k_hat=cert.k_hat,
        ess=cert.ess,
        ess_raw=cert.ess_raw,
        efficiency=cert.efficiency,
        log_z=evidence.log_z,
        log_z_err=float(log_z_err),
        log_z_err_reliable=reliable,
        certified=bool(certified),
        n_planned=n_planned,
        n_top_up=int(n_top_up),
        shortfall=int(shortfall),
        warnings=messages,
        evidence_n_raw=evidence.n_raw,
        evidence_k_hat=evidence.k_hat,
    )


def resample(
    u: np.ndarray,
    log_w_smoothed: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Multinomial resampling of weighted draws into equal-weight rows.

    Parameters:
        u: Weighted draws, shape `(m, d)`.
        log_w_smoothed: Log weights, shape `(m,)`, normalised or not; `-inf`
            rows are never selected.
        n: Number of rows to return.
        rng: Generator.

    Returns:
        `(rows, n_unique)`: `n` rows drawn with replacement with probability
        proportional to the weights, shape `(n, d)`, and the number of distinct
        source rows among them.

    Raises:
        TypeError: If *n* is not an integer.
        ValueError: On inconsistent shapes, a negative *n*, weights with `NaN`
            or `+inf`, or no positive weight.
    """
    u = np.asarray(u)
    lw = np.asarray(log_w_smoothed, dtype=np.float64).ravel()
    if u.ndim != 2 or u.shape[0] != lw.size:
        raise ValueError(f"u must have shape ({lw.size}, d), got {u.shape}.")
    try:
        n = operator.index(n)
    except TypeError:
        raise TypeError(f"n must be an integer, got {type(n).__name__}.") from None
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}.")
    if np.any(np.isnan(lw)) or np.any(np.isposinf(lw)):
        raise ValueError("log weights must not contain NaN or +inf.")
    finite = np.isfinite(lw)
    if not finite.any():
        raise ValueError("cannot resample: no draw has a positive weight.")
    p = np.zeros(lw.size)
    p[finite] = np.exp(lw[finite] - logsumexp(lw[finite]))
    p /= p.sum()
    index = rng.choice(lw.size, size=int(n), replace=True, p=p)
    return u[index], int(np.unique(index).size)
