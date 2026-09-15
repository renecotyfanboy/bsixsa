r"""Truncated proposals: training draws on the high-density region of a flow.

Every SIXSA round after the first trains on points spread over

$$
A = [0, 1]^d \cap R \cap \{u : \log q(u) > t\},
$$

where $q$ is the raw (not cube-rejected) density of the selected proposal,
$R$ the history-matching region when there is one, and $t$ the
``quantile``-quantile of $\log q$ over the proposal's own in-cube draws
([`density_threshold`][bsixsa.backend.sixsa.truncation.density_threshold]).
The plain NPE loss learns $q \propto$ (sampling density) $\times \mathcal{L}$,
so a training set that is flat on $A$ yields the posterior restricted to $A$.

[`sample_truncated`][bsixsa.backend.sixsa.truncation.sample_truncated] offers
four samplers:

- ``"sir"`` (default, parity with the v2 reference): sbi's
  ``RestrictedPrior(sample_with="sir")`` with the proposal as candidate
  generator. sbi passes the 0/1 acceptance mask as a LOG potential and always
  weighs 32 candidates per draw, so the draws are tilted toward $q$ and a few
  percent land outside $A$. No budget applies.
- ``"flow"``: exact rejection with $q$ as envelope; a candidate in $A$ is kept
  with probability $e^{t - \log q(u)} \le 1$, so the draws are exactly flat on
  $A$.
- ``"ellipsoid"``: uniform draws in an ellipsoid fitted to the support draws in
  $A$, kept on $A$; exact whenever the ellipsoid covers $A$, which a shell
  test and a coverage check on fresh flow draws enforce.
- ``"auto"``: the cheaper of the two exact samplers, from estimates that cost
  one uniform pilot.

The exact samplers have a candidate budget; over it, only the ``"flow"`` path
raises the quantile (up to $10^{-2}$) before raising
[`TruncationBudgetError`][bsixsa.exceptions.TruncationBudgetError]. None of
them falls back to ``"sir"``. Certificates always target the full prior, so a
training set that is not flat on $A$ costs efficiency, never correctness.

This module imports numpy and scipy at import time; torch and sbi are imported
by the ``"sir"`` path only. It imports neither XSPEC nor the solver.
"""

from __future__ import annotations

import math
import operator
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import linalg

from ..._cube import in_cube
from ...exceptions import TruncationBudgetError

if TYPE_CHECKING:
    import torch

__all__ = [
    "METHODS",
    "QUANTILE_CAP",
    "SbiProposalProxy",
    "density_threshold",
    "sample_truncated",
]

METHODS = ("sir", "flow", "ellipsoid", "auto")
"""Accepted values of ``method`` in [`sample_truncated`][bsixsa.backend.sixsa.truncation.sample_truncated]."""

QUANTILE_CAP = 1e-2
"""Largest quantile the ``"flow"`` budget escalation may reach."""

ELLIPSOID_PAD = 1.1
ELLIPSOID_GROWTH = 1.15
SHELL_FRACTION = 0.95
MAX_ELLIPSOID_GROWTHS = 3
PILOT_DRAWS = 100_000
FLOW_SAMPLE_COST = 10.0
"""Cost of one flow draw in units of one ``log_prob`` evaluation (``"auto"``)."""

_MIN_SUPPORT_IN_CUBE = 10
_MIN_BATCH = 1_024
_MAX_BATCH = 100_000
_NO_HIT_LIMIT = 1_000_000


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _check_rng(rng: Any) -> np.random.Generator:
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy Generator, got {type(rng).__name__}.")
    return rng


def _check_quantile(quantile: float) -> float:
    quantile = float(quantile)
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must lie in (0, 1), got {quantile}.")
    return quantile


def _positive_int(value: Any, name: str) -> int:
    value = operator.index(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")
    return value


def _raw_draws(proposal: Any, n: int, rng: np.random.Generator) -> np.ndarray:
    """``proposal.sample_raw(n, rng)`` as a float64 ``(n, d)`` array."""
    u = np.asarray(proposal.sample_raw(int(n), rng), dtype=np.float64)
    if u.ndim != 2 or u.shape[0] != n:
        raise ValueError(f"proposal.sample_raw({n}) returned shape {u.shape}, expected ({n}, d).")
    return u


def _iid_raw_draws(proposal: Any, n: int, rng: np.random.Generator) -> np.ndarray:
    """Raw draws in random order.

    A mixture may return its draws grouped by member. Only a whole batch is
    then a mixture sample, not a prefix of it, and both the last batch of a
    rejection loop and sbi's 32-candidate groups take prefixes.
    """
    u = _raw_draws(proposal, n, rng)
    return u[rng.permutation(n)]


def _log_prob(proposal: Any, u: np.ndarray) -> np.ndarray:
    """``proposal.log_prob_raw(u)`` as a float64 ``(n,)`` array."""
    logq = np.asarray(proposal.log_prob_raw(u), dtype=np.float64).reshape(-1)
    if logq.shape[0] != u.shape[0]:
        raise ValueError(
            f"proposal.log_prob_raw returned {logq.shape[0]} values for {u.shape[0]} points."
        )
    return logq


def _region_accepts(region: Any, u: np.ndarray) -> np.ndarray:
    accepted = np.asarray(region.accepts(u), dtype=bool).reshape(-1)
    if accepted.shape[0] != u.shape[0]:
        raise ValueError(f"region.accepts returned {accepted.shape[0]} values for {u.shape[0]} points.")
    return accepted


def _acceptance_mask(proposal: Any, u: np.ndarray, threshold: float, region: Any) -> np.ndarray:
    """Membership in $A$: closed cube, then $\\log q > t$, then the region.

    The cheap tests run first so that the density and the surrogate only see
    the rows that survived them.
    """
    mask = np.zeros(u.shape[0], dtype=bool)
    idx = np.flatnonzero(in_cube(u))
    if idx.size:
        idx = idx[_log_prob(proposal, u[idx]) > threshold]
    if idx.size and region is not None:
        idx = idx[_region_accepts(region, u[idx])]
    mask[idx] = True
    return mask


def _over_budget(got: int, attempt_drawn: int, n: int, spent: int, max_candidates: int) -> bool:
    """Whether finishing *n* draws would exceed *max_candidates*.

    The acceptance is taken at an optimistic upper bound (about 5 Poisson
    sigmas above the observed hit count), so a false stop is negligible; this
    is the rule of [`Region.sample`][bsixsa.restriction.region.Region.sample].
    """
    if spent >= max_candidates:
        return True
    if attempt_drawn == 0:
        return False
    p_high = (got + 5.0 * math.sqrt(got + 1.0) + 5.0) / attempt_drawn
    return spent + (n - got) / p_high > max_candidates


def _batch_size(missing: int, acceptance: float, room: int) -> int:
    acceptance = max(acceptance, 1.0 / _MAX_BATCH)
    ask = int(math.ceil(1.2 * missing / acceptance)) + 64
    return max(1, min(max(_MIN_BATCH, min(ask, _MAX_BATCH)), room))


def _budget_error(
    method: str, n: int, quantiles: list[float], n_candidates: int, max_candidates: int
) -> TruncationBudgetError:
    tried = ", ".join(f"{q:g}" for q in quantiles)
    err = TruncationBudgetError(
        f"The exact {method!r} truncated sampler cannot draw {n:,} points within "
        f"{max_candidates:,} candidates, even at quantile {quantiles[-1]:g} "
        f"(tried {tried}; {n_candidates:,} candidates spent). Raise "
        f"max_candidates, or use the approximate 'sir' sampler explicitly."
    )
    err.quantiles = tuple(quantiles)
    err.n_candidates = int(n_candidates)
    return err


# ---------------------------------------------------------------------------
# Threshold and support
# ---------------------------------------------------------------------------


def density_threshold(
    proposal: Any,
    *,
    quantile: float,
    n_support: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, np.ndarray]:
    r"""Place the density threshold $t$ of a raw proposal.

    Draws *n_support* raw points from the proposal, keeps those in the closed
    unit cube and returns $t$ = ``np.quantile(log_q, quantile)`` over them, so
    that about a fraction *quantile* of the proposal's in-cube mass lies at
    $\log q \le t$. The threshold uses the same raw density as the acceptance
    test $\log q > t$, which is what makes the exact samplers flat.

    Parameters:
        proposal:
            Object with ``sample_raw(n, rng)`` and ``log_prob_raw(u)``.
        quantile:
            Fraction in $(0, 1)$ of the in-cube draws placed below $t$.
        n_support:
            Number of raw draws.
        rng:
            Random generator consumed by the draws.

    Returns:
        ``(t, kept_u, kept_logq)``: the threshold, the in-cube draws of shape
        ``(m, d)`` and their raw log densities of shape ``(m,)``.

    Raises:
        TypeError: If *rng* is not a ``np.random.Generator``.
        ValueError: If *quantile* is outside $(0, 1)$, fewer than 10 draws
            fall in the cube, or a log density of an in-cube draw is not
            finite.
    """
    quantile = _check_quantile(quantile)
    n_support = _positive_int(n_support, "n_support")
    rng = _check_rng(rng)
    u = _raw_draws(proposal, n_support, rng)
    kept = u[in_cube(u)]
    if kept.shape[0] < _MIN_SUPPORT_IN_CUBE:
        raise ValueError(
            f"Only {kept.shape[0]} of {n_support:,} proposal draws fall in the unit cube; "
            f"at least {_MIN_SUPPORT_IN_CUBE} are needed to place the density threshold."
        )
    logq = _log_prob(proposal, kept)
    if not np.all(np.isfinite(logq)):
        raise ValueError("The proposal returned a non-finite log density on its own in-cube draws.")
    return float(np.quantile(logq, quantile)), kept, logq


@dataclass
class _Support:
    """Support draws of one threshold, reduced to those in $A$."""

    quantile: float
    threshold: float
    n_raw: int
    u: np.ndarray
    logq: np.ndarray

    @property
    def flow_acceptance(self) -> float:
        r"""Acceptance of the exact flow sampler, $E_q[1_A \, e^{t - \log q}]$."""
        return float(np.sum(np.exp(self.threshold - self.logq)) / self.n_raw)


def _build_support(
    proposal: Any, quantile: float, n_support: int, region: Any, rng: np.random.Generator
) -> _Support:
    threshold, kept, logq = density_threshold(proposal, quantile=quantile, n_support=n_support, rng=rng)
    inside = logq > threshold
    kept, logq = kept[inside], logq[inside]
    if region is not None and kept.shape[0]:
        accepted = _region_accepts(region, kept)
        kept, logq = kept[accepted], logq[accepted]
    return _Support(quantile, threshold, n_support, kept, logq)


# ---------------------------------------------------------------------------
# Exact flow rejection
# ---------------------------------------------------------------------------


@dataclass
class _Outcome:
    """What one sampler call produced; ``u`` is ``None`` when it gave up."""

    u: np.ndarray | None
    drawn: int = 0
    acceptance: float = float("nan")
    growths: int = 0
    growth_reasons: list[str] = field(default_factory=list)
    fallback: str | None = None


def _flow_attempt(
    proposal: Any,
    n: int,
    support: _Support,
    region: Any,
    rng: np.random.Generator,
    max_candidates: int,
) -> _Outcome:
    """Exact rejection from the flow at one threshold, within one budget."""
    t = support.threshold
    kept: list[np.ndarray] = []
    got = hits = drawn = 0
    while got < n:
        if _over_budget(got, drawn, n, drawn, max_candidates):
            return _Outcome(None, drawn=drawn)
        acceptance = hits / drawn if hits else support.flow_acceptance
        ask = _batch_size(n - got, acceptance, max_candidates - drawn)
        candidates = _iid_raw_draws(proposal, ask, rng)
        uniform = rng.random(ask)
        drawn += ask

        idx = np.flatnonzero(in_cube(candidates))
        if idx.size:
            logq = _log_prob(proposal, candidates[idx])
            # On A, log q > t makes exp(t - log q) a valid probability; the
            # region runs last because the surrogate is the costly test.
            keep = (logq > t) & (uniform[idx] < np.exp(np.minimum(t - logq, 0.0)))
            idx = idx[keep]
        if idx.size and region is not None:
            idx = idx[_region_accepts(region, candidates[idx])]
        if idx.size:
            hits += idx.size
            kept.append(candidates[idx[: n - got]])
            got += min(idx.size, n - got)
    return _Outcome(np.concatenate(kept, axis=0), drawn=drawn, acceptance=hits / drawn)


def _sample_flow(
    proposal: Any,
    n: int,
    support: _Support,
    region: Any,
    rng: np.random.Generator,
    max_candidates: int,
    n_support: int,
    spent: int,
) -> tuple[_Outcome, _Support, list[float]]:
    """Exact flow rejection with the quantile escalation of the budget rule.

    Each quantile gets its own *max_candidates*; the rows accepted at a lower
    quantile are discarded, since they are flat on a different set.

    Raises:
        TruncationBudgetError: If the budget is exceeded at ``QUANTILE_CAP``.
    """
    quantiles: list[float] = []
    total = spent
    while True:
        quantiles.append(support.quantile)
        outcome = _flow_attempt(proposal, n, support, region, rng, max_candidates)
        total += outcome.drawn
        if outcome.u is not None:
            outcome.drawn = total
            return outcome, support, quantiles
        if support.quantile >= QUANTILE_CAP * (1.0 - 1e-9):
            raise _budget_error("flow", n, quantiles, total, max_candidates)
        next_quantile = min(support.quantile * 10.0, QUANTILE_CAP)
        support = _build_support(proposal, next_quantile, n_support, region, rng)


# ---------------------------------------------------------------------------
# Ellipsoid rejection
# ---------------------------------------------------------------------------


@dataclass
class _Ellipsoid:
    r"""The set $\{u : \lVert L^{-1}(u - m) \rVert \le R\}$ with $S = L L^\top$."""

    mean: np.ndarray
    chol: np.ndarray
    radius: float

    @classmethod
    def fit(cls, rows: np.ndarray) -> _Ellipsoid | None:
        """Fit to the support rows in $A$; ``None`` when degenerate."""
        n, d = rows.shape
        if n < 2 * (d + 1):
            return None
        mean = rows.mean(axis=0)
        cov = np.atleast_2d(np.cov(rows, rowvar=False))
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            return None
        diag = np.diag(chol)
        if not (np.all(np.isfinite(chol)) and diag.min() > 1e-8 * diag.max()):
            return None
        ellipsoid = cls(mean, chol, 1.0)
        radius = ELLIPSOID_PAD * float(np.max(ellipsoid.whitened_radius(rows)))
        if not (math.isfinite(radius) and radius > 0.0):
            return None
        ellipsoid.radius = radius
        return ellipsoid

    def whitened_radius(self, u: np.ndarray) -> np.ndarray:
        whitened = linalg.solve_triangular(self.chol, (u - self.mean).T, lower=True)
        return np.sqrt(np.sum(whitened * whitened, axis=0))

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Uniform draws in the ellipsoid and their whitened radii."""
        d = self.mean.shape[0]
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        r = self.radius * rng.random(n) ** (1.0 / d)
        return self.mean + (z * r[:, None]) @ self.chol.T, r


def _sample_ellipsoid(
    proposal: Any,
    n: int,
    support: _Support,
    region: Any,
    rng: np.random.Generator,
    max_candidates: int,
    n_support: int,
    ellipsoid: _Ellipsoid | None = None,
    acceptance_hint: float | None = None,
    cost_limit: float | None = None,
    members: int = 1,
) -> _Outcome:
    """Uniform rejection from a fitted ellipsoid; ``u=None`` asks for ``"flow"``.

    With *cost_limit* (the ``"auto"`` estimate of the flow sampler), every
    growth re-prices the ellipsoid: its acceptance is scaled by the volume
    ratio $(R_{old} / R_{new})^d$, and the call gives up (``fallback="cost"``)
    once *n* draws would cost more than the flow.
    """
    if ellipsoid is None:
        ellipsoid = _Ellipsoid.fit(support.u)
    if ellipsoid is None:
        n_rows, d = support.u.shape
        return _Outcome(None, fallback="support" if n_rows < 2 * (d + 1) else "degenerate")
    d = ellipsoid.mean.shape[0]
    t = support.threshold
    outcome = _Outcome(None)
    reference = (acceptance_hint, ellipsoid.radius) if acceptance_hint else None

    def grow(radius: float, reason: str) -> bool:
        if outcome.growths >= MAX_ELLIPSOID_GROWTHS:
            outcome.fallback = "growths"
            return False
        if cost_limit is not None and reference is not None:
            acceptance = reference[0] * (reference[1] / radius) ** d
            if n * members / acceptance >= cost_limit:
                outcome.fallback = "cost"
                return False
        ellipsoid.radius = radius
        outcome.growths += 1
        outcome.growth_reasons.append(reason)
        return True

    # A piece of A the support rows missed entirely would never be drawn, and
    # the shell test cannot see it either: look for it in fresh flow draws.
    fresh = _raw_draws(proposal, n_support, rng)
    in_a = _acceptance_mask(proposal, fresh, t, region)
    if in_a.any():
        r_max = float(np.max(ellipsoid.whitened_radius(fresh[in_a])))
        if r_max > ellipsoid.radius and not grow(ELLIPSOID_PAD * r_max, "disconnected"):
            return outcome

    kept: list[np.ndarray] = []
    got = hits = attempt = drawn = 0
    while got < n:
        if _over_budget(got, attempt, n, drawn, max_candidates):
            outcome.drawn, outcome.fallback = drawn, "budget"
            return outcome
        if hits:
            acceptance = hits / attempt
        elif reference is not None:
            acceptance = reference[0] * (reference[1] / ellipsoid.radius) ** d
        else:
            acceptance = 0.5 if attempt == 0 else 1.0 / attempt
        ask = _batch_size(n - got, acceptance, max_candidates - drawn)
        candidates, radii = ellipsoid.sample(ask, rng)
        drawn += ask
        attempt += ask
        mask = _acceptance_mask(proposal, candidates, t, region)
        n_mask = int(mask.sum())
        if n_mask and np.any(radii[mask] > SHELL_FRACTION * ellipsoid.radius):
            # A reaches the edge of E, so E may clip it: every row accepted so
            # far is flat on A ∩ E only and must go.
            reference = ((hits + n_mask) / attempt, ellipsoid.radius)
            if not grow(ELLIPSOID_GROWTH * ellipsoid.radius, "shell"):
                outcome.drawn = drawn
                return outcome
            kept, got, hits, attempt = [], 0, 0, 0
            continue
        if n_mask:
            accepted = candidates[mask]
            hits += n_mask
            kept.append(accepted[: n - got])
            got += min(n_mask, n - got)
    outcome.u = np.concatenate(kept, axis=0)
    outcome.drawn = drawn
    outcome.acceptance = hits / attempt
    return outcome


# ---------------------------------------------------------------------------
# sbi SIR path
# ---------------------------------------------------------------------------


class SbiProposalProxy:
    r"""A raw proposal exposed as the ``posterior`` of sbi's SIR sampler.

    In sbi 0.27, ``RestrictedPrior.sample(sample_with="sir")`` reaches the
    proposal only through ``importance_sample``, which calls
    ``sample((m,), show_progress_bar=...)`` (retrying without the keyword on
    ``TypeError``) and then ``log_prob(samples)``. Nothing else is used.

    ``sample`` returns draws rejected to the closed unit cube, in random
    order. The v2 reference handed sbi a ``DirectPosterior``, whose ``sample``
    rejects leaked draws to the prior support, so candidates never left the
    box; raw draws would let SIR resample points outside the cube, which
    cannot be folded. ``log_prob`` is the raw log density: the in-cube
    normaliser the v2 posterior divided by is a constant, and SIR's softmax
    over each candidate group cancels it.

    Parameters:
        proposal:
            Object with ``sample_raw(n, rng)`` and ``log_prob_raw(u)``.
        rng:
            Random generator consumed by every ``sample`` call.

    Attributes:
        proposal: The wrapped proposal.
        rng: The generator.
        requests: Number of rows requested by each ``sample`` call.
        n_raw: Raw draws spent, including those rejected outside the cube.
    """

    def __init__(self, proposal: Any, rng: np.random.Generator):
        self.proposal = proposal
        self.rng = _check_rng(rng)
        self.requests: list[int] = []
        self.n_raw = 0

    @property
    def n_requested(self) -> int:
        """Total number of rows sbi has requested."""
        return int(sum(self.requests))

    def sample(self, sample_shape: Any = (), **kwargs: Any) -> torch.Tensor:
        """Draw in-cube proposal rows.

        Parameters:
            sample_shape:
                Batch shape, e.g. ``(m,)``.
            **kwargs:
                Ignored (sbi passes ``show_progress_bar``).

        Returns:
            float32 tensor of shape ``(*sample_shape, d)``.

        Raises:
            ValueError: If no raw draw lands in the cube within one million
                draws.
        """
        import torch

        shape = (int(sample_shape),) if isinstance(sample_shape, int) else tuple(int(s) for s in sample_shape)
        n = math.prod(shape)
        self.requests.append(n)
        u = self._draw_in_cube(n)
        return torch.from_numpy(u.astype(np.float32)).reshape(*shape, -1)

    def log_prob(self, theta: Any, **kwargs: Any) -> torch.Tensor:
        """Raw log density of the proposal.

        Parameters:
            theta:
                Tensor or array of shape ``(m, d)``.
            **kwargs:
                Ignored.

        Returns:
            float32 tensor of shape ``(m,)``.
        """
        import torch

        if isinstance(theta, torch.Tensor):
            theta = theta.detach().cpu().numpy()
        u = np.asarray(theta, dtype=np.float64)
        u = u.reshape(-1, u.shape[-1])
        return torch.from_numpy(_log_prob(self.proposal, u).astype(np.float32))

    def set_default_x(self, x: Any) -> SbiProposalProxy:
        """No-op kept for sbi code that attaches a context; returns ``self``."""
        return self

    def _draw_in_cube(self, n: int) -> np.ndarray:
        kept: list[np.ndarray] = []
        got = raw = 0
        while got < n:
            if got == 0 and raw >= _NO_HIT_LIMIT:
                raise ValueError(f"No proposal draw fell in the unit cube in {raw:,} raw draws.")
            missing = n - got
            if raw == 0:
                ask = missing
            else:
                ask = int(math.ceil(1.2 * missing * raw / max(got, 1))) + 16
                ask = min(ask, max(_MAX_BATCH, 4 * missing))
            draws = _iid_raw_draws(self.proposal, ask, self.rng)
            raw += ask
            inside = draws[in_cube(draws)]
            kept.append(inside[:missing])
            got += min(inside.shape[0], missing)
        self.n_raw += raw
        return np.concatenate(kept, axis=0)


def _sample_sir(
    proposal: Any, n: int, support: _Support, region: Any, rng: np.random.Generator, d: int
) -> tuple[np.ndarray, SbiProposalProxy]:
    """v2 truncation: sbi ``RestrictedPrior(sample_with="sir")`` on the proxy."""
    import torch
    from sbi.utils import BoxUniform, RestrictedPrior

    t = support.threshold

    def accept_reject_fn(theta: torch.Tensor) -> torch.Tensor:
        u = theta.detach().cpu().numpy().astype(np.float64).reshape(-1, d)
        return torch.from_numpy(_acceptance_mask(proposal, u, t, region))

    proxy = SbiProposalProxy(proposal, rng)
    restricted = RestrictedPrior(
        prior=BoxUniform(torch.zeros(d), torch.ones(d)),
        accept_reject_fn=accept_reject_fn,
        posterior=proxy,
        sample_with="sir",
    )
    # SIR's resampling decision reads the global torch generator.
    seed = int(rng.integers(0, 2**63 - 1))
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        samples = restricted.sample((n,), show_progress_bars=False, print_rejected_frac=False)
    u = samples.detach().cpu().numpy().astype(np.float64).reshape(n, d)
    return u, proxy


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def sample_truncated(
    proposal: Any,
    *,
    n: int,
    quantile: float,
    n_support: int,
    region: Any = None,
    method: str = "sir",
    rng: np.random.Generator,
    max_candidates: int = 20_000_000,
    log: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    r"""Draw *n* training points on $A = [0,1]^d \cap R \cap \{\log q > t\}$.

    The threshold $t$ comes from
    [`density_threshold`][bsixsa.backend.sixsa.truncation.density_threshold]
    at *quantile* over *n_support* raw draws.

    - ``"sir"``: sbi ``RestrictedPrior(BoxUniform(0, 1), accept_fn,
      SbiProposalProxy, sample_with="sir")``. The draws are NOT flat on $A$:
      sbi weighs 32 in-cube candidates per draw by $e^{1_A}/q$, so they lean
      toward $q$ and some fall outside $A$ (measured in
      ``fraction_outside_contour`` and ``fraction_outside_region``). No budget.
    - ``"flow"``: raw candidates kept on $A$ with probability
      $e^{t - \log q}$; exactly flat on $A$. Over *max_candidates* the
      quantile is multiplied by 10 (at most ``QUANTILE_CAP``), $t$ is
      recomputed and the accepted rows are discarded; over budget at the cap,
      [`TruncationBudgetError`][bsixsa.exceptions.TruncationBudgetError].
    - ``"ellipsoid"``: mean $m$ and covariance $S$ of the support rows in $A$,
      radius $R = 1.1 \max r$ with $r = \lVert L^{-1}(u - m) \rVert$; uniform
      draws in the ellipsoid kept on $A$. A coverage check on *n_support*
      fresh flow draws enlarges $R$ to $1.1\,r$ if any of them in $A$ lies
      beyond $R$. If an accepted draw has $r > 0.95 R$, every accepted draw is
      discarded and $R$ grows by 1.15. Both count as growths. It falls back
      to ``"flow"``, recording ``fallback``, with fewer than $2(d+1)$ support
      rows in $A$ (``"support"``), a degenerate covariance
      (``"degenerate"``), a fourth growth (``"growths"``) or over budget
      (``"budget"``).
    - ``"auto"``: ``"ellipsoid"`` if
      $n K / \mathrm{acc}_E < n (10 + K) / \hat P_{flow}$, else ``"flow"``,
      where $\hat P_{flow}$ is the flow sampler's acceptance estimated on the
      support draws, $\mathrm{acc}_E$ the acceptance of a 100 000-point
      uniform pilot in the ellipsoid (0 if degenerate) and $K$ the number of
      mixture members (1 for a single flow). Each growth re-prices the
      ellipsoid by the volume ratio $(R_{old}/R_{new})^d$ and switches to
      ``"flow"`` (``fallback="cost"``) once it is no longer the cheaper.

    Each exact sampler (and each quantile of ``"flow"``) has its own budget of
    *max_candidates*; ``n_candidates`` reports the total. No path falls back to
    ``"sir"``.

    Parameters:
        proposal:
            Object with ``sample_raw(n, rng)`` and ``log_prob_raw(u)``; an
            optional ``members`` list sets $K$.
        n:
            Number of draws.
        quantile:
            Quantile of the density threshold, in $(0, 1)$.
        n_support:
            Raw draws used to place the threshold (and for the coverage check).
        region:
            Optional object with ``accepts(u) -> bool (n,)``, intersected with
            the contour.
        method:
            One of ``"sir"``, ``"flow"``, ``"ellipsoid"``, ``"auto"``.
        rng:
            Random generator consumed by every draw (and seeding torch inside
            a forked RNG scope for ``"sir"``).
        max_candidates:
            Candidate budget of the exact samplers.
        log:
            Optional callable receiving a one-line summary.

    Returns:
        ``(u, info)``: float64 draws of shape ``(n, d)`` and a dict with keys
        ``method``, ``method_used`` (``"sir"``, ``"flow"`` or
        ``"ellipsoid"``), ``threshold``, ``quantile_used``, ``n_candidates``,
        ``acceptance`` (accepted per candidate of the sampler that produced
        the rows), ``growths``, ``fraction_outside_contour`` and
        ``fraction_outside_region`` (0 by construction for the exact
        samplers), plus ``growth_reasons``, ``fallback``,
        ``quantiles_tried``, ``estimates`` (``"auto"`` only, else ``None``)
        and ``seconds``.

    Raises:
        TypeError: If *rng* is not a ``np.random.Generator``.
        ValueError: On an unknown *method*, a non-positive *n*,
            *n_support* or *max_candidates*, or a threshold that cannot be
            placed (see ``density_threshold``).
        TruncationBudgetError: If the exact ``"flow"`` path exceeds its budget
            at ``QUANTILE_CAP``.
    """
    started = time.perf_counter()
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}.")
    n = _positive_int(n, "n")
    n_support = _positive_int(n_support, "n_support")
    max_candidates = _positive_int(max_candidates, "max_candidates")
    quantile = _check_quantile(quantile)
    rng = _check_rng(rng)

    support = _build_support(proposal, quantile, n_support, region, rng)
    d = int(getattr(proposal, "d", None) or support.u.shape[1])
    info: dict[str, Any] = {
        "method": method,
        "method_used": None,
        "threshold": support.threshold,
        "quantile_used": quantile,
        "n_candidates": 0,
        "acceptance": float("nan"),
        "growths": 0,
        "fraction_outside_contour": 0.0,
        "fraction_outside_region": 0.0,
        "growth_reasons": [],
        "fallback": None,
        "quantiles_tried": [quantile],
        "estimates": None,
        "seconds": 0.0,
    }

    if method == "sir":
        u, proxy = _sample_sir(proposal, n, support, region, rng, d)
        info["method_used"] = "sir"
        info["n_candidates"] = proxy.n_requested
        info["acceptance"] = n / max(proxy.n_requested, 1)
        in_contour = in_cube(u)
        if in_contour.any():
            in_contour[in_contour] = _log_prob(proposal, u[in_contour]) > support.threshold
        info["fraction_outside_contour"] = float(np.mean(~in_contour))
        if region is not None:
            info["fraction_outside_region"] = float(np.mean(~_region_accepts(region, u)))
    else:
        choice = method
        ellipsoid = None
        acceptance_hint = None
        if method == "auto":
            choice, ellipsoid, acceptance_hint, info["estimates"] = _choose_exact(
                proposal, n, support, region, rng
            )

        outcome = None
        spent = 0
        if choice == "ellipsoid":
            outcome = _sample_ellipsoid(
                proposal,
                n,
                support,
                region,
                rng,
                max_candidates,
                n_support,
                ellipsoid,
                acceptance_hint,
                cost_limit=None if info["estimates"] is None else info["estimates"]["cost_flow"],
                members=len(getattr(proposal, "members", [0])),
            )
            info["growths"] = outcome.growths
            info["growth_reasons"] = list(outcome.growth_reasons)
            spent = outcome.drawn
            if outcome.u is None:
                info["fallback"] = outcome.fallback
                outcome = None
            else:
                info["method_used"] = "ellipsoid"
        if outcome is None:
            outcome, support, quantiles = _sample_flow(
                proposal, n, support, region, rng, max_candidates, n_support, spent
            )
            info["method_used"] = "flow"
            info["quantiles_tried"] = quantiles
        info["n_candidates"] = int(outcome.drawn)
        info["acceptance"] = float(outcome.acceptance)
        info["threshold"] = support.threshold
        info["quantile_used"] = support.quantile
        u = outcome.u

    info["seconds"] = time.perf_counter() - started
    if log is not None:
        route = info["method_used"] if method == info["method_used"] else f"{method}->{info['method_used']}"
        log(
            f"[truncation] {route}: {n:,} draws at quantile {info['quantile_used']:g} "
            f"(t={info['threshold']:.4g}), acceptance {info['acceptance']:.3g} over "
            f"{info['n_candidates']:,} candidates, {info['growths']} growths, outside contour "
            f"{100 * info['fraction_outside_contour']:.1f}%, outside region "
            f"{100 * info['fraction_outside_region']:.1f}%, {info['seconds']:.1f}s"
        )
    return u, info


def _choose_exact(
    proposal: Any, n: int, support: _Support, region: Any, rng: np.random.Generator
) -> tuple[str, _Ellipsoid | None, float | None, dict[str, float]]:
    """Pick the cheaper exact sampler in ``log_prob`` equivalents."""
    p_flow = support.flow_acceptance
    ellipsoid = _Ellipsoid.fit(support.u)
    acc_ellipsoid = 0.0
    if ellipsoid is not None:
        pilot, _ = ellipsoid.sample(PILOT_DRAWS, rng)
        acc_ellipsoid = float(np.mean(_acceptance_mask(proposal, pilot, support.threshold, region)))
    k = len(getattr(proposal, "members", [0]))
    cost_ellipsoid = n / acc_ellipsoid * k if acc_ellipsoid > 0 else math.inf
    cost_flow = n / p_flow * (FLOW_SAMPLE_COST + k) if p_flow > 0 else math.inf
    choice = "ellipsoid" if cost_ellipsoid < cost_flow else "flow"
    estimates = {
        "acceptance_flow": p_flow,
        "acceptance_ellipsoid": acc_ellipsoid,
        "cost_flow": cost_flow,
        "cost_ellipsoid": cost_ellipsoid,
        "members": k,
    }
    return choice, ellipsoid, (acc_ellipsoid or None), estimates
