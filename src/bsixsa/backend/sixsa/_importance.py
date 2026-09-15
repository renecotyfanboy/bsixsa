"""Defensive importance sampling for the SIXSA backend (spec section 7).

The exactness layer of SIXSA. Every convergence CHECK and GATE draws from a
*defensive mixture* proposal

    q_def(z) = alpha * prior(z) + (1 - alpha) * q_flow(z)

with ``q_flow`` the equal-weight, leakage-corrected mixture density of the
member posteriors at ``x_obs`` (an sbi ``EnsemblePosterior``/``DirectPosterior``
``log_prob``). ``round(alpha*n)`` draws come from the prior and the rest from the
flow; each is simulated exactly. The importance log-weights, in the
unit-Gaussian latent space,

    log w = log prior(z) + log L(z) - log q_def(z),   log L = -0.5 * cstat,

are bounded by ``w <= L / alpha`` because ``q_def >= alpha * prior`` -- the prior
floor guarantees finite variance by construction. Diagnostics: PSIS ``khat``
(arviz ``psislw``) on ``log w``; ``ESS = 1 / sum(w_i^2)`` from the RAW normalized
weights; ``efficiency = 100 * ESS / n``; ``log Z = logmeanexp(log w)`` with the
Dax et al. (2022) error/bias.

Every check/gate batch is *banked*. The headline posterior / evidence come from
the banked pool, combined via a multiple-IS balance-heuristic estimator whose
denominator for each pooled point is the deterministic mixture over ALL banked
proposals -- so a prior-floor draw is scored against a denominator that
acknowledges every batch that could have produced it, and is never underweighted
by ``e^(thousands)``.

Latent-space note: the whole computation is done in the unit-Gaussian latent.
``log Z`` is reparameterization-invariant (``from_unit_gaussian`` pushes
``N(0, I)`` exactly onto the physical prior), so the reported evidence needs no
Jacobian term.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
from arviz import psislw
from scipy.special import logsumexp

# Minimum effective sample size below which the Pareto k_hat fit is
# uninformative (a too-small ESS makes psislw report a spuriously LOW k_hat on
# a catastrophic proposal). Below this we report k_hat = inf so the batch is
# never chosen best and never passes a gate. Mirrors `_sixsa_dev`.
_KHAT_MIN_ESS = 10.0


# --------------------------------------------------------------------------- #
# Evidence / ESS conventions (Dax et al. 2022, arXiv:2210.05686)
# --------------------------------------------------------------------------- #
def compute_log_evidence(raw_log_weights) -> tuple[float, float, float]:
    """``(log_Z, log_Z_err, log_Z_bias)`` from RAW importance log-weights.

    ``log_Z = logsumexp(log_w) - log(N)`` where ``N`` counts ALL draws including
    non-finite-weight ones (a non-finite weight is a zero-weight draw from the
    proposal, not a missing one). Error and negative finite-sample bias follow
    Dax et al. (2022), Suppl. Eqs. (6)/(8):
    ``err = sqrt((N - ess)/(N ess))``, ``bias = -err^2 / 2``.
    """
    log_w = np.asarray(raw_log_weights, dtype=np.float64).ravel()
    n_total = len(log_w)
    finite = log_w[np.isfinite(log_w)]
    if finite.size == 0 or n_total == 0:
        return float("nan"), float("nan"), float("nan")
    w = np.exp(finite - finite.max())
    w_norm = w / w.sum()
    ess = float(1.0 / np.sum(w_norm**2))
    log_z = float(logsumexp(finite) - np.log(n_total))
    log_z_err = float(np.sqrt((n_total - ess) / (n_total * ess)))
    log_z_bias = float(-0.5 * (n_total - ess) / (n_total * ess))
    return log_z, log_z_err, log_z_bias


def ess_from_log_weights(log_w) -> tuple[float, float]:
    """``(ESS, efficiency%)`` from RAW log-weights (ESS from raw normalized w)."""
    log_w = np.asarray(log_w, dtype=np.float64).ravel()
    n_total = len(log_w)
    finite = log_w[np.isfinite(log_w)]
    if finite.size == 0 or n_total == 0:
        return 0.0, 0.0
    w = np.exp(finite - finite.max())
    w /= w.sum()
    ess = float(1.0 / np.sum(w**2))
    return ess, float(100.0 * ess / n_total)


def _khat(log_w_finite: np.ndarray, ess: float) -> float:
    """PSIS k_hat with the min-ESS guard."""
    try:
        if not np.isfinite(ess) or ess < _KHAT_MIN_ESS:
            return float("inf")
        _, khat_val = psislw(np.asarray(log_w_finite, dtype=np.float64).copy())
        return float(khat_val)
    except Exception:
        return float("nan")


def defensive_is_diagnostics(log_w, n_total: int | None = None) -> dict:
    """Full defensive-IS diagnostics from a 1D array of RAW log-weights.

    ``n_total`` defaults to ``len(log_w)`` (non-finite weights are zero-weight
    draws and still count toward N and the efficiency denominator).
    """
    log_w = np.asarray(log_w, dtype=np.float64).ravel()
    if n_total is None:
        n_total = len(log_w)
    finite = log_w[np.isfinite(log_w)]
    out = {
        "ess": 0.0,
        "efficiency": 0.0,
        "khat": float("inf"),
        "log_Z": float("nan"),
        "log_Z_err": float("nan"),
        "log_Z_bias": float("nan"),
        "n_finite": int(finite.size),
        "n_total": int(n_total),
    }
    if finite.size == 0 or n_total == 0:
        return out
    w = np.exp(finite - finite.max())
    w_norm = w / w.sum()
    ess = float(1.0 / np.sum(w_norm**2))
    out["ess"] = ess
    out["efficiency"] = 100.0 * ess / n_total
    out["khat"] = _khat(finite, ess)
    out["log_Z"], out["log_Z_err"], out["log_Z_bias"] = compute_log_evidence(log_w)
    return out


def passes_gate(diag: dict, khat_threshold, min_eff: float) -> bool:
    """PASS iff k_hat < threshold AND efficiency >= min_eff (percent)."""
    khat_ok = khat_threshold is None or (
        np.isfinite(diag["khat"]) and diag["khat"] < khat_threshold
    )
    eff_ok = float(diag.get("efficiency", 0.0)) >= float(min_eff)
    return bool(khat_ok and eff_ok)


# --------------------------------------------------------------------------- #
# The defensive-IS draw + weight primitive
# --------------------------------------------------------------------------- #
@dataclass
class DefensiveISBatch:
    """One defensive-IS batch: draws, provenance and self-consistent weights."""

    z: torch.Tensor            # (n, d) latent draws (float32)
    log_prior: torch.Tensor    # (n,) prior.log_prob(z), double
    log_lik: torch.Tensor      # (n,) -0.5*cstat, double
    log_q_def: torch.Tensor    # (n,) defensive-mixture density at z, double
    log_w: torch.Tensor        # (n,) self-consistent log-weights, double
    flow: Any                  # the sbi posterior this batch's flow part drew from
    alpha: float
    n_requested: int
    kind: str = "check"        # "check" | "round_gate" | "final_gate" | "emulator"
    round_: int = 0
    member: int | None = None
    n_prior: int = 0
    extra: dict = field(default_factory=dict)


def _mixture_log_prob(flow, prior, alpha, z, *, log_prior=None):
    """``(log q_def, log_prior, log_q_flow)`` of the defensive mixture at ``z``."""
    if log_prior is None:
        log_prior = prior.log_prob(z).double()
    log_q_flow = flow.log_prob(z).double()
    la = math.log(alpha)
    l1a = math.log1p(-alpha)
    log_q_def = torch.logaddexp(la + log_prior, l1a + log_q_flow)
    return log_q_def, log_prior, log_q_flow


def _sample(dist, n):
    if n <= 0:
        return None
    try:
        s = dist.sample((n,), show_progress_bars=False)
    except TypeError:
        s = dist.sample((n,))
    return s.detach()


def defensive_is_draw_and_weight(
    *,
    flow,
    prior,
    log_likelihood_fn: Callable,
    n: int,
    alpha: float,
    kind: str = "check",
    round_: int = 0,
    member: int | None = None,
) -> DefensiveISBatch:
    """Draw ``n`` defensive-mixture samples, simulate them, and weight them.

    ``round(alpha*n)`` draws come from ``prior`` and the rest from ``flow``; the
    weights use the mixture density ``q_def`` so they are self-consistent (the
    denominator is exactly the density the batch was drawn from).
    """
    n_prior = int(round(alpha * n))
    n_prior = max(0, min(n, n_prior))
    n_flow = n - n_prior

    z_prior = _sample(prior, n_prior)
    z_flow = _sample(flow, n_flow)
    parts = [p for p in (z_prior, z_flow) if p is not None and len(p) > 0]
    if not parts:
        raise ValueError("defensive_is_draw_and_weight drew zero samples.")
    z = torch.cat([p.to(torch.float32) for p in parts], dim=0)

    log_q_def, log_prior, _ = _mixture_log_prob(flow, prior, alpha, z)
    log_lik = torch.as_tensor(log_likelihood_fn(z)).double().reshape(-1)
    log_w = log_prior + log_lik - log_q_def

    return DefensiveISBatch(
        z=z.detach(),
        log_prior=log_prior.detach(),
        log_lik=log_lik.detach(),
        log_q_def=log_q_def.detach(),
        log_w=log_w.detach(),
        flow=flow,
        alpha=float(alpha),
        n_requested=int(n),
        kind=kind,
        round_=round_,
        member=member,
        n_prior=int(n_prior),
    )


# --------------------------------------------------------------------------- #
# The bank + pooled multiple-IS estimator
# --------------------------------------------------------------------------- #
class DefensiveISBank:
    """Collects defensive-IS batches and combines them via multiple IS.

    The pooled estimator uses the balance heuristic: each pooled point is scored
    against the deterministic mixture over ALL banked proposals (weighted by
    each batch's sample count). A cheaper stratified fallback -- each point kept
    against its own batch's mixture -- is used when the balance-heuristic cost
    ``B * N`` exceeds ``max_bn`` (both estimators are unbiased; the balance
    heuristic has lower variance).
    """

    def __init__(self, max_bn: int = 40_000_000):
        self.batches: list[DefensiveISBatch] = []
        self.max_bn = int(max_bn)
        self._pooled_cache: dict | None = None

    def __len__(self) -> int:
        return len(self.batches)

    def add(self, batch: DefensiveISBatch) -> None:
        self.batches.append(batch)
        self._pooled_cache = None  # invalidate

    # -- per-batch helpers --
    def batch_diagnostics(self, batch: DefensiveISBatch) -> dict:
        return defensive_is_diagnostics(
            batch.log_w.cpu().numpy(), n_total=len(batch.z)
        )

    def best_batch(self, khat_threshold, min_eff: float) -> DefensiveISBatch | None:
        """The khat-best RELIABLE banked batch (least-bad khat if none reliable)."""
        if not self.batches:
            return None
        scored = [(self.batch_diagnostics(b), b) for b in self.batches]
        reliable = [
            (d, b) for d, b in scored if passes_gate(d, khat_threshold, min_eff)
        ]
        pool = reliable if reliable else scored
        # lowest khat, then highest efficiency
        pool = sorted(
            pool,
            key=lambda db: (
                db[0]["khat"] if np.isfinite(db[0]["khat"]) else float("inf"),
                -db[0]["efficiency"],
            ),
        )
        return pool[0][1]

    def best_flow(self, khat_threshold, min_eff: float):
        b = self.best_batch(khat_threshold, min_eff)
        return None if b is None else b.flow

    # -- pooled multiple-IS --
    def pooled_estimate(self, prior, *, balance_heuristic: bool = True) -> dict:
        """Combine every banked batch into a single pooled IS estimate."""
        if not self.batches:
            return defensive_is_diagnostics(np.array([]))

        z_all = torch.cat([b.z for b in self.batches], dim=0).to(torch.float32)
        log_prior_all = torch.cat([b.log_prior for b in self.batches]).double()
        log_lik_all = torch.cat([b.log_lik for b in self.batches]).double()
        counts = [len(b.z) for b in self.batches]
        n_total = int(sum(counts))

        use_balance = balance_heuristic and (len(self.batches) * n_total <= self.max_bn)
        if use_balance:
            # Balance heuristic: deterministic mixture over all banked proposals.
            log_n_over_N = [math.log(c) - math.log(n_total) for c in counts]
            comps = []
            for b, log_frac in zip(self.batches, log_n_over_N):
                log_q_def_b, _, _ = _mixture_log_prob(
                    b.flow, prior, b.alpha, z_all, log_prior=log_prior_all
                )
                comps.append(log_frac + log_q_def_b)
            log_q_mix = torch.logsumexp(torch.stack(comps, dim=0), dim=0)
            log_w_pool = log_prior_all + log_lik_all - log_q_mix
            mode = "balance_heuristic"
        else:
            # Stratified: each point kept against its own batch's mixture.
            log_w_pool = torch.cat([b.log_w for b in self.batches]).double()
            mode = "stratified"

        log_w_np = log_w_pool.cpu().numpy()
        diag = defensive_is_diagnostics(log_w_np, n_total=n_total)
        diag["mode"] = mode
        diag["n_batches"] = len(self.batches)
        self._pooled_cache = {"z": z_all.detach(), "log_w": log_w_pool.detach(), "diag": diag}
        return diag

    def pooled_weighted(self, prior=None) -> tuple[torch.Tensor, torch.Tensor]:
        """``(z_all, normalized_weights)`` over the pooled banked draws.

        Weights are the PSIS-smoothed pooled weights (smoothing stabilizes the
        returned posterior resample; the RAW pooled weights drive the evidence).
        """
        if self._pooled_cache is None:
            if prior is None:
                raise ValueError("Call pooled_estimate(prior) first, or pass prior.")
            self.pooled_estimate(prior)
        z = self._pooled_cache["z"]
        log_w = self._pooled_cache["log_w"].cpu().numpy()
        finite = np.isfinite(log_w)
        if not np.any(finite):
            raise ValueError("No finite pooled weights.")
        z_finite = z[torch.as_tensor(finite)]
        try:
            log_w_smoothed, _ = psislw(log_w[finite].copy())
        except Exception:
            log_w_smoothed = log_w[finite]
        w = np.exp(log_w_smoothed - log_w_smoothed.max())
        w_norm = torch.as_tensor(w / w.sum(), dtype=torch.float64)
        return z_finite, w_norm


# --------------------------------------------------------------------------- #
# CHECK / GATE runners
# --------------------------------------------------------------------------- #
def run_check(
    bank: DefensiveISBank,
    *,
    flow,
    prior,
    log_likelihood_fn,
    n: int,
    alpha: float,
    khat_threshold,
    min_eff: float,
    ess_mode: str = "off",
    ess_target: float | None = None,
    round_: int = 0,
    member: int | None = None,
) -> tuple[bool, dict]:
    """Per-member CHECK: defensive IS on a single member flow (banked, reported)."""
    batch = defensive_is_draw_and_weight(
        flow=flow, prior=prior, log_likelihood_fn=log_likelihood_fn,
        n=n, alpha=alpha, kind="check", round_=round_, member=member,
    )
    bank.add(batch)
    diag = bank.batch_diagnostics(batch)
    passed = passes_gate(diag, khat_threshold, min_eff)
    if passed and ess_mode == "floor" and ess_target is not None:
        pooled = bank.pooled_estimate(prior)
        passed = passed and (pooled["ess"] >= float(ess_target))
        diag["pooled_ess"] = pooled["ess"]
    return passed, diag


def run_gate(
    bank: DefensiveISBank,
    *,
    flow,
    prior,
    log_likelihood_fn,
    n: int,
    alpha: float,
    khat_threshold,
    min_eff: float,
    ess_mode: str = "off",
    ess_target: float | None = None,
    kind: str = "round_gate",
    round_: int = 0,
    member_selection: str = "apply",
) -> tuple[bool, dict]:
    """ROUND/FINAL GATE: defensive IS on the full ensemble flow (banked)."""
    batch = defensive_is_draw_and_weight(
        flow=flow, prior=prior, log_likelihood_fn=log_likelihood_fn,
        n=n, alpha=alpha, kind=kind, round_=round_, member=None,
    )
    batch.extra["member_selection"] = member_selection
    bank.add(batch)
    diag = bank.batch_diagnostics(batch)
    passed = passes_gate(diag, khat_threshold, min_eff)
    if passed and ess_mode == "floor" and ess_target is not None:
        pooled = bank.pooled_estimate(prior)
        passed = passed and (pooled["ess"] >= float(ess_target))
        diag["pooled_ess"] = pooled["ess"]
    return passed, diag
