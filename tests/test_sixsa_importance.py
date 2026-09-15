"""Tests for SIXSA defensive importance sampling (`_importance.py`).

Closed-form setting throughout: latent prior N(0, I_d), Gaussian "likelihood"
L(z) = N(z; mu, sigma^2 I) -- posterior and evidence are analytic:

    posterior = N(mu / (1 + sigma^2), sigma^2 / (1 + sigma^2) I)
    log Z     = sum_d log N(mu_d; 0, sqrt(1 + sigma^2))

so the pass's log_Z, ESS, k_hat and resampled moments can be checked exactly.
"""

from __future__ import annotations

import math
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

sys.modules.setdefault("xspec", MagicMock())

from bsixsa.backend.sixsa._importance import (
    DefensiveISBank,
    compute_log_evidence,
    defensive_is_diagnostics,
    defensive_is_draw_and_weight,
    ess_from_log_weights,
    passes_gate,
)

D = 2
MU = torch.tensor([1.0, -0.5])
SIGMA = 0.8
ALPHA = 0.05


class _GaussianFlow:
    """Stub posterior with the `sample`/`log_prob` surface the primitive needs."""

    def __init__(self, mean, std):
        self.dist = torch.distributions.MultivariateNormal(
            torch.as_tensor(mean, dtype=torch.float64),
            torch.eye(len(mean), dtype=torch.float64) * float(std) ** 2,
        )

    def sample(self, sample_shape):
        return self.dist.sample(sample_shape)

    def log_prob(self, theta):
        return self.dist.log_prob(theta.double())


def _latent_prior():
    return torch.distributions.MultivariateNormal(torch.zeros(D), torch.eye(D))


def _log_likelihood(z):
    like = torch.distributions.MultivariateNormal(
        MU.double(), torch.eye(D, dtype=torch.float64) * SIGMA**2
    )
    return like.log_prob(z.double())


def _analytic_posterior():
    s2 = SIGMA**2 / (1.0 + SIGMA**2)
    mean = MU.double() / (1.0 + SIGMA**2)
    return mean.numpy(), np.sqrt(s2)


def _analytic_log_z():
    scale = np.sqrt(1.0 + SIGMA**2)
    return float(
        sum(
            float(torch.distributions.Normal(0.0, scale).log_prob(m.double()))
            for m in MU
        )
    )


def test_defensive_is_recovers_analytic_logz_and_bounds_weights():
    torch.manual_seed(0)
    np.random.seed(0)
    post_mean, post_std = _analytic_posterior()
    flow = _GaussianFlow(post_mean, 1.3 * post_std)  # mass-covering

    batch = defensive_is_draw_and_weight(
        flow=flow, prior=_latent_prior(), log_likelihood_fn=_log_likelihood,
        n=8000, alpha=ALPHA,
    )
    diag = defensive_is_diagnostics(batch.log_w.numpy(), n_total=len(batch.z))

    # round(alpha*n) prior draws, the rest from the flow.
    assert batch.n_prior == round(ALPHA * 8000)
    assert len(batch.z) == 8000
    assert diag["khat"] < 0.7
    assert diag["ess"] > 1000.0
    assert abs(diag["log_Z"] - _analytic_log_z()) < 3.5 * diag["log_Z_err"] + 0.05

    # Prior floor: w = prior*L/q_def <= L/alpha  =>  log_w - log_lik <= -log(alpha).
    slack = (batch.log_w - batch.log_lik).max().item()
    assert slack <= -math.log(ALPHA) + 1e-6


def test_bank_pooled_estimate_balance_heuristic():
    torch.manual_seed(1)
    np.random.seed(1)
    post_mean, post_std = _analytic_posterior()
    # Two DIFFERENT flows so the balance-heuristic mixture is non-trivial.
    flow_a = _GaussianFlow(post_mean, 1.3 * post_std)
    flow_b = _GaussianFlow(post_mean + 0.2, 1.0 * post_std)

    bank = DefensiveISBank()
    for flow in (flow_a, flow_b):
        bank.add(
            defensive_is_draw_and_weight(
                flow=flow, prior=_latent_prior(),
                log_likelihood_fn=_log_likelihood, n=4000, alpha=ALPHA,
            )
        )

    diag = bank.pooled_estimate(_latent_prior(), balance_heuristic=True)
    assert diag["mode"] == "balance_heuristic"
    assert diag["n_batches"] == 2
    assert diag["n_total"] == 8000
    assert abs(diag["log_Z"] - _analytic_log_z()) < 3.5 * diag["log_Z_err"] + 0.05
    assert diag["khat"] < 0.7

    # Stratified fallback is also unbiased and should agree closely.
    diag_s = bank.pooled_estimate(_latent_prior(), balance_heuristic=False)
    assert diag_s["mode"] == "stratified"
    assert abs(diag_s["log_Z"] - _analytic_log_z()) < 3.5 * diag_s["log_Z_err"] + 0.1


def test_pooled_weighted_resamples_analytic_posterior():
    torch.manual_seed(2)
    np.random.seed(2)
    post_mean, post_std = _analytic_posterior()
    flow = _GaussianFlow(post_mean, 1.3 * post_std)

    bank = DefensiveISBank()
    bank.add(
        defensive_is_draw_and_weight(
            flow=flow, prior=_latent_prior(),
            log_likelihood_fn=_log_likelihood, n=8000, alpha=ALPHA,
        )
    )
    bank.pooled_estimate(_latent_prior())
    z, w = bank.pooled_weighted()
    assert len(z) == len(w)
    assert np.isclose(float(w.sum()), 1.0)

    idx = np.random.default_rng(0).choice(len(z), size=40000, replace=True, p=w.numpy())
    resampled = z.numpy()[idx]
    np.testing.assert_allclose(resampled.mean(axis=0), post_mean, atol=0.05)
    np.testing.assert_allclose(resampled.std(axis=0), post_std, rtol=0.15)


def test_best_flow_prefers_reliable_lowest_khat():
    torch.manual_seed(3)
    np.random.seed(3)
    post_mean, post_std = _analytic_posterior()
    good = _GaussianFlow(post_mean, 1.3 * post_std)          # reliable
    bad = _GaussianFlow(post_mean + 8.0, 2.0 * post_std)     # displaced, high khat

    bank = DefensiveISBank()
    bank.add(defensive_is_draw_and_weight(
        flow=bad, prior=_latent_prior(), log_likelihood_fn=_log_likelihood,
        n=2000, alpha=ALPHA))
    bank.add(defensive_is_draw_and_weight(
        flow=good, prior=_latent_prior(), log_likelihood_fn=_log_likelihood,
        n=2000, alpha=ALPHA))

    assert bank.best_flow(khat_threshold=0.7, min_eff=1.0) is good


def test_compute_log_evidence_and_ess_conventions():
    # Uniform weights: ESS = N, efficiency = 100%, Z_hat = mean(w) = 1 -> log_Z = 0.
    ess, eff = ess_from_log_weights(np.zeros(100))
    assert np.isclose(ess, 100.0)
    assert np.isclose(eff, 100.0)
    log_z, log_z_err, log_z_bias = compute_log_evidence(np.zeros(100))
    assert np.isclose(log_z, 0.0)
    assert np.isclose(log_z_err, 0.0)
    assert np.isclose(log_z_bias, 0.0)

    # A -inf half halves the evidence but leaves the finite-weight ESS.
    log_z, _, _ = compute_log_evidence(np.concatenate([np.zeros(50), np.full(50, -np.inf)]))
    assert np.isclose(log_z, np.log(0.5))
    ess, eff = ess_from_log_weights(np.concatenate([np.zeros(50), np.full(50, -np.inf)]))
    assert np.isclose(ess, 50.0)
    assert np.isclose(eff, 50.0)  # 50 effective / 100 total draws


def test_defensive_is_diagnostics_counts_nonfinite_in_n():
    log_w = np.concatenate([np.zeros(80), np.full(20, -np.inf)])
    diag = defensive_is_diagnostics(log_w)
    assert diag["n_total"] == 100
    assert diag["n_finite"] == 80
    assert np.isclose(diag["ess"], 80.0)
    assert np.isclose(diag["efficiency"], 80.0)  # 80 / 100 total


def test_passes_gate_logic():
    good = {"khat": 0.4, "efficiency": 5.0}
    assert passes_gate(good, khat_threshold=0.7, min_eff=1.0)
    assert not passes_gate({"khat": 0.9, "efficiency": 5.0}, 0.7, 1.0)   # khat too high
    assert not passes_gate({"khat": 0.4, "efficiency": 0.5}, 0.7, 1.0)   # eff too low
    assert not passes_gate({"khat": float("inf"), "efficiency": 90.0}, 0.7, 1.0)
    # khat gate disabled -> only efficiency matters.
    assert passes_gate({"khat": float("inf"), "efficiency": 5.0}, None, 1.0)
