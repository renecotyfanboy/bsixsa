"""Tests for the SIXSA final importance-sampling refinement (`_importance.py`).

The analytic setting throughout: latent prior N(0, I_d), Gaussian "likelihood"
L(z) = N(z; mu, sigma^2 I) -- the posterior and evidence are closed-form:

    posterior = N(mu / (1 + sigma^2), sigma^2 / (1 + sigma^2) I)
    log Z     = sum_d log N(mu_d; 0, sqrt(1 + sigma^2))

so the pass's log_Z, ESS, and resampled moments can be checked exactly.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

sys.modules.setdefault("xspec", MagicMock())

from bsixsa.backend.sixsa._importance import (
    combined_pool_evidence,
    ess_from_log_weights,
    final_importance_pass,
    resample_from_log_weights,
)

D = 2
MU = torch.tensor([1.0, -0.5])
SIGMA = 0.8


class _GaussianProposal:
    """Stub proposal with the `sample`/`log_prob` surface the pass needs."""

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
            float(
                torch.distributions.Normal(0.0, scale).log_prob(m.double())
            )
            for m in MU
        )
    )


def test_final_pass_recovers_analytic_logz_and_posterior():
    torch.manual_seed(0)
    np.random.seed(0)

    post_mean, post_std = _analytic_posterior()
    # Slightly widened true posterior: mass-covering, so k_hat stays low.
    proposal = _GaussianProposal(post_mean, 1.3 * post_std)

    info = final_importance_pass(
        proposal=proposal,
        latent_prior=_latent_prior(),
        log_likelihood_fn=_log_likelihood,
        n_per_round=4096,
        target_ess=2_000.0,
        fallback_target_ess=1_000.0,
        max_rounds=10,
    )

    assert info["reached_target_ess"]
    assert not info["hopeless"]
    assert info["mean_khat"] < 0.7
    assert info["ess"] >= 2_000.0
    assert abs(info["log_Z"] - _analytic_log_z()) < 3.0 * info["log_Z_err"]
    # The negative finite-sample bias correction is tiny but well-defined.
    assert info["log_Z_bias"] <= 0.0
    assert info["log_Z_corrected"] >= info["log_Z"]

    samples, weights = info["pooled_weighted"]
    assert len(samples) == len(weights)
    resampled = resample_from_log_weights(
        samples, np.log(weights.numpy()), 20_000, rng=np.random.default_rng(0)
    ).numpy()
    np.testing.assert_allclose(resampled.mean(axis=0), post_mean, atol=0.05)
    np.testing.assert_allclose(
        resampled.std(axis=0), post_std, rtol=0.1
    )


def test_final_pass_hopeless_exit():
    torch.manual_seed(0)
    np.random.seed(0)

    post_mean, post_std = _analytic_posterior()
    # Proposal displaced far from the posterior: a lucky near-mode draw
    # dominates the weights, k_hat blows up, and the pass must abort instead
    # of returning a garbage posterior.
    proposal = _GaussianProposal(post_mean + 8.0, 2.0 * post_std)

    info = final_importance_pass(
        proposal=proposal,
        latent_prior=_latent_prior(),
        log_likelihood_fn=_log_likelihood,
        n_per_round=512,
        target_ess=1_000.0,
        fallback_target_ess=500.0,
        max_rounds=5,
    )

    assert info["hopeless"]
    assert info["n_rounds"] == 2
    assert info["pooled_weighted"] is None


def test_final_pass_extension_and_plateau():
    torch.manual_seed(0)
    np.random.seed(0)

    post_mean, post_std = _analytic_posterior()
    proposal = _GaussianProposal(post_mean, 1.3 * post_std)

    # Unreachable targets: the main loop exhausts its rounds, the extension
    # kicks in and must stop on the ESS plateau (each round's gain is far
    # below 1% of the fallback target), never reaching the hard ceiling.
    info = final_importance_pass(
        proposal=proposal,
        latent_prior=_latent_prior(),
        log_likelihood_fn=_log_likelihood,
        n_per_round=256,
        target_ess=1e9,
        fallback_target_ess=1e8,
        max_rounds=2,
        max_rounds_extended=50,
    )

    assert not info["hopeless"]
    assert not info["reached_target_ess"]
    assert info["reached_fallback_ess"] is False
    # 2 main rounds + 2 stalled extension rounds.
    assert info["n_rounds"] == 4
    assert info["ess"] > 0.0
    assert info["pooled_weighted"] is not None


def test_resample_from_log_weights_filters_nonfinite_consistently():
    """Non-finite weights must drop their samples too (regression test for the
    reference's mask bug, where filtered-space indices were applied to the
    unfiltered sample stack)."""
    samples = torch.arange(10, dtype=torch.float64).reshape(-1, 1)
    log_w = np.zeros(10)
    log_w[[0, 3, 7]] = -np.inf
    log_w[5] = np.nan

    resampled = resample_from_log_weights(
        samples, log_w, 1_000, rng=np.random.default_rng(0)
    ).numpy()

    bad_rows = {0.0, 3.0, 7.0, 5.0}
    assert bad_rows.isdisjoint(set(np.unique(resampled)))

    with pytest.raises(ValueError, match="finite"):
        resample_from_log_weights(samples, np.full(10, -np.inf), 10)
    with pytest.raises(ValueError, match="entries"):
        resample_from_log_weights(samples, np.zeros(9), 10)


def test_ess_and_pooled_evidence_conventions():
    """Uniform weights give ESS = N; non-finite weights count toward N in the
    evidence (a zero-weight draw is still a draw) but not toward the ESS."""
    ess, eff = ess_from_log_weights(np.zeros(100))
    assert np.isclose(ess, 100.0)
    assert np.isclose(eff, 100.0)

    # Two rounds of uniform weights w = 1: Z_hat = mean(w) = 1 -> log_Z = 0.
    ess, log_z, log_z_err, log_z_bias = combined_pool_evidence(
        [np.zeros(50), np.zeros(50)]
    )
    assert np.isclose(ess, 100.0)
    assert np.isclose(log_z, 0.0)
    assert np.isclose(log_z_err, 0.0)

    # A -inf half halves the evidence but leaves the finite-weight ESS.
    ess, log_z, _, _ = combined_pool_evidence(
        [np.zeros(50), np.full(50, -np.inf)]
    )
    assert np.isclose(ess, 50.0)
    assert np.isclose(log_z, np.log(0.5))
