"""Tests for `bsixsa.backend.sixsa.truncation` with analytic proposals.

The proposals are `tests/_analytic.GaussianProposal`, a local two-member
mixture and a volume-preserving banana, whose raw log densities are exact. The
set $A = [0, 1]^d \\cap R \\cap \\{\\log q > t\\}$ is then known in closed form
(an ellipse for a Gaussian) or by quadrature on a uniform reference, so the
flatness of the exact samplers is checked with chi-square tests, and the
documented departures of sbi's SIR path are pinned.
"""

from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy import linalg, special, stats

sys.modules.setdefault("xspec", MagicMock())
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _analytic import GaussianProposal  # noqa: E402

from bsixsa._cube import in_cube  # noqa: E402
from bsixsa.backend.sixsa import truncation  # noqa: E402
from bsixsa.backend.sixsa.truncation import (  # noqa: E402
    QUANTILE_CAP,
    SbiProposalProxy,
    density_threshold,
    sample_truncated,
)
from bsixsa.exceptions import TruncationBudgetError  # noqa: E402

P_MIN = 1e-3
_LOG_2PI = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Local proposals and helpers
# ---------------------------------------------------------------------------


class _Mixture:
    """Weighted mixture of raw Gaussian proposals, draws grouped by member.

    The grouping mimics `MixtureProposal.sample_stratified`: only a whole batch
    is a mixture sample, so any sampler that takes a prefix must shuffle.
    """

    name = "mixture"

    def __init__(self, members, weights):
        self.members = list(members)
        self.weights = np.asarray(weights, dtype=np.float64) / np.sum(weights)
        self.d = self.members[0].d

    def sample_raw(self, n, rng):
        counts = rng.multinomial(int(n), self.weights)
        return np.concatenate([m.sample_raw(int(c), rng) for m, c in zip(self.members, counts)], axis=0)

    def log_prob_raw(self, u):
        u = np.atleast_2d(np.asarray(u, dtype=np.float64))
        parts = np.stack([m.log_prob_raw(u) + math.log(w) for m, w in zip(self.members, self.weights)])
        return special.logsumexp(parts, axis=0)


class _Banana:
    """Gaussian in warped coordinates, $u = c + a y$ with $u_{2k+1} \\mathrel{+}= b (y_{2k}^2 - 1)$.

    The map is triangular with a constant Jacobian $a^d$, so the raw log
    density is exact: $\\log q(u) = \\log N(y(u); 0, I) - d \\log a$.
    """

    name = "banana"

    def __init__(self, d=6, scale=0.02, bend=0.035):
        self.d, self.a, self.b = d, scale, bend
        self.c = np.full(d, 0.5)
        self.c[1::2] = 0.15

    def sample_raw(self, n, rng):
        y = rng.standard_normal((int(n), self.d))
        u = self.c + self.a * y
        u[:, 1::2] += self.b * (y[:, 0::2] ** 2 - 1.0)
        return u

    def log_prob_raw(self, u):
        u = np.atleast_2d(np.asarray(u, dtype=np.float64))
        y = (u - self.c) / self.a
        y[:, 1::2] = (u[:, 1::2] - self.c[1::2] - self.b * (y[:, 0::2] ** 2 - 1.0)) / self.a
        return -0.5 * np.sum(y * y, axis=1) - 0.5 * self.d * _LOG_2PI - self.d * math.log(self.a)


class _HalfPlane:
    """Stub region $\\{u \\in [0,1]^d : u_0 < c\\}$ that counts its calls."""

    def __init__(self, cut):
        self.cut = float(cut)
        self.calls = 0

    def accepts(self, u):
        self.calls += 1
        u = np.atleast_2d(np.asarray(u, dtype=np.float64))
        return in_cube(u) & (u[:, 0] < self.cut)


def _gaussian_2d():
    return GaussianProposal([0.5, 0.45], [[0.008, 0.003], [0.003, 0.006]])


def _contour_radius(proposal, threshold):
    """Whitened radius of the Gaussian level set $\\log q = t$."""
    r2 = -2.0 * threshold - proposal.d * _LOG_2PI - proposal._logdet
    return math.sqrt(r2)


def _whitened(proposal, u):
    return linalg.solve_triangular(proposal._chol, (u - proposal.mean).T, lower=True).T


def _chi2_uniform(values, bins):
    """p-value of a chi-square test that *values* in [0, 1] are uniform."""
    observed, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    return stats.chisquare(observed).pvalue


def _radial_pvalue(proposal, u, threshold):
    r"""Uniformity of $(r / r_t)^d$, which is U(0, 1) for a flat draw on the whitened ball."""
    r = np.linalg.norm(_whitened(proposal, u), axis=1)
    return _chi2_uniform((r / _contour_radius(proposal, threshold)) ** proposal.d, bins=10)


def _grid_pvalue(u, inside_fn, rng, *, bins=10, n_reference=2_000_000):
    """Chi-square test of flatness on a 2-D grid, cells weighted by their area in A.

    The area of A in each cell is measured with a uniform reference sample in
    the padded bounding box of the draws. Cells expecting fewer than 5 draws
    (slivers on the boundary of A) are pooled into one bin; membership of the
    draws in A is asserted by the callers.
    """
    lo = u.min(axis=0) - 0.02
    hi = u.max(axis=0) + 0.02
    reference = lo + (hi - lo) * rng.random((n_reference, 2))
    reference = reference[inside_fn(reference)]
    edges = [np.linspace(lo[j], hi[j], bins + 1) for j in range(2)]
    observed, _, _ = np.histogram2d(u[:, 0], u[:, 1], bins=edges)
    area, _, _ = np.histogram2d(reference[:, 0], reference[:, 1], bins=edges)
    expected = area * (len(u) / area.sum())
    small = expected < 5.0
    obs = np.append(observed[~small], observed[small].sum())
    exp = np.append(expected[~small], expected[small].sum())
    if exp[-1] == 0.0:
        obs, exp = obs[:-1], exp[:-1] * (obs[:-1].sum() / exp[:-1].sum())
    return stats.chisquare(obs, exp).pvalue


def _info_without_timing(info):
    return {k: v for k, v in info.items() if k != "seconds"}


# ---------------------------------------------------------------------------
# density_threshold
# ---------------------------------------------------------------------------


def test_density_threshold_quantile_semantics():
    proposal = _gaussian_2d()
    quantile = 0.05

    t, kept_u, kept_logq = density_threshold(
        proposal, quantile=quantile, n_support=200_000, rng=np.random.default_rng(1)
    )

    assert kept_u.dtype == np.float64 and kept_logq.dtype == np.float64
    assert np.all(in_cube(kept_u))
    np.testing.assert_allclose(kept_logq, proposal.log_prob_raw(kept_u))
    assert t == pytest.approx(np.quantile(kept_logq, quantile))
    # np.quantile interpolates, so the empirical fraction below t is within one row.
    assert abs(np.mean(kept_logq <= t) - quantile) <= 1.0 / len(kept_logq)
    # The proposal sits inside the cube: t is the analytic chi2 level set.
    r2 = stats.chi2.ppf(1.0 - quantile, proposal.d)
    analytic = -0.5 * (r2 + proposal.d * _LOG_2PI + proposal._logdet)
    assert t == pytest.approx(analytic, abs=0.02)


def test_density_threshold_keeps_only_in_cube_rows():
    proposal = GaussianProposal([0.02, 0.5], np.diag([0.05**2, 0.05**2]))

    t, kept_u, kept_logq = density_threshold(proposal, quantile=1e-2, n_support=20_000, rng=np.random.default_rng(2))

    assert np.all(in_cube(kept_u))
    # About a third of the draws leak through the u_0 = 0 face.
    assert 0.5 * 20_000 < len(kept_u) < 0.8 * 20_000
    assert t == pytest.approx(np.quantile(kept_logq, 1e-2))


def test_density_threshold_rejects_bad_inputs():
    far = GaussianProposal([5.0, 5.0], np.eye(2) * 0.01)
    with pytest.raises(ValueError, match="at least 10"):
        density_threshold(far, quantile=1e-3, n_support=1000, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="quantile"):
        density_threshold(_gaussian_2d(), quantile=0.0, n_support=1000, rng=np.random.default_rng(0))
    with pytest.raises(TypeError, match="Generator"):
        density_threshold(_gaussian_2d(), quantile=1e-3, n_support=1000, rng=0)


def test_sample_truncated_validates_arguments():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="method"):
        sample_truncated(_gaussian_2d(), n=10, quantile=1e-3, n_support=1000, method="rejection", rng=rng)
    with pytest.raises(ValueError, match="n must be"):
        sample_truncated(_gaussian_2d(), n=0, quantile=1e-3, n_support=1000, method="flow", rng=rng)
    with pytest.raises(TypeError, match="Generator"):
        sample_truncated(_gaussian_2d(), n=10, quantile=1e-3, n_support=1000, method="flow", rng=None)


# ---------------------------------------------------------------------------
# Flatness of the exact samplers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["flow", "ellipsoid"])
def test_exact_samplers_are_flat_on_a_gaussian_contour(method):
    proposal = _gaussian_2d()

    u, info = sample_truncated(
        proposal, n=20_000, quantile=1e-3, n_support=10_000, method=method, rng=np.random.default_rng(3)
    )

    assert u.shape == (20_000, 2) and u.dtype == np.float64
    assert info["method_used"] == method
    t = info["threshold"]
    assert np.all(in_cube(u)) and np.all(proposal.log_prob_raw(u) > t)
    assert info["fraction_outside_contour"] == 0.0 and info["fraction_outside_region"] == 0.0

    w = _whitened(proposal, u)
    assert _radial_pvalue(proposal, u, t) > P_MIN
    angle = (np.arctan2(w[:, 1], w[:, 0]) + np.pi) / (2.0 * np.pi)
    assert _chi2_uniform(angle, bins=12) > P_MIN
    assert _grid_pvalue(u, lambda x: in_cube(x) & (proposal.log_prob_raw(x) > t), np.random.default_rng(30)) > P_MIN


@pytest.mark.parametrize("method", ["flow", "ellipsoid"])
def test_exact_samplers_are_flat_on_a_mixture_contour(method):
    members = [
        GaussianProposal([0.4, 0.45], [[0.006, 0.002], [0.002, 0.005]], name="m0"),
        GaussianProposal([0.6, 0.55], [[0.004, -0.001], [-0.001, 0.007]], name="m1"),
    ]
    proposal = _Mixture(members, [0.6, 0.4])

    u, info = sample_truncated(
        proposal, n=20_000, quantile=1e-3, n_support=10_000, method=method, rng=np.random.default_rng(4)
    )

    assert info["method_used"] == method
    t = info["threshold"]
    assert np.all(in_cube(u)) and np.all(proposal.log_prob_raw(u) > t)
    assert _grid_pvalue(u, lambda x: in_cube(x) & (proposal.log_prob_raw(x) > t), np.random.default_rng(40)) > P_MIN


@pytest.mark.parametrize("method", ["flow", "ellipsoid"])
def test_exact_samplers_respect_the_region_exactly(method):
    proposal = _gaussian_2d()
    region = _HalfPlane(cut=0.5)

    u, info = sample_truncated(
        proposal,
        n=20_000,
        quantile=1e-3,
        n_support=10_000,
        region=region,
        method=method,
        rng=np.random.default_rng(5),
    )

    assert info["method_used"] == method
    t = info["threshold"]
    assert region.calls > 0
    assert np.all(u[:, 0] < 0.5)
    assert np.all(proposal.log_prob_raw(u) > t)
    inside = lambda x: in_cube(x) & (proposal.log_prob_raw(x) > t) & (x[:, 0] < 0.5)  # noqa: E731
    assert _grid_pvalue(u, inside, np.random.default_rng(50)) > P_MIN


def test_ellipsoid_shell_test_catches_a_face_pegged_gaussian():
    """A Gaussian centred on the u_0 = 0 face leaves a half-ball A.

    The ellipsoid fitted to the support draws is squeezed and shifted inward,
    so it clips the far tip of A. Fresh flow draws reach that tip as rarely as
    the support did, so the coverage check does not see it; the shell test
    does, and the grown ellipsoid gives a flat half-ball.
    """
    d, sigma = 4, 0.05
    proposal = GaussianProposal([0.0] + [0.5] * (d - 1), np.eye(d) * sigma**2)

    u, info = sample_truncated(
        proposal, n=20_000, quantile=1e-3, n_support=10_000, method="ellipsoid", rng=np.random.default_rng(0)
    )

    assert info["method_used"] == "ellipsoid"
    assert info["growths"] >= 1
    assert "shell" in info["growth_reasons"]
    assert "disconnected" not in info["growth_reasons"]
    assert info["growths"] == len(info["growth_reasons"])
    t = info["threshold"]
    assert np.all(in_cube(u)) and np.all(proposal.log_prob_raw(u) > t)
    # (r / r_t)^d is uniform on a half-ball exactly as on a ball.
    assert _radial_pvalue(proposal, u, t) > P_MIN
    # Each tangential axis is symmetric about the face-pegged mean.
    for j in range(1, d):
        assert stats.binomtest(int(np.sum(u[:, j] > 0.5)), len(u)).pvalue > P_MIN


# ---------------------------------------------------------------------------
# auto
# ---------------------------------------------------------------------------


def test_auto_picks_the_ellipsoid_for_a_gaussian():
    u, info = sample_truncated(
        _gaussian_2d(), n=5_000, quantile=1e-3, n_support=10_000, method="auto", rng=np.random.default_rng(6)
    )

    estimates = info["estimates"]
    assert info["method_used"] == "ellipsoid"
    assert estimates["members"] == 1
    assert estimates["cost_ellipsoid"] < estimates["cost_flow"]
    assert estimates["acceptance_ellipsoid"] > 10 * estimates["acceptance_flow"]
    assert u.shape == (5_000, 2)


def test_auto_counts_mixture_members():
    members = [GaussianProposal([0.45, 0.5], np.eye(2) * 0.004), GaussianProposal([0.55, 0.5], np.eye(2) * 0.004)]

    _, info = sample_truncated(
        _Mixture(members, [0.5, 0.5]),
        n=1_000,
        quantile=1e-3,
        n_support=10_000,
        method="auto",
        rng=np.random.default_rng(7),
    )

    estimates = info["estimates"]
    assert estimates["members"] == 2
    assert estimates["cost_flow"] == pytest.approx(1_000 / estimates["acceptance_flow"] * 12)
    assert estimates["cost_ellipsoid"] == pytest.approx(1_000 / estimates["acceptance_ellipsoid"] * 2)


def test_auto_picks_flow_for_a_banana():
    proposal = _Banana(d=6)

    u, info = sample_truncated(
        proposal, n=2_000, quantile=1e-3, n_support=10_000, method="auto", rng=np.random.default_rng(0)
    )

    estimates = info["estimates"]
    assert info["method_used"] == "flow"
    # Chosen up front, not reached through an ellipsoid growth.
    assert info["fallback"] is None and info["growths"] == 0
    assert estimates["cost_flow"] < estimates["cost_ellipsoid"]
    assert estimates["acceptance_ellipsoid"] < 0.2 * estimates["acceptance_flow"]
    assert u.shape == (2_000, 6)
    assert np.all(in_cube(u)) and np.all(proposal.log_prob_raw(u) > info["threshold"])

    _, forced = sample_truncated(
        proposal, n=200, quantile=1e-3, n_support=10_000, method="ellipsoid", rng=np.random.default_rng(1)
    )
    assert forced["method_used"] == "ellipsoid"
    assert forced["acceptance"] < 0.2 * info["acceptance"]


def test_banana_log_density_is_normalised():
    """The banana's log density is exact: its mass on a bounding box is 1."""
    proposal = _Banana(d=2, scale=0.05, bend=0.05)
    rng = np.random.default_rng(8)
    lo, hi = np.array([0.1, -0.1]), np.array([0.9, 1.2])
    u = lo + (hi - lo) * rng.random((400_000, 2))
    mass = np.mean(np.exp(proposal.log_prob_raw(u))) * np.prod(hi - lo)
    assert mass == pytest.approx(1.0, abs=0.02)


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def _forbid_sir(monkeypatch):
    import sbi.utils

    def refuse(*args, **kwargs):
        raise AssertionError("an exact sampler fell back to sbi's SIR")

    monkeypatch.setattr(sbi.utils, "RestrictedPrior", refuse)


@pytest.mark.parametrize("method", ["flow", "ellipsoid", "auto"])
def test_budget_escalates_the_quantile_then_raises(method, monkeypatch):
    _forbid_sir(monkeypatch)
    # 5 000 draws need ~6 400 ellipsoid candidates and at least ~110 000 flow
    # candidates (4.6% acceptance at quantile 1e-2): over a 1 000 budget everywhere.
    proposal = _gaussian_2d()

    with pytest.raises(TruncationBudgetError) as err:
        sample_truncated(
            proposal,
            n=5_000,
            quantile=1e-4,
            n_support=5_000,
            method=method,
            rng=np.random.default_rng(9),
            max_candidates=1_000,
        )

    assert err.value.quantiles == pytest.approx((1e-4, 1e-3, 1e-2))
    assert err.value.quantiles[-1] == QUANTILE_CAP
    assert "0.01" in str(err.value)


def test_budget_escalation_can_succeed_at_a_higher_quantile(monkeypatch):
    _forbid_sir(monkeypatch)
    proposal = _gaussian_2d()

    u, info = sample_truncated(
        proposal,
        n=500,
        quantile=1e-3,
        n_support=10_000,
        method="flow",
        rng=np.random.default_rng(10),
        max_candidates=30_000,
    )

    assert info["quantiles_tried"] == pytest.approx([1e-3, 1e-2])
    assert info["quantile_used"] == pytest.approx(1e-2)
    assert np.all(proposal.log_prob_raw(u) > info["threshold"])
    # The threshold was recomputed at the new quantile.
    r2 = stats.chi2.ppf(1.0 - 1e-2, 2)
    assert info["threshold"] == pytest.approx(-0.5 * (r2 + 2 * _LOG_2PI + proposal._logdet), abs=0.1)
    assert info["n_candidates"] > 30_000


def test_budget_above_the_cap_raises_without_escalation(monkeypatch):
    _forbid_sir(monkeypatch)

    with pytest.raises(TruncationBudgetError) as err:
        sample_truncated(
            _gaussian_2d(),
            n=5_000,
            quantile=0.05,
            n_support=5_000,
            method="flow",
            rng=np.random.default_rng(11),
            max_candidates=1_000,
        )

    assert err.value.quantiles == pytest.approx((0.05,))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["flow", "ellipsoid", "auto"])
def test_exact_samplers_are_deterministic(method):
    proposal = _gaussian_2d()
    kwargs = dict(n=3_000, quantile=1e-3, n_support=5_000, method=method, region=_HalfPlane(0.55))

    u1, info1 = sample_truncated(proposal, rng=np.random.default_rng(12), **kwargs)
    u2, info2 = sample_truncated(proposal, rng=np.random.default_rng(12), **kwargs)
    u3, _ = sample_truncated(proposal, rng=np.random.default_rng(13), **kwargs)

    np.testing.assert_array_equal(u1, u2)
    assert _info_without_timing(info1) == _info_without_timing(info2)
    assert not np.array_equal(u1, u3)


# ---------------------------------------------------------------------------
# sbi SIR path
# ---------------------------------------------------------------------------


def test_proxy_matches_what_sbi_sir_calls():
    import torch

    proposal = GaussianProposal([0.05, 0.5], np.eye(2) * 0.04**2)
    proxy = SbiProposalProxy(proposal, np.random.default_rng(14))

    draws = proxy.sample((4_000,), show_progress_bar=False)
    assert draws.dtype == torch.float32 and tuple(draws.shape) == (4_000, 2)
    assert np.all(in_cube(draws.numpy()))
    assert proxy.requests == [4_000]
    assert proxy.n_raw > 4_000  # a quarter of the raw draws leak through u_0 = 0

    log_prob = proxy.log_prob(draws)
    assert log_prob.dtype == torch.float32 and tuple(log_prob.shape) == (4_000,)
    np.testing.assert_allclose(
        log_prob.numpy(), proposal.log_prob_raw(draws.numpy().astype(np.float64)), rtol=1e-6
    )
    assert proxy.set_default_x(torch.zeros(3)) is proxy


def test_sir_uses_the_sbi_restricted_prior(monkeypatch):
    import torch
    import sbi.utils
    from sbi.utils import BoxUniform, RestrictedPrior

    calls = {}

    class SpyRestrictedPrior(RestrictedPrior):
        def __init__(self, prior, accept_reject_fn, posterior=None, sample_with="rejection", device="cpu"):
            calls["init"] = dict(prior=prior, fn=accept_reject_fn, posterior=posterior, sample_with=sample_with)
            super().__init__(prior, accept_reject_fn, posterior=posterior, sample_with=sample_with, device=device)

        def sample(self, sample_shape=torch.Size(), **kwargs):
            calls["sample"] = dict(shape=sample_shape, kwargs=kwargs)
            return super().sample(sample_shape, **kwargs)

    monkeypatch.setattr(sbi.utils, "RestrictedPrior", SpyRestrictedPrior)
    proposal = _gaussian_2d()
    region = _HalfPlane(0.6)
    n = 4_000

    u, info = sample_truncated(
        proposal, n=n, quantile=1e-3, n_support=10_000, region=region, method="sir", rng=np.random.default_rng(15)
    )

    init = calls["init"]
    assert init["sample_with"] == "sir"
    assert isinstance(init["prior"], BoxUniform)
    np.testing.assert_array_equal(init["prior"].base_dist.low.numpy(), np.zeros(2))
    np.testing.assert_array_equal(init["prior"].base_dist.high.numpy(), np.ones(2))
    assert isinstance(init["posterior"], SbiProposalProxy)
    assert init["posterior"].proposal is proposal
    assert calls["sample"]["shape"] == (n,)
    assert "oversampling_factor" not in calls["sample"]["kwargs"]

    # The accept function is the contour intersected with the region, as a bool tensor.
    probe = proposal.sample_raw(2_000, np.random.default_rng(16)).astype(np.float32)
    mask = init["fn"](torch.from_numpy(probe))
    assert mask.dtype == torch.bool
    expected = in_cube(probe) & (proposal.log_prob_raw(probe.astype(np.float64)) > info["threshold"])
    expected &= probe[:, 0] < 0.6
    np.testing.assert_array_equal(mask.numpy(), expected)

    assert u.shape == (n, 2) and u.dtype == np.float64
    assert np.all(in_cube(u))
    assert info["method_used"] == "sir"
    # sbi weighs a fixed 32 candidates per draw.
    assert init["posterior"].requests == [32 * n]
    assert info["n_candidates"] == 32 * n
    assert info["acceptance"] == pytest.approx(1.0 / 32)


def test_sir_draws_leave_the_contour_and_the_region():
    """Pin the v2 quirk: the 0/1 mask is a log potential, so SIR keeps rows outside A."""
    proposal = _gaussian_2d()
    region = _HalfPlane(0.5)

    u, info = sample_truncated(
        proposal, n=8_000, quantile=0.05, n_support=10_000, region=region, method="sir", rng=np.random.default_rng(17)
    )

    outside_contour = ~(proposal.log_prob_raw(u) > info["threshold"])
    outside_region = ~(u[:, 0] < 0.5)
    assert info["fraction_outside_contour"] == pytest.approx(np.mean(outside_contour))
    assert info["fraction_outside_region"] == pytest.approx(np.mean(outside_region))
    assert info["fraction_outside_contour"] > 0.01
    assert info["fraction_outside_region"] > 0.05
    # Still tilted toward A: e^1 vs e^0 in the potential.
    assert info["fraction_outside_region"] < 0.5


def test_sir_default_quantile_leaves_a_few_percent_outside_the_contour():
    u, info = sample_truncated(
        _gaussian_2d(), n=20_000, quantile=1e-3, n_support=10_000, method="sir", rng=np.random.default_rng(18)
    )

    assert 0.005 < info["fraction_outside_contour"] < 0.08
    assert info["fraction_outside_region"] == 0.0


def _unbalanced_mixture():
    """Two separated modes with weights 0.8 / 0.2, drawn grouped by member."""
    members = [GaussianProposal([0.35, 0.5], np.eye(2) * 0.003), GaussianProposal([0.65, 0.5], np.eye(2) * 0.003)]
    return _Mixture(members, [0.8, 0.2])


def _left_area_fraction(proposal, threshold, rng):
    """Fraction of the area of A with u_0 < 0.5 (the heavier member's mode)."""
    reference = rng.random((2_000_000, 2))
    inside = proposal.log_prob_raw(reference) > threshold
    return float(np.mean(reference[inside, 0] < 0.5))


def test_flow_prefixes_of_a_grouped_mixture_are_flat():
    """Small draws come from ONE truncated batch; unshuffled, its prefix favours member 0."""
    proposal = _unbalanced_mixture()
    rows = []
    for seed in range(30):
        u, info = sample_truncated(
            proposal, n=200, quantile=1e-3, n_support=5_000, method="flow", rng=np.random.default_rng(100 + seed)
        )
        rows.append(u)
    u = np.concatenate(rows, axis=0)

    # The threshold differs per call by its support noise only; its area split does not move.
    expected = _left_area_fraction(proposal, info["threshold"], np.random.default_rng(23))
    assert 0.5 < expected < 0.6
    assert stats.binomtest(int(np.sum(u[:, 0] < 0.5)), len(u), expected).pvalue > P_MIN


def test_sir_through_a_mixture_proxy():
    """SIR groups 32 consecutive candidates, so the proxy must shuffle member-grouped draws.

    Unshuffled, each group is one member's block and the output follows the
    member weights (80% in mode 0) instead of the area of A (about 55%).
    """
    proposal = _unbalanced_mixture()

    u, info = sample_truncated(
        proposal, n=6_000, quantile=1e-3, n_support=10_000, method="sir", rng=np.random.default_rng(19)
    )

    assert u.shape == (6_000, 2) and np.all(in_cube(u))
    expected = _left_area_fraction(proposal, info["threshold"], np.random.default_rng(24))
    assert abs(np.mean(u[:, 0] < 0.5) - expected) < 0.06
    assert info["fraction_outside_contour"] < 0.1


def test_sir_is_deterministic_and_leaves_the_torch_rng_alone():
    import torch

    state = torch.get_rng_state().clone()
    kwargs = dict(n=2_000, quantile=1e-3, n_support=5_000, method="sir")

    u1, _ = sample_truncated(_gaussian_2d(), rng=np.random.default_rng(20), **kwargs)
    u2, _ = sample_truncated(_gaussian_2d(), rng=np.random.default_rng(20), **kwargs)

    np.testing.assert_array_equal(u1, u2)
    assert torch.equal(torch.get_rng_state(), state)


def test_sir_oversampling_factor_is_inert_in_sbi_027():
    """`RestrictedPrior.sample(oversampling_factor=...)` never reaches SIR.

    sbi 0.27 hands the keyword to `sampling_importance_resampling`, whose
    `**kwargs` swallows it; the number of candidates per draw stays the
    hard-coded `num_candidate_samples=32`. If a future sbi honours the knob,
    this test fails and the SIXSA docs must change.
    """
    import torch
    from sbi.samplers.importance.sir import sampling_importance_resampling as sir
    from sbi.utils import BoxUniform, RestrictedPrior

    params = inspect.signature(sir).parameters
    assert "oversampling_factor" not in params
    assert params["num_candidate_samples"].default == 32
    assert "oversampling_factor" in inspect.signature(RestrictedPrior.sample).parameters

    proposal = _gaussian_2d()
    n = 1_000
    for factor in (1, 2, 1024):
        proxy = SbiProposalProxy(proposal, np.random.default_rng(21))
        restricted = RestrictedPrior(
            BoxUniform(torch.zeros(2), torch.ones(2)),
            lambda theta: torch.ones(theta.shape[0], dtype=torch.bool),
            posterior=proxy,
            sample_with="sir",
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            restricted.sample((n,), oversampling_factor=factor, show_progress_bars=False)
        assert proxy.requests == [32 * n]


def test_sir_log_summary_line():
    lines = []

    sample_truncated(
        _gaussian_2d(), n=500, quantile=1e-3, n_support=2_000, method="auto", rng=np.random.default_rng(22), log=lines.append
    )

    assert len(lines) == 1
    assert lines[0].startswith("[truncation] auto->")


def test_module_exposes_the_documented_methods():
    assert truncation.METHODS == ("sir", "flow", "ellipsoid", "auto")
    assert inspect.signature(sample_truncated).parameters["method"].default == "sir"


def test_ellipsoid_with_too_few_support_rows_falls_back_to_flow():
    proposal = _gaussian_2d()
    # The contour's tail left of u_0 = 0.2 holds a handful of the 10 000 support rows.
    region = _HalfPlane(0.2)

    u, info = sample_truncated(
        proposal, n=20, quantile=1e-3, n_support=10_000, region=region, method="ellipsoid", rng=np.random.default_rng(25)
    )

    assert info["method_used"] == "flow"
    assert info["fallback"] == "support"
    assert u.shape == (20, 2) and np.all(u[:, 0] < 0.2)
    assert np.all(proposal.log_prob_raw(u) > info["threshold"])


def test_module_is_xspec_free_and_torch_lazy():
    from test_import_hygiene import FORBIDDEN_MODULES, _collect_imports, _read, _violations

    relative_path = "bsixsa/backend/sixsa/truncation.py"
    imports = _collect_imports(_read(relative_path), relative_path)

    assert _violations(imports, FORBIDDEN_MODULES, module_level_only=False) == []
    assert _violations(imports, ("torch", "sbi"), module_level_only=True) == []
