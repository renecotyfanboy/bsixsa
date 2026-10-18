"""Self-checks of the shared analytic fixtures in `tests/_analytic.py`."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from scipy.stats import chi2, multivariate_normal

from _analytic import (
    AnalyticSpectrum,
    GaussianProposal,
    QuadCstat,
    check_grid_convergence,
    grid_truth,
    log_integral_grid,
    marginal_cdf,
)


# ---------------------------------------------------------------------------
# AnalyticSpectrum
# ---------------------------------------------------------------------------


def test_spectrum_layout_and_normalisation():
    spectrum = AnalyticSpectrum(d=2)

    assert spectrum.edges.shape == (65,)
    assert spectrum.edges[0] == pytest.approx(0.5) and spectrum.edges[-1] == pytest.approx(10.0)
    np.testing.assert_allclose(np.diff(np.log(spectrum.edges)), np.log(20.0) / 64)
    assert spectrum.mu_true.sum() == pytest.approx(4000.0)
    assert spectrum.observed.dtype == np.float64
    assert np.all(spectrum.observed == np.round(spectrum.observed))
    assert abs(spectrum.observed.sum() - 4000.0) < 5 * np.sqrt(4000.0)
    assert spectrum.parameter_names == ["Gamma", "norm"]
    assert AnalyticSpectrum(d=3).parameter_names == ["Gamma", "norm", "nH"]


def test_spectrum_is_seeded():
    np.testing.assert_array_equal(AnalyticSpectrum(seed=7).observed, AnalyticSpectrum(seed=7).observed)
    assert not np.array_equal(AnalyticSpectrum(seed=7).observed, AnalyticSpectrum(seed=8).observed)


def test_spectrum_d2_fixes_nh_to_truth():
    two = AnalyticSpectrum(d=2)
    three = AnalyticSpectrum(d=3)
    theta2 = np.array([[2.1, 0.7], [1.3, 3.0]])
    theta3 = np.column_stack([theta2, np.full(2, 0.3)])

    np.testing.assert_allclose(two.model_counts(theta2), three.model_counts(theta3))
    with pytest.raises(ValueError):
        two.model_counts(theta3)
    with pytest.raises(ValueError):
        AnalyticSpectrum(d=4)


def test_cash_is_vectorised_and_handles_zero_counts():
    spectrum = AnalyticSpectrum(d=2)
    theta = np.array([[1.8, 1.0], [2.5, 0.3], [1.2, 6.0]])
    mu = spectrum.model_counts(theta)

    expected = np.array([spectrum.cash(row, spectrum.observed) for row in mu])
    np.testing.assert_allclose(spectrum.cstat(theta), expected)

    k = np.array([0.0, 2.0, 5.0])
    m = np.array([1.5, 2.0, 4.0])
    manual = 2.0 * (1.5 + 0.0 + (4.0 - 5.0 + 5.0 * np.log(5.0 / 4.0)))
    assert spectrum.cash(m, k) == pytest.approx(manual)


def test_prior_list_families_and_supports():
    families = [dist.dist.name for dist in AnalyticSpectrum(d=3).prior_list()]
    supports = [tuple(dist.support()) for dist in AnalyticSpectrum(d=3).prior_list()]

    assert families == ["uniform", "loguniform", "uniform"]
    np.testing.assert_allclose(supports, [(1.0, 3.0), (0.1, 10.0), (0.0, 1.5)])


# ---------------------------------------------------------------------------
# grid_truth
# ---------------------------------------------------------------------------


def test_grid_truth_d2_self_check_converges():
    check = check_grid_convergence(AnalyticSpectrum(d=2), n_grid=512)

    assert check["ok"], check
    assert check["delta"] <= 1e-3
    assert np.isfinite(check["log_z_fine"])


def test_grid_truth_d2_posterior_is_consistent():
    spectrum = AnalyticSpectrum(d=2)
    truth = grid_truth(spectrum)

    assert truth["n_grid"] == 512
    assert abs(truth["mean_theta"][0] - 1.8) < 0.1
    assert abs(truth["mean_theta"][1] - 1.0) < 0.1
    assert np.all(truth["std_theta"] > 0) and np.all(truth["std_u"] > 0)
    np.testing.assert_allclose(np.diag(truth["corr_theta"]), 1.0)
    np.testing.assert_allclose(np.diag(truth["corr_u"]), 1.0)

    # The minimum is below the statistic at the truth and at the grid MAP.
    assert truth["c_min"] <= float(spectrum.cstat(spectrum.truth)[0])
    assert truth["c_min"] == pytest.approx(float(spectrum.cstat_u(truth["map_u"])[0]))
    np.testing.assert_allclose(spectrum.prior().to_unit_cube(truth["map_theta"]), truth["map_u"])

    # With ~4000 counts the cube posterior is close to Gaussian, so the HPD
    # C-stat levels sit near the chi-square quantiles above the minimum.
    for level in (0.68, 0.95):
        delta = truth["hpd_cstat"][level] - truth["c_min"]
        assert delta == pytest.approx(chi2.ppf(level, df=2), rel=0.05)
        assert truth["hpd_density"][level] == pytest.approx(np.exp(-0.5 * truth["hpd_cstat"][level] - truth["log_z"]))
    assert truth["hpd_density"][0.68] > truth["hpd_density"][0.95]

    for axis in range(2):
        assert marginal_cdf(truth, axis, truth["window_lo"][axis]) == pytest.approx(0.0)
        assert marginal_cdf(truth, axis, truth["window_hi"][axis]) == pytest.approx(1.0)
        assert marginal_cdf(truth, axis, truth["mean_u"][axis]) == pytest.approx(0.5, abs=0.05)


def test_grid_truth_log_z_matches_laplace_order_of_magnitude():
    spectrum = AnalyticSpectrum(d=2)
    truth = grid_truth(spectrum)

    cov_u = truth["corr_u"] * np.outer(truth["std_u"], truth["std_u"])
    laplace = -0.5 * truth["c_min"] + np.log(2 * np.pi) + 0.5 * np.linalg.slogdet(cov_u)[1]
    assert truth["log_z"] == pytest.approx(laplace, abs=0.05)


def test_grid_truth_is_cached():
    spectrum = AnalyticSpectrum(d=2)

    first = grid_truth(spectrum, 256)
    second = grid_truth(spectrum, 256)

    assert first is not second
    assert first["log_z"] == second["log_z"]


@pytest.mark.slow
def test_grid_truth_d3_self_check_converges():
    spectrum = AnalyticSpectrum(d=3)
    check = check_grid_convergence(spectrum, n_grid=128)
    truth = grid_truth(spectrum)

    assert check["ok"], check
    assert abs(truth["mean_theta"][0] - 1.8) < 0.15


# ---------------------------------------------------------------------------
# QuadCstat
# ---------------------------------------------------------------------------


def _quad(d=2, c0=37.2):
    if d == 2:
        std = np.array([0.03, 0.05])
        corr = np.array([[1.0, 0.6], [0.6, 1.0]])
        mu = np.array([0.45, 0.55])
    else:
        std = np.array([0.03, 0.05, 0.04])
        corr = np.array([[1.0, 0.4, -0.2], [0.4, 1.0, 0.3], [-0.2, 0.3, 1.0]])
        mu = np.array([0.45, 0.55, 0.5])
    cov = corr * np.outer(std, std)
    return QuadCstat(mu, np.linalg.inv(cov), c0=c0)


def test_quad_evidence_matches_grid_integration():
    quad = _quad()

    grid = log_integral_grid(quad, np.zeros(2), np.ones(2), n_grid=1024)

    assert grid == pytest.approx(quad.log_evidence_unbounded(), abs=1e-3)


def test_quad_basic_identities():
    quad = _quad()

    assert quad(quad.mu) == pytest.approx(quad.c0)
    np.testing.assert_allclose(quad.posterior_cov, np.linalg.inv(quad.P))
    assert quad.level_volume(quad.c0) == 0.0
    lo, hi = quad.aabb(quad.c0 + 4.0)
    np.testing.assert_allclose(0.5 * (lo + hi), quad.mu)
    np.testing.assert_allclose(0.5 * (hi - lo), 2.0 * np.sqrt(np.diag(quad.posterior_cov)))
    with pytest.raises(ValueError):
        QuadCstat([0.5, 0.5], [[1.0, 2.0], [2.0, 1.0]])


@pytest.mark.parametrize("d", [2, 3])
def test_quad_level_volume_matches_monte_carlo(d):
    quad = _quad(d)
    t = quad.c0 + 6.0
    lo, hi = quad.aabb(t)
    assert np.all(lo > 0) and np.all(hi < 1)

    n = 400_000
    u = np.random.default_rng(11).random((n, d))
    fraction = np.mean(quad(u) <= t)
    sigma = np.sqrt(fraction * (1 - fraction) / n)

    assert abs(fraction - quad.level_volume(t)) <= 3 * sigma


@pytest.mark.parametrize("d", [2, 3])
def test_quad_level_set_samples_are_uniform_in_the_ellipsoid(d):
    quad = _quad(d)
    t = quad.c0 + 5.0
    samples = quad.sample_level_set(t, 100_000, np.random.default_rng(5))
    values = quad(samples)
    lo, hi = quad.aabb(t)

    assert np.all(values <= t + 1e-9)
    assert np.all(samples >= lo - 1e-12) and np.all(samples <= hi + 1e-12)
    np.testing.assert_allclose(samples.mean(axis=0), quad.mu, atol=5e-3)
    # Uniformity: the inner level set holds its volume fraction of the draws.
    inner = quad.c0 + 0.5 * (t - quad.c0)
    expected = quad.level_volume(inner) / quad.level_volume(t)
    assert np.mean(values <= inner) == pytest.approx(expected, abs=0.01)


def test_quad_posterior_sample_covariance():
    quad = _quad(3)
    samples = quad.posterior_sample(200_000, np.random.default_rng(3))

    np.testing.assert_allclose(samples.mean(axis=0), quad.mu, atol=1e-3)
    np.testing.assert_allclose(np.cov(samples.T), quad.posterior_cov, rtol=0.03, atol=1e-6)


# ---------------------------------------------------------------------------
# GaussianProposal
# ---------------------------------------------------------------------------


def test_gaussian_proposal_raw_protocol():
    mean = np.array([0.1, 0.8])
    cov = np.array([[0.04, 0.01], [0.01, 0.02]])
    proposal = GaussianProposal(mean, cov, name="q")
    draws = proposal.sample_raw(50_000, np.random.default_rng(0))

    assert draws.shape == (50_000, 2)
    assert np.any(draws < 0) or np.any(draws > 1)
    np.testing.assert_allclose(draws.mean(axis=0), mean, atol=5e-3)

    log_q = proposal.log_prob_raw(draws[:100])
    assert log_q.dtype == np.float64 and log_q.shape == (100,)
    np.testing.assert_allclose(log_q, multivariate_normal(mean, cov).logpdf(draws[:100]))
    assert proposal.log_prob_raw(mean) == pytest.approx(multivariate_normal(mean, cov).logpdf(mean))


def test_gaussian_proposal_in_cube_mass_analytic_matches_monte_carlo():
    proposal = GaussianProposal([0.15, 0.7], np.diag([0.1**2, 0.25**2]))

    analytic = proposal.in_cube_mass()
    mc = proposal.in_cube_mass(method="mc")
    sigma = np.sqrt(analytic * (1 - analytic) / GaussianProposal.LEAKAGE_MC_DRAWS)

    assert 0.3 < analytic < 0.95
    assert analytic == proposal.in_cube_mass(method="analytic")
    assert abs(analytic - mc) <= 4 * sigma


def test_gaussian_proposal_leakage_is_the_mass_outside_the_cube():
    """Regression: `leakage()` used to return the in-cube mass, the opposite of its name."""
    proposal = GaussianProposal([0.15, 0.7], np.diag([0.1**2, 0.25**2]))
    # Independent reference: the x-marginal alone leaks P(u < 0) = Phi(-1.5).
    x_inside = stats.norm.cdf(0.85 / 0.1) - stats.norm.cdf(-1.5)
    y_inside = stats.norm.cdf(0.3 / 0.25) - stats.norm.cdf(-0.7 / 0.25)

    assert proposal.in_cube_mass() == pytest.approx(x_inside * y_inside, rel=1e-12)
    assert proposal.leakage() == pytest.approx(1.0 - x_inside * y_inside, rel=1e-12)
    assert proposal.leakage(method="mc") == 1.0 - proposal.in_cube_mass(method="mc")
    # Nearly all the mass of a tight Gaussian at the centre stays inside.
    assert GaussianProposal([0.5, 0.5], np.diag([1e-4, 1e-4])).leakage() < 1e-12


def test_gaussian_proposal_in_cube_mass_uses_monte_carlo_when_correlated():
    proposal = GaussianProposal([0.5, 0.5], [[0.09, 0.05], [0.05, 0.09]])

    assert not proposal.is_diagonal
    assert proposal.in_cube_mass() == proposal.in_cube_mass(method="mc")
    with pytest.raises(ValueError, match="diagonal"):
        proposal.in_cube_mass(method="analytic")
    with pytest.raises(ValueError, match="diagonal"):
        proposal.leakage(method="analytic")
    # A correlated Gaussian keeps more mass in the cube than its decorrelated twin.
    assert proposal.in_cube_mass() > GaussianProposal([0.5, 0.5], np.diag([0.09, 0.09])).in_cube_mass()
