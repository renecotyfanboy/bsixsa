from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.modules.setdefault("xspec", MagicMock())

from bsixsa.backend.config import LevenbergMarquardt
from bsixsa.backend.levenberg_marquart import (
    LevenbergMarquardtBackend,
    _delta_method_to_unconstrained,
    _log_prior_residuals,
    _poisson_deviance_residuals,
    covariance_from_jacobian,
)
from bsixsa.priors import MultipleIndependent, loguniform, norm, uniform


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _make_stub_solver(prior, *, output_dir, observed=None, simulate=None, n_bins=3):
    """Minimal stub solver exposing only what the LM backend actually touches."""
    n_params = prior.ndim
    solver = MagicMock()
    solver.num_parameters = n_params
    solver.parameter_names = [f"p{i}" for i in range(n_params)]
    solver.observed_spectrum = (
        observed if observed is not None else np.full(n_bins, 5.0)
    )
    solver.outputfiles_basename = str(output_dir)
    solver.prior = prior
    if simulate is None:
        simulate = MagicMock(
            return_value={
                "total_model_counts": np.full((1, n_bins), 5.0),
                "cstat": np.zeros(1),
            }
        )
    solver.simulate = simulate
    return solver


def _make_backend(prior, tmp_path, *, simulate=None, observed=None, n_bins=3, **config_kw):
    solver = _make_stub_solver(
        prior,
        output_dir=tmp_path,
        observed=observed,
        simulate=simulate,
        n_bins=n_bins,
    )
    config = LevenbergMarquardt(trace=False, **config_kw)
    return LevenbergMarquardtBackend(solver=solver, config=config)


# -----------------------------------------------------------------------------
# Existing tests (kept)
# -----------------------------------------------------------------------------

def test_poisson_deviance_residuals_match_closed_form_for_positive_bins():
    observed = np.array([3.0, 7.0])
    model = np.array([2.0, 10.0])

    residuals = _poisson_deviance_residuals(observed, model)

    expected = np.sign(observed - model) * np.sqrt(
        2.0 * (observed * np.log(observed / model) + model - observed)
    )
    np.testing.assert_allclose(residuals, expected)


def test_poisson_deviance_residuals_handle_zero_observed_counts():
    observed = np.array([0.0, 0.0])
    model = np.array([2.5, 0.0])

    residuals = _poisson_deviance_residuals(observed, model)

    expected = np.array([-np.sqrt(5.0), 0.0])
    np.testing.assert_allclose(residuals, expected)


def test_poisson_deviance_residuals_stay_finite_for_zero_model_counts():
    observed = np.array([4.0, 1.0])
    model = np.array([0.0, -3.0])

    residuals = _poisson_deviance_residuals(observed, model)

    assert np.all(np.isfinite(residuals))
    assert np.all(residuals > 0.0)


# -----------------------------------------------------------------------------
# New tests
# -----------------------------------------------------------------------------

def test_covariance_no_chi2_rescale():
    """Covariance equals (J^T J)^-1 directly — no χ²-rescaling for Poisson stats."""
    rng = np.random.default_rng(0)
    jac = rng.standard_normal((10, 3))
    expected = np.linalg.inv(jac.T @ jac)
    cov = covariance_from_jacobian(jac)
    np.testing.assert_allclose(cov, expected)


def test_covariance_for_zero_jacobian_is_finite_zero():
    """Zero Jacobian: the SVD-based pseudoinverse returns a finite zero matrix."""
    jac = np.zeros((5, 3))
    cov = covariance_from_jacobian(jac)
    assert cov.shape == (3, 3)
    assert np.all(np.isfinite(cov))
    np.testing.assert_array_equal(cov, np.zeros((3, 3)))


def test_covariance_for_rank_deficient_jacobian_is_finite_and_symmetric():
    """Rank-deficient Jacobian (duplicated column) still yields a finite symmetric covariance."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal((10, 2))
    jac = np.column_stack([base, base[:, 0]])  # rank 2 in a 3-column matrix

    cov = covariance_from_jacobian(jac)
    assert cov.shape == (3, 3)
    assert np.all(np.isfinite(cov))
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)


def test_unconstrained_round_trip_for_interior_values(tmp_path):
    """logit-of-CDF round-trip recovers interior physical values."""
    prior = MultipleIndependent([uniform(0.0, 1.0), uniform(-2.0, 5.0)])
    backend = _make_backend(prior, tmp_path, reparametrise=True)

    physical = np.array([0.4, 1.5])
    unconstrained = backend._physical_to_unconstrained(physical)
    physical_back = backend._unconstrained_to_physical(unconstrained).ravel()
    np.testing.assert_allclose(physical_back, physical, rtol=1e-12)


def test_run_uses_trf_with_bounds_when_reparametrise_false(tmp_path):
    """Default mode forwards method='trf', x_scale='jac', and prior bounds to scipy."""
    prior = MultipleIndependent([uniform(0.0, 1.0), uniform(-2.0, 5.0)])
    backend = _make_backend(prior, tmp_path, reparametrise=False)

    fake_result = MagicMock()
    fake_result.x = np.array([0.5, 1.5])
    fake_result.jac = np.eye(2)

    with patch(
        "bsixsa.backend.levenberg_marquart.least_squares",
        return_value=fake_result,
    ) as mock_ls:
        backend.run()

    _, kwargs = mock_ls.call_args
    assert kwargs["method"] == "trf"
    assert kwargs["x_scale"] == "jac"
    assert kwargs["diff_step"] == pytest.approx(1e-2)
    low, high = kwargs["bounds"]
    np.testing.assert_array_equal(low, [0.0, -2.0])
    np.testing.assert_array_equal(high, [1.0, 5.0])
    np.testing.assert_array_equal(backend.best_fit_params, fake_result.x)
    # Physical-space mode still populates the unconstrained-space equivalents
    # (via the delta method) so posterior samples stay inside the prior support
    # — see test_physical_mode_samples_stay_in_prior_support.
    assert backend.best_fit_unconstrained is not None
    assert backend.covariance_unconstrained is not None


def test_run_uses_lm_without_bounds_when_reparametrise_true(tmp_path):
    """Legacy mode forwards method='lm' and no bounds (lmder cannot handle them)."""
    prior = MultipleIndependent([uniform(0.0, 1.0), uniform(-2.0, 5.0)])
    backend = _make_backend(prior, tmp_path, reparametrise=True)

    fake_result = MagicMock()
    fake_result.x = np.array([0.0, 0.0])
    fake_result.jac = np.eye(2)

    with patch(
        "bsixsa.backend.levenberg_marquart.least_squares",
        return_value=fake_result,
    ) as mock_ls:
        backend.run()

    _, kwargs = mock_ls.call_args
    assert kwargs["method"] == "lm"
    assert "bounds" not in kwargs
    assert backend.best_fit_unconstrained is not None
    assert backend.covariance_unconstrained is not None


def test_fit_results_expose_best_fit_and_stat(tmp_path):
    """FitResults from the LM backend carries best_fit (Series) and best_fit_stat
    (float). Uniform priors map to XSPEC cons priors, so the bayes-on term is 0
    and best_fit_stat equals the bare cstat."""
    prior = MultipleIndependent([uniform(0.0, 1.0), uniform(-2.0, 5.0)])

    target_cstat = 42.0
    simulate = MagicMock(
        return_value={
            "total_model_counts": np.full((1, 3), 5.0),
            "cstat": np.array([target_cstat]),
        }
    )
    backend = _make_backend(
        prior, tmp_path, simulate=simulate, reparametrise=False
    )

    fake_result = MagicMock()
    fake_result.x = np.array([0.7, 2.5])
    fake_result.jac = np.eye(2)

    with patch(
        "bsixsa.backend.levenberg_marquart.least_squares",
        return_value=fake_result,
    ):
        fit_result = backend.run()

    assert isinstance(fit_result.best_fit, pd.Series)
    assert list(fit_result.best_fit.index) == backend.solver.parameter_names
    np.testing.assert_array_equal(fit_result.best_fit.values, [0.7, 2.5])
    assert fit_result.best_fit_stat == pytest.approx(target_cstat)


def test_sample_with_zero_covariance_returns_repeated_best_fit_rows(tmp_path):
    """A rank-deficient fit (zero covariance) produces samples sitting on the best fit."""
    prior = MultipleIndependent([uniform(0.0, 1.0), uniform(-2.0, 5.0)])
    backend = _make_backend(prior, tmp_path, reparametrise=False)
    backend.best_fit_params = np.array([0.3, 1.2])
    backend.best_fit_unconstrained = backend._physical_to_unconstrained(
        backend.best_fit_params
    )
    backend.covariance_unconstrained = np.zeros((2, 2))

    samples = backend.sample(100)
    assert samples.shape == (100, 2)
    np.testing.assert_allclose(
        samples, np.tile(backend.best_fit_params, (100, 1)), rtol=1e-12
    )


def test_max_log_pdf_per_axis_uniform():
    """For uniform(low, high), max log pdf is -log(high - low)."""
    prior = MultipleIndependent([uniform(0.0, 1.0), uniform(-2.0, 5.0)])
    shift = prior.max_log_pdf_per_axis()
    np.testing.assert_allclose(shift, [-np.log(1.0), -np.log(7.0)], rtol=1e-10)


def test_max_log_pdf_per_axis_loguniform():
    """For loguniform(low, high), the mode sits at x = low."""
    prior = MultipleIndependent([loguniform(0.1, 100.0)])
    shift = prior.max_log_pdf_per_axis()
    expected = float(prior.dists[0].logpdf(0.1))
    np.testing.assert_allclose(shift, [expected], rtol=1e-10)


def test_max_log_pdf_per_axis_truncnorm_interior_mode():
    """truncnorm with loc inside the support: mode sits at loc."""
    # truncnorm(a=-2, b=2, loc=0.3, scale=1.0) -> support (-1.7, 2.3), loc inside.
    prior = MultipleIndependent([norm(loc=0.3, scale=1.0, low=-2.0, high=2.0)])
    shift = prior.max_log_pdf_per_axis()
    expected = float(prior.dists[0].logpdf(0.3))
    np.testing.assert_allclose(shift, [expected], rtol=1e-12)


def test_max_log_pdf_per_axis_truncnorm_clipped_mode():
    """truncnorm with loc outside the support: mode clips to the closer boundary."""
    # truncnorm(a=1, b=2, loc=0.0, scale=1.0) -> support (1, 2), loc=0 below support.
    prior = MultipleIndependent([norm(loc=0.0, scale=1.0, low=1.0, high=2.0)])
    shift = prior.max_log_pdf_per_axis()
    expected = float(prior.dists[0].logpdf(1.0))
    np.testing.assert_allclose(shift, [expected], rtol=1e-12)


def test_log_prior_residuals_zero_for_uniform_prior():
    """Uniform priors contribute nothing to the LM residual vector."""
    prior = MultipleIndependent([uniform(0.0, 1.0), uniform(-2.0, 5.0)])
    shift = prior.max_log_pdf_per_axis()
    physical = np.array([0.3, 1.7])
    r_prior = _log_prior_residuals(prior, physical, shift)
    np.testing.assert_allclose(r_prior, [0.0, 0.0], atol=1e-12)


def test_log_prior_residuals_match_loguniform_closed_form():
    """For loguniform(low, high), r_prior(theta) = sqrt(2 * log(theta / low))."""
    prior = MultipleIndependent([loguniform(0.1, 100.0)])
    shift = prior.max_log_pdf_per_axis()
    theta = np.array([1.0])
    r = _log_prior_residuals(prior, theta, shift)
    expected = np.sqrt(2.0 * np.log(theta[0] / 0.1))
    np.testing.assert_allclose(r[0], expected, rtol=1e-10)


def test_log_prior_residual_gradient_matches_score_function():
    """At any theta, J_prior^T r_prior == -d log_prior / d theta (the LM gradient
    contribution is exactly the prior score, the standard MAP requirement)."""
    prior = MultipleIndependent([loguniform(0.1, 100.0)])
    shift = prior.max_log_pdf_per_axis()
    theta = np.array([3.0])

    # r_prior at theta and at theta+h, finite-difference J_prior wrt theta.
    h = 1e-6
    r0 = _log_prior_residuals(prior, theta, shift)
    rh = _log_prior_residuals(prior, theta + h, shift)
    j_prior = (rh - r0) / h  # shape (1,) — diagonal for independent priors.

    # Compare J_prior^T r_prior with -d log_pdf / d theta (which for
    # loguniform is +1/theta, since logpdf = -log(theta) + const).
    grad_residual_part = float(j_prior[0] * r0[0])
    expected_score = 1.0 / float(theta[0])
    np.testing.assert_allclose(grad_residual_part, expected_score, rtol=1e-3)


def test_xspec_bayes_contribution_uniform_is_zero():
    """uniform maps to XSPEC's cons prior, which contributes exactly 0."""
    prior = MultipleIndependent([uniform(0.0, 1.0), uniform(-2.0, 5.0)])
    assert prior.xspec_bayes_contribution([0.3, 1.7]) == 0.0


def test_xspec_bayes_contribution_loguniform_is_two_log_theta():
    """loguniform maps to XSPEC's jeffreys prior: -2*Log(prior) = 2*ln(theta),
    with no normalisation constant (unlike scipy's loguniform.logpdf)."""
    prior = MultipleIndependent([loguniform(0.1, 100.0)])
    np.testing.assert_allclose(
        prior.xspec_bayes_contribution([3.0]), 2.0 * np.log(3.0), rtol=1e-12
    )


def test_xspec_bayes_contribution_other_priors_fall_back_to_scipy_logpdf():
    """Priors without an XSPEC-exact mapping (e.g. truncnorm) contribute
    -2 * scipy logpdf. NOTE: this is NOT identical to XSPEC's gauss prior
    (untruncated, no truncation-mass term) — only uniform/loguniform are
    cross-checked against XSPEC."""
    dist = norm(loc=0.3, scale=2.0, low=-1.0, high=1.0)
    prior = MultipleIndependent([dist])
    expected = -2.0 * dist.logpdf(0.5)
    np.testing.assert_allclose(
        prior.xspec_bayes_contribution([0.5]), expected, rtol=1e-12
    )


def test_xspec_bayes_contribution_sums_over_axes():
    truncated = norm(loc=0.0, scale=1.0, low=-2.0, high=2.0)
    prior = MultipleIndependent(
        [uniform(0.0, 1.0), loguniform(0.1, 100.0), truncated]
    )
    theta = [0.5, 3.0, 0.7]
    expected = 2.0 * np.log(3.0) - 2.0 * truncated.logpdf(0.7)
    np.testing.assert_allclose(
        prior.xspec_bayes_contribution(theta), expected, rtol=1e-12
    )


def test_best_fit_stat_includes_xspec_bayes_term(tmp_path):
    """best_fit_stat = bare cstat + XSPEC bayes-on prior term at the best fit."""
    prior = MultipleIndependent([loguniform(0.1, 100.0)])

    target_cstat = 42.0
    simulate = MagicMock(
        return_value={
            "total_model_counts": np.full((1, 3), 5.0),
            "cstat": np.array([target_cstat]),
        }
    )
    backend = _make_backend(
        prior, tmp_path, simulate=simulate, reparametrise=False
    )

    fake_result = MagicMock()
    fake_result.x = np.array([3.0])
    fake_result.jac = np.eye(1)

    with patch(
        "bsixsa.backend.levenberg_marquart.least_squares",
        return_value=fake_result,
    ):
        fit_result = backend.run()

    assert fit_result.best_fit_stat == pytest.approx(
        target_cstat + 2.0 * np.log(3.0)
    )


def test_delta_method_unconstrained_covariance_for_uniform_prior():
    """Delta-method pushforward matches the closed form for a uniform prior."""
    # uniform(0, 1) at theta=0.5: CDF=0.5, pdf=1.0  -> du/dtheta = 1/(0.5*0.5) = 4
    # uniform(-2, 5) at theta=1.5: CDF=0.5, pdf=1/7 -> du/dtheta = (1/7)/0.25 = 4/7
    prior = MultipleIndependent([uniform(0.0, 1.0), uniform(-2.0, 5.0)])
    params_phys = np.array([0.5, 1.5])
    cov_phys = np.eye(2)

    cov_unc = _delta_method_to_unconstrained(prior, params_phys, cov_phys)

    expected_diag = np.array([16.0, (4.0 / 7.0) ** 2])
    np.testing.assert_allclose(np.diag(cov_unc), expected_diag, rtol=1e-12)
    # Off-diagonals stay zero for a diagonal pushforward of a diagonal Sigma_phys.
    np.testing.assert_allclose(cov_unc[0, 1], 0.0, atol=1e-15)
    np.testing.assert_allclose(cov_unc[1, 0], 0.0, atol=1e-15)


def test_physical_mode_samples_stay_in_prior_support(tmp_path):
    """Posterior samples in physical-space mode are bounded by the prior support."""
    prior = MultipleIndependent([uniform(0.0, 1.0), uniform(-2.0, 5.0)])
    backend = _make_backend(prior, tmp_path, reparametrise=False)

    # Best fit near a boundary, plus a covariance wide enough to leak out
    # under naive physical-space Gaussian sampling.
    fake_result = MagicMock()
    fake_result.x = np.array([0.05, 4.8])
    # JJ^T = 0.01 * I  =>  inv = 100 * I  =>  std ~ 10 in physical space.
    fake_result.jac = 0.1 * np.eye(2)
    fake_result.cost = 0.0

    with patch(
        "bsixsa.backend.levenberg_marquart.least_squares",
        return_value=fake_result,
    ):
        fit_result = backend.run()

    samples = fit_result.posterior_samples.values
    assert samples.shape[0] == 10_000
    # Every sample sits inside the prior support — the unconstrained-space
    # Gaussian + sigmoid + inverse-CDF cannot escape it by construction.
    assert np.all(samples[:, 0] >= 0.0) and np.all(samples[:, 0] <= 1.0)
    assert np.all(samples[:, 1] >= -2.0) and np.all(samples[:, 1] <= 5.0)
