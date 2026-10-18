from __future__ import annotations

import sys
import warnings
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault("xspec", MagicMock())

from bsixsa.backend.config import Iminuit  # noqa: E402
from bsixsa.backend.iminuit import IminuitBackend  # noqa: E402
from bsixsa.priors import MultipleIndependent, loguniform, norm, uniform  # noqa: E402


def _make_iminuit_backend(prior, tmp_path, simulate_fn, **config_kw):
    solver = MagicMock()
    solver.num_parameters = prior.ndim
    solver.parameter_names = [f"p{i}" for i in range(prior.ndim)]
    solver.outputfiles_basename = str(tmp_path)
    solver.prior = prior
    solver.simulate = simulate_fn
    config = Iminuit(trace=False, **config_kw)
    return IminuitBackend(solver=solver, config=config)


def test_iminuit_objective_equals_cstat_under_uniform_prior(tmp_path):
    """Sanity: with a uniform prior, ``-2 log_prior`` is a constant offset, so the
    MAP that MIGRAD lands on matches the MLE."""
    prior = MultipleIndependent([uniform(-3.0, 3.0)])

    def simulate_fn(parameters, return_kind="cstat", progress_bar=False):
        theta = float(np.atleast_2d(parameters)[0, 0])
        return {"cstat": np.array([(theta - 1.2) ** 2])}

    backend = _make_iminuit_backend(
        prior,
        tmp_path,
        simulate_fn,
        x0_physical=np.array([0.0]),
    )
    result = backend.run()
    np.testing.assert_allclose(result.best_fit["p0"], 1.2, atol=1e-3)


def test_iminuit_map_shifts_toward_loguniform_prior_mode(tmp_path):
    """MIGRAD finds the MAP, not the MLE, when the prior is informative.

    Stub the C-stat as a quadratic with MLE at theta=5.  Combined with a
    ``loguniform(0.1, 100)`` prior (score ``-d log pi / d theta = 1/theta``),
    the MAP condition is

        2 (theta - 5) + 2 / theta = 0   <=>   theta^2 - 5 theta + 1 = 0,

    which has positive-curvature root  ``theta = (5 + sqrt(21)) / 2 ~ 4.7913``.
    A backend that minimised the bare likelihood would land at 5.
    """
    prior = MultipleIndependent([loguniform(0.1, 100.0)])

    def simulate_fn(parameters, return_kind="cstat", progress_bar=False):
        theta = float(np.atleast_2d(parameters)[0, 0])
        return {"cstat": np.array([(theta - 5.0) ** 2])}

    backend = _make_iminuit_backend(
        prior,
        tmp_path,
        simulate_fn,
        x0_physical=np.array([3.0]),
    )
    result = backend.run()

    expected_map = 0.5 * (5.0 + np.sqrt(21.0))
    np.testing.assert_allclose(result.best_fit["p0"], expected_map, rtol=1e-3)
    # The prior actually shifts the optimum away from the MLE (theta=5).
    assert result.best_fit["p0"] < 5.0 - 1e-2


def test_iminuit_precision_config_is_applied(tmp_path):
    """config.precision propagates to Minuit.precision (FD step sizing for
    noisy tabulated-model objectives)."""
    prior = MultipleIndependent([uniform(-3.0, 3.0)])

    def simulate_fn(parameters, return_kind="cstat", progress_bar=False):
        theta = float(np.atleast_2d(parameters)[0, 0])
        return {"cstat": np.array([(theta - 1.2) ** 2])}

    backend = _make_iminuit_backend(
        prior,
        tmp_path,
        simulate_fn,
        x0_physical=np.array([0.0]),
        precision=1e-5,
    )
    backend.run()
    assert backend.minuit.precision == 1e-5


def test_iminuit_best_fit_stat_includes_xspec_bayes_term(tmp_path):
    """best_fit_stat = bare cstat + XSPEC bayes-on prior term at the best fit
    (for loguniform/jeffreys: + 2*ln(theta_hat))."""
    prior = MultipleIndependent([loguniform(0.1, 100.0)])

    def simulate_fn(parameters, return_kind="cstat", progress_bar=False):
        theta = float(np.atleast_2d(parameters)[0, 0])
        return {"cstat": np.array([(theta - 5.0) ** 2])}

    backend = _make_iminuit_backend(
        prior,
        tmp_path,
        simulate_fn,
        x0_physical=np.array([3.0]),
    )
    result = backend.run()

    theta_hat = float(result.best_fit["p0"])
    expected = (theta_hat - 5.0) ** 2 + 2.0 * np.log(theta_hat)
    np.testing.assert_allclose(result.best_fit_stat, expected, rtol=1e-10)


def test_iminuit_default_start_is_exactly_the_prior_median_in_logit_space(tmp_path):
    """Without x0_physical, MIGRAD starts at u = 0 exactly (keeping its default 0.1 step),
    not at logit(F(median)), which is ~1e-16 for many loguniform priors."""
    prior = MultipleIndependent([loguniform(0.1, 100.0), uniform(-3.0, 3.0)])

    def simulate_fn(parameters, return_kind="cstat", progress_bar=False):
        theta = np.atleast_2d(parameters)[0]
        return {"cstat": np.array([(np.log(theta[0]) - 1.0) ** 2 + theta[1] ** 2])}

    backend = _make_iminuit_backend(prior, tmp_path, simulate_fn)
    result = backend.run()

    assert [parameter.value for parameter in backend.minuit.init_params] == [0.0, 0.0]
    assert result.n_likelihood_evaluations == backend.minuit.nfcn
    samples = backend.sample(50)
    assert samples.shape == (50, 2)


def test_iminuit_warns_about_a_parameter_pegged_on_a_prior_bound(tmp_path):
    """The likelihood pulls p0 beyond its upper bound: MIGRAD stops at u ~ 14 (CDF ~ 1 - 1e-6,
    not on the bound), but the Gaussian draws pile up on the bounds, so run() warns for p0 only."""
    prior = MultipleIndependent([uniform(0.0, 1.0), loguniform(0.1, 10.0)])

    def simulate_fn(parameters, return_kind="cstat", progress_bar=False):
        theta = np.atleast_2d(parameters)[0]
        return {"cstat": np.array([100.0 * (theta[0] - 2.0) ** 2 + 50.0 * (np.log(theta[1]) - 0.5) ** 2])}

    backend = _make_iminuit_backend(prior, tmp_path, simulate_fn)
    with pytest.warns(UserWarning, match="p0 is at its prior bound") as caught:
        backend.run()

    assert not [w for w in caught if "p1" in str(w.message)]


def test_iminuit_best_fit_far_in_a_gaussian_prior_tail_is_not_a_prior_bound(tmp_path):
    """Regression: a best fit ~8.7 prior sigmas below a Gaussian prior's mean (CDF ~1e-18,
    below float eps but far from the truncation bound 1.0) was reported as on a prior bound."""
    prior = MultipleIndependent([norm(loc=2.75, scale=0.1, low=1.0, high=4.0), uniform(-3.0, 3.0)])

    def simulate_fn(parameters, return_kind="cstat", progress_bar=False):
        theta = np.atleast_2d(parameters)[0]
        return {"cstat": np.array([((theta[0] - 1.8436) / 0.02) ** 2 + theta[1] ** 2])}

    backend = _make_iminuit_backend(prior, tmp_path, simulate_fn)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = backend.run()

    assert not [w for w in caught if "prior bound" in str(w.message)]
    assert prior.to_unit_cube(result.best_fit.to_numpy()[None])[0, 0] < np.finfo(np.float64).eps
    draws = result.posterior_samples["p0"].to_numpy()
    assert draws.mean() == pytest.approx(result.best_fit["p0"], abs=0.005)
    assert draws.std() > 0.01
