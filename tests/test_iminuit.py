from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np

sys.modules.setdefault("xspec", MagicMock())

from bsixsa.backend.config import Iminuit
from bsixsa.backend.iminuit import IminuitBackend
from bsixsa.priors import MultipleIndependent, loguniform, uniform


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
