"""Tests for SIXSASolver.build_inference_data (and build_dataframe for parity)."""

import sys
from unittest.mock import MagicMock, patch, PropertyMock

# Provide a fake xspec module so bsixsa.solver can be imported without HEASoft
_xspec_mock = MagicMock()
sys.modules.setdefault("xspec", _xspec_mock)

import arviz as az
import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from bsixsa.priors import MultipleIndependent
from bsixsa.backend.abc import Backend
from bsixsa.solver import Solver, FitResults, likelihood_per_bin


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_solver():
    """Create a SIXSASolver-like object without touching XSPEC."""

    param_names = ["powerlaw_PhoIndex", "powerlaw_norm"]
    n_params = len(param_names)

    # Build a real prior from uniform distributions
    from scipy.stats import uniform

    prior = MultipleIndependent([uniform(1, 2), uniform(0, 1)])

    # Create a fake "posterior" backend whose .sample() returns deterministic data
    rng = np.random.default_rng(42)
    posterior_samples = rng.random((20_000, n_params))

    fake_posterior = MagicMock(spec=Backend)
    fake_posterior.sample = MagicMock(
        side_effect=lambda n, **kw: posterior_samples[:n]
    )

    # Patch __init__ to avoid XSPEC calls, then manually set attributes
    with patch.object(Solver, "__init__", lambda self, *a, **kw: None):
        solver = Solver.__new__(Solver)

    solver.prior = prior
    solver.distributions = {"prior": prior, "posterior": fake_posterior}
    solver.indexes = list(range(1, n_params + 1))
    solver.model_indexes = ["powerlaw"] * n_params
    solver.bounds = [(1.0, 3.0), (0.0, 1.0)]
    solver.posterior_samples = None
    solver.backend_name = "mock"
    solver.backend = fake_posterior
    solver.fit_result = FitResults(
        time=1.0,
        posterior_samples=pd.DataFrame(),
        n_likelihood_evaluations=500,
        log_Z=-120.5,
        log_Z_err=0.3,
    )

    # Patch the property so it doesn't call XSPEC
    solver._parameter_names = param_names

    return solver


def _patched_parameter_names(solver):
    """Patch ``parameter_names`` property to avoid XSPEC dependency."""
    return solver._parameter_names


# ---------------------------------------------------------------------------
# Tests — build_dataframe (baseline parity check)
# ---------------------------------------------------------------------------


class TestBuildDataframe:

    def test_returns_dataframe(self, mock_solver):
        with patch.object(
            type(mock_solver), "parameter_names", new_callable=lambda: property(
                lambda self: _patched_parameter_names(self)
            ),
        ):
            df = mock_solver.build_dataframe(num_samples=100)

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (100, 2)
        assert list(df.columns) == mock_solver._parameter_names


# ---------------------------------------------------------------------------
# Tests — build_inference_data
# ---------------------------------------------------------------------------


class TestBuildInferenceData:

    @pytest.fixture(autouse=True)
    def _patch_param_names(self, mock_solver):
        """Auto-patch parameter_names for every test in this class."""
        self.patcher = patch.object(
            type(mock_solver),
            "parameter_names",
            new_callable=lambda: property(
                lambda self: _patched_parameter_names(self)
            ),
        )
        self.patcher.start()
        yield
        self.patcher.stop()

    def test_returns_inference_data(self, mock_solver):
        idata = mock_solver.build_inference_data(
            num_samples=100, include_log_likelihood=False,
        )
        assert isinstance(idata, az.InferenceData)

    def test_posterior_group_shape(self, mock_solver):
        n = 200
        idata = mock_solver.build_inference_data(
            num_samples=n, include_log_likelihood=False,
        )
        assert "posterior" in idata.groups()
        for name in mock_solver._parameter_names:
            assert name in idata.posterior
            assert idata.posterior[name].shape == (1, n)

    def test_prior_group_included_by_default(self, mock_solver):
        idata = mock_solver.build_inference_data(
            num_samples=50, include_log_likelihood=False,
        )
        assert "prior" in idata.groups()
        for name in mock_solver._parameter_names:
            assert name in idata.prior
            assert idata.prior[name].shape == (1, 50)

    def test_prior_group_excluded(self, mock_solver):
        idata = mock_solver.build_inference_data(
            num_samples=50, include_prior=False,
            include_log_likelihood=False,
        )
        assert "prior" not in idata.groups()

    def test_observed_data_excluded_when_xspec_unavailable(self, mock_solver):
        """observed_spectrum errors are surfaced to the caller."""
        with patch.object(
            type(mock_solver),
            "observed_spectrum",
            new_callable=lambda: property(
                lambda self: (_ for _ in ()).throw(RuntimeError("no XSPEC"))
            ),
        ):
            with pytest.raises(RuntimeError, match="no XSPEC"):
                mock_solver.build_inference_data(
                    num_samples=50,
                    include_log_likelihood=False,
                )

    def test_observed_data_included(self, mock_solver):
        fake_obs = np.ones(100, dtype=np.float32)
        with patch.object(
            type(mock_solver),
            "observed_spectrum",
            new_callable=lambda: property(lambda self: fake_obs),
        ):
            idata = mock_solver.build_inference_data(
                num_samples=50, include_log_likelihood=False,
            )
        assert "observed_data" in idata.groups()
        np.testing.assert_array_equal(
            idata.observed_data["spectrum"].values, fake_obs
        )

    def test_log_evidence_in_attrs(self, mock_solver):
        idata = mock_solver.build_inference_data(
            num_samples=50, include_log_likelihood=False,
        )
        assert idata.attrs["log_Z"] == pytest.approx(-120.5)
        assert idata.attrs["log_Z_err"] == pytest.approx(0.3)

    def test_no_fit_result(self, mock_solver):
        mock_solver.fit_result = None
        idata = mock_solver.build_inference_data(
            num_samples=50, include_observed=False,
            include_log_likelihood=False,
        )
        assert "log_Z" not in idata.attrs

    def test_consistency_with_build_dataframe(self, mock_solver):
        """Posterior samples should match those from build_dataframe."""
        n = 150
        df = mock_solver.build_dataframe(num_samples=n)
        idata = mock_solver.build_inference_data(
            num_samples=n, include_prior=False,
            include_observed=False,
            include_log_likelihood=False,
        )

        for name in mock_solver._parameter_names:
            np.testing.assert_array_equal(
                idata.posterior[name].values.squeeze(), df[name].values
            )

    def test_log_likelihood_group_included(self, mock_solver):
        """log_likelihood group is present with correct shape."""
        n = 80
        n_bins = 50
        fake_obs = np.random.default_rng(0).poisson(10, size=n_bins).astype(np.float32)
        fake_model_counts = np.random.default_rng(1).poisson(10, size=(n, n_bins)).astype(np.float64)

        with patch.object(
            type(mock_solver),
            "observed_spectrum",
            new_callable=lambda: property(lambda self: fake_obs),
        ), patch.object(
            mock_solver,
            "simulate",
            return_value={"total_model_counts": fake_model_counts},
        ):
            idata = mock_solver.build_inference_data(
                num_samples=n,
                include_prior=False,
                include_observed=False,
                include_log_likelihood=True,
            )

        assert "log_likelihood" in idata.groups()
        assert "spectrum" in idata.log_likelihood
        assert idata.log_likelihood["spectrum"].shape == (1, n, n_bins)

    def test_log_likelihood_values_correct(self, mock_solver):
        """log_likelihood values match manual poisson_deviance computation."""
        n = 30
        n_bins = 20
        rng = np.random.default_rng(99)
        fake_obs = rng.poisson(5, size=n_bins).astype(np.float64)
        fake_model = rng.poisson(5, size=(n, n_bins)).astype(np.float64) + 0.1

        expected = likelihood_per_bin(fake_obs, fake_model)

        with patch.object(
            type(mock_solver),
            "observed_spectrum",
            new_callable=lambda: property(lambda self: fake_obs),
        ), patch.object(
            mock_solver,
            "simulate",
            return_value={"total_model_counts": fake_model},
        ):
            idata = mock_solver.build_inference_data(
                num_samples=n,
                include_prior=False,
                include_observed=False,
                include_log_likelihood=True,
            )

        np.testing.assert_array_almost_equal(
            idata.log_likelihood["spectrum"].values.squeeze(), expected
        )

    def test_log_likelihood_excluded(self, mock_solver):
        """log_likelihood group absent when disabled."""
        idata = mock_solver.build_inference_data(
            num_samples=50, include_log_likelihood=False,
        )
        assert "log_likelihood" not in idata.groups()


# ---------------------------------------------------------------------------
# Tests — likelihood_per_bin
# ---------------------------------------------------------------------------


class TestLikelihoodPerBin:

    def test_shape(self):
        observed = np.array([10.0, 0.0, 5.0])
        model = np.array([[10.0, 1.0, 5.0], [8.0, 2.0, 7.0]])
        result = likelihood_per_bin(observed, model)
        assert result.shape == (2, 3)

    def test_known_values(self):
        """Check against SciPy's Poisson log-PMF implementation."""
        observed = np.array([10.0, 5.0, 3.0])
        model = np.array([[10.0, 5.0, 3.0]])
        result = likelihood_per_bin(observed, model)
        expected = poisson.logpmf(observed, model)
        np.testing.assert_array_almost_equal(result, expected)

    def test_zero_observed(self):
        observed = np.array([0.0])
        model = np.array([[5.0]])
        result = likelihood_per_bin(observed, model)
        np.testing.assert_almost_equal(result[0, 0], -5.0)

    def test_1d_model(self):
        """Works with 1-D model array (single sample)."""
        observed = np.array([10.0, 0.0, 5.0])
        model = np.array([10.0, 1.0, 5.0])
        result = likelihood_per_bin(observed, model)
        assert result.shape == (3,)

    def test_maximised_at_observed(self):
        """Log-likelihood is maximised when model equals observed."""
        rng = np.random.default_rng(7)
        observed = rng.poisson(10, size=50).astype(np.float64)
        observed = np.maximum(observed, 1)  # avoid y=0 where max is trivial
        model = rng.poisson(10, size=(100, 50)).astype(np.float64) + 0.1
        ll_model = likelihood_per_bin(observed, model)
        ll_saturated = likelihood_per_bin(observed, observed)
        assert np.all(ll_model <= ll_saturated + 1e-12)
