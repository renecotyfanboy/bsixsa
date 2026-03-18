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

from bsixsa.priors import MultipleIndependent
from bsixsa.backend.abc import Backend
from bsixsa.solver import SIXSASolver, FitResults


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
    with patch.object(SIXSASolver, "__init__", lambda self, *a, **kw: None):
        solver = SIXSASolver.__new__(SIXSASolver)

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
            num_samples=100
        )
        assert isinstance(idata, az.InferenceData)

    def test_posterior_group_shape(self, mock_solver):
        n = 200
        idata = mock_solver.build_inference_data(
            num_samples=n
        )
        assert "posterior" in idata.groups()
        for name in mock_solver._parameter_names:
            assert name in idata.posterior
            assert idata.posterior[name].shape == (1, n)

    def test_prior_group_included_by_default(self, mock_solver):
        idata = mock_solver.build_inference_data(
            num_samples=50
        )
        assert "prior" in idata.groups()
        for name in mock_solver._parameter_names:
            assert name in idata.prior
            assert idata.prior[name].shape == (1, 50)

    def test_prior_group_excluded(self, mock_solver):
        idata = mock_solver.build_inference_data(
            num_samples=50, include_prior=False
        )
        assert "prior" not in idata.groups()

    def test_observed_data_excluded_when_xspec_unavailable(self, mock_solver):
        """observed_spectrum raises → observed_data group is simply absent."""
        with patch.object(
            type(mock_solver),
            "observed_spectrum",
            new_callable=lambda: property(
                lambda self: (_ for _ in ()).throw(RuntimeError("no XSPEC"))
            ),
        ):
            idata = mock_solver.build_inference_data(
                num_samples=50
            )
        assert "observed_data" not in idata.groups()

    def test_observed_data_included(self, mock_solver):
        fake_obs = np.ones(100, dtype=np.float32)
        with patch.object(
            type(mock_solver),
            "observed_spectrum",
            new_callable=lambda: property(lambda self: fake_obs),
        ):
            idata = mock_solver.build_inference_data(
                num_samples=50
            )
        assert "observed_data" in idata.groups()
        np.testing.assert_array_equal(
            idata.observed_data["spectrum"].values, fake_obs
        )

    def test_log_evidence_in_attrs(self, mock_solver):
        idata = mock_solver.build_inference_data(
            num_samples=50
        )
        assert idata.attrs["log_Z"] == pytest.approx(-120.5)
        assert idata.attrs["log_Z_err"] == pytest.approx(0.3)

    def test_no_fit_result(self, mock_solver):
        mock_solver.fit_result = None
        idata = mock_solver.build_inference_data(
            num_samples=50, include_observed=False
        )
        assert "log_Z" not in idata.attrs

    def test_consistency_with_build_dataframe(self, mock_solver):
        """Posterior samples should match those from build_dataframe."""
        n = 150
        df = mock_solver.build_dataframe(num_samples=n)
        idata = mock_solver.build_inference_data(
            num_samples=n, include_prior=False,
            include_observed=False,
        )

        for name in mock_solver._parameter_names:
            np.testing.assert_array_equal(
                idata.posterior[name].values.squeeze(), df[name].values
            )
