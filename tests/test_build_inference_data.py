"""Tests for Solver.build_inference_data (and build_dataframe for parity)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import arviz as az
import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

import bsixsa.solver as solver_module
import bsixsa.xspec as xspec_module
from bsixsa import Nessai
from bsixsa.priors import MultipleIndependent
from bsixsa.backend.abc import Backend
from bsixsa.solver import Solver, FitResults, likelihood_per_bin


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PARAMETER_NAMES = ["powerlaw_PhoIndex", "powerlaw_norm"]


def _solver_without_xspec(tmp_path, posterior_sample):
    """A `Solver` initialised by `_setup` only (no XSPEC), with a fake posterior backend."""
    from scipy.stats import uniform

    prior = MultipleIndependent([uniform(1, 2), uniform(0, 1)])
    fake_posterior = MagicMock(spec=Backend)
    fake_posterior.sample = MagicMock(side_effect=posterior_sample)

    solver = Solver.__new__(Solver)
    solver._setup(
        prior=prior,
        parameter_indices=[1, 2],
        model_names=["", ""],
        parameter_names=PARAMETER_NAMES,
        outputfiles_basename=str(tmp_path / "out"),
        overwrite=False,
        n_jobs=1,
        backend=Nessai(trace=False),
        restrict_prior=False,
    )
    solver.distributions["posterior"] = fake_posterior
    solver.backend = fake_posterior
    solver.fit_result = FitResults(
        time=1.0,
        posterior_samples=pd.DataFrame(),
        n_likelihood_evaluations=500,
        log_Z=-120.5,
        log_Z_err=0.3,
    )
    return solver


@pytest.fixture
def mock_solver(tmp_path):
    """Create a Solver-like object without touching XSPEC."""
    # A fake "posterior" backend whose .sample() returns deterministic data.
    posterior_samples = np.random.default_rng(42).random((20_000, len(PARAMETER_NAMES)))
    solver = _solver_without_xspec(tmp_path, lambda n, **kw: posterior_samples[:n])

    # Keep these pure tests independent of whether HEASoft is activated.
    with patch.object(Solver, "observed_counts", new_callable=PropertyMock,
                      return_value=np.ones(100, dtype=np.float64)):
        yield solver


# ---------------------------------------------------------------------------
# Tests — build_dataframe (baseline parity check)
# ---------------------------------------------------------------------------


class TestBuildDataframe:

    def test_returns_dataframe(self, mock_solver):
        df = mock_solver.build_dataframe(num_samples=100)

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (100, 2)
        assert list(df.columns) == mock_solver.parameter_names

    def test_accepts_a_backend_returning_a_dataframe(self, tmp_path):
        """Regression: the Emcee backend's `sample` returns a DataFrame, which broke both builders."""
        draws = pd.DataFrame(np.random.default_rng(0).random((50, 2)), columns=PARAMETER_NAMES)
        solver = _solver_without_xspec(tmp_path, lambda n, **kw: draws.iloc[:n])

        df = solver.build_dataframe(num_samples=20)
        idata = solver.build_inference_data(
            num_samples=20, include_prior=False, include_observed=False, include_log_likelihood=False,
        )

        np.testing.assert_array_equal(df.to_numpy(), draws.to_numpy()[:20])
        np.testing.assert_array_equal(idata.posterior["powerlaw_norm"].values.squeeze(), draws["powerlaw_norm"][:20])


class TestSampleParameters:

    def test_returns_an_array_of_physical_draws(self, mock_solver):
        prior_draws = mock_solver.sample_parameters(30)
        posterior_draws = mock_solver.sample_parameters(30, "posterior")

        assert isinstance(prior_draws, np.ndarray) and prior_draws.shape == (30, 2)
        assert np.all((prior_draws[:, 0] >= 1.0) & (prior_draws[:, 0] <= 3.0))
        assert isinstance(posterior_draws, np.ndarray) and posterior_draws.shape == (30, 2)

    def test_names_the_available_distributions(self, mock_solver):
        """Regression: a missing distribution (e.g. "posterior" before run) raised a bare KeyError."""
        del mock_solver.distributions["posterior"]

        with pytest.raises(KeyError, match=r"No 'posterior' distribution.*\['prior'\].*run\(\)"):
            mock_solver.sample_parameters(10, "posterior")
        with pytest.raises(KeyError, match="No 'posterior' distribution"):
            mock_solver.build_dataframe()

    def test_accepts_an_added_torch_distribution(self, mock_solver):
        """Regression: an added torch or sbi distribution got `sample(n)` and failed with "'int' object is not iterable"."""
        torch = pytest.importorskip("torch")
        mock_solver.distributions["custom"] = torch.distributions.Independent(
            torch.distributions.Normal(torch.zeros(2), torch.ones(2)), 1
        )

        draws = mock_solver.sample_parameters(4, "custom")
        idata = mock_solver.build_inference_data(
            "custom", 4, include_prior=False, include_observed=False, include_log_likelihood=False,
        )

        assert isinstance(draws, np.ndarray) and draws.shape == (4, 2)
        assert mock_solver.build_dataframe("custom", 4).shape == (4, 2)
        assert idata.posterior["powerlaw_norm"].shape == (1, 4)
        # A bsixsa backend still receives a number of draws.
        mock_solver.sample_parameters(3, "posterior")
        mock_solver.distributions["posterior"].sample.assert_called_with(3)
        with pytest.raises(KeyError, match=r"No 'other' distribution.*'custom'"):
            mock_solver.sample_parameters(4, "other")


# ---------------------------------------------------------------------------
# Tests — build_inference_data
# ---------------------------------------------------------------------------


class TestBuildInferenceData:

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
        for name in mock_solver.parameter_names:
            assert name in idata.posterior
            assert idata.posterior[name].shape == (1, n)

    def test_prior_group_included_by_default(self, mock_solver):
        idata = mock_solver.build_inference_data(
            num_samples=50, include_log_likelihood=False,
        )
        assert "prior" in idata.groups()
        for name in mock_solver.parameter_names:
            assert name in idata.prior
            assert idata.prior[name].shape == (1, 50)

    def test_prior_group_excluded(self, mock_solver):
        idata = mock_solver.build_inference_data(
            num_samples=50, include_prior=False,
            include_log_likelihood=False,
        )
        assert "prior" not in idata.groups()

    def test_observed_data_errors_propagate(self, mock_solver):
        """observed_counts errors are surfaced to the caller."""
        with patch.object(
            type(mock_solver),
            "observed_counts",
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
        fake_obs = np.ones(100, dtype=np.float64)
        with patch.object(
            type(mock_solver),
            "observed_counts",
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

        for name in mock_solver.parameter_names:
            np.testing.assert_array_equal(
                idata.posterior[name].values.squeeze(), df[name].values
            )

    def test_log_likelihood_group_included(self, mock_solver):
        """log_likelihood group is present with correct shape."""
        n = 80
        n_bins = 50
        fake_obs = np.random.default_rng(0).poisson(10, size=n_bins).astype(np.float64)
        fake_model_counts = np.random.default_rng(1).poisson(10, size=(n, n_bins)).astype(np.float64)

        with patch.object(
            type(mock_solver),
            "observed_counts",
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
            "observed_counts",
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


def test_log_likelihood_and_observed_data_use_the_exact_counts(tmp_path, monkeypatch):
    """Regression: the float32 rate x exposure is not an integer in some bins, and the
    Poisson log-pmf of a non-integer count is -inf, which broke az.loo and az.compare."""
    counts = np.array([54321.0, 98765.0, 1234.0, 7.0])
    exposure = 12345.6
    rate = counts / exposure
    float32_counts = np.asarray(rate, dtype=np.float32) * exposure
    assert not np.array_equal(float32_counts, counts)  # the old data path was inexact here

    fake = MagicMock()
    fake.AllData.nSpectra = 1
    fake.AllData.return_value = SimpleNamespace(values=list(rate), exposure=exposure)
    monkeypatch.setattr(solver_module, "xspec", fake)
    monkeypatch.setattr(xspec_module, "xspec", fake)  # the spectra are read through bsixsa.xspec
    solver = _solver_without_xspec(tmp_path, lambda n, **kw: np.full((n, 2), 0.5))
    model_counts = np.tile(counts, (5, 1)) + 0.5

    with patch.object(solver, "simulate", return_value={"total_model_counts": model_counts}):
        idata = solver.build_inference_data(num_samples=5, include_prior=False)

    log_likelihood = idata.log_likelihood["spectrum"].values.squeeze()
    assert np.all(np.isfinite(log_likelihood))
    np.testing.assert_allclose(log_likelihood, poisson.logpmf(counts, model_counts))
    np.testing.assert_array_equal(idata.observed_data["spectrum"].values, counts)
    np.testing.assert_array_equal(solver.observed_spectrum, counts)


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
