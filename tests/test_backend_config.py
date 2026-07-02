"""Tests for the public backend configuration API and solver wiring."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import ValidationError

sys.modules.setdefault("xspec", MagicMock())

from bsixsa import (
    BackendConfig,
    Emcee,
    Iminuit,
    LevenbergMarquardt,
    Nautilus,
    Nessai,
    Ultranest,
)
from bsixsa.backend.abc import Backend
from bsixsa.solver import FitResults, Solver


@pytest.fixture
def solver_stub(tmp_path):
    return SimpleNamespace(
        num_parameters=2,
        outputfiles_basename=str(tmp_path),
        parameter_names=["par_a", "par_b"],
    )


@pytest.fixture
def mock_solver(tmp_path):
    with patch.object(Solver, "__init__", lambda self, *args, **kwargs: None):
        solver = Solver.__new__(Solver)

    solver.backend_name = "nessai"
    solver.backend_config = Nessai(trace=False)
    solver.outputfiles_basename = str(tmp_path)
    solver.distributions = {}
    solver.indexes = [1, 2]
    solver._parameter_names = ["par_a", "par_b"]
    solver.posterior_samples = None
    solver.fit_result = None

    return solver


def _fit_result() -> FitResults:
    return FitResults(
        time=1.0,
        posterior_samples=pd.DataFrame({"par_a": [1.0], "par_b": [2.0]}),
        n_likelihood_evaluations=10,
        log_Z=0.0,
        log_Z_err=0.1,
    )


def test_root_exports_public_backend_configs():
    configs = [
        Nautilus(),
        Nessai(),
        Ultranest(),
        Iminuit(x0_physical=[1.0, 2.0]),
        LevenbergMarquardt(x0_physical=[1.0, 2.0]),
        Emcee(init_from_prior=True),
    ]

    assert [config.backend_name for config in configs] == [
        "nautilus",
        "nessai",
        "ultranest",
        "iminuit",
        "levenberg_marquardt",
        "emcee",
    ]


def test_nessai_config_rejects_invalid_live_points():
    with pytest.raises(ValidationError, match="num_live_points"):
        Nessai(num_live_points=0)


def test_nessai_config_rejects_reserved_engine_kwargs():
    with pytest.raises(ValidationError, match="importance_nested_sampler"):
        Nessai(engine_kwargs={"importance_nested_sampler": False})


def test_iminuit_config_rejects_conflicting_start_options():
    with pytest.raises(ValidationError, match="mutually exclusive"):
        Iminuit(x0_physical=[1.0, 2.0], init_from_prior=True)


def test_iminuit_config_default_strategy_and_tol():
    config = Iminuit()
    assert config.strategy == 1
    assert config.tol is None


def test_iminuit_config_rejects_out_of_range_strategy():
    with pytest.raises(ValidationError, match="strategy"):
        Iminuit(strategy=3)
    with pytest.raises(ValidationError, match="strategy"):
        Iminuit(strategy=-1)


def test_iminuit_config_rejects_non_positive_tol():
    with pytest.raises(ValidationError, match="tol"):
        Iminuit(tol=0.0)
    with pytest.raises(ValidationError, match="tol"):
        Iminuit(tol=-1.0)


def test_emcee_config_requires_initialisation_strategy():
    with pytest.raises(ValidationError, match="requires either `x0_physical`"):
        Emcee()


def test_ultranest_config_rejects_unknown_extras():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Ultranest(foo="bar")


def test_trace_config_exposes_shared_plot_settings():
    config = BackendConfig(plot_every=25, plot_step_percent=20, trace=False)

    assert config.plot_every == 25
    assert config.plot_step_percent == 20
    assert config.trace is False


def test_emcee_validate_for_solver_requires_enough_walkers(solver_stub):
    config = Emcee(init_from_prior=True, num_walkers=3)

    with pytest.raises(ValueError, match="num_walkers"):
        config.validate_for_solver(solver_stub)


def test_iminuit_validate_for_solver_rejects_wrong_x0_length(solver_stub):
    config = Iminuit(x0_physical=[1.0])

    with pytest.raises(ValueError, match="x0_physical"):
        config.validate_for_solver(solver_stub)


def test_backend_base_class_rejects_mismatched_config_type(solver_stub):
    class FakeBackend(Backend):
        name = "nessai"
        config_cls = Nessai

        def run(self):
            raise NotImplementedError

        def sample(self, n):
            raise NotImplementedError

    with pytest.raises(TypeError, match="Nessai"):
        FakeBackend(solver=solver_stub, config=Iminuit(x0_physical=[1.0, 2.0]))


def test_solver_run_uses_constructor_backend_config(mock_solver):
    created = {}
    config = Nessai(num_live_points=321, trace=False)
    mock_solver.backend_config = config
    mock_solver.backend_name = config.backend_name

    class FakeBackend(Backend):
        name = "nessai"
        config_cls = Nessai

        def __init__(self, *, solver, config):
            super().__init__(solver=solver, config=config)
            created["solver"] = solver
            created["config"] = config
            self.tracer = SimpleNamespace(plot_progress=lambda: None, save=lambda: None)

        def run(self):
            created["run"] = True
            return _fit_result()

        def sample(self, n):
            raise NotImplementedError

    with patch(
        "bsixsa.backend.get_backend_class",
        return_value=FakeBackend,
    ), patch.object(
        type(mock_solver),
        "parameter_names",
        new_callable=lambda: property(lambda self: self._parameter_names),
    ):
        result = mock_solver.run()

    assert result.posterior_samples.equals(_fit_result().posterior_samples)
    assert created["solver"] is mock_solver
    assert created["config"] is config
    assert created["run"] is True
    assert mock_solver.distributions["posterior"] is mock_solver.backend


def test_solver_run_rejects_mismatched_backend_config(mock_solver):
    mock_solver.backend_name = "nessai"
    mock_solver.backend_config = Iminuit(x0_physical=[1.0, 2.0])

    class FakeBackend(Backend):
        name = "nessai"
        config_cls = Nessai

        def __init__(self, *, solver, config):
            super().__init__(solver=solver, config=config)

        def run(self):
            return _fit_result()

        def sample(self, n):
            raise NotImplementedError

    with patch("bsixsa.backend.get_backend_class", return_value=FakeBackend):
        with pytest.raises(TypeError, match="Nessai"):
            mock_solver.run()


def test_sixsa_backend_run_mock(tmp_path):
    import torch
    import numpy as np
    from unittest.mock import MagicMock, patch
    from bsixsa.backend.sixsa import SIXSABackend
    from bsixsa.backend.config import SIXSA

    # Mock Solver
    solver = MagicMock()
    solver.prior.ndim = 2
    solver.prior.from_unit_cube = lambda x: x * 10.0
    solver.prior.to_unit_cube = lambda x: x / 10.0
    solver.outputfiles_basename = str(tmp_path)
    solver.parameter_names = ["p1", "p2"]

    # Mock simulations results
    def mock_simulate(params, *args, **kwargs):
        n = len(params)
        return {
            "total_model_counts": np.ones((n, 50), dtype=np.float32),
            "cstat": np.random.uniform(50, 100, size=n).astype(np.float32)
        }
    solver.simulate.side_effect = mock_simulate
    solver.observed_spectrum = np.ones(50, dtype=np.float32)

    # Mock embedding
    from bsixsa.backend.sixsa.embedding import IdentityEmbedding
    class MockEmbedding(IdentityEmbedding):
        pass
    embedding = MockEmbedding()
    embedding.transform = MagicMock(return_value=np.ones(50, dtype=np.float32))

    config = SIXSA(
        trace=False,
        num_simulations_per_round=[10, 10],
        embedding=embedding,
        n_ensemble=1,
        is_filter_fraction=0.2,
        khat_threshold=None,      # disable early stopping for a deterministic run
        plot_diagnostics=False,   # skip the xspec-backed diagnostic plots under mocks
    )
    backend = SIXSABackend(solver=solver, config=config)

    # Patch training_job to return mock posteriors
    mock_post = MagicMock()
    def mock_log_prob(theta, *args, **kwargs):
        n = theta.shape[0] if hasattr(theta, "shape") else 1
        return torch.ones(n, dtype=torch.float32)
    mock_post.log_prob.side_effect = mock_log_prob
    def mock_sample(shape, *args, **kwargs):
        n = shape[0] if isinstance(shape, (tuple, list, torch.Size)) else shape
        # Interior of the unit cube so the prior log-prob is finite and the
        # importance weights are well-defined (boundary points give -inf).
        return torch.full((n, 2), 0.5, dtype=torch.float32)
    mock_post.sample.side_effect = mock_sample
    mock_post.potential_fn = MagicMock()
    mock_post.potential_fn.device = "cpu"
    mock_post._device = "cpu"

    # Each training job now returns (posterior, training_stats); mock that shape.
    mock_stats = {
        "nde_index": 0,
        "training_time_s": 0.0,
        "best_val_loss": 1.0,
        "best_train_loss": 1.0,
        "n_epochs": 1,
    }
    mock_parallel = MagicMock()
    mock_parallel.return_value = [(mock_post, mock_stats)]
    with patch("bsixsa.backend.sixsa._sixsa.Parallel", return_value=mock_parallel):
        results = backend.run()

    assert results is not None
    assert results.posterior_samples.shape == (10000, 2)
    # The run records a per-round history that drives the diagnostics.
    assert len(backend.history) >= 1
    assert backend.history[0]["nde_stats"] == [mock_stats]
    # The returned posterior is importance-resampled (no flow rejection sampling).
    assert backend.best_weighted is not None
    # sample() returns physical-space parameters, not the unit cube. The mock
    # `from_unit_cube` scales by 10, so physical values exceed the unit cube.
    physical = np.asarray(backend.sample(50))
    assert physical.shape == (50, 2)
    assert float(physical.max()) > 1.0

