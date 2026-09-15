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
    solver.pool = None
    solver._pool_depth = 0
    solver._n_jobs = 1
    # `run` forks the folding pool up front; these fakes never fold.
    solver._get_pool = lambda: None

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


def test_available_cpu_count_is_positive_int():
    from bsixsa.convenience import available_cpu_count

    n = available_cpu_count()
    assert isinstance(n, int)
    assert n >= 1


def test_nautilus_config_worker_defaults_and_overrides():
    # Memory-safe defaults, decoupled from os.cpu_count().
    default = Nautilus()
    assert default.n_networks == 4
    assert default.n_pool == 4

    override = Nautilus(n_networks=8, n_pool=2)
    assert override.n_networks == 8
    assert override.n_pool == 2

    with pytest.raises(ValidationError):
        Nautilus(n_pool=0)
    with pytest.raises(ValidationError):
        Nautilus(n_networks=0)


def _nautilus_mock_solver(tmp_path):
    solver = SimpleNamespace(
        outputfiles_basename=str(tmp_path),
        parameter_names=["p1", "p2"],
        prior=SimpleNamespace(dists=[object(), object()]),
    )
    return solver


def test_nautilus_backend_sizes_pool_from_config_not_cpu_count(tmp_path):
    """The sampler must be built from config (not os.cpu_count()), and the
    sampler pool must be capped at the available CPU count."""
    from bsixsa.backend.nautilus import NautilusBackend

    with patch("bsixsa.backend.nautilus.Prior", return_value=MagicMock()), \
         patch("bsixsa.backend.nautilus.available_cpu_count", return_value=8), \
         patch("bsixsa.backend.nautilus.NestedSampler") as sampler_cls:

        # Defaults: n_networks=4, n_pool=4 -> pool=(None, 4)
        NautilusBackend(
            solver=_nautilus_mock_solver(tmp_path),
            config=Nautilus(trace=False),
        )
        kwargs = sampler_cls.call_args.kwargs
        assert kwargs["n_networks"] == 4
        assert kwargs["pool"] == (None, 4)

        # n_pool larger than available cores is capped at available_cpu_count()
        sampler_cls.reset_mock()
        NautilusBackend(
            solver=_nautilus_mock_solver(tmp_path),
            config=Nautilus(trace=False, n_networks=16, n_pool=100),
        )
        kwargs = sampler_cls.call_args.kwargs
        assert kwargs["n_networks"] == 16
        assert kwargs["pool"] == (None, 8)  # min(100, 8)
