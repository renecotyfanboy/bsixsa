"""Tests for the public backend configuration API and solver wiring."""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

sys.modules.setdefault("xspec", MagicMock())

from bsixsa import (  # noqa: E402
    SIXSA,
    BackendConfig,
    Emcee,
    HistoryMatching,
    Iminuit,
    LevenbergMarquardt,
    Nautilus,
    Nessai,
    Ultranest,
)
from bsixsa.backend.abc import Backend  # noqa: E402
from bsixsa.solver import FitResults, Solver  # noqa: E402


@pytest.fixture
def solver_stub(tmp_path):
    return SimpleNamespace(
        num_parameters=2,
        outputfiles_basename=str(tmp_path),
        parameter_names=["par_a", "par_b"],
    )


@pytest.fixture
def mock_solver(tmp_path):
    import collections

    from bsixsa.priors import MultipleIndependent, uniform

    with patch.object(Solver, "__init__", lambda self, *args, **kwargs: None):
        solver = Solver.__new__(Solver)

    solver.backend_config = Nessai(trace=False)
    solver.outputfiles_basename = str(tmp_path)
    solver.prior = solver.base_prior = MultipleIndependent([uniform(0.0, 1.0), uniform(0.0, 1.0)])
    solver.distributions = {}
    solver.parameter_indices = [1, 2]
    solver._parameter_names = ["par_a", "par_b"]
    solver.fit_result = None
    solver.restriction = solver.restriction_config = None
    solver._restriction_injected = solver._restriction_folds_reported = False
    solver._fold_counts = collections.Counter()
    solver._fold_phase_name = "backend"
    solver.pool = None
    solver._pool_depth = 0
    solver._n_jobs = 1
    # `run` checks the xspec session and forks the folding pool up front;
    # these fakes never fold.
    solver._check_xspec_state = lambda: None
    solver._get_pool = lambda: None

    return solver


@pytest.fixture
def without_sampler_check(monkeypatch):
    """The fake backends need no sampler package, so `run` must not look for one."""
    monkeypatch.setattr("bsixsa.solver._check_sampler_installed", lambda config: None)


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
        config_cls = Nessai

        def run(self):
            raise NotImplementedError

        def sample(self, n):
            raise NotImplementedError

    with pytest.raises(TypeError, match="Nessai"):
        FakeBackend(solver=solver_stub, config=Iminuit(x0_physical=[1.0, 2.0]))


def test_solver_run_uses_constructor_backend_config(mock_solver, without_sampler_check):
    created = {}
    config = Nessai(num_live_points=321, trace=False)
    mock_solver.backend_config = config

    class FakeBackend(Backend):
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


def test_solver_run_rejects_mismatched_backend_config(mock_solver, without_sampler_check):
    mock_solver.backend_config = Iminuit(x0_physical=[1.0, 2.0])

    class FakeBackend(Backend):
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


def _run_mock_nautilus(tmp_path, config, *, cpu_count=8):
    """Run NautilusBackend with nautilus and the process pool mocked; return (sampler class, pool class)."""
    from bsixsa.backend.nautilus import NautilusBackend

    with patch("bsixsa.backend.nautilus.Prior", return_value=MagicMock()), \
         patch("bsixsa.backend.nautilus.available_cpu_count", return_value=cpu_count), \
         patch("bsixsa.backend.nautilus.Pool") as pool_cls, \
         patch("bsixsa.backend.nautilus.NestedSampler") as sampler_cls:
        sampler = sampler_cls.return_value
        sampler.posterior.return_value = (np.zeros((3, 2)), np.zeros(3), np.zeros(3))
        sampler.n_like, sampler.log_z, sampler.n_eff = 10, -1.0, 100.0
        NautilusBackend(solver=_nautilus_mock_solver(tmp_path), config=config).run()
    return sampler_cls, pool_cls


def test_nautilus_backend_sizes_pool_from_config_not_cpu_count(tmp_path):
    """The sampler must be built from config (not os.cpu_count()), and the
    sampler pool must be capped at the available CPU count."""
    # Defaults: n_networks=4, n_pool=4 -> a 4-process pool for nautilus' own work
    sampler_cls, pool_cls = _run_mock_nautilus(tmp_path, Nautilus(trace=False))
    pool_cls.assert_called_once_with(4)
    kwargs = sampler_cls.call_args.kwargs
    assert kwargs["n_networks"] == 4
    assert kwargs["pool"] == (None, pool_cls.return_value)
    assert kwargs["n_live"] == Nautilus().num_live_points
    assert kwargs["seed"] is None

    # n_pool larger than available cores is capped at available_cpu_count()
    sampler_cls, pool_cls = _run_mock_nautilus(
        tmp_path, Nautilus(trace=False, n_networks=16, n_pool=100, num_live_points=500, seed=3)
    )
    pool_cls.assert_called_once_with(8)  # min(100, 8)
    kwargs = sampler_cls.call_args.kwargs
    assert kwargs["n_networks"] == 16
    assert kwargs["n_live"] == 500
    assert kwargs["seed"] == 3
    assert kwargs["pool"] == (None, pool_cls.return_value)

    # One process: nautilus runs its own work serially, without a pool.
    sampler_cls, pool_cls = _run_mock_nautilus(tmp_path, Nautilus(trace=False, n_pool=1))
    pool_cls.assert_not_called()
    assert sampler_cls.call_args.kwargs["pool"] == (None, None)


def test_nautilus_backend_shuts_down_its_pool_after_the_run(tmp_path):
    """Regression: nautilus never closes its pool, so its worker processes outlived
    every run and printed a `Pool.__del__` traceback at exit."""
    from bsixsa.backend.nautilus import NautilusBackend

    _, pool_cls = _run_mock_nautilus(tmp_path, Nautilus(trace=False))
    pool = pool_cls.return_value
    pool.terminate.assert_called_once_with()
    pool.join.assert_called_once_with()

    # Also when the run fails.
    with patch("bsixsa.backend.nautilus.Prior", return_value=MagicMock()), \
         patch("bsixsa.backend.nautilus.available_cpu_count", return_value=8), \
         patch("bsixsa.backend.nautilus.Pool") as pool_cls, \
         patch("bsixsa.backend.nautilus.NestedSampler") as sampler_cls:
        sampler_cls.return_value.run.side_effect = RuntimeError("likelihood failed")
        backend = NautilusBackend(solver=_nautilus_mock_solver(tmp_path), config=Nautilus(trace=False))
        with pytest.raises(RuntimeError, match="likelihood failed"):
            backend.run()
    pool_cls.return_value.terminate.assert_called_once_with()
    pool_cls.return_value.join.assert_called_once_with()


# Library each backend module needs, besides bsixsa's own dependencies.
_BACKEND_LIBRARIES = {
    "nautilus": "nautilus",
    "nessai": "nessai",
    "ultranest": "ultranest",
    "levenberg_marquardt": "scipy",
    "iminuit": "iminuit",
    "emcee": "emcee",
    "sixsa": "sbi",
}


@pytest.mark.parametrize("name", sorted(_BACKEND_LIBRARIES))
def test_every_registered_backend_resolves_to_its_config(name):
    from bsixsa import backend as backend_package

    pytest.importorskip(_BACKEND_LIBRARIES[name])
    backend_cls = backend_package.get_backend_class(name)

    assert issubclass(backend_cls, Backend)
    assert backend_cls.config_cls.backend_name == name


def test_registry_lists_every_backend_config():
    from bsixsa import SIXSA
    from bsixsa.backend import BACKEND_REGISTRY

    configs = [Nautilus, Nessai, Ultranest, LevenbergMarquardt, Iminuit, Emcee, SIXSA]
    assert sorted(config.backend_name for config in configs) == sorted(BACKEND_REGISTRY)


def test_unknown_backend_name_lists_the_available_ones():
    """Regression: the message listed only the backends imported so far ('(none)' at first)."""
    from bsixsa.backend import get_backend_class

    with pytest.raises(ValueError, match="Unknown backend 'nautlius'.*emcee, iminuit, levenberg_marquardt"):
        get_backend_class("nautlius")


def test_star_import_of_the_backend_package_works():
    namespace = {}
    exec("from bsixsa.backend import *", namespace)

    assert namespace["Nautilus"] is Nautilus
    assert "get_backend_class" in namespace


def _sixsa_solver(**constraints):
    from bsixsa.priors import MultipleIndependent, uniform

    values = {"n_spectra": 1, "stat_method": "cstat", "has_background": False, **constraints}
    return SimpleNamespace(
        base_prior=MultipleIndependent([uniform(0.0, 1.0)]),
        _xspec_constraints=lambda: values,
    )


@pytest.mark.parametrize(
    "constraints, message",
    [
        ({"n_spectra": 2}, "exactly one spectrum, but 2 are loaded"),
        ({"n_spectra": None}, "number of loaded spectra could not be read"),
        ({"has_background": True}, "does not model a background"),
        ({"stat_method": "chi"}, "statMethod is 'chi'"),
        ({"stat_method": None}, "statMethod could not be read"),
    ],
)
def test_sixsa_session_errors_say_what_was_found(constraints, message):
    from bsixsa import SIXSA

    with pytest.raises(ValueError, match=message):
        SIXSA().validate_for_solver(_sixsa_solver(**constraints))


def test_sixsa_accepts_a_single_cstat_spectrum_without_background():
    from bsixsa import SIXSA

    SIXSA().validate_for_solver(_sixsa_solver())


def test_config_errors_quote_the_offending_values(solver_stub):
    from bsixsa import SIXSA

    with pytest.raises(ValidationError, match=r"n_training \(60000\) must not exceed train_max \(50000\)"):
        SIXSA(n_training=60_000)
    with pytest.raises(ValueError, match="got 3"):
        Emcee(init_from_prior=True, num_walkers=3).validate_for_solver(solver_stub)


@pytest.mark.parametrize("config_cls", [Nautilus, Nessai, Ultranest, Iminuit, Emcee, LevenbergMarquardt, SIXSA])
def test_required_package_is_the_sampler_of_the_backend_and_its_extra(config_cls):
    if config_cls in (LevenbergMarquardt, SIXSA):
        # scipy, torch and sbi are dependencies of bsixsa itself.
        assert config_cls.required_package is None
        return
    module, pip_name = config_cls.required_package
    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    extra = pyproject["project"]["optional-dependencies"][config_cls.backend_name]

    assert module == _BACKEND_LIBRARIES[config_cls.backend_name]
    assert [requirement.split(">")[0] for requirement in extra] == [pip_name]


def test_missing_sampler_package_is_reported_before_anything_is_touched(tmp_path, monkeypatch):
    output = tmp_path / "out"
    output.mkdir()
    (output / "previous_run.txt").write_text("kept")
    looked_up = []
    # Not installed: every lookup fails. The package is looked up, never imported.
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args: looked_up.append(name))

    expected = r"Nautilus backend needs the 'nautilus-sampler' package.*pip install nautilus-sampler"
    with pytest.raises(ImportError, match=expected):
        Solver([("powerlaw", "PhoIndex", None)], outputfiles_basename=str(output), overwrite=True, backend=Nautilus())

    assert looked_up == ["nautilus"]
    assert (output / "previous_run.txt").read_text() == "kept"


def test_missing_sampler_package_is_reported_by_run_when_the_config_was_replaced(mock_solver, monkeypatch):
    mock_solver.backend_config = Nautilus(trace=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args: None)
    mock_solver._get_pool = lambda: pytest.fail("the pool must not be forked")

    with pytest.raises(ImportError, match="pip install nautilus-sampler"):
        mock_solver.run()
    assert mock_solver.fit_result is None and mock_solver._pool_depth == 0


@pytest.mark.parametrize(
    "config_cls, field, value, expected",
    [
        (Nautilus, "num_live_points", 1e3, 1000),
        (Nautilus, "seed", np.int64(3), 3),
        (BackendConfig, "plot_step_percent", 20.0, 20),
        (Iminuit, "strategy", 2.0, 2),
        (SIXSA, "train_max", 5e4, 50_000),
        (SIXSA, "check_draws", np.float64(1e3), 1000),
        (SIXSA, "truncated_support_draws", np.float32(4096.0), 4096),
        (SIXSA, "n_training", 2e3, 2000),
        (SIXSA, "refresh_compressor", 2.0, 2),
        (HistoryMatching, "simulations_per_parameter", 1e3, 1000),
        (HistoryMatching, "max_candidates", 2e9, 2_000_000_000),
        (HistoryMatching, "box_samples", 0.0, 0),
        (HistoryMatching, "seed", np.int64(7), 7),
    ],
)
def test_integer_options_accept_whole_numbers_written_as_floats(config_cls, field, value, expected):
    stored = getattr(config_cls(**{field: value}), field)

    assert stored == expected and type(stored) is int


@pytest.mark.parametrize(
    "value", [True, np.True_, "1000", 1000.5, np.float32(2.5), float("inf"), float("nan"), np.float32("inf")]
)
@pytest.mark.parametrize(
    "config_cls, field",
    [(Nautilus, "num_live_points"), (SIXSA, "check_draws"), (HistoryMatching, "n_waves"), (HistoryMatching, "seed")],
)
def test_integer_options_refuse_booleans_strings_and_fractions(config_cls, field, value):
    with pytest.raises(ValidationError, match=field):
        config_cls(**{field: value})


def test_refresh_compressor_keeps_booleans_apart_from_integers():
    assert SIXSA(refresh_compressor=True).refresh_compressor is True
    assert SIXSA(refresh_compressor=False).refresh_compressor is False
    assert type(SIXSA(refresh_compressor=1.0).refresh_compressor) is int
