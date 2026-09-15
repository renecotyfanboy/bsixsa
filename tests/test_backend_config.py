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


def _sixsa_mock_solver(tmp_path, d=2, seed=None):
    """Mock Solver working in the unit-Gaussian latent (from/to_unit_gaussian)."""
    import numpy as np
    from unittest.mock import MagicMock

    solver = MagicMock()
    solver.prior.ndim = d
    solver.prior.from_unit_gaussian = lambda z: np.asarray(z) * 10.0
    solver.prior.to_unit_gaussian = lambda x: np.asarray(x) / 10.0
    solver.outputfiles_basename = str(tmp_path)
    solver.parameter_names = [f"p{i}" for i in range(d)]

    rng = np.random.default_rng(seed)

    def mock_simulate(params, *args, **kwargs):
        n = len(params)
        return {
            "total_model_counts": np.ones((n, 50), dtype=np.float32),
            "cstat": rng.uniform(50, 100, size=n).astype(np.float64),
        }

    solver.simulate.side_effect = mock_simulate
    solver.observed_spectrum = np.ones(50, dtype=np.float32)
    return solver


def _mock_member_posterior(d):
    """A stand-in NDE posterior with real-tensor `log_prob`/`sample`."""
    import torch
    from unittest.mock import MagicMock

    p = MagicMock()
    p.log_prob.side_effect = lambda theta, *a, **k: torch.zeros(
        theta.shape[0], dtype=torch.float64
    )

    def _sample(shape, *a, **k):
        n = shape[0] if isinstance(shape, (tuple, list, torch.Size)) else shape
        return 0.1 * torch.randn(int(n), d)

    p.sample.side_effect = _sample
    p.set_default_x.return_value = p
    return p


def _patch_sixsa_internals(d):
    """Patch training/proposal/ensemble internals so run() needs no real flows."""
    from unittest.mock import patch

    stats = {"nde_index": 0, "training_time_s": 0.0, "best_val_loss": 1.0,
             "best_train_loss": 1.0, "n_epochs": 1}

    def fake_training_job(number, theta, x, **kwargs):
        s = dict(stats, nde_index=number)
        return _mock_member_posterior(d), s

    def fake_sample_flat(proposal, n, *args, **kwargs):
        import torch
        return 0.1 * torch.randn(int(n), d)

    return [
        patch("bsixsa.backend.sixsa._sixsa.training_job", side_effect=fake_training_job),
        patch("bsixsa.backend.sixsa._sixsa.build_truncated_proposal",
              side_effect=lambda *a, **k: object()),
        patch("bsixsa.backend.sixsa._sixsa.sample_flat", side_effect=fake_sample_flat),
        patch("bsixsa.backend.sixsa._sixsa.EnsemblePosterior",
              side_effect=lambda members, *a, **k: _mock_member_posterior(d)),
    ]


def test_sixsa_backend_run_mock(tmp_path):
    """End-to-end mocked run of the spec loop: checks + gates banked, pooled
    posterior returned, FitResults carries the pooled evidence."""
    import contextlib
    import numpy as np
    from bsixsa.backend.sixsa import SIXSABackend
    from bsixsa.backend.config import SIXSA

    solver = _sixsa_mock_solver(tmp_path, d=2, seed=0)

    config = SIXSA(
        trace=False,
        n_initial_per_param=8,      # d=2 -> 16 initial
        n_picked_per_param=4,
        n_sub_batch=4,
        n_ensemble=2,
        n_rounds=2,
        khat_from_round=1,          # gate/check from round 1
        early_member_n=64,
        gate_n=128,
        round_gate_n=64,
        gate_alpha=0.05,
        # Impossible efficiency floor -> checks/gates never PASS -> deterministic
        # full run that still banks every check/gate batch.
        early_member_min_eff=200.0,
        run_emulator_pass=False,
        plot_diagnostics=False,
    )
    backend = SIXSABackend(solver=solver, config=config)

    with contextlib.ExitStack() as stack:
        for p in _patch_sixsa_internals(2):
            stack.enter_context(p)
        results = backend.run()

    assert results is not None
    assert results.posterior_samples.shape == (10000, 2)
    # Two rounds recorded; each round trained the 2-member ensemble.
    assert len(backend.history) == 2
    assert len(backend.history[0]["nde_stats"]) == 2
    # Defensive-IS bank populated (per-member checks + round/final gates).
    assert backend.bank is not None and len(backend.bank) > 0
    assert backend.pool_info is not None
    assert np.isfinite(results.log_Z)
    assert results.n_likelihood_evaluations > 16
    # sample() returns physical-space parameters (mock scales latent by 10).
    physical = np.asarray(backend.sample(50))
    assert physical.shape == (50, 2)
    assert float(np.abs(physical).max()) > 1.0
    # best_fit is the lowest-cstat pooled sample.
    assert results.best_fit is not None
    assert np.isfinite(results.best_fit_stat)


def test_sixsa_backend_run_mock_with_emulator_pass(tmp_path):
    """Mocked run with the emulator pass enabled: it must populate
    `emulator_pass_info` and extend the run's likelihood-evaluation budget."""
    import contextlib
    import numpy as np
    from bsixsa.backend.sixsa import SIXSABackend
    from bsixsa.backend.config import SIXSA

    solver = _sixsa_mock_solver(tmp_path, d=2, seed=1)

    config = SIXSA(
        trace=False,
        n_initial_per_param=8,
        n_picked_per_param=4,
        n_sub_batch=4,
        n_ensemble=2,
        n_rounds=2,
        khat_from_round=1,
        early_member_n=64,
        gate_n=128,
        round_gate_n=64,
        gate_alpha=0.05,
        early_member_min_eff=200.0,
        run_emulator_pass=True,
        emulator_min_per_param=8,   # d=2 -> 16 surrogate draws
        train_likelihood_emulator=False,
        plot_diagnostics=False,
    )
    backend = SIXSABackend(solver=solver, config=config)

    with contextlib.ExitStack() as stack:
        for p in _patch_sixsa_internals(2):
            stack.enter_context(p)
        results = backend.run()

    assert backend.emulator_pass_info is not None
    assert "log_Z" in backend.emulator_pass_info
    assert backend.emulator_pass_info["n_draws"] == 16
    # The emulator-pass sims are included in the reported budget.
    assert results.n_likelihood_evaluations >= 16
    assert results.posterior_samples.shape == (10000, 2)


def test_sixsa_nde_kwargs_merge_keeps_reference_defaults():
    """A partial `nde_kwargs` override merges over the reference defaults instead
    of replacing them (e.g. `{"model": "nsf"}` must keep the MAF-era capacity
    and z-scoring settings)."""
    from bsixsa.backend.sixsa._sixsa import DEFAULT_NDE_KWARGS, merge_nde_kwargs

    merged = merge_nde_kwargs({"model": "nsf"})
    assert merged["model"] == "nsf"
    assert merged["hidden_features"] == 100
    assert merged["num_transforms"] == 10
    assert merged["z_score_theta"] == "independent"
    assert merged["z_score_x"] == "independent"

    assert merge_nde_kwargs(None) == DEFAULT_NDE_KWARGS
    # User values win over defaults.
    assert merge_nde_kwargs({"hidden_features": 64})["hidden_features"] == 64


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

    params = [SimpleNamespace(name="p1"), SimpleNamespace(name="p2")]

    with patch("bsixsa.backend.nautilus.iter_thawn_parameters", return_value=params), \
         patch("bsixsa.backend.nautilus.Prior", return_value=MagicMock()), \
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

