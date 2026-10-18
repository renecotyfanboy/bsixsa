"""XSPEC-free tests of `Solver.run` (fold accounting, failures, XSPEC state) and of the backends' naming.

The backends are fake ones registered in `BACKEND_REGISTRY`, or the real
nautilus and UltraNest backends with their samplers mocked.
"""

from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from _analytic import AnalyticSolver, AnalyticSpectrum

import bsixsa.backend as backend_package
from bsixsa.backend.abc import Backend
from bsixsa.backend.config import BackendConfig, Nautilus
from bsixsa.solver import FitResults

SRC_DIR = Path(__file__).resolve().parents[1] / "src"


class _FakeConfig(BackendConfig):
    backend_name: ClassVar[str] = "fake"


def _default_result(solver) -> FitResults:
    u = np.random.default_rng(1).uniform(0.4, 0.6, size=(200, solver.num_parameters))
    samples = solver.prior.from_unit_cube(u)
    return FitResults(
        time=1.0,
        posterior_samples=pd.DataFrame(samples, columns=solver.parameter_names),
        n_likelihood_evaluations=1000,
        log_Z=-10.0,
        log_Z_err=0.1,
    )


@pytest.fixture
def backends(monkeypatch):
    """Register a fake backend for `_FakeConfig`; `state.result_fn` builds the results."""
    state = SimpleNamespace(constructed=[], result_fn=_default_result)

    class FakeBackend(Backend):
        config_cls = _FakeConfig

        def __init__(self, *, solver, config):
            super().__init__(solver=solver, config=config)
            self.tracer = SimpleNamespace(plot_progress=lambda: None, save=lambda: None)
            state.constructed.append(self)

        def run(self, **kwargs):
            return state.result_fn(self.solver)

        def sample(self, n):
            return self.solver.prior.sample(n, random_state=2)

    monkeypatch.setitem(backend_package.BACKEND_REGISTRY, "fake", FakeBackend)
    return state


def _make_solver(tmp_path, *, cls=AnalyticSolver):
    return cls(AnalyticSpectrum(d=2), backend=_FakeConfig(trace=False), outputfiles_basename=tmp_path / "out")


class _BayesOnSolver(AnalyticSolver):
    def _check_xspec_state(self) -> None:
        raise RuntimeError("xspec.Fit.bayes is 'on'")


def test_nautilus_config_seed():
    assert Nautilus().seed is None
    assert Nautilus(seed=7).seed == 7
    for bad in (-1, 1.5, "3"):
        with pytest.raises(Exception):
            Nautilus(seed=bad)


def test_post_run_folds_are_counted_under_post(tmp_path, backends):
    solver = _make_solver(tmp_path)
    theta = solver.prior.from_unit_cube(np.random.default_rng(0).random((7, 2)))

    def result_fn(s):
        s.log_likelihood_fn(theta, None, progress_bar=False)
        return _default_result(s)

    backends.result_fn = result_fn
    solver.run()
    assert solver.fold_counts == {"backend": 7}

    solver.build_inference_data(num_samples=5, include_prior=False, include_observed=False)
    with solver.fold_phase("custom"):
        solver.simulate(theta[:2], return_kind="cstat")
    assert solver.fold_counts == {"backend": 7, "post": 5, "custom": 2}

    # A new run attributes its backend folds to "backend" again.
    solver.run()
    assert solver.fold_counts == {"backend": 14, "post": 5, "custom": 2}
    solver.simulate(theta[:3], return_kind="cstat")
    assert solver.fold_counts["post"] == 8


def test_backend_warnings_are_issued_when_the_backend_fails(tmp_path, backends):
    solver = _make_solver(tmp_path)

    def result_fn(s):
        warnings.warn("explains the failure", UserWarning, stacklevel=2)
        raise RuntimeError("backend crashed")

    backends.result_fn = result_fn
    with pytest.warns(UserWarning, match="explains the failure"), pytest.raises(RuntimeError, match="backend crashed"):
        solver.run()
    assert solver.fit_result is None and solver._pool_depth == 0


def test_run_rejects_bayes_on_before_building_the_backend(tmp_path, backends):
    solver = _make_solver(tmp_path, cls=_BayesOnSolver)

    with pytest.raises(RuntimeError, match="bayes"):
        solver.run()

    assert backends.constructed == []
    assert solver.fold_counts == {} and solver._pool_depth == 0


# ---------------------------------------------------------------------------
# Backends naming and package exports
# ---------------------------------------------------------------------------


def test_nautilus_backend_uses_solver_names_seed_and_shuffled_samples(tmp_path):
    from bsixsa.backend.nautilus import NautilusBackend

    points = np.arange(40, dtype=np.float64).reshape(20, 2)
    solver = SimpleNamespace(
        outputfiles_basename=str(tmp_path),
        parameter_names=["alpha", "beta"],
        prior=SimpleNamespace(dists=["dist_a", "dist_b"]),
    )
    prior = MagicMock()

    with patch("bsixsa.backend.nautilus.Prior", return_value=prior), \
         patch("bsixsa.backend.nautilus.Pool"), \
         patch("bsixsa.backend.nautilus.NestedSampler") as sampler_cls:
        sampler = sampler_cls.return_value
        sampler.posterior.return_value = (points, np.zeros(20), np.zeros(20))
        sampler.n_like = 77
        sampler.log_z = -3.0
        sampler.n_eff = 100.0
        backend = NautilusBackend(solver=solver, config=Nautilus(trace=False, seed=5))
        result = backend.run()
        first = backend.sample(8)
        again = backend.sample(8)
        everything = backend.sample(100)

    assert [c.args for c in prior.add_parameter.call_args_list] == [("alpha", "dist_a"), ("beta", "dist_b")]
    assert sampler_cls.call_args.kwargs["seed"] == 5
    assert list(result.posterior_samples.columns) == ["alpha", "beta"]
    np.testing.assert_array_equal(result.posterior_samples.to_numpy(), points)
    assert result.n_likelihood_evaluations == 77

    assert first.shape == (8, 2)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, points[:8])
    assert len({tuple(row) for row in first}) == 8
    assert {tuple(row) for row in first} <= {tuple(row) for row in points}
    assert everything.shape == (20, 2)
    assert sorted(map(tuple, everything)) == sorted(map(tuple, points))


def test_ultranest_backend_uses_solver_names(tmp_path):
    pytest.importorskip("ultranest")
    from bsixsa.backend.config import Ultranest
    from bsixsa.backend.ultranest import UltranestBackend

    solver = SimpleNamespace(outputfiles_basename=str(tmp_path), parameter_names=["alpha", "beta"])
    with patch("bsixsa.backend.ultranest.ultranest.ReactiveNestedSampler") as sampler_cls:
        UltranestBackend(solver=solver, config=Ultranest(trace=False))

    assert sampler_cls.call_args.args[0] == ["alpha", "beta"]
    assert sampler_cls.call_args.kwargs["resume"] == "resume"

    with patch("bsixsa.backend.ultranest.ultranest.ReactiveNestedSampler") as sampler_cls:
        UltranestBackend(solver=solver, config=Ultranest(trace=False, resume=False))
    # UltraNest 4.x asserts on resume=False; "overwrite" is its fresh start.
    assert sampler_cls.call_args.kwargs["resume"] == "overwrite"


def test_package_import_does_not_load_torch_and_exports_history_matching():
    code = f"""
import sys
from unittest.mock import MagicMock
sys.path.insert(0, {str(SRC_DIR)!r})
sys.modules.setdefault("xspec", MagicMock())
import bsixsa
assert "torch" not in sys.modules, "torch imported by the package"
assert "HistoryMatching" in bsixsa.__all__
from bsixsa.restriction import HistoryMatching
assert bsixsa.HistoryMatching is HistoryMatching
assert "HistoryMatching" in dir(bsixsa) and "Solver" in dir(bsixsa)
assert "torch" not in sys.modules, "torch imported by HistoryMatching"
try:
    bsixsa.not_an_export
except AttributeError:
    pass
else:
    raise AssertionError("missing attribute did not raise")
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
