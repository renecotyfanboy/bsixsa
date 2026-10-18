"""XSPEC-free tests of the Solver-level prior restriction (`restrict_prior`, `restrict`, `run`).

The history matching itself is replaced by a monkeypatched
`bsixsa.restriction.waves.run_history_matching` returning a small stub that
implements the `Restriction` contract, so these tests exercise the solver
wiring only: resolution of `restrict_prior`, caching, pool scopes, prior swap,
evidence and fold accounting, and box-edge diagnostics. The same
paths with the real restriction package are covered by
`test_restriction_integration.py`.
"""

from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from _analytic import AnalyticSolver, AnalyticSpectrum, grid_truth

import bsixsa.backend as backend_package
from bsixsa.backend.abc import Backend
from bsixsa.backend.config import BackendConfig, Nautilus
from bsixsa.priors import norm
from bsixsa.restriction import Restriction, waves
from bsixsa.restriction.config import HistoryMatching
from bsixsa.solver import FitResults

SRC_DIR = Path(__file__).resolve().parents[1] / "src"


# ---------------------------------------------------------------------------
# Restriction stand-ins
# ---------------------------------------------------------------------------


@pytest.fixture
def hm_cls():
    """The `HistoryMatching` class the solver resolves."""
    return HistoryMatching


class StubRestriction(Restriction):
    """A `Restriction` with a fixed box: it implements only what the solver uses.

    It skips the dataclass `__init__`, so it has no region, history or folds.
    """

    def __init__(
        self,
        *,
        lo=(0.2, 0.2),
        hi=(0.8, 0.8),
        core_lo=(0.3, 0.3),
        core_hi=(0.7, 0.7),
        config=None,
        n_simulations: int = 123,
        seconds: float = 4.5,
        box_info: dict | None = None,
    ):
        self.lo = np.asarray(lo, dtype=np.float64)
        self.hi = np.asarray(hi, dtype=np.float64)
        self._core = (np.asarray(core_lo, dtype=np.float64), np.asarray(core_hi, dtype=np.float64))
        self.config = config
        self.n_simulations = n_simulations
        self.seconds = seconds
        self.box_info = box_info if box_info is not None else {}
        self.threshold = 1.0
        self.truncated_calls: list = []
        self.saved_to: list[str] = []
        self._cache: dict = {}

    def box(self):
        return self.lo.copy(), self.hi.copy()

    @property
    def box_core(self):
        return self._core[0].copy(), self._core[1].copy()

    def truncated_prior(self, base_prior):
        self.truncated_calls.append(base_prior)
        cached = self._cache.get(id(base_prior))
        if cached is None or cached[0] is not base_prior:
            cached = (base_prior, base_prior.truncated(self.lo, self.hi))
            self._cache[id(base_prior)] = cached
        return cached[1]

    def save(self, directory: str) -> None:
        self.saved_to.append(directory)

    def __repr__(self) -> str:
        # `Restriction.__repr__` reads the region and folds, which the stub lacks.
        return "StubRestriction()"


@pytest.fixture
def fake_hm(monkeypatch, hm_cls):
    """Monkeypatch `run_history_matching` with a stub that folds one batch."""
    state = SimpleNamespace(calls=[], box={}, n_design=16)

    def run_history_matching(simulate_cstat, d, cfg, *, outdir=None,
                             param_names=None, log=None):
        solver = state.solver
        u = np.random.default_rng(0).random((state.n_design, d))
        cstat = simulate_cstat(u)
        restriction = StubRestriction(
            config=cfg, n_simulations=len(u), **state.box
        )
        state.calls.append(
            {
                "d": d,
                "cfg": cfg,
                "outdir": outdir,
                "param_names": param_names,
                "u": u,
                "cstat": cstat,
                "pool_depth": solver._pool_depth,
                "pool_requests": solver.n_pool_requests,
                "log": log,
                "restriction": restriction,
            }
        )
        return restriction

    monkeypatch.setattr(waves, "run_history_matching", run_history_matching)
    return state


# ---------------------------------------------------------------------------
# Fake backends
# ---------------------------------------------------------------------------


class _BoxConfig(BackendConfig):
    backend_name: ClassVar[str] = "fake_box"


class _RegionConfig(BackendConfig):
    backend_name: ClassVar[str] = "fake_region"
    restriction_mode: ClassVar[str] = "region"
    restrict_prior_default: ClassVar[bool] = True


class _X0Config(BackendConfig):
    backend_name: ClassVar[str] = "fake_x0"
    x0_physical: Any = None


def _default_result(solver) -> FitResults:
    # Central draws of the working prior: inside the core of every stub box.
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
    """Register one fake backend per fake config; `state.result_fn` builds the results."""
    state = SimpleNamespace(constructed=[], result_fn=_default_result, tracer_saves=[])

    def make(name, config_cls):
        class FakeBackend(Backend):
            def __init__(self, *, solver, config):
                super().__init__(solver=solver, config=config)
                self.tracer = SimpleNamespace(
                    plot_progress=lambda: None, save=lambda: state.tracer_saves.append(self)
                )
                self.prior_at_construction = solver.prior
                state.constructed.append(self)

            def run(self, **kwargs):
                return state.result_fn(self.solver)

            def sample(self, n):
                return self.solver.prior.sample(n, random_state=2)

        FakeBackend.config_cls = config_cls
        monkeypatch.setitem(backend_package.BACKEND_REGISTRY, name, FakeBackend)

    make("fake_box", _BoxConfig)
    make("fake_region", _RegionConfig)
    make("fake_x0", _X0Config)
    return state


def _make_solver(tmp_path, *, backend=None, spectrum=None, name="out", cls=AnalyticSolver, **kwargs):
    return cls(
        spectrum if spectrum is not None else AnalyticSpectrum(d=2),
        backend=backend if backend is not None else _BoxConfig(trace=False),
        outputfiles_basename=tmp_path / name,
        **kwargs,
    )


def _bounds(prior) -> np.ndarray:
    return np.asarray([dist.support() for dist in prior.dists], dtype=np.float64)


class _NormalPriorSpectrum(AnalyticSpectrum):
    """Infinite-support Gamma prior."""

    def prior_list(self):
        from scipy.stats import norm as scipy_norm

        dists = super().prior_list()
        dists[0] = scipy_norm(2.0, 0.5)
        return dists


class _TruncnormSpectrum(AnalyticSpectrum):
    """Truncated normal Gamma prior, whose XSPEC-convention term depends on the box."""

    def prior_list(self):
        dists = super().prior_list()
        dists[0] = norm(2.0, 0.5, 1.0, 3.0)
        return dists


class _BayesOnSolver(AnalyticSolver):
    def _check_xspec_state(self) -> None:
        raise RuntimeError("xspec.Fit.bayes is 'on'")


def _with_solver(fake_hm, solver):
    fake_hm.solver = solver
    return solver


# ---------------------------------------------------------------------------
# FitResults and configs
# ---------------------------------------------------------------------------


def test_fit_results_restriction_fields_default_to_neutral_values():
    result = FitResults(time=1.0, posterior_samples=pd.DataFrame(), n_likelihood_evaluations=3,
                        log_Z=0.0, log_Z_err=0.1)

    assert result.n_restriction_evaluations == 0
    assert result.restriction_time == 0.0
    assert result.log_Z_restriction_correction == 0.0


def test_backend_config_restriction_class_variables():
    assert BackendConfig.restriction_mode == "box"
    assert BackendConfig.restrict_prior_default is False
    assert Nautilus.restriction_mode == "box" and Nautilus.restrict_prior_default is False
    # Class variables are not pydantic fields.
    assert "restriction_mode" not in BackendConfig.model_fields
    assert "restrict_prior_default" not in BackendConfig.model_fields
    assert _RegionConfig.restriction_mode == "region" and _RegionConfig.restrict_prior_default is True


def test_nautilus_config_seed():
    assert Nautilus().seed is None
    assert Nautilus(seed=7).seed == 7
    for bad in (-1, 1.5, "3"):
        with pytest.raises(Exception):
            Nautilus(seed=bad)


# ---------------------------------------------------------------------------
# restrict_prior resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config_cls", "arg", "expected"),
    [
        (_BoxConfig, None, None),
        (_RegionConfig, None, "default"),
        (_BoxConfig, True, "default"),
        (_BoxConfig, np.True_, "default"),
        (_BoxConfig, False, None),
        (_RegionConfig, False, None),
    ],
)
def test_restrict_prior_resolution_table(tmp_path, hm_cls, config_cls, arg, expected):
    solver = _make_solver(tmp_path, backend=config_cls(trace=False), restrict_prior=arg)

    assert solver.restriction is None
    assert solver._restriction_injected is False
    if expected is None:
        assert solver.restriction_config is None
    else:
        assert isinstance(solver.restriction_config, hm_cls)
        assert solver.restriction_config == hm_cls()


def test_restrict_prior_history_matching_instance_is_kept(tmp_path, hm_cls):
    config = hm_cls(n_waves=2)

    solver = _make_solver(tmp_path, restrict_prior=config)

    assert solver.restriction_config is config
    assert solver.restriction is None and solver._restriction_injected is False


def test_restrict_prior_injected_restriction(tmp_path, hm_cls):
    stub = StubRestriction(config=hm_cls(n_waves=3))

    solver = _make_solver(tmp_path, restrict_prior=stub)

    assert solver.restriction is stub
    assert solver._restriction_injected is True
    assert solver.restriction_config is stub.config


def test_restrict_prior_rejects_unknown_objects_before_touching_the_directory(tmp_path, hm_cls):
    target = tmp_path / "out"
    target.mkdir()
    (target / "keep.txt").write_text("x")

    # A duck-typed look-alike of a Restriction is refused too: only real ones are reused.
    look_alike = SimpleNamespace(
        truncated_prior=lambda prior: None, config=None
    )
    for value in (object(), SimpleNamespace(kind="history-matching"), _BoxConfig(trace=False), look_alike):
        with pytest.raises(TypeError, match="restrict_prior"):
            _make_solver(tmp_path, overwrite=True, restrict_prior=value)
    assert (target / "keep.txt").exists()


def test_restrict_prior_requires_a_finite_support_only_when_active(tmp_path, hm_cls):
    spectrum = _NormalPriorSpectrum(d=2)

    for arg in (True, hm_cls(), StubRestriction()):
        with pytest.raises(ValueError, match="finite"):
            _make_solver(tmp_path, spectrum=spectrum, restrict_prior=arg)
    with pytest.raises(ValueError, match="finite"):
        _make_solver(tmp_path, spectrum=spectrum, backend=_RegionConfig(trace=False))

    assert _make_solver(tmp_path, spectrum=spectrum, restrict_prior=False).restriction_config is None
    assert _make_solver(tmp_path, spectrum=spectrum).restriction_config is None


# ---------------------------------------------------------------------------
# restrict()
# ---------------------------------------------------------------------------


def test_restrict_runs_once_in_a_single_pool_scope_and_caches(tmp_path, fake_hm, hm_cls):
    solver = _with_solver(fake_hm, _make_solver(tmp_path, restrict_prior=hm_cls(seed=11)))

    result = solver.restrict()

    assert len(fake_hm.calls) == 1
    call = fake_hm.calls[0]
    assert result is call["restriction"]
    assert solver.restriction is result and solver._restriction_injected is False
    assert solver.restriction_config is call["cfg"]
    # The pool was requested before any fold, inside the one scope restrict() opened.
    assert call["pool_depth"] == 1 and call["pool_requests"] >= 1
    assert solver._pool_depth == 0
    assert {entry["phase"] for entry in solver.fold_log} == {"restriction"}
    assert all(entry["pool_depth"] == 1 and entry["return_kind"] == "cstat" for entry in solver.fold_log)
    assert solver.fold_counts == {"restriction": fake_hm.n_design}
    # Arguments of the contract.
    assert call["d"] == 2
    assert call["param_names"] == ["Gamma", "norm"]
    assert call["cfg"].seed == 11
    assert call["log"] is print
    outdir = str(Path(solver.outputfiles_basename) / "restriction")
    assert Path(call["outdir"]) == Path(outdir)
    assert [Path(p) for p in result.saved_to] == [Path(outdir)]
    # simulate_cstat maps the cube through the base prior and returns float64.
    assert call["cstat"].dtype == np.float64
    np.testing.assert_allclose(call["cstat"], solver.spectrum.cstat_u(call["u"]), rtol=1e-12)

    assert solver.restrict() is result
    assert solver.restrict(hm_cls(seed=11)) is result
    assert len(fake_hm.calls) == 1


def test_restrict_force_and_new_config_recompute(tmp_path, fake_hm, hm_cls):
    solver = _with_solver(fake_hm, _make_solver(tmp_path, restrict_prior=True))

    first = solver.restrict()
    forced = solver.restrict(force=True)
    other_config = hm_cls(n_waves=2)
    third = solver.restrict(other_config)

    assert len(fake_hm.calls) == 3
    assert forced is not first and third is not forced
    assert fake_hm.calls[0]["cfg"] == hm_cls()
    assert solver.restriction is third and solver.restriction_config is other_config
    assert solver.fold_counts == {"restriction": 3 * fake_hm.n_design}


def test_restrict_without_configuration_uses_defaults(tmp_path, fake_hm, hm_cls):
    solver = _with_solver(fake_hm, _make_solver(tmp_path, restrict_prior=False))

    solver.restrict()

    assert fake_hm.calls[0]["cfg"] == hm_cls()
    assert solver.restriction_config == hm_cls()


def test_restrict_replaces_an_injected_restriction_only_when_asked(tmp_path, fake_hm, hm_cls):
    stub = StubRestriction(config=hm_cls(n_waves=3))
    solver = _with_solver(fake_hm, _make_solver(tmp_path, restrict_prior=stub))

    assert solver.restrict() is stub
    assert solver.restrict(hm_cls(n_waves=3)) is stub
    assert fake_hm.calls == []

    fresh = solver.restrict(force=True)

    assert fresh is not stub and solver._restriction_injected is False
    assert fake_hm.calls[0]["cfg"] == hm_cls(n_waves=3)


def test_restrict_forwards_the_logger(tmp_path, fake_hm):
    solver = _with_solver(fake_hm, _make_solver(tmp_path, restrict_prior=True))
    lines: list[str] = []

    solver.restrict(log=lines.append)
    solver.restrict(force=True, log=None)

    assert fake_hm.calls[0]["log"] == lines.append and fake_hm.calls[1]["log"] is None


@pytest.mark.parametrize("config", [True, False, 3, {"n_waves": 2}, "default"])
def test_restrict_refuses_a_non_configuration_before_any_work(tmp_path, fake_hm, config):
    solver = _with_solver(fake_hm, _make_solver(tmp_path, restrict_prior=False))

    with pytest.raises(TypeError, match="HistoryMatching"):
        solver.restrict(config)

    assert fake_hm.calls == [] and solver.fold_counts == {}
    assert solver.n_pool_requests == 0 and solver._pool_depth == 0


def test_restrict_prior_refuses_the_restriction_class_itself(tmp_path):
    from bsixsa.restriction import Restriction

    target = tmp_path / "out"
    target.mkdir()
    (target / "keep.txt").write_text("x")

    with pytest.raises(TypeError, match="restrict_prior"):
        _make_solver(tmp_path, overwrite=True, restrict_prior=Restriction)
    with pytest.raises(TypeError, match="restrict_prior"):
        _make_solver(tmp_path, overwrite=True, restrict_prior=StubRestriction)
    assert (target / "keep.txt").exists()


def test_restrict_checks_xspec_state_before_folding(tmp_path, fake_hm):
    solver = _with_solver(fake_hm, _make_solver(tmp_path, cls=_BayesOnSolver, restrict_prior=True))

    with pytest.raises(RuntimeError, match="bayes"):
        solver.restrict()

    assert fake_hm.calls == [] and solver.fold_counts == {}
    assert solver._pool_depth == 0


# ---------------------------------------------------------------------------
# run(): prior swap, evidence and fold accounting
# ---------------------------------------------------------------------------


def test_run_box_mode_installs_the_truncated_prior(tmp_path, fake_hm, backends):
    solver = _with_solver(fake_hm, _make_solver(tmp_path, restrict_prior=True))
    base = solver.base_prior

    result = solver.run()

    assert len(fake_hm.calls) == 1
    restriction = solver.restriction
    box_prior, log_mass = restriction.truncated_prior(base)
    assert solver.prior is box_prior and solver.prior is not base
    assert backends.constructed[0].prior_at_construction is box_prior
    assert solver.distributions["prior"] is base
    assert solver.distributions["restricted_prior"] is box_prior
    assert solver.distributions["posterior"] is backends.constructed[0]
    np.testing.assert_allclose(_bounds(box_prior), _bounds(base.truncated(restriction.lo, restriction.hi)[0]))
    assert log_mass == pytest.approx(2 * np.log(0.6))

    assert result is solver.fit_result
    assert result.log_Z == pytest.approx(-10.0 + log_mass)
    assert result.log_Z_restriction_correction == pytest.approx(log_mass)
    assert result.n_likelihood_evaluations == 1000 + fake_hm.n_design
    assert result.n_restriction_evaluations == fake_hm.n_design
    assert result.restriction_time == restriction.seconds
    # The restriction ran inside run()'s scope, after the pool was requested.
    assert fake_hm.calls[0]["pool_depth"] == 2 and fake_hm.calls[0]["pool_requests"] >= 1
    assert solver._pool_depth == 0


def test_run_region_mode_keeps_the_base_prior(tmp_path, fake_hm, backends):
    solver = _with_solver(fake_hm, _make_solver(tmp_path, backend=_RegionConfig(trace=False)))

    result = solver.run()

    assert len(fake_hm.calls) == 1
    assert solver.restriction.truncated_calls == []
    assert solver.prior is solver.base_prior
    assert backends.constructed[0].prior_at_construction is solver.base_prior
    assert "restricted_prior" not in solver.distributions
    assert result.log_Z == -10.0
    assert result.log_Z_restriction_correction == 0.0
    assert result.n_likelihood_evaluations == 1000 + fake_hm.n_design
    assert result.n_restriction_evaluations == fake_hm.n_design


@pytest.mark.parametrize("log_z", [np.nan, -np.inf])
def test_run_box_mode_leaves_a_non_finite_evidence_alone(tmp_path, fake_hm, backends, log_z):
    solver = _with_solver(fake_hm, _make_solver(tmp_path, restrict_prior=True))

    def result_fn(s):
        result = _default_result(s)
        result.log_Z = log_z
        return result

    backends.result_fn = result_fn
    result = solver.run()

    assert np.isnan(result.log_Z) if np.isnan(log_z) else result.log_Z == -np.inf
    assert result.log_Z_restriction_correction == pytest.approx(2 * np.log(0.6))


def test_run_without_restriction_is_untouched(tmp_path, fake_hm, backends):
    solver = _with_solver(fake_hm, _make_solver(tmp_path))

    result = solver.run()

    assert fake_hm.calls == [] and solver.restriction is None
    assert solver.prior is solver.base_prior
    assert result.log_Z == -10.0 and result.n_likelihood_evaluations == 1000
    assert result.n_restriction_evaluations == 0 and result.log_Z_restriction_correction == 0.0


def test_run_does_not_recount_an_injected_restriction(tmp_path, fake_hm, backends, hm_cls):
    stub = StubRestriction(config=hm_cls(), n_simulations=999, seconds=7.0)
    solver = _with_solver(fake_hm, _make_solver(tmp_path, restrict_prior=stub))

    result = solver.run()

    assert fake_hm.calls == []
    assert solver.prior is stub.truncated_prior(solver.base_prior)[0]
    assert result.n_likelihood_evaluations == 1000
    assert result.n_restriction_evaluations == 0
    assert result.restriction_time == 0.0
    assert result.log_Z == pytest.approx(-10.0 + 2 * np.log(0.6))


def test_run_uses_a_manual_restriction_even_when_disabled(tmp_path, fake_hm, backends):
    solver = _with_solver(fake_hm, _make_solver(tmp_path, restrict_prior=False))
    restriction = solver.restrict()

    result = solver.run()

    assert len(fake_hm.calls) == 1
    assert solver.restriction is restriction
    assert solver.prior is restriction.truncated_prior(solver.base_prior)[0]
    assert result.n_restriction_evaluations == fake_hm.n_design


def test_rerunning_never_truncates_twice(tmp_path, fake_hm, backends):
    solver = _with_solver(fake_hm, _make_solver(tmp_path, restrict_prior=True))
    base = solver.base_prior

    solver.run()
    first_bounds = _bounds(solver.prior)
    second = solver.run()

    assert len(fake_hm.calls) == 1
    assert solver.restriction.truncated_calls == [base, base]
    np.testing.assert_array_equal(_bounds(solver.prior), first_bounds)
    assert solver.distributions["prior"] is base
    assert second.log_Z == pytest.approx(-10.0 + 2 * np.log(0.6))
    # The restriction folds were reported by the first run; none were spent since.
    assert second.n_likelihood_evaluations == 1000
    assert second.n_restriction_evaluations == 0 and second.restriction_time == 0.0

    # Switching to a region backend restores the base prior and drops the box;
    # `backend_name` follows the configuration.
    solver.backend_config = _RegionConfig(trace=False)
    assert solver.backend_name == "fake_region"
    third = solver.run()

    assert solver.prior is base and "restricted_prior" not in solver.distributions
    assert third.log_Z == -10.0


def test_run_saves_the_box_of_a_computed_restriction_only(tmp_path, fake_hm, backends, hm_cls):
    solver = _with_solver(fake_hm, _make_solver(tmp_path, restrict_prior=True))
    solver.restrict()
    solver.run()
    outdir = Path(solver.outputfiles_basename) / "restriction"
    # restrict() saved before the box existed; the box-mode run records it.
    assert [Path(p) for p in solver.restriction.saved_to] == [outdir, outdir]

    stub = StubRestriction(config=hm_cls())
    injected = _with_solver(fake_hm, _make_solver(tmp_path, name="injected", restrict_prior=stub))
    injected.run()
    assert stub.saved_to == []

    region = _with_solver(fake_hm, _make_solver(tmp_path, name="region", backend=_RegionConfig(trace=False)))
    region.run()
    assert len(region.restriction.saved_to) == 1


def test_restriction_time_includes_the_box(tmp_path, fake_hm, backends):
    solver = _with_solver(fake_hm, _make_solver(tmp_path, restrict_prior=True))
    restriction = solver.restrict()
    restriction.box_info = {"seconds": 2.25}

    result = solver.run()

    assert result.restriction_time == pytest.approx(restriction.seconds + 2.25)


def test_post_run_folds_are_counted_under_post(tmp_path, fake_hm, backends):
    solver = _with_solver(fake_hm, _make_solver(tmp_path))
    theta = solver.base_prior.from_unit_cube(np.random.default_rng(0).random((7, 2)))

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


def test_box_edge_warning_as_error_keeps_the_finished_run(tmp_path, fake_hm, backends):
    solver = _with_solver(fake_hm, _make_solver(tmp_path, restrict_prior=True))
    rng = np.random.default_rng(3)

    def result_fn(s):
        result = _default_result(s)
        result.posterior_samples = _box_cube_samples(s, rng, [0.0, 0.4], [0.1, 0.6])
        return result

    backends.result_fn = result_fn
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match="Gamma"):
            solver.run()

    backend = backends.constructed[0]
    assert solver.fit_result is not None
    assert solver.posterior_samples is solver.fit_result.posterior_samples
    assert solver.distributions["posterior"] is backend
    assert backends.tracer_saves == [backend]
    assert solver.fit_result.log_Z == pytest.approx(-10.0 + 2 * np.log(0.6))
    assert solver._pool_depth == 0


def test_backend_warnings_keep_their_category_message_and_line(tmp_path, fake_hm, backends):
    solver = _with_solver(fake_hm, _make_solver(tmp_path, restrict_prior=True))
    rng = np.random.default_rng(3)

    def result_fn(s):
        # Points at the caller of Solver.run (result_fn <- backend.run <- Solver.run <- caller).
        warnings.warn("SIXSA final importance sample not certified", RuntimeWarning, stacklevel=4)
        for _ in range(3):
            warnings.warn("repeated", UserWarning, stacklevel=2)
        result = _default_result(s)
        result.posterior_samples = _box_cube_samples(s, rng, [0.0, 0.4], [0.1, 0.6])
        return result

    backends.result_fn = result_fn
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        run_line = sys._getframe().f_lineno + 1
        solver.run()

    first, repeated, box_edge = caught
    assert (first.category, str(first.message)) == (RuntimeWarning, "SIXSA final importance sample not certified")
    assert (first.filename, first.lineno) == (__file__, run_line)
    # Repeated at one line: shown once, as Python does by default.
    assert (repeated.category, str(repeated.message)) == (UserWarning, "repeated")
    # The box-edge diagnostic comes after the backend's own warnings.
    assert "Gamma" in str(box_edge.message) and "lower" in str(box_edge.message)


def test_backend_warnings_are_issued_when_the_backend_fails(tmp_path, backends):
    solver = _make_solver(tmp_path)

    def result_fn(s):
        warnings.warn("explains the failure", UserWarning, stacklevel=2)
        raise RuntimeError("backend crashed")

    backends.result_fn = result_fn
    with pytest.warns(UserWarning, match="explains the failure"), pytest.raises(RuntimeError, match="backend crashed"):
        solver.run()
    assert solver.fit_result is None and solver._pool_depth == 0


def test_run_rejects_bayes_on_before_restricting_or_building_the_backend(tmp_path, fake_hm, backends):
    solver = _with_solver(fake_hm, _make_solver(tmp_path, cls=_BayesOnSolver, restrict_prior=True))

    with pytest.raises(RuntimeError, match="bayes"):
        solver.run()

    assert fake_hm.calls == [] and backends.constructed == []
    assert solver.fold_counts == {} and solver._pool_depth == 0


def test_run_rejects_x0_outside_the_box(tmp_path, fake_hm, backends):
    # Gamma ~ U(1, 3): the box u in [0.2, 0.8] is Gamma in [1.4, 2.6].
    config = _X0Config(trace=False, x0_physical=np.array([1.1, 1.0]))
    solver = _with_solver(fake_hm, _make_solver(tmp_path, backend=config, restrict_prior=True))

    with pytest.raises(ValueError, match="restricted prior box") as excinfo:
        solver.run()

    assert "Gamma" in str(excinfo.value)
    assert backends.constructed == []
    assert solver.prior is solver.base_prior and "restricted_prior" not in solver.distributions

    solver.backend_config = _X0Config(trace=False, x0_physical=np.array([2.0, 1.0]))
    solver.run()
    assert len(backends.constructed) == 1


# ---------------------------------------------------------------------------
# best_fit_stat and UserWarning
# ---------------------------------------------------------------------------


def test_best_fit_stat_is_reported_with_the_base_prior_term(tmp_path, fake_hm, backends):
    fake_hm.box = {"lo": (0.2, 0.1), "hi": (0.7, 0.9), "core_lo": (0.0, 0.0), "core_hi": (1.0, 1.0)}
    solver = _with_solver(
        fake_hm, _make_solver(tmp_path, spectrum=_TruncnormSpectrum(d=2), restrict_prior=True)
    )
    x = np.array([1.9, 1.2])
    cstat = float(solver.spectrum.cstat(x)[0])

    def result_fn(s):
        result = _default_result(s)
        result.best_fit = pd.Series(x, index=s.parameter_names)
        result.best_fit_stat = cstat + s.prior.xspec_bayes_contribution(x)
        return result

    backends.result_fn = result_fn
    result = solver.run()

    box_term = solver.prior.xspec_bayes_contribution(x)
    base_term = solver.base_prior.xspec_bayes_contribution(x)
    assert result.best_fit_stat == pytest.approx(cstat + base_term, rel=1e-12)
    # The truncnorm term is renormalised by the box width 0.5 on that axis.
    assert box_term - base_term == pytest.approx(2 * np.log(0.5), rel=1e-9)


def _box_cube_samples(solver, rng, low, high, n=2000):
    """Samples uniform in `[low, high]` of the box's own unit cube, mapped to physical space."""
    u = rng.uniform(low, high, size=(n, len(low)))
    return pd.DataFrame(solver.prior.from_unit_cube(u), columns=solver.parameter_names)


def _record_box_warnings(solver) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver.run()
    return [str(w.message) for w in caught if issubclass(w.category, UserWarning)]


def test_box_edge_warning_for_samples_crowding_an_interior_shell(tmp_path, fake_hm, backends):
    # Box [0.2, 0.8], core [0.3, 0.7]: each shell is 1/6 of the box-cube width.
    solver = _with_solver(fake_hm, _make_solver(tmp_path, restrict_prior=True))
    rng = np.random.default_rng(3)

    def result_fn(s):
        result = _default_result(s)
        result.posterior_samples = _box_cube_samples(s, rng, [0.0, 0.4], [0.1, 0.6])
        return result

    backends.result_fn = result_fn
    messages = _record_box_warnings(solver)

    assert len(messages) == 1
    assert "Gamma" in messages[0] and "lower" in messages[0]


def test_no_box_edge_warning_for_samples_in_the_core(tmp_path, fake_hm, backends):
    solver = _with_solver(fake_hm, _make_solver(tmp_path, restrict_prior=True))
    rng = np.random.default_rng(4)

    def result_fn(s):
        result = _default_result(s)
        result.posterior_samples = _box_cube_samples(s, rng, [0.25, 0.25], [0.75, 0.75])
        result.best_fit = pd.Series([2.0, 1.0], index=s.parameter_names)
        return result

    backends.result_fn = result_fn

    assert _record_box_warnings(solver) == []


def test_no_box_edge_warning_on_faces_of_the_prior_cube(tmp_path, fake_hm, backends):
    fake_hm.box = {"lo": (0.0, 0.2), "hi": (0.8, 1.0), "core_lo": (0.1, 0.3), "core_hi": (0.7, 0.9)}
    solver = _with_solver(fake_hm, _make_solver(tmp_path, restrict_prior=True))
    rng = np.random.default_rng(5)

    def result_fn(s):
        result = _default_result(s)
        # Crowd the lower face of Gamma and the upper face of norm: both are prior faces.
        result.posterior_samples = _box_cube_samples(s, rng, [0.0, 0.95], [0.05, 1.0])
        return result

    backends.result_fn = result_fn

    assert _record_box_warnings(solver) == []


def test_box_edge_warning_for_a_best_fit_on_an_interior_face(tmp_path, fake_hm, backends):
    solver = _with_solver(fake_hm, _make_solver(tmp_path, restrict_prior=True))
    rng = np.random.default_rng(6)

    def result_fn(s):
        result = _default_result(s)
        result.posterior_samples = _box_cube_samples(s, rng, [0.25, 0.25], [0.75, 0.75])
        upper_norm = float(s.prior.dists[1].support()[1])
        result.best_fit = pd.Series([2.0, upper_norm], index=s.parameter_names)
        result.best_fit_stat = 50.0
        return result

    backends.result_fn = result_fn
    messages = _record_box_warnings(solver)

    assert len(messages) == 1
    assert "best fit" in messages[0] and "norm" in messages[0] and "upper" in messages[0]


def test_box_edge_diagnostics_never_raise(tmp_path, fake_hm, backends, hm_cls):
    stub = StubRestriction(config=hm_cls())
    stub.box = MagicMock(side_effect=[stub.box(), RuntimeError("box unavailable")])
    solver = _with_solver(fake_hm, _make_solver(tmp_path, restrict_prior=stub))

    def result_fn(s):
        result = _default_result(s)
        result.posterior_samples = pd.DataFrame({"only": [1.0]})
        return result

    backends.result_fn = result_fn

    assert _record_box_warnings(solver) == []


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


def _central_box_restriction(spectrum, tmp_path, spans=12.0):
    """A stub restriction boxing the grid posterior at `mean_u +- spans * std_u`."""
    truth = grid_truth(spectrum)
    mean, std = truth["mean_u"], truth["std_u"]
    return truth, StubRestriction(
        lo=np.clip(mean - spans * std, 0.0, 1.0),
        hi=np.clip(mean + spans * std, 0.0, 1.0),
        core_lo=np.clip(mean - 2 * spans / 3 * std, 0.0, 1.0),
        core_hi=np.clip(mean + 2 * spans / 3 * std, 0.0, 1.0),
    )


def test_levenberg_marquardt_samples_stay_in_the_box_after_run(tmp_path):
    from bsixsa.backend.config import LevenbergMarquardt

    spectrum = AnalyticSpectrum(d=2)
    truth, stub = _central_box_restriction(spectrum, tmp_path)
    solver = AnalyticSolver(
        spectrum,
        backend=LevenbergMarquardt(trace=False),
        outputfiles_basename=tmp_path / "run",
        restrict_prior=stub,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = solver.run()

    bounds = _bounds(solver.prior)
    assert solver.prior is not solver.base_prior
    assert not [w for w in caught if issubclass(w.category, UserWarning)]
    assert np.isnan(result.log_Z)
    assert result.log_Z_restriction_correction == pytest.approx(float(np.sum(np.log(stub.hi - stub.lo))))
    best = result.best_fit.to_numpy()
    np.testing.assert_allclose(best, truth["map_theta"], atol=3 * truth["std_theta"].max())
    expected_stat = float(spectrum.cstat(best)[0]) + solver.base_prior.xspec_bayes_contribution(best)
    assert result.best_fit_stat == pytest.approx(expected_stat, rel=1e-9)
    for samples in (solver.backend.sample(2000), solver.build_dataframe(num_samples=2000).to_numpy()):
        assert np.all((samples >= bounds[:, 0]) & (samples <= bounds[:, 1]))


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
