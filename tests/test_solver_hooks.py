"""XSPEC-free tests of the Solver hooks, `_setup` split and fold accounting."""

from __future__ import annotations

import collections
import os
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from _analytic import AnalyticSolver, AnalyticSpectrum

import bsixsa.solver as solver_module
import bsixsa.xspec as xspec_module
from bsixsa.backend.config import BackendConfig, Nessai
from bsixsa.priors import MultipleIndependent, loguniform, uniform
from bsixsa.restriction import HistoryMatching
from bsixsa.solver import Solver


class _DummyConfig(BackendConfig):
    backend_name: ClassVar[str] = "dummy"


class _RecordingConfig(BackendConfig):
    """Records the solver state visible to `validate_for_solver`."""

    backend_name: ClassVar[str] = "recording"
    seen: ClassVar[dict] = {}

    def validate_for_solver(self, solver) -> None:
        type(self).seen = {
            "num_parameters": solver.num_parameters,
            "fold_counts": solver.fold_counts,
            "has_pool_attr": hasattr(solver, "pool"),
            "prior_is_base": solver.prior is solver.base_prior,
        }


def _config():
    return _DummyConfig(trace=False)


def _make_solver(tmp_path, name="out", spectrum=None, **kwargs):
    spectrum = spectrum if spectrum is not None else AnalyticSpectrum(d=2)
    return AnalyticSolver(
        spectrum,
        backend=kwargs.pop("backend", _config()),
        outputfiles_basename=tmp_path / name,
        **kwargs,
    )


def _prior_draws(solver, n, seed=0):
    return solver.prior.from_unit_cube(np.random.default_rng(seed).random((n, solver.num_parameters)))


def _bare_solver():
    """A `Solver` built without `__init__` or `_setup`: only what a test sets exists."""
    return Solver.__new__(Solver)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_analytic_solver_builds(tmp_path):
    solver = _make_solver(tmp_path, restrict_prior=True)

    assert solver.num_parameters == 2
    assert solver.parameter_names == ["Gamma", "norm"]
    assert solver.parameter_indices == [1, 2] and solver.indexes is solver.parameter_indices
    assert solver.model_names == ["analytic"] * 2 and solver.model_indexes is solver.model_names
    assert solver.n_bins == 64
    assert solver.backend_name == "dummy"
    assert solver.base_prior is solver.prior
    assert solver.distributions == {"prior": solver.prior}
    assert solver.fold_counts == {}
    assert solver.restriction is None
    # `True` resolves to a default history-matching configuration.
    assert solver.restriction_config == HistoryMatching()
    assert solver.fit_result is None and solver.posterior_samples is None
    assert solver.pool is None and solver._pool_depth == 0
    assert solver._n_jobs == 1
    assert os.path.isdir(solver.outputfiles_basename)
    assert solver.outputfiles_basename.endswith(os.sep)
    assert solver.observed_counts.dtype == np.float64
    np.testing.assert_array_equal(solver.observed_counts, solver.spectrum.observed)
    # `observed_spectrum` is the old name of the same exact counts.
    assert solver.observed_spectrum.dtype == np.float64
    np.testing.assert_array_equal(solver.observed_spectrum, solver.observed_counts)


def test_setup_validates_backend_after_full_initialisation(tmp_path):
    _make_solver(tmp_path, backend=_RecordingConfig(trace=False))

    assert _RecordingConfig.seen == {
        "num_parameters": 2,
        "fold_counts": {},
        "has_pool_attr": True,
        "prior_is_base": True,
    }


def test_setup_initialises_every_attribute_without_xspec(tmp_path):
    prior = MultipleIndependent([uniform(0.0, 1.0), uniform(1.0, 2.0)])
    sentinel = HistoryMatching(n_waves=2)
    solver = _bare_solver()

    solver._setup(
        prior=prior,
        parameter_indices=[1, 2],
        model_names=["m", "m"],
        parameter_names=["a", "b"],
        outputfiles_basename=str(tmp_path / "pure"),
        overwrite=False,
        n_jobs=None,
        backend=_config(),
        restrict_prior=sentinel,
    )

    from bsixsa.convenience import available_cpu_count

    assert solver.base_prior is prior and solver.prior is prior
    assert solver.distributions == {"prior": prior}
    assert solver.parameter_indices == [1, 2] and solver.model_names == ["m", "m"]
    assert solver.parameter_names == ["a", "b"]
    assert solver.posterior_samples is None and solver.fit_result is None
    assert solver.backend is None and solver.backend_name == "dummy"
    assert solver._n_jobs == available_cpu_count()
    assert solver._fold_phase_name == "backend"
    assert solver.fold_counts == {}
    assert solver.restriction_config is sentinel and solver.restriction is None
    assert (tmp_path / "pure").is_dir()


@pytest.mark.parametrize(
    ("parameter_names", "model_names"), [(["a"], ["m", "m"]), (["a", "b", "c"], ["m", "m"]), (["a", "b"], ["m"])]
)
def test_setup_refuses_names_not_aligned_with_the_indices(tmp_path, parameter_names, model_names):
    with pytest.raises(ValueError, match="one entry per fitted parameter"):
        _bare_solver()._setup(
            prior=MultipleIndependent([uniform(0.0, 1.0), uniform(1.0, 2.0)]),
            parameter_indices=[1, 2],
            model_names=model_names,
            parameter_names=parameter_names,
            outputfiles_basename=str(tmp_path / "out"),
            overwrite=False,
            n_jobs=1,
            backend=_config(),
        )
    assert not (tmp_path / "out").exists()


def test_setup_respects_overwrite_guard(tmp_path):
    target = tmp_path / "busy"
    target.mkdir()
    (target / "file.txt").write_text("x")

    with pytest.raises(FileExistsError, match="not empty"):
        _make_solver(tmp_path, name="busy", overwrite=False)


# ---------------------------------------------------------------------------
# Folding through the analytic hooks
# ---------------------------------------------------------------------------


def test_simulate_cstat_matches_spectrum(tmp_path):
    solver = _make_solver(tmp_path)
    theta = _prior_draws(solver, 17)

    out = solver.simulate(theta, return_kind="cstat", progress_bar=False)

    assert set(out) == {"cstat"}
    assert out["cstat"].shape == (17,)
    np.testing.assert_allclose(out["cstat"], solver.spectrum.cstat(theta), rtol=0, atol=0)


def test_simulate_full_model_counts_matches_spectrum(tmp_path):
    solver = _make_solver(tmp_path)
    theta = _prior_draws(solver, 9, seed=1)

    out = solver.simulate(theta)

    assert set(out) == {"cstat", "total_model_counts"}
    assert out["total_model_counts"].shape == (9, 64)
    np.testing.assert_array_equal(out["total_model_counts"], solver.spectrum.model_counts(theta))
    np.testing.assert_array_equal(out["cstat"], solver.spectrum.cstat(theta))


def test_simulate_models_and_components(tmp_path):
    solver = _make_solver(tmp_path)
    theta = _prior_draws(solver, 5, seed=2)

    out = solver.simulate(theta, return_kind="models_and_components")

    assert set(out) == {"cstat", "total_model_counts", "model_counts", "component_counts"}
    np.testing.assert_array_equal(out["model_counts"]["analytic"], out["total_model_counts"])
    assert out["component_counts"] == {}


def test_simulate_rejects_non_finite_without_counting(tmp_path):
    solver = _make_solver(tmp_path)
    theta = _prior_draws(solver, 3)
    theta[1, 0] = np.nan

    with pytest.raises(ValueError, match="Gamma"):
        solver.simulate(theta)
    assert solver.fold_counts == {}


def test_cash_statistic_is_minimal_at_the_data():
    spectrum = AnalyticSpectrum(d=2)
    k = spectrum.observed

    assert spectrum.cash(np.where(k > 0, k, 1e-300), k) == pytest.approx(0.0, abs=1e-9)
    mu = np.full(64, 3.0)
    zero = np.zeros(64)
    assert spectrum.cash(mu, zero) == pytest.approx(2.0 * mu.sum())


# ---------------------------------------------------------------------------
# Fold accounting
# ---------------------------------------------------------------------------


def test_fold_counts_follow_nested_phases(tmp_path):
    solver = _make_solver(tmp_path)
    theta = _prior_draws(solver, 10)

    solver.simulate(theta[:3], return_kind="cstat")
    with solver.fold_phase("restriction"):
        solver.simulate(theta[:5], return_kind="cstat")
        with solver.fold_phase("inner"):
            solver.log_likelihood_fn(theta[:2], None)
        solver.simulate(theta[:1], return_kind="full_model_counts")
    solver.log_likelihood_fn(theta[:4], None, no_pool=True)

    assert solver.fold_counts == {"backend": 7, "restriction": 6, "inner": 2}
    assert solver._fold_phase_name == "backend"
    assert sum(solver.fold_counts.values()) == 15
    assert [entry["phase"] for entry in solver.fold_log] == [
        "backend",
        "restriction",
        "inner",
        "restriction",
        "backend",
    ]


def test_fold_phase_restores_on_error_and_counts_are_copies(tmp_path):
    solver = _make_solver(tmp_path)

    with pytest.raises(RuntimeError):
        with solver.fold_phase("outer"):
            with solver.fold_phase("boom"):
                raise RuntimeError("fail")
    assert solver._fold_phase_name == "backend"

    solver.simulate(_prior_draws(solver, 3), return_kind="cstat")
    counts = solver.fold_counts
    counts["backend"] = 999
    assert solver.fold_counts == {"backend": 3}

    with pytest.raises(TypeError):
        with solver.fold_phase(1):
            pass


def test_log_likelihood_is_minus_half_cstat_and_counted(tmp_path):
    solver = _make_solver(tmp_path)
    theta = _prior_draws(solver, 11, seed=3)
    expected = -0.5 * solver.spectrum.cstat(theta)

    pooled = solver.log_likelihood_fn(theta, None, progress_bar=False)
    assert solver.n_pool_requests == 1
    serial = solver.log_likelihood_fn(theta, None, progress_bar=False, no_pool=True)
    assert solver.n_pool_requests == 1

    np.testing.assert_array_equal(pooled, expected)
    np.testing.assert_array_equal(serial, expected)
    assert solver.fold_counts == {"backend": 22}
    assert solver.pool is None and solver._pool_depth == 0


def test_folds_inside_an_open_pool_scope_share_it(tmp_path):
    solver = _make_solver(tmp_path)
    theta = _prior_draws(solver, 2)

    with solver.pool_scope():
        assert solver._pool_depth == 1
        solver.simulate(theta, return_kind="cstat")
        assert solver._pool_depth == 1
    assert solver._pool_depth == 0
    assert solver.fold_log[-1]["pool_depth"] == 1


def test_log_prob_fn_combines_likelihood_and_prior(tmp_path):
    solver = _make_solver(tmp_path)
    theta = _prior_draws(solver, 6, seed=4)

    expected = -0.5 * solver.spectrum.cstat(theta) + solver.prior.log_prob(theta)
    np.testing.assert_allclose(solver.log_prob_fn(theta, None), expected)
    # `x_o` is optional: it is ignored.
    np.testing.assert_allclose(solver.log_prob_fn(theta), expected)
    np.testing.assert_array_equal(solver.log_likelihood_fn(theta), -0.5 * solver.spectrum.cstat(theta))


def test_log_prior_fn_reads_working_prior(tmp_path):
    solver = _make_solver(tmp_path)
    theta = np.array([[1.5, 1.0], [1.9, 0.5]])
    narrower = MultipleIndependent([uniform(1.0, 2.0), loguniform(0.2, 5.0)])

    solver.prior = narrower

    np.testing.assert_allclose(solver.log_prior_fn(theta, None), narrower.log_prob(theta))
    assert not np.allclose(solver.log_prior_fn(theta, None), solver.distributions["prior"].log_prob(theta))
    assert solver.distributions["prior"] is solver.base_prior


# ---------------------------------------------------------------------------
# Base Solver.simulate (parallel_folding patched)
# ---------------------------------------------------------------------------


def _patched_bare_solver():
    solver = _bare_solver()
    solver.parameter_indices = [1, 2]
    solver.model_names = ["m", "m"]
    solver._parameter_names = ("a", "b")
    solver._fold_counts = collections.Counter()
    solver._fold_phase_name = "backend"
    solver.pool = None
    solver._pool_depth = 0
    solver._n_jobs = 1
    return solver


def test_base_fold_routes_through_parallel_folding():
    solver = _patched_bare_solver()
    pool = object()
    calls = []

    def fake_parallel_folding(parameter_batch, parameter_indices, model_names, **kwargs):
        calls.append(kwargs)
        return {"cstat": np.arange(len(parameter_batch), dtype=np.float64)}

    theta = np.ones((4, 2))
    with patch.object(solver_module, "parallel_folding", side_effect=fake_parallel_folding), patch.object(
        Solver, "_get_pool", return_value=pool
    ), patch.object(Solver, "close_pool"):
        out = solver.simulate(theta, return_kind="cstat", progress_bar=False)
        with solver.fold_phase("post"):
            ll = solver.log_likelihood_fn(theta[:3], None, progress_bar=False, no_pool=True)

    np.testing.assert_array_equal(out["cstat"], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(ll, [-0.0, -0.5, -1.0])
    assert calls[0]["pool"] is pool and calls[0]["return_kind"] == "cstat"
    assert calls[1]["pool"] is None and calls[1]["return_kind"] == "cstat"
    assert solver.fold_counts == {"backend": 4, "post": 3}


def test_base_fold_reports_non_finite_parameter_names():
    solver = _patched_bare_solver()
    theta = np.array([[1.0, np.inf], [1.0, 2.0]])

    with patch.object(solver_module, "parallel_folding") as folding, patch.object(
        Solver, "parameter_names", new_callable=lambda: property(lambda self: ["a", "b"])
    ):
        with pytest.raises(ValueError, match=r"b \(samples \[0\]\)"):
            solver.log_likelihood_fn(theta, None)
    folding.assert_not_called()
    assert solver.fold_counts == {}


# ---------------------------------------------------------------------------
# One fold implementation: AnalyticSolver only replaces the leaf hook
# ---------------------------------------------------------------------------


def test_analytic_solver_runs_the_production_fold():
    """Regression: AnalyticSolver used to re-implement the fold, so the real one went untested."""
    assert "simulate" not in AnalyticSolver.__dict__
    assert AnalyticSolver.simulate is Solver.simulate
    assert "_fold_batch" in AnalyticSolver.__dict__


@pytest.fixture(params=["analytic", "xspec"])
def fold_solver(request, tmp_path):
    """`(solver, fold_mock)` for the analytic solver and a bare Solver with `parallel_folding` patched."""
    if request.param == "analytic":
        yield _make_solver(tmp_path), None
        return

    solver = _patched_bare_solver()
    folding = MagicMock(side_effect=lambda params, *args, **kwargs: {"cstat": np.zeros(len(params))})
    with patch.object(solver_module, "parallel_folding", folding), patch.object(
        Solver, "_get_pool", return_value=object()
    ), patch.object(Solver, "close_pool"), patch.object(
        Solver, "parameter_names", new_callable=lambda: property(lambda self: ["a", "b"])
    ), patch.object(Solver, "n_bins", new_callable=lambda: property(lambda self: 64)):
        yield solver, folding


@pytest.mark.parametrize(
    ("return_kind", "expected_keys"),
    [
        ("cstat", {"cstat"}),
        ("full_model_counts", {"cstat", "total_model_counts"}),
        ("models_and_components", {"cstat", "total_model_counts", "model_counts", "component_counts"}),
    ],
)
def test_empty_batch_contract_is_shared(fold_solver, return_kind, expected_keys):
    """Regression: an empty batch raised `KeyError: 'cstat'` on the XSPEC solver only."""
    solver, folding = fold_solver
    for empty in (np.empty((0, 2)), np.empty((0,)), []):
        out = solver.simulate(empty, return_kind=return_kind, progress_bar=False)

        assert set(out) == expected_keys
        assert out["cstat"].shape == (0,) and out["cstat"].dtype == np.float64
        if "total_model_counts" in out:
            assert out["total_model_counts"].shape == (0, 64)
        if return_kind == "models_and_components":
            assert out["model_counts"] == {} and out["component_counts"] == {}
    assert solver.log_likelihood_fn(np.empty((0, 2)), None).shape == (0,)
    assert solver.fold_counts == {}
    if folding is not None:
        folding.assert_not_called()
    else:
        assert solver.fold_log == [] and solver.n_pool_requests == 0


def test_fold_rejects_bad_shapes_and_return_kinds(fold_solver):
    """Regression: a 1-D vector was counted as d folds and iterated coordinate by coordinate."""
    solver, folding = fold_solver
    with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
        solver.simulate(np.array([1.8, 1.0]), return_kind="cstat")
    with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
        solver.log_likelihood_fn(np.ones((3, 3)), None)
    with pytest.raises(ValueError, match="return_kind"):
        solver.simulate(np.ones((3, 2)), return_kind="counts")
    assert solver.fold_counts == {}
    if folding is not None:
        folding.assert_not_called()


def test_analytic_fold_log_keeps_pool_semantics(tmp_path):
    solver = _make_solver(tmp_path)
    theta = _prior_draws(solver, 3)

    solver.simulate(theta, return_kind="cstat")
    solver.log_likelihood_fn(theta, None, no_pool=True)
    with solver.pool_scope():
        solver.log_likelihood_fn(theta, None, no_pool=True)

    assert [(e["no_pool"], e["pool_depth"]) for e in solver.fold_log] == [(False, 0), (True, 0), (True, 1)]
    assert solver.n_pool_requests == 1
    assert solver.pool is None


# ---------------------------------------------------------------------------
# Constructor ordering and argument checks
# ---------------------------------------------------------------------------


@pytest.fixture
def loaded_fake_xspec(monkeypatch):
    fake = MagicMock()
    fake.AllData.nSpectra = 1
    fake.AllModels.sources = {1: "m"}
    monkeypatch.setattr(solver_module, "xspec", fake)
    return fake


def test_constructor_checks_the_output_directory_before_build_prior(tmp_path, loaded_fake_xspec):
    """Regression: build_prior set the xspec parameters before FileExistsError was raised."""
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "file.txt").write_text("x")
    not_a_dir = tmp_path / "file"
    not_a_dir.write_text("x")

    with patch.object(solver_module, "build_prior") as build_prior:
        with pytest.raises(FileExistsError, match="not empty"):
            Solver([], outputfiles_basename=str(busy), backend=_config())
        with pytest.raises(ValueError, match="not a directory"):
            Solver([], outputfiles_basename=str(not_a_dir), backend=_config())
        with pytest.raises(TypeError, match="restrict_prior"):
            Solver([], outputfiles_basename=str(tmp_path / "new"), backend=_config(), restrict_prior="yes")
    build_prior.assert_not_called()
    assert (busy / "file.txt").exists()
    assert not (tmp_path / "new").exists()


def test_parameter_names_are_read_once_at_construction(tmp_path, loaded_fake_xspec):
    """Regression: the names were re-read from XSPEC at every access, so freezing a
    parameter after construction silently relabelled the posterior columns."""
    free = [SimpleNamespace(name="TBabs_1_nH"), SimpleNamespace(name="powerlaw_2_PhoIndex")]
    prior = MultipleIndependent([uniform(0.0, 1.0), uniform(1.0, 3.0)])

    with patch.object(solver_module, "build_prior", return_value=(prior, [1, 2], ["", ""])), patch.object(
        solver_module, "iter_free_parameters", side_effect=lambda: iter(list(free))
    ) as iter_free:
        solver = Solver([], outputfiles_basename=str(tmp_path / "run"), backend=_config())
        free.pop()  # the user freezes PhoIndex after building the solver

        assert solver.parameter_names == ["TBabs_1_nH", "powerlaw_2_PhoIndex"]
    iter_free.assert_called_once_with()
    assert solver.parameter_indices == [1, 2] and solver.model_names == ["", ""]


@pytest.mark.parametrize(
    ("backend", "message"),
    [
        (Nessai, r"write backend=Nessai\(\) \(with any options\), not backend=Nessai\."),
        ("nautilus", r"backend must be a backend configuration such as bsixsa\.Nautilus\(\).*got 'nautilus'"),
        (BackendConfig(), "backend must be a backend configuration"),
    ],
)
def test_constructor_explains_a_wrong_backend_argument(tmp_path, loaded_fake_xspec, backend, message):
    with patch.object(solver_module, "build_prior") as build_prior:
        with pytest.raises(TypeError, match=message):
            Solver([], outputfiles_basename=str(tmp_path / "run"), backend=backend)
    build_prior.assert_not_called()


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [("nSpectra", 0, r"No spectrum is loaded.*xspec\.Spectrum"), ("sources", {}, r"No model is defined.*xspec\.Model")],
)
def test_constructor_explains_an_empty_xspec_session(tmp_path, loaded_fake_xspec, attribute, value, message):
    owner = loaded_fake_xspec.AllData if attribute == "nSpectra" else loaded_fake_xspec.AllModels
    setattr(owner, attribute, value)

    with pytest.raises(ValueError, match=message):
        Solver([], outputfiles_basename=str(tmp_path / "run"), backend=_config())


@pytest.mark.parametrize("value", ["yes", 1, 0, 0.5, [True], {"n": 1}, np.float64(1.0)])
def test_restrict_prior_rejects_plain_values_before_touching_the_directory(tmp_path, value):
    target = tmp_path / "out"
    target.mkdir()
    (target / "keep.txt").write_text("x")

    with pytest.raises(TypeError, match="restrict_prior"):
        _make_solver(tmp_path, overwrite=True, restrict_prior=value)
    assert (target / "keep.txt").exists()


@pytest.mark.parametrize(("value", "restricted"), [(None, False), (True, True), (False, False), (np.True_, True)])
def test_restrict_prior_accepts_none_and_bools(tmp_path, value, restricted):
    solver = _make_solver(tmp_path, restrict_prior=value)

    assert solver.restriction is None
    assert solver.restriction_config == (HistoryMatching() if restricted else None)


def test_restrict_prior_accepts_configuration_objects(tmp_path):
    config = HistoryMatching(n_waves=2)
    solver = _make_solver(tmp_path, restrict_prior=config)
    assert solver.restriction_config is config


# ---------------------------------------------------------------------------
# XSPEC hooks under a mocked module
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_xspec(monkeypatch):
    fake = MagicMock()
    monkeypatch.setattr(solver_module, "xspec", fake)
    return fake


@pytest.mark.parametrize("value", ["on", "ON", "cons", "jeffreys"])
def test_check_xspec_state_rejects_bayes_on(fake_xspec, value):
    fake_xspec.Fit.bayes = value

    with pytest.raises(RuntimeError, match="twice"):
        _bare_solver()._check_xspec_state()


@pytest.mark.parametrize("value", ["off", "OFF"])
def test_check_xspec_state_accepts_off(fake_xspec, value):
    fake_xspec.Fit.bayes = value

    _bare_solver()._check_xspec_state()


def test_analytic_solver_check_is_a_noop(tmp_path, fake_xspec):
    fake_xspec.Fit.bayes = "on"

    assert _make_solver(tmp_path)._check_xspec_state() is None


class _PyXspecSpectrum:
    """Mimics PyXspec's `Spectrum`: reading `background` raises when none is loaded."""

    def __init__(self, background_file=None, **attributes):
        self._background_file = background_file
        self.__dict__.update(attributes)

    @property
    def background(self):
        if self._background_file is None:
            # Message of PySpectrum.cxx getBackgrnd (HEASoft 6.34).
            raise Exception("Error: Spectrum has no background.")
        return SimpleNamespace(fileName=self._background_file)


def test_xspec_constraints_read_real_types(fake_xspec):
    """Regression: PyXspec raises on a missing background, which was reported as unknown."""
    fake_xspec.AllData.nSpectra = 1
    fake_xspec.Fit.statMethod = "cstat"
    fake_xspec.AllData.return_value = _PyXspecSpectrum()

    assert _bare_solver()._xspec_constraints() == {
        "n_spectra": 1,
        "stat_method": "cstat",
        "has_background": False,
    }

    fake_xspec.AllData.return_value = _PyXspecSpectrum(background_file="bkg.pha")
    assert _bare_solver()._xspec_constraints()["has_background"] is True

    fake_xspec.AllData.return_value = SimpleNamespace(background=None)
    assert _bare_solver()._xspec_constraints()["has_background"] is False


def _load_fake_spectra(monkeypatch, fake_xspec, *spectra) -> None:
    """Make the fake session hold *spectra*; bsixsa.xspec reads the spectra for the solver."""
    monkeypatch.setattr(xspec_module, "xspec", fake_xspec)
    fake_xspec.AllData.nSpectra = len(spectra)
    fake_xspec.AllData.side_effect = lambda index: spectra[index - 1]


def test_observed_counts_are_rounded_float64(fake_xspec, monkeypatch):
    _load_fake_spectra(monkeypatch, fake_xspec, SimpleNamespace(values=[0.12, 0.26, 0.0], exposure=100.0 + 1e-9))

    solver = _bare_solver()
    counts = solver.observed_counts

    assert counts.dtype == np.float64
    np.testing.assert_array_equal(counts, [12.0, 26.0, 0.0])
    assert solver.n_bins == 3
    # The old name returns the same exact counts (it used to be float32 rate x exposure).
    np.testing.assert_array_equal(solver.observed_spectrum, counts)
    assert solver.observed_spectrum.dtype == np.float64


def test_observed_counts_concatenate_the_spectra(fake_xspec, monkeypatch):
    _load_fake_spectra(
        monkeypatch,
        fake_xspec,
        SimpleNamespace(values=[0.12, 0.26], exposure=100.0),
        SimpleNamespace(values=[0.5, 0.25, 0.0], exposure=20.0),
    )

    solver = _bare_solver()

    np.testing.assert_array_equal(solver.observed_counts, [12.0, 26.0, 10.0, 5.0, 0.0])
    assert solver.n_bins == 5
