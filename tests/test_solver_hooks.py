"""XSPEC-free tests of the Solver hooks, `_setup` split and fold accounting."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from _analytic import AnalyticSolver, AnalyticSpectrum

import bsixsa.solver as solver_module
from bsixsa.backend.config import BackendConfig
from bsixsa.priors import MultipleIndependent, loguniform, uniform
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
    """A `Solver` built without `__init__`, as the existing test fixtures do."""
    with patch.object(Solver, "__init__", lambda self, *args, **kwargs: None):
        return Solver.__new__(Solver)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_analytic_solver_builds(tmp_path):
    solver = _make_solver(tmp_path, restrict_prior=True)

    assert solver.num_parameters == 2
    assert solver.parameter_names == ["Gamma", "norm"]
    assert solver.n_bins == 64
    assert solver.backend_name == "dummy"
    assert solver.base_prior is solver.prior
    assert solver.distributions == {"prior": solver.prior}
    assert solver.fold_counts == {}
    assert solver.restriction is None
    assert solver.restriction_config is None
    assert solver._restrict_prior_arg is True
    assert solver.pool is None and solver._pool_depth == 0
    assert solver._n_jobs == 1
    assert os.path.isdir(solver.outputfiles_basename)
    assert solver.outputfiles_basename.endswith(os.sep)
    assert solver.observed_counts.dtype == np.float64
    assert solver.observed_spectrum.dtype == np.float32
    np.testing.assert_array_equal(solver.observed_counts, solver.spectrum.observed)


def test_setup_validates_backend_after_full_initialisation(tmp_path):
    _make_solver(tmp_path, backend=_RecordingConfig(trace=False))

    assert _RecordingConfig.seen == {
        "num_parameters": 2,
        "fold_counts": {},
        "has_pool_attr": True,
        "prior_is_base": True,
    }


def test_setup_is_pure_and_does_not_touch_prior_dists(tmp_path):
    prior = SimpleNamespace(ndim=2, bounds=[(0.0, 1.0), (1.0, 2.0)])
    sentinel = object()
    solver = _bare_solver()

    solver._setup(
        prior=prior,
        indexes=[1, 2],
        model_indexes=["m", "m"],
        outputfiles_basename=str(tmp_path / "pure"),
        overwrite=False,
        n_jobs=None,
        backend=_config(),
        restrict_prior=sentinel,
    )

    from bsixsa.convenience import available_cpu_count

    assert solver.base_prior is prior and solver.prior is prior
    assert solver.distributions == {"prior": prior}
    assert solver.indexes == [1, 2] and solver.model_indexes == ["m", "m"]
    assert solver.posterior_samples is None and solver.fit_result is None
    assert solver.backend is None and solver.backend_name == "dummy"
    assert solver._n_jobs == available_cpu_count()
    assert solver._fold_phase_name == "backend"
    assert solver.fold_counts == {}
    assert solver._restrict_prior_arg is sentinel
    assert (tmp_path / "pure").is_dir()


def test_setup_respects_overwrite_guard(tmp_path):
    target = tmp_path / "busy"
    target.mkdir()
    (target / "file.txt").write_text("x")

    with pytest.raises(FileExistsError, match="not empty"):
        _make_solver(tmp_path, name="busy", overwrite=False)


def test_bare_solver_class_defaults():
    solver = _bare_solver()

    assert solver.fold_counts == {}
    assert solver.restriction is None
    assert solver.restriction_config is None
    assert solver._fold_phase_name == "backend"
    assert solver._restrict_prior_arg is None

    solver._count_folds(4)
    with solver.fold_phase("post"):
        solver._count_folds(2)
    assert solver.fold_counts == {"backend": 4, "post": 2}
    # The lazily created dict belongs to the instance, not the class.
    assert _bare_solver().fold_counts == {}


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

    solver._count_folds(3)
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


def test_log_prior_fn_reads_working_prior(tmp_path):
    solver = _make_solver(tmp_path)
    theta = np.array([[1.5, 1.0], [1.9, 0.5]])
    narrower = MultipleIndependent([uniform(1.0, 2.0), loguniform(0.2, 5.0)])

    solver.prior = narrower

    np.testing.assert_allclose(solver.log_prior_fn(theta, None), narrower.log_prob(theta))
    assert not np.allclose(solver.log_prior_fn(theta, None), solver.distributions["prior"].log_prob(theta))
    assert solver.distributions["prior"] is solver.base_prior


# ---------------------------------------------------------------------------
# Base Solver._fold (parallel_folding patched)
# ---------------------------------------------------------------------------


def _patched_bare_solver():
    solver = _bare_solver()
    solver.indexes = [1, 2]
    solver.model_indexes = ["m", "m"]
    solver.pool = None
    solver._pool_depth = 0
    solver._n_jobs = 1
    return solver


def test_base_fold_routes_through_parallel_folding():
    solver = _patched_bare_solver()
    pool = object()
    calls = []

    def fake_parallel_folding(params, indexes, model_indexes, **kwargs):
        calls.append(kwargs)
        return {"cstat": np.arange(len(params), dtype=np.float64)}

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
    """Regression: AnalyticSolver used to re-implement `_fold`, so the real one went untested."""
    assert "_fold" not in AnalyticSolver.__dict__
    assert AnalyticSolver._fold is Solver._fold
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


@pytest.mark.parametrize("value", ["yes", 1, 0, 0.5, [True], {"n": 1}, np.float64(1.0)])
def test_restrict_prior_rejects_plain_values_before_touching_the_directory(tmp_path, value):
    target = tmp_path / "out"
    target.mkdir()
    (target / "keep.txt").write_text("x")

    with pytest.raises(TypeError, match="restrict_prior"):
        _make_solver(tmp_path, overwrite=True, restrict_prior=value)
    assert (target / "keep.txt").exists()


@pytest.mark.parametrize("value", [None, True, False, np.True_])
def test_restrict_prior_accepts_none_and_bools(tmp_path, value):
    assert _make_solver(tmp_path, restrict_prior=value)._restrict_prior_arg is value


def test_restrict_prior_accepts_configuration_objects(tmp_path):
    config = SimpleNamespace(kind="history-matching")
    assert _make_solver(tmp_path, restrict_prior=config)._restrict_prior_arg is config


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


class _WiderGammaSpectrum(AnalyticSpectrum):
    def prior_list(self):
        dists = super().prior_list()
        dists[0] = uniform(1.0, 3.5)
        return dists


class _FloatBoundsSpectrum(AnalyticSpectrum):
    def prior_list(self):
        dists = super().prior_list()
        dists[0] = uniform(np.float64(1.0), np.float64(3.0))
        return dists


def test_fingerprint_is_stable_for_identical_problems(tmp_path):
    first = _make_solver(tmp_path, name="a")
    second = _make_solver(tmp_path, name="b", spectrum=_FloatBoundsSpectrum(d=2))

    assert len(first.fingerprint()) == 64
    assert first.fingerprint() == first.fingerprint()
    assert first.fingerprint() == second.fingerprint()


def test_fingerprint_changes_with_spectrum_or_prior(tmp_path):
    reference = _make_solver(tmp_path, name="ref").fingerprint()

    other_seed = _make_solver(tmp_path, name="seed", spectrum=AnalyticSpectrum(d=2, seed=1)).fingerprint()
    wider = _make_solver(tmp_path, name="wider", spectrum=_WiderGammaSpectrum(d=2)).fingerprint()
    three_d = _make_solver(tmp_path, name="d3", spectrum=AnalyticSpectrum(d=3)).fingerprint()

    assert len({reference, other_seed, wider, three_d}) == 4


def test_fingerprint_ignores_the_working_prior(tmp_path):
    solver = _make_solver(tmp_path)
    before = solver.fingerprint()

    solver.prior = MultipleIndependent([uniform(1.5, 2.0), loguniform(0.5, 2.0)])

    assert solver.fingerprint() == before


def test_fingerprint_changes_with_n_spectra(tmp_path):
    solver = _make_solver(tmp_path)
    before = solver.fingerprint()

    with patch.object(
        AnalyticSolver,
        "_xspec_constraints",
        lambda self: {"n_spectra": 2, "stat_method": "cstat", "has_background": False},
    ):
        assert solver.fingerprint() != before


def _fingerprint_with_base(solver, dists):
    solver.base_prior = MultipleIndependent(dists)
    return solver.fingerprint()


def test_fingerprint_normalises_positional_and_keyword_loc_scale(tmp_path):
    from scipy import stats

    solver = _make_solver(tmp_path)
    positional = _fingerprint_with_base(solver, [stats.uniform(1.0, 2.0), loguniform(0.1, 10.0)])
    keyword = _fingerprint_with_base(solver, [stats.uniform(loc=1.0, scale=2.0), loguniform(0.1, 10.0)])
    shifted = _fingerprint_with_base(solver, [stats.uniform(loc=1.5, scale=1.5), loguniform(0.1, 10.0)])

    assert positional == keyword
    assert shifted != positional


def test_fingerprint_describes_the_base_of_a_truncated_marginal(tmp_path):
    """A `TruncatedMarginal` has empty `args`: the base family must enter the hash."""
    from scipy import stats

    from bsixsa.priors import TruncatedMarginal

    solver = _make_solver(tmp_path)
    # Both marginals have support [0, 1]; only the wrapped family differs.
    beta_23 = TruncatedMarginal(stats.beta(2.0, 3.0), 0.0, 1.0)
    beta_32 = TruncatedMarginal(stats.beta(3.0, 2.0), 0.0, 1.0)
    assert beta_23.support() == beta_32.support()

    first = _fingerprint_with_base(solver, [beta_23, loguniform(0.1, 10.0)])
    second = _fingerprint_with_base(solver, [beta_32, loguniform(0.1, 10.0)])
    narrower = _fingerprint_with_base(
        solver, [TruncatedMarginal(stats.beta(2.0, 3.0), 0.0, 0.5), loguniform(0.1, 10.0)]
    )

    assert first != second
    assert narrower not in (first, second)
    assert first == _fingerprint_with_base(solver, [TruncatedMarginal(stats.beta(2, 3), 0, 1), loguniform(0.1, 10.0)])


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


@pytest.mark.parametrize("value", ["off", "OFF", None])
def test_check_xspec_state_accepts_off(fake_xspec, value):
    fake_xspec.Fit.bayes = value

    _bare_solver()._check_xspec_state()


def test_check_xspec_state_is_inert_under_magicmock(fake_xspec):
    assert isinstance(fake_xspec.Fit.bayes, MagicMock)

    _bare_solver()._check_xspec_state()


def test_analytic_solver_check_is_a_noop(tmp_path, fake_xspec):
    fake_xspec.Fit.bayes = "on"

    assert _make_solver(tmp_path)._check_xspec_state() is None


def test_xspec_constraints_are_none_under_magicmock(fake_xspec):
    assert _bare_solver()._xspec_constraints() == {
        "n_spectra": None,
        "stat_method": None,
        "has_background": None,
    }


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


def test_xspec_constraints_other_background_errors_stay_unknown(fake_xspec):
    class _Broken:
        @property
        def background(self):
            raise Exception("Error: Cannot parse getBackgrnd arguments.")

    fake_xspec.AllData.return_value = _Broken()
    assert _bare_solver()._xspec_constraints()["has_background"] is None


# ---------------------------------------------------------------------------
# Problem state entering the fingerprint
# ---------------------------------------------------------------------------


class _FakeParameter(SimpleNamespace):
    pass


class _FakeModel:
    def __init__(self, expression, parameters):
        self.expression = expression
        self.nParameters = len(parameters)
        self._parameters = parameters

    def __call__(self, index):
        return self._parameters[index - 1]


def _install_problem(fake_xspec, *, exposure=1000.0, rmf="pn.rmf", nh=0.05, thawed=1.9, abund="wilm"):
    spectrum = _PyXspecSpectrum(
        fileName="obs.pha",
        exposure=exposure,
        response=SimpleNamespace(rmf=rmf, arf="pn.arf"),
        noticed=[3, 4, 5],
    )
    model = _FakeModel(
        "TBabs*powerlaw",
        [
            _FakeParameter(frozen=True, link="", values=[nh, 0.01]),  # 1 nH, fixed
            _FakeParameter(frozen=False, link="", values=[thawed, 0.01]),  # 2 PhoIndex, folded
            _FakeParameter(frozen=False, link="", values=[3e-3, 0.01]),  # 3 norm, folded
            _FakeParameter(frozen=False, link="= 2", values=[thawed, 0.0]),  # 4 linked
            _FakeParameter(frozen=False, link="", values=[7.0, 0.01]),  # 5 thawed, no prior
        ],
    )
    fake_xspec.AllData.nSpectra = 1
    fake_xspec.AllData.side_effect = lambda index: spectrum
    fake_xspec.AllModels.sources = {1: "src"}
    fake_xspec.AllModels.side_effect = lambda group, modName="": model
    fake_xspec.Xset.abund = abund
    fake_xspec.Xset.xsect = "vern"
    fake_xspec.Xset.cosmo = "70.0 0.0 0.73"


def _problem_state(fake_xspec, **kwargs):
    _install_problem(fake_xspec, **kwargs)
    solver = _bare_solver()
    solver.indexes = [2, 3]
    solver.model_indexes = ["src", "src"]
    return solver._xspec_problem_state()


def test_xspec_problem_state_reads_responses_and_fixed_parameters(fake_xspec):
    state = _problem_state(fake_xspec)

    assert state == {
        "spectra": [
            {
                "file": "obs.pha",
                "exposure": 1000.0,
                "rmf": "pn.rmf",
                "arf": "pn.arf",
                "background": False,
                "noticed": [3, 4, 5],
            }
        ],
        "models": [
            {
                "source": 1,
                "name": "src",
                "expression": "TBabs*powerlaw",
                # Folded (2, 3) and plain thawed (5) parameters are excluded;
                # the linked one contributes its link only.
                "fixed": [[1, 0.05, ""], [4, None, "= 2"]],
            }
        ],
        "settings": {"abund": "wilm", "xsect": "vern", "cosmo": "70.0 0.0 0.73"},
    }


def test_xspec_problem_state_tracks_what_changes_the_likelihood(fake_xspec):
    reference = _problem_state(fake_xspec)

    assert _problem_state(fake_xspec, exposure=2000.0) != reference
    assert _problem_state(fake_xspec, rmf="mos.rmf") != reference
    assert _problem_state(fake_xspec, nh=0.5) != reference
    assert _problem_state(fake_xspec, abund="angr") != reference
    # Folding moves the thawed values; they must not change the state.
    assert _problem_state(fake_xspec, thawed=2.7) == reference


def test_xspec_problem_state_is_inert_under_magicmock(fake_xspec):
    solver = _bare_solver()
    solver.indexes, solver.model_indexes = [1], ["m"]

    assert solver._xspec_problem_state() == {
        "spectra": [],
        "models": [],
        "settings": {"abund": None, "xsect": None, "cosmo": None},
    }


def test_fingerprint_changes_with_statistic_background_and_problem_state(tmp_path):
    """Regression: the hash ignored everything but the counts, names, priors and nSpectra."""
    solver = _make_solver(tmp_path)
    reference = solver.fingerprint()
    constraints = {"n_spectra": 1, "stat_method": "cstat", "has_background": False}

    with patch.object(AnalyticSolver, "_xspec_constraints", lambda self: {**constraints, "stat_method": "pstat"}):
        assert solver.fingerprint() != reference
    with patch.object(AnalyticSolver, "_xspec_constraints", lambda self: {**constraints, "has_background": True}):
        assert solver.fingerprint() != reference
    assert solver.fingerprint() == reference

    # Same observed counts, different forward model (an exposure-like rescale).
    rescaled = _make_solver(tmp_path, name="rescaled")
    rescaled.spectrum.amplitude *= 1.5
    np.testing.assert_array_equal(rescaled.observed_counts, solver.observed_counts)
    assert rescaled.fingerprint() != reference


def test_fingerprint_separates_opaque_families_by_their_quantiles(tmp_path):
    """`rv_histogram` keeps its histogram out of `args`; only its values tell two apart."""
    from scipy import stats

    solver = _make_solver(tmp_path)
    edges = np.linspace(0.0, 1.0, 5)
    rising = stats.rv_histogram((np.array([1.0, 2.0, 3.0, 4.0]), edges))
    falling = stats.rv_histogram((np.array([4.0, 3.0, 2.0, 1.0]), edges))
    assert rising.support() == falling.support()

    first = _fingerprint_with_base(solver, [rising, loguniform(0.1, 10.0)])
    second = _fingerprint_with_base(solver, [falling, loguniform(0.1, 10.0)])
    again = _fingerprint_with_base(
        solver, [stats.rv_histogram((np.array([1.0, 2.0, 3.0, 4.0]), edges)), loguniform(0.1, 10.0)]
    )

    assert first != second
    assert first == again


def test_canonical_form_drops_memory_addresses():
    first, second = object(), object()
    assert solver_module._canonical(first) == solver_module._canonical(second) == "object"
    assert solver_module._canonical(SimpleNamespace(a=1)) == "namespace(a=1)"


def test_xspec_constraints_tolerate_errors_and_bools(fake_xspec):
    fake_xspec.AllData.nSpectra = True
    fake_xspec.AllData.side_effect = RuntimeError("no spectrum")

    constraints = _bare_solver()._xspec_constraints()

    assert constraints["n_spectra"] is None
    assert constraints["has_background"] is None


def test_observed_counts_are_rounded_float64(fake_xspec):
    spectrum = fake_xspec.AllData.return_value
    spectrum.values = [0.12, 0.26, 0.0]
    spectrum.exposure = 100.0 + 1e-9

    solver = _bare_solver()
    counts = solver.observed_counts

    assert counts.dtype == np.float64
    np.testing.assert_array_equal(counts, [12.0, 26.0, 0.0])
    assert solver.n_bins == 3
