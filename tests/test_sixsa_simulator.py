"""Tests for `bsixsa.backend.sixsa._simulator.CubeSimulator` on the analytic solver."""

from __future__ import annotations

import numpy as np
import pytest

from _analytic import AnalyticSolver, AnalyticSpectrum

from bsixsa._cube import child_rng, to_physical
from bsixsa.backend.config import Nautilus
from bsixsa.backend.sixsa._simulator import CubeSimulator
from bsixsa.exceptions import FoldError


@pytest.fixture
def spectrum():
    return AnalyticSpectrum(d=2)


@pytest.fixture
def solver(tmp_path, spectrum):
    return AnalyticSolver(spectrum, backend=Nautilus(trace=False), outputfiles_basename=tmp_path / "out")


def _simulator(solver, seed=0, **kwargs):
    return CubeSimulator(solver, seed_seq=np.random.SeedSequence(seed), **kwargs)


def _cube_points(n, d=2, seed=1):
    return np.random.default_rng(seed).random((n, d))


class _RecordingTracer:
    def __init__(self):
        self.calls = []

    def record(self, params, cstat):
        self.calls.append((np.array(params, copy=True), np.array(cstat, copy=True)))


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def test_attributes_mirror_the_solver(solver, spectrum):
    sim = _simulator(solver)

    assert sim.d == 2
    assert sim.n_bins == 64
    assert sim.x_obs.dtype == np.float64
    np.testing.assert_array_equal(sim.x_obs, spectrum.observed)
    assert not sim.x_obs.flags.writeable
    assert sim.seconds == {}


def test_constructor_validates_arguments(solver):
    with pytest.raises(TypeError, match="SeedSequence"):
        CubeSimulator(solver, seed_seq=0)
    with pytest.raises(ValueError, match="fold_chunk"):
        _simulator(solver, fold_chunk=0)
    with pytest.raises(TypeError, match="fold_chunk"):
        _simulator(solver, fold_chunk=2.5)


# ---------------------------------------------------------------------------
# cstat
# ---------------------------------------------------------------------------


def test_cstat_matches_the_analytic_statistic(solver, spectrum):
    sim = _simulator(solver)
    u = _cube_points(50)

    values = sim.cstat(u, phase="screen")

    assert values.dtype == np.float64
    assert values.shape == (50,)
    expected = spectrum.cstat(to_physical(solver.base_prior, u))
    np.testing.assert_allclose(values, expected, rtol=0, atol=0)
    assert solver.fold_counts == {"screen": 50}
    assert sim.seconds["screen"] > 0.0


def test_cstat_out_of_cube_rows_are_infinite_and_not_folded(solver, spectrum):
    sim = _simulator(solver)
    u = _cube_points(10)
    u[1] = [-1e-12, 0.5]
    u[3] = [0.5, 1.0 + 1e-12]
    u[5] = [np.nan, 0.5]
    u[7] = [np.inf, 0.2]
    u[8] = [0.0, 1.0]  # the closed cube keeps its faces

    values = sim.cstat(u, phase="screen")

    outside = [1, 3, 5, 7]
    inside = [i for i in range(10) if i not in outside]
    assert np.all(values[outside] == np.inf)
    expected = spectrum.cstat(to_physical(solver.base_prior, u[inside]))
    np.testing.assert_array_equal(values[inside], expected)
    assert solver.fold_counts == {"screen": len(inside)}


def test_cstat_all_rows_outside_folds_nothing(solver):
    sim = _simulator(solver)

    values = sim.cstat(np.full((4, 2), 2.0), phase="screen")

    assert np.all(values == np.inf)
    assert solver.fold_counts == {}
    assert solver.fold_log == []


def test_cstat_chunking_gives_identical_results_and_fold_counts(tmp_path, spectrum):
    u = _cube_points(33)
    u[4] = [1.5, 0.5]
    results, counts, logs = [], [], []
    for chunk in (10_000, 7):
        solver = AnalyticSolver(spectrum, backend=Nautilus(trace=False), outputfiles_basename=tmp_path / f"c{chunk}")
        sim = _simulator(solver, fold_chunk=chunk)
        results.append(sim.cstat(u, phase="screen"))
        counts.append(solver.fold_counts)
        logs.append([entry["n"] for entry in solver.fold_log])

    np.testing.assert_array_equal(results[0], results[1])
    assert counts[0] == counts[1] == {"screen": 32}
    assert logs == [[32], [7, 7, 7, 7, 4]]


def test_cstat_empty_input_folds_nothing(solver):
    sim = _simulator(solver)

    for empty in (np.empty((0, 2)), np.empty(0), []):
        values = sim.cstat(empty, phase="screen")
        assert values.shape == (0,)
        assert values.dtype == np.float64
    assert solver.fold_counts == {}


@pytest.mark.parametrize("bad", [np.zeros(2), np.zeros((3, 3)), np.zeros((2, 2, 2)), np.empty((0, 3))])
def test_cstat_rejects_bad_shapes(solver, bad):
    with pytest.raises(ValueError, match="shape"):
        _simulator(solver).cstat(bad, phase="screen")


def test_phase_is_attributed_and_restored(solver):
    sim = _simulator(solver)
    u = _cube_points(5)

    with solver.fold_phase("outer"):
        sim.cstat(u, phase="screen")
        sim.counts(u, phase="training", key=(1, 0))
        assert solver._fold_phase_name == "outer"

    assert solver.fold_counts == {"screen": 5, "training": 5}
    assert set(sim.seconds) == {"screen", "training"}


@pytest.mark.parametrize("bad_value", [np.nan, -np.inf])
def test_cstat_invalid_fold_raises_fold_error(solver, monkeypatch, bad_value):
    original = solver._analytic_outputs

    def broken(parameters, return_kind):
        outputs = original(parameters, return_kind)
        outputs["cstat"] = outputs["cstat"].copy()
        outputs["cstat"][1] = bad_value
        return outputs

    monkeypatch.setattr(solver, "_analytic_outputs", broken)
    tracer = _RecordingTracer()
    sim = _simulator(solver, tracer=tracer)

    with pytest.raises(FoldError, match="invalid C-stat"):
        sim.cstat(_cube_points(4), phase="screen")
    with pytest.raises(FoldError, match="invalid C-stat"):
        sim.counts(_cube_points(4), phase="training", key=(0,))
    assert tracer.calls == []


def test_patched_fold_returning_nan_raises(solver, monkeypatch):
    def nan_fold(parameters, return_kind="cstat", progress_bar=True, no_pool=False):
        n = len(parameters)
        return {"cstat": np.full(n, np.nan), "total_model_counts": np.ones((n, 64))}

    monkeypatch.setattr(solver, "_fold", nan_fold)

    with pytest.raises(FoldError):
        _simulator(solver).cstat(_cube_points(3), phase="screen")


def test_positive_infinite_fold_is_a_legitimate_zero_likelihood(solver, monkeypatch):
    original = solver._analytic_outputs

    def infinite(parameters, return_kind):
        outputs = original(parameters, return_kind)
        outputs["cstat"] = np.full(len(parameters), np.inf)
        return outputs

    monkeypatch.setattr(solver, "_analytic_outputs", infinite)

    values = _simulator(solver).cstat(_cube_points(3), phase="screen")

    assert np.all(values == np.inf)
    assert solver.fold_counts == {"screen": 3}


def test_wrong_length_fold_raises(solver, monkeypatch):
    monkeypatch.setattr(solver, "_fold", lambda parameters, **kwargs: {"cstat": np.zeros(len(parameters) + 1)})

    with pytest.raises(FoldError, match="C-stat values"):
        _simulator(solver).cstat(_cube_points(3), phase="screen")


# ---------------------------------------------------------------------------
# counts
# ---------------------------------------------------------------------------


def test_counts_shapes_dtypes_and_cstat(solver, spectrum):
    sim = _simulator(solver)
    u = _cube_points(20)

    cstat, counts = sim.counts(u, phase="training", key=(3, 1))

    assert cstat.dtype == np.float64 and cstat.shape == (20,)
    assert counts.dtype == np.float32 and counts.shape == (20, 64)
    np.testing.assert_array_equal(cstat, spectrum.cstat(to_physical(solver.base_prior, u)))
    assert np.all(counts >= 0) and np.all(counts == np.round(counts))
    assert solver.fold_counts == {"training": 20}


def test_counts_uses_one_child_stream_per_chunk(solver, spectrum):
    seed_seq = np.random.SeedSequence(42)
    sim = CubeSimulator(solver, seed_seq=seed_seq, fold_chunk=4)
    u = _cube_points(10)

    _, counts = sim.counts(u, phase="training", key=(2, 5))

    mu = spectrum.model_counts(to_physical(solver.base_prior, u))
    for j, (lo, hi) in enumerate([(0, 4), (4, 8), (8, 10)]):
        expected = child_rng(seed_seq, 2, 5, j).poisson(np.clip(mu[lo:hi], 0.0, None)).astype(np.float32)
        np.testing.assert_array_equal(counts[lo:hi], expected)
    assert [entry["n"] for entry in solver.fold_log] == [4, 4, 2]


def test_counts_is_deterministic_per_key(tmp_path, spectrum):
    u = _cube_points(30)
    outputs = []
    for run in range(2):
        solver = AnalyticSolver(spectrum, backend=Nautilus(trace=False), outputfiles_basename=tmp_path / f"r{run}")
        sim = _simulator(solver, seed=7)
        outputs.append((sim.counts(u, phase="training", key=(1, 2))[1], sim.counts(u, phase="training", key=(1, 3))[1]))

    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
    assert not np.array_equal(outputs[0][0], outputs[0][1])


def test_counts_differs_between_seeds(solver):
    u = _cube_points(10)

    a = _simulator(solver, seed=0).counts(u, phase="training", key=(0,))[1]
    b = _simulator(solver, seed=1).counts(u, phase="training", key=(0,))[1]

    assert not np.array_equal(a, b)


def test_counts_poisson_mean_matches_the_model(solver, spectrum):
    sim = _simulator(solver, seed=3, fold_chunk=5000)
    u = np.tile([[0.4, 0.6]], (20_000, 1))

    _, counts = sim.counts(u, phase="training", key=(9,))

    mu = spectrum.model_counts(to_physical(solver.base_prior, u[:1]))[0]
    mean = counts.astype(np.float64).mean(axis=0)
    # Poisson: the standard error of the mean is sqrt(mu / n).
    z = (mean - mu) / np.sqrt(np.maximum(mu, 1e-12) / len(u))
    assert np.max(np.abs(z)) < 5.0
    variance = counts.astype(np.float64).var(axis=0)
    bright = mu > 5.0
    np.testing.assert_allclose(variance[bright], mu[bright], rtol=0.1)


def test_counts_rejects_rows_outside_the_cube(solver):
    sim = _simulator(solver)
    u = _cube_points(5)
    u[2] = [0.5, 1.0 + 1e-9]

    with pytest.raises(ValueError, match="closed unit cube"):
        sim.counts(u, phase="training", key=(0,))
    u[2] = [np.nan, 0.5]
    with pytest.raises(ValueError, match="closed unit cube"):
        sim.counts(u, phase="training", key=(0,))
    assert solver.fold_counts == {}


def test_counts_rejects_bad_keys_before_folding(solver):
    sim = _simulator(solver)

    with pytest.raises(TypeError):
        sim.counts(_cube_points(3), phase="training", key=("a",))
    with pytest.raises(ValueError):
        sim.counts(_cube_points(3), phase="training", key=(2**32,))
    assert solver.fold_counts == {}


def test_counts_empty_input(solver):
    cstat, counts = _simulator(solver).counts(np.empty((0, 2)), phase="training", key=(0,))

    assert cstat.shape == (0,) and cstat.dtype == np.float64
    assert counts.shape == (0, 64) and counts.dtype == np.float32
    assert solver.fold_counts == {}


def test_counts_non_finite_model_counts_raise(solver, monkeypatch):
    original = solver._analytic_outputs

    def broken(parameters, return_kind):
        outputs = original(parameters, return_kind)
        if "total_model_counts" in outputs:
            outputs["total_model_counts"] = outputs["total_model_counts"].copy()
            outputs["total_model_counts"][0, 3] = np.nan
        return outputs

    monkeypatch.setattr(solver, "_analytic_outputs", broken)

    with pytest.raises(FoldError, match="non-finite model counts"):
        _simulator(solver).counts(_cube_points(3), phase="training", key=(0,))


# ---------------------------------------------------------------------------
# tracer
# ---------------------------------------------------------------------------


def test_tracer_records_physical_parameters(solver, spectrum):
    tracer = _RecordingTracer()
    sim = _simulator(solver, tracer=tracer, fold_chunk=3)
    u = _cube_points(5)
    u[0] = [2.0, 0.5]

    values = sim.cstat(u, phase="screen")
    cstat, _ = sim.counts(u[1:], phase="training", key=(0,))

    params = np.concatenate([call[0] for call in tracer.calls])
    recorded = np.concatenate([call[1] for call in tracer.calls])
    theta = to_physical(solver.base_prior, u[1:])
    np.testing.assert_array_equal(params, np.concatenate([theta, theta]))
    np.testing.assert_array_equal(recorded, np.concatenate([values[1:], cstat]))
    assert [len(call[1]) for call in tracer.calls] == [3, 1, 3, 1]
    # Physical, not cube: the Gamma prior is uniform(1, 3).
    assert np.all((params[:, 0] >= 1.0) & (params[:, 0] <= 3.0))


def test_pool_scope_is_shared_by_the_chunks(solver):
    sim = _simulator(solver, fold_chunk=2)

    sim.cstat(_cube_points(5), phase="screen")

    assert [entry["pool_depth"] for entry in solver.fold_log] == [1, 1, 1]
    assert solver._pool_depth == 0
