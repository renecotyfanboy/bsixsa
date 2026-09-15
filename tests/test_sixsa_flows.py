"""Tests for the SIXSA flow proposals and the ensemble training (`flows.py`, `_nde_worker.py`).

The flows are tiny (d = 2, zuko MAF 16 x 2, 5 epochs on 1500 pairs), so every
in-process test runs in about a second; only the loky pool, which spawns fresh
interpreters, is marked slow. The mixture algebra is checked on the analytic
Gaussian proposals of `tests/_analytic.py`, where every density is exact.
"""

from __future__ import annotations

import ast
import os
import pickle
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault("xspec", MagicMock())

torch = pytest.importorskip("torch")
pytest.importorskip("sbi")

import joblib  # noqa: E402
from _analytic import GaussianProposal  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

from bsixsa._cube import child_seed  # noqa: E402
from bsixsa.backend.sixsa import _nde_worker, flows  # noqa: E402
from bsixsa.backend.sixsa.flows import (  # noqa: E402
    FlowProposal,
    MixtureProposal,
    member_seeds,
    shutdown_workers,
    train_members,
)

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
WORKER_PATH = SRC_DIR / "bsixsa" / "backend" / "sixsa" / "_nde_worker.py"

D = 2
N_TRAIN = 1500
# num_bins and the None entry check the kwarg filtering: an affine flow has no
# splines, and None means "sbi default".
NDE = {
    "model": "zuko_maf",
    "hidden_features": 16,
    "num_transforms": 2,
    "num_bins": 5,
    "embedding_net": None,
}
TRAINING = {"max_num_epochs": 5, "training_batch_size": 256}
MEMBER_SEEDS = [11, 12]


def _dataset(n: int = N_TRAIN, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(size=(n, D))
    z = np.concatenate(
        [theta + 0.05 * rng.normal(size=theta.shape), rng.normal(size=(n, 2))], axis=1
    )
    return theta, z


def _z_obs() -> np.ndarray:
    return np.array([0.4, 0.6, 0.0, 0.0])


def _grid() -> np.ndarray:
    axis = np.linspace(0.05, 0.95, 7)
    return np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, D)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    theta, z = _dataset()
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp_path_factory.mktemp("member"))
        return _nde_worker.train_member(theta, z, NDE, TRAINING, seed=3)


@pytest.fixture(scope="module")
def flow(trained):
    return FlowProposal(trained[0], _z_obs(), name="member-0", generation=1)


@pytest.fixture(scope="module")
def in_process_members(tmp_path_factory):
    theta, z = _dataset()
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp_path_factory.mktemp("members"))
        return train_members(theta, z, nde=NDE, training=TRAINING, seeds=MEMBER_SEEDS, n_jobs=1)


def _grid_log_probs(estimators) -> np.ndarray:
    return np.stack([FlowProposal(e, _z_obs(), name="g").log_prob_raw(_grid()) for e in estimators])


# ---------------------------------------------------------------------------
# _nde_worker
# ---------------------------------------------------------------------------


def test_nde_worker_is_self_contained():
    tree = ast.parse(WORKER_PATH.read_text())
    allowed = {"__future__", "time", "warnings", "typing", "numpy", "torch", "sbi"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0, "no relative imports in the worker"
            assert node.module.split(".")[0] in allowed, node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in allowed, alias.name


def test_null_tracker_satisfies_the_sbi_protocol():
    tracker = _nde_worker.NullTracker()
    assert tracker.log_dir is None
    tracker.log_metric("x", 1.0, step=2)
    tracker.log_metrics({"x": 1.0}, step=None)
    tracker.log_params({"a": 1})
    tracker.add_figure("f", object(), step=1)
    tracker.flush()


def test_train_member_returns_a_zuko_flow_and_writes_no_logs(tmp_path, monkeypatch):
    from sbi.neural_nets.estimators.zuko_flow import ZukoFlow

    monkeypatch.chdir(tmp_path)
    theta, z = _dataset()
    threads_before = torch.get_num_threads()
    rng_before = torch.get_rng_state()

    estimator, info = _nde_worker.train_member(theta, z, NDE, TRAINING, seed=5, n_threads=2)

    assert isinstance(estimator, ZukoFlow)
    assert not estimator.training
    assert tuple(estimator.input_shape) == (D,)
    assert next(estimator.parameters()).device.type == "cpu"
    assert not (tmp_path / "sbi-logs").exists()
    assert list(tmp_path.iterdir()) == []
    assert torch.get_num_threads() == threads_before
    assert torch.equal(torch.get_rng_state(), rng_before)

    assert set(info) >= {"best_validation_loss", "epochs_trained", "seconds", "validation_curve"}
    assert np.isfinite(info["best_validation_loss"])
    assert info["epochs_trained"] >= 1
    assert len(info["validation_curve"]) == info["epochs_trained"]
    assert info["best_validation_loss"] == pytest.approx(min(info["validation_curve"]))
    assert info["seconds"] > 0
    assert all(isinstance(m, str) for m in info["warnings"])
    pickle.loads(pickle.dumps(estimator))


def test_train_member_is_deterministic_in_its_seed(trained, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    theta, z = _dataset()
    again, _ = _nde_worker.train_member(theta, z, NDE, TRAINING, seed=3)
    other, _ = _nde_worker.train_member(theta, z, NDE, TRAINING, seed=4)
    reference, repeat, different = _grid_log_probs([trained[0], again, other])
    np.testing.assert_array_equal(reference, repeat)
    assert np.max(np.abs(reference - different)) > 1e-3


@pytest.mark.parametrize(
    ("theta", "z", "threads"),
    [
        (np.zeros(5), np.zeros((5, 2)), 1),
        (np.zeros((5, 2)), np.zeros((4, 2)), 1),
        (np.zeros((5, 2)), np.zeros((5, 2)), 0),
    ],
)
def test_train_member_rejects_bad_inputs(theta, z, threads):
    with pytest.raises(ValueError):
        _nde_worker.train_member(theta, z, NDE, TRAINING, seed=0, n_threads=threads)


def test_worker_runs_by_value_without_bsixsa(tmp_path):
    flows._register_worker()
    from joblib.externals.cloudpickle import dumps

    theta, z = _dataset(n=200)
    payload = tmp_path / "call.pkl"
    payload.write_bytes(
        dumps(
            (
                _nde_worker.train_member,
                theta,
                z,
                NDE,
                {"max_num_epochs": 1, "training_batch_size": 64},
            )
        )
    )
    script = (
        "import pickle, sys\n"
        "sys.modules['bsixsa'] = None\n"  # any import of bsixsa now raises
        f"fn, theta, z, nde, training = pickle.loads(open({str(payload)!r}, 'rb').read())\n"
        "est, info = fn(theta, z, nde, training, 0)\n"
        "assert not any(m == 'xspec' or m.startswith('bsixsa.') for m in sys.modules)\n"
        "print(type(est).__name__, info['epochs_trained'])\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split()[0] == "ZukoFlow"
    assert not (tmp_path / "sbi-logs").exists()


# ---------------------------------------------------------------------------
# FlowProposal
# ---------------------------------------------------------------------------


def test_flow_proposal_attributes(flow):
    assert flow.d == D
    assert flow.name == "member-0"
    assert flow.generation == 1
    np.testing.assert_array_equal(flow.z_obs, _z_obs())


def test_flow_sample_raw_shape_dtype_and_determinism(flow):
    rng_state = torch.get_rng_state()
    a = flow.sample_raw(500, np.random.default_rng(7))
    b = flow.sample_raw(500, np.random.default_rng(7))
    c = flow.sample_raw(500, np.random.default_rng(8))
    assert a.shape == (500, D)
    assert a.dtype == np.float64
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert torch.equal(torch.get_rng_state(), rng_state)


def test_flow_sample_raw_batches_and_empty_requests(trained):
    small = FlowProposal(trained[0], _z_obs(), name="small", batch=64)
    draws = small.sample_raw(200, np.random.default_rng(1))
    assert draws.shape == (200, D)
    assert np.isfinite(draws).all()

    rng = np.random.default_rng(2)
    empty = small.sample_raw(0, rng)
    assert empty.shape == (0, D)
    # the seed is consumed even for an empty request
    assert rng.bit_generator.state != np.random.default_rng(2).bit_generator.state
    with pytest.raises(ValueError):
        small.sample_raw(-1, rng)


def test_flow_dimension_is_inferred_without_input_shape(trained):
    class Bare:
        def __init__(self, estimator):
            self._estimator = estimator

        def sample(self, shape, condition):
            return self._estimator.sample(shape, condition=condition)

        def log_prob(self, inputs, condition):
            return self._estimator.log_prob(inputs, condition=condition)

    bare = FlowProposal(Bare(trained[0]), _z_obs(), name="bare")
    assert bare.d == D
    assert bare.log_prob_raw(_grid()).shape == (_grid().shape[0],)


def test_flow_log_prob_raw_matches_the_estimator(flow, trained):
    u = flow.sample_raw(300, np.random.default_rng(3))
    values = flow.log_prob_raw(u)
    assert values.shape == (300,)
    assert values.dtype == np.float64
    assert np.isfinite(values).all()
    with torch.no_grad():
        direct = trained[0].log_prob(
            torch.as_tensor(u, dtype=torch.float32).reshape(-1, 1, D),
            condition=torch.as_tensor(_z_obs(), dtype=torch.float32).reshape(1, -1),
        )
    np.testing.assert_allclose(values, direct.reshape(-1).double().numpy(), rtol=0, atol=0)

    batched = FlowProposal(trained[0], _z_obs(), name="b", batch=17).log_prob_raw(u)
    np.testing.assert_allclose(batched, values, atol=1e-5)


def test_flow_log_prob_raw_handles_non_finite_rows_and_shapes(flow):
    u = np.array([[0.5, 0.5], [np.nan, 0.5], [0.2, np.inf], [1e300, 0.1], [0.3, 0.7]])
    values = flow.log_prob_raw(u)
    assert np.isfinite(values[[0, 4]]).all()
    assert np.all(values[1:4] == -np.inf)

    single = flow.log_prob_raw(np.array([0.5, 0.5]))
    assert single.shape == (1,)
    assert single[0] == pytest.approx(values[0])
    assert flow.log_prob_raw(np.empty((0, D))).shape == (0,)
    with pytest.raises(ValueError):
        flow.log_prob_raw(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        flow.log_prob_raw(np.zeros(4))


def test_flow_draws_are_not_rejected_to_the_cube(flow):
    u = flow.sample_raw(20_000, np.random.default_rng(4))
    outside = ~((u >= 0.0) & (u <= 1.0)).all(axis=1)
    assert outside.any()
    # the density outside the cube is the flow's own, not zero
    assert np.isfinite(flow.log_prob_raw(u[outside])).all()


def test_flow_proposal_rejects_bad_arguments(trained):
    with pytest.raises(ValueError):
        FlowProposal(trained[0], np.array([]), name="x")
    with pytest.raises(ValueError):
        FlowProposal(trained[0], _z_obs(), name="x", batch=0)


# ---------------------------------------------------------------------------
# MixtureProposal
# ---------------------------------------------------------------------------


def _gaussians() -> list[GaussianProposal]:
    return [
        GaussianProposal([0.2, 0.3], np.diag([0.01, 0.02]), name="g0"),
        GaussianProposal([5.0, 5.0], np.diag([0.05, 0.03]), name="g1"),
        GaussianProposal([-5.0, 0.5], [[0.04, 0.01], [0.01, 0.02]], name="g2"),
    ]


@pytest.mark.parametrize(
    ("n", "expected"), [(10, [4, 3, 3]), (9, [3, 3, 3]), (2, [1, 1, 0]), (0, [0, 0, 0])]
)
def test_mixture_allocation(n, expected):
    mixture = MixtureProposal(_gaussians())
    assert mixture.allocation(n) == expected
    assert sum(mixture.allocation(n)) == n


def test_mixture_sample_raw_is_stratified_in_member_order():
    members = _gaussians()
    mixture = MixtureProposal(members, name="mix")
    assert mixture.d == 2
    assert mixture.name == "mix"
    assert len(mixture) == 3

    u = mixture.sample_raw(1001, np.random.default_rng(0))
    assert u.shape == (1001, 2)
    assert u.dtype == np.float64
    assert mixture.last_counts == [334, 334, 333]
    # the three means are far apart, so each stratum is recognisable
    blocks = np.split(u, np.cumsum(mixture.last_counts)[:-1])
    for block, member in zip(blocks, members):
        np.testing.assert_allclose(block.mean(axis=0), member.mean, atol=0.05)

    again = mixture.sample_raw(1001, np.random.default_rng(0))
    np.testing.assert_array_equal(u, again)

    rng = np.random.default_rng(0)
    manual = np.concatenate([m.sample_raw(c, rng) for m, c in zip(members, [334, 334, 333])])
    np.testing.assert_array_equal(u, manual)


def test_mixture_sample_stratified_with_zero_counts():
    mixture = MixtureProposal(_gaussians())
    u = mixture.sample_stratified([0, 5, 0], np.random.default_rng(1))
    assert u.shape == (5, 2)
    assert mixture.last_counts == [0, 5, 0]
    np.testing.assert_allclose(u.mean(axis=0), [5.0, 5.0], atol=0.5)
    assert mixture.sample_stratified([0, 0, 0], np.random.default_rng(1)).shape == (0, 2)
    with pytest.raises(ValueError):
        mixture.sample_stratified([1, 2], np.random.default_rng(1))
    with pytest.raises(ValueError):
        mixture.sample_stratified([1, -1, 2], np.random.default_rng(1))


def test_mixture_log_prob_raw_is_the_log_mean_density():
    members = _gaussians()
    mixture = MixtureProposal(members)
    u = np.random.default_rng(2).normal(size=(50, 2)) * 3.0
    per_member = np.stack([m.log_prob_raw(u) for m in members])
    np.testing.assert_allclose(mixture.log_prob_members(u), per_member)
    np.testing.assert_allclose(mixture.log_prob_raw(u), logsumexp(per_member, axis=0) - np.log(3))
    # not the mean of the log densities
    assert np.all(mixture.log_prob_raw(u) > per_member.mean(axis=0))


def test_mixture_log_prob_stratified_with_unequal_counts():
    members = _gaussians()
    mixture = MixtureProposal(members)
    u = np.random.default_rng(3).normal(size=(40, 2)) * 3.0
    counts = [700, 250, 50]
    per_member = np.stack([m.log_prob_raw(u) for m in members])
    expected = logsumexp(per_member + np.log(np.array(counts) / 1000.0)[:, None], axis=0)
    np.testing.assert_allclose(mixture.log_prob_stratified(u, counts), expected)

    equal = mixture.log_prob_stratified(u, [10, 10, 10])
    np.testing.assert_allclose(equal, mixture.log_prob_raw(u))

    only_first = mixture.log_prob_stratified(u, [3, 0, 0])
    np.testing.assert_allclose(only_first, per_member[0])

    with pytest.raises(ValueError):
        mixture.log_prob_stratified(u, [0, 0, 0])
    with pytest.raises(ValueError):
        mixture.log_prob_stratified(u, [1, 1])


def test_mixture_stratified_weights_are_unbiased():
    # Integral of a normalised N(m, s^2 I) target: the stratified importance
    # sum with the stratified density must estimate 1 for any allocation.
    members = [
        GaussianProposal([0.0, 0.0], np.eye(2)),
        GaussianProposal([1.0, -0.5], 0.5 * np.eye(2)),
    ]
    target = GaussianProposal([0.4, -0.2], 0.3 * np.eye(2))
    mixture = MixtureProposal(members)
    counts = [3000, 17000]
    u = mixture.sample_stratified(counts, np.random.default_rng(5))
    log_w = target.log_prob_raw(u) - mixture.log_prob_stratified(u, counts)
    estimate = np.exp(logsumexp(log_w) - np.log(len(u)))
    assert estimate == pytest.approx(1.0, abs=0.03)


def test_mixture_of_flows(flow, trained):
    other = FlowProposal(trained[0], np.array([0.7, 0.2, 0.0, 0.0]), name="member-1")
    mixture = MixtureProposal([flow, other])
    assert mixture.d == D
    u = mixture.sample_raw(101, np.random.default_rng(6))
    assert u.shape == (101, D)
    assert mixture.last_counts == [51, 50]
    np.testing.assert_array_equal(u[:51], flow.sample_raw(51, np.random.default_rng(6)))
    expected = logsumexp(np.stack([flow.log_prob_raw(u), other.log_prob_raw(u)]), axis=0) - np.log(
        2
    )
    np.testing.assert_allclose(mixture.log_prob_raw(u), expected)


def test_mixture_rejects_empty_or_mismatched_members():
    with pytest.raises(ValueError):
        MixtureProposal([])
    with pytest.raises(ValueError):
        MixtureProposal([GaussianProposal([0.0, 0.0], np.eye(2)), GaussianProposal([0.0], [[1.0]])])


# ---------------------------------------------------------------------------
# Seeds and ensemble training
# ---------------------------------------------------------------------------


def test_member_seeds_are_distinct_and_deterministic():
    seq = np.random.SeedSequence(1234)
    seeds = member_seeds(seq, 3, 5)
    assert len(seeds) == 5
    assert len(set(seeds)) == 5
    assert seeds == member_seeds(np.random.SeedSequence(1234), 3, 5)
    assert seeds == [child_seed(seq, 10, 3, k) for k in range(5)]
    assert member_seeds(seq, 3, 3) == seeds[:3]
    assert not set(seeds) & set(member_seeds(seq, 4, 5))
    assert member_seeds(seq, 0, 0) == []
    with pytest.raises(ValueError):
        member_seeds(seq, 0, -1)


def test_train_members_in_process_is_deterministic(in_process_members, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    estimators, infos = in_process_members
    assert len(estimators) == len(infos) == 2
    assert [info["seed"] for info in infos] == MEMBER_SEEDS
    assert {info["executor"] for info in infos} == {"in_process"}

    threads_before = torch.get_num_threads()
    theta, z = _dataset()
    again, _ = train_members(theta, z, nde=NDE, training=TRAINING, seeds=MEMBER_SEEDS, n_jobs=1)
    assert torch.get_num_threads() == threads_before
    assert not (tmp_path / "sbi-logs").exists()

    first, repeat = _grid_log_probs(estimators), _grid_log_probs(again)
    np.testing.assert_array_equal(first, repeat)
    # different seeds, different members
    assert np.max(np.abs(first[0] - first[1])) > 1e-3

    single, _ = _nde_worker.train_member(theta, z, NDE, TRAINING, seed=MEMBER_SEEDS[1])
    np.testing.assert_array_equal(_grid_log_probs([single])[0], first[1])


def test_train_members_rejects_bad_arguments():
    theta, z = _dataset(n=10)
    with pytest.raises(ValueError):
        train_members(theta, z, nde=NDE, training=TRAINING, seeds=[])
    with pytest.raises(ValueError):
        train_members(theta, z, nde=NDE, training=TRAINING, seeds=[1], n_jobs=0)
    with pytest.raises(ValueError):
        train_members(theta, z, nde=NDE, training=TRAINING, seeds=[1], threads_per_member=0)
    with pytest.raises(ValueError):
        train_members(theta, z[:-1], nde=NDE, training=TRAINING, seeds=[1, 2], n_jobs=2)
    with pytest.raises(ValueError):
        train_members(theta[:, 0], z, nde=NDE, training=TRAINING, seeds=[1, 2], n_jobs=2)


def test_effective_jobs():
    assert flows._effective_jobs(None, 3) == min(3, os.cpu_count() or 1)
    assert flows._effective_jobs(8, 3) == 3
    assert flows._effective_jobs(2, 5) == 2
    assert flows._effective_jobs(-1, 2) == min(2, joblib.cpu_count())
    with pytest.raises(ValueError):
        flows._effective_jobs(0, 2)


def test_train_members_falls_back_in_process_when_the_pool_fails(
    in_process_members, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    def broken_parallel(*args, **kwargs):
        raise RuntimeError("no pool today")

    monkeypatch.setattr(joblib, "Parallel", broken_parallel)
    messages = []
    theta, z = _dataset()
    threads_before = torch.get_num_threads()
    with pytest.warns(RuntimeWarning, match="no pool today"):
        estimators, infos = train_members(
            theta, z, nde=NDE, training=TRAINING, seeds=MEMBER_SEEDS, n_jobs=2, log=messages.append
        )
    assert torch.get_num_threads() == threads_before
    assert len(messages) == 1 and "in-process" in messages[0]
    assert {info["executor"] for info in infos} == {"in_process"}
    np.testing.assert_array_equal(
        _grid_log_probs(estimators), _grid_log_probs(in_process_members[0])
    )


def test_train_members_does_not_swallow_interrupts(monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(joblib, "Parallel", interrupted)
    theta, z = _dataset(n=10)
    with pytest.raises(KeyboardInterrupt):
        train_members(theta, z, nde=NDE, training=TRAINING, seeds=[1, 2], n_jobs=2)


@pytest.mark.slow
def test_train_members_loky_matches_in_process(in_process_members, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    theta, z = _dataset()
    threads_before = torch.get_num_threads()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(99)
            rng_before = torch.get_rng_state()
            estimators, infos = train_members(
                theta, z, nde=NDE, training=TRAINING, seeds=MEMBER_SEEDS, n_jobs=2
            )
            assert torch.equal(torch.get_rng_state(), rng_before)
    finally:
        shutdown_workers()
    assert torch.get_num_threads() == threads_before
    assert {info["executor"] for info in infos} == {"loky"}
    assert [info["seed"] for info in infos] == MEMBER_SEEDS
    assert not (tmp_path / "sbi-logs").exists()
    np.testing.assert_allclose(
        _grid_log_probs(estimators), _grid_log_probs(in_process_members[0]), rtol=0, atol=1e-5
    )


def test_shutdown_workers_is_safe_without_a_pool():
    shutdown_workers()
    shutdown_workers()
