"""Tests for the SIXSA flow proposals and their training (`flows.py`).

The flows are tiny (d = 2, zuko MAF 16 x 2, 5 epochs on 1500 pairs), so every
test runs in about a second. The mixture algebra is checked on the analytic
Gaussian proposals of `tests/_analytic.py`, where every density is exact.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault("xspec", MagicMock())

torch = pytest.importorskip("torch")
pytest.importorskip("sbi")

from _analytic import GaussianProposal  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

from bsixsa.backend.sixsa.flows import FlowProposal, MixtureProposal, train_flow  # noqa: E402

D = 2
N_TRAIN = 1500
# The full NDEConfig dict, num_bins included: sbi accepts it on an affine flow and ignores it.
NDE = {"model": "zuko_maf", "hidden_features": 16, "num_transforms": 2, "num_bins": 5,
       "z_score_theta": "independent", "z_score_x": "independent"}
TRAINING = {"max_num_epochs": 5, "training_batch_size": 256}


def _dataset(n: int = N_TRAIN, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(size=(n, D))
    z = np.concatenate([theta + 0.05 * rng.normal(size=theta.shape), rng.normal(size=(n, 2))], axis=1)
    return theta, z


def _z_obs() -> np.ndarray:
    return np.array([0.4, 0.6, 0.0, 0.0])


def _grid() -> np.ndarray:
    axis = np.linspace(0.05, 0.95, 7)
    return np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, D)


def _grid_log_probs(estimator) -> np.ndarray:
    return FlowProposal(estimator, _z_obs(), name="g").log_prob_raw(_grid())


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    theta, z = _dataset()
    log_dir = tmp_path_factory.mktemp("member") / "round_1" / "member_0"
    estimator, info = train_flow(theta, z, nde=NDE, training=TRAINING, seed=3, log_dir=log_dir)
    return estimator, info, log_dir


@pytest.fixture(scope="module")
def flow(trained):
    return FlowProposal(trained[0], _z_obs(), name="member-0")


# ---------------------------------------------------------------------------
# train_flow
# ---------------------------------------------------------------------------


def test_train_flow_returns_an_estimator_and_logs_to_tensorboard(trained):
    estimator, info, log_dir = trained
    assert tuple(estimator.input_shape) == (D,)
    assert not estimator.training
    assert info["seed"] == 3 and info["log_dir"] == str(log_dir)
    assert info["epochs_trained"] >= 1 and np.isfinite(info["best_validation_loss"])
    assert any(path.name.startswith("events.out.tfevents") for path in log_dir.iterdir())


def test_train_flow_is_deterministic_in_its_seed(trained, tmp_path):
    theta, z = _dataset()
    again, _ = train_flow(theta, z, nde=NDE, training=TRAINING, seed=3, log_dir=tmp_path / "again")
    other, _ = train_flow(theta, z, nde=NDE, training=TRAINING, seed=4, log_dir=tmp_path / "other")
    np.testing.assert_array_equal(_grid_log_probs(trained[0]), _grid_log_probs(again))
    assert np.max(np.abs(_grid_log_probs(trained[0]) - _grid_log_probs(other))) > 1e-3


# ---------------------------------------------------------------------------
# FlowProposal
# ---------------------------------------------------------------------------


def test_flow_proposal_attributes(flow):
    assert flow.d == D and flow.name == "member-0"
    np.testing.assert_array_equal(flow.z_obs, _z_obs())


def test_flow_sample_raw_shape_dtype_and_determinism(flow):
    u = flow.sample_raw(500, np.random.default_rng(0))
    assert u.shape == (500, D) and u.dtype == np.float64
    np.testing.assert_array_equal(u, flow.sample_raw(500, np.random.default_rng(0)))
    assert not np.array_equal(u, flow.sample_raw(500, np.random.default_rng(1)))
    assert flow.sample_raw(0, np.random.default_rng(0)).shape == (0, D)


def test_flow_sample_raw_batches(trained):
    batched = FlowProposal(trained[0], _z_obs(), name="b", batch=100)
    assert batched.sample_raw(250, np.random.default_rng(2)).shape == (250, D)


def test_flow_log_prob_raw_matches_the_estimator(flow, trained):
    u = _grid()
    with torch.no_grad():
        expected = trained[0].log_prob(
            torch.as_tensor(u, dtype=torch.float32).reshape(-1, 1, D),
            condition=torch.as_tensor(_z_obs(), dtype=torch.float32).reshape(1, -1),
        ).reshape(-1).numpy()
    np.testing.assert_allclose(flow.log_prob_raw(u), expected, rtol=1e-5, atol=1e-5)


def test_flow_log_prob_raw_handles_non_finite_rows_and_shapes(flow):
    u = _grid()[:3].copy()
    u[1, 0] = np.nan
    log_q = flow.log_prob_raw(u)
    assert log_q.shape == (3,) and np.isneginf(log_q[1]) and np.isfinite(log_q[[0, 2]]).all()
    assert flow.log_prob_raw(u[0]).shape == (1,)


# ---------------------------------------------------------------------------
# MixtureProposal
# ---------------------------------------------------------------------------


def _gaussians():
    return [
        GaussianProposal([0.2, 0.3], np.diag([0.01, 0.02]), name="g0"),
        GaussianProposal([5.0, 5.0], np.diag([0.05, 0.03]), name="g1"),
        GaussianProposal([-5.0, 0.5], [[0.04, 0.01], [0.01, 0.02]], name="g2"),
    ]


def test_mixture_sample_raw_mixes_the_members_at_random():
    members = _gaussians()
    mixture = MixtureProposal(members, name="mix")
    assert mixture.d == 2 and mixture.name == "mix"

    u = mixture.sample_raw(3000, np.random.default_rng(0))
    assert u.shape == (3000, 2) and u.dtype == np.float64
    # the three means are far apart, so each draw's member is recognisable
    origin = np.argmin(np.stack([np.linalg.norm(u - m.mean, axis=1) for m in members]), axis=0)
    assert np.all(np.abs(np.bincount(origin, minlength=3) - 1000) < 120)
    assert len(set(origin[:30])) == 3  # interleaved, not grouped
    np.testing.assert_array_equal(u, mixture.sample_raw(3000, np.random.default_rng(0)))
    assert mixture.sample_raw(0, np.random.default_rng(1)).shape == (0, 2)


def test_mixture_log_prob_raw_is_the_log_mean_density():
    members = _gaussians()
    mixture = MixtureProposal(members)
    u = np.random.default_rng(2).normal(size=(50, 2)) * 3.0
    per_member = np.stack([m.log_prob_raw(u) for m in members])
    np.testing.assert_allclose(mixture.log_prob_raw(u), logsumexp(per_member, axis=0) - np.log(3))
    assert np.all(mixture.log_prob_raw(u) > per_member.mean(axis=0))


def test_mixture_importance_weights_are_unbiased():
    members = [GaussianProposal([0.0, 0.0], np.eye(2)), GaussianProposal([1.0, -0.5], 0.5 * np.eye(2))]
    target = GaussianProposal([0.4, -0.2], 0.3 * np.eye(2))
    mixture = MixtureProposal(members)
    u = mixture.sample_raw(20_000, np.random.default_rng(5))
    log_w = target.log_prob_raw(u) - mixture.log_prob_raw(u)
    assert np.exp(logsumexp(log_w) - np.log(len(u))) == pytest.approx(1.0, abs=0.03)


def test_mixture_of_flows(flow, trained):
    other = FlowProposal(trained[0], np.array([0.7, 0.2, 0.0, 0.0]), name="member-1")
    mixture = MixtureProposal([flow, other])
    u = mixture.sample_raw(101, np.random.default_rng(6))
    assert u.shape == (101, D)
    expected = logsumexp(np.stack([flow.log_prob_raw(u), other.log_prob_raw(u)]), axis=0) - np.log(2)
    np.testing.assert_allclose(mixture.log_prob_raw(u), expected)
