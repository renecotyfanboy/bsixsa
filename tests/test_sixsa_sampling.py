"""Tests for `bsixsa.backend.sixsa.sampling` (final importance sample and resampling).

The fold is `FakeSimulator` on the quadratic C-stat of `tests/_analytic.QuadCstat`
(rows outside the closed cube get `+inf`), and proposals are raw
`tests/_analytic.GaussianProposal`s wrapped to record their draw sizes. The
target is interior, so its evidence is the closed-form unbounded one.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault("xspec", MagicMock())
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _analytic import GaussianProposal, QuadCstat  # noqa: E402

from bsixsa.backend.sixsa.sampling import importance_sample, resample  # noqa: E402

MU = np.array([0.5, 0.5])
COV = np.diag([0.05**2, 0.08**2])
GATE = {"khat_threshold": 0.7, "min_efficiency": 5.0}


class FakeSimulator:
    """Exact-statistic callback on an analytic C-stat; out-of-cube rows get ``+inf``."""

    def __init__(self, quad: QuadCstat):
        self.quad = quad
        self.sizes: list[int] = []

    def __call__(self, u) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        self.sizes.append(len(u))
        out = np.full(len(u), np.inf)
        inside = np.all(np.isfinite(u), axis=1) & np.all((u >= 0.0) & (u <= 1.0), axis=1)
        out[inside] = self.quad.cstat(u[inside])
        return out


class RecordingProposal(GaussianProposal):
    """Gaussian proposal that records the size of every raw draw."""

    def __init__(self, mean, cov, name: str = "gauss"):
        super().__init__(mean, cov, name=name)
        self.draw_sizes: list[int] = []

    def sample_raw(self, n, rng=None):
        self.draw_sizes.append(int(n))
        return super().sample_raw(n, rng)


def target() -> QuadCstat:
    return QuadCstat(MU, np.linalg.inv(COV), c0=10.0)


def good_proposal() -> RecordingProposal:
    return RecordingProposal(MU + [0.005, 0.0], COV * 1.3**2, name="good")


def narrow_proposal() -> RecordingProposal:
    return RecordingProposal(MU, COV / 25.0, name="narrow")


def test_importance_sample_stops_once_the_target_ess_is_reached():
    quad, proposal, sim = target(), good_proposal(), FakeSimulator(target())

    post = importance_sample(proposal, sim, target_ess=1000, max_draws=50_000, batch=5000,
                             rng=np.random.default_rng(0), **GATE)

    assert post.name == "good" and post.round_index is None and post.proposal is proposal
    assert post.n_raw % 5000 == 0 and post.n_raw < 50_000
    assert proposal.draw_sizes == sim.sizes == [5000] * (post.n_raw // 5000)
    assert post.u.shape == (post.n_raw, 2) and post.log_w_smoothed.shape == (post.n_raw,)
    assert post.ess >= 1000 > post.ess - 5000 * post.efficiency / 100  # the previous batch fell short
    assert post.passed and post.k_hat < 0.5
    assert post.ess == pytest.approx(1.0 / np.sum(np.exp(post.log_w_smoothed) ** 2))
    assert abs(post.log_z - quad.log_evidence_unbounded()) < 4.0 * post.log_z_err


def test_importance_sample_stops_at_the_budget():
    proposal, sim = narrow_proposal(), FakeSimulator(target())

    post = importance_sample(proposal, sim, target_ess=1e6, max_draws=12_000, batch=5000,
                             rng=np.random.default_rng(1), **GATE)

    assert sim.sizes == [5000, 5000, 2000] and post.n_raw == 12_000
    assert post.ess < 1e6
    assert not post.passed  # a proposal five times too narrow has k-hat well above the gate


def test_importance_sample_is_deterministic():
    runs = [
        importance_sample(good_proposal(), FakeSimulator(target()), target_ess=100, max_draws=3000, batch=1000,
                          rng=np.random.default_rng(9), **GATE)
        for _ in range(2)
    ]
    np.testing.assert_array_equal(runs[0].u, runs[1].u)
    assert runs[0].log_z == runs[1].log_z


def test_importance_sample_evidence_is_calibrated_for_a_good_proposal():
    truth = target().log_evidence_unbounded()
    pulls = []
    for seed in range(20):
        post = importance_sample(good_proposal(), FakeSimulator(target()), target_ess=500, max_draws=20_000,
                                 batch=2000, rng=np.random.default_rng(100 + seed), **GATE)
        assert post.k_hat < 0.5
        pulls.append((post.log_z - truth) / post.log_z_err)
    pulls = np.asarray(pulls)
    assert np.all(np.abs(pulls) < 4.0)
    assert 0.5 < pulls.std() < 2.0


# ---------------------------------------------------------------------------
# resample
# ---------------------------------------------------------------------------


def test_resample_counts_unique_rows_and_is_deterministic():
    u = np.arange(10.0).reshape(5, 2)
    log_w = np.log([0.5, 0.3, 0.1, 0.1, 0.0]) if False else np.array([0.0, -0.5, -1.0, -np.inf, -2.0])
    rows, n_unique = resample(u, log_w, 2000, np.random.default_rng(7))
    again, n_again = resample(u, log_w, 2000, np.random.default_rng(7))
    np.testing.assert_array_equal(rows, again)
    assert rows.shape == (2000, 2) and n_unique == n_again == 4
    assert not np.any(np.all(rows == u[3], axis=1))


def test_resample_follows_the_weights():
    u = np.arange(8.0).reshape(4, 2)
    log_w = np.log([0.4, 0.3, 0.2, 0.1])
    rows, _ = resample(u, log_w, 100_000, np.random.default_rng(8))
    frequencies = [np.mean(np.all(rows == u[i], axis=1)) for i in range(4)]
    np.testing.assert_allclose(frequencies, [0.4, 0.3, 0.2, 0.1], atol=0.01)
    single, n_single = resample(u, np.array([0.0, -np.inf, -np.inf, -np.inf]), 50, np.random.default_rng(8))
    assert n_single == 1 and np.all(single == u[0])
    empty, n_empty = resample(u, log_w, 0, np.random.default_rng(8))
    assert empty.shape == (0, 2) and n_empty == 0


def test_resample_validation():
    u = np.zeros((3, 2))
    with pytest.raises(ValueError, match="no draw has a positive weight"):
        resample(u, np.full(3, -np.inf), 5, np.random.default_rng(0))
    with pytest.raises(ValueError, match="NaN"):
        resample(u, np.array([0.0, np.nan, 0.0]), 5, np.random.default_rng(0))
    with pytest.raises(ValueError):
        resample(u, np.zeros(3), -1, np.random.default_rng(0))
