"""Tests for `bsixsa.backend.sixsa.sampling` (planned final sampling and resampling).

The fold is `FakeSimulator` on the quadratic C-stat of `tests/_analytic.QuadCstat`
(rows outside the closed cube get `+inf`), and proposals are raw
`tests/_analytic.GaussianProposal`s wrapped to record their draw sizes. The
target is interior, so its evidence is the closed-form unbounded one. No torch
is needed.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault("xspec", MagicMock())

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _analytic import GaussianProposal, QuadCstat  # noqa: E402

from bsixsa.backend.sixsa.sampling import FinalPosterior, resample, sample_final  # noqa: E402
from bsixsa.exceptions import FoldError  # noqa: E402

MU = np.array([0.5, 0.5])
COV = np.diag([0.05**2, 0.08**2])


class FakeSimulator:
    """Cube-simulator contract (`cstat` only) on an analytic statistic."""

    def __init__(self, quad: QuadCstat):
        self.quad = quad
        self.d = quad.d
        self.calls = 0
        self.sizes: list[int] = []
        self.phases: list[str] = []

    def cstat(self, u, *, phase: str) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        self.calls += 1
        self.sizes.append(len(u))
        self.phases.append(phase)
        out = np.full(len(u), np.inf)
        inside = np.all(np.isfinite(u), axis=1) & np.all((u >= 0.0) & (u <= 1.0), axis=1)
        out[inside] = self.quad.cstat(u[inside])
        if np.any(np.isnan(out) | np.isneginf(out)):
            raise FoldError("the fold returned NaN or -inf")
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


def _delta_error(post: FinalPosterior) -> float:
    finite = np.isfinite(post.log_w)
    w = np.exp(post.log_w[finite] - post.log_w[finite].max())
    ess_raw = w.sum() ** 2 / np.sum(w * w)
    return math.sqrt(max(1.0 / ess_raw - 1.0 / post.n_raw, 0.0))


def test_sample_final_draws_exactly_the_planned_number():
    quad = target()
    proposal = good_proposal()
    sim = FakeSimulator(quad)

    post = sample_final(
        proposal, sim, n_planned=12_345, target_ess=1000, max_draws=50_000, rng=np.random.default_rng(0), certified=True
    )

    assert post.n_raw == 12_345 and post.n_planned == 12_345 and post.n_top_up == 0
    assert post.u.shape == (12_345, 2) and post.log_w.shape == (12_345,)
    assert proposal.draw_sizes == [5000, 5000, 2345]
    assert sim.sizes == [5000, 5000, 2345] and set(sim.phases) == {"sampling"}
    assert post.shortfall == 0 and post.warnings == []
    assert post.certified and post.log_z_err_reliable and post.k_hat < 0.5
    assert post.log_z_err == pytest.approx(_delta_error(post))
    assert post.ess == pytest.approx(1.0 / np.sum(np.exp(post.log_w_smoothed) ** 2))
    assert post.efficiency == pytest.approx(100.0 * post.ess / post.n_in)
    assert abs(post.log_z - quad.log_evidence_unbounded()) < 4.0 * post.log_z_err


def test_sample_final_is_deterministic_and_uses_the_phase():
    quad = target()
    runs = [
        sample_final(
            good_proposal(),
            FakeSimulator(quad),
            n_planned=3000,
            target_ess=100,
            max_draws=3000,
            rng=np.random.default_rng(9),
            certified=True,
            chunk=1000,
            phase="final",
        )
        for _ in range(2)
    ]
    sim = FakeSimulator(quad)
    sample_final(
        good_proposal(),
        sim,
        n_planned=10,
        target_ess=1,
        max_draws=10,
        rng=np.random.default_rng(0),
        certified=True,
        phase="final",
    )

    np.testing.assert_array_equal(runs[0].u, runs[1].u)
    assert runs[0].log_z == runs[1].log_z
    assert sim.phases == ["final"]


def test_sample_final_tops_up_once_in_whole_chunks():
    quad = target()
    proposal = RecordingProposal(MU, COV * 3.0**2, name="wide")
    sim = FakeSimulator(quad)
    chunk = 2000

    post = sample_final(
        proposal,
        sim,
        n_planned=4000,
        target_ess=2500,
        max_draws=100_000,
        rng=np.random.default_rng(1),
        certified=True,
        chunk=chunk,
    )

    assert post.n_top_up > 0 and post.n_top_up % chunk == 0
    assert post.n_raw == 4000 + post.n_top_up
    assert sum(proposal.draw_sizes) == post.n_raw
    assert all(size <= chunk for size in proposal.draw_sizes)
    assert sim.calls == 2 + post.n_top_up // chunk
    # The rate measured on the planned draws predicts the target closely.
    assert post.ess == pytest.approx(2500, rel=0.25)


def test_sample_final_caps_the_top_up_at_the_budget_and_reports_the_shortfall():
    quad = target()
    proposal = narrow_proposal()
    sim = FakeSimulator(quad)

    post = sample_final(
        proposal, sim, n_planned=5000, target_ess=1e6, max_draws=12_000, rng=np.random.default_rng(2), certified=True
    )

    assert post.n_top_up == 7000 and post.n_raw == 12_000
    assert proposal.draw_sizes == [5000, 5000, 2000]
    assert post.shortfall == math.ceil(1e6 - post.ess)
    assert any("short of the target" in message for message in post.warnings)


def test_sample_final_never_tops_up_twice_even_with_budget_left():
    quad = target()
    planned_only = sample_final(
        narrow_proposal(),
        FakeSimulator(quad),
        n_planned=5000,
        target_ess=1e9,
        max_draws=5000,
        rng=np.random.default_rng(12),
        certified=True,
    )
    # Heavy tails make the ESS grow sublinearly, so extrapolating the planned
    # rate to 4x its ESS undershoots and a second top-up would be tempting.
    target_ess = 4.0 * planned_only.ess
    wanted = math.ceil(target_ess / (planned_only.ess / 5000) - 5000)
    expected_top_up = math.ceil(wanted / 5000) * 5000
    proposal = narrow_proposal()
    sim = FakeSimulator(quad)

    post = sample_final(
        proposal,
        sim,
        n_planned=5000,
        target_ess=target_ess,
        max_draws=1_000_000,
        rng=np.random.default_rng(12),
        certified=True,
    )

    np.testing.assert_array_equal(post.u[:5000], planned_only.u)
    assert post.n_top_up == expected_top_up
    assert post.n_raw == 5000 + expected_top_up < 1_000_000
    assert sum(proposal.draw_sizes) == post.n_raw and sim.calls == len(proposal.draw_sizes)
    assert post.ess < target_ess and post.shortfall > 0


def test_sample_final_without_budget_left_does_not_top_up():
    quad = target()
    sim = FakeSimulator(quad)

    post = sample_final(
        good_proposal(),
        sim,
        n_planned=5000,
        target_ess=1e6,
        max_draws=5000,
        rng=np.random.default_rng(3),
        certified=True,
    )

    assert post.n_top_up == 0 and post.n_raw == 5000 and sim.calls == 1
    assert post.shortfall > 0


def test_narrow_proposal_doubles_the_error_and_marks_it_unreliable():
    quad = target()

    post = sample_final(
        narrow_proposal(),
        FakeSimulator(quad),
        n_planned=20_000,
        target_ess=10,
        max_draws=20_000,
        rng=np.random.default_rng(4),
        certified=True,
    )

    assert post.k_hat >= 0.5
    assert post.log_z_err == pytest.approx(2.0 * _delta_error(post))
    assert not post.log_z_err_reliable
    assert post.certified
    assert any("doubled" in message for message in post.warnings)
    if post.k_hat >= 0.7:
        assert any("posterior itself is unreliable" in message for message in post.warnings)


def test_uncertified_proposal_marks_the_error_unreliable_without_doubling():
    quad = target()

    post = sample_final(
        good_proposal(),
        FakeSimulator(quad),
        n_planned=5000,
        target_ess=10,
        max_draws=5000,
        rng=np.random.default_rng(5),
        certified=False,
    )

    assert post.k_hat < 0.5
    assert not post.certified and not post.log_z_err_reliable
    assert post.log_z_err == pytest.approx(_delta_error(post))
    assert any("did not pass the gate" in message for message in post.warnings)


def test_log_z_error_is_calibrated_below_khat_one_half():
    quad = target()
    truth = quad.log_evidence_unbounded()
    # Offset and wider than the target: bounded weights, but a visible error.
    proposal = GaussianProposal(MU + [0.03, -0.04], COV * 1.4**2)

    scores = []
    for repeat in range(200):
        post = sample_final(
            proposal,
            FakeSimulator(quad),
            n_planned=5000,
            target_ess=10,
            max_draws=5000,
            rng=np.random.default_rng(1000 + repeat),
            certified=True,
        )
        assert post.k_hat < 0.5 and post.log_z_err_reliable
        scores.append((post.log_z - truth) / post.log_z_err)
    scores = np.asarray(scores)

    assert abs(scores.mean()) < 0.25
    assert 0.8 < scores.std(ddof=1) < 1.25


def test_sample_final_validation():
    quad = target()
    base = {
        "n_planned": 100,
        "target_ess": 10.0,
        "max_draws": 100,
        "rng": np.random.default_rng(0),
        "certified": True,
    }

    for override in (
        {"n_planned": 200},
        {"n_planned": 0},
        {"target_ess": 0.0},
        {"chunk": 0},
        {"max_draws": -1},
    ):
        with pytest.raises(ValueError):
            sample_final(good_proposal(), FakeSimulator(quad), **{**base, **override})
    with pytest.raises(TypeError):
        sample_final(good_proposal(), FakeSimulator(quad), **{**base, "n_planned": 100.0})


# ---------------------------------------------------------------------------
# resample
# ---------------------------------------------------------------------------


def test_resample_counts_unique_rows_and_is_deterministic():
    rng = np.random.default_rng(6)
    u = rng.random((500, 3))
    log_w = rng.normal(size=500)
    log_w[:100] = -np.inf

    rows, n_unique = resample(u, log_w, 2000, np.random.default_rng(7))
    again, n_again = resample(u, log_w, 2000, np.random.default_rng(7))

    assert rows.shape == (2000, 3)
    np.testing.assert_array_equal(rows, again)
    assert n_unique == n_again == len(np.unique(rows, axis=0))
    assert n_unique <= 400
    excluded = {tuple(row) for row in u[:100]}
    assert not any(tuple(row) in excluded for row in rows)


def test_resample_follows_the_weights():
    u = np.arange(4, dtype=np.float64).reshape(4, 1)
    log_w = np.array([math.log(0.5), math.log(0.3), math.log(0.2), -np.inf])

    rows, n_unique = resample(u, log_w, 100_000, np.random.default_rng(8))

    freq = np.bincount(rows[:, 0].astype(int), minlength=4) / 100_000
    np.testing.assert_allclose(freq, [0.5, 0.3, 0.2, 0.0], atol=0.01)
    assert n_unique == 3

    single, n_single = resample(u, np.array([0.0, -np.inf, -np.inf, -np.inf]), 50, np.random.default_rng(8))
    assert n_single == 1 and np.all(single == 0.0)

    empty, n_empty = resample(u, log_w, 0, np.random.default_rng(8))
    assert empty.shape == (0, 1) and n_empty == 0


def test_resample_validation():
    u = np.zeros((3, 2))

    with pytest.raises(ValueError):
        resample(u, np.zeros(4), 5, np.random.default_rng(0))
    with pytest.raises(ValueError):
        resample(u, np.full(3, -np.inf), 5, np.random.default_rng(0))
    with pytest.raises(ValueError):
        resample(u, np.array([0.0, np.nan, 0.0]), 5, np.random.default_rng(0))
    with pytest.raises(ValueError):
        resample(u, np.zeros(3), -1, np.random.default_rng(0))
    with pytest.raises(TypeError):
        resample(u, np.zeros(3), 2.5, np.random.default_rng(0))


def test_adaptive_top_up_does_not_change_the_planned_evidence():
    kwargs = dict(n_planned=4000, target_ess=2500, certified=True, chunk=2000)
    fixed = sample_final(RecordingProposal(MU, COV * 9, name="wide"), FakeSimulator(target()),
                         max_draws=4000, rng=np.random.default_rng(1), **kwargs)
    extended = sample_final(RecordingProposal(MU, COV * 9, name="wide"), FakeSimulator(target()),
                            max_draws=100000, rng=np.random.default_rng(1), **kwargs)
    assert extended.n_top_up > 0
    assert extended.evidence_n_raw == fixed.evidence_n_raw == 4000
    assert extended.log_z == fixed.log_z
    assert extended.log_z_err == fixed.log_z_err
    assert extended.evidence_k_hat == fixed.evidence_k_hat
