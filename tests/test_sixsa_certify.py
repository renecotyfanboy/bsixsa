"""Tests for `bsixsa.backend.sixsa.certify` against analytic Gaussian targets.

The exact fold is replaced by `FakeSimulator`, which follows the cube-simulator
contract on the quadratic C-stat of `tests/_analytic.QuadCstat`: rows outside
the closed cube are not folded and get `+inf`, and a `NaN` fold raises
`FoldError`. Proposals are `tests/_analytic.GaussianProposal` (raw, not
cube-rejected), so evidences and leakages are known in closed form or by
dense quadrature. No torch is needed.
"""

from __future__ import annotations

import math
import pickle
import sys
from functools import reduce
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault("xspec", MagicMock())

from arviz.stats import psislw  # noqa: E402
from scipy.special import logsumexp, ndtr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _analytic import GaussianProposal, QuadCstat, log_integral_grid  # noqa: E402

from bsixsa.backend.sixsa.certify import (  # noqa: E402
    Certificate,
    build_certificate,
    certify_fresh,
    certify_many,
    ess_from_log_weights,
    is_alive,
    log_evidence,
    psis,
    rank_candidates,
    tier,
)
from bsixsa.exceptions import FoldError  # noqa: E402

GATE = {"khat_threshold": 0.7, "khat_safe": 0.5, "min_efficiency": 5.0}

# Interior target: 6 sigma or more from every face, so the unbounded
# evidence is the in-cube evidence to ~1e-9.
INTERIOR_MU = np.array([0.5, 0.5])
INTERIOR_COV = np.diag([0.05**2, 0.08**2])

# Target truncated by the face u_0 = 0 (16% of its mass beyond it).
FACE_MU = np.array([0.1, 0.5])
FACE_SD = 0.1


class FakeSimulator:
    """Cube-simulator contract (`cstat` only) on an analytic statistic.

    Parameters:
        quad: The statistic.
        inf_every: If set, every `inf_every`-th folded row of each call
            returns `+inf` (a legitimate zero likelihood).
        nan_row: If set, the fold of that row of each call returns `NaN`,
            which the contract turns into a `FoldError`.
    """

    def __init__(self, quad: QuadCstat, *, inf_every: int | None = None, nan_row: int | None = None):
        self.quad = quad
        self.d = quad.d
        self.inf_every = inf_every
        self.nan_row = nan_row
        self.calls = 0
        self.rows = 0
        self.folded = 0
        self.phases: list[str] = []

    def cstat(self, u, *, phase: str) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        self.calls += 1
        self.rows += len(u)
        self.phases.append(phase)
        out = np.full(len(u), np.inf)
        inside = np.all(np.isfinite(u), axis=1) & np.all((u >= 0.0) & (u <= 1.0), axis=1)
        self.folded += int(inside.sum())
        values = self.quad.cstat(u[inside])
        if self.inf_every is not None:
            values[:: self.inf_every] = np.inf
        out[inside] = values
        if self.nan_row is not None:
            out[self.nan_row] = np.nan
        if np.any(np.isnan(out) | np.isneginf(out)):
            raise FoldError("the fold returned NaN or -inf")
        return out


class StratifiedMixture:
    """Minimal mixture following the stratified-mixture contract."""

    def __init__(self, members, name: str = "mixture"):
        self.members = list(members)
        self.name = name
        self.d = self.members[0].d

    def log_prob_stratified(self, u, counts) -> np.ndarray:
        counts = np.asarray(counts, dtype=np.float64)
        log_q = np.stack([member.log_prob_raw(u) for member in self.members])
        return logsumexp(log_q + np.log(counts / counts.sum())[:, None], axis=0)

    def sample_stratified(self, counts, rng) -> np.ndarray:
        return np.concatenate([m.sample_raw(int(c), rng) for m, c in zip(self.members, counts)])


def interior_target() -> QuadCstat:
    return QuadCstat(INTERIOR_MU, np.linalg.inv(INTERIOR_COV), c0=10.0)


def face_target() -> QuadCstat:
    return QuadCstat(FACE_MU, np.eye(2) / FACE_SD**2, c0=5.0)


def face_truth(quad: QuadCstat) -> float:
    """In-cube evidence of `face_target` by dense trapezoid quadrature."""
    return log_integral_grid(quad.cstat, [0.0, 0.0], [1.0, 1.0], 1001)


# ---------------------------------------------------------------------------
# psis / ess / log_evidence
# ---------------------------------------------------------------------------


def test_psis_matches_arviz_on_finite_weights():
    log_w = np.random.default_rng(0).normal(size=2000) * 2.0

    smoothed, k_hat = psis(log_w)
    expected, expected_k = psislw(log_w.copy(), reff=1.0, normalize=True)

    np.testing.assert_allclose(smoothed, expected)
    assert k_hat == pytest.approx(float(expected_k))


def test_psis_preserves_zero_weights_and_normalises():
    log_w = np.random.default_rng(1).normal(size=3000) * 3.0
    log_w[::7] = -np.inf
    original = log_w.copy()

    smoothed, k_hat = psis(log_w)

    np.testing.assert_array_equal(log_w, original)
    np.testing.assert_array_equal(np.isneginf(smoothed), np.isneginf(original))
    assert logsumexp(smoothed) == pytest.approx(0.0, abs=1e-12)
    assert math.isfinite(k_hat)


def test_psis_with_fewer_than_five_finite_weights_returns_normalised_raw_and_inf():
    log_w = np.full(100, -np.inf)
    log_w[:4] = [0.0, 1.0, 2.0, 3.0]

    smoothed, k_hat = psis(log_w)

    assert k_hat == math.inf
    np.testing.assert_allclose(smoothed[:4], log_w[:4] - logsumexp(log_w[:4]))
    assert np.all(np.isneginf(smoothed[4:]))

    empty, k_empty = psis(np.full(10, -np.inf))
    assert k_empty == math.inf
    assert np.all(np.isneginf(empty))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_weight_helpers_reject_nan_and_positive_inf(bad):
    log_w = np.zeros(50)
    log_w[3] = bad

    with pytest.raises(ValueError):
        psis(log_w)
    with pytest.raises(ValueError):
        ess_from_log_weights(log_w)
    with pytest.raises(ValueError):
        log_evidence(log_w, 50)


def _raw_log_weights(proposal: GaussianProposal, quad: QuadCstat, n: int, seed: int) -> np.ndarray:
    u = proposal.sample_raw(n, np.random.default_rng(seed))
    return -0.5 * quad.cstat(u) - proposal.log_prob_raw(u)


def test_khat_small_for_exact_proposal_and_large_for_five_times_too_narrow():
    quad = interior_target()
    exact = GaussianProposal(INTERIOR_MU, INTERIOR_COV)
    narrow = GaussianProposal(INTERIOR_MU, INTERIOR_COV / 25.0)

    exact_k = [psis(_raw_log_weights(exact, quad, 20_000, seed))[1] for seed in range(3)]
    narrow_k = [psis(_raw_log_weights(narrow, quad, 20_000, seed))[1] for seed in range(5)]

    # The true shape is 1 - (1/5)^2 = 0.96 for the narrow proposal; the
    # estimate is noisy, so the median of a few fixed seeds is checked.
    assert max(exact_k) < 0.5
    assert np.median(narrow_k) > 0.7


def test_ess_basics():
    assert ess_from_log_weights(np.zeros(250)) == pytest.approx(250.0)
    assert ess_from_log_weights(np.array([0.0, -np.inf, -np.inf])) == pytest.approx(1.0)
    assert ess_from_log_weights(np.array([np.log(3.0), 0.0, -np.inf])) == pytest.approx(16.0 / 10.0)
    assert ess_from_log_weights(np.full(5, -np.inf)) == 0.0
    # Invariant to a constant shift.
    log_w = np.random.default_rng(2).normal(size=100)
    assert ess_from_log_weights(log_w + 123.0) == pytest.approx(ess_from_log_weights(log_w))


def test_log_evidence_basics():
    log_z, err = log_evidence(np.full(400, np.log(2.0)), 400)
    assert log_z == pytest.approx(np.log(2.0))
    assert err == pytest.approx(0.0, abs=1e-7)

    half = np.full(400, np.log(2.0))
    half[::2] = -np.inf
    log_z, err = log_evidence(half, 400)
    assert log_z == pytest.approx(np.log(2.0) + np.log(0.5))
    assert err == pytest.approx(math.sqrt(1.0 / 200.0 - 1.0 / 400.0))

    # Draws missing from the array count as zeros.
    log_z, _ = log_evidence(np.full(200, np.log(2.0)), 400)
    assert log_z == pytest.approx(np.log(2.0) + np.log(0.5))

    assert log_evidence(np.full(10, -np.inf), 10) == (-math.inf, math.inf)
    with pytest.raises(ValueError):
        log_evidence(np.zeros(10), 9)


# ---------------------------------------------------------------------------
# Certificates against analytic evidences
# ---------------------------------------------------------------------------


def test_certificate_log_z_matches_the_analytic_interior_evidence():
    quad = interior_target()
    proposal = GaussianProposal(INTERIOR_MU + [0.01, -0.01], INTERIOR_COV * 1.3**2)
    sim = FakeSimulator(quad)

    cert = certify_fresh(proposal, 20_000, sim, rng=np.random.default_rng(3), phase="check", round_index=2, **GATE)

    assert cert.kind == "fresh" and cert.round_index == 2 and cert.name == proposal.name
    assert cert.n_raw == 20_000 and cert.n_in == 20_000
    assert cert.u.shape == (20_000, 2)
    assert cert.passed and cert.tier == 0 and cert.k_hat < 0.5
    assert cert.proposal is proposal
    assert sim.calls == 1 and sim.phases == ["check"]
    assert abs(cert.log_z - quad.log_evidence_unbounded()) < 4.0 * cert.log_z_err
    assert cert.log_z_err == pytest.approx(math.sqrt(1.0 / cert.ess_raw - 1.0 / cert.n_raw))
    assert cert.efficiency == pytest.approx(100.0 * cert.ess / cert.n_in)


def test_certificate_is_exact_under_leakage_where_the_in_cube_mean_is_not():
    quad = face_target()
    truth = face_truth(quad)
    analytic = (
        -0.5 * quad.c0
        + math.log(2.0 * math.pi * FACE_SD**2)
        + math.log((ndtr((1.0 - 0.1) / FACE_SD) - ndtr(-0.1 / FACE_SD)) * (ndtr(0.5 / FACE_SD) - ndtr(-0.5 / FACE_SD)))
    )
    assert truth == pytest.approx(analytic, abs=1e-5)
    proposal = GaussianProposal([0.0, 0.5], np.diag([0.2**2, 0.15**2]))
    assert proposal.leakage() == pytest.approx(0.5, abs=0.01)
    sim = FakeSimulator(quad)

    cert = certify_fresh(proposal, 20_000, sim, rng=np.random.default_rng(1), phase="check", round_index=0, **GATE)

    finite = np.isfinite(cert.log_w)
    naive = logsumexp(cert.log_w[finite]) - math.log(cert.n_in)
    assert cert.n_in == sim.folded
    assert abs(cert.log_z - truth) < 4.0 * cert.log_z_err
    assert abs(naive - truth) > 10.0 * cert.log_z_err


def _mixture_setup():
    quad = face_target()
    heavy = GaussianProposal([0.0, 0.5], np.diag([0.2**2, 0.15**2]), name="leaky")
    light = GaussianProposal([0.2, 0.5], np.diag([0.12**2, 0.12**2]), name="tight")
    return quad, heavy, light, StratifiedMixture([heavy, light])


def test_mixture_certificate_is_unbiased_under_unequal_leakage_with_no_extra_fold():
    quad, heavy, light, mixture = _mixture_setup()
    assert heavy.leakage() > 0.4 and light.leakage() < 0.06
    truth = face_truth(quad)
    sim = FakeSimulator(quad)

    certs = certify_many(
        [heavy, light],
        20_000,
        sim,
        rng=np.random.default_rng(2),
        phase="screen",
        round_index=1,
        mixture=mixture,
        **GATE,
    )

    assert [c.name for c in certs] == ["leaky", "tight", "mixture"]
    assert sim.calls == 1 and sim.rows == 40_000
    member_a, member_b, pooled = certs
    assert pooled.n_raw == 40_000 and pooled.proposal is mixture
    assert pooled.extra == {"counts": [20_000, 20_000]}
    np.testing.assert_array_equal(pooled.u, np.concatenate([member_a.u, member_b.u]))
    np.testing.assert_array_equal(pooled.cstat, np.concatenate([member_a.cstat, member_b.cstat]))
    assert abs(pooled.log_z - truth) < 4.0 * pooled.log_z_err
    for member in (member_a, member_b):
        assert abs(member.log_z - truth) < 4.0 * member.log_z_err
    finite = np.isfinite(pooled.log_w)
    naive = logsumexp(pooled.log_w[finite]) - math.log(pooled.n_in)
    assert abs(naive - truth) > 10.0 * pooled.log_z_err


def test_mixture_certificate_evidence_is_unbiased_over_repeats():
    quad, heavy, light, mixture = _mixture_setup()
    truth = face_truth(quad)
    sim = FakeSimulator(quad)

    ratios = np.array(
        [
            math.exp(
                certify_many(
                    [heavy, light],
                    2000,
                    sim,
                    rng=np.random.default_rng(100 + repeat),
                    phase="screen",
                    round_index=0,
                    mixture=mixture,
                    **GATE,
                )[-1].log_z
                - truth
            )
            for repeat in range(200)
        ]
    )

    # Z itself (not log Z) is the unbiased quantity.
    assert sim.calls == 200
    assert abs(ratios.mean() - 1.0) < 3.0 * ratios.std(ddof=1) / math.sqrt(len(ratios))


def test_mixture_members_must_be_the_certified_proposals():
    quad, heavy, light, mixture = _mixture_setup()
    other = GaussianProposal([0.2, 0.5], np.diag([0.12**2, 0.12**2]), name="tight")
    sim = FakeSimulator(quad)

    for proposals in ([heavy], [light, heavy], [heavy, other]):
        with pytest.raises(ValueError, match="mixture members"):
            certify_many(
                proposals, 100, sim, rng=np.random.default_rng(0), phase="s", round_index=0, mixture=mixture, **GATE
            )
    assert sim.calls == 0


def test_forced_infinite_folds_shift_log_z_by_log_point_nine():
    quad = interior_target()
    proposal = GaussianProposal(INTERIOR_MU, INTERIOR_COV)

    clean = certify_fresh(
        proposal, 20_000, FakeSimulator(quad), rng=np.random.default_rng(4), phase="check", round_index=0, **GATE
    )
    sim = FakeSimulator(quad, inf_every=10)
    holed = certify_fresh(proposal, 20_000, sim, rng=np.random.default_rng(4), phase="check", round_index=0, **GATE)

    np.testing.assert_array_equal(clean.u, holed.u)
    assert holed.n_raw == 20_000 and holed.n_in == 20_000
    assert int(np.isposinf(holed.cstat).sum()) == 2000
    assert holed.log_z - clean.log_z == pytest.approx(math.log(0.9), abs=0.01)
    assert holed.log_z == pytest.approx(quad.log_evidence_unbounded() + math.log(0.9), abs=0.01)
    assert holed.ess_raw == pytest.approx(18_000.0, rel=1e-6)


def test_nan_fold_raises_fold_error():
    quad = interior_target()
    proposal = GaussianProposal(INTERIOR_MU, INTERIOR_COV)
    sim = FakeSimulator(quad, nan_row=5)

    with pytest.raises(FoldError):
        certify_fresh(proposal, 100, sim, rng=np.random.default_rng(0), phase="check", round_index=0, **GATE)

    u = proposal.sample_raw(20, np.random.default_rng(0))
    for bad in (np.nan, -np.inf):
        cstat = quad.cstat(u)
        cstat[3] = bad
        with pytest.raises(FoldError):
            build_certificate("p", u, cstat, proposal.log_prob_raw(u), n_raw=20, kind="screen", round_index=0, **GATE)


def test_build_certificate_gives_zero_weight_outside_the_cube_even_for_a_finite_statistic():
    u = np.array([[0.5, 0.5], [1.0, 0.0], [1.0 + 1e-12, 0.5], [-0.1, 0.5], [np.nan, 0.5]])
    cstat = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # finite everywhere, as a buggy simulator might
    log_q = np.zeros(5)

    cert = build_certificate("p", u, cstat, log_q, n_raw=5, kind="screen", round_index=None, **GATE)

    assert cert.n_in == 2  # the closed faces are inside
    np.testing.assert_array_equal(np.isposinf(cert.cstat), [False, False, True, True, True])
    np.testing.assert_array_equal(np.isneginf(cert.log_w), [False, False, True, True, True])
    assert cert.log_z == pytest.approx(logsumexp([-0.5, -1.0]) - math.log(5))
    assert not cert.alive and cert.tier == 3


def test_build_certificate_validation():
    u = np.full((4, 2), 0.5)
    ok = {"n_raw": 4, "round_index": 0, **GATE}

    with pytest.raises(ValueError, match="kind"):
        build_certificate("p", u, np.zeros(4), np.zeros(4), kind="other", **ok)
    with pytest.raises(ValueError):
        build_certificate("p", u, np.zeros(3), np.zeros(4), kind="screen", **ok)
    with pytest.raises(ValueError):
        build_certificate("p", u[:3], np.zeros(4), np.zeros(4), kind="screen", **ok)
    with pytest.raises(ValueError, match="log_q"):
        build_certificate("p", u, np.zeros(4), np.array([0.0, -np.inf, 0.0, 0.0]), kind="screen", **ok)
    # log_q may be anything where the statistic is +inf.
    cert = build_certificate(
        "p", u, np.array([0.0, np.inf, 0.0, 0.0]), np.array([0.0, np.nan, 0.0, 0.0]), kind="screen", **ok
    )
    assert np.isneginf(cert.log_w[1])


def test_alive_threshold():
    n_raw = 1000
    edge = math.ceil(3.0 * math.sqrt(n_raw)) + 4

    assert not is_alive(0.2, edge, n_raw)
    assert is_alive(0.2, edge + 1, n_raw)
    assert not is_alive(math.inf, 10 * n_raw, n_raw)
    assert not is_alive(math.nan, 10 * n_raw, n_raw)


@pytest.mark.parametrize(("extra_rows", "alive"), [(0, False), (1, True)])
def test_khat_is_ignored_when_too_few_draws_land_in_the_cube(extra_rows, alive):
    n_raw = 1000
    n_in = math.ceil(3.0 * math.sqrt(n_raw)) + 4 + extra_rows
    rng = np.random.default_rng(5)
    u = np.full((n_raw, 2), 2.0)
    u[:n_in] = rng.random((n_in, 2))
    cstat = rng.chisquare(2, size=n_raw)
    log_q = 0.1 * rng.normal(size=n_raw)

    cert = build_certificate("p", u, cstat, log_q, n_raw=n_raw, kind="screen", round_index=0, **GATE)

    assert cert.n_in == n_in
    assert math.isfinite(cert.k_hat) and cert.k_hat < 0.7
    assert cert.efficiency > 5.0
    assert cert.alive is alive
    assert cert.passed is alive
    assert cert.tier == (0 if alive else 3)


def test_certify_many_uses_independent_deterministic_streams_and_one_fold_call():
    quad = interior_target()
    first = GaussianProposal(INTERIOR_MU, INTERIOR_COV, name="a")
    second = GaussianProposal(INTERIOR_MU, INTERIOR_COV, name="b")
    sim = FakeSimulator(quad)

    one = certify_many([first, second], 500, sim, rng=np.random.default_rng(7), phase="p1", round_index=3, **GATE)
    two = certify_many([first, second], 500, sim, rng=np.random.default_rng(7), phase="p2", round_index=3, **GATE)

    assert sim.calls == 2 and sim.phases == ["p1", "p2"]
    assert [c.kind for c in one] == ["screen", "screen"]
    for a, b in zip(one, two):
        np.testing.assert_array_equal(a.u, b.u)
        assert a.log_z == b.log_z
    assert not np.array_equal(one[0].u, one[1].u)
    with pytest.raises(ValueError):
        certify_many([], 10, sim, rng=np.random.default_rng(0), phase="p", round_index=0, **GATE)
    with pytest.raises(ValueError):
        certify_many([first], 0, sim, rng=np.random.default_rng(0), phase="p", round_index=0, **GATE)


def test_certificate_serialisation_drops_the_proposal():
    quad = interior_target()
    proposal = GaussianProposal(INTERIOR_MU, INTERIOR_COV)
    cert = certify_fresh(
        proposal, 200, FakeSimulator(quad), rng=np.random.default_rng(0), phase="p", round_index=0, **GATE
    )

    restored = pickle.loads(pickle.dumps(cert))

    assert cert.proposal is proposal
    assert restored.proposal is None
    np.testing.assert_array_equal(restored.log_w, cert.log_w)
    assert cert.without_proposal().proposal is None
    summary = cert.summary()
    assert summary["log_z"] == cert.log_z and summary["tier"] == cert.tier
    assert not any(isinstance(value, np.ndarray) for value in summary.values())


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _cert(k_hat: float, efficiency: float, *, alive: bool = True, passed: bool = False, name: str = "c") -> Certificate:
    one = np.zeros(1)
    cert = Certificate(
        name=name,
        kind="fresh",
        round_index=0,
        u=np.zeros((1, 2)),
        cstat=one,
        log_q=one,
        log_w=one,
        log_w_smoothed=one,
        n_raw=1000,
        n_in=1000,
        k_hat=k_hat,
        ess=10.0 * efficiency,
        ess_raw=10.0 * efficiency,
        efficiency=efficiency,
        log_z=0.0,
        log_z_err=0.0,
        alive=alive,
        passed=passed,
        tier=-1,
    )
    cert.tier = tier(cert, khat_safe=GATE["khat_safe"])
    return cert


def _rank(certs, tolerance: float = 0.05) -> int:
    return rank_candidates(certs, khat_safe=GATE["khat_safe"], tolerance=tolerance)


def test_tiers():
    assert _cert(0.3, 50.0, passed=True).tier == 0
    assert _cert(0.3, 1.0).tier == 1
    assert _cert(0.8, 50.0).tier == 2
    assert _cert(0.1, 50.0, alive=False).tier == 3


def test_rank_prefers_the_lowest_tier():
    certs = [
        _cert(0.1, 99.0, alive=False),
        _cert(0.9, 90.0),
        _cert(0.4, 3.0),
        _cert(0.69, 5.0, passed=True),
    ]

    assert _rank(certs) == 3
    assert _rank(certs[:3]) == 2
    assert _rank(certs[:2]) == 1
    assert _rank(certs[:1]) == 0


def test_rank_within_tiers_zero_and_one_ignores_khat():
    assert _rank([_cert(0.1, 10.0), _cert(0.45, 30.0), _cert(0.3, 20.0)]) == 1
    assert _rank([_cert(0.1, 10.0, passed=True), _cert(0.6, 30.0, passed=True)]) == 1
    # Ties go to the lower index.
    assert _rank([_cert(0.4, 30.0), _cert(0.1, 30.0)]) == 0


def test_rank_tier_two_keeps_the_khat_tolerance_band_then_efficiency():
    certs = [_cert(0.9, 90.0), _cert(0.6, 10.0), _cert(0.64, 20.0), _cert(0.66, 80.0)]

    assert _rank(certs) == 2
    assert _rank(certs, tolerance=0.0) == 1
    assert _rank(certs, tolerance=1.0) == 0


def test_rank_tier_three_takes_the_lowest_index():
    certs = [_cert(math.inf, 100.0, alive=False), _cert(0.2, 90.0, alive=False)]

    assert _rank(certs) == 0


def test_rank_recomputes_tiers_with_the_given_khat_safe():
    certs = [_cert(0.55, 90.0), _cert(0.45, 10.0)]

    assert _rank(certs) == 1  # 0.45 is tier 1 at khat_safe 0.5
    assert rank_candidates(certs, khat_safe=0.6, tolerance=0.05) == 0


def test_rank_is_a_function_of_the_set_and_does_not_drift():
    rounds = [_cert(0.80, 10.0), _cert(0.84, 20.0), _cert(0.88, 30.0), _cert(0.92, 40.0)]
    assert all(cert.tier == 2 for cert in rounds)

    best_so_far = [_rank(rounds[: i + 1]) for i in range(len(rounds))]
    # Chaining two-candidate comparisons walks up the k_hat ramp.
    chained = reduce(lambda best, new: [best, new][_rank([best, new])], rounds)

    assert best_so_far == [0, 1, 1, 1]
    assert chained is rounds[3]


def test_a_passing_certificate_always_beats_a_failing_one():
    rng = np.random.default_rng(11)
    for _ in range(500):
        n = int(rng.integers(2, 8))
        certs = []
        for _ in range(n):
            alive = bool(rng.random() < 0.8)
            k_hat = float(rng.uniform(-0.5, 1.5))
            efficiency = float(rng.uniform(0.0, 100.0))
            passed = alive and k_hat < 0.7 and efficiency >= 5.0 and bool(rng.random() < 0.3)
            certs.append(_cert(k_hat, efficiency, alive=alive, passed=passed))
        best = _rank(certs)
        if any(cert.passed for cert in certs):
            assert certs[best].passed
    assert _rank([_cert(0.01, 99.0), _cert(0.69, 5.0, passed=True)]) == 1


def test_rank_rejects_an_empty_set():
    with pytest.raises(ValueError):
        _rank([])
