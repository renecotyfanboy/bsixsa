"""Tests for `bsixsa.backend.sixsa.certify` against analytic Gaussian targets.

The exact fold is `FakeSimulator` on the quadratic C-stat of
`tests/_analytic.QuadCstat`: rows outside the closed cube are not folded and
get `+inf`. Proposals are raw `tests/_analytic.GaussianProposal`s (not
cube-rejected), so evidences and leakages are known in closed form or by dense
quadrature. No torch is needed.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault("xspec", MagicMock())

from arviz.stats import psislw  # noqa: E402
from scipy.special import logsumexp, ndtr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _analytic import GaussianProposal, QuadCstat, log_integral_grid  # noqa: E402

from bsixsa._cube import in_cube  # noqa: E402
from bsixsa.backend.sixsa.certify import (  # noqa: E402
    KHAT_SAFE,
    Certificate,
    build_certificate,
    certify,
    ess_from_log_weights,
    log_evidence,
    psis,
    rank,
)

GATE = {"khat_threshold": 0.7, "min_efficiency": 5.0}

# Interior target: 6 sigma or more from every face, so the unbounded
# evidence is the in-cube evidence to ~1e-9.
INTERIOR_MU = np.array([0.5, 0.5])
INTERIOR_COV = np.diag([0.05**2, 0.08**2])

# Target truncated by the face u_0 = 0 (16% of its mass beyond it).
FACE_MU = np.array([0.1, 0.5])
FACE_SD = 0.1


class FakeSimulator:
    """Exact-statistic callback on an analytic C-stat.

    Parameters:
        quad: The statistic.
        inf_every: If set, every `inf_every`-th folded row of each call
            returns `+inf` (a legitimate zero likelihood).
        nan_row: If set, the fold of that row of each call returns `NaN`.
    """

    def __init__(self, quad: QuadCstat, *, inf_every: int | None = None, nan_row: int | None = None):
        self.quad = quad
        self.inf_every = inf_every
        self.nan_row = nan_row
        self.calls = 0
        self.rows = 0
        self.folded = 0

    def __call__(self, u) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        self.calls += 1
        self.rows += len(u)
        out = np.full(len(u), np.inf)
        inside = in_cube(u)
        self.folded += int(inside.sum())
        values = self.quad.cstat(u[inside])
        if self.inf_every is not None:
            values[:: self.inf_every] = np.inf
        out[inside] = values
        if self.nan_row is not None:
            out[self.nan_row] = np.nan
        return out


class EqualMixture:
    """Minimal equal-weight mixture following the raw-proposal contract."""

    def __init__(self, members, name: str = "mixture"):
        self.members = list(members)
        self.name = name
        self.d = self.members[0].d

    def log_prob_raw(self, u) -> np.ndarray:
        log_q = np.stack([member.log_prob_raw(u) for member in self.members])
        return logsumexp(log_q, axis=0) - math.log(len(self.members))

    def sample_raw(self, n, rng) -> np.ndarray:
        counts = rng.multinomial(int(n), np.full(len(self.members), 1.0 / len(self.members)))
        u = np.concatenate([m.sample_raw(int(c), rng) for m, c in zip(self.members, counts)])
        return u[rng.permutation(len(u))]


def interior_target() -> QuadCstat:
    return QuadCstat(INTERIOR_MU, np.linalg.inv(INTERIOR_COV), c0=10.0)


def face_target() -> QuadCstat:
    return QuadCstat(FACE_MU, np.eye(2) / FACE_SD**2, c0=5.0)


def face_truth(quad: QuadCstat) -> float:
    """In-cube evidence of `face_target` by dense trapezoid quadrature."""
    return log_integral_grid(quad.cstat, [0.0, 0.0], [1.0, 1.0], 1001)


def certify_one(proposal, n, sim, *, seed, round_index=0) -> Certificate:
    return certify(proposal, n, sim, rng=np.random.default_rng(seed), round_index=round_index, **GATE)


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
    assert k_empty == math.inf and np.all(np.isneginf(empty))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_weight_helpers_reject_nan_and_positive_inf(bad):
    log_w = np.zeros(50)
    log_w[3] = bad
    for helper in (psis, ess_from_log_weights, lambda lw: log_evidence(lw, 50)):
        with pytest.raises(ValueError):
            helper(log_w)


def _raw_log_weights(proposal: GaussianProposal, quad: QuadCstat, n: int, seed: int) -> np.ndarray:
    u = proposal.sample_raw(n, np.random.default_rng(seed))
    return -0.5 * quad.cstat(u) - proposal.log_prob_raw(u)


def test_khat_small_for_exact_proposal_and_large_for_five_times_too_narrow():
    quad = interior_target()
    exact = GaussianProposal(INTERIOR_MU, INTERIOR_COV)
    narrow = GaussianProposal(INTERIOR_MU, INTERIOR_COV / 25.0)
    exact_k = [psis(_raw_log_weights(exact, quad, 20_000, seed))[1] for seed in range(3)]
    narrow_k = [psis(_raw_log_weights(narrow, quad, 20_000, seed))[1] for seed in range(5)]
    # The true shape is 1 - (1/5)^2 = 0.96 for the narrow proposal.
    assert max(exact_k) < 0.5
    assert np.median(narrow_k) > 0.7


def test_ess_basics():
    assert ess_from_log_weights(np.zeros(250)) == pytest.approx(250.0)
    assert ess_from_log_weights(np.array([0.0, -np.inf, -np.inf])) == pytest.approx(1.0)
    assert ess_from_log_weights(np.array([np.log(3.0), 0.0, -np.inf])) == pytest.approx(16.0 / 10.0)
    assert ess_from_log_weights(np.full(5, -np.inf)) == 0.0
    log_w = np.random.default_rng(2).normal(size=100)
    assert ess_from_log_weights(log_w + 123.0) == pytest.approx(ess_from_log_weights(log_w))


def test_log_evidence_basics():
    log_z, err = log_evidence(np.full(400, np.log(2.0)), 400)
    assert log_z == pytest.approx(np.log(2.0)) and err == pytest.approx(0.0, abs=1e-7)
    half = np.full(400, np.log(2.0))
    half[::2] = -np.inf
    log_z, err = log_evidence(half, 400)
    assert log_z == pytest.approx(np.log(2.0) + np.log(0.5))
    assert err == pytest.approx(math.sqrt(1.0 / 200.0 - 1.0 / 400.0))
    # Draws missing from the array count as zeros.
    assert log_evidence(np.full(200, np.log(2.0)), 400)[0] == pytest.approx(np.log(2.0) + np.log(0.5))
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

    cert = certify_one(proposal, 20_000, sim, seed=3, round_index=2)

    assert cert.round_index == 2 and cert.name == proposal.name and cert.proposal is proposal
    assert cert.n_raw == 20_000 and cert.n_in == 20_000 and cert.u.shape == (20_000, 2)
    assert cert.passed and cert.khat_reliable and cert.k_hat < 0.5
    assert sim.calls == 1
    assert abs(cert.log_z - quad.log_evidence_unbounded()) < 4.0 * cert.log_z_err
    assert cert.log_z_err == pytest.approx(math.sqrt(1.0 / ess_from_log_weights(cert.log_w) - 1.0 / cert.n_raw))
    assert cert.efficiency == pytest.approx(100.0 * cert.ess / cert.n_in)
    assert not any(isinstance(value, np.ndarray) for value in cert.summary().values())


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

    cert = certify_one(proposal, 20_000, sim, seed=1)

    finite = np.isfinite(cert.log_w)
    naive = logsumexp(cert.log_w[finite]) - math.log(cert.n_in)
    assert cert.n_in == sim.folded
    assert abs(cert.log_z - truth) < 4.0 * cert.log_z_err
    assert abs(naive - truth) > 10.0 * cert.log_z_err


def _mixture_setup():
    quad = face_target()
    heavy = GaussianProposal([0.0, 0.5], np.diag([0.2**2, 0.15**2]), name="leaky")
    light = GaussianProposal([0.2, 0.5], np.diag([0.12**2, 0.12**2]), name="tight")
    return quad, heavy, light, EqualMixture([heavy, light])


def test_mixture_certificate_is_unbiased_under_unequal_leakage():
    quad, heavy, light, mixture = _mixture_setup()
    assert heavy.leakage() > 0.4 and light.leakage() < 0.06
    truth = face_truth(quad)
    sim = FakeSimulator(quad)

    certs = [certify_one(proposal, 20_000, sim, seed=2, round_index=1) for proposal in (heavy, light, mixture)]

    assert [c.name for c in certs] == ["leaky", "tight", "mixture"]
    assert sim.calls == 3
    pooled = certs[-1]
    assert pooled.n_raw == 20_000 and pooled.proposal is mixture
    for cert in certs:
        assert abs(cert.log_z - truth) < 4.0 * cert.log_z_err
    finite = np.isfinite(pooled.log_w)
    naive = logsumexp(pooled.log_w[finite]) - math.log(pooled.n_in)
    assert abs(naive - truth) > 10.0 * pooled.log_z_err


def test_mixture_certificate_evidence_is_unbiased_over_repeats():
    quad, heavy, light, mixture = _mixture_setup()
    truth = face_truth(quad)
    sim = FakeSimulator(quad)
    ratios = np.array([
        math.exp(certify_one(mixture, 4000, sim, seed=100 + repeat).log_z - truth)
        for repeat in range(200)
    ])
    # Z itself (not log Z) is the unbiased quantity.
    assert sim.calls == 200
    assert abs(ratios.mean() - 1.0) < 3.0 * ratios.std(ddof=1) / math.sqrt(len(ratios))


def test_forced_infinite_folds_shift_log_z_by_log_point_nine():
    quad = interior_target()
    proposal = GaussianProposal(INTERIOR_MU, INTERIOR_COV)

    clean = certify_one(proposal, 20_000, FakeSimulator(quad), seed=4)
    holed = certify_one(proposal, 20_000, FakeSimulator(quad, inf_every=10), seed=4)

    np.testing.assert_array_equal(clean.u, holed.u)
    assert holed.n_raw == 20_000 and holed.n_in == 20_000
    assert int(np.isposinf(holed.cstat).sum()) == 2000
    assert holed.log_z - clean.log_z == pytest.approx(math.log(0.9), abs=0.01)
    assert holed.log_z == pytest.approx(quad.log_evidence_unbounded() + math.log(0.9), abs=0.01)
    assert ess_from_log_weights(holed.log_w) == pytest.approx(18_000.0, rel=1e-6)


def test_invalid_statistic_inside_the_cube_is_rejected():
    quad = interior_target()
    proposal = GaussianProposal(INTERIOR_MU, INTERIOR_COV)
    with pytest.raises(ValueError, match="NaN or -inf"):
        certify_one(proposal, 100, FakeSimulator(quad, nan_row=5), seed=0)
    u = proposal.sample_raw(20, np.random.default_rng(0))
    for bad in (np.nan, -np.inf):
        cstat = quad.cstat(u)
        cstat[3] = bad
        with pytest.raises(ValueError, match="'p'.*NaN or -inf for 1 parameter vector"):
            build_certificate("p", u, cstat, proposal.log_prob_raw(u), round_index=0, **GATE)
    # Outside the cube the statistic is never used, whatever its value.
    u[3] = [1.5, 0.5]
    cstat = quad.cstat(u)
    cstat[3] = np.nan
    cert = build_certificate("p", u, cstat, proposal.log_prob_raw(u), round_index=0, **GATE)
    assert np.isposinf(cert.cstat[3]) and np.isneginf(cert.log_w[3])


def test_build_certificate_gives_zero_weight_outside_the_cube_even_for_a_finite_statistic():
    u = np.array([[0.5, 0.5], [1.0, 0.0], [1.0 + 1e-12, 0.5], [-0.1, 0.5], [np.nan, 0.5]])
    cstat = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # finite everywhere, as a buggy simulator might
    cert = build_certificate("p", u, cstat, np.zeros(5), round_index=None, **GATE)
    assert cert.n_in == 2  # the closed faces are inside
    np.testing.assert_array_equal(np.isposinf(cert.cstat), [False, False, True, True, True])
    np.testing.assert_array_equal(np.isneginf(cert.log_w), [False, False, True, True, True])
    assert cert.log_z == pytest.approx(logsumexp([-0.5, -1.0]) - math.log(5))
    assert not cert.khat_reliable and not cert.passed


def test_build_certificate_validation():
    u = np.full((4, 2), 0.5)
    ok = {"round_index": 0, **GATE}
    with pytest.raises(ValueError):
        build_certificate("p", u, np.zeros(3), np.zeros(4), **ok)
    with pytest.raises(ValueError):
        build_certificate("p", u[:0], np.zeros(0), np.zeros(0), **ok)
    with pytest.raises(ValueError, match="log_q"):
        build_certificate("p", u, np.zeros(4), np.array([0.0, -np.inf, 0.0, 0.0]), **ok)
    # log_q may be anything where the statistic is +inf.
    cert = build_certificate("p", u, np.array([0.0, np.inf, 0.0, 0.0]), np.array([0.0, np.nan, 0.0, 0.0]), **ok)
    assert np.isneginf(cert.log_w[1])


@pytest.mark.parametrize(("extra_rows", "reliable"), [(0, False), (1, True)])
def test_khat_is_ignored_when_too_few_draws_land_in_the_cube(extra_rows, reliable):
    n_raw = 1000
    n_in = math.ceil(3.0 * math.sqrt(n_raw)) + 4 + extra_rows
    rng = np.random.default_rng(5)
    u = np.full((n_raw, 2), 2.0)
    u[:n_in] = rng.random((n_in, 2))
    cstat = rng.chisquare(2, size=n_raw)
    log_q = 0.1 * rng.normal(size=n_raw)

    cert = build_certificate("p", u, cstat, log_q, round_index=0, **GATE)

    assert cert.n_in == n_in
    assert math.isfinite(cert.k_hat) and cert.k_hat < 0.7 and cert.efficiency > 5.0
    assert cert.khat_reliable is reliable and cert.passed is reliable


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _cert(k_hat: float, efficiency: float, *, passed: bool = False, name: str = "c") -> Certificate:
    n = 1000
    log_w = np.zeros(n) if math.isfinite(k_hat) else np.full(n, -np.inf)
    return Certificate(name=name, round_index=0, u=np.zeros((n, 2)), cstat=np.zeros(n), log_w=log_w,
                       log_w_smoothed=log_w, n_raw=n, n_in=n, k_hat=k_hat, ess=10.0 * efficiency,
                       efficiency=efficiency, log_z=0.0, log_z_err=0.0, passed=passed)


def test_rank_prefers_a_passing_certificate():
    assert rank([_cert(0.1, 99.0, passed=False), _cert(0.69, 5.0, passed=True), _cert(0.3, 60.0)]) == 1


def test_rank_compares_efficiencies_below_khat_safe():
    assert KHAT_SAFE == 0.5
    assert rank([_cert(0.45, 20.0), _cert(-0.2, 30.0), _cert(0.1, 25.0)]) == 1
    # a dead certificate (no finite k-hat) never wins over a readable one
    assert rank([_cert(math.inf, 100.0), _cert(0.9, 1.0)]) == 1


def test_rank_above_khat_safe_takes_the_lowest_khat_within_tolerance_then_efficiency():
    assert rank([_cert(0.9, 30.0), _cert(0.62, 3.0), _cert(0.65, 8.0), _cert(0.8, 40.0)]) == 2
    assert rank([_cert(0.9, 30.0), _cert(0.6, 3.0), _cert(0.7, 8.0)]) == 1


def test_rank_keeps_the_first_when_nothing_is_readable():
    assert rank([_cert(math.inf, 100.0), _cert(math.nan, 100.0)]) == 0
    with pytest.raises(ValueError):
        rank([])
