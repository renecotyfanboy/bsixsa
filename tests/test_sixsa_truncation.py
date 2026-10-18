"""Tests for `bsixsa.backend.sixsa.truncation` (SIR draws inside a density contour).

The proposals are `tests/_analytic.GaussianProposal`, whose raw log density is
exact, so the set $A = [0, 1]^d \\cap R \\cap \\{\\log q > t\\}$ is an ellipse
(intersected with a half-plane) and the flatness of the draws can be checked
on the whitened radius: for a flat draw on the whitened ball of radius $R$,
$(r / R)^d$ is uniform on $[0, 1]$.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy import linalg

sys.modules.setdefault("xspec", MagicMock())
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _analytic import GaussianProposal  # noqa: E402

from bsixsa._cube import in_cube  # noqa: E402
from bsixsa.backend.sixsa.truncation import density_threshold, sample_truncated  # noqa: E402

_LOG_2PI = math.log(2.0 * math.pi)


class _HalfPlane:
    """Stub region $\\{u \\in [0,1]^d : u_0 < c\\}$ that counts its calls."""

    def __init__(self, cut):
        self.cut = float(cut)
        self.calls = 0

    def accepts(self, u):
        self.calls += 1
        u = np.atleast_2d(np.asarray(u, dtype=np.float64))
        return in_cube(u) & (u[:, 0] < self.cut)


def _gaussian_2d():
    return GaussianProposal([0.5, 0.45], [[0.008, 0.003], [0.003, 0.006]])


def _contour_radius(proposal, threshold):
    """Whitened radius of the Gaussian level set $\\log q = t$."""
    return math.sqrt(-2.0 * threshold - proposal.d * _LOG_2PI - proposal._logdet)


def _radial_cdf(proposal, u, threshold):
    r"""$(r / R)^d$ of the draws, uniform on $[0, 1]$ for a flat draw inside the contour."""
    whitened = linalg.solve_triangular(proposal._chol, (u - proposal.mean).T, lower=True).T
    r = np.linalg.norm(whitened, axis=1)
    return (r / _contour_radius(proposal, threshold)) ** proposal.d


# ---------------------------------------------------------------------------
# density_threshold
# ---------------------------------------------------------------------------


def test_density_threshold_quantile_semantics():
    proposal = _gaussian_2d()
    quantile = 0.05

    t = density_threshold(proposal, quantile=quantile, n_support=20_000, rng=np.random.default_rng(0))

    # For a Gaussian, log q > t is the ellipse r^2 < chi2_d(1 - quantile).
    r2 = -2.0 * t - proposal.d * _LOG_2PI - proposal._logdet
    assert r2 == pytest.approx(-2.0 * math.log(quantile), rel=0.05)
    draws = proposal.sample_raw(50_000, np.random.default_rng(1))
    assert np.mean(proposal.log_prob_raw(draws) > t) == pytest.approx(1.0 - quantile, abs=0.01)


def test_density_threshold_keeps_only_in_cube_rows():
    # Centred on a face: half the draws leak out and must not enter the quantile.
    proposal = GaussianProposal([0.0, 0.5], np.eye(2) * 0.05**2)
    rng = np.random.default_rng(2)
    t = density_threshold(proposal, quantile=0.1, n_support=20_000, rng=rng)
    draws = proposal.sample_raw(50_000, np.random.default_rng(3))
    kept = draws[in_cube(draws)]
    assert np.mean(proposal.log_prob_raw(kept) > t) == pytest.approx(0.9, abs=0.02)


def test_density_threshold_rejects_bad_inputs():
    far = GaussianProposal([5.0, 5.0], np.eye(2) * 0.01)
    with pytest.raises(ValueError, match="fall in the unit cube"):
        density_threshold(far, quantile=1e-3, n_support=1000, rng=np.random.default_rng(0))

    class _Broken(GaussianProposal):
        def log_prob_raw(self, u):
            out = super().log_prob_raw(u)
            out[0] = np.nan
            return out

    with pytest.raises(ValueError, match="non-finite"):
        density_threshold(_Broken([0.5, 0.5], np.eye(2) * 0.01), quantile=1e-3, n_support=1000,
                          rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------
# sample_truncated
# ---------------------------------------------------------------------------


def test_sir_draws_lie_inside_the_contour_the_cube_and_the_region():
    proposal = _gaussian_2d()
    region = _HalfPlane(0.5)

    u, info = sample_truncated(
        proposal, n=3000, quantile=1e-3, n_support=10_000, oversampling=16, region=region,
        rng=np.random.default_rng(4),
    )

    assert u.shape == (3000, 2) and u.dtype == np.float64
    assert np.all(in_cube(u))
    assert np.all(proposal.log_prob_raw(u) > info["threshold"])
    assert np.all(u[:, 0] < 0.5)
    assert region.calls == 1
    assert info["n_candidates"] == 16 * 3000
    # Half the contour is cut away by the region, and 0.1% of the mass by the quantile.
    assert info["acceptance"] == pytest.approx(0.5, abs=0.05)
    assert 0 < info["sir_ess"] <= info["acceptance"] * info["n_candidates"]
    assert 1 < info["n_unique"] <= 3000


def test_sir_draws_are_approximately_flat_inside_a_gaussian_contour():
    proposal = _gaussian_2d()
    n = 4000

    u, info = sample_truncated(
        proposal, n=n, quantile=1e-3, n_support=10_000, oversampling=64, rng=np.random.default_rng(5)
    )

    cdf = _radial_cdf(proposal, u, info["threshold"])
    assert np.all(cdf < 1.0)
    # Flat draws have E[(r/R)^d] = 1/2 and put half their mass below (r/R)^d = 1/2;
    # the proposal's own draws would put ~90% of theirs there.
    assert cdf.mean() == pytest.approx(0.5, abs=0.03)
    assert np.mean(cdf < 0.5) == pytest.approx(0.5, abs=0.04)
    # SIR reweighs the proposal towards its own contour: the weights are spread.
    assert info["sir_ess"] > n
    assert info["n_unique"] > 0.7 * n


def test_more_oversampling_spreads_the_resample_over_more_distinct_rows():
    proposal = _gaussian_2d()
    n = 4000
    infos = {}
    for oversampling in (1, 64):
        u, info = sample_truncated(
            proposal, n=n, quantile=1e-3, n_support=10_000, oversampling=oversampling,
            rng=np.random.default_rng(6),
        )
        # Whatever the oversampling, the resample is flat on average ...
        assert _radial_cdf(proposal, u, info["threshold"]).mean() == pytest.approx(0.5, abs=0.05)
        infos[oversampling] = info
    # ... but with one candidate per draw it repeats a few candidates near the contour.
    assert infos[1]["sir_ess"] < 0.1 * n < infos[64]["sir_ess"]
    assert infos[1]["n_unique"] < 0.5 * n < infos[64]["n_unique"]


def test_sir_is_deterministic():
    proposal = _gaussian_2d()
    kwargs = dict(n=500, quantile=1e-3, n_support=2000, oversampling=8, region=_HalfPlane(0.6))
    first, info_first = sample_truncated(proposal, rng=np.random.default_rng(7), **kwargs)
    second, info_second = sample_truncated(proposal, rng=np.random.default_rng(7), **kwargs)
    other, _ = sample_truncated(proposal, rng=np.random.default_rng(8), **kwargs)

    np.testing.assert_array_equal(first, second)
    assert info_first == info_second
    assert not np.array_equal(first, other)


def test_sir_raises_when_the_region_excludes_the_contour():
    proposal = _gaussian_2d()
    with pytest.raises(RuntimeError, match="do not overlap"):
        sample_truncated(
            proposal, n=100, quantile=1e-3, n_support=2000, oversampling=4, region=_HalfPlane(0.0),
            rng=np.random.default_rng(9),
        )


def test_module_is_xspec_free_and_torch_free():
    from test_import_hygiene import FORBIDDEN_MODULES, TORCH_STACK, _collect_imports, _read, _violations

    relative_path = "bsixsa/backend/sixsa/truncation.py"
    imports = _collect_imports(_read(relative_path), relative_path)

    assert _violations(imports, FORBIDDEN_MODULES, module_level_only=False) == []
    assert _violations(imports, TORCH_STACK, module_level_only=False) == []
