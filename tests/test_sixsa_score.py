"""Tests for `bsixsa.backend.sixsa.score`, on the analytic 64-channel spectrum."""

from __future__ import annotations

import functools
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.optimize import minimize

sys.modules.setdefault("xspec", MagicMock())

from _analytic import AnalyticSpectrum  # noqa: E402

from bsixsa._cube import to_physical  # noqa: E402
from bsixsa.backend.sixsa.score import PRIOR_PRECISION, ScoreCompressor  # noqa: E402


def _problem(d: int):
    """The analytic spectrum, its fold function on the cube (logging the rows folded), and its maximum likelihood."""
    spectrum = AnalyticSpectrum(d=d)
    prior = spectrum.prior()
    rows = []

    def fold(u):
        u = np.asarray(u, dtype=np.float64)
        rows.append(len(u))
        mu = spectrum.model_counts(to_physical(prior, u))
        return spectrum.cash(mu, spectrum.observed), mu

    return spectrum, fold, rows, _maximum_likelihood(d)


@functools.cache
def _maximum_likelihood(d: int):
    """Reference maximum likelihood in the cube, by Nelder-Mead from the truth."""
    spectrum = AnalyticSpectrum(d=d)
    u_truth = spectrum.prior().to_unit_cube(spectrum.truth[None, :])[0]
    return minimize(lambda u: spectrum.cstat_u(u[None, :])[0], u_truth, method="Nelder-Mead",
                    options=dict(xatol=1e-10, fatol=1e-10, maxiter=20_000))


@pytest.fixture(scope="module", params=[2, 3])
def fitted(request):
    spectrum, fold, rows, best = _problem(request.param)
    compressor = ScoreCompressor(fold, spectrum.observed)
    u_start = np.clip(best.x + 0.05 * np.random.default_rng(1).standard_normal(best.x.size), 0.01, 0.99)
    record = compressor.fit(u_start)
    return compressor, fold, rows, best, record


def test_fit_reaches_the_maximum_likelihood(fitted):
    compressor, _, _, best, record = fitted
    assert compressor.cstat_star == pytest.approx(best.fun, abs=1e-6)
    assert np.all(np.abs(compressor.u_star - best.x) < 1e-3 * compressor.sigma)
    assert record["cstat_start"] > record["cstat_star"] == compressor.cstat_star
    assert 1 <= record["iterations"] <= 5
    # At the maximum likelihood the score of the observation vanishes: its code is the fiducial.
    code = compressor.encode(compressor.observed)
    assert code.shape == (1, compressor.u_star.size) and code.dtype == np.float32
    assert np.all(np.abs(code[0] - compressor.u_star) < 1e-3 * compressor.sigma)


def test_derivatives_and_fisher_match_the_model(fitted):
    compressor, fold, _, _, _ = fitted
    d, u = compressor.u_star.size, compressor.u_star
    h = 1e-6
    reference = np.array([(fold(u + h * e)[1][0] - fold(u - h * e)[1][0]) / (2 * h) for e in np.eye(d)[:, None, :]])
    weights = 1.0 / compressor.mu_star
    error = np.sqrt(((compressor.dmu - reference) ** 2 * weights).sum(1) / (reference**2 * weights).sum(1))
    assert np.all(error < 1e-3), error
    fisher = (reference * weights) @ reference.T
    scale = np.sqrt(np.diag(fisher))
    np.testing.assert_allclose(compressor.fisher / np.outer(scale, scale), fisher / np.outer(scale, scale), atol=2e-3)
    np.testing.assert_allclose(compressor.sigma, np.sqrt(np.diag(np.linalg.inv(compressor.fisher))), rtol=1e-8)
    assert not compressor.unstable.any() and np.all(compressor.steps <= 1e-2)


def test_codes_have_the_regularised_fisher_covariance_at_the_fiducial(fitted):
    compressor = fitted[0]
    counts = np.random.default_rng(2).poisson(compressor.mu_star, size=(40_000, compressor.mu_star.size))
    codes = compressor.encode(counts).astype(np.float64)
    offset = (codes.mean(0) - compressor.u_star) / compressor.sigma
    assert np.all(np.abs(offset) < 0.03), offset
    # Cov(t) = A F A with A = (F + P)^-1, close to F^-1 when the data dominate the prior.
    a = np.linalg.inv(compressor.fisher + PRIOR_PRECISION * np.eye(compressor.u_star.size))
    scale = np.outer(compressor.sigma, compressor.sigma)
    np.testing.assert_allclose(np.cov(codes.T) / scale, a @ compressor.fisher @ a / scale, atol=0.03)


def test_every_row_folded_is_counted():
    spectrum, fold, rows, best = _problem(2)
    compressor = ScoreCompressor(fold, spectrum.observed)
    compressor.fit(np.clip(best.x + 0.05, 0.0, 1.0))
    assert compressor.n_folds == sum(rows)
    assert compressor.generation == len(compressor.history) > 1
    # The fit folds its candidates outside the maps it builds.
    assert sum(record["n_folds"] for record in compressor.history) < compressor.n_folds


def test_build_reuses_a_fold_given_at_the_fiducial():
    spectrum, fold, rows, best = _problem(2)
    compressor = ScoreCompressor(fold, spectrum.observed)
    compressor.build(best.x)
    folds = compressor.n_folds
    cstat, mu = fold(best.x[None, :])
    compressor.build(best.x, cstat=float(cstat[0]), mu=mu[0], reuse_steps=True)
    # Two folds per parameter, and none for the fiducial itself.
    assert compressor.n_folds - folds == 2 * best.x.size
    assert compressor.history[-1]["n_folds"] == 2 * best.x.size and compressor.generation == 2


def test_a_fiducial_on_a_face_of_the_cube_uses_one_sided_differences():
    spectrum, fold, _, best = _problem(2)
    u = best.x.copy()
    u[0] = 0.0
    compressor = ScoreCompressor(fold, spectrum.observed)
    compressor.build(u)
    h = 1e-6
    forward = (fold((u + h * np.eye(2)[0])[None, :])[1][0] - fold(u[None, :])[1][0]) / h
    weights = 1.0 / compressor.mu_star
    error = np.sqrt(((compressor.dmu[0] - forward) ** 2 * weights).sum() / (forward**2 * weights).sum())
    assert error < 0.03 and np.all(np.isfinite(compressor.encode(spectrum.observed)))


def test_a_parameter_the_spectrum_ignores_keeps_its_code_at_the_fiducial():
    spectrum, fold, _, best = _problem(2)

    def fold_with_a_dead_axis(u):
        return fold(np.asarray(u)[:, :2])

    compressor = ScoreCompressor(fold_with_a_dead_axis, spectrum.observed)
    u_star = np.append(best.x, 0.5)
    compressor.build(u_star)
    assert compressor.fisher[2, 2] == 0 and np.isinf(compressor.sigma[2]) and np.isfinite(compressor.condition)
    counts = np.random.default_rng(3).poisson(compressor.mu_star, size=(100, spectrum.n_bins))
    codes = compressor.encode(counts)
    assert np.all(codes[:, 2] == np.float32(0.5))
    assert np.all(np.isfinite(compressor.sigma[:2]))


def test_a_derivative_buried_in_noise_is_flagged():
    spectrum, fold, _, best = _problem(2)
    rng = np.random.default_rng(4)

    def noisy_fold(u):
        cstat, mu = fold(np.asarray(u)[:, :2])
        # The third axis only moves the counts by noise, at the 1e-6 level.
        return cstat, mu * (1.0 + 1e-6 * rng.standard_normal(mu.shape))

    compressor = ScoreCompressor(noisy_fold, spectrum.observed)
    compressor.build(np.append(best.x, 0.5))
    assert compressor.unstable.tolist() == [False, False, True]
    assert compressor.history[-1]["unstable"] == [2]


def test_encoding_before_a_map_says_what_to_do():
    spectrum, fold, _, _ = _problem(2)
    with pytest.raises(RuntimeError, match=r"call build\(\) or fit\(\) first"):
        ScoreCompressor(fold, spectrum.observed).encode(spectrum.observed)


@pytest.mark.parametrize("prior_precision", [0.0, PRIOR_PRECISION])
def test_the_prior_precision_keeps_a_degenerate_direction_at_the_scale_of_the_prior(prior_precision):
    channels = np.linspace(0.0, 1.0, 64)
    profiles = ((0.3, 0.1), (0.7, 0.1), (0.5, 0.3))
    line_a, line_b, continuum = (np.exp(-0.5 * ((channels - c) / w) ** 2) for c, w in profiles)

    def fold(u):
        # Axes 0 and 2 have nearly the same profile: their difference is barely constrained.
        u = np.asarray(u, dtype=np.float64)
        mu = 20.0 + 1000.0 * (u[:, :1] * line_a + u[:, 2:] * (line_a + 3e-4 * line_b)) + 500.0 * u[:, 1:2] * continuum
        return np.zeros(len(u)), mu

    u_star = np.array([0.4, 0.5, 0.4])
    rng = np.random.default_rng(5)
    compressor = ScoreCompressor(fold, rng.poisson(fold(u_star[None, :])[1][0]), prior_precision=prior_precision)
    compressor.build(u_star)
    assert compressor.condition > 1e6 and not compressor.unstable.any()
    codes = compressor.encode(rng.poisson(compressor.mu_star, size=(2000, channels.size))).astype(np.float64)
    spread = np.std(codes[:, 0] - codes[:, 2])
    if prior_precision:
        # Same information as the score, but the poorly measured difference stays within the prior.
        assert spread < 1.0 / np.sqrt(12.0)
        a = np.linalg.inv(compressor.fisher + prior_precision * np.eye(3))
        np.testing.assert_allclose(np.std(codes, axis=0), np.sqrt(np.diag(a @ compressor.fisher @ a)), rtol=0.1)
    else:
        assert spread > 10.0
