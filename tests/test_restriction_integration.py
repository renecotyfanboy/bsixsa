"""End-to-end test of history matching with the real `bsixsa.restriction` package.

The history matching runs for real (torch regressor, waves, region) on the
C-stat of the d=2 `tests/_analytic.AnalyticSpectrum`, and is checked against
the tensor-grid quadrature of `grid_truth`: the region keeps the posterior.
SIXSA's use of the restriction is covered by `test_sixsa_backend.py`.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import special

pytest.importorskip("torch")

from _analytic import AnalyticSpectrum, grid_truth  # noqa: E402

from bsixsa.restriction import HistoryMatching, Restriction, run_history_matching  # noqa: E402

SPECTRUM_SEED = 12345


def _small_config() -> HistoryMatching:
    return HistoryMatching(
        n_waves=3,
        simulations_per_parameter=200,
        hidden=32,
        blocks=2,
        epochs=200,
        patience=10,
        mass_probe=20_000,
        seed=0,
    )


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


def _tensor_grid(lo, hi, n_grid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nodes, per-axis spacing and log trapezoid weights of a tensor grid.

    Returns:
        `(u, h, log_w)`: nodes of shape `(n_grid**d, d)` in C order, spacings
        of shape `(d,)` and log weights of shape `(n_grid**d,)`.
    """
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    axes, log_weights = [], []
    for a, b in zip(lo, hi):
        weights = np.full(n_grid, (b - a) / (n_grid - 1))
        weights[[0, -1]] *= 0.5
        axes.append(np.linspace(a, b, n_grid))
        log_weights.append(np.log(weights))
    mesh = np.meshgrid(*axes, indexing="ij")
    u = np.stack([m.ravel() for m in mesh], axis=1)
    log_w = sum(np.meshgrid(*log_weights, indexing="ij")).ravel()
    return u, (hi - lo) / (n_grid - 1), log_w


def _draw_from_grid(u, h, log_mass, n: int, rng, lo, hi) -> np.ndarray:
    """Inverse-CDF draws of grid nodes by mass, jittered uniformly within their cell."""
    cdf = np.cumsum(np.exp(log_mass - special.logsumexp(log_mass)))
    index = np.minimum(np.searchsorted(cdf, rng.random(n) * cdf[-1]), len(cdf) - 1)
    jitter = (rng.random((n, u.shape[1])) - 0.5) * h
    return np.clip(u[index] + jitter, lo, hi)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spectrum() -> AnalyticSpectrum:
    return AnalyticSpectrum(d=2, seed=SPECTRUM_SEED)


@pytest.fixture(scope="module")
def truth(spectrum) -> dict:
    return grid_truth(spectrum)


@pytest.fixture(scope="module")
def posterior_u(spectrum, truth) -> np.ndarray:
    """20 000 posterior draws on the grid of `grid_truth`, mapped back to the cube by the prior."""
    lo, hi = truth["window_lo"], truth["window_hi"]
    u, h, log_w = _tensor_grid(lo, hi, truth["n_grid"])
    log_mass = -0.5 * spectrum.cstat_u(u) + log_w
    assert special.logsumexp(log_mass) == pytest.approx(truth["log_z"], abs=1e-9)
    draws = _draw_from_grid(u, h, log_mass, 20_000, np.random.default_rng(7), lo, hi)
    prior = spectrum.prior()
    return prior.to_unit_cube(prior.from_unit_cube(draws))


# ---------------------------------------------------------------------------
# History matching
# ---------------------------------------------------------------------------


def test_history_matching_keeps_the_posterior(spectrum, truth, posterior_u):
    restriction = run_history_matching(spectrum.cstat_u, 2, _small_config(), param_names=spectrum.parameter_names)

    assert isinstance(restriction, Restriction)
    assert restriction.config == _small_config()
    assert restriction.region.accepts(posterior_u).mean() >= 0.99
    assert 0.0 < restriction.prior_mass < 1.0
    assert restriction.best == pytest.approx(truth["c_min"], abs=restriction.span)
    assert restriction.n_simulations == len(restriction.cstat) == sum(row["n"] for row in restriction.history)
    np.testing.assert_allclose(restriction.cstat, spectrum.cstat_u(restriction.u), rtol=1e-12)
