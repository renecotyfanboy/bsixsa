"""Tests for SIXSA's unit-Gaussian latent space and first-round QMC sampling."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault("xspec", MagicMock())

from bsixsa.priors import MultipleIndependent, loguniform, norm, uniform


def _example_prior() -> MultipleIndependent:
    return MultipleIndependent(
        [uniform(-2.0, 3.0), loguniform(1e-2, 1e2), norm(1.0, 0.5, -1.0, 3.0)]
    )


def test_unit_gaussian_round_trips_with_to_unit_gaussian():
    """`to_unit_gaussian` inverts `from_unit_gaussian` over a moderate latent range."""
    prior = _example_prior()

    rng = np.random.default_rng(0)
    z = rng.normal(size=(256, prior.ndim))  # moderate range -> no tail clipping
    physical = prior.from_unit_gaussian(z)
    z_back = prior.to_unit_gaussian(physical)

    assert physical.shape == (256, prior.ndim)
    np.testing.assert_allclose(z_back, z, rtol=0, atol=1e-6)


def test_from_unit_gaussian_maps_into_prior_bounds():
    """Standard-normal latent draws land in-bounds after `from_unit_gaussian`."""
    prior = _example_prior()

    rng = np.random.default_rng(1)
    # Include extreme draws to confirm finite, in-bounds physical values at the tails.
    z = np.concatenate(
        [rng.normal(size=(512, prior.ndim)), np.full((1, prior.ndim), 8.0)]
    )
    physical = prior.from_unit_gaussian(z)

    assert np.all(np.isfinite(physical))
    for j, (low, high) in enumerate(prior.bounds):
        assert np.all(physical[:, j] >= low)
        assert np.all(physical[:, j] <= high)


def test_qmc_draws_are_approximately_standard_normal_and_in_bounds():
    """`MultivariateNormalQMC` fills N(0, I) and maps in-bounds via `from_unit_gaussian`."""
    from scipy.stats import qmc

    prior = _example_prior()
    n = 4096
    z = qmc.MultivariateNormalQMC(mean=np.zeros(prior.ndim), seed=0).random(n)

    assert z.shape == (n, prior.ndim)
    assert np.allclose(z.mean(axis=0), 0.0, atol=0.1)
    assert np.allclose(z.std(axis=0), 1.0, atol=0.1)

    physical = prior.from_unit_gaussian(z)
    for j, (low, high) in enumerate(prior.bounds):
        assert np.all(physical[:, j] >= low)
        assert np.all(physical[:, j] <= high)


def _make_backend(ndim: int, first_round_sampling: str):
    from bsixsa.backend.config import SIXSA
    from bsixsa.backend.sixsa import SIXSABackend

    solver = MagicMock()
    solver.prior.ndim = ndim
    solver.outputfiles_basename = "/tmp"
    solver.parameter_names = [f"p{i}" for i in range(ndim)]

    config = SIXSA(
        trace=False,
        num_simulations_per_round=[16, 16],
        first_round_sampling=first_round_sampling,
        plot_diagnostics=False,
    )
    return SIXSABackend(solver=solver, config=config)


@pytest.mark.parametrize("mode", ["qmc", "lhs", "prior"])
def test_initial_latent_shape_and_dtype(mode):
    """`_initial_latent` returns a float32 `(n, ndim)` tensor in the unbounded latent."""
    pytest.importorskip("torch")
    pytest.importorskip("sbi")
    import torch

    backend = _make_backend(ndim=3, first_round_sampling=mode)
    latent = backend._initial_latent(50)

    assert isinstance(latent, torch.Tensor)
    assert latent.dtype == torch.float32
    assert tuple(latent.shape) == (50, 3)


def test_initial_latent_lhs_is_stratified():
    """The LHS design is per-marginal stratified: mapping the latent back to the
    unit cube (`Phi(z)`) must place exactly one sample in each of the n bins of
    every dimension -- the Latin-hypercube property `_sixsa_dev` relies on."""
    pytest.importorskip("torch")
    pytest.importorskip("sbi")
    from scipy.stats import norm as scipy_norm

    n = 64
    backend = _make_backend(ndim=3, first_round_sampling="lhs")
    z = backend._initial_latent(n).numpy()

    u = scipy_norm.cdf(z)
    bins = np.floor(u * n).astype(int)
    for j in range(z.shape[1]):
        assert sorted(bins[:, j]) == list(range(n))


def test_first_round_sampling_rejects_unknown_mode():
    """`first_round_sampling` is validated -- a typo can no longer silently fall
    through to plain i.i.d. prior sampling."""
    from pydantic import ValidationError

    from bsixsa.backend.config import SIXSA

    with pytest.raises(ValidationError, match="first_round_sampling"):
        SIXSA(num_simulations_per_round=[16], first_round_sampling="bogus")
