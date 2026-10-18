"""Tests for the priors and their unit-cube maps in ``bsixsa.priors``."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from bsixsa.priors import (
    MultipleIndependent,
    loguniform,
    norm,
    uniform,
    validate_finite_support,
)


def _example_prior() -> MultipleIndependent:
    return MultipleIndependent(
        [uniform(-2.0, 3.0), loguniform(1e-2, 1e2), norm(1.0, 0.5, -1.0, 3.0)]
    )


def test_max_log_pdf_resolves_positional_truncnorm_loc():
    """A truncnorm built with a positional ``loc`` peaks at ``loc``, not at 0."""
    dist = stats.truncnorm(-4.0, 4.0, 1.0, 0.5)
    (max_log_pdf,) = MultipleIndependent([dist]).max_log_pdf_per_axis()
    assert max_log_pdf == pytest.approx(float(dist.logpdf(1.0)))


def test_validate_finite_support():
    with pytest.raises(ValueError, match="axis 1"):
        validate_finite_support(MultipleIndependent([uniform(0.0, 1.0), stats.norm()]))

    validate_finite_support(_example_prior())


def test_xspec_bayes_contribution_batch_matches_single_points():
    prior = MultipleIndependent(
        [
            uniform(-2.0, 3.0),
            loguniform(1e-2, 1e2),
            norm(1.0, 0.5, -1.0, 3.0),
            stats.truncnorm(-4.0, 4.0, 1.0, 0.5),
        ]
    )
    x = prior.sample(size=97, random_state=4)

    batch = prior.xspec_bayes_contribution(x)
    assert batch.shape == (97,)

    single = [prior.xspec_bayes_contribution(row) for row in x]
    assert all(type(value) is float for value in single)
    np.testing.assert_allclose(batch, single, rtol=1e-12, atol=1e-12)

    # Independent reference: cons -> 0, jeffreys -> 2 ln x, otherwise -2 logpdf.
    reference = (
        2.0 * np.log(x[:, 1])
        - 2.0 * prior.dists[2].logpdf(x[:, 2])
        - 2.0 * prior.dists[3].logpdf(x[:, 3])
    )
    np.testing.assert_allclose(batch, reference, rtol=1e-12, atol=1e-12)

    # Same shape convention as log_prob: a one-row batch stays a batch.
    assert prior.xspec_bayes_contribution(x[:1]).shape == (1,)


def test_xspec_bayes_contribution_rejects_the_wrong_width():
    """Extra columns used to be ignored, and a 1-D prior read a length-n vector as one point."""
    prior = _example_prior()
    with pytest.raises(ValueError, match="shape"):
        prior.xspec_bayes_contribution(np.ones((4, 4)))
    with pytest.raises(ValueError, match="shape"):
        prior.xspec_bayes_contribution([0.0, 1.0, 1.0, 5.0])
    with pytest.raises(ValueError, match="shape"):
        prior.xspec_bayes_contribution(np.ones((2, 2, 3)))
    with pytest.raises(ValueError, match="shape"):
        MultipleIndependent([loguniform(1e-2, 1e2)]).xspec_bayes_contribution([1.0, 2.0, 3.0])
    assert MultipleIndependent([loguniform(1e-2, 1e2)]).xspec_bayes_contribution([[1.0], [2.0]]).shape == (2,)


def test_prior_maps_reject_points_of_the_wrong_width():
    """Regression: a too-wide input used to return uninitialised columns (from/to_unit_cube)
    or have its extra columns silently ignored (log_prob)."""
    prior = _example_prior()
    for method in (prior.from_unit_cube, prior.to_unit_cube, prior.log_prob):
        for bad in (np.full((2, 4), 0.5), np.full((2, 2), 0.5), np.full(4, 0.5), 0.5):
            with pytest.raises(ValueError, match="shape"):
                method(bad)
    # log_prob reads one point per row, so it refuses more than two dimensions...
    with pytest.raises(ValueError, match="shape"):
        prior.log_prob(np.full((2, 2, 3), 0.5))
    # ...while the cube maps keep accepting any leading shape.
    with pytest.raises(ValueError, match=r"\(\.\.\., 3\)"):
        prior.from_unit_cube(np.full((2, 5, 4), 0.5))
    with pytest.raises(ValueError, match=r"\(3,\) or \(n, 3\)"):
        prior.log_prob(np.full(4, 0.5))
    cube = np.full((2, 5, 3), 0.5)
    np.testing.assert_array_equal(prior.from_unit_cube(cube)[1, 4], prior.from_unit_cube(cube[0, 0]))
    assert prior.to_unit_cube(prior.from_unit_cube(cube)).shape == (2, 5, 3)


def test_log_prob_returns_a_float_for_one_point_and_an_array_for_a_batch():
    prior = _example_prior()
    x = prior.sample(size=5, random_state=0)
    assert prior.sample(size=(4, 5), random_state=0).shape == (4, 5, 3)
    assert np.ndim(prior.log_prob(x[0])) == 0
    assert prior.log_prob(x).shape == (5,)
    np.testing.assert_allclose(prior.log_prob(x), [prior.log_prob(row) for row in x])


@pytest.mark.parametrize(
    "make, match",
    [
        (lambda: uniform(3.0, 1.0), "low < high"),
        (lambda: uniform(0.0, np.inf), "finite"),
        (lambda: loguniform(0.0, 10.0), "0 < low"),
        (lambda: loguniform(10.0, 1.0), "0 < low"),
        (lambda: norm(0.0, 0.0, -1.0, 1.0), "scale > 0"),
        (lambda: norm(0.0, 1.0, 1.0, -1.0), "low < high"),
    ],
)
def test_prior_helpers_reject_invalid_bounds(make, match):
    """Regression: these used to fail later, inside build_prior, with an unrelated OverflowError."""
    with pytest.raises(ValueError, match=match):
        make()


def test_norm_bounds_are_physical():
    """``low``/``high`` of ``norm`` are bounds on the parameter, not standardised truncnorm limits."""
    dist = norm(loc=10.0, scale=2.0, low=7.0, high=15.0)
    assert dist.support() == pytest.approx((7.0, 15.0))
    assert float(dist.ppf(0.5)) == pytest.approx(10.0 + 2.0 * stats.truncnorm(-1.5, 2.5).ppf(0.5))
