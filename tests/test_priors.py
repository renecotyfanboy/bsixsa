"""Tests for the prior transforms and unit-cube truncation in ``bsixsa.priors``."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from scipy import integrate, stats

from bsixsa.priors import (
    MultipleIndependent,
    TruncatedMarginal,
    loguniform,
    norm,
    uniform,
    validate_finite_support,
)


def _example_prior() -> MultipleIndependent:
    return MultipleIndependent(
        [uniform(-2.0, 3.0), loguniform(1e-2, 1e2), norm(1.0, 0.5, -1.0, 3.0)]
    )


def _truncate_1d(dist, lo: float, hi: float):
    prior, log_mass = MultipleIndependent([dist]).truncated([lo], [hi])
    return prior.dists[0], log_mass


def _interior_grid(dist, n: int = 201) -> np.ndarray:
    low, high = (float(b) for b in dist.support())
    return np.linspace(low, high, n)[1:-1]


# name, factory, lo_u, hi_u, expected family after truncation
FAMILY_CASES = [
    ("uniform", lambda: uniform(-2.0, 3.0), 0.2, 0.7, "uniform"),
    ("loguniform", lambda: loguniform(1e-2, 1e2), 0.25, 0.75, "loguniform"),
    ("truncnorm", lambda: norm(1.0, 0.5, -1.0, 3.0), 0.1, 0.9, "truncnorm"),
    ("reciprocal", lambda: stats.reciprocal(1e-2, 1e2), 0.25, 0.75, "loguniform"),
    ("truncnorm-positional-loc", lambda: stats.truncnorm(-4.0, 4.0, 1.0, 0.5), 0.1, 0.9, "truncnorm"),
]


@pytest.mark.parametrize(
    "factory, lo, hi, family",
    [case[1:] for case in FAMILY_CASES],
    ids=[case[0] for case in FAMILY_CASES],
)
def test_closed_form_truncation_identities(factory, lo, hi, family):
    """The rebuilt marginal is the base restricted to the box and renormalised by its mass."""
    old = factory()
    new, log_mass = _truncate_1d(old, lo, hi)
    width = hi - lo

    assert new.dist.name == family
    assert log_mass == pytest.approx(np.log(width), rel=1e-15)
    np.testing.assert_allclose(new.support(), (old.ppf(lo), old.ppf(hi)), rtol=1e-12)

    v = np.linspace(0.0, 1.0, 101)
    np.testing.assert_allclose(new.ppf(v), old.ppf(lo + v * width), rtol=1e-9, atol=1e-12)

    assert hasattr(new, "isf")
    np.testing.assert_allclose(new.isf(v), new.ppf(1.0 - v), rtol=1e-9, atol=1e-12)

    theta = _interior_grid(new)
    np.testing.assert_allclose(new.logpdf(theta), old.logpdf(theta) - np.log(width), rtol=1e-9)
    np.testing.assert_allclose(new.cdf(theta), (old.cdf(theta) - lo) / width, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize(
    "factory, lo, hi, family",
    [case[1:] for case in FAMILY_CASES],
    ids=[case[0] for case in FAMILY_CASES],
)
def test_truncated_max_log_pdf_is_the_maximum_over_the_box(factory, lo, hi, family):
    """``max_log_pdf_per_axis`` still returns the log-density maximum after truncation."""
    old = factory()
    prior, _ = MultipleIndependent([old]).truncated([lo], [hi])
    new = prior.dists[0]

    low, high = (float(b) for b in new.support())
    grid = np.concatenate([np.linspace(low, high, 2001), [np.clip(1.0, low, high)]])
    expected = float(np.max(new.logpdf(grid)))

    (max_log_pdf,) = prior.max_log_pdf_per_axis()
    assert max_log_pdf == pytest.approx(expected, rel=1e-9, abs=1e-12)
    assert max_log_pdf >= expected - 1e-12


def test_truncnorm_keeps_loc_in_kwds():
    """A truncated ``norm`` keeps ``loc``/``scale`` in ``kwds``, like the ``norm()`` factory."""
    for base in (norm(1.0, 0.5, -1.0, 3.0), stats.truncnorm(-4.0, 4.0, 1.0, 0.5)):
        new, _ = _truncate_1d(base, 0.1, 0.9)
        assert new.kwds["loc"] == pytest.approx(1.0)
        assert new.kwds["scale"] == pytest.approx(0.5)


def test_max_log_pdf_resolves_positional_truncnorm_loc():
    """A truncnorm built with a positional ``loc`` peaks at ``loc``, not at 0."""
    dist = stats.truncnorm(-4.0, 4.0, 1.0, 0.5)
    (max_log_pdf,) = MultipleIndependent([dist]).max_log_pdf_per_axis()
    assert max_log_pdf == pytest.approx(float(dist.logpdf(1.0)))


@pytest.mark.parametrize(
    "factory, lo, hi",
    [(lambda: uniform(-2.0, 3.0), 0.2, 0.7), (lambda: loguniform(1e-2, 1e2), 0.25, 0.75)],
    ids=["uniform", "loguniform"],
)
def test_xspec_bayes_contribution_unchanged_for_xspec_families(factory, lo, hi):
    """XSPEC's ``cons``/``jeffreys`` terms are unnormalised, so truncation leaves them alone."""
    old = MultipleIndependent([factory()])
    new, _ = old.truncated([lo], [hi])
    for theta in _interior_grid(new.dists[0], n=23):
        assert new.xspec_bayes_contribution([theta]) == pytest.approx(
            old.xspec_bayes_contribution([theta]), rel=1e-12, abs=1e-12
        )


def test_xspec_bayes_contribution_of_truncnorm_shifts_by_the_mass():
    """The scipy-logpdf fallback term picks up the renormalisation ``2 log(hi - lo)``."""
    old = MultipleIndependent([norm(1.0, 0.5, -1.0, 3.0)])
    new, log_mass = old.truncated([0.1], [0.9])
    for theta in _interior_grid(new.dists[0], n=23):
        assert new.xspec_bayes_contribution([theta]) == pytest.approx(
            old.xspec_bayes_contribution([theta]) + 2.0 * log_mass, rel=1e-10
        )


def test_generic_fallback_identities():
    """Families without a closed form are wrapped in a ``TruncatedMarginal``."""
    old = stats.beta(2.0, 3.0)
    lo, hi = 0.15, 0.8
    width = hi - lo
    new, log_mass = _truncate_1d(old, lo, hi)

    assert isinstance(new, TruncatedMarginal)
    assert new.dist.name == "truncated"
    assert new.args == () and new.kwds == {}
    assert new.base is old
    assert log_mass == pytest.approx(np.log(width))
    np.testing.assert_allclose(new.support(), (old.ppf(lo), old.ppf(hi)), rtol=1e-12)

    v = np.linspace(0.0, 1.0, 101)
    np.testing.assert_allclose(new.ppf(v), old.ppf(lo + v * width), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(new.isf(v), new.ppf(1.0 - v), rtol=1e-9, atol=1e-12)
    assert new.ppf(0.0) == new.support()[0]
    assert new.ppf(1.0) == new.support()[1]
    assert np.isnan(new.ppf(-0.1)) and np.isnan(new.ppf(1.1))

    theta = _interior_grid(new)
    np.testing.assert_allclose(new.logpdf(theta), old.logpdf(theta) - np.log(width), rtol=1e-12)
    np.testing.assert_allclose(new.pdf(theta), old.pdf(theta) / width, rtol=1e-12)
    np.testing.assert_allclose(new.cdf(theta), (old.cdf(theta) - lo) / width, rtol=1e-12)
    np.testing.assert_allclose(new.cdf(theta) + new.sf(theta), 1.0, rtol=1e-12)

    low, high = new.support()
    outside = np.array([low - 1e-3, high + 1e-3])
    assert np.all(new.logpdf(outside) == -np.inf)
    np.testing.assert_array_equal(new.cdf(outside), [0.0, 1.0])
    np.testing.assert_array_equal(new.sf(outside), [1.0, 0.0])

    # Scalar in, scalar out -- SciPy's convention, relied on by float(dist.logpdf(x)).
    assert np.ndim(new.logpdf(0.4)) == 0
    assert np.ndim(new.ppf(0.5)) == 0


def test_generic_fallback_pickle_and_rvs():
    """A ``TruncatedMarginal`` survives pickling and samples inside its support."""
    new, _ = _truncate_1d(stats.beta(2.0, 3.0), 0.15, 0.8)

    restored = pickle.loads(pickle.dumps(new))
    assert restored.dist.name == "truncated"
    assert (restored.lo_u, restored.hi_u, restored.a, restored.b) == (new.lo_u, new.hi_u, new.a, new.b)
    v = np.linspace(0.0, 1.0, 11)
    np.testing.assert_array_equal(restored.ppf(v), new.ppf(v))

    draws = new.rvs(size=4000, random_state=0)
    low, high = new.support()
    assert draws.shape == (4000,)
    assert np.all((draws >= low) & (draws <= high))
    np.testing.assert_array_equal(draws, new.rvs(size=4000, random_state=0))
    assert stats.kstest(new.cdf(draws), "uniform").pvalue > 1e-3
    assert np.ndim(new.rvs()) == 0

    legacy = new.rvs(size=5, random_state=np.random.RandomState(1))
    assert np.all((legacy >= low) & (legacy <= high))

    prior = MultipleIndependent([uniform(0.0, 1.0), new])
    samples = prior.sample(size=64, random_state=3)
    assert samples.shape == (64, 2)
    assert np.all((samples[:, 1] >= low) & (samples[:, 1] <= high))


def test_generic_fallback_max_log_pdf_raises():
    prior, _ = MultipleIndependent([uniform(0.0, 1.0), stats.beta(2.0, 3.0)]).truncated(
        [0.0, 0.15], [1.0, 0.8]
    )
    with pytest.raises(NotImplementedError, match="TruncatedMarginal"):
        prior.max_log_pdf_per_axis()


def test_shifted_loguniform_falls_back_to_wrapper():
    """A log-uniform with a non-zero ``loc`` is not log-uniform on the box: wrap it."""
    old = stats.loguniform(1.0, 10.0, loc=1.0)
    new, _ = _truncate_1d(old, 0.2, 0.6)
    assert isinstance(new, TruncatedMarginal)
    v = np.linspace(0.0, 1.0, 11)
    np.testing.assert_allclose(new.ppf(v), old.ppf(0.2 + 0.4 * v), rtol=1e-12)


def test_retruncating_a_wrapper_composes_in_the_base_cube():
    """Truncating a ``TruncatedMarginal`` again does not nest wrappers."""
    old = stats.beta(2.0, 3.0)
    once, _ = _truncate_1d(old, 0.2, 0.8)
    twice, _ = _truncate_1d(once, 0.5, 1.0)
    assert isinstance(twice, TruncatedMarginal) and twice.base is old
    assert (twice.lo_u, twice.hi_u) == pytest.approx((0.5, 0.8))


def test_untouched_axes_keep_identity_and_mass_is_exact():
    base = _example_prior()
    lo = np.array([0.0, 0.2, 0.0])
    hi = np.array([1.0, 0.6, 1.0])
    prior, log_mass = base.truncated(lo, hi)

    assert prior.dists[0] is base.dists[0]
    assert prior.dists[2] is base.dists[2]
    assert prior.dists[1] is not base.dists[1]
    assert log_mass == pytest.approx(np.log(0.4), rel=1e-15)

    full, full_log_mass = base.truncated([0.2, 0.25, 0.1], [0.7, 0.75, 0.9])
    assert full_log_mass == pytest.approx(np.log(0.5) + np.log(0.5) + np.log(0.8), rel=1e-15)

    # The whole-prior cube map is the per-axis affine rescaling of the base cube.
    rng = np.random.default_rng(0)
    v = rng.uniform(size=(128, 3))
    lo_full, hi_full = np.array([0.2, 0.25, 0.1]), np.array([0.7, 0.75, 0.9])
    np.testing.assert_allclose(
        full.from_unit_cube(v), base.from_unit_cube(lo_full + v * (hi_full - lo_full)), rtol=1e-9
    )
    np.testing.assert_allclose(full.to_unit_cube(full.from_unit_cube(v)), v, rtol=1e-8, atol=1e-10)


def test_whole_cube_box_is_a_no_op():
    base = _example_prior()
    prior, log_mass = base.truncated(np.zeros(3), np.ones(3))
    assert log_mass == 0.0
    assert all(new is old for new, old in zip(prior.dists, base.dists))


@pytest.mark.parametrize(
    "lower, upper, match",
    [
        ([0.5, 0.0, 0.0], [0.5, 1.0, 1.0], "axis 0"),
        ([0.0, 0.7, 0.0], [1.0, 0.6, 1.0], "axis 1"),
        ([-0.1, 0.0, 0.0], [1.0, 1.0, 1.0], "axis 0"),
        ([0.0, 0.0, 0.0], [1.0, 1.0, 1.1], "axis 2"),
        ([0.0, 0.0, 0.3], [1.0, 1.0, 0.3 + 1e-12], "min_width"),
        ([0.0, np.nan, 0.0], [1.0, 1.0, 1.0], "non-finite"),
        ([0.0, 0.0], [1.0, 1.0], "shape"),
    ],
)
def test_invalid_box_raises(lower, upper, match):
    with pytest.raises(ValueError, match=match):
        _example_prior().truncated(lower, upper)


def test_truncated_marginal_rejects_invalid_interval():
    with pytest.raises(ValueError):
        TruncatedMarginal(stats.beta(2.0, 3.0), 0.6, 0.6)


def _smooth_likelihood(*theta: float) -> float:
    centres = (0.3, 1.2, 0.5)
    widths = (0.7, 0.4, 0.2)
    return float(np.exp(-0.5 * sum(((t - c) / w) ** 2 for t, c, w in zip(theta, centres, widths))))


@pytest.mark.parametrize(
    "factory, lo, hi",
    [
        (lambda: loguniform(1e-2, 1e2), 0.25, 0.75),
        (lambda: norm(1.0, 0.5, -1.0, 3.0), 0.1, 0.9),
        (lambda: stats.beta(2.0, 3.0), 0.15, 0.8),
    ],
    ids=["loguniform", "truncnorm", "beta"],
)
def test_evidence_identity_by_quadrature_1d(factory, lo, hi):
    r"""$M_B \int_B \pi_B L = \int_B \pi L$: the restricted evidence plus $\log M_B$ is exact."""
    base = MultipleIndependent([factory()])
    prior, log_mass = base.truncated([lo], [hi])
    a, b = prior.bounds[0]

    def restricted(t):
        return np.exp(prior.log_prob(np.array([t]))) * _smooth_likelihood(t)

    def full(t):
        return np.exp(base.log_prob(np.array([t]))) * _smooth_likelihood(t)

    z_box, _ = integrate.quad(restricted, a, b, epsabs=0.0, epsrel=1e-11, limit=200)
    z_full, _ = integrate.quad(full, a, b, epsabs=0.0, epsrel=1e-11, limit=200)
    assert np.exp(log_mass) * z_box == pytest.approx(z_full, rel=1e-6)


def test_evidence_identity_by_quadrature_2d():
    base = MultipleIndependent([uniform(-2.0, 3.0), norm(1.0, 0.5, -1.0, 3.0)])
    prior, log_mass = base.truncated([0.2, 0.1], [0.7, 0.9])
    (a0, b0), (a1, b1) = prior.bounds

    def integrand(model):
        # dblquad integrates f(y, x) with x the outer variable.
        return lambda t1, t0: np.exp(model.log_prob(np.array([t0, t1]))) * _smooth_likelihood(t0, t1)

    z_box, _ = integrate.dblquad(integrand(prior), a0, b0, a1, b1, epsabs=0.0, epsrel=1e-10)
    z_full, _ = integrate.dblquad(integrand(base), a0, b0, a1, b1, epsabs=0.0, epsrel=1e-10)
    assert np.exp(log_mass) * z_box == pytest.approx(z_full, rel=1e-6)


def test_validate_finite_support():
    with pytest.raises(ValueError, match="axis 1"):
        validate_finite_support(MultipleIndependent([uniform(0.0, 1.0), stats.norm()]))

    validate_finite_support(_example_prior())
    truncated, _ = _example_prior().truncated([0.2, 0.25, 0.1], [0.7, 0.75, 0.9])
    validate_finite_support(truncated)


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


# -- truncated supports never leave the original support -------------------------------


def test_truncated_uniform_upper_face_does_not_overshoot():
    """Regression: ``uniform(low, high)`` re-rounds ``low + (high - low)`` one ulp past ``high``."""
    from bsixsa._cube import to_physical

    base = MultipleIndependent([uniform(-4.04110756526943, 2.8301776623698247)])
    prior, _ = base.truncated([0.3443654068365892], [1.0])

    assert prior.dists[0].support()[1] <= base.dists[0].support()[1]
    theta = to_physical(prior, [[0.0], [1.0]])
    assert np.all(np.isfinite(base.log_prob(theta)))


def _random_marginal(rng):
    kind = int(rng.integers(5))
    low = float(rng.uniform(-10.0, 10.0))
    width = float(10.0 ** rng.uniform(-3.0, 2.0))
    if kind == 0:
        return uniform(low, low + width)
    if kind == 1:
        a = float(10.0 ** rng.uniform(-5.0, 2.0))
        return loguniform(a, a * 10.0 ** rng.uniform(0.1, 5.0))
    if kind == 2:
        a = float(10.0 ** rng.uniform(-5.0, 2.0))
        return stats.loguniform(a, a * 10.0 ** rng.uniform(0.1, 3.0), scale=10.0 ** rng.uniform(-3.0, 3.0))
    loc, scale = float(rng.uniform(-5.0, 5.0)), float(10.0 ** rng.uniform(-2.0, 1.0))
    if kind == 3:
        return norm(loc, scale, low, low + width)
    return stats.truncnorm((low - loc) / scale, (low + width - loc) / scale, loc, scale)


def test_truncated_support_stays_inside_the_original_randomized():
    from bsixsa._cube import to_physical

    rng = np.random.default_rng(20260915)
    for _ in range(3000):
        base = MultipleIndependent([_random_marginal(rng)])
        u = float(rng.uniform(0.0, 0.99))
        side = int(rng.integers(3))
        if side == 0:
            lo, hi = 0.0, u + 0.005
        elif side == 1:
            lo, hi = u, 1.0
        else:
            lo, hi = u, min(1.0, u + float(rng.uniform(1e-6, 0.5)))
        prior, _ = base.truncated([lo], [hi])

        (old_low, old_high), (new_low, new_high) = base.bounds[0], prior.bounds[0]
        assert old_low <= new_low < new_high <= old_high, (base.dists[0], lo, hi)
        # A face moved inward to stay inside is moved by a few ulps, no more.
        assert new_low == pytest.approx(float(base.dists[0].ppf(lo)), rel=1e-14, abs=1e-300)
        assert new_high == pytest.approx(float(base.dists[0].ppf(hi)), rel=1e-14, abs=1e-300)
        edges = to_physical(prior, [[0.0], [1.0]])
        if np.all(np.isfinite(base.log_prob(np.array([[old_low], [old_high]])))):
            assert np.all(np.isfinite(base.log_prob(edges))), (base.dists[0], lo, hi)


def test_untouched_truncnorm_face_is_bit_identical():
    base = norm(1.0, 0.5, -1.0, 3.0)
    lower, _ = _truncate_1d(base, 0.0, 0.4)
    upper, _ = _truncate_1d(base, 0.6, 1.0)

    assert lower.support()[0] == base.support()[0]
    assert upper.support()[1] == base.support()[1]
    assert lower.kwds == {"loc": 1.0, "scale": 0.5}


def test_generic_fallback_rejects_a_collapsed_physical_interval():
    """Regression: the wrapper used to accept ``a == b`` and return a non-normalised density."""
    narrow = stats.triang(0.5, loc=1e10, scale=1.0)
    with pytest.raises(ValueError, match="collapses"):
        MultipleIndependent([narrow]).truncated([0.5], [0.5 + 2e-9])
    with pytest.raises(ValueError, match="collapses"):
        TruncatedMarginal(narrow, 0.5, 0.5 + 2e-9)
    # The closed-form families already refused the same box.
    with pytest.raises(ValueError, match="collapses"):
        MultipleIndependent([uniform(1e10, 1e10 + 1.0)]).truncated([0.5], [0.5 + 2e-9])


def test_truncated_marginal_support_is_clipped_to_the_base_support():
    class _Overshooting:
        """ppf that rounds past the support at both ends."""

        def support(self):
            return (0.0, 1.0)

        def ppf(self, q):
            return np.asarray(q, dtype=float) * 1.2 - 0.1

    wrapped = TruncatedMarginal(_Overshooting(), 0.0, 1.0)
    assert wrapped.support() == (0.0, 1.0)


def test_composed_truncation_keeps_the_outer_faces_exact():
    once, _ = _truncate_1d(stats.beta(2.0, 3.0), 0.1, 0.7)
    rng = np.random.default_rng(3)
    for lo in rng.uniform(0.0, 0.9, size=200):
        upper, _ = _truncate_1d(once, float(lo), 1.0)
        lower, _ = _truncate_1d(once, 0.0, float(lo) + 0.05)
        assert upper.hi_u == once.hi_u and upper.support()[1] == once.support()[1]
        assert lower.lo_u == once.lo_u and lower.support()[0] == once.support()[0]
        assert once.lo_u <= upper.lo_u < upper.hi_u <= once.hi_u
