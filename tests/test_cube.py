"""Tests for the unit-cube primitives in `bsixsa._cube`."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault("xspec", MagicMock())

from bsixsa._cube import (  # noqa: E402
    in_cube,
    latin_hypercube,
    to_physical,
)
from bsixsa.priors import MultipleIndependent, loguniform, norm, uniform  # noqa: E402

SRC_DIR = Path(__file__).resolve().parents[1] / "src" / "bsixsa"


def _example_prior() -> MultipleIndependent:
    return MultipleIndependent(
        [uniform(0.05, 0.3), loguniform(1e-5, 1e-2), norm(1.0, 0.5, -1.0, 3.0)]
    )


# --------------------------------------------------------------------- in_cube


def test_in_cube_faces_are_inside():
    u = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.5, 1.0]])
    np.testing.assert_array_equal(in_cube(u), [True, True, True])


def test_in_cube_rejects_points_just_outside_a_face():
    u = np.array([[1.0 + 1e-12, 0.5], [0.5, -1e-12], [0.5, 0.5]])
    np.testing.assert_array_equal(in_cube(u), [False, False, True])


def test_in_cube_compares_in_float64():
    """A float32 view would round 1 + 1e-12 onto the face; float64 must not."""
    u = [[1.0 + 1e-12, 0.5]]
    assert np.float32(u[0][0]) == 1.0
    assert not in_cube(u)[0]


def test_in_cube_non_finite_rows_are_outside():
    u = np.array([[np.nan, 0.5], [0.5, np.inf], [-np.inf, 0.5], [0.2, 0.8]])
    np.testing.assert_array_equal(in_cube(u), [False, False, False, True])


def test_in_cube_accepts_one_dimensional_input():
    inside = in_cube(np.array([0.0, 0.3, 1.0]))
    outside = in_cube(np.array([0.0, 1.5]))
    assert inside.shape == (1,) and inside.dtype == bool and inside[0]
    assert outside.shape == (1,) and not outside[0]


def test_in_cube_rejects_three_dimensional_input():
    with pytest.raises(ValueError, match="shape"):
        in_cube(np.zeros((2, 2, 2)))


def test_in_cube_empty_input_has_no_rows():
    """Regression: an empty 1-D input used to become one vacuously-inside point."""
    for empty in (np.array([]), [], np.empty((0, 3))):
        result = in_cube(empty)
        assert result.shape == (0,) and result.dtype == bool


def test_to_physical_maps_non_finite_coordinates_to_nan():
    """Regression: +-inf used to be clipped onto a face and folded as a valid point."""
    prior = _example_prior()
    u = np.array([[np.inf, -np.inf, 0.5], [np.nan, 0.5, 0.5], [0.2, 0.4, 0.6]])

    theta = to_physical(prior, u)

    assert np.all(np.isnan(theta[0, :2])) and np.isfinite(theta[0, 2])
    assert np.isnan(theta[1, 0]) and np.all(np.isfinite(theta[1, 1:]))
    np.testing.assert_allclose(theta[2], prior.from_unit_cube(u[2]))
    assert not np.any(in_cube(u[:2]))


# ----------------------------------------------------------------- to_physical


def test_to_physical_clips_u_outside_the_cube_to_the_support():
    prior = _example_prior()
    u = np.array([[-0.5, -0.5, -0.5], [1.5, 1.5, 1.5], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    theta = to_physical(prior, u)

    assert theta.dtype == np.float64 and theta.shape == (4, 3)
    assert np.all(np.isfinite(theta))
    low = np.array([b[0] for b in prior.bounds])
    high = np.array([b[1] for b in prior.bounds])
    np.testing.assert_allclose(theta[0], low, rtol=1e-12)
    np.testing.assert_allclose(theta[1], high, rtol=1e-12)
    np.testing.assert_allclose(theta[2], low, rtol=1e-12)
    np.testing.assert_allclose(theta[3], high, rtol=1e-12)


def test_to_physical_is_support_bounded_where_ppf_overshoots():
    """`loguniform.ppf` lands below its lower bound for tiny u; the output must not."""
    prior = _example_prior()
    tiny = np.concatenate([[0.0], np.logspace(-17, -1, 1000), 1.0 - np.logspace(-17, -1, 1000), [1.0]])
    u = np.column_stack([tiny, tiny, tiny])

    theta = to_physical(prior, u)

    for j, (low, high) in enumerate(prior.bounds):
        assert np.all(theta[:, j] >= low)
        assert np.all(theta[:, j] <= high)


def test_to_physical_clips_a_ppf_that_leaves_the_support():
    """Deterministic overshoot, independent of scipy's rounding."""
    prior = SimpleNamespace(
        bounds=[(0.0, 1.0), (2.0, 3.0)],
        from_unit_cube=lambda u: np.column_stack([1.1 * u[:, 0] - 0.05, 2.0 + 1.2 * u[:, 1] - 0.1]),
    )
    theta = to_physical(prior, [[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    np.testing.assert_allclose(theta, [[0.0, 2.0], [1.0, 3.0], [0.5, 2.5]])


def test_to_physical_casts_float32_input_to_float64():
    prior = _example_prior()
    u = np.full((2, 3), 0.25, dtype=np.float32)
    assert to_physical(prior, u).dtype == np.float64


def test_to_physical_accepts_one_dimensional_input_and_checks_width():
    prior = _example_prior()
    assert to_physical(prior, [0.5, 0.5, 0.5]).shape == (1, 3)
    with pytest.raises(ValueError, match="axes"):
        to_physical(prior, [[0.5, 0.5]])


def test_to_physical_round_trips_with_to_unit_cube():
    prior = _example_prior()
    rng = np.random.default_rng(0)
    u = rng.uniform(0.01, 0.99, size=(512, prior.ndim))

    theta = to_physical(prior, u)
    u_back = prior.to_unit_cube(theta)

    assert u_back.dtype == np.float64 and u_back.shape == u.shape
    np.testing.assert_allclose(u_back, u, rtol=0, atol=1e-9)
    np.testing.assert_allclose(to_physical(prior, u_back), theta, rtol=1e-9, atol=0)


# ------------------------------------------------------------- latin_hypercube


@pytest.mark.parametrize(("d", "n"), [(1, 7), (3, 50), (6, 128)])
def test_latin_hypercube_is_exactly_stratified(d, n):
    points = latin_hypercube(d, n, np.random.default_rng(3))

    assert points.shape == (n, d) and points.dtype == np.float64
    assert np.all(points >= 0.0) and np.all(points < 1.0)
    for j in range(d):
        strata = np.floor(points[:, j] * n).astype(int)
        np.testing.assert_array_equal(np.sort(strata), np.arange(n))


def test_latin_hypercube_is_deterministic_for_a_seed():
    a = latin_hypercube(4, 64, np.random.default_rng(11))
    b = latin_hypercube(4, 64, np.random.default_rng(11))
    c = latin_hypercube(4, 64, np.random.default_rng(12))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_latin_hypercube_rejects_empty_designs():
    with pytest.raises(ValueError):
        latin_hypercube(0, 10, np.random.default_rng(0))
    with pytest.raises(ValueError):
        latin_hypercube(2, 0, np.random.default_rng(0))




def test_cube_utilities_import_without_xspec_or_torch():
    """Loaded by file path, so the package `__init__` (which imports XSPEC) is bypassed."""
    code = f"""
import importlib.util, sys
for name in ("_cube", "_torch_utils"):
    spec = importlib.util.spec_from_file_location("m_" + name, {str(SRC_DIR)!r} + "/" + name + ".py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
bad = sorted(m for m in ("xspec", "torch", "bsixsa") if m in sys.modules)
assert not bad, bad
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
