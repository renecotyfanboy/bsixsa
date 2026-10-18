r"""Unit-cube helpers.

The prior unit cube is $u = F(\theta)$, where $F$ applies each prior marginal
CDF independently.
"""

from __future__ import annotations

import operator
from typing import Any

import numpy as np
from scipy.stats import qmc

__all__ = [
    "in_cube",
    "latin_hypercube",
    "to_physical",
]


def _as_2d_float64(x: Any, name: str) -> np.ndarray:
    """Return *x* as a float64 array of shape ``(n, d)``; an empty 1-D input gives shape ``(0, 0)``."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim > 2:
        raise ValueError(f"{name} must have shape (d,) or (n, d), got {arr.shape}.")
    if arr.ndim == 1 and arr.size == 0:
        # atleast_2d would make this one 0-dimensional point, which every
        # all()-based test vacuously accepts.
        return arr.reshape(0, 0)
    return np.atleast_2d(arr)


def in_cube(u: Any) -> np.ndarray:
    r"""Test membership in the closed unit cube $[0, 1]^d$.

    The test is done in float64, so a point just outside a face is not rounded
    back onto it.

    Parameters:
        u: Points of shape ``(d,)`` or ``(n, d)``.

    Returns:
        Boolean array of shape ``(n,)``; rows with ``NaN`` or $\pm\infty$ are ``False``.
    """
    u = _as_2d_float64(u, "u")
    # NaN compares False on both sides, so non-finite rows fail.
    return np.all((u >= 0.0) & (u <= 1.0), axis=1)


def to_physical(prior: Any, u: Any) -> np.ndarray:
    r"""Map unit-cube points to physical parameters, $\theta = F^{-1}(u)$.

    *u* is clipped to $[0, 1]$ and the result to the prior support, which
    absorbs rounding errors of the inverse CDFs. Non-finite coordinates map to
    ``NaN``.

    Parameters:
        prior: A [`MultipleIndependent`][bsixsa.priors.MultipleIndependent] prior.
        u: Unit-cube points of shape ``(d,)`` or ``(n, d)``.

    Returns:
        Float64 array of shape ``(n, d)``.

    Raises:
        ValueError: If the width of *u* does not match the prior.
    """
    u = _as_2d_float64(u, "u")
    u = np.where(np.isfinite(u), np.clip(u, 0.0, 1.0), np.nan)
    bounds = np.asarray(prior.bounds, dtype=np.float64).reshape(-1, 2)
    if len(bounds) != u.shape[1]:
        raise ValueError(f"Prior has {len(bounds)} axes but the points have {u.shape[1]} columns.")
    theta = np.asarray(prior.from_unit_cube(u), dtype=np.float64)
    return np.clip(theta, bounds[:, 0], bounds[:, 1])


def latin_hypercube(d: int, n: int, rng: np.random.Generator) -> np.ndarray:
    r"""Draw a Latin hypercube design of *n* points on $[0, 1)^d$.

    Each axis is split into *n* equal strata holding exactly one point each.

    Parameters:
        d: Dimension of the cube.
        n: Number of points.
        rng: Random generator.

    Returns:
        Float64 array of shape ``(n, d)``.
    """
    d, n = operator.index(d), operator.index(n)
    if d < 1 or n < 1:
        raise ValueError(f"latin_hypercube expects d >= 1 and n >= 1, got d={d}, n={n}.")
    sampler = qmc.LatinHypercube(d, rng=rng)
    return np.asarray(sampler.random(n), dtype=np.float64)
