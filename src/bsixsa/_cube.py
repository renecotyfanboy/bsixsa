r"""Unit-cube helpers shared by the restriction and SIXSA layers.

Every restriction works in the unit cube $u = F(\theta)$, where $F$ applies
each prior marginal CDF independently. This module holds the few primitives
that must behave identically everywhere:

- the closed-cube membership test used in importance weights;
- the float64 maps between the cube and physical space;
- exact Latin hypercube designs;
- deterministic child random streams derived from one ``SeedSequence``.

The module itself imports numpy and scipy only (no XSPEC, no torch, no
``bsixsa.solver``). Importing it as ``bsixsa._cube`` still executes the package
``bsixsa/__init__``, which imports the solver and therefore XSPEC; code that
must run without XSPEC (e.g. pickled workers) has to load this file without
going through the package, or ship the functions it needs by value.
"""

from __future__ import annotations

import operator
from typing import Any

import numpy as np
from scipy.stats import qmc

__all__ = [
    "child_rng",
    "child_seed",
    "in_cube",
    "latin_hypercube",
    "make_seed_sequence",
    "to_cube",
    "to_physical",
]


def _as_2d_float64(x: Any, name: str) -> np.ndarray:
    """Return *x* as a float64 array of shape ``(n, d)``.

    Parameters:
        x:
            Array-like of shape ``(d,)`` or ``(n, d)``.
        name:
            Argument name used in the error message.

    Returns:
        A float64 array of shape ``(n, d)``; a non-empty 1-D input becomes one
        row and an empty 1-D input becomes zero rows, shape ``(0, 0)``.

    Raises:
        ValueError: If *x* has more than two dimensions.
    """
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

    The comparison is done in float64 so that a float32 input cannot round a
    point just outside a face (e.g. $1 + 10^{-12}$) back onto it. Faces are
    included: $u_j = 0$ and $u_j = 1$ are inside.

    Parameters:
        u:
            Points of shape ``(d,)`` or ``(n, d)``.

    Returns:
        Boolean array of shape ``(n,)`` (``(1,)`` for a non-empty 1-D input,
        ``(0,)`` for an empty one), ``True``
        where every coordinate of the row lies in $[0, 1]$. Rows containing
        ``NaN`` or $\pm\infty$ are ``False``.

    Raises:
        ValueError: If *u* has more than two dimensions.
    """
    u = _as_2d_float64(u, "u")
    # NaN compares False on both sides, so non-finite rows fail without an
    # explicit isfinite pass.
    return np.all((u >= 0.0) & (u <= 1.0), axis=1)


def _support_bounds(prior: Any, ndim: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis support ``(low, high)`` of *prior* as float64 arrays.

    Parameters:
        prior:
            A prior exposing ``bounds`` (a sequence of ``(low, high)``) or
            ``dists`` (distributions with a ``support()`` method).
        ndim:
            Expected number of axes.

    Returns:
        ``(low, high)``, each of shape ``(ndim,)``.

    Raises:
        ValueError: If the number of bounds does not match *ndim*.
    """
    bounds = getattr(prior, "bounds", None)
    if bounds is None:
        bounds = [dist.support() for dist in prior.dists]
    bounds = np.asarray(bounds, dtype=np.float64).reshape(-1, 2)
    if bounds.shape[0] != ndim:
        raise ValueError(
            f"Prior has {bounds.shape[0]} axes but the points have {ndim} columns."
        )
    return bounds[:, 0], bounds[:, 1]


def to_physical(prior: Any, u: Any) -> np.ndarray:
    r"""Map unit-cube points to physical parameters, $\theta = F^{-1}(u)$.

    Finite coordinates of *u* are clipped to $[0, 1]$ before the inverse CDF
    (a truncated normal ``ppf`` returns ``NaN`` outside it), and the result is
    clipped to each axis' support because ``ppf`` can land a rounding error
    outside it (e.g. a ``loguniform`` near its lower bound). Non-finite
    coordinates (``NaN`` and $\pm\infty$) map to ``NaN`` instead of being
    clipped onto a face, so a corrupted row stays non-finite and fails the
    fold's finiteness check; filter such rows with
    [`in_cube`][bsixsa._cube.in_cube] first.

    Parameters:
        prior:
            A [`MultipleIndependent`][bsixsa.priors.MultipleIndependent]-like
            prior exposing ``from_unit_cube`` and ``bounds`` (or ``dists``).
        u:
            Unit-cube points of shape ``(d,)`` or ``(n, d)``.

    Returns:
        Float64 array of shape ``(n, d)``, inside the prior support except
        for the ``NaN`` images of non-finite coordinates.

    Raises:
        ValueError: If *u* has more than two dimensions or its width does not
            match the prior.
    """
    u = _as_2d_float64(u, "u")
    u = np.where(np.isfinite(u), np.clip(u, 0.0, 1.0), np.nan)
    low, high = _support_bounds(prior, u.shape[1])
    theta = np.asarray(prior.from_unit_cube(u), dtype=np.float64)
    return np.clip(theta, low, high)


def to_cube(prior: Any, theta: Any) -> np.ndarray:
    r"""Map physical parameters to the unit cube, $u = F(\theta)$.

    Parameters:
        prior:
            A [`MultipleIndependent`][bsixsa.priors.MultipleIndependent]-like
            prior exposing ``to_unit_cube``.
        theta:
            Physical points of shape ``(d,)`` or ``(n, d)``.

    Returns:
        Float64 array of shape ``(n, d)`` clipped to $[0, 1]$.

    Raises:
        ValueError: If *theta* has more than two dimensions.
    """
    theta = _as_2d_float64(theta, "theta")
    u = np.asarray(prior.to_unit_cube(theta), dtype=np.float64)
    return np.clip(u, 0.0, 1.0)


def latin_hypercube(d: int, n: int, rng: np.random.Generator) -> np.ndarray:
    r"""Draw an exact Latin hypercube design on $[0, 1)^d$.

    Each axis is split into *n* equal strata and every stratum holds exactly
    one point, with a uniform position inside it.

    Parameters:
        d:
            Dimension of the cube.
        n:
            Number of points (and of strata per axis).
        rng:
            Random generator consumed by the design.

    Returns:
        Float64 array of shape ``(n, d)`` with values in $[0, 1)$.

    Raises:
        ValueError: If *d* or *n* is smaller than 1.
    """
    d, n = operator.index(d), operator.index(n)
    if d < 1 or n < 1:
        raise ValueError(f"latin_hypercube expects d >= 1 and n >= 1, got d={d}, n={n}.")
    try:
        sampler = qmc.LatinHypercube(d, rng=rng)
    except TypeError:
        # scipy < 1.15 only knows the legacy ``seed`` keyword.
        sampler = qmc.LatinHypercube(d, seed=rng)
    return np.asarray(sampler.random(n), dtype=np.float64)


def make_seed_sequence(
    seed: int | np.random.SeedSequence | None,
) -> np.random.SeedSequence:
    """Normalise a user seed into a ``SeedSequence``.

    Parameters:
        seed:
            A non-negative integer, an existing ``SeedSequence`` (returned
            unchanged), or ``None`` for fresh OS entropy.

    Returns:
        The corresponding ``np.random.SeedSequence``.

    Raises:
        TypeError: If *seed* is of another type.
        ValueError: If *seed* is a negative integer.
    """
    if isinstance(seed, np.random.SeedSequence):
        return seed
    if seed is None:
        return np.random.SeedSequence()
    try:
        value = operator.index(seed)
    except TypeError:
        raise TypeError(
            f"seed must be an int, a numpy SeedSequence or None, got {type(seed).__name__}."
        ) from None
    return np.random.SeedSequence(value)


_KEY_LIMIT = 2**32


def _child_sequence(seq: np.random.SeedSequence, keys: tuple[Any, ...]) -> np.random.SeedSequence:
    """Build the child ``SeedSequence`` addressed by *keys* without mutating *seq*.

    Parameters:
        seq:
            Parent sequence.
        keys:
            Integers in $[0, 2^{32})$ appended to the parent's spawn key.

    Returns:
        A new ``SeedSequence`` sharing the parent's entropy and pool size.

    Raises:
        TypeError: If a key is not an integer.
        ValueError: If a key is outside $[0, 2^{32})$.
    """
    checked = []
    for key in keys:
        try:
            value = operator.index(key)
        except TypeError:
            raise TypeError(f"child stream keys must be integers, got {type(key).__name__}.") from None
        # SeedSequence splits an integer >= 2**32 into 32-bit words, so the key
        # 2**32 would address the same stream as the key pair (0, 1).
        if not 0 <= value < _KEY_LIMIT:
            raise ValueError(f"child stream keys must lie in [0, 2**32), got {value}.")
        checked.append(value)
    return np.random.SeedSequence(
        entropy=seq.entropy,
        spawn_key=tuple(seq.spawn_key) + tuple(checked),
        pool_size=seq.pool_size,
    )


def child_rng(seq: np.random.SeedSequence, *keys: int) -> np.random.Generator:
    """Return a generator addressed by *keys* under *seq*.

    The stream is a pure function of ``(seq.entropy, seq.spawn_key, keys)``,
    so the same keys always give the same stream and different keys give
    statistically independent ones, whatever order the calls happen in.
    Unlike ``seq.spawn``, *seq* is not mutated. Because the keys extend the
    spawn key, ``child_rng(seq, k)`` matches the ``k``-th child of a fresh
    ``seq.spawn``; do not mix both schemes on the same parent.

    Parameters:
        seq:
            Parent seed sequence.
        *keys:
            Integers in $[0, 2^{32})$ naming the stream (e.g. phase, round,
            chunk).

    Returns:
        A ``np.random.Generator`` seeded from the child sequence.

    Raises:
        TypeError: If a key is not an integer.
        ValueError: If a key is outside $[0, 2^{32})$; larger keys would
            alias other key tuples.
    """
    return np.random.default_rng(_child_sequence(seq, keys))


def child_seed(seq: np.random.SeedSequence, *keys: int) -> int:
    """Return a 32-bit integer seed addressed by *keys* under *seq*.

    Derived like [`child_rng`][bsixsa._cube.child_rng], for libraries that
    take a plain integer seed (torch, sbi).

    Parameters:
        seq:
            Parent seed sequence.
        *keys:
            Integers in $[0, 2^{32})$ naming the stream.

    Returns:
        A Python ``int`` in $[0, 2^{32})$.

    Raises:
        TypeError: If a key is not an integer.
        ValueError: If a key is outside $[0, 2^{32})$.
    """
    state = _child_sequence(seq, keys).generate_state(1, dtype=np.uint32)
    return int(state[0])
