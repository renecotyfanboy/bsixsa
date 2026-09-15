"""Tests for the SIXSA persistent replace-worst training buffer (`_buffer.py`)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault("xspec", MagicMock())

from bsixsa.backend.sixsa._buffer import TrainingBuffer


def _buffer(cstat, capacity=None):
    cstat = np.asarray(cstat, dtype=float)
    n = len(cstat)
    # Encode the identity of each row in theta/x so we can track which rows survive.
    theta = np.stack([cstat, np.arange(n)], axis=1)
    x = np.tile(cstat.reshape(-1, 1), (1, 4))
    return TrainingBuffer(theta, x, cstat, capacity=capacity)


def test_buffer_alignment_and_ceiling():
    buf = _buffer([3.0, 1.0, 2.0])
    assert len(buf) == 3
    assert buf.ceiling == 3.0
    assert buf.d == 2
    theta, x = buf.training_arrays()
    assert theta.shape == (3, 2)
    assert x.shape == (3, 4)


def test_replace_worst_keeps_best_of_union():
    """Keep-best-of-union: only the lowest-cstat rows survive; count is exact."""
    buf = _buffer([0.0, 1.0, 2.0, 3.0, 4.0], capacity=5)

    # New rows: one better than the ceiling, two worse than everything kept.
    n_new = buf.replace_worst(
        new_theta=np.array([[0.5, 100], [5.0, 101], [6.0, 102]], dtype=float),
        new_x=np.zeros((3, 4)),
        new_cstat=np.array([0.5, 5.0, 6.0]),
    )
    # Union lowest-5 = {0, 0.5, 1, 2, 3}; exactly one new row (0.5) made it in.
    assert n_new == 1
    assert len(buf) == 5
    np.testing.assert_allclose(np.sort(buf.cstat), [0.0, 0.5, 1.0, 2.0, 3.0])
    # Ceiling dropped from 4 -> 3 (monotone).
    assert buf.ceiling == 3.0


def test_replace_worst_all_worse_is_noop_and_monotone():
    buf = _buffer([0.0, 1.0, 2.0], capacity=3)
    before = buf.cstat.copy()
    n_new = buf.replace_worst(
        new_theta=np.array([[9.0, 0], [8.0, 1]], dtype=float),
        new_x=np.zeros((2, 4)),
        new_cstat=np.array([9.0, 8.0]),
    )
    assert n_new == 0
    np.testing.assert_allclose(np.sort(buf.cstat), np.sort(before))
    assert buf.ceiling == 2.0


def test_replace_worst_is_monotone_over_many_moves():
    rng = np.random.default_rng(0)
    buf = _buffer(rng.uniform(50, 100, size=200), capacity=200)
    ceilings = [buf.ceiling]
    for _ in range(20):
        k = 40
        new_c = rng.uniform(30, 110, size=k)  # some better, some worse
        buf.replace_worst(
            new_theta=np.stack([new_c, rng.normal(size=k)], axis=1),
            new_x=np.zeros((k, 4)),
            new_cstat=new_c,
        )
        assert len(buf) == 200
        ceilings.append(buf.ceiling)
    # Ceiling must never rise.
    assert all(b <= a + 1e-9 for a, b in zip(ceilings, ceilings[1:]))


def test_row_alignment_preserved_after_move():
    """theta/x rows must track their cstat through the sort."""
    buf = _buffer([5.0, 1.0, 3.0], capacity=3)
    buf.replace_worst(
        new_theta=np.array([[0.0, 99]], dtype=float),
        new_x=np.full((1, 4), 0.0),
        new_cstat=np.array([0.0]),
    )
    # theta[:,0] was set equal to cstat at construction; the new row carries 0.0.
    np.testing.assert_allclose(np.sort(buf.theta[:, 0]), np.sort(buf.cstat))


def test_length_and_shape_validation():
    with pytest.raises(ValueError, match="same number of rows"):
        TrainingBuffer(np.zeros((3, 2)), np.zeros((2, 4)), np.zeros(3))
    with pytest.raises(ValueError, match="2D"):
        TrainingBuffer(np.zeros(3), np.zeros((3, 4)), np.zeros(3))
