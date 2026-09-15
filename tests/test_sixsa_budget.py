"""Tests for `bsixsa.backend.sixsa.budget`."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("xspec", MagicMock())

from bsixsa.backend.sixsa.budget import final_draws, next_training_size, training_set_size

# ---------------------------------------------------------------------------
# training_set_size
# ---------------------------------------------------------------------------

_V2_TABLE = {
    2500: [2000, 3750, 12000, 27500, 45500],
    1000: [1500, 3000, 9000, 22500, 38500],
    5000: [2500, 4500, 15000, 32500, 56000],
}


@pytest.mark.parametrize("n_bins", sorted(_V2_TABLE))
@pytest.mark.parametrize("column, d", list(enumerate([2, 3, 6, 10, 14])))
def test_training_set_size_matches_the_v2_table(n_bins, column, d):
    assert training_set_size(d, n_bins) == _V2_TABLE[n_bins][column]


def test_training_set_size_short_spectrum_hits_the_floors():
    # The binning factor clips at 0.5 and the per-parameter count at 750.
    assert training_set_size(2, 64) == 1500


def test_training_set_size_is_a_python_int_multiple_of_d():
    n = training_set_size(7, 3000)
    assert type(n) is int
    assert n % 7 == 0


def test_training_set_size_clips_the_binning_factor_at_three():
    assert training_set_size(6, 10**9) == training_set_size(6, 2500 * 81)
    assert training_set_size(6, 2500 * 81) == 36000


def test_training_set_size_skips_the_binning_factor_without_bins():
    assert training_set_size(6, 0) == training_set_size(6, 2500) == 12000
    assert training_set_size(6, 10, bins_exponent=0.0) == 12000


def test_training_set_size_honours_the_keyword_overrides():
    assert training_set_size(6, 2500, base=1000.0) == 6000
    assert training_set_size(2, 64, per_param_min=100) == 1000
    assert training_set_size(6, 2500, round_to=1) == 12000


@pytest.mark.parametrize("d", [0, -1])
def test_training_set_size_rejects_a_bad_dimension(d):
    with pytest.raises(ValueError, match="d must be"):
        training_set_size(d, 100)


def test_training_set_size_rejects_non_integers():
    with pytest.raises(TypeError):
        training_set_size(2.5, 100)
    with pytest.raises(ValueError, match="n_bins"):
        training_set_size(2, -1)


# ---------------------------------------------------------------------------
# next_training_size
# ---------------------------------------------------------------------------


def test_next_training_size_ramps_and_caps():
    assert next_training_size(1500, ramp=1.25, n_max=50_000) == 1875
    assert next_training_size(45_000, ramp=1.25, n_max=50_000) == 50_000
    assert next_training_size(1500, ramp=1.0, n_max=None) == 1500
    assert next_training_size(10**6, ramp=2.0, n_max=None) == 2 * 10**6


def test_next_training_size_rounds_to_nearest():
    assert next_training_size(3, ramp=1.25, n_max=None) == 4  # 3.75
    assert next_training_size(2, ramp=1.2, n_max=None) == 2  # 2.4


@pytest.mark.parametrize("ramp", [0.99, 0.0, float("nan"), float("inf")])
def test_next_training_size_rejects_a_shrinking_ramp(ramp):
    with pytest.raises(ValueError, match="ramp"):
        next_training_size(1000, ramp=ramp, n_max=None)


def test_next_training_size_rejects_bad_sizes():
    with pytest.raises(ValueError, match="n must be"):
        next_training_size(0, ramp=1.25, n_max=None)
    with pytest.raises(ValueError, match="n_max"):
        next_training_size(10, ramp=1.25, n_max=0)


# ---------------------------------------------------------------------------
# final_draws
# ---------------------------------------------------------------------------


def test_final_draws_scales_with_the_fresh_efficiency():
    # 20 % efficiency: 1.25 * 2000 / 0.2 = 12500, rounded up to 15000.
    assert final_draws(target_ess=2000, ess_fresh=200, n_fresh=1000, max_draws=200_000) == 15_000


def test_final_draws_exact_multiple_is_kept():
    # 25 % efficiency: 1.25 * 2000 / 0.25 = 10000 exactly.
    assert final_draws(target_ess=2000, ess_fresh=250, n_fresh=1000, max_draws=200_000) == 10_000


def test_final_draws_applies_the_floor():
    assert final_draws(target_ess=100, ess_fresh=900, n_fresh=1000, max_draws=200_000) == 5000
    assert final_draws(target_ess=100, ess_fresh=900, n_fresh=1000, max_draws=200_000, min_draws=0) == 5000
    assert (
        final_draws(target_ess=100, ess_fresh=900, n_fresh=1000, max_draws=200_000, min_draws=0, chunk=1) == 139
    )


def test_final_draws_caps_at_max_draws():
    assert final_draws(target_ess=2000, ess_fresh=1, n_fresh=1000, max_draws=200_000) == 200_000


def test_final_draws_never_rounds_above_a_non_multiple_cap():
    # 12500 rounds up to 15000, above the cap of 13000.
    assert final_draws(target_ess=2000, ess_fresh=200, n_fresh=1000, max_draws=13_000) == 13_000
    # The cap wins over the floor.
    assert final_draws(target_ess=10, ess_fresh=900, n_fresh=1000, max_draws=3000) == 3000


@pytest.mark.parametrize("ess", [0.0, -3.0, float("nan")])
def test_final_draws_degenerate_efficiency_requests_the_cap(ess):
    assert final_draws(target_ess=2000, ess_fresh=ess, n_fresh=1000, max_draws=30_000) == 30_000


def test_final_draws_zero_fresh_draws_does_not_divide_by_zero():
    assert final_draws(target_ess=2000, ess_fresh=0.0, n_fresh=0, max_draws=40_000) == 40_000


def test_final_draws_returns_a_python_int():
    assert type(final_draws(target_ess=2000, ess_fresh=200, n_fresh=1000, max_draws=200_000)) is int


def test_final_draws_rejects_bad_arguments():
    with pytest.raises(ValueError, match="target_ess"):
        final_draws(target_ess=0, ess_fresh=1, n_fresh=10, max_draws=10)
    with pytest.raises(ValueError, match="max_draws"):
        final_draws(target_ess=1, ess_fresh=1, n_fresh=10, max_draws=0)
    with pytest.raises(ValueError, match="chunk"):
        final_draws(target_ess=1, ess_fresh=1, n_fresh=10, max_draws=10, chunk=0)
