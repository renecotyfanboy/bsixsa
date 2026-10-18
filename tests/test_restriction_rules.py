"""Tests for the pure decision rules in `bsixsa.restriction.rules`."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault("xspec", MagicMock())

from scipy.stats import chi2

from bsixsa.restriction.rules import (
    pad_box,
    posterior_span,
    recall_check,
    stop_reason,
    threshold_rule,
)

# ------------------------------------------------------------ posterior_span


@pytest.mark.parametrize("d", [1, 2, 5, 17])
def test_posterior_span_is_the_chi2_quantile(d):
    assert posterior_span(d) == pytest.approx(chi2.ppf(0.999, d))
    assert posterior_span(d, level=0.68) == pytest.approx(chi2.ppf(0.68, d))


def test_posterior_span_grows_with_dimension():
    spans = [posterior_span(d) for d in range(1, 10)]
    assert np.all(np.diff(spans) > 0)


@pytest.mark.parametrize(("d", "level"), [(0, 0.999), (2, 0.0), (2, 1.0)])
def test_posterior_span_refuses_bad_arguments(d, level):
    with pytest.raises(ValueError):
        posterior_span(d, level=level)


# ------------------------------------------------------------ threshold_rule

# best=100, span=10 -> request 130 (target_spans=3); safety = 110 + 3 * rmse.
_PROBE = np.linspace(0.0, 1000.0, 1001)  # quantile q of the probe is 1000 * q


@pytest.mark.parametrize(
    ("rmse", "min_sampleable_mass", "prev", "expected", "wall"),
    [
        # request binds: safety 110 + 3 = 113, cost 10
        (1.0, 0.01, None, 130.0, "request"),
        # safety binds: 110 + 3 * 10 = 140 > 130
        (10.0, 0.01, None, 140.0, "safety"),
        # cost binds: 1000 * 0.2 = 200 > 140
        (10.0, 0.2, None, 200.0, "cost"),
        # the clamp never lets the threshold rise
        (10.0, 0.2, 150.0, 150.0, "cost+clamped"),
        (1.0, 0.01, 120.0, 120.0, "request+clamped"),
        # a previous threshold above the new one is not a clamp
        (1.0, 0.01, 500.0, 130.0, "request"),
    ],
)
def test_threshold_rule_table(rmse, min_sampleable_mass, prev, expected, wall):
    thr, got_wall = threshold_rule(
        best=100.0,
        span=10.0,
        target_spans=3.0,
        rmse_cstat=rmse,
        min_margin_sigma=3.0,
        prev_threshold=prev,
        pred_probe=_PROBE,
        min_sampleable_mass=min_sampleable_mass,
    )
    assert thr == pytest.approx(expected)
    assert got_wall == wall


def test_threshold_rule_reports_the_request_on_a_tie():
    thr, wall = threshold_rule(
        best=100.0,
        span=10.0,
        target_spans=3.0,
        rmse_cstat=10.0,
        min_margin_sigma=2.0,  # safety 110 + 20 == request 130
        prev_threshold=None,
        pred_probe=_PROBE,
        min_sampleable_mass=0.01,
    )
    assert (thr, wall) == (130.0, "request")


@pytest.mark.parametrize(("min_sampleable_mass", "quantile"), [(0.0, 1e-9), (-1.0, 1e-9), (5.0, 1.0)])
def test_threshold_rule_clips_the_min_sampleable_mass(min_sampleable_mass, quantile):
    probe = np.linspace(0.0, 1e6, 10_001)
    thr, wall = threshold_rule(
        best=0.0,
        span=1.0,
        target_spans=1.0,
        rmse_cstat=0.0,
        min_margin_sigma=0.0,
        prev_threshold=None,
        pred_probe=probe,
        min_sampleable_mass=min_sampleable_mass,
    )
    assert thr == pytest.approx(max(1.0, np.quantile(probe, quantile)))
    assert wall == ("cost" if quantile == 1.0 else "request")


def test_threshold_rule_refuses_an_empty_probe():
    with pytest.raises(ValueError, match="probe"):
        threshold_rule(
            best=0.0,
            span=1.0,
            target_spans=3.0,
            rmse_cstat=0.0,
            min_margin_sigma=3.0,
            prev_threshold=None,
            pred_probe=np.array([]),
            min_sampleable_mass=1e-3,
        )


# ------------------------------------------------------------ recall_check


def _validation_set(rng, n=400, best=100.0, span=10.0, noise=0.5):
    cstat = best + rng.uniform(0.0, 20.0 * span, n)
    pred = cstat + rng.normal(0.0, noise, n)
    return pred, cstat


def test_recall_check_does_nothing_when_recall_is_perfect():
    rng = np.random.default_rng(0)
    pred, cstat = _validation_set(rng, noise=0.0)
    thr, triggered, info = recall_check(
        pred_val=pred, cstat_val=cstat, best=100.0, span=10.0, threshold=130.0, rmse_cstat=0.5
    )
    assert not triggered
    assert thr == 130.0
    assert info["skipped"] is False
    assert info["recall_core"] == 1.0
    assert info["recall_threshold"] == 1.0
    assert info["threshold_after"] == 130.0


def test_recall_check_widens_on_a_planted_core_miss():
    rng = np.random.default_rng(1)
    pred, cstat = _validation_set(rng, noise=0.0)
    core = np.flatnonzero(cstat <= 120.0)
    pred[core[0]] = 400.0  # a core point the surrogate predicts far outside

    thr, triggered, info = recall_check(
        pred_val=pred, cstat_val=cstat, best=100.0, span=10.0, threshold=130.0, rmse_cstat=2.0
    )

    assert triggered
    assert info["recall_core"] < 1.0
    assert thr == pytest.approx(400.0 + 3.0 * 2.0)
    assert thr > 130.0
    assert info["threshold_before"] == 130.0
    assert info["threshold_after"] == thr


def test_recall_check_widens_on_a_low_recall_below_threshold():
    rng = np.random.default_rng(2)
    cstat = 100.0 + rng.uniform(0.0, 200.0, 2000)
    pred = cstat.copy()
    # Core fully recalled, but 30 % of the band (120, 130] predicted just above 130.
    band = np.flatnonzero((cstat > 120.0) & (cstat <= 130.0))
    miss = band[: max(1, int(0.3 * band.size))]
    pred[miss] = 135.0

    thr, triggered, info = recall_check(
        pred_val=pred, cstat_val=cstat, best=100.0, span=10.0, threshold=130.0, rmse_cstat=1.0
    )

    assert info["recall_core"] == 1.0
    assert info["recall_threshold"] < 0.99
    assert triggered
    expected = max(
        130.0, pred[cstat <= 120.0].max() + 3.0, np.quantile(pred[cstat <= 130.0], 0.99, method="higher")
    )
    assert thr == pytest.approx(expected)
    assert thr >= 135.0


@pytest.mark.parametrize("seed", range(20))
def test_recall_check_never_lowers_the_threshold(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 300))
    cstat = 100.0 + rng.exponential(30.0, n)
    pred = cstat + rng.normal(0.0, rng.uniform(0.0, 40.0), n)
    threshold = float(100.0 + rng.uniform(0.0, 80.0))

    thr, triggered, _ = recall_check(
        pred_val=pred, cstat_val=cstat, best=100.0, span=10.0, threshold=threshold, rmse_cstat=0.0
    )

    assert thr >= threshold
    if not triggered:
        assert thr == threshold


def test_recall_check_is_skipped_with_fewer_than_five_core_points():
    cstat = np.array([101.0, 105.0, 110.0, 115.0, 500.0, 600.0])  # 4 points <= 120
    pred = np.array([900.0, 900.0, 900.0, 900.0, 500.0, 600.0])  # all core points missed

    thr, triggered, info = recall_check(
        pred_val=pred, cstat_val=cstat, best=100.0, span=10.0, threshold=130.0, rmse_cstat=1.0
    )

    assert (thr, triggered) == (130.0, False)
    assert info["skipped"] is True
    assert (info["n_core"], info["n_threshold"]) == (4, 4)

    thr, triggered, info = recall_check(
        pred_val=pred, cstat_val=cstat, best=100.0, span=10.0, threshold=130.0, rmse_cstat=1.0, min_core=4
    )
    assert triggered and thr > 130.0


def test_recall_check_core_stops_three_rmse_below_a_low_request():
    # target_spans = 1: threshold = best + span + 3 * rmse = 113. A perfect
    # surrogate predicts the points in (113, 120] above it, rightly: the core
    # C <= best + 2 * span would fail them, forcing ~2 spans whatever was asked.
    rng = np.random.default_rng(5)
    cstat = 100.0 + rng.uniform(0.0, 40.0, 2000)
    pred = cstat.copy()

    thr, triggered, info = recall_check(
        pred_val=pred, cstat_val=cstat, best=100.0, span=10.0, threshold=113.0, rmse_cstat=1.0
    )

    assert (thr, triggered) == (113.0, False)
    assert info["core_top"] == pytest.approx(110.0)
    assert info["recall_core"] == 1.0 and info["recall_threshold"] == 1.0


def test_recall_check_core_always_holds_the_posterior_span():
    # A threshold below best + span + 3 * rmse (e.g. clamped): the core never
    # shrinks below best + span, so a posterior point cut by it still widens.
    cstat = np.array([101.0, 102.0, 103.0, 104.0, 105.0, 109.0, 150.0])
    pred = np.array([101.0, 102.0, 103.0, 104.0, 105.0, 111.0, 150.0])

    thr, triggered, info = recall_check(
        pred_val=pred, cstat_val=cstat, best=100.0, span=10.0, threshold=110.0, rmse_cstat=2.0
    )

    assert triggered and thr == pytest.approx(111.0 + 6.0)
    assert info["core_top"] == pytest.approx(110.0) and info["recall_core"] < 1.0


def test_recall_check_threshold_test_runs_without_enough_core_points():
    # 3 core points (C <= 120), but 20 points below the threshold 130 with 2
    # of them predicted above it: the 99 % test must still run and widen.
    cstat = np.concatenate([[101.0, 110.0, 119.0], np.linspace(121.0, 129.0, 17), [300.0, 400.0]])
    pred = cstat.copy()
    pred[[5, 9]] = [131.0, 133.0]

    thr, triggered, info = recall_check(
        pred_val=pred, cstat_val=cstat, best=100.0, span=10.0, threshold=130.0, rmse_cstat=0.1
    )

    assert triggered and thr == pytest.approx(133.0)
    assert info["skipped"] is False
    assert info["n_core"] == 3 and np.isnan(info["recall_core"])
    assert info["n_threshold"] == 20 and info["recall_threshold"] == pytest.approx(0.9)


@pytest.mark.parametrize("n", [7, 20, 60, 99, 100, 101, 250, 1000])
def test_recall_check_restores_99_percent_recall_for_any_count(n):
    # n points below the threshold, the highest 2 % (at least 1) predicted
    # above it: the widened threshold must recall at least 99 % of them.
    rng = np.random.default_rng(n)
    cstat = np.sort(125.0 + rng.uniform(0.0, 5.0, n))  # all above the core
    pred = cstat.copy()
    k = max(1, int(np.ceil(0.02 * n)))
    pred[-k:] = 130.0 + np.linspace(0.5, 3.0, k)
    core = np.array([101.0, 102.0, 103.0, 104.0, 105.0])
    pred_all, cstat_all = np.concatenate([core, pred]), np.concatenate([core, cstat])

    thr, triggered, info = recall_check(
        pred_val=pred_all, cstat_val=cstat_all, best=100.0, span=10.0, threshold=130.0, rmse_cstat=0.1
    )

    assert triggered
    below = cstat_all <= 130.0
    assert np.mean(pred_all[below] <= thr) >= 0.99


def test_recall_check_refuses_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        recall_check(
            pred_val=np.zeros(3), cstat_val=np.zeros(4), best=0.0, span=1.0, threshold=1.0, rmse_cstat=0.0
        )


# ------------------------------------------------------------ stop_reason


def _stop(**overrides):
    kwargs = dict(
        wave=2,
        n_waves=5,
        prior_mass=0.1,
        stop_when_acceptance=5e-3,
        threshold=100.0,
        prev_threshold=200.0,
        stop_when_threshold=0.01,
        recall_triggered=False,
    )
    kwargs.update(overrides)
    return stop_reason(**kwargs)


def test_stop_reason_continues_while_the_region_shrinks():
    assert _stop() is None


def test_stop_reason_acceptance_comes_first():
    assert _stop(prior_mass=1e-3, threshold=200.0, wave=5) == "acceptance"


def test_stop_reason_threshold_before_n_waves():
    assert _stop(threshold=199.5, wave=5) == "threshold"
    # a clamped wave has zero gain
    assert _stop(threshold=200.0) == "threshold"


def test_stop_reason_threshold_needs_a_previous_wave():
    assert _stop(prev_threshold=None, wave=1) is None


def test_stop_reason_recall_wave_suppresses_the_threshold_stop():
    assert _stop(threshold=250.0, recall_triggered=True) is None
    assert _stop(threshold=250.0, recall_triggered=True, wave=5) == "n_waves"
    # but not the acceptance stop
    assert _stop(threshold=250.0, recall_triggered=True, prior_mass=0.0) == "acceptance"


def test_stop_reason_n_waves():
    assert _stop(wave=5) == "n_waves"
    assert _stop(wave=4) is None


# ------------------------------------------------------------ pad_box


def test_pad_box_uses_the_fractional_padding_when_the_surrogate_is_accurate():
    lo, hi = pad_box([0.4, 0.2], [0.6, 0.3], box_padding=0.1, rmse_cstat=0.0, delta=10.0)
    np.testing.assert_allclose(lo, [0.38, 0.19])
    np.testing.assert_allclose(hi, [0.62, 0.31])


def test_pad_box_floor_from_the_surrogate_error():
    # 3 * rmse / (2 * delta) = 3 * 4 / 20 = 0.6 -> pad = 0.6 * w / 2 = 0.3 w > 0.1 w
    lo, hi = pad_box([0.4], [0.6], box_padding=0.1, rmse_cstat=4.0, delta=10.0)
    np.testing.assert_allclose(lo, [0.34])
    np.testing.assert_allclose(hi, [0.66])


def test_pad_box_clips_to_the_cube():
    lo, hi = pad_box([0.01, 0.2], [0.5, 0.99], box_padding=0.5, rmse_cstat=0.0, delta=1.0)
    np.testing.assert_allclose(lo, [0.0, 0.0])
    np.testing.assert_allclose(hi, [0.745, 1.0])
    assert lo.dtype == np.float64 and hi.dtype == np.float64


def test_pad_box_survives_a_zero_delta():
    lo, hi = pad_box([0.4], [0.6], box_padding=0.1, rmse_cstat=1.0, delta=0.0)
    np.testing.assert_array_equal(lo, [0.0])
    np.testing.assert_array_equal(hi, [1.0])


def test_pad_box_refuses_inverted_or_mismatched_corners():
    with pytest.raises(ValueError):
        pad_box([0.6], [0.4], box_padding=0.1, rmse_cstat=0.0, delta=1.0)
    with pytest.raises(ValueError):
        pad_box([0.1, 0.2], [0.4], box_padding=0.1, rmse_cstat=0.0, delta=1.0)
