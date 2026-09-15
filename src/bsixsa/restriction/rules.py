r"""Pure decision rules of the history-matching waves.

Every rule that decides where a region's boundary goes, when the waves stop,
and how far a bounding box is padded lives here as a pure numpy function, so
that it can be tested in isolation from the regressor that feeds it.

Notation: $C$ is the C-stat, ``best`` the smallest exact C-stat folded so far,
``span`` the posterior width $\chi^2_d(0.999)$ in C-stat units, and
``rmse_cstat`` the regressor's C-stat error near the region boundary.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import chi2

__all__ = [
    "pad_box",
    "posterior_span",
    "recall_override",
    "stop_reason",
    "threshold_rule",
]


def posterior_span(d: int, level: float = 0.999) -> float:
    r"""Width of the posterior in C-stat units, $\chi^2_d(\text{level})$.

    By Wilks' theorem $C - C_{min}$ is asymptotically $\chi^2_d$ distributed
    under the posterior, so this is the C-stat excess that contains *level* of
    the posterior mass.

    Parameters:
        d:
            Number of free parameters.
        level:
            Posterior mass enclosed, in $(0, 1)$.

    Returns:
        The quantile $\chi^2_d(\text{level})$.

    Raises:
        ValueError: If *d* is smaller than 1 or *level* is outside $(0, 1)$.
    """
    if int(d) < 1:
        raise ValueError(f"posterior_span expects d >= 1, got {d}.")
    if not 0.0 < float(level) < 1.0:
        raise ValueError(f"posterior_span expects a level in (0, 1), got {level}.")
    return float(chi2.ppf(float(level), int(d)))


def threshold_rule(
    *,
    best: float,
    span: float,
    target_spans: float,
    rmse_cstat: float,
    min_margin_sigma: float,
    prev_threshold: float | None,
    pred_probe: np.ndarray,
    mass_floor: float,
) -> tuple[float, str]:
    r"""Place the C-stat threshold of a wave.

    The threshold is the largest of three terms:

    - the request, ``best + target_spans * span``;
    - the safety floor, ``best + span + min_margin_sigma * rmse_cstat``: the
      region must clear the posterior's own extent by the regressor's error;
    - the cost floor, the ``mass_floor`` quantile of the regressor's
      predictions on a uniform probe: a smaller region could not be filled
      within the candidate budget.

    The threshold is then clamped to ``prev_threshold``: the cost floor is a
    quantile of the surrogate, so it moves with every refit, in either
    direction, and ground already taken is never given back here (only the
    recall check may widen a region).

    Parameters:
        best:
            Smallest exact C-stat folded so far.
        span:
            Posterior span in C-stat units.
        target_spans:
            Requested width in spans.
        rmse_cstat:
            Regressor C-stat RMSE near the boundary.
        min_margin_sigma:
            Safety margin, in RMSEs.
        prev_threshold:
            Threshold of the previous wave, or ``None`` on the first wave.
        pred_probe:
            Regressor predictions on uniform cube points, shape ``(n,)``.
        mass_floor:
            Smallest prior mass the sampler can fill, clipped to
            $[10^{-9}, 1]$ before taking the quantile.

    Returns:
        ``(threshold, wall)`` where *wall* names the binding term,
        ``"request"``, ``"safety"`` or ``"cost"``, with ``"+clamped"``
        appended when the clamp applied.

    Raises:
        ValueError: If *pred_probe* is empty.
    """
    pred = np.asarray(pred_probe, dtype=np.float64).reshape(-1)
    if pred.size == 0:
        raise ValueError("threshold_rule needs at least one probe prediction.")
    want = float(best) + float(target_spans) * float(span)
    floor_safe = float(best) + float(span) + float(min_margin_sigma) * float(rmse_cstat)
    floor_cost = float(np.quantile(pred, float(np.clip(mass_floor, 1e-9, 1.0))))

    threshold = max(want, floor_safe, floor_cost)
    if want >= threshold:
        wall = "request"
    elif floor_safe >= threshold:
        wall = "safety"
    else:
        wall = "cost"

    if prev_threshold is not None and threshold > float(prev_threshold):
        threshold = float(prev_threshold)
        wall += "+clamped"
    return threshold, wall


def recall_override(
    *,
    pred_val: np.ndarray,
    cstat_val: np.ndarray,
    best: float,
    span: float,
    threshold: float,
    rmse_cstat: float,
    min_core: int = 5,
) -> tuple[float, bool, dict]:
    r"""Widen a threshold that would exclude validation points it should keep.

    A region that cuts posterior mass is invisible afterwards: its draws are
    never proposed, so no importance weight can reveal them. This check reads
    held-out folds whose exact C-stat is known:

    - the core must be fully recalled (every core point predicted at or below
      the threshold). The core is $C \le \min(C_{best} + 2\Delta,\,
      \max(C_{best} + \Delta,\, t - 3\sigma_C))$: it stops $3\sigma_C$ below
      the threshold $t$, because a point closer to the boundary may be
      predicted across it by a surrogate that is right to within its error
      (the second test covers those), but it always holds the posterior span
      $\Delta$ itself;
    - the points with $C \le$ *threshold* must be recalled at 99 % or more.

    Each test runs only when at least *min_core* points fall in its set; the
    check is skipped when neither does. On failure the threshold becomes
    ``max(threshold, max(pred_core) + 3 * rmse_cstat, q_0.99(pred[C <= threshold]))``,
    where $q_{0.99}$ is the $\lceil 0.99\,(n - 1) \rceil$-th order statistic
    (numpy's ``method="higher"``), so that at least 99 % of those points are
    recalled afterwards whatever their number. The rule can only widen the
    region, never tighten it. It lowers the risk of a cut posterior; it does
    not rule it out.

    Parameters:
        pred_val:
            Regressor predictions on the validation points, shape ``(n,)``.
        cstat_val:
            Exact C-stat of the same points, shape ``(n,)``.
        best:
            Smallest exact C-stat folded so far.
        span:
            Posterior span in C-stat units.
        threshold:
            Threshold proposed by
            [`threshold_rule`][bsixsa.restriction.rules.threshold_rule].
        rmse_cstat:
            Regressor C-stat RMSE near the boundary.
        min_core:
            Smallest number of points a test needs to run.

    Returns:
        ``(threshold, triggered, info)``. *threshold* is never below the input.
        *info* holds ``skipped``, the core's upper C-stat ``core_top`` and the
        point counts ``n_core`` and ``n_threshold``; when the check ran, also
        the recalls (``NaN`` for a test that did not run), ``triggered`` and
        the thresholds before and after.

    Raises:
        ValueError: If *pred_val* and *cstat_val* have different lengths.
    """
    pred = np.asarray(pred_val, dtype=np.float64).reshape(-1)
    cstat = np.asarray(cstat_val, dtype=np.float64).reshape(-1)
    if pred.shape != cstat.shape:
        raise ValueError(
            f"pred_val and cstat_val must have the same length, got {pred.size} and {cstat.size}."
        )
    threshold = float(threshold)
    best, span, rmse = float(best), float(span), float(rmse_cstat)
    # Uncapped, a request below 2 spans could never pass: core points above
    # the threshold are rightly predicted above it.
    core_top = min(best + 2.0 * span, max(best + span, threshold - 3.0 * rmse))
    core = cstat <= core_top
    in_thr = cstat <= threshold
    n_core, n_thr = int(core.sum()), int(in_thr.sum())
    # Gated separately: a sparse core must not silence the 99 % test.
    run_core, run_thr = n_core >= int(min_core), n_thr >= int(min_core)
    counts = {"core_top": core_top, "n_core": n_core, "n_threshold": n_thr}
    if not (run_core or run_thr):
        return threshold, False, {"skipped": True, **counts}

    recall_core = float(np.mean(pred[core] <= threshold)) if run_core else math.nan
    recall_thr = float(np.mean(pred[in_thr] <= threshold)) if run_thr else math.nan
    triggered = (run_core and recall_core < 1.0) or (run_thr and recall_thr < 0.99)

    new = threshold
    if triggered:
        candidates = [threshold]
        if n_core:
            candidates.append(float(pred[core].max()) + 3.0 * rmse)
        if n_thr:
            # Interpolated quantiles fall short of 99 % recall below 100 points.
            candidates.append(float(np.quantile(pred[in_thr], 0.99, method="higher")))
        new = max(candidates)
    info = {
        "skipped": False,
        **counts,
        "recall_core": recall_core,
        "recall_threshold": recall_thr,
        "triggered": bool(triggered),
        "threshold_before": threshold,
        "threshold_after": new,
    }
    return new, bool(triggered), info


def stop_reason(
    *,
    wave: int,
    n_waves: int,
    prior_mass: float,
    stop_when_acceptance: float,
    threshold: float,
    prev_threshold: float | None,
    stop_when_threshold: float,
    recall_triggered: bool,
) -> str | None:
    """Decide whether the waves stop after this one.

    The tests run in order and the first that holds names the reason:

    1. ``"acceptance"``: the region's prior mass is below
       *stop_when_acceptance*, so it is too small to keep sampling;
    2. ``"threshold"``: the threshold fell by less than
       *stop_when_threshold* relative to the previous wave (not tested on the
       first wave, nor on a wave widened by the recall check, whose threshold
       says nothing about convergence). As in SIXSA v2 the gain is relative
       to the absolute C-stat, $1 - t_w / t_{w-1}$, so it depends on the
       C-stat's offset: a spectrum whose best C-stat is large against the
       posterior span (many bins) stops earlier, with a wider region;
    3. ``"n_waves"``: the wave budget is spent.

    Parameters:
        wave:
            1-based index of the wave just finished.
        n_waves:
            Maximum number of waves.
        prior_mass:
            Prior mass of the region after this wave.
        stop_when_acceptance:
            Mass below which the waves stop.
        threshold:
            Threshold of this wave.
        prev_threshold:
            Threshold of the previous wave, ``None`` on the first wave.
        stop_when_threshold:
            Relative threshold gain below which the waves stop.
        recall_triggered:
            Whether the recall check widened this wave's threshold.

    Returns:
        The stop reason, or ``None`` to run another wave.
    """
    if float(prior_mass) < float(stop_when_acceptance):
        return "acceptance"
    if prev_threshold is not None and not recall_triggered:
        gain = 1.0 - float(threshold) / max(float(prev_threshold), 1e-12)
        if gain < float(stop_when_threshold):
            return "threshold"
    if int(wave) >= int(n_waves):
        return "n_waves"
    return None


def pad_box(
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    box_padding: float,
    rmse_cstat: float,
    delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Pad a unit-cube bounding box and clip it to $[0, 1]^d$.

    Each side of an axis of width $w$ grows by
    $\max(\text{box\_padding}\cdot w,\; \tfrac{3\,\sigma_C}{2\Delta}\cdot\tfrac{w}{2})$.
    The second term is the boundary displacement a $3\sigma_C$ surrogate
    error causes on a quadratic bowl: with $C - C_{best} = \Delta (r/R)^2$
    the boundary radius $R \approx w/2$ moves by
    $\delta R = 3\sigma_C R / (2\Delta)$.

    Parameters:
        lo:
            Lower corner, shape ``(d,)``.
        hi:
            Upper corner, shape ``(d,)``, with ``hi >= lo``.
        box_padding:
            Minimum padding as a fraction of the width.
        rmse_cstat:
            Regressor C-stat RMSE near the boundary.
        delta:
            Threshold minus best C-stat; floored at $10^{-12}$.

    Returns:
        ``(lo, hi)`` padded and clipped to $[0, 1]$, as float64 arrays.

    Raises:
        ValueError: If the corners have different shapes or ``hi < lo``.
    """
    lo = np.asarray(lo, dtype=np.float64).reshape(-1)
    hi = np.asarray(hi, dtype=np.float64).reshape(-1)
    if lo.shape != hi.shape:
        raise ValueError(f"lo and hi must have the same shape, got {lo.shape} and {hi.shape}.")
    if np.any(hi < lo):
        raise ValueError("pad_box expects hi >= lo on every axis.")
    width = hi - lo
    surrogate = 3.0 * float(rmse_cstat) / (2.0 * max(float(delta), 1e-12))
    pad = np.maximum(float(box_padding) * width, surrogate * width / 2.0)
    return np.clip(lo - pad, 0.0, 1.0), np.clip(hi + pad, 0.0, 1.0)
