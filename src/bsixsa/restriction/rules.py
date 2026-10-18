r"""Decision rules of the history-matching waves: threshold, recall check and stop.

Notation: ``best`` is the smallest exact C-stat folded so far, ``span`` the
posterior span $\chi^2_d(0.999)$ and ``rmse_cstat`` ($\sigma_C$) the
surrogate's held-out C-stat RMSE near the region.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import chi2

__all__ = [
    "posterior_span",
    "recall_check",
    "stop_reason",
    "threshold_rule",
]


def posterior_span(d: int, level: float = 0.999) -> float:
    r"""Width of the posterior in C-stat units, $\chi^2_d(\text{level})$.

    For a near-Gaussian posterior, $C - C_{min}$ follows $\chi^2_d$, so this
    C-stat excess holds *level* of the posterior mass.

    Parameters:
        d: Number of free parameters.
        level: Posterior mass enclosed, in $(0, 1)$.

    Returns:
        The quantile $\chi^2_d(\text{level})$.

    Raises:
        ValueError: If *d* < 1 or *level* is outside $(0, 1)$.
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
    min_sampleable_mass: float,
) -> tuple[float, str]:
    r"""Place the C-stat threshold of a wave.

    The threshold is the largest of the request ``best + target_spans * span``,
    the safety floor ``best + span + min_margin_sigma * rmse_cstat`` and the
    cost floor (the ``min_sampleable_mass`` quantile of the surrogate on a
    uniform probe, below which the region could not be filled). It is then
    clamped to ``prev_threshold``, so the region never grows here.

    Parameters:
        best: Smallest exact C-stat folded so far.
        span: Posterior span in C-stat units.
        target_spans: Requested width in spans.
        rmse_cstat: The surrogate's C-stat error.
        min_margin_sigma: Safety margin, in RMSEs.
        prev_threshold: Previous wave's threshold, ``None`` on the first wave.
        pred_probe: Surrogate predictions on uniform cube points, shape ``(n,)``.
        min_sampleable_mass: Smallest prior mass the sampler can fill.

    Returns:
        ``(threshold, wall)``, where *wall* is ``"request"``, ``"safety"`` or
        ``"cost"``, with ``"+clamped"`` appended when the clamp applied.

    Raises:
        ValueError: If *pred_probe* is empty.
    """
    pred = np.asarray(pred_probe, dtype=np.float64).reshape(-1)
    if pred.size == 0:
        raise ValueError("threshold_rule needs at least one probe prediction.")
    requested = float(best) + float(target_spans) * float(span)
    safety_floor = float(best) + float(span) + float(min_margin_sigma) * float(rmse_cstat)
    cost_floor = float(np.quantile(pred, float(np.clip(min_sampleable_mass, 1e-9, 1.0))))

    threshold = max(requested, safety_floor, cost_floor)
    if requested >= threshold:
        wall = "request"
    elif safety_floor >= threshold:
        wall = "safety"
    else:
        wall = "cost"

    if prev_threshold is not None and threshold > float(prev_threshold):
        threshold = float(prev_threshold)
        wall += "+clamped"
    return threshold, wall


def recall_check(
    *,
    pred_val: np.ndarray,
    cstat_val: np.ndarray,
    best: float,
    span: float,
    threshold: float,
    rmse_cstat: float,
    min_core: int = 5,
) -> tuple[float, bool, dict]:
    r"""Raise the threshold if the surrogate would wrongly discard held-out folds.

    A region that cuts posterior mass cannot be detected afterwards, since its
    points are never proposed. Two tests run on folds the surrogate never
    trained on:

    - every point of the core $C \le \min(C_{best} + 2\Delta,\,
      \max(C_{best} + \Delta,\, t - 3\sigma_C))$ must be predicted below $t$;
    - at least 99 % of the points with $C \le t$ must be.

    A test runs only with at least *min_core* points. On failure the threshold
    becomes ``max(t, max(pred_core) + 3 * rmse_cstat, q_0.99(pred[C <= t]))``.
    The check can only widen the region; it lowers the risk of a cut
    posterior without ruling it out.

    Parameters:
        pred_val: Surrogate predictions on the validation points, shape ``(n,)``.
        cstat_val: Exact C-stat of the same points, shape ``(n,)``.
        best: Smallest exact C-stat folded so far.
        span: Posterior span in C-stat units.
        threshold: Threshold from
            [`threshold_rule`][bsixsa.restriction.rules.threshold_rule].
        rmse_cstat: The surrogate's C-stat error.
        min_core: Smallest number of points a test needs to run.

    Returns:
        ``(threshold, triggered, info)``; *info* holds the core bound, point
        counts and, when the check ran, the recalls and thresholds.

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
    # Capped so that a request below 2 spans can still pass.
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
    r"""Decide whether the waves stop after this one.

    The first test that holds names the reason:

    1. ``"acceptance"``: the region's prior mass is below *stop_when_acceptance*;
    2. ``"threshold"``: the threshold fell by less than *stop_when_threshold*,
       relative ($1 - t_w / t_{w-1}$); not tested on the first wave nor after
       a recall widening;
    3. ``"n_waves"``: the wave budget is spent.

    Parameters:
        wave: 1-based index of the wave just finished.
        n_waves: Maximum number of waves.
        prior_mass: Prior mass of the region after this wave.
        stop_when_acceptance: Mass below which the waves stop.
        threshold: Threshold of this wave.
        prev_threshold: Previous wave's threshold, ``None`` on the first wave.
        stop_when_threshold: Relative threshold gain below which the waves stop.
        recall_triggered: Whether the recall check widened this wave's threshold.

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

