"""The history-matching loop on the unit cube."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np

from .._cube import latin_hypercube
from .._torch_utils import torch_threads
from .config import HistoryMatching
from .region import Region, Restriction
from .rules import posterior_span, recall_check, stop_reason, threshold_rule

__all__ = ["run_history_matching"]


def _no_log(message: str) -> None:
    """Default logger: discard the message."""


def _hold_out(cstat: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Hold out ``fraction`` of a wave's finite folds (at least 1, never all) for validation."""
    finite = np.flatnonzero(np.isfinite(cstat))
    held = np.zeros(cstat.size, dtype=bool)
    if finite.size >= 2:
        k = min(max(1, int(round(fraction * finite.size))), finite.size - 1)
        held[rng.permutation(finite)[:k]] = True
    return held


def _simulate(simulate_cstat: Callable[[np.ndarray], np.ndarray], u: np.ndarray, wave: int) -> np.ndarray:
    """Fold a design and validate the returned C-stat."""
    cstat = np.asarray(simulate_cstat(u), dtype=np.float64).reshape(-1)
    if cstat.shape[0] != u.shape[0]:
        raise ValueError(
            f"simulate_cstat returned {cstat.shape[0]} values for {u.shape[0]} points in wave {wave}."
        )
    return cstat


def _wave_summary(row: dict, n_waves: int, stop: str | None) -> str:
    """One log line summarising a wave, built from its history row."""

    def fmt(value: float | None, spec: str) -> str:
        return "-" if value is None else format(value, spec)

    return (
        f"[restriction] wave {row['wave']}/{n_waves}: {row['n']:,} folds (total {row['n_total']:,}), "
        f"best {row['best']:.6g}, threshold {row['threshold']:.6g} ({row['spans']:.2f} spans, wall {row['wall']}), "
        f"prior mass {row['prior_mass']:.4g}, purity {fmt(row['purity'], '.3f')}, "
        f"recall {fmt(row['recall'], '.3f')}, C-stat RMSE {row['rmse_cstat']:.4g}, "
        f"margin {row['margin_sigma']:.1f} sigma, {row['seconds']:.1f} s"
        + ("" if stop is None else f"; stop: {stop}")
    )


def run_history_matching(
    simulate_cstat: Callable[[np.ndarray], np.ndarray],
    d: int,
    cfg: HistoryMatching,
    *,
    param_names: list[str] | None = None,
    log: Callable[[str], None] | None = None,
) -> Restriction:
    r"""Restrict the unit cube to the region worth sampling.

    For each wave:

    1. draw ``cfg.simulations_per_parameter * d`` points (Latin hypercube on
       wave 1, uniform on the previous region afterwards) and fold them;
    2. fit the surrogate on the folds so far, except the held-out ones, on
       $\log C$ for wave 1 and on $\log(C - c_{ref})$ with
       $c_{ref} = C_{best} - \Delta/2$ after;
    3. place the threshold with
       [`threshold_rule`][bsixsa.restriction.rules.threshold_rule], widened by
       [`recall_check`][bsixsa.restriction.rules.recall_check] if needed;
    4. measure the region's prior mass;
    5. stop per [`stop_reason`][bsixsa.restriction.rules.stop_reason].

    Parameters:
        simulate_cstat: Maps unit-cube points ``(n, d)`` to their exact C-stat
            ``(n,)``, finite or ``+inf``.
        d: Dimension of the cube.
        cfg: History-matching settings.
        param_names: Parameter names, stored in the result for its reports.
        log: Receives one summary line per wave; silent by default.

    Returns:
        The [`Restriction`][bsixsa.restriction.region.Restriction].

    Raises:
        ValueError: On a bad *d* or *param_names*, a simulator returning the
            wrong number of values, or fewer than 2 finite C-stats in wave 1.
        RuntimeError: If a wave cannot be drawn within ``cfg.max_candidates`` candidates.
    """
    d = int(d)
    if d < 1:
        raise ValueError(f"run_history_matching expects d >= 1, got {d}.")
    if param_names is not None and len(param_names) != d:
        raise ValueError(f"param_names has {len(param_names)} entries for d = {d}.")
    rng = np.random.default_rng(cfg.seed)
    log = log or _no_log
    start = time.perf_counter()

    span = posterior_span(d)
    n_per_wave = cfg.simulations_per_parameter * d
    # Smallest prior mass from which one wave can still be drawn, with a 5-sigma margin.
    min_sampleable_mass = (1.0 + 5.0 / math.sqrt(n_per_wave)) * n_per_wave / cfg.max_candidates

    u_all = np.empty((0, d), dtype=np.float64)
    cstat_all = np.empty(0, dtype=np.float64)
    wave_all = np.empty(0, dtype=np.int64)
    held_all = np.empty(0, dtype=bool)
    history: list[dict] = []
    region: Region | None = None  # the previous wave's region
    prev_threshold: float | None = None

    with torch_threads(cfg.torch_threads):
        # Deferred: torch must not load before the solver forks its workers.
        from .regressor import train_regressor

        for w in range(1, cfg.n_waves + 1):
            wave_start = time.perf_counter()

            # 1. Draw the wave's points and fold them.
            if w == 1:
                u = latin_hypercube(d, n_per_wave, rng)
            else:
                u = region.sample(n_per_wave, rng, max_candidates=cfg.max_candidates)
            cstat = _simulate(simulate_cstat, u, w)
            if w == 1:
                n_finite = int(np.isfinite(cstat).sum())
                if n_finite < 2:
                    raise ValueError(
                        f"History matching: only {n_finite} of {len(cstat)} folds of the first wave "
                        "returned a finite C-stat, too few to train the C-stat surrogate (it needs at "
                        "least 2). An infinite C-stat means the model predicts zero counts in a bin "
                        "where counts were observed; check the prior ranges and the model set-up, or "
                        "run without history matching (SIXSA(history_matching=None))."
                    )

            purity = None if prev_threshold is None else float(np.mean(cstat <= prev_threshold))
            u_all = np.concatenate([u_all, u], axis=0)
            cstat_all = np.concatenate([cstat_all, cstat])
            wave_all = np.concatenate([wave_all, np.full(len(u), w, dtype=np.int64)])
            # Held-out folds stay out of every later fit, so the recall check stays out-of-sample.
            held = _hold_out(cstat, cfg.validation_fraction, rng)
            held_all = np.concatenate([held_all, held])
            best = float(cstat_all.min())

            # 2. Refit the surrogate on every fold so far, except the held-out ones.
            model, info = train_regressor(
                u_all,
                cstat_all,
                cfg=cfg,
                held_out=held_all,
                c_ref=None if w == 1 else best - 0.5 * span,
                focus_below=prev_threshold,
                seed=int(rng.integers(2**31 - 1)),
            )
            rmse_cstat = float(info["rmse_cstat"])

            # 3. Place the threshold, then let the recall check widen it.
            probe = rng.random((cfg.mass_probe, d))
            pred_probe = model.predict_cstat(probe)
            threshold, wall = threshold_rule(
                best=best,
                span=span,
                target_spans=cfg.target_spans,
                rmse_cstat=rmse_cstat,
                min_margin_sigma=cfg.min_margin_sigma,
                prev_threshold=prev_threshold,
                pred_probe=pred_probe,
                min_sampleable_mass=min_sampleable_mass,
            )
            val_idx = np.flatnonzero(held_all)
            pred_all = model.predict_cstat(u_all)
            threshold, recall_triggered, recall_info = recall_check(
                pred_val=pred_all[val_idx],
                cstat_val=cstat_all[val_idx],
                best=best,
                span=span,
                threshold=threshold,
                rmse_cstat=rmse_cstat,
            )
            if recall_triggered:
                wall = "recall"

            # 4. Measure the region.
            region = Region(model, threshold, d=d)
            prior_mass = region.prior_mass(
                rng,
                n_probe=cfg.mass_probe,
                max_candidates=cfg.mass_probe_budget,
            )
            truly = cstat_all <= threshold
            recall = float(np.mean(pred_all[truly] <= threshold)) if truly.any() else None

            # 5. Record, decide whether to stop.
            row = {
                "wave": w,
                "n": int(len(u)),
                "n_total": int(len(cstat_all)),
                "n_validation": int(val_idx.size),
                "threshold": threshold,
                "best": best,
                "prior_mass": prior_mass,
                "prior_mass_probe": region.last_probe_size,
                "delta": threshold - best,
                "spans": (threshold - best) / span,
                "purity": purity,
                "recall": recall,
                "margin_sigma": (threshold - (best + span)) / max(rmse_cstat, 1e-9),
                "rmse_log": float(info["rmse_log"]),
                "rmse_cstat": rmse_cstat,
                "epochs": int(info["epochs"]),
                "wall": wall,
                "recall_info": recall_info,
                "seconds": 0.0,
            }
            history.append(row)

            stop = stop_reason(
                wave=w,
                n_waves=cfg.n_waves,
                prior_mass=prior_mass,
                stop_when_acceptance=cfg.stop_when_acceptance,
                threshold=threshold,
                prev_threshold=prev_threshold,
                stop_when_threshold=cfg.stop_when_threshold,
                recall_triggered=recall_triggered,
            )
            row["seconds"] = time.perf_counter() - wave_start
            log(_wave_summary(row, cfg.n_waves, stop))
            prev_threshold = threshold
            if stop is not None:
                break

    return Restriction(
        region=region,
        threshold=threshold,
        best=best,
        span=span,
        prior_mass=prior_mass,
        rmse_cstat=rmse_cstat,
        u=u_all,
        cstat=cstat_all,
        wave=wave_all,
        history=history,
        stop_reason=stop,
        n_simulations=int(len(cstat_all)),
        seconds=time.perf_counter() - start,
        config=cfg,
        param_names=None if param_names is None else list(param_names),
    )
