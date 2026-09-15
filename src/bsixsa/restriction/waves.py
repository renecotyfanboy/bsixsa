r"""History-matching waves on the unit cube.

Port of SIXSA v2's ``run_restriction``: each wave folds a design exactly,
refits a C-stat surrogate on every fold so far, and cuts the cube on the
surrogate's predictions. Wave 1 is an exact Latin hypercube over the cube;
later waves are uniform draws from the previous region.

Every random stream is a child of one ``SeedSequence``, addressed by
``(purpose, wave)``: 1 = design, 2 = regressor, 3 = cost-floor probe,
4 = bounding-box draws (in [`Restriction.box`][bsixsa.restriction.region.Restriction.box]),
5 = prior-mass probe, 6 = validation split.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np

from .._cube import child_rng, child_seed, latin_hypercube, make_seed_sequence
from .._torch_utils import torch_threads
from ..exceptions import FoldError
from .config import HistoryMatching
from .region import Region, Restriction, _mass_probe_budget
from .rules import posterior_span, recall_override, stop_reason, threshold_rule

__all__ = ["run_history_matching"]


def _no_log(message: str) -> None:
    """Default logger: discard the message."""


def _hold_out(cstat: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Validation mask of one wave's new folds.

    Parameters:
        cstat:
            Exact C-stat of the wave's folds, shape ``(n,)``.
        fraction:
            Fraction of the finite folds to hold out.
        rng:
            Random generator of the split.

    Returns:
        Boolean mask of shape ``(n,)``: ``round(fraction * m)`` of the ``m``
        finite folds, clipped to ``[1, m - 1]``, or none when ``m < 2``.
    """
    finite = np.flatnonzero(np.isfinite(cstat))
    held = np.zeros(cstat.size, dtype=bool)
    if finite.size >= 2:
        k = min(max(1, int(round(fraction * finite.size))), finite.size - 1)
        held[rng.permutation(finite)[:k]] = True
    return held


def _fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def _simulate(simulate_cstat: Callable[[np.ndarray], np.ndarray], u: np.ndarray, wave: int) -> np.ndarray:
    """Fold a design and validate the returned C-stat."""
    cstat = np.asarray(simulate_cstat(u), dtype=np.float64).reshape(-1)
    if cstat.shape[0] != u.shape[0]:
        raise ValueError(
            f"simulate_cstat returned {cstat.shape[0]} values for {u.shape[0]} points in wave {wave}."
        )
    bad = np.isnan(cstat) | np.isneginf(cstat)
    if bad.any():
        first = int(np.flatnonzero(bad)[0])
        raise FoldError(
            f"Wave {wave}: {int(bad.sum())} of {cstat.size} folds returned a NaN or -inf C-stat "
            f"(first at unit-cube point {u[first].tolist()}). +inf is a valid zero likelihood; "
            "NaN and -inf point to a broken model evaluation."
        )
    return cstat


def run_history_matching(
    simulate_cstat: Callable[[np.ndarray], np.ndarray],
    d: int,
    cfg: HistoryMatching,
    *,
    seed_seq: np.random.SeedSequence,
    outdir: str | None = None,
    fingerprint: str = "",
    param_names: list[str] | None = None,
    log: Callable[[str], None] | None = None,
) -> Restriction:
    r"""Restrict the unit cube to the region worth sampling.

    For each wave $w = 1, \dots,$ ``cfg.n_waves``:

    1. draw ``cfg.simulations_per_parameter * d`` points (exact LHS on wave 1,
       uniform on the previous region afterwards) and fold them;
    2. fit the surrogate on every fold so far, on $\log C$ for wave 1 and on
       $\log(C - c_{ref})$ with $c_{ref} = C_{best} - \max(\Delta/2, 1)$ after;
    3. place the threshold with
       [`threshold_rule`][bsixsa.restriction.rules.threshold_rule], then let
       [`recall_override`][bsixsa.restriction.rules.recall_override] widen it
       if held-out folds would be lost. The validation split is sticky: each
       wave holds out ``cfg.validation_fraction`` of its own folds, and they
       stay out of every later fit, so the check reads the union of every
       wave's validation folds, none of which the current surrogate has
       seen;
    4. measure the region's prior mass on a uniform probe grown until 100
       hits, ``50 * cfg.mass_probe`` points (or the ``400 /
       cfg.stop_when_acceptance`` that resolve the acceptance stop, if more),
       or ``cfg.max_candidates``;
    5. stop per [`stop_reason`][bsixsa.restriction.rules.stop_reason].

    torch runs with ``cfg.torch_threads`` intra-op threads for the whole call,
    and the caller's global torch random state is left untouched.

    Parameters:
        simulate_cstat:
            Maps float64 unit-cube points of shape ``(n, d)`` to their exact
            C-stat, shape ``(n,)``. ``+inf`` is allowed.
        d:
            Dimension of the cube.
        cfg:
            History-matching settings.
        seed_seq:
            Root of every random stream of the run (an integer is accepted
            and converted).
        outdir:
            Directory of the per-wave PDFs; nothing is written when ``None``
            or when ``cfg.plot`` is false.
        fingerprint:
            Fingerprint of the inference problem, stored on the result.
        param_names:
            Parameter names for logs and plots.
        log:
            Receives one summary line per wave; silent by default.

    Returns:
        The [`Restriction`][bsixsa.restriction.region.Restriction].

    Raises:
        ValueError: If *d* is smaller than 1, *param_names* has the wrong
            length, or the simulator returns the wrong number of values.
        FoldError: If a fold returns a NaN or $-\infty$ C-stat.
        TruncationBudgetError: If a wave cannot be drawn from the previous
            region within ``cfg.max_candidates`` candidates.
    """
    d = int(d)
    if d < 1:
        raise ValueError(f"run_history_matching expects d >= 1, got {d}.")
    if param_names is not None and len(param_names) != d:
        raise ValueError(f"param_names has {len(param_names)} entries for d = {d}.")
    seed_seq = make_seed_sequence(seed_seq)
    log = log or _no_log
    t_start = time.perf_counter()

    span = posterior_span(d)
    n_per_wave = cfg.simulations_per_parameter * d
    mass_floor = (1.0 + 5.0 / math.sqrt(n_per_wave)) * n_per_wave / cfg.max_candidates
    mass_budget = _mass_probe_budget(cfg)

    u_all = np.empty((0, d), dtype=np.float64)
    cstat_all = np.empty(0, dtype=np.float64)
    wave_all = np.empty(0, dtype=np.int64)
    held_all = np.empty(0, dtype=bool)
    history: list[dict] = []
    region: Region | None = None
    prev_threshold: float | None = None
    stop: str | None = None

    with torch_threads(cfg.torch_threads):
        from .regressor import train_regressor

        for w in range(1, cfg.n_waves + 1):
            t_wave = time.perf_counter()
            design_rng = child_rng(seed_seq, 1, w)
            if region is None:
                u = latin_hypercube(d, n_per_wave, design_rng)
            else:
                u = region.sample(n_per_wave, design_rng, max_candidates=cfg.max_candidates)
            cstat = _simulate(simulate_cstat, u, w)

            purity = None if prev_threshold is None else float(np.mean(cstat <= prev_threshold))
            u_all = np.concatenate([u_all, u], axis=0)
            cstat_all = np.concatenate([cstat_all, cstat])
            wave_all = np.concatenate([wave_all, np.full(len(u), w, dtype=np.int64)])
            held_all = np.concatenate(
                [held_all, _hold_out(cstat, cfg.validation_fraction, child_rng(seed_seq, 6, w))]
            )
            best = float(cstat_all.min())

            # A fresh split per wave would train on earlier validation folds,
            # and the recall check would then read in-sample predictions.
            val_rows = np.flatnonzero(held_all)
            if val_rows.size == 0 or val_rows.size == int(np.isfinite(cstat_all).sum()):
                val_rows = None  # too few finite folds to hold any out per wave
            c_ref = None if w < 2 else best - max(0.5 * span, 1.0)
            model, info = train_regressor(
                u_all,
                cstat_all,
                cfg=cfg,
                c_ref=c_ref,
                focus_below=prev_threshold,
                seed=child_seed(seed_seq, 2, w),
                val_idx=val_rows,
            )
            rmse_cstat = float(info["rmse_cstat"])

            probe = child_rng(seed_seq, 3, w).random((cfg.mass_probe, d))
            pred = model.predict_cstat(probe)
            threshold, wall = threshold_rule(
                best=best,
                span=span,
                target_spans=cfg.target_spans,
                rmse_cstat=rmse_cstat,
                min_margin_sigma=cfg.min_margin_sigma,
                prev_threshold=prev_threshold,
                pred_probe=pred,
                mass_floor=mass_floor,
            )

            val_idx = np.asarray(info["val_idx"], dtype=np.int64)
            pred_all = model.predict_cstat(u_all)
            threshold, recall_triggered, recall_info = recall_override(
                pred_val=pred_all[val_idx],
                cstat_val=cstat_all[val_idx],
                best=best,
                span=span,
                threshold=threshold,
                rmse_cstat=rmse_cstat,
            )
            if recall_triggered:
                wall = "recall"

            region = Region(model, threshold, d=d)
            prior_mass = region.prior_mass(
                child_rng(seed_seq, 5, w), n_probe=cfg.mass_probe, max_candidates=mass_budget
            )
            truly = cstat_all <= threshold
            recall = float(np.mean(pred_all[truly] <= threshold)) if truly.any() else None

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
                "spans": (threshold - best) / max(span, 1e-12),
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

            if cfg.plot and outdir:
                try:
                    from .plots import plot_wave

                    path = plot_wave(
                        history, model, u_all, cstat_all, probe, pred, w, outdir, span, param_names=param_names
                    )
                    log(f"[restriction] wave {w} figure: {path}")
                except Exception as exc:  # a figure must never end a run
                    log(f"[restriction] wave {w} figure failed: {type(exc).__name__}: {exc}")

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
            row["seconds"] = time.perf_counter() - t_wave
            log(
                f"[restriction] wave {w}/{cfg.n_waves}: {len(u):,} folds (total {len(cstat_all):,}), "
                f"best {best:.6g}, threshold {threshold:.6g} ({row['spans']:.2f} spans, wall {wall}), "
                f"prior mass {prior_mass:.4g}, purity {_fmt(purity, '.3f')}, recall {_fmt(recall, '.3f')}, "
                f"C-stat RMSE {rmse_cstat:.4g}, margin {row['margin_sigma']:.1f} sigma, "
                f"{row['seconds']:.1f} s" + ("" if stop is None else f"; stop: {stop}")
            )
            prev_threshold = threshold
            if stop is not None:
                break

    return Restriction(
        region=region,
        threshold=float(threshold),
        best=float(cstat_all.min()),
        span=span,
        prior_mass=float(prior_mass),
        rmse_cstat=rmse_cstat,
        u=u_all,
        cstat=cstat_all,
        wave=wave_all,
        history=history,
        stop_reason=stop or "n_waves",
        n_simulations=int(len(cstat_all)),
        seconds=time.perf_counter() - t_start,
        fingerprint=fingerprint,
        config=cfg,
        seed_entropy=seed_seq.entropy,
        param_names=None if param_names is None else list(param_names),
        seed_spawn_key=tuple(seed_seq.spawn_key),
    )
