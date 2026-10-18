"""The restricted region and the result of a history-matching run."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._cube import in_cube
from .config import HistoryMatching

__all__ = ["Region", "Restriction"]

_DEFAULT_BATCH = 200_000


class Region:
    r"""The set $\{u \in [0, 1]^d : \hat C(u) \le t\}$ of a surrogate.

    Draws are made by rejection from the uniform cube, so they are exactly
    uniform on the region.

    Parameters:
        model: Surrogate with ``predict_cstat(u)`` mapping ``(n, d)`` points to ``(n,)`` C-stats.
        threshold: C-stat threshold $t$.
        d: Dimension of the cube; read from ``model.d`` when omitted.

    Attributes:
        last_probe_size: Points probed by the last
            [`prior_mass`][bsixsa.restriction.region.Region.prior_mass] call.
        last_probe_hits: How many of those points the region accepted.

    Raises:
        ValueError: If *d* is omitted and the surrogate has no ``d`` attribute.
    """

    def __init__(self, model: Any, threshold: float, d: int | None = None):
        if d is None:
            d = getattr(model, "d", None)
        if d is None:
            raise ValueError(
                "Region needs the dimension of the cube: pass d=... or use a surrogate with a `d` attribute."
            )
        self.model = model
        self.threshold = float(threshold)
        self.d = int(d)
        self.last_probe_size: int | None = None
        self.last_probe_hits: int | None = None

    def __repr__(self) -> str:
        return f"Region(d={self.d}, threshold={self.threshold:.6g})"

    def accepts(self, u: np.ndarray) -> np.ndarray:
        """Return a boolean mask of the points of shape ``(n, d)`` that lie in the region."""
        u = np.asarray(u, dtype=np.float64)
        if u.ndim == 1:
            u = u.reshape(1, -1)
        # Cube test in float64: the float32 surrogate would round 1 + 1e-9 onto the face.
        inside = in_cube(u)
        out = np.zeros(inside.shape[0], dtype=bool)
        if not inside.any():
            return out
        rows = u if inside.all() else u[inside]
        pred = np.asarray(self.model.predict_cstat(rows), dtype=np.float64)
        out[inside] = pred.reshape(-1) <= self.threshold
        return out

    def sample(
        self,
        n: int,
        rng: np.random.Generator,
        *,
        max_candidates: int,
        batch: int = _DEFAULT_BATCH,
    ) -> np.ndarray:
        """Draw *n* points uniformly on the region by rejection from the cube.

        Parameters:
            n: Number of draws.
            rng: Random generator.
            max_candidates: Largest number of uniform candidates the call may draw.
            batch: Candidates per surrogate call.

        Returns:
            float64 array of shape ``(n, d)``.

        Raises:
            ValueError: If *n* is negative.
            RuntimeError: If *n* draws need more than *max_candidates* candidates.
        """
        n = int(n)
        if n < 0:
            raise ValueError(f"sample expects n >= 0, got {n}.")
        budget, batch = max(int(max_candidates), 0), max(1, int(batch))
        kept: list[np.ndarray] = []
        got = drawn = 0
        while got < n and drawn < budget:
            if drawn > 0:
                # Give up once even an optimistic (about 5-sigma) acceptance cannot fill n draws.
                p_high = (got + 5.0 * math.sqrt(got + 1.0) + 5.0) / drawn
                if drawn + (n - got) / p_high > budget:
                    break
            ask = min(batch, budget - drawn)
            candidates = rng.random((ask, self.d))
            drawn += ask
            accepted = candidates[self.accepts(candidates)][: n - got]
            kept.append(accepted)
            got += len(accepted)
        if got < n:
            rate = got / drawn if drawn else 0.0
            need = f"about {n / rate:,.0f}" if rate > 0 else "an unknown number of"
            raise RuntimeError(
                f"Sampling {n:,} points from the region (threshold {self.threshold:.6g}) needs {need} "
                f"uniform candidates, over the budget of {int(max_candidates):,}: {got:,} accepted from "
                f"{drawn:,} candidates (acceptance {rate:.3g}). The region is too small for rejection "
                "from the cube; widen it (larger HistoryMatching.target_spans) or raise "
                "HistoryMatching.max_candidates."
            )
        return np.concatenate(kept, axis=0) if kept else np.empty((0, self.d), dtype=np.float64)

    def prior_mass(
        self,
        rng: np.random.Generator,
        *,
        n_probe: int,
        min_hits: int = 100,
        max_candidates: int,
    ) -> float:
        """Estimate the region's prior mass (its volume in the cube).

        The uniform probe grows fourfold until *min_hits* points are accepted
        or *max_candidates* points have been probed.

        Parameters:
            rng: Random generator.
            n_probe: Size of the first probe.
            min_hits: Accepted points wanted before stopping.
            max_candidates: Largest total probe size.

        Returns:
            ``hits / probed``; ``0.0`` means the mass is below about ``3 / probed``.

        Raises:
            ValueError: If *n_probe* < 1.
        """
        n_probe = int(n_probe)
        if n_probe < 1:
            raise ValueError(f"prior_mass expects n_probe >= 1, got {n_probe}.")
        limit = max(int(max_candidates), 1)
        target = min(n_probe, limit)
        probed = hits = 0
        while True:
            while probed < target:
                ask = min(_DEFAULT_BATCH, target - probed)
                hits += int(self.accepts(rng.random((ask, self.d))).sum())
                probed += ask
            if hits >= int(min_hits) or probed >= limit:
                break
            target = min(4 * target, limit)
        self.last_probe_size = probed
        self.last_probe_hits = hits
        return hits / probed


def _to_jsonable(value: Any) -> Any:
    """Recursively convert numpy values to JSON types; non-finite floats become ``None``."""
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


@dataclass(eq=False, repr=False)
class Restriction:
    r"""Result of a history-matching run.

    Attributes:
        region: Final [`Region`][bsixsa.restriction.region.Region].
        threshold: Final C-stat threshold.
        best: Smallest exact C-stat folded.
        span: Posterior span $\chi^2_d(0.999)$ in C-stat units.
        prior_mass: Final region's prior mass.
        rmse_cstat: Last surrogate's held-out C-stat RMSE near the region.
        u: Every folded unit-cube point, shape ``(N, d)``.
        cstat: Their exact C-stat, shape ``(N,)``.
        wave: 1-based wave of each fold, shape ``(N,)``.
        history: One dict per wave, with the keys

            - ``wave``: 1-based wave index;
            - ``n``, ``n_total``: folds made in this wave, and so far;
            - ``n_validation``: held-out folds so far;
            - ``threshold``, ``best``: C-stat threshold and best exact C-stat;
            - ``delta``, ``spans``: ``threshold - best``, in C-stat and in spans;
            - ``prior_mass``, ``prior_mass_probe``: region's prior mass, and the
              uniform points used to measure it;
            - ``purity``: fraction of this wave's folds below the previous
              threshold (``None`` on wave 1); low purity means wasted folds;
            - ``recall``: fraction of the folds below the threshold that the
              surrogate keeps; below 1 the region misses points;
            - ``margin_sigma``: threshold above ``best + span``, in surrogate errors;
            - ``rmse_log``, ``rmse_cstat``, ``epochs``: surrogate validation
              errors and training epochs;
            - ``wall``: rule that set the threshold (``"request"``, ``"safety"``,
              ``"cost"``, ``+clamped`` if held at the previous value, or ``"recall"``);
            - ``recall_info``: details of the recall check;
            - ``seconds``: wall-clock time of the wave.
        stop_reason: ``"acceptance"``, ``"threshold"`` or ``"n_waves"``.
        n_simulations: Number of exact folds spent.
        seconds: Wall time of the run.
        config: Settings of the run.
        param_names: Parameter names, saved in ``history.json``.
    """

    region: Region
    threshold: float
    best: float
    span: float
    prior_mass: float
    rmse_cstat: float
    u: np.ndarray
    cstat: np.ndarray
    wave: np.ndarray
    history: list[dict]
    stop_reason: str
    n_simulations: int
    seconds: float
    config: HistoryMatching
    param_names: list[str] | None = None

    def __post_init__(self) -> None:
        self.u = np.asarray(self.u, dtype=np.float64)
        self.cstat = np.asarray(self.cstat, dtype=np.float64).reshape(-1)
        self.wave = np.asarray(self.wave, dtype=np.int64).reshape(-1)
        if self.u.ndim != 2 or not (self.u.shape[0] == self.cstat.size == self.wave.size):
            raise ValueError(
                "Restriction expects u of shape (N, d) with cstat and wave of shape (N,), got "
                f"{self.u.shape}, {self.cstat.shape} and {self.wave.shape}."
            )

    def __repr__(self) -> str:
        return (
            f"Restriction(d={self.u.shape[1]}, threshold={self.threshold:.6g}, best={self.best:.6g}, "
            f"prior_mass={self.prior_mass:.4g}, waves={len(self.history)}, "
            f"n_simulations={self.n_simulations}, stop_reason={self.stop_reason!r})"
        )

    @property
    def d(self) -> int:
        """Dimension of the cube."""
        return int(self.u.shape[1])

    def save(self, directory: str) -> None:
        """Write ``restriction.npz`` and ``history.json`` to *directory*."""
        os.makedirs(directory, exist_ok=True)
        np.savez(
            os.path.join(directory, "restriction.npz"),
            u=self.u,
            cstat=self.cstat,
            wave=self.wave,
            threshold=self.threshold,
            best=self.best,
            span=self.span,
            prior_mass=self.prior_mass,
            rmse_cstat=self.rmse_cstat,
        )
        payload = {
            "history": self.history,
            "stop_reason": self.stop_reason,
            "n_simulations": self.n_simulations,
            "seconds": self.seconds,
            "config": self.config.model_dump(),
            "param_names": self.param_names,
        }
        with open(os.path.join(directory, "history.json"), "w", encoding="utf-8") as handle:
            json.dump(_to_jsonable(payload), handle, indent=2)

    def report(self) -> str:
        """Human-readable summary of the waves."""

        def fmt(value: Any, spec: str) -> str:
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                return "-".rjust(int(spec.split(".")[0]))
            return format(value, spec)

        lines = [
            f"History matching: {len(self.history)} wave(s), {self.n_simulations:,} exact folds "
            f"in {self.seconds:.1f} s, stopped by {self.stop_reason!r}.",
            f"Region: cstat <= {self.threshold:.6g} (best {self.best:.6g}, "
            f"{(self.threshold - self.best) / self.span:.2f} spans of {self.span:.4g}), "
            f"prior mass {self.prior_mass:.4g}, surrogate C-stat RMSE {self.rmse_cstat:.4g}.",
            "",
            f"{'wave':>4} {'n':>7} {'threshold':>12} {'best':>12} {'spans':>7} {'mass':>10} "
            f"{'purity':>7} {'recall':>7} {'rmse_C':>9}  wall",
        ]
        for row in self.history:
            lines.append(
                f"{row.get('wave', 0):>4} {row.get('n', 0):>7,} {fmt(row.get('threshold'), '12.6g')} "
                f"{fmt(row.get('best'), '12.6g')} {fmt(row.get('spans'), '7.2f')} "
                f"{fmt(row.get('prior_mass'), '10.4g')} {fmt(row.get('purity'), '7.3f')} "
                f"{fmt(row.get('recall'), '7.3f')} {fmt(row.get('rmse_cstat'), '9.4g')}  {row.get('wall', '')}"
            )
        return "\n".join(lines)
