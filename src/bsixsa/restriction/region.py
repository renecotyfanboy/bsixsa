r"""The restricted region and the result of a history-matching run.

A [`Region`][bsixsa.restriction.region.Region] is the set
$\{u \in [0, 1]^d : \hat C(u) \le t\}$ of a trained surrogate. It is sampled
by plain rejection from the uniform cube, with no envelope, so its draws are
exactly uniform on it: everything downstream that treats them as a flat
sample relies on that, and there is nothing else to verify.

A [`Restriction`][bsixsa.restriction.region.Restriction] bundles the final
region with every exact fold of the waves, the per-wave history, and the
axis-aligned bounding box consumed by the backends that cannot use the region
itself.

This module imports neither XSPEC nor, at import time, torch.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np

from .._cube import child_rng, in_cube
from ..exceptions import TruncationBudgetError
from .config import HistoryMatching
from .rules import pad_box

if TYPE_CHECKING:
    from ..priors import MultipleIndependent

__all__ = ["Region", "Restriction"]

_DEFAULT_BATCH = 200_000


def _mass_probe_budget(cfg: HistoryMatching) -> int:
    """Largest total probe size of a prior-mass measurement.

    Parameters:
        cfg:
            History-matching settings.

    Returns:
        ``min(max_candidates, max(50 * mass_probe, ceil(400 / stop_when_acceptance)))``.
    """
    # 400 / stop_when_acceptance points expect 100 hits at a quarter of the
    # acceptance stop, so the stop is decided on a measured mass; the whole
    # candidate budget would let a tiny region cost hours of surrogate calls.
    resolve = math.ceil(400.0 / float(cfg.stop_when_acceptance))
    return int(min(cfg.max_candidates, max(50 * cfg.mass_probe, resolve)))


def _box_candidate_budget(cfg: HistoryMatching) -> int:
    """Largest number of uniform candidates spent on the bounding-box draws.

    Parameters:
        cfg:
            History-matching settings.

    Returns:
        ``min(max_candidates, 50 * max(mass_probe, box_samples))``.
    """
    return int(min(cfg.max_candidates, 50 * max(cfg.mass_probe, cfg.box_samples)))


def _model_dimension(model: Any) -> int | None:
    """Input dimension advertised by a surrogate, if any."""
    d = getattr(model, "d", None)
    if d is not None:
        return int(d)
    t_mu = getattr(model, "t_mu", None)
    shape = getattr(t_mu, "shape", None)
    if shape:
        return int(shape[-1])
    return None


class Region:
    r"""The set $\{u \in [0, 1]^d : \hat C(u) \le t\}$ of a surrogate.

    Parameters:
        model:
            Surrogate exposing ``predict_cstat(u, batch=...)`` that maps
            float64 points of shape ``(n, d)`` to C-stat predictions of shape
            ``(n,)``.
        threshold:
            C-stat threshold $t$.
        d:
            Dimension of the cube; read from ``model.d`` (or the shape of
            ``model.t_mu``) when omitted.

    Attributes:
        model: The surrogate.
        threshold: The threshold $t$.
        d: Dimension of the cube, or ``None`` if it could not be inferred.
        last_probe_size: Number of points probed by the last
            [`prior_mass`][bsixsa.restriction.region.Region.prior_mass] call.
        last_probe_hits: Number of accepted points of that probe.
    """

    def __init__(self, model: Any, threshold: float, d: int | None = None):
        self.model = model
        self.threshold = float(threshold)
        self.d = int(d) if d is not None else _model_dimension(model)
        self.last_probe_size: int | None = None
        self.last_probe_hits: int | None = None

    def __repr__(self) -> str:
        return f"Region(d={self.d}, threshold={self.threshold:.6g})"

    def _dimension(self) -> int:
        if self.d is None:
            raise ValueError(
                "The region's dimension is unknown: pass d= to Region or give the model a `d` attribute."
            )
        return self.d

    def accepts(self, u) -> np.ndarray:
        """Test membership in the region.

        The cube test is done in float64 before the surrogate sees the points,
        because the surrogate works in float32 and would round $1 + 10^{-9}$
        onto the face.

        Parameters:
            u:
                Points of shape ``(n, d)`` (or ``(d,)`` for one point).

        Returns:
            Boolean array of shape ``(n,)``, ``True`` for the points inside the
            closed cube whose predicted C-stat is at most the threshold.
        """
        u = np.asarray(u, dtype=np.float64)
        if u.ndim == 1:
            u = u.reshape(1, -1)
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
        """Draw points uniformly on the region by rejection from the cube.

        Candidates are uniform cube points kept on the surrogate's verdict, so
        the accepted points are uniform on the region exactly. The candidate
        budget is enforced before it is exhausted too: once the observed
        acceptance makes the remaining draws unaffordable even at an
        optimistic (roughly 5-sigma) acceptance rate, the call stops.

        Parameters:
            n:
                Number of draws.
            rng:
                Random generator consumed by the candidates.
            max_candidates:
                Largest number of candidates the call may draw.
            batch:
                Candidates per surrogate call.

        Returns:
            float64 array of shape ``(n, d)``.

        Raises:
            ValueError: If *n* is negative, or the dimension is unknown.
            TruncationBudgetError: If filling *n* draws needs more than
                *max_candidates* candidates.
        """
        n = int(n)
        if n < 0:
            raise ValueError(f"sample expects n >= 0, got {n}.")
        d = self._dimension()
        if n == 0:
            return np.empty((0, d), dtype=np.float64)
        max_candidates = int(max_candidates)
        batch = max(1, int(batch))

        kept: list[np.ndarray] = []
        got = drawn = 0
        while got < n:
            if drawn > 0:
                # Upper bound on the acceptance: a false stop needs the hit
                # count ~5 Poisson sigmas below its mean.
                p_high = (got + 5.0 * math.sqrt(got + 1.0) + 5.0) / drawn
                if drawn + (n - got) / p_high > max_candidates:
                    self._raise_budget(n, got, drawn, max_candidates)
            ask = min(batch, max_candidates - drawn)
            if ask <= 0:
                self._raise_budget(n, got, drawn, max_candidates)
            candidates = rng.random((ask, d))
            drawn += ask
            accepted = candidates[self.accepts(candidates)]
            if len(accepted):
                kept.append(accepted[: n - got])
                got += min(len(accepted), n - got)
        return np.concatenate(kept, axis=0)

    def _sample_within(
        self,
        n: int,
        rng: np.random.Generator,
        *,
        max_candidates: int,
        batch: int = _DEFAULT_BATCH,
    ) -> tuple[np.ndarray, int]:
        """Draw up to *n* points uniformly on the region within a candidate budget.

        Unlike [`sample`][bsixsa.restriction.region.Region.sample], running
        out of candidates is not an error: the draws obtained so far are
        returned. They are still uniform on the region, since the stop depends
        on the candidate count only.

        Parameters:
            n:
                Largest number of draws.
            rng:
                Random generator consumed by the candidates.
            max_candidates:
                Largest number of candidates the call may draw.
            batch:
                Candidates per surrogate call.

        Returns:
            ``(draws, n_candidates)``: float64 draws of shape ``(m, d)`` with
            ``m <= n``, and the number of candidates spent.
        """
        d = self._dimension()
        n, budget, batch = max(int(n), 0), max(int(max_candidates), 0), max(1, int(batch))
        kept: list[np.ndarray] = []
        got = drawn = 0
        while got < n and drawn < budget:
            ask = min(batch, budget - drawn)
            candidates = rng.random((ask, d))
            drawn += ask
            accepted = candidates[self.accepts(candidates)]
            if len(accepted):
                kept.append(accepted[: n - got])
                got += min(len(accepted), n - got)
        draws = np.concatenate(kept, axis=0) if kept else np.empty((0, d), dtype=np.float64)
        return draws, drawn

    def _raise_budget(self, n: int, got: int, drawn: int, max_candidates: int) -> NoReturn:
        rate = got / drawn if drawn else 0.0
        need = f"about {n / rate:,.0f}" if rate > 0 else "an unknown number of"
        raise TruncationBudgetError(
            f"Sampling {n:,} points from the region (threshold {self.threshold:.6g}) needs {need} "
            f"uniform candidates, over the budget of {max_candidates:,}: {got:,} accepted from "
            f"{drawn:,} candidates (acceptance {rate:.3g}). The region is too small for rejection "
            "from the cube; widen it (larger target_spans) or raise max_candidates."
        )

    def prior_mass(
        self,
        rng: np.random.Generator,
        *,
        n_probe: int,
        min_hits: int = 100,
        max_candidates: int,
    ) -> float:
        """Estimate the region's prior mass (its volume in the cube).

        A uniform probe of *n_probe* points is grown fourfold (in total size)
        until at least *min_hits* points are accepted or *max_candidates*
        points have been probed, so that a small mass is measured rather
        than reported as zero.

        Parameters:
            rng:
                Random generator consumed by the probe.
            n_probe:
                Size of the first probe.
            min_hits:
                Accepted points wanted before stopping.
            max_candidates:
                Largest total probe size.

        Returns:
            ``hits / probed``, ``0.0`` if no probe point was accepted (the
            mass is then below about ``3 / probed`` at 95 %). The probe size
            and hits are stored in ``last_probe_size`` and ``last_probe_hits``.

        Raises:
            ValueError: If *n_probe* is smaller than 1, or the dimension is
                unknown.
        """
        d = self._dimension()
        n_probe = int(n_probe)
        if n_probe < 1:
            raise ValueError(f"prior_mass expects n_probe >= 1, got {n_probe}.")
        limit = max(int(max_candidates), 1)
        target = min(n_probe, limit)
        probed = hits = 0
        while True:
            while probed < target:
                ask = min(_DEFAULT_BATCH, target - probed)
                hits += int(self.accepts(rng.random((ask, d))).sum())
                probed += ask
            if hits >= int(min_hits) or probed >= limit:
                break
            target = min(4 * target, limit)
        self.last_probe_size = probed
        self.last_probe_hits = hits
        return hits / probed if hits else 0.0


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
        rmse_cstat: Last regressor's C-stat RMSE near the boundary.
        u: Every folded unit-cube point, shape ``(N, d)``.
        cstat: Their exact C-stat, shape ``(N,)``.
        wave: 1-based wave of each fold, shape ``(N,)``.
        history: One dict per wave (threshold, mass, recall, binding wall...).
        stop_reason: ``"acceptance"``, ``"threshold"`` or ``"n_waves"``.
        n_simulations: Number of exact folds spent.
        seconds: Wall time of the run.
        fingerprint: Fingerprint of the inference problem it was computed for.
        config: Settings of the run.
        seed_entropy: Entropy of the run's ``SeedSequence``, from which the
            bounding-box draws are derived.
        param_names: Parameter names, for reports and plots.
        seed_spawn_key: Spawn key of the run's ``SeedSequence``, so that the
            box stream stays inside the run's own namespace.
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
    fingerprint: str
    config: HistoryMatching
    seed_entropy: int | None = None
    param_names: list[str] | None = None
    seed_spawn_key: tuple[int, ...] = ()
    box_info: dict = field(default_factory=dict, init=False)
    _box: tuple[np.ndarray, np.ndarray] | None = field(default=None, init=False)
    _box_core: tuple[np.ndarray, np.ndarray] | None = field(default=None, init=False)
    _truncated: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.u = np.asarray(self.u, dtype=np.float64)
        self.cstat = np.asarray(self.cstat, dtype=np.float64).reshape(-1)
        self.wave = np.asarray(self.wave, dtype=np.int64).reshape(-1)
        if self.u.ndim != 2 or not (self.u.shape[0] == self.cstat.size == self.wave.size):
            raise ValueError(
                "Restriction expects u of shape (N, d) with cstat and wave of shape (N,), got "
                f"{self.u.shape}, {self.cstat.shape} and {self.wave.shape}."
            )
        self.seed_spawn_key = tuple(int(k) for k in self.seed_spawn_key)
        if self.region.d is None:
            self.region.d = self.u.shape[1]

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

    @property
    def _inside(self) -> np.ndarray:
        return self.cstat <= self.threshold

    @property
    def u_in(self) -> np.ndarray:
        """Folded points whose exact C-stat is at most the threshold, shape ``(n, d)``."""
        return self.u[self._inside]

    @property
    def cstat_in(self) -> np.ndarray:
        """Exact C-stat of [`u_in`][bsixsa.restriction.region.Restriction.u_in], shape ``(n,)``."""
        return self.cstat[self._inside]

    def _compute_box(self) -> None:
        t_start = time.perf_counter()
        d = self.d
        info: dict[str, Any] = {"n_exact_inside": int(self._inside.sum())}
        parts = [self.u_in]
        folded = self.u[self.region.accepts(self.u)]
        parts.append(folded)
        info["n_folded_accepted"] = int(len(folded))
        info["n_region_samples"] = 0
        info["n_candidates"] = 0
        info["sampling"] = "skipped"
        if self.config.box_samples > 0:
            seq = np.random.SeedSequence(self.seed_entropy, spawn_key=self.seed_spawn_key)
            budget = _box_candidate_budget(self.config)
            drawn, n_candidates = self.region._sample_within(
                self.config.box_samples, child_rng(seq, 4), max_candidates=budget
            )
            parts.append(drawn)
            info["n_region_samples"] = int(len(drawn))
            info["n_candidates"] = int(n_candidates)
            info["candidate_budget"] = budget
            info["sampling"] = "ok" if len(drawn) == self.config.box_samples else "partial"
        points = np.concatenate(parts, axis=0)
        info["n_points"] = int(len(points))

        if len(points) < 2:
            lo, hi = np.zeros(d), np.ones(d)
            info["full_cube"] = True
            info["seconds"] = time.perf_counter() - t_start
            self._box_core = (lo.copy(), hi.copy())
            self._box = (lo, hi)
            self.box_info = info
            return
        lo, hi = points.min(axis=0), points.max(axis=0)
        self._box_core = (lo.copy(), hi.copy())
        lo, hi = pad_box(
            lo,
            hi,
            box_padding=self.config.box_padding,
            rmse_cstat=self.rmse_cstat,
            delta=self.threshold - self.best,
        )
        # A zero-width axis (all points share one coordinate) cannot be padded
        # proportionally; keeping the whole axis never cuts posterior mass.
        flat = (hi - lo) < 1e-9
        lo[flat], hi[flat] = 0.0, 1.0
        info["full_cube"] = False
        info["full_axes"] = np.flatnonzero(flat).tolist()
        info["seconds"] = time.perf_counter() - t_start
        self._box = (lo, hi)
        self.box_info = info

    def box(self) -> tuple[np.ndarray, np.ndarray]:
        """Padded unit-cube bounding box of the region, computed once.

        The box encloses the folded points with an exact C-stat below the
        threshold, the folded points the region accepts, and up to
        ``config.box_samples`` region draws. The draws spend at most
        ``50 * max(config.mass_probe, config.box_samples)`` uniform candidates
        (and ``config.max_candidates``), about the cost of one mass probe; a
        region too small to fill them within that budget keeps the draws it
        got (``box_info["sampling"] == "partial"``). The box is padded by
        [`pad_box`][bsixsa.restriction.rules.pad_box] and clipped to the
        cube; with fewer than 2 points it is the whole cube. ``box_info``
        records the counts and the wall time (``seconds``).

        Nothing computes the box but this method (and the members that call
        it), so a restriction consumed only as a region never pays for it.

        Returns:
            ``(lo, hi)`` float64 copies of shape ``(d,)``.
        """
        if self._box is None:
            self._compute_box()
        return self._box[0].copy(), self._box[1].copy()

    @property
    def box_core(self) -> tuple[np.ndarray, np.ndarray]:
        """Unpadded bounding box ``(lo, hi)`` behind [`box`][bsixsa.restriction.region.Restriction.box]."""
        if self._box_core is None:
            self._compute_box()
        return self._box_core[0].copy(), self._box_core[1].copy()

    def truncated_prior(self, base_prior: MultipleIndependent) -> tuple[MultipleIndependent, float]:
        r"""Truncate a prior to the bounding box, cached per prior object.

        Parameters:
            base_prior:
                The untruncated prior whose unit cube the restriction lives in.

        Returns:
            ``(prior, log_mass)`` from ``base_prior.truncated(*self.box())``,
            where ``log_mass`` is $\log M_B$, the evidence correction.
        """
        key = id(base_prior)
        cached = self._truncated.get(key)
        # The cache keeps the base alive, so its id cannot be reused by
        # another object while the entry exists; the identity check is cheap
        # insurance anyway.
        if cached is not None and cached[0] is base_prior:
            return cached[1]
        result = base_prior.truncated(*self.box())
        self._truncated[key] = (base_prior, result)
        return result

    def check_fingerprint(self, fingerprint: str) -> None:
        """Refuse to be used for another inference problem.

        Parameters:
            fingerprint:
                Fingerprint of the problem at hand (``Solver.fingerprint()``).

        Raises:
            ValueError: If it differs from the one the restriction was
                computed for.
        """
        if self.fingerprint != fingerprint:
            raise ValueError(
                "This restriction was computed for another inference problem (fingerprint "
                f"{self.fingerprint!r}, expected {fingerprint!r}). A region describes the likelihood "
                "of one spectrum under one prior; reusing it after the data, the parameters or the "
                "prior changed would restrict the wrong part of parameter space. Recompute it."
            )

    def save(self, directory: str) -> None:
        """Write ``restriction.npz`` and ``history.json`` to *directory*.

        The bounding box (``box_lo``, ``box_hi``, ``box_core_lo``,
        ``box_core_hi``) is written only once it has been computed: saving
        never spends region draws. Save again after
        [`box`][bsixsa.restriction.region.Restriction.box] to record it.

        Parameters:
            directory:
                Output directory, created if needed.
        """
        os.makedirs(directory, exist_ok=True)
        arrays: dict[str, Any] = {}
        if self._box is not None and self._box_core is not None:
            arrays = {
                "box_lo": self._box[0],
                "box_hi": self._box[1],
                "box_core_lo": self._box_core[0],
                "box_core_hi": self._box_core[1],
            }
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
            **arrays,
        )
        payload = {
            "history": self.history,
            "stop_reason": self.stop_reason,
            "n_simulations": self.n_simulations,
            "seconds": self.seconds,
            "fingerprint": self.fingerprint,
            "config": self.config.model_dump(),
            "param_names": self.param_names,
            "seed_entropy": None if self.seed_entropy is None else str(self.seed_entropy),
            "seed_spawn_key": list(self.seed_spawn_key),
            "box_info": self.box_info,
        }
        with open(os.path.join(directory, "history.json"), "w", encoding="utf-8") as handle:
            json.dump(_to_jsonable(payload), handle, indent=2)

    def report(self) -> str:
        """Human-readable summary of the waves and the result.

        The bounding box is included only once it has been computed, so that
        the report never spends region draws.

        Returns:
            A multi-line string.
        """

        def fmt(value: Any, spec: str) -> str:
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                width = spec.split(".")[0]
                return "-".rjust(int(width) if width.isdigit() else 0)
            return format(value, spec)

        lines = [
            f"History matching: {len(self.history)} wave(s), {self.n_simulations:,} exact folds "
            f"in {self.seconds:.1f} s, stopped by {self.stop_reason!r}.",
            f"Region: cstat <= {self.threshold:.6g} (best {self.best:.6g}, "
            f"{(self.threshold - self.best) / max(self.span, 1e-12):.2f} spans of {self.span:.4g}), "
            f"prior mass {self.prior_mass:.4g}, regressor C-stat RMSE {self.rmse_cstat:.4g}.",
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
        if self._box is not None:
            lo, hi = self._box
            names = self.param_names or [f"u{j}" for j in range(self.d)]
            lines += ["", f"Bounding box (unit cube), log M_B = {float(np.sum(np.log(hi - lo))):.4g}:"]
            width = max(len(str(name)) for name in names)
            for name, a, b in zip(names, lo, hi):
                lines.append(f"  {str(name):<{width}}  [{a:.4f}, {b:.4f}]")
            if self.box_info.get("sampling") == "partial":
                lines.append(
                    f"  ({self.box_info.get('n_region_samples', 0):,} of {self.config.box_samples:,} region "
                    "draws: candidate budget reached)"
                )
        return "\n".join(lines)
