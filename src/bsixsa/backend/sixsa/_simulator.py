r"""The simulator SIXSA folds through: unit-cube points in, C-stat and counts out.

[`CubeSimulator`][bsixsa.backend.sixsa._simulator.CubeSimulator] is the only
place where the SIXSA backend touches the forward model. It works in the unit
cube $u = F(\theta)$ of the solver's base prior and routes every fold through
``solver._fold`` under ``solver.fold_phase``, so that the solver's per-phase
fold counters stay the single source of accounting.

Two conventions matter to the importance weights built downstream:

- a row outside the closed cube $[0, 1]^d$ (or with a non-finite coordinate)
  has zero prior density, so it is not folded and gets $C = +\infty$, i.e. a
  zero weight that still counts in the number of raw draws;
- a fold returning ``NaN`` or $-\infty$ is a broken forward model, not a
  zero-likelihood point, and raises
  [`FoldError`][bsixsa.exceptions.FoldError].

The Poisson realisation of the simulated spectra happens here, in the parent
process, from child streams of one ``SeedSequence``: the counts are a pure
function of the seed, the key and the chunking, whatever the fold pool does.

The module imports numpy only; the solver is used through its attributes, so
neither XSPEC nor ``bsixsa.solver`` is imported.
"""

from __future__ import annotations

import contextlib
import operator
from collections.abc import Sequence
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

from ..._cube import child_rng, in_cube, to_physical
from ...exceptions import FoldError

if TYPE_CHECKING:  # pragma: no cover
    from ...solver import Solver

__all__ = ["CubeSimulator"]


class CubeSimulator:
    r"""Fold unit-cube points through a solver, in chunks, with seeded Poisson noise.

    Parameters:
        solver:
            A [`Solver`][bsixsa.solver.Solver]-like object exposing
            ``base_prior``, ``observed_counts``, ``n_bins``,
            ``num_parameters``, ``fold_phase(name)`` and
            ``_fold(parameters, return_kind=..., progress_bar=...)``; an
            optional ``pool_scope()`` is entered around each call so that
            its chunks share one folding pool.
        seed_seq:
            Parent seed sequence of the Poisson realisations.
        fold_chunk:
            Maximum number of rows passed to one ``_fold`` call. It bounds the
            memory of the model counts, and it also delimits the Poisson
            streams (see [`counts`][bsixsa.backend.sixsa._simulator.CubeSimulator.counts]).
        tracer:
            Optional object with a ``record(params, cstat)`` method (e.g. an
            [`EvaluationTracer`][bsixsa.backend.tracer.EvaluationTracer]),
            called once per folded chunk with physical parameters.

    Attributes:
        d:
            Number of free parameters.
        n_bins:
            Number of channels of the spectrum.
        x_obs:
            Observed counts, float64 of shape ``(n_bins,)``, read once at
            construction and read-only.
        seconds:
            Wall-clock seconds spent per phase label, accumulated over calls.

    Raises:
        TypeError: If *seed_seq* is not a ``np.random.SeedSequence`` or
            *fold_chunk* is not an integer.
        ValueError: If *fold_chunk* is smaller than 1, or the observed counts
            do not have ``n_bins`` entries.
    """

    def __init__(
        self,
        solver: Solver,
        *,
        seed_seq: np.random.SeedSequence,
        fold_chunk: int = 10_000,
        tracer: Any = None,
    ) -> None:
        if not isinstance(seed_seq, np.random.SeedSequence):
            raise TypeError(f"seed_seq must be a numpy SeedSequence, got {type(seed_seq).__name__}.")
        try:
            fold_chunk = operator.index(fold_chunk)
        except TypeError:
            raise TypeError(f"fold_chunk must be an integer, got {type(fold_chunk).__name__}.") from None
        if fold_chunk < 1:
            raise ValueError(f"fold_chunk must be >= 1, got {fold_chunk}.")

        self._solver = solver
        self._prior = solver.base_prior
        self.seed_seq = seed_seq
        self.fold_chunk = int(fold_chunk)
        self.tracer = tracer
        self.d = int(solver.num_parameters)
        self.n_bins = int(solver.n_bins)
        # The XSPEC-backed property re-reads the spectrum on every access.
        x_obs = np.array(solver.observed_counts, dtype=np.float64).reshape(-1)
        if x_obs.shape != (self.n_bins,):
            raise ValueError(f"observed_counts has shape {x_obs.shape} but the solver reports n_bins={self.n_bins}.")
        x_obs.setflags(write=False)
        self.x_obs = x_obs
        self.seconds: dict[str, float] = {}

    # ------------------------------------------------------------------
    # public protocol
    # ------------------------------------------------------------------

    def cstat(self, u: Any, *, phase: str) -> np.ndarray:
        r"""Exact C-stat of unit-cube points, with $+\infty$ outside the cube.

        Rows outside the closed cube, or with a non-finite coordinate, are not
        folded, not counted by the solver and not traced; they get
        $C = +\infty$. The other rows are folded in chunks of ``fold_chunk``
        under ``solver.fold_phase(phase)``.

        Parameters:
            u:
                Unit-cube points of shape ``(n, d)``; an empty array gives an
                empty result without folding.
            phase:
                Fold-phase label, e.g. ``"screen"`` or ``"final"``.

        Returns:
            float64 array of shape ``(n,)``.

        Raises:
            ValueError: If *u* does not have shape ``(n, d)``.
            FoldError: If a fold returns ``NaN`` or $-\infty$, or a result of
                the wrong length.
        """
        u = self._as_points(u)
        out = np.full(u.shape[0], np.inf, dtype=np.float64)
        rows = np.flatnonzero(in_cube(u))
        if rows.size == 0:
            return out
        start = perf_counter()
        try:
            with self._scope(phase):
                for lo in range(0, rows.size, self.fold_chunk):
                    idx = rows[lo : lo + self.fold_chunk]
                    theta = to_physical(self._prior, u[idx])
                    result = self._solver._fold(theta, return_kind="cstat", progress_bar=False)
                    values = self._checked_cstat(result, theta, phase)
                    self._record(theta, values)
                    out[idx] = values
        finally:
            self._add_seconds(phase, perf_counter() - start)
        return out

    def counts(self, u: Any, *, phase: str, key: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        r"""Exact C-stat and a seeded Poisson realisation of the model counts.

        Every row must lie in the closed cube: a training spectrum outside the
        prior support has no meaning. Rows are folded in chunks of
        ``fold_chunk``; chunk ``j`` (rows ``j * fold_chunk`` to
        ``(j + 1) * fold_chunk``) is realised with
        ``child_rng(seed_seq, *key, j)``. The same *key* therefore reproduces
        the same counts, different keys give independent ones, and the
        realisation also depends on ``fold_chunk``.

        Parameters:
            u:
                Unit-cube points of shape ``(n, d)``; an empty array gives
                empty results without folding.
            phase:
                Fold-phase label, e.g. ``"harvest"`` or ``"training"``.
            key:
                Integers in $[0, 2^{32})$ naming the Poisson streams, e.g.
                ``(stage, round)``.

        Returns:
            ``(cstat, counts)``: float64 of shape ``(n,)`` and float32 of
            shape ``(n, n_bins)``.

        Raises:
            ValueError: If *u* does not have shape ``(n, d)`` or a row is
                outside the closed cube.
            TypeError: If a key is not an integer.
            FoldError: If a fold returns ``NaN`` or $-\infty$ C-stat,
                non-finite model counts, or outputs of the wrong shape.
        """
        key = tuple(key)
        u = self._as_points(u)
        n = u.shape[0]
        cstat = np.empty(n, dtype=np.float64)
        counts = np.empty((n, self.n_bins), dtype=np.float32)
        if n == 0:
            return cstat, counts
        inside = in_cube(u)
        if not np.all(inside):
            bad = int(np.count_nonzero(~inside))
            raise ValueError(
                f"counts() needs every row inside the closed unit cube; {bad} of {n} rows are outside it or non-finite."
            )
        # Validate the key before any fold is spent on it.
        child_rng(self.seed_seq, *key, 0)
        start = perf_counter()
        try:
            with self._scope(phase):
                for j, lo in enumerate(range(0, n, self.fold_chunk)):
                    hi = min(lo + self.fold_chunk, n)
                    theta = to_physical(self._prior, u[lo:hi])
                    result = self._solver._fold(theta, return_kind="full_model_counts", progress_bar=False)
                    values = self._checked_cstat(result, theta, phase)
                    mu = self._checked_model_counts(result, theta, phase)
                    self._record(theta, values)
                    cstat[lo:hi] = values
                    rng = child_rng(self.seed_seq, *key, j)
                    counts[lo:hi] = rng.poisson(np.clip(mu, 0.0, None)).astype(np.float32)
        finally:
            self._add_seconds(phase, perf_counter() - start)
        return cstat, counts

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _as_points(self, u: Any) -> np.ndarray:
        """Return *u* as float64 of shape ``(n, d)``.

        Parameters:
            u:
                Array-like of shape ``(n, d)``, or an empty 1-D array.

        Returns:
            A float64 array of shape ``(n, d)``.

        Raises:
            ValueError: If the shape is not ``(n, d)``.
        """
        arr = np.asarray(u, dtype=np.float64)
        if arr.ndim == 1 and arr.size == 0:
            arr = arr.reshape(0, self.d)
        if arr.ndim != 2 or arr.shape[1] != self.d:
            raise ValueError(f"u must have shape (n, {self.d}), got {arr.shape}.")
        return arr

    @contextlib.contextmanager
    def _scope(self, phase: str):
        """Enter the solver's pool scope (if any) and the fold phase *phase*."""
        pool_scope = getattr(self._solver, "pool_scope", None)
        with contextlib.ExitStack() as stack:
            if callable(pool_scope):
                stack.enter_context(pool_scope())
            stack.enter_context(self._solver.fold_phase(phase))
            yield

    def _checked_cstat(self, result: dict, theta: np.ndarray, phase: str) -> np.ndarray:
        r"""Extract the C-stat of one fold and refuse ``NaN`` and $-\infty$.

        Parameters:
            result:
                Output dict of ``solver._fold``.
            theta:
                The physical parameters folded, shape ``(m, d)``.
            phase:
                Fold-phase label, for the message.

        Returns:
            float64 array of shape ``(m,)``.

        Raises:
            FoldError: On a wrong length, ``NaN`` or $-\infty$.
        """
        values = np.asarray(result["cstat"], dtype=np.float64).reshape(-1)
        if values.shape[0] != theta.shape[0]:
            raise FoldError(
                f"fold in phase {phase!r} returned {values.shape[0]} C-stat values for {theta.shape[0]} parameter rows."
            )
        bad = np.isnan(values) | (values == -np.inf)
        if np.any(bad):
            first = int(np.flatnonzero(bad)[0])
            raise FoldError(
                f"fold in phase {phase!r} returned an invalid C-stat ({values[first]}) for "
                f"{int(bad.sum())} of {values.size} rows, e.g. at parameters {theta[first].tolist()}."
            )
        return values

    def _checked_model_counts(self, result: dict, theta: np.ndarray, phase: str) -> np.ndarray:
        """Extract the model counts of one fold and refuse bad shapes and non-finite values.

        Parameters:
            result:
                Output dict of ``solver._fold`` with ``"total_model_counts"``.
            theta:
                The physical parameters folded, shape ``(m, d)``.
            phase:
                Fold-phase label, for the message.

        Returns:
            float64 array of shape ``(m, n_bins)``.

        Raises:
            FoldError: If the key is missing, the shape is wrong or a value is
                not finite.
        """
        if "total_model_counts" not in result:
            raise FoldError(f"fold in phase {phase!r} returned no 'total_model_counts'.")
        mu = np.asarray(result["total_model_counts"], dtype=np.float64)
        expected = (theta.shape[0], self.n_bins)
        if mu.shape != expected:
            raise FoldError(f"fold in phase {phase!r} returned model counts of shape {mu.shape}, expected {expected}.")
        finite = np.isfinite(mu)
        if not np.all(finite):
            first = int(np.flatnonzero(~np.all(finite, axis=1))[0])
            raise FoldError(
                f"fold in phase {phase!r} returned non-finite model counts, e.g. at parameters {theta[first].tolist()}."
            )
        return mu

    def _record(self, theta: np.ndarray, values: np.ndarray) -> None:
        """Forward one folded chunk to the tracer, if any."""
        if self.tracer is not None:
            self.tracer.record(theta, values)

    def _add_seconds(self, phase: str, seconds: float) -> None:
        """Accumulate wall-clock time under *phase*."""
        self.seconds[phase] = self.seconds.get(phase, 0.0) + float(seconds)
