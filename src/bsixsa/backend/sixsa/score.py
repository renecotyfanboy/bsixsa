r"""Score compression: each spectrum becomes one number per parameter.

At a fiducial point $u_\star$ of the unit cube, with expected counts
$\mu = \mu(u_\star)$ and their derivatives $\partial_a \mu$, the Poisson score of
a spectrum $k$ and the Fisher matrix are

$$
s_a(k) = \sum_i \frac{k_i - \mu_i}{\mu_i}\, \partial_a \mu_i, \qquad
F_{ab} = \sum_i \frac{\partial_a \mu_i\, \partial_b \mu_i}{\mu_i},
$$

and the code of $k$ is $t(k) = u_\star + (F + P)^{-1} s(k)$: one Fisher-scoring
step from $u_\star$ towards the maximum likelihood of $k$ (Alsing & Wandelt
2018). Near $u_\star$ the code keeps all the information the spectrum carries
about the parameters, so the closer $u_\star$ is to the posterior, the sharper
the proposals. $P = 12\,I$ is the precision of the uniform prior in the cube:
with $F^{-1}$ alone, a direction the spectrum barely constrains (two
degenerate normalisations, say) would carry Poisson noise amplified by the
degeneracy; with $P$ its codes stay at the scale of the prior. The code is
still an invertible linear function of the score, so it keeps the same
information. The derivatives are central differences of exact folds; once
they are known, encoding is a matrix product, with no network to train.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from time import perf_counter
from typing import Any

import numpy as np

__all__ = ["ScoreCompressor"]

# Largest finite-difference step, in unit-cube coordinates; the ladder halves it.
_LARGEST_STEP = 1e-2
_SMALLEST_STEP = 1e-6
# A step is accepted when the derivative agrees with the one at twice the step to this
# fraction, in the Fisher norm.
_STEP_TOLERANCE = 0.02
# The Fisher correlation matrix is inverted down to this relative eigenvalue.
_RCOND = 1e-12
# Precision of the uniform prior on [0, 1], added to the Fisher matrix of the codes.
PRIOR_PRECISION = 12.0
# Step halvings tried by each Fisher-scoring iteration before the fit stops.
_MAX_HALVINGS = 8

# Exact C-stat (n,) and expected counts (n, n_bins) of cube points (n, d).
FoldFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


class ScoreCompressor:
    """Score compression at a fiducial point; `generation` counts the maps built.

    Parameters:
        fold: Exact C-stat and expected counts of cube points.
        observed: Observed counts, shape `(n_bins,)`.
        prior_precision: The $P$ added to the Fisher matrix of the codes, per axis; 0 gives $F^{-1}$.
        log: Receives the progress messages; silent by default.

    Attributes:
        u_star: Fiducial point in the unit cube, shape `(d,)`.
        cstat_star: C-stat of the observation at the fiducial.
        fisher: Fisher matrix in cube coordinates, shape `(d, d)`.
        sigma: Marginal Fisher standard deviation of each parameter, `sqrt(diag(F^-1))`.
        condition: Condition number of the Fisher correlation matrix.
        steps: Finite-difference step of each parameter.
        unstable: Whether the derivative of each parameter never settled on the step ladder.
        n_folds: Folds spent on the maps and the fit.
        history: One record per map built.
    """

    def __init__(self, fold: FoldFn, observed: Any, *, prior_precision: float = PRIOR_PRECISION,
                 log: Callable[[str], None] | None = None) -> None:
        self._fold = fold
        self.observed = np.asarray(observed, dtype=np.float64).reshape(-1)
        self.prior_precision = float(prior_precision)
        self._log = log or (lambda message: None)
        self.generation = 0
        self.n_folds = 0
        self.history: list[dict] = []
        self.u_star: np.ndarray | None = None
        self.cstat_star = math.inf
        self.mu_star: np.ndarray | None = None
        self.dmu: np.ndarray | None = None
        self.fisher: np.ndarray | None = None
        self.sigma: np.ndarray | None = None
        self.condition = math.nan
        self.steps: np.ndarray | None = None
        self.unstable: np.ndarray | None = None
        self._projection: np.ndarray | None = None

    def __repr__(self) -> str:
        return (f"ScoreCompressor(n_bins={self.observed.size}, cstat_star={self.cstat_star:.6g}, "
                f"condition={self.condition:.3g}, generation={self.generation})")

    # ---- folds ----

    def _folds(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cstat, mu = self._fold(u)
        self.n_folds += len(u)
        return np.asarray(cstat, dtype=np.float64).reshape(-1), np.asarray(mu, dtype=np.float64)

    def _differences(self, u: np.ndarray, axes: list[int], step: np.ndarray) -> dict[int, np.ndarray]:
        """Central differences of the expected counts along *axes*, one batch of folds.

        A step that would leave the cube is cut at its face, and the
        difference divided by the separation actually used.
        """
        rows = []
        for a in axes:
            for sign in (1.0, -1.0):
                row = u.copy()
                row[a] = min(max(u[a] + sign * step[a], 0.0), 1.0)
                rows.append(row)
        _, mu = self._folds(np.array(rows))
        return {a: (mu[2 * i] - mu[2 * i + 1]) / (rows[2 * i][a] - rows[2 * i + 1][a]) for i, a in enumerate(axes)}

    def _ladder(self, u: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Derivatives with a step per parameter: the largest that agrees with twice itself.

        The step is halved from `_LARGEST_STEP` until the derivative agrees with the one at
        twice the step to `_STEP_TOLERANCE` in the Fisher norm. A derivative that never
        settles (numerical noise, a parameter the spectrum barely responds to) takes the most
        consistent step tried and is flagged.
        """
        d = u.size

        def norm(x: np.ndarray) -> float:
            return float(np.sqrt(np.sum(x * x * weights)))

        step = np.full(d, _LARGEST_STEP)
        previous = self._differences(u, list(range(d)), step)
        dmu = np.zeros((d, weights.size))
        best = np.full(d, math.inf)
        chosen = np.zeros(d)
        pending = list(range(d))
        while pending and step[pending[0]] / 2 >= _SMALLEST_STEP:
            step[pending] /= 2
            current = self._differences(u, pending, step)
            still = []
            for a in pending:
                scale = max(norm(previous[a]), norm(current[a]), np.finfo(float).tiny)
                disagreement = norm(previous[a] - current[a]) / scale
                if disagreement < best[a]:
                    best[a], dmu[a], chosen[a] = disagreement, current[a], step[a]
                if disagreement > _STEP_TOLERANCE:
                    still.append(a)
            previous = current
            pending = still
        unstable = np.zeros(d, dtype=bool)
        unstable[pending] = True
        return dmu, chosen, unstable

    # ---- the map ----

    def build(self, u_star: Any, *, cstat: float | None = None, mu: Any = None, reuse_steps: bool = False) -> dict:
        """Compute the derivatives and the Fisher matrix at *u_star*; return the build record.

        Parameters:
            u_star: Fiducial point in the unit cube, shape `(d,)`.
            cstat: C-stat at *u_star*, if already folded (with *mu*).
            mu: Expected counts at *u_star*, if already folded.
            reuse_steps: Keep the steps of the previous map instead of running the ladder.
        """
        start, folds_before = perf_counter(), self.n_folds
        u_star = np.clip(np.asarray(u_star, dtype=np.float64).reshape(-1), 0.0, 1.0)
        if mu is None:
            c, m = self._folds(u_star[None, :])
            cstat, mu = float(c[0]), m[0]
        mu = np.asarray(mu, dtype=np.float64).reshape(-1)
        # A channel the model predicts no counts in carries no information at the fiducial.
        weights = np.divide(1.0, mu, out=np.zeros_like(mu), where=mu > 0)
        if reuse_steps and self.steps is not None:
            dmu_by_axis = self._differences(u_star, list(range(u_star.size)), self.steps)
            dmu = np.array([dmu_by_axis[a] for a in range(u_star.size)])
            steps, unstable = self.steps, self.unstable
        else:
            dmu, steps, unstable = self._ladder(u_star, weights)

        fisher = (dmu * weights) @ dmu.T
        responsive = np.diag(fisher) > 0
        fisher_inv, correlation = _inverse(fisher)
        eigenvalues = np.linalg.eigvalsh(correlation[np.ix_(responsive, responsive)])
        codes_inv, _ = _inverse(fisher + self.prior_precision * np.eye(u_star.size))

        self.u_star, self.cstat_star, self.mu_star = u_star, float(cstat), mu
        self.dmu, self.fisher, self.steps, self.unstable = dmu, fisher, steps, unstable
        self.sigma = np.where(responsive, np.sqrt(np.clip(np.diag(fisher_inv), 0.0, None)), np.inf)
        self.condition = (float(eigenvalues.max() / max(eigenvalues.min(), np.finfo(float).tiny))
                          if eigenvalues.size else math.nan)
        self._projection = codes_inv @ (dmu * weights)
        self.generation += 1
        record = {
            "generation": self.generation, "cstat_star": self.cstat_star, "condition": self.condition,
            "n_folds": self.n_folds - folds_before, "seconds": perf_counter() - start,
            "u_star": u_star.tolist(), "sigma": self.sigma.tolist(), "steps": steps.tolist(),
            "unstable": np.flatnonzero(unstable).tolist(),
        }
        self.history.append(record)
        self._log(f"[score] map {self.generation} at C-stat {self.cstat_star:,.2f}: {record['n_folds']} folds, "
                  f"Fisher correlation condition {self.condition:.3g}"
                  + (f", derivative unsettled for axes {record['unstable']}" if unstable.any() else ""))
        return dict(record)

    def fit(self, u_start: Any, *, max_iterations: int = 20, min_improvement: float = 0.1) -> dict:
        """Move the fiducial to the maximum likelihood of the observation by Fisher scoring, from *u_start*.

        Each iteration builds the map at the current point (the steps of the
        first map are reused), and steps to the observation's code, halving the
        step until the C-stat falls. The fit stops when an iteration gains less
        than *min_improvement* in C-stat, when no halving helps, or after
        *max_iterations*; the map is left at the best point.

        Returns:
            The record of the last map, with the C-stat at the start and the iterations.
        """
        start = perf_counter()
        self.build(u_start)
        cstat_start = self.cstat_star
        iterations = 0
        for _ in range(int(max_iterations)):
            direction = self._codes(self.observed[None, :])[0] - self.u_star
            for halving in range(_MAX_HALVINGS):
                candidate = np.clip(self.u_star + direction * 0.5**halving, 0.0, 1.0)
                cstat, mu = self._folds(candidate[None, :])
                if cstat[0] < self.cstat_star:
                    break
            else:
                break
            gain = self.cstat_star - float(cstat[0])
            self.build(candidate, cstat=float(cstat[0]), mu=mu[0], reuse_steps=True)
            iterations += 1
            if gain < min_improvement:
                break
        self._log(f"[score] fiducial fitted by Fisher scoring in {iterations} iterations: C-stat "
                  f"{cstat_start:,.2f} -> {self.cstat_star:,.2f}, {perf_counter() - start:.1f} s")
        return dict(self.history[-1], cstat_start=cstat_start, iterations=iterations)

    # ---- codes ----

    def _codes(self, counts: Any) -> np.ndarray:
        if self._projection is None:
            raise RuntimeError("the score compressor has no map yet: call build() or fit() first.")
        k = np.asarray(counts, dtype=np.float64)
        k = k.reshape(1, -1) if k.ndim == 1 else k
        return self.u_star + (k - self.mu_star) @ self._projection.T

    def encode(self, counts: Any) -> np.ndarray:
        """Codes `(n, d)` of the spectra `(n, n_bins)` (or of one spectrum `(n_bins,)`)."""
        return self._codes(counts).astype(np.float32)


def _inverse(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pseudo-inverse of a symmetric positive semi-definite matrix, and its correlation matrix.

    Inverted as a correlation matrix: the diagonal spans the parameters'
    scales, not their degeneracy, and would fool the eigenvalue cut-off. A
    row of zeros (a parameter the spectrum does not respond to) stays zero.
    """
    diagonal = np.diag(matrix)
    scale = np.sqrt(np.where(diagonal > 0, diagonal, 1.0))
    correlation = matrix / np.outer(scale, scale)
    return np.linalg.pinv(correlation, rcond=_RCOND, hermitian=True) / np.outer(scale, scale), correlation
