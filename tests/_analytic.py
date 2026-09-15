r"""Analytic, XSPEC-free fixtures shared by the solver, restriction and SIXSA tests.

This is an importable helper module (not a test file). It provides:

- [`AnalyticSpectrum`][]: a 64-channel absorbed power law with a seeded
  Poisson realisation and an exact, vectorised Cash statistic;
- [`AnalyticSolver`][]: a [`Solver`][bsixsa.solver.Solver] whose folds are
  computed analytically through the overridable hooks, so that every solver
  code path (fold accounting, pool scopes, fingerprint) runs without XSPEC;
- [`QuadCstat`][]: a quadratic C-stat on the unit cube with closed-form level
  sets and evidence;
- [`GaussianProposal`][]: a raw (not cube-rejected) Gaussian proposal with its
  analytic in-cube mass;
- [`grid_truth`][]: reference evidence and posterior moments of an
  `AnalyticSpectrum` by tensor-grid quadrature in cube coordinates.

Conventions: the likelihood is $\mathcal{L} = e^{-C/2}$ and, in cube
coordinates $u = F(\theta)$, the prior density is $\pi_u = 1$ on $[0, 1]^d$.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Callable, Iterator, Sequence

import numpy as np
from scipy import linalg, optimize, special
from scipy.integrate import cumulative_trapezoid

try:  # pragma: no cover - depends on the local HEASoft install
    import xspec  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    from unittest.mock import MagicMock

    sys.modules["xspec"] = MagicMock()

from bsixsa._cube import latin_hypercube, to_physical  # noqa: E402
from bsixsa.priors import MultipleIndependent, loguniform, uniform  # noqa: E402
from bsixsa.solver import Solver  # noqa: E402

__all__ = [
    "NO_WORKER_POOL",
    "AnalyticSolver",
    "AnalyticSpectrum",
    "GaussianProposal",
    "QuadCstat",
    "check_grid_convergence",
    "grid_truth",
    "log_integral_grid",
    "marginal_cdf",
]

_LOG_2PI = math.log(2.0 * math.pi)


def _rng(rng) -> np.random.Generator:
    """Accept a `Generator`, a seed or `None`."""
    return np.random.default_rng(rng)


def _unit_ball_volume(d: int) -> float:
    r"""Volume $V_d = \pi^{d/2} / \Gamma(d/2 + 1)$ of the unit ball in $\mathbb{R}^d$."""
    return float(math.pi ** (d / 2.0) / special.gamma(d / 2.0 + 1.0))


# ---------------------------------------------------------------------------
# Analytic spectrum
# ---------------------------------------------------------------------------


class AnalyticSpectrum:
    r"""Absorbed power law folded on 64 log-spaced channels, with Poisson data.

    The expected counts in channel $i$ are

    $$
    \mu_i = A \, \mathrm{norm} \, E_i^{-\Gamma} \, \Delta E_i \,
        \exp\left(-0.4 \, n_H \, E_i^{-2.5}\right),
    $$

    with $E_i$ the channel centre and $\Delta E_i$ its width, on edges
    log-spaced between 0.5 and 10 keV. $A$ is set so that the expected total
    at the truth ($\Gamma = 1.8$, $\mathrm{norm} = 1$, $n_H = 0.3$) is 4000
    counts. For `d=2` the free parameters are `(Gamma, norm)` and $n_H$ is
    fixed to 0.3; for `d=3` they are `(Gamma, norm, nH)`.

    Parameters:
        d: Number of free parameters, 2 or 3.
        seed: Seed of the Poisson realisation of the observed counts.

    Raises:
        ValueError: If *d* is not 2 or 3.
    """

    n_channels = 64
    e_min = 0.5
    e_max = 10.0
    target_total_counts = 4000.0
    fixed_nH = 0.3

    def __init__(self, d: int = 2, seed: int = 12345):
        if d not in (2, 3):
            raise ValueError(f"AnalyticSpectrum supports d=2 or d=3, got {d}.")
        self.d = int(d)
        self.seed = int(seed)

        self.edges = np.geomspace(self.e_min, self.e_max, self.n_channels + 1)
        self.energy = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.width = np.diff(self.edges)
        self._log_energy = np.log(self.energy)
        self._absorption_shape = 0.4 * self.energy ** -2.5

        self.parameter_names = ["Gamma", "norm", "nH"][: self.d]
        self.truth = np.array([1.8, 1.0, self.fixed_nH][: self.d], dtype=np.float64)

        self.amplitude = 1.0
        self.amplitude = self.target_total_counts / float(self.model_counts(self.truth)[0].sum())
        self.mu_true = self.model_counts(self.truth)[0]
        self.observed = np.random.default_rng(self.seed).poisson(self.mu_true).astype(np.float64)

    @property
    def n_bins(self) -> int:
        """Number of channels."""
        return self.n_channels

    def model_counts(self, theta) -> np.ndarray:
        """Expected counts per channel.

        Parameters:
            theta: Physical parameters, shape `(n, d)` or `(d,)`.

        Returns:
            Array of shape `(n, 64)` (`(1, 64)` for a single vector).

        Raises:
            ValueError: If the last dimension of *theta* is not `d`.
        """
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        if theta.ndim != 2 or theta.shape[1] != self.d:
            raise ValueError(f"theta must have shape (n, {self.d}), got {theta.shape}.")
        gamma = theta[:, 0:1]
        norm = theta[:, 1:2]
        n_h = theta[:, 2:3] if self.d == 3 else self.fixed_nH
        return (
            self.amplitude
            * norm
            * np.exp(-gamma * self._log_energy)
            * self.width
            * np.exp(-n_h * self._absorption_shape)
        )

    @staticmethod
    def cash(mu, k) -> np.ndarray:
        r"""Cash statistic $C = 2 \sum_i \left(\mu_i - k_i + k_i \ln(k_i/\mu_i)\right)$.

        The term $k_i \ln(k_i / \mu_i)$ is taken as 0 where $k_i = 0$.

        Parameters:
            mu: Expected counts, shape `(n, n_bins)` or `(n_bins,)`.
            k: Observed counts, shape `(n_bins,)`.

        Returns:
            Array of shape `(n,)`, or a scalar for a 1-D *mu*.
        """
        mu = np.asarray(mu, dtype=np.float64)
        k = np.asarray(k, dtype=np.float64)
        positive = k > 0
        safe_k = np.where(positive, k, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(positive, k * np.log(safe_k / mu), 0.0)
        return 2.0 * np.sum(mu - k + term, axis=-1)

    def cstat(self, theta) -> np.ndarray:
        """Exact Cash statistic of the observed counts at *theta*.

        Parameters:
            theta: Physical parameters, shape `(n, d)` or `(d,)`.

        Returns:
            Array of shape `(n,)`.
        """
        return self.cash(self.model_counts(theta), self.observed)

    def prior_list(self) -> list:
        """Frozen prior per axis: `uniform(1, 3)`, `loguniform(0.1, 10)` and, for d=3, `uniform(0, 1.5)`."""
        dists = [uniform(1.0, 3.0), loguniform(0.1, 10.0)]
        if self.d == 3:
            dists.append(uniform(0.0, 1.5))
        return dists

    def prior(self) -> MultipleIndependent:
        """Joint prior built from `prior_list`."""
        return MultipleIndependent(self.prior_list())

    def cstat_u(self, u) -> np.ndarray:
        """Cash statistic in cube coordinates, $C(F^{-1}(u))$.

        Parameters:
            u: Cube coordinates, shape `(n, d)`; clipped to `[0, 1]`.

        Returns:
            Array of shape `(n,)`.
        """
        # Frozen scipy distributions are slow to build and the optimiser calls
        # this one row at a time, so the prior is built once per instance.
        prior = self.__dict__.get("_cube_prior")
        if prior is None:
            prior = self._cube_prior = self.prior()
        # Same u -> theta map as the restriction and SIXSA layers (float64,
        # clipped to the support), so the reference sees the production map.
        return self.cstat(to_physical(prior, u))


# ---------------------------------------------------------------------------
# Analytic solver
# ---------------------------------------------------------------------------


class _NoWorkerPool:
    """Stand-in pool returned by `AnalyticSolver._get_pool`; it has no workers.

    It is not `None`, so `Solver._fold` still takes its pooled branch, and any
    code that tried to dispatch work to it would fail loudly.
    """

    def __repr__(self) -> str:
        return "<analytic pool: no workers>"


NO_WORKER_POOL = _NoWorkerPool()


class AnalyticSolver(Solver):
    """`Solver` whose folds are evaluated analytically on an `AnalyticSpectrum`.

    Every XSPEC hook is overridden, and only the leaf folding hook
    `_fold_batch` is replaced, so the production `Solver._fold` (validation,
    empty batches, fold accounting, pool scopes) runs without XSPEC, as do the
    fingerprint and prior handling. The worker pool is never created:
    `_get_pool` returns the `NO_WORKER_POOL` sentinel, leaves `self.pool`
    unset and counts its calls in `n_pool_requests`. Each folded batch is
    appended to `fold_log` as a dict with keys `phase`, `n`, `return_kind`,
    `no_pool` and `pool_depth` (the pool-scope depth *before* the fold opened
    its own scope, so a positive value means an outer scope was already open).
    Empty batches are answered by `Solver._fold` and never reach the log.

    Parameters:
        spectrum: Analytic forward model and observed counts.
        backend: Backend configuration object.
        outputfiles_basename: Output directory.
        overwrite: Whether a non-empty output directory may be cleared.
        n_jobs: Nominal pool size (no pool is ever created).
        restrict_prior: Forwarded to `Solver._setup`.
    """

    def __init__(
        self,
        spectrum: AnalyticSpectrum,
        *,
        backend,
        outputfiles_basename,
        overwrite: bool = True,
        n_jobs: int = 1,
        restrict_prior=None,
    ):
        self.spectrum = spectrum
        self.n_pool_requests = 0
        self.fold_log: list[dict] = []
        self.nb_models = 1
        d = spectrum.d
        self._setup(
            prior=MultipleIndependent(spectrum.prior_list()),
            indexes=list(range(1, d + 1)),
            model_indexes=["analytic"] * d,
            outputfiles_basename=os.fspath(outputfiles_basename),
            overwrite=overwrite,
            n_jobs=n_jobs,
            backend=backend,
            restrict_prior=restrict_prior,
        )

    @property
    def parameter_names(self) -> list[str]:
        return list(self.spectrum.parameter_names)

    @property
    def observed_counts(self) -> np.ndarray:
        return np.array(self.spectrum.observed, dtype=np.float64)

    @property
    def observed_spectrum(self) -> np.ndarray:
        return np.asarray(self.spectrum.observed, dtype=np.float32)

    @property
    def n_bins(self) -> int:
        return int(self.spectrum.n_bins)

    def _check_xspec_state(self) -> None:
        return None

    def _xspec_constraints(self) -> dict:
        return {"n_spectra": 1, "stat_method": "cstat", "has_background": False}

    def _xspec_problem_state(self) -> dict:
        """Forward-model settings that change the likelihood but not the counts."""
        spectrum = self.spectrum
        return {
            "model": "analytic absorbed power law",
            "grid": [spectrum.e_min, spectrum.e_max, spectrum.n_channels],
            "amplitude": spectrum.amplitude,
            "fixed_nH": spectrum.fixed_nH if spectrum.d == 2 else None,
        }

    def _get_pool(self):
        self.n_pool_requests += 1
        return NO_WORKER_POOL

    def _fold_batch(self, parameters, *, return_kind, desc, progress_bar, pool) -> dict:
        """Vectorised analytic replacement of the `parallel_folding` call.

        Parameters:
            parameters: Validated physical parameters, shape `(n, d)`, `n >= 1`.
            return_kind: One of `"cstat"`, `"full_model_counts"`, or
                `"models_and_components"`.
            desc: Ignored.
            progress_bar: Ignored.
            pool: `NO_WORKER_POOL` for a pooled fold, `None` for a serial one.

        Returns:
            Dict with `"cstat"` `(n,)`, plus `"total_model_counts"` `(n, 64)`
            for `"full_model_counts"`, plus `"model_counts"`
            (`{"analytic": (n, 64)}`) and `"component_counts"` (`{}`) for
            `"models_and_components"`.
        """
        self.fold_log.append(
            {
                "phase": self._fold_phase_name,
                "n": int(len(parameters)),
                "return_kind": return_kind,
                "no_pool": pool is None,
                # A pooled fold runs inside the scope `_fold` just opened.
                "pool_depth": int(self._pool_depth) - (0 if pool is None else 1),
            }
        )
        return self._analytic_outputs(parameters, return_kind)

    def _analytic_outputs(self, parameters: np.ndarray, return_kind: str) -> dict:
        mu = self.spectrum.model_counts(parameters)
        outputs = {"cstat": self.spectrum.cash(mu, self.spectrum.observed)}
        if return_kind in ("full_model_counts", "models_and_components"):
            outputs["total_model_counts"] = mu
        if return_kind == "models_and_components":
            outputs["model_counts"] = {"analytic": mu.copy()}
            outputs["component_counts"] = {}
        return outputs


# ---------------------------------------------------------------------------
# Quadratic C-stat on the unit cube
# ---------------------------------------------------------------------------


class QuadCstat:
    r"""Quadratic C-stat on the unit cube, $C(u) = c_0 + (u - \mu)^\top P (u - \mu)$.

    With $\mathcal{L} = e^{-C/2}$ the unbounded posterior is Gaussian with
    mean $\mu$ and covariance $P^{-1}$. Level sets $\{C \le t\}$ are ellipsoids;
    the closed forms below ignore the cube boundary, so they are exact only
    when the relevant ellipsoid lies inside $[0, 1]^d$.

    Parameters:
        mu: Centre, shape `(d,)`.
        P: Symmetric positive-definite matrix, shape `(d, d)`.
        c0: Minimum of the statistic.

    Raises:
        ValueError: If *P* is not a symmetric positive-definite `(d, d)` matrix.
    """

    def __init__(self, mu, P, c0: float = 0.0):
        self.mu = np.asarray(mu, dtype=np.float64).ravel()
        self.P = np.atleast_2d(np.asarray(P, dtype=np.float64))
        self.d = self.mu.size
        if self.P.shape != (self.d, self.d):
            raise ValueError(f"P must have shape ({self.d}, {self.d}), got {self.P.shape}.")
        if not np.allclose(self.P, self.P.T):
            raise ValueError("P must be symmetric.")
        try:
            self._chol = np.linalg.cholesky(self.P)
        except np.linalg.LinAlgError as err:
            raise ValueError("P must be positive definite.") from err
        self.c0 = float(c0)
        self._logdet = 2.0 * float(np.sum(np.log(np.diag(self._chol))))

    def __call__(self, u) -> np.ndarray:
        return self.cstat(u)

    def cstat(self, u) -> np.ndarray:
        """Evaluate $C(u)$.

        Parameters:
            u: Points, shape `(n, d)` or `(d,)`.

        Returns:
            Array of shape `(n,)`, or a float for a 1-D *u*.
        """
        u = np.asarray(u, dtype=np.float64)
        delta = np.atleast_2d(u) - self.mu
        values = self.c0 + np.einsum("ni,ij,nj->n", delta, self.P, delta)
        return float(values[0]) if u.ndim == 1 else values

    @property
    def posterior_cov(self) -> np.ndarray:
        r"""Covariance $P^{-1}$ of the unbounded posterior."""
        return np.linalg.inv(self.P)

    def level_volume(self, t: float) -> float:
        r"""Volume of $\{C \le t\}$: $V_d (t - c_0)^{d/2} / \sqrt{\det P}$ (0 for $t \le c_0$)."""
        r2 = float(t) - self.c0
        if r2 <= 0.0:
            return 0.0
        return _unit_ball_volume(self.d) * r2 ** (self.d / 2.0) * math.exp(-0.5 * self._logdet)

    def aabb(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        r"""Axis-aligned bounding box of $\{C \le t\}$, not clipped to the cube.

        Returns:
            `(lo, hi)` with $\mu \mp \sqrt{(t - c_0)\,\mathrm{diag}(P^{-1})}$.
        """
        r2 = max(float(t) - self.c0, 0.0)
        half = np.sqrt(r2 * np.diag(self.posterior_cov))
        return self.mu - half, self.mu + half

    def sample_level_set(self, t: float, n: int, rng=None) -> np.ndarray:
        r"""Uniform draws inside the ellipsoid $\{C \le t\}$ (not clipped to the cube).

        Parameters:
            t: Level, must exceed $c_0$.
            n: Number of draws.
            rng: `Generator`, seed or `None`.

        Returns:
            Array of shape `(n, d)`.

        Raises:
            ValueError: If `t <= c0`.
        """
        r2 = float(t) - self.c0
        if r2 <= 0.0:
            raise ValueError("The level t must exceed c0.")
        gen = _rng(rng)
        z = gen.standard_normal((int(n), self.d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        z *= gen.random(int(n))[:, None] ** (1.0 / self.d)
        # With P = L L^T, x = L^{-T} z gives x^T P x = |z|^2.
        x = linalg.solve_triangular(self._chol.T, z.T, lower=False).T
        return self.mu + math.sqrt(r2) * x

    def log_evidence_unbounded(self) -> float:
        r"""$\log \int_{\mathbb{R}^d} e^{-C/2}\,du = -c_0/2 + \tfrac{d}{2}\log 2\pi - \tfrac12 \log\det P$."""
        return -0.5 * self.c0 + 0.5 * self.d * _LOG_2PI - 0.5 * self._logdet

    def posterior_sample(self, n: int, rng=None) -> np.ndarray:
        """Draws from the unbounded Gaussian posterior (not cube-rejected).

        Parameters:
            n: Number of draws.
            rng: `Generator`, seed or `None`.

        Returns:
            Array of shape `(n, d)`.
        """
        z = _rng(rng).standard_normal((int(n), self.d))
        return self.mu + linalg.solve_triangular(self._chol.T, z.T, lower=False).T


# ---------------------------------------------------------------------------
# Raw Gaussian proposal
# ---------------------------------------------------------------------------


class GaussianProposal:
    r"""Gaussian proposal following the raw protocol: draws are NOT cube-rejected.

    `log_prob_raw` is the untruncated Gaussian log-density, so importance
    weights built from it must count the draws that fall outside the cube.

    Parameters:
        mean: Mean, shape `(d,)`.
        cov: Covariance, shape `(d, d)`.
        name: Label.

    Raises:
        ValueError: If *cov* is not a positive-definite `(d, d)` matrix.
    """

    LEAKAGE_MC_DRAWS = 200_000
    LEAKAGE_MC_SEED = 20_260_915

    def __init__(self, mean, cov, name: str = "gauss"):
        self.mean = np.asarray(mean, dtype=np.float64).ravel()
        self.d = self.mean.size
        self.cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
        if self.cov.shape != (self.d, self.d):
            raise ValueError(f"cov must have shape ({self.d}, {self.d}), got {self.cov.shape}.")
        try:
            self._chol = np.linalg.cholesky(self.cov)
        except np.linalg.LinAlgError as err:
            raise ValueError("cov must be positive definite.") from err
        self._logdet = 2.0 * float(np.sum(np.log(np.diag(self._chol))))
        self.name = name

    @property
    def is_diagonal(self) -> bool:
        """Whether the covariance has exactly zero off-diagonal entries."""
        return not np.any(self.cov - np.diag(np.diag(self.cov)))

    def sample_raw(self, n: int, rng=None) -> np.ndarray:
        """Draw from the untruncated Gaussian.

        Parameters:
            n: Number of draws.
            rng: `Generator`, seed or `None`.

        Returns:
            Array of shape `(n, d)`, possibly outside the unit cube.
        """
        z = _rng(rng).standard_normal((int(n), self.d))
        return self.mean + z @ self._chol.T

    def log_prob_raw(self, u) -> np.ndarray:
        """Untruncated Gaussian log-density.

        Parameters:
            u: Points, shape `(n, d)` or `(d,)`.

        Returns:
            float64 array of shape `(n,)`, or a float for a 1-D *u*.
        """
        u = np.asarray(u, dtype=np.float64)
        delta = np.atleast_2d(u) - self.mean
        whitened = linalg.solve_triangular(self._chol, delta.T, lower=True)
        maha = np.sum(whitened**2, axis=0)
        values = -0.5 * (maha + self.d * _LOG_2PI + self._logdet)
        return float(values[0]) if u.ndim == 1 else values

    def leakage(self, method: str = "auto") -> float:
        r"""Mass $P(u \notin [0, 1]^d)$ that the raw proposal leaks out of the cube.

        Parameters:
            method: As in `in_cube_mass`.

        Returns:
            `1 - in_cube_mass(method)`.

        Raises:
            ValueError: As in `in_cube_mass`.
        """
        return 1.0 - self.in_cube_mass(method)

    def in_cube_mass(self, method: str = "auto") -> float:
        r"""In-cube mass $P(u \in [0, 1]^d)$ of the raw proposal.

        This is the normaliser of the cube-truncated proposal (sbi's
        "leakage correction"); the leaked mass is `leakage()`.

        Parameters:
            method: `"analytic"` (product of $\Phi$ differences, diagonal
                covariance only), `"mc"` (`LEAKAGE_MC_DRAWS` draws with the
                fixed `LEAKAGE_MC_SEED`), or `"auto"` (analytic when diagonal).

        Returns:
            Probability in `[0, 1]`.

        Raises:
            ValueError: On an unknown method, or `"analytic"` with a
                non-diagonal covariance.
        """
        if method == "auto":
            method = "analytic" if self.is_diagonal else "mc"
        if method == "analytic":
            if not self.is_diagonal:
                raise ValueError("The analytic in-cube mass requires a diagonal covariance.")
            sigma = np.sqrt(np.diag(self.cov))
            upper = special.ndtr((1.0 - self.mean) / sigma)
            lower = special.ndtr((0.0 - self.mean) / sigma)
            return float(np.prod(upper - lower))
        if method == "mc":
            draws = self.sample_raw(self.LEAKAGE_MC_DRAWS, np.random.default_rng(self.LEAKAGE_MC_SEED))
            inside = np.all((draws >= 0.0) & (draws <= 1.0), axis=1)
            return float(np.mean(inside))
        raise ValueError(f"Unknown in-cube mass method {method!r}.")


# ---------------------------------------------------------------------------
# Grid quadrature
# ---------------------------------------------------------------------------


def _tensor_grid(lo, hi, n_grid: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-axis nodes and log trapezoid weights."""
    lo = np.asarray(lo, dtype=np.float64).ravel()
    hi = np.asarray(hi, dtype=np.float64).ravel()
    if n_grid < 2:
        raise ValueError("n_grid must be at least 2.")
    if np.any(hi <= lo):
        raise ValueError("Every grid axis must have hi > lo.")
    axes, log_weights = [], []
    for a, b in zip(lo, hi):
        nodes = np.linspace(a, b, n_grid)
        weights = np.full(n_grid, (b - a) / (n_grid - 1))
        weights[[0, -1]] *= 0.5
        axes.append(nodes)
        log_weights.append(np.log(weights))
    return axes, log_weights


def _iter_grid(
    axes: Sequence[np.ndarray],
    log_weights: Sequence[np.ndarray],
    chunk: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield `(u, log_w)` chunks over the tensor grid in C (row-major) order."""
    shape = tuple(len(axis) for axis in axes)
    total = int(np.prod(shape))
    for start in range(0, total, chunk):
        multi = np.unravel_index(np.arange(start, min(start + chunk, total)), shape)
        u = np.stack([axis[index] for axis, index in zip(axes, multi)], axis=1)
        log_w = np.sum([lw[index] for lw, index in zip(log_weights, multi)], axis=0)
        yield u, log_w


def log_integral_grid(
    cstat: Callable[[np.ndarray], np.ndarray],
    lo,
    hi,
    n_grid: int,
    chunk: int = 1 << 16,
) -> float:
    r"""$\log \int_{[lo, hi]} e^{-C(u)/2}\,du$ by the tensor trapezoid rule.

    Parameters:
        cstat: Vectorised statistic, `(m, d) -> (m,)`.
        lo: Lower corner, shape `(d,)`.
        hi: Upper corner, shape `(d,)`.
        n_grid: Nodes per axis.
        chunk: Points evaluated per call of *cstat*.

    Returns:
        The log-integral.
    """
    axes, log_weights = _tensor_grid(lo, hi, n_grid)
    pieces = [
        special.logsumexp(-0.5 * np.asarray(cstat(u), dtype=np.float64) + log_w)
        for u, log_w in _iter_grid(axes, log_weights, chunk)
    ]
    return float(special.logsumexp(pieces))


_MAP_CACHE: dict[tuple[int, int], tuple[np.ndarray, float, np.ndarray]] = {}
_GRID_CACHE: dict[tuple[int, int, int], dict] = {}


def _fd_hessian(f: Callable[[np.ndarray], float], x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Central finite-difference Hessian with per-axis steps, kept inside the cube."""
    d = x.size
    x = np.clip(x, 2.0 * h, 1.0 - 2.0 * h)
    f0 = f(x)
    hess = np.empty((d, d))
    for i in range(d):
        ei = np.zeros(d)
        ei[i] = h[i]
        hess[i, i] = (f(x + ei) - 2.0 * f0 + f(x - ei)) / h[i] ** 2
        for j in range(i):
            ej = np.zeros(d)
            ej[j] = h[j]
            value = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4.0 * h[i] * h[j])
            hess[i, j] = hess[j, i] = value
    return hess


def _laplace_sigma(hessian: np.ndarray) -> np.ndarray:
    """Marginal standard deviations from the Hessian of C/2 (eigenvalues made positive)."""
    eigval, eigvec = np.linalg.eigh(hessian)
    eigval = np.maximum(np.abs(eigval), 1e-12)
    cov = (eigvec / eigval) @ eigvec.T
    return np.sqrt(np.diag(cov))


def _find_map(spectrum: AnalyticSpectrum) -> tuple[np.ndarray, float, np.ndarray]:
    """MAP in cube coordinates from 32 LHS starts, and Laplace marginal sigmas."""
    key = (spectrum.d, spectrum.seed)
    if key in _MAP_CACHE:
        map_u, c_map, sigma = _MAP_CACHE[key]
        return map_u.copy(), c_map, sigma.copy()

    d = spectrum.d

    def objective(u: np.ndarray) -> float:
        return float(spectrum.cstat_u(u[None, :])[0])

    starts = latin_hypercube(d, 32, np.random.default_rng(spectrum.seed))
    best = None
    for start in starts:
        result = optimize.minimize(objective, start, method="L-BFGS-B", bounds=[(0.0, 1.0)] * d)
        if best is None or result.fun < best.fun:
            best = result
    map_u = np.asarray(best.x, dtype=np.float64)
    c_map = float(best.fun)

    def half_cstat(u: np.ndarray) -> float:
        return 0.5 * objective(u)

    # Two passes: the second step is sized on the first estimate so that the
    # finite difference resolves the curvature whatever the posterior width.
    sigma = _laplace_sigma(_fd_hessian(half_cstat, map_u, np.full(d, 1e-4)))
    step = np.clip(0.2 * sigma, 1e-7, 1e-3)
    sigma = _laplace_sigma(_fd_hessian(half_cstat, map_u, step))

    _MAP_CACHE[key] = (map_u.copy(), c_map, sigma.copy())
    return map_u, c_map, sigma


def _weighted_moments(sum_w1: np.ndarray, sum_w2: np.ndarray, centre: np.ndarray):
    """Mean, std and correlation from centred first and second weighted sums."""
    mean_offset = sum_w1
    cov = sum_w2 - np.outer(mean_offset, mean_offset)
    std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    corr = cov / np.outer(std, std)
    return centre + mean_offset, std, corr


def grid_truth(spectrum: AnalyticSpectrum, n_grid: int | None = None) -> dict:
    r"""Reference evidence and posterior moments of *spectrum* by grid quadrature.

    Works in cube coordinates $u = F(\theta)$ under the spectrum's prior, where
    $\pi_u = 1$ and $\mathcal{L} = e^{-C/2}$, so
    $Z = \int_{[0,1]^d} e^{-C(u)/2}\,du$. The MAP is found by L-BFGS-B from 32
    Latin-hypercube starts; a finite-difference Hessian of $C/2$ gives the
    Laplace marginal widths $\sigma_u$, and the grid spans
    $\mathrm{MAP} \pm 10\,\sigma_u$ clipped to the cube. Results are cached
    per `(d, n_grid, seed)`; a shallow copy is returned.

    Parameters:
        spectrum: The analytic problem.
        n_grid: Nodes per axis; defaults to 512 for d=2 and 128 for d=3.

    Returns:
        Dict with keys `log_z`, `mean_u`, `std_u`, `corr_u`, `mean_theta`,
        `std_theta`, `corr_theta`, `c_min`, `map_u`, `map_theta`,
        `hpd_density` and `hpd_cstat` (dicts keyed by `0.68` and `0.95`:
        thresholds on the normalised cube density, and the equivalent
        C-stat levels $C \le c$), `sigma_u_laplace`, `window_lo`,
        `window_hi`, `grid_axes_u` and `marginal_cdf_u` (per-axis node CDFs,
        see `marginal_cdf`), `n_grid`, `d` and `seed`.
    """
    d = spectrum.d
    n_grid = int(n_grid if n_grid is not None else (512 if d == 2 else 128))
    key = (d, n_grid, spectrum.seed)
    if key in _GRID_CACHE:
        return dict(_GRID_CACHE[key])

    prior = spectrum.prior()
    map_u, c_map, sigma_u = _find_map(spectrum)
    lo = np.clip(map_u - 10.0 * sigma_u, 0.0, 1.0)
    hi = np.clip(map_u + 10.0 * sigma_u, 0.0, 1.0)
    axes, log_weights = _tensor_grid(lo, hi, n_grid)
    chunk = 1 << 16

    cstat_grid = np.concatenate([spectrum.cstat_u(u) for u, _ in _iter_grid(axes, log_weights, chunk)])
    log_w_grid = np.concatenate([log_w for _, log_w in _iter_grid(axes, log_weights, chunk)])
    log_z = float(special.logsumexp(-0.5 * cstat_grid + log_w_grid))
    mass = np.exp(-0.5 * cstat_grid + log_w_grid - log_z)

    map_theta = prior.from_unit_cube(map_u[None, :])[0]
    s1_u = np.zeros(d)
    s2_u = np.zeros((d, d))
    s1_t = np.zeros(d)
    s2_t = np.zeros((d, d))
    offset = 0
    for u, _ in _iter_grid(axes, log_weights, chunk):
        m = mass[offset : offset + len(u)]
        offset += len(u)
        du = u - map_u
        dt = prior.from_unit_cube(u) - map_theta
        s1_u += m @ du
        s2_u += (du * m[:, None]).T @ du
        s1_t += m @ dt
        s2_t += (dt * m[:, None]).T @ dt
    mean_u, std_u, corr_u = _weighted_moments(s1_u, s2_u, map_u)
    mean_theta, std_theta, corr_theta = _weighted_moments(s1_t, s2_t, map_theta)

    # Refine the minimum from the best grid node.
    best_node = np.array(np.unravel_index(int(np.argmin(cstat_grid)), (n_grid,) * d))
    best_u = np.array([axes[i][best_node[i]] for i in range(d)])
    refined = optimize.minimize(
        lambda x: float(spectrum.cstat_u(x[None, :])[0]),
        best_u,
        method="L-BFGS-B",
        bounds=[(0.0, 1.0)] * d,
    )
    candidates = [(float(cstat_grid.min()), best_u), (float(refined.fun), np.asarray(refined.x)), (c_map, map_u)]
    c_min, best_map_u = min(candidates, key=lambda item: item[0])

    # HPD: accumulate cell masses by decreasing density (increasing C).
    order = np.argsort(cstat_grid, kind="stable")
    cumulative = np.cumsum(mass[order])
    hpd_cstat, hpd_density = {}, {}
    for level in (0.68, 0.95):
        index = min(int(np.searchsorted(cumulative, level)), len(order) - 1)
        c_level = float(cstat_grid[order[index]])
        hpd_cstat[level] = c_level
        hpd_density[level] = float(np.exp(-0.5 * c_level - log_z))

    mass_grid = mass.reshape((n_grid,) * d)
    marginal_cdfs = []
    for i in range(d):
        node_mass = mass_grid.sum(axis=tuple(j for j in range(d) if j != i))
        density = node_mass / np.exp(log_weights[i])
        cdf = cumulative_trapezoid(density, axes[i], initial=0.0)
        marginal_cdfs.append(cdf / cdf[-1])

    result = {
        "log_z": log_z,
        "mean_u": mean_u,
        "std_u": std_u,
        "corr_u": corr_u,
        "mean_theta": mean_theta,
        "std_theta": std_theta,
        "corr_theta": corr_theta,
        "c_min": float(c_min),
        "map_u": np.asarray(best_map_u, dtype=np.float64),
        "map_theta": prior.from_unit_cube(np.asarray(best_map_u)[None, :])[0],
        "hpd_density": hpd_density,
        "hpd_cstat": hpd_cstat,
        "sigma_u_laplace": sigma_u,
        "window_lo": lo,
        "window_hi": hi,
        "grid_axes_u": axes,
        "marginal_cdf_u": marginal_cdfs,
        "n_grid": n_grid,
        "d": d,
        "seed": spectrum.seed,
    }
    _GRID_CACHE[key] = result
    return dict(result)


def marginal_cdf(truth: dict, axis: int, u) -> np.ndarray:
    """Marginal posterior CDF along one cube axis, interpolated from `grid_truth`.

    Parameters:
        truth: Output of `grid_truth`.
        axis: Parameter index.
        u: Cube coordinates along *axis*.

    Returns:
        CDF values, 0 below and 1 above the grid window.
    """
    return np.interp(
        np.asarray(u, dtype=np.float64),
        truth["grid_axes_u"][axis],
        truth["marginal_cdf_u"][axis],
        left=0.0,
        right=1.0,
    )


def check_grid_convergence(spectrum: AnalyticSpectrum, n_grid: int | None = None, tol: float = 1e-3) -> dict:
    """Compare `grid_truth` at `n_grid // 2` and `n_grid` nodes per axis.

    Both grids share the same MAP-centred window, so the difference isolates
    the quadrature error.

    Parameters:
        spectrum: The analytic problem.
        n_grid: Fine grid size; defaults as in `grid_truth`.
        tol: Tolerance on the absolute log-evidence difference.

    Returns:
        Dict with `log_z_fine`, `log_z_coarse`, `delta` and `ok`.
    """
    fine = grid_truth(spectrum, n_grid)
    coarse = grid_truth(spectrum, fine["n_grid"] // 2)
    delta = abs(fine["log_z"] - coarse["log_z"])
    return {
        "log_z_fine": fine["log_z"],
        "log_z_coarse": coarse["log_z"],
        "delta": float(delta),
        "ok": bool(delta <= tol),
    }
