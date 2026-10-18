"""What every backend provides, and the shared Gaussian approximation of the two optimisers."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
from scipy.special import expit

from ..solver import FitResults
from .tracer import EvaluationTracer

if TYPE_CHECKING:
    from ..priors import MultipleIndependent
    from ..solver import Solver
    from .config import BackendConfig

LAPLACE_POSTERIOR_SAMPLES = 10_000
"""Draws of the Gaussian approximation stored in `FitResults` by the optimisers."""

# Clips of the prior CDF that keep its logit finite. They are asymmetric because a
# float near 0 keeps full relative precision (1e-25 is ~10 sigmas into a Gaussian
# prior's tail), while near 1 it cannot get closer than eps.
_CDF_LOW = 1e-25
_CDF_HIGH = 1.0 - np.finfo(np.float64).eps

_MAX_FRACTION_ON_A_BOUND = 0.01  # fraction of draws on a prior bound above which `run` warns


def traced_log_likelihood(solver: Solver, tracer: EvaluationTracer, theta: np.ndarray) -> np.ndarray:
    """Log-likelihood `-C/2` of each row of *theta* (shape `(d,)` or `(n, d)`), recorded by *tracer*."""
    theta = np.atleast_2d(theta)
    log_like = solver.log_likelihood_fn(theta, progress_bar=False)
    tracer.record(theta, -2.0 * log_like)
    return log_like


def random_rows(samples: np.ndarray, n: int, seed: int | None = None) -> np.ndarray:
    """Up to *n* distinct rows of *samples*, in random order (`seed=None` is unseeded)."""
    order = np.random.default_rng(seed).permutation(len(samples))
    return np.asarray(samples)[order[: int(n)]]


def _prior_cdf_clipped(prior: MultipleIndependent, theta: np.ndarray) -> np.ndarray:
    """Prior CDF of each parameter of one point, clipped so that its logit stays finite at the bounds."""
    cdf = prior.to_unit_cube(np.atleast_2d(theta)).ravel()
    return np.clip(cdf, _CDF_LOW, _CDF_HIGH)


def _on_a_prior_bound(cdf: np.ndarray) -> np.ndarray:
    """Whether each prior CDF value is on a prior bound (at either clip)."""
    return (cdf <= _CDF_LOW) | (cdf >= _CDF_HIGH)


class Backend(ABC):
    """A fitting method run on a solver's likelihood and prior.

    [`Solver.run`][bsixsa.solver.Solver.run] calls `run()` once, which returns a
    [`FitResults`][bsixsa.solver.FitResults], and keeps the backend as
    `solver.distributions["posterior"]`; `sample(n)` then returns up to *n*
    posterior draws in physical units.

    To add a backend, write a [`BackendConfig`][bsixsa.BackendConfig] subclass
    with a `backend_name`, a `Backend` subclass with that `config_cls`, and an
    entry in `bsixsa.backend.BACKEND_REGISTRY`.

    Parameters:
        solver: The solver to fit.
        config: The backend configuration, an instance of `config_cls`.

    Attributes:
        solver: The solver to fit.
        config: The backend configuration.
        tracer: The [`EvaluationTracer`][bsixsa.backend.tracer.EvaluationTracer] of the run.

    Raises:
        TypeError: If *config* is not an instance of `config_cls`.
        ValueError: If *config* does not fit the solver.
    """

    config_cls: ClassVar[type[BackendConfig]]

    def __init__(self, *, solver: Solver, config: BackendConfig) -> None:
        if not isinstance(config, self.config_cls):
            raise TypeError(
                f"{type(self).__name__} needs a configuration of type {self.config_cls.__name__}, "
                f"got {type(config).__name__}."
            )
        # Checked again: the XSPEC session may have changed since the solver was built.
        config.validate_for_solver(solver)
        self.solver = solver
        self.config = config
        self.tracer = EvaluationTracer(
            solver.outputfiles_basename,
            solver.parameter_names,
            enabled=config.trace,
            plot_every=config.plot_every,
            plot_step_percent=config.plot_step_percent,
        )

    def _log_likelihood(self, theta: np.ndarray) -> np.ndarray:
        """Traced log-likelihood `-C/2` of each row of *theta*."""
        return traced_log_likelihood(self.solver, self.tracer, theta)

    def _posterior_dataframe(self, samples: np.ndarray) -> pd.DataFrame:
        """One column per parameter."""
        return pd.DataFrame(samples, columns=self.solver.parameter_names)

    @abstractmethod
    def run(self) -> FitResults:
        """Fit the solver's data and return the results."""

    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        """Draw posterior samples after `run`, shape `(m, d)` with `m <= n`."""


class OptimiserBackend(Backend):
    """Best fit plus a Gaussian approximation of the posterior (iminuit, Levenberg-Marquardt).

    The posterior is approximated by a Gaussian around the maximum a posteriori
    point (Laplace approximation) in the logit of the prior CDF,
    `u = log(F(theta) / (1 - F(theta)))`, so every draw lies inside the prior.
    `run` warns when a parameter sits on a prior bound, where this
    approximation is meaningless.

    Attributes:
        best_fit_params: Best fit in physical units, `None` before `run`.
        best_fit_unconstrained: Best fit in logit-of-CDF space.
        covariance_unconstrained: Covariance in that space, or `None` (the
            draws then repeat the best fit).
    """

    def __init__(self, *, solver: Solver, config: BackendConfig) -> None:
        super().__init__(solver=solver, config=config)
        self.best_fit_params: np.ndarray | None = None
        self.best_fit_unconstrained: np.ndarray | None = None
        self.covariance_unconstrained: np.ndarray | None = None

    def _starting_point(self) -> np.ndarray | None:
        """`x0_physical`, a prior draw, or `None` for the prior median."""
        if self.config.x0_physical is not None:
            return self.config.x0_physical
        if self.config.init_from_prior:
            return self.solver.prior.sample(size=1).ravel()
        return None

    def _physical_to_unconstrained(self, physical: np.ndarray) -> np.ndarray:
        """Logit of the prior CDF of one physical point."""
        cdf = _prior_cdf_clipped(self.solver.prior, physical)
        return np.log(cdf / (1.0 - cdf))

    def _unconstrained_to_physical(self, unconstrained: np.ndarray) -> np.ndarray:
        """Physical parameters of logit-of-CDF points, shape `(n, d)`."""
        return self.solver.prior.from_unit_cube(expit(np.atleast_2d(unconstrained)))

    def sample(self, n: int) -> np.ndarray:
        """Draw *n* points of the Gaussian approximation, in physical units, shape `(n, d)`."""
        if self.covariance_unconstrained is None:
            draws = np.tile(self.best_fit_unconstrained, (n, 1))
        else:
            draws = np.random.default_rng().multivariate_normal(
                self.best_fit_unconstrained, self.covariance_unconstrained, size=n
            )
        return self._unconstrained_to_physical(draws)

    def _best_fit_stat(self) -> float:
        """C-stat at the best fit plus XSPEC's prior term, i.e. XSPEC's `Fit.statistic` with `Fit.bayes = "on"`.

        It differs from the optimiser's objective (SciPy's normalised prior
        densities) by a constant: same minimum, values not comparable.
        """
        best = self.best_fit_params
        cstat = self.solver.simulate(best[None, :], return_kind="cstat", progress_bar=False)["cstat"]
        return float(np.ravel(cstat)[0]) + self.solver.prior.xspec_bayes_contribution(best)

    def _fit_results(self, seconds: float, n_evaluations: int, best_fit_stat: float) -> FitResults:
        """Package the best fit and its Gaussian draws (NaN evidence); warn about parameters on a prior bound."""
        names = self.solver.parameter_names
        samples = self.sample(LAPLACE_POSTERIOR_SAMPLES)
        fit = FitResults(
            time=seconds,
            posterior_samples=self._posterior_dataframe(samples),
            n_likelihood_evaluations=int(n_evaluations),
            log_Z=float("nan"),
            log_Z_err=float("nan"),
            best_fit=pd.Series(self.best_fit_params, index=names),
            best_fit_stat=best_fit_stat,
        )
        # Optimisers stop just inside a bound, so pegged parameters are also
        # detected through their draws, which then pile up on the bounds.
        prior = self.solver.prior
        best_fit_on_a_bound = _on_a_prior_bound(prior.to_unit_cube(np.atleast_2d(self.best_fit_params)).ravel())
        draws_on_a_bound = _on_a_prior_bound(prior.to_unit_cube(samples)).mean(axis=0) > _MAX_FRACTION_ON_A_BOUND
        for name in np.asarray(names)[best_fit_on_a_bound | draws_on_a_bound]:
            warnings.warn(
                f"{name} is at its prior bound, or much closer to it than its uncertainty: its Gaussian "
                "posterior approximation is not meaningful (the draws pile up on the bounds). "
                "Widen the prior or use a sampler.",
                UserWarning,
                # Points at the caller of Solver.run.
                stacklevel=4,
            )
        return fit
