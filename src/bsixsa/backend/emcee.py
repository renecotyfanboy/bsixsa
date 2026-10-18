"""Ensemble MCMC with emcee."""

from __future__ import annotations

import time

import emcee
import numpy as np

from ..solver import FitResults
from .abc import Backend, random_rows
from .config import Emcee


class EmceeBackend(Backend):
    """Ensemble MCMC with emcee (half differential-evolution, half stretch moves).

    Attributes:
        sampler: The `emcee.EnsembleSampler` after `run`, for emcee's diagnostics.
    """

    config_cls = Emcee

    def log_prob(self, theta: np.ndarray) -> np.ndarray:
        """Log posterior `-C/2 + log prior` (up to a constant) of each walker, shape `(n_walkers,)`.

        Walkers outside the prior are not folded.
        """
        log_prior = self.solver.log_prior_fn(theta)
        inside = np.isfinite(log_prior)
        log_like = np.full(len(theta), -np.inf)
        log_like[inside] = self._log_likelihood(theta[inside])
        # Outside the prior: the most negative finite float, so that a move outside is
        # always rejected. A NaN statistic (failed evaluation): -inf, never accepted.
        with np.errstate(all="ignore"):
            return np.nan_to_num(
                log_like + log_prior, nan=-np.inf, posinf=-np.inf, neginf=np.finfo(float).min
            )

    def run(self) -> FitResults:
        """Run the chain and return the steps after the burn-in as the posterior."""
        config = self.config
        d = self.solver.num_parameters
        folds_before = sum(self.solver.fold_counts.values())
        self.sampler = emcee.EnsembleSampler(
            config.num_walkers,
            d,
            self.log_prob,
            vectorize=True,
            moves=[(emcee.moves.DEMove(), 0.5), (emcee.moves.StretchMove(), 0.5)],
        )
        x0 = config.x0_physical if config.x0_physical is not None else self.solver.prior.sample(size=1).ravel()
        walkers = x0 * np.random.normal(loc=1, scale=config.init_spread, size=(config.num_walkers, d))

        start = time.perf_counter()
        self.sampler.run_mcmc(walkers, config.num_warmup + config.num_samples, progress=True)
        seconds = time.perf_counter() - start

        self._chain = self.sampler.get_chain(discard=config.num_warmup, flat=True)
        return FitResults(
            time=seconds,
            posterior_samples=self._posterior_dataframe(self._chain),
            n_likelihood_evaluations=sum(self.solver.fold_counts.values()) - folds_before,
            log_Z=float("nan"),
            log_Z_err=float("nan"),
        )

    def sample(self, n: int) -> np.ndarray:
        """Up to *n* random rows of the chain after burn-in (not its first, most correlated steps)."""
        return random_rows(self._chain, n)
