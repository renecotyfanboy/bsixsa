"""CMA-ES backend using *pycma*."""

from __future__ import annotations

import typing

import emcee
import numpy as np
import pandas as pd

from . import register_backend
from .abc import Backend
from .config import Emcee
from ..solver import FitResults

if typing.TYPE_CHECKING:
    from ..solver import SIXSASolver


@register_backend
class EmceeBackend(Backend):

    name = "emcee"
    config_cls = Emcee

    def __init__(
        self,
        *,
        solver: SIXSASolver,
        config: Emcee,
    ):
        super().__init__(solver=solver, config=config)


    def log_prob(self, parameters):

        log_prior = self.solver.log_prior_fn(parameters, None)
        in_prior_parameters = np.isfinite(log_prior)
        log_likelihood = -np.inf * np.empty_like(log_prior)
        log_likelihood[in_prior_parameters] = self.solver.log_likelihood_fn(
            parameters[in_prior_parameters], None, progress_bar=False
        )

        self.tracer.record(parameters, -2 * log_likelihood)

        with np.errstate(all='ignore'):
            log_prob = np.nan_to_num(log_likelihood + log_prior, nan=-np.inf, posinf=-np.inf)

        return log_prob

    def run(self) -> FitResults:

        from ..convenience import catchtime

        num_warmup = self.config.num_warmup
        num_samples = self.config.num_samples
        num_walkers = self.config.num_walkers
        x0_physical = self.config.x0_physical
        init_from_prior = self.config.init_from_prior
        init_spread = self.config.init_spread

        self.sampler = emcee.EnsembleSampler(
            num_walkers,
            self.solver.num_parameters,
            self.log_prob,
            vectorize=True,
            moves=[
                (emcee.moves.DEMove(), 0.5),
                (emcee.moves.StretchMove(), 0.5),
            ]
        )

        if x0_physical is not None:
            x0 = np.asarray(x0_physical, dtype=np.float64)
        elif init_from_prior:
            x0 = self.solver.prior.sample(size=1).ravel()
        else:
            raise ValueError("Either x0_physical or init_from_prior must be specified.")

        x0 = np.asarray(x0, dtype=np.float64).ravel()
        x0 = x0 * np.random.normal(loc=1, scale=init_spread, size=(num_walkers, self.solver.num_parameters))

        with catchtime("Run emcee", print_time=False) as run_time:
            self.sampler.run_mcmc(x0, num_warmup + num_samples, progress=True)

        posterior_samples = self.sampler.get_chain(discard=num_warmup, thin=1, flat=True)
        posterior_dict = {
            name: posterior_samples[:, i]
            for i, name in enumerate(self.solver.parameter_names)
        }

        self.fit_result = FitResults(
            time=float(run_time()),
            posterior_samples=pd.DataFrame.from_dict(posterior_dict),
            n_likelihood_evaluations=int(self.tracer.n_evals),
            log_Z=float("nan"),
            log_Z_err=float("nan"),
        )

        return self.fit_result

    def sample(self, n: int) -> np.ndarray:
        samples = self.fit_result.posterior_samples
        size = min(n, samples.shape[0])
        return samples[:size]
