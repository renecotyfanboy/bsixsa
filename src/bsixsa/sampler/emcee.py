"""
WORK IN PROGRESS
"""

import numpy as np
import emcee
from .abc import Sampler
import typing

if typing.TYPE_CHECKING:
    from ..solver import SIXSASolver

class EmceeSampler(Sampler):

    def __init__(self, *, solver:'SIXSASolver', n_live_points:int=1_000, **kwargs):

        super().__init__(solver=solver)

        def is_truthy(x):
            return bool(x) and isinstance(x[0], np.ndarray) and x[0].size > 0

        def log_prob(x):
            prior = solver.log_prior_fn(x, None)
            likelihood = np.zeros_like(prior)
            finite_idx = np.where(~np.isinf(prior))

            if is_truthy(finite_idx):
                likelihood[finite_idx] = solver.log_likelihood_fn(x[finite_idx], None, progress_bar=False,
                                                                  no_pool=False)

            log_prob = prior + likelihood

            return log_prob

        ndim, nwalkers = len(solver.parameter_names), 100
        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, vectorize=True)

    def run(self):

        p0 = live_points_to_array(sampler.posterior_samples, names=model.names, copy=True)[
            :nwalkers]  # np.median(, axis=0)[None, :] + np.random.normal(0, 1e-6, size=(nwalkers, ndim))

        emcee_sampler.run_mcmc(
            p0, 1000, progress=True
        )

        emcee_results = emcee_sampler.get_chain(flat=True)

    def sample(self, shape):

        samples = self.sampler.posterior(equal_weight=True)[0]
        size = min(shape[0], samples.shape[0])
        return samples[:size]
