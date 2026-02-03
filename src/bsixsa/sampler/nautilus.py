import numpy as np
from .abc import Sampler
from nautilus import Prior, Sampler as NestedSampler


class NautilusSampler(Sampler):

    def __init__(self, *, solver:'SIXSASolver', n_live_points:int=1_000, **kwargs):

        super().__init__(solver=solver)

        prior = Prior()

        for index, dist in zip(self.solver.indexes, self.solver.prior.dists):
            prior.add_parameter(self.solver.parameter_names[index], dist)

        self.n_live_points = n_live_points

        def likelihood(x):
            return solver.log_likelihood_fn(x, None, progress_bar=False, no_pool=False)

        self.sampler = NestedSampler(prior, likelihood, n_live=n_live_points, vectorized=True, pass_dict=False)

    def run(self):

        self.sampler.run(verbose=True)
        return self.sampler

    def sample(self, shape):

        samples = self.sampler.posterior(equal_weight=True)[0]
        size = min(shape[0], samples.shape[0])
        return samples[:size]