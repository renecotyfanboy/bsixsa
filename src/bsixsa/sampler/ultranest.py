import numpy as np
from .abc import Sampler
import typing
import ultranest


if typing.TYPE_CHECKING:
    from ..solver import SIXSASolver


class UltranestSampler(Sampler):

    def __init__(self, *, solver:'SIXSASolver', n_live_points:int=1_000, resume=True, **kwargs):

        super().__init__(solver=solver)

        self.sampler = ultranest.ReactiveNestedSampler(
            [solver.parameter_names[i] for i in solver.indexes],
            self.likelihood,
            transform=self.prior_transform,
            log_dir=self.solver.outputfiles_basename,
            vectorized=True
        )

        self.results = None

        self.sampler_kwargs = {"min_num_live_points" : n_live_points, "resume":resume}

    def prior_transform(self, cube):
        params = np.asarray([dist.ppf(quantile) for quantile, dist in zip(cube.T, self.solver.prior.dists)]).T
        return params

    def likelihood(self, x):
        return self.solver.log_likelihood_fn(x, None, progress_bar=False, no_pool=False)

    def run(self):

        results = self.sampler.run(**self.sampler_kwargs)
        self.results = results
        return results

    def sample(self, shape):

        samples = self.results["samples"]
        size = min(shape[0], samples.shape[0])
        return samples[:size]