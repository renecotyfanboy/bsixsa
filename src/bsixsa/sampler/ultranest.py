import numpy as np
from .abc import Sampler
import typing
import ultranest
import pandas as pd
from ..convenience import iter_thawn_parameters

if typing.TYPE_CHECKING:
    from ..solver import SIXSASolver


class UltranestSampler(Sampler):

    def __init__(self, *, solver:'SIXSASolver', n_live_points:int=1_000, resume=True, **kwargs):

        super().__init__(solver=solver)

        self.sampler = ultranest.ReactiveNestedSampler(
            [par.name for par in iter_thawn_parameters()],
            self.likelihood,
            transform=lambda cube: self.prior_transform(cube),
            log_dir=self.solver.outputfiles_basename,
            vectorized=True,
            resume=resume
        )

        self.results = None

        self.sampler_kwargs = {"min_num_live_points" : n_live_points}

    def prior_transform(self, cube):
        params = np.asarray([dist.ppf(quantile) for quantile, dist in zip(cube.T, self.solver.prior.dists)]).T
        return params

    def likelihood(self, x):
        return self.solver.log_likelihood_fn(x, None, progress_bar=False, no_pool=False)

    def run(self):
        from ..solver import FitResults
        from ..convenience import catchtime
        with catchtime("Running nested sampler", print_time=False) as run_time:
            results = self.sampler.run(**self.sampler_kwargs)

        self.results = results

        posterior_dict = {name:results["samples"][:, i] for i, name in enumerate(self.solver.parameter_names)}

        self.solver.fit_result = FitResults(
            time=float(run_time()),
            posterior_samples=pd.DataFrame.from_dict(posterior_dict),
            n_likelihood_evaluations=results["ncall"],
            log_Z=float(results["logz"]),
            log_Z_err=float(results["logzerr"])
        )

        return results

    def sample(self, shape):

        samples = self.results["samples"]
        size = min(shape[0], samples.shape[0])
        return samples[:size]