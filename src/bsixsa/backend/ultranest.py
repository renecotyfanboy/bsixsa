import os
import typing

import numpy as np
import pandas as pd
import ultranest
import ultranest.popstepsampler
import ultranest.calibrator

from ..convenience import iter_thawn_parameters
from .abc import Backend
from . import register_backend
from .config import Ultranest
from ..solver import FitResults

if typing.TYPE_CHECKING:
    from ..solver import Solver


@register_backend
class UltranestBackend(Backend):

    name = "ultranest"
    config_cls = Ultranest

    def __init__(
        self,
        *,
        solver: 'Solver',
        config: Ultranest,
    ):
        super().__init__(solver=solver, config=config)

        ultranest_output = os.path.join(self.solver.outputfiles_basename, "ultranest_outputs")
        os.makedirs(ultranest_output, exist_ok=True)

        self._reactive_sampler = ultranest.ReactiveNestedSampler(
            [par.name for par in iter_thawn_parameters()],
            self._likelihood,
            transform=lambda cube: self._prior_transform(cube),
            log_dir=ultranest_output,
            vectorized=True,
            resume=self.config.resume
        )

        if self.config.use_step_sampler:
            popsize = 100
            nsteps = 50
            self._reactive_sampler.stepsampler = ultranest.popstepsampler.PopulationSimpleSliceSampler(
                popsize=popsize,
                nsteps=nsteps,
                scale=1,
                generate_direction=ultranest.popstepsampler.generate_region_oriented_direction,
            )

        self._results = None
        self._run_kwargs = {"min_num_live_points": self.config.num_live_points}

    def _prior_transform(self, cube):
        return self._cube_to_physical(cube)

    def _likelihood(self, x):
        ll = self.solver.log_likelihood_fn(x, None, progress_bar=False, no_pool=False)
        self.tracer.record(np.atleast_2d(x), -2.0 * np.atleast_1d(ll))
        return ll

    def run(self, **kwargs) -> 'FitResults':
        from ..solver import FitResults
        from ..convenience import catchtime

        with catchtime("Running nested sampler", print_time=False) as run_time:
            results = self._reactive_sampler.run(**self._run_kwargs)

        self._reactive_sampler.plot()
        self._results = results

        posterior_dict = {
            name: results["samples"][:, i]
            for i, name in enumerate(self.solver.parameter_names)
        }

        return FitResults(
            time=float(run_time()),
            posterior_samples=pd.DataFrame.from_dict(posterior_dict),
            n_likelihood_evaluations=results["ncall"],
            log_Z=float(results["logz"]),
            log_Z_err=float(results["logzerr"]),
        )

    def sample(self, n: int) -> np.ndarray:
        samples = self._results["samples"]
        size = min(n, samples.shape[0])
        return samples[:size]
