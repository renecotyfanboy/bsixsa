import typing

import numpy as np
import pandas as pd
from nautilus import Prior, Sampler as NestedSampler

from ..convenience import iter_thawn_parameters, available_cpu_count
from .abc import Backend
from . import register_backend
from .config import Nautilus
from ..solver import FitResults


if typing.TYPE_CHECKING:
    from ..solver import Solver


@register_backend
class NautilusBackend(Backend):

    name = "nautilus"
    config_cls = Nautilus

    def __init__(
        self,
        *,
        solver: 'Solver',
        config: Nautilus,
    ):
        super().__init__(solver=solver, config=config)

        prior = Prior()

        for par, dist in zip(iter_thawn_parameters(), self.solver.prior.dists):
            prior.add_parameter(par.name, dist)

        self.n_live_points = config.num_live_points

        def likelihood(x):
            ll = solver.log_likelihood_fn(x, None, progress_bar=False, no_pool=False)
            self.tracer.record(np.atleast_2d(x), -2.0 * np.atleast_1d(ll))
            return ll

        n_pool = min(config.n_pool, available_cpu_count())

        self._nested_sampler = NestedSampler(
            prior, likelihood,
            n_live=config.num_live_points,
            vectorized=True,
            pass_dict=False,
            n_batch=config.n_batch,
            n_networks=config.n_networks,
            pool=(None, n_pool),
        )

    def run(self, **kwargs) -> 'FitResults':
        from ..solver import FitResults
        from ..convenience import catchtime

        with catchtime("Running nested sampler", print_time=False) as run_time:
            self._nested_sampler.run(verbose=True, discard_exploration=True)

        return FitResults(
            time=float(run_time()),
            posterior_samples=pd.DataFrame.from_dict(
                self._nested_sampler.posterior(equal_weight=True)[0]
            ),
            n_likelihood_evaluations=self._nested_sampler.n_like,
            log_Z=float(self._nested_sampler.log_z),
            log_Z_err=float(1 / np.sqrt(self._nested_sampler.n_eff)),
        )

    def sample(self, n: int) -> np.ndarray:
        samples = self._nested_sampler.posterior(equal_weight=True)[0]
        size = min(n, samples.shape[0])
        return samples[:size]
