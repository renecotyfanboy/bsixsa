"""Nested sampling with nautilus, which learns the likelihood's shape with neural networks."""

from __future__ import annotations

import time
from multiprocessing import Pool
from typing import TYPE_CHECKING

import numpy as np
from nautilus import Prior
from nautilus import Sampler as NestedSampler

from ..convenience import available_cpu_count
from ..solver import FitResults
from .abc import Backend, random_rows
from .config import Nautilus

if TYPE_CHECKING:
    from ..solver import Solver


class NautilusBackend(Backend):
    """Neural-network-boosted nested sampling with nautilus; returns the equal-weight posterior and log Z."""

    config_cls = Nautilus

    def __init__(self, *, solver: Solver, config: Nautilus) -> None:
        super().__init__(solver=solver, config=config)
        self._prior = Prior()
        # The solver's working prior, which a restriction may have boxed.
        for name, dist in zip(solver.parameter_names, solver.prior.dists):
            self._prior.add_parameter(name, dist)
        self._nested_sampler = None

    def run(self) -> FitResults:
        """Run nautilus and return its equal-weight posterior and log evidence."""
        config = self.config
        n_processes = min(config.n_pool, available_cpu_count())
        # nautilus never shuts down a pool, even its own, so the backend owns it.
        sampler_pool = Pool(n_processes) if n_processes > 1 else None
        try:
            self._nested_sampler = NestedSampler(
                self._prior,
                self._log_likelihood,
                n_live=config.num_live_points,
                vectorized=True,
                pass_dict=False,
                n_batch=config.n_batch,
                n_networks=config.n_networks,
                # No likelihood pool: the solver folds each batch on its own pool.
                pool=(None, sampler_pool),
                seed=config.seed,
            )
            start = time.perf_counter()
            # Dropping the exploration phase avoids biased posterior and evidence.
            self._nested_sampler.run(verbose=True, discard_exploration=True)
            seconds = time.perf_counter() - start
        finally:
            if sampler_pool is not None:
                sampler_pool.terminate()
                sampler_pool.join()

        points = np.asarray(self._nested_sampler.posterior(equal_weight=True)[0], dtype=np.float64)
        return FitResults(
            time=seconds,
            posterior_samples=self._posterior_dataframe(points),
            n_likelihood_evaluations=self._nested_sampler.n_like,
            log_Z=float(self._nested_sampler.log_z),
            log_Z_err=float(1 / np.sqrt(self._nested_sampler.n_eff)),
        )

    def sample(self, n: int) -> np.ndarray:
        """Up to *n* rows of the equal-weight posterior in random order (nautilus groups them by bound)."""
        return random_rows(self._nested_sampler.posterior(equal_weight=True)[0], n, self.config.seed)
