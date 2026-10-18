"""Nested sampling with UltraNest."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np
import ultranest
import ultranest.popstepsampler

from ..solver import FitResults
from .abc import Backend
from .config import Ultranest

if TYPE_CHECKING:
    from ..solver import Solver


class UltranestBackend(Backend):
    """Nested sampling with UltraNest's reactive sampler; returns the equal-weight posterior and log Z."""

    config_cls = Ultranest

    def __init__(self, *, solver: Solver, config: Ultranest) -> None:
        super().__init__(solver=solver, config=config)
        output = os.path.join(solver.outputfiles_basename, "ultranest_outputs")
        os.makedirs(output, exist_ok=True)

        self._reactive_sampler = ultranest.ReactiveNestedSampler(
            list(solver.parameter_names),
            self._log_likelihood,
            transform=self._transform,
            log_dir=output,
            vectorized=True,
            # UltraNest asserts on `resume=False`; "overwrite" is its fresh start.
            resume="resume" if config.resume else "overwrite",
        )
        if config.use_step_sampler:
            self._reactive_sampler.stepsampler = ultranest.popstepsampler.PopulationSimpleSliceSampler(
                popsize=100,
                nsteps=50,
                scale=1,
                generate_direction=ultranest.popstepsampler.generate_region_oriented_direction,
            )

    def _transform(self, cube: np.ndarray) -> np.ndarray:
        """Map unit-cube points to physical parameters."""
        return self.solver.prior.from_unit_cube(np.atleast_2d(cube))

    def run(self) -> FitResults:
        """Run UltraNest and return its equal-weight posterior and log evidence."""
        start = time.perf_counter()
        self._results = self._reactive_sampler.run(min_num_live_points=self.config.num_live_points)
        seconds = time.perf_counter() - start

        self._reactive_sampler.plot()
        return FitResults(
            time=seconds,
            posterior_samples=self._posterior_dataframe(self._results["samples"]),
            n_likelihood_evaluations=self._results["ncall"],
            log_Z=float(self._results["logz"]),
            log_Z_err=float(self._results["logzerr"]),
        )

    def sample(self, n: int) -> np.ndarray:
        """Up to *n* rows of the equal-weight posterior (already shuffled by UltraNest)."""
        return self._results["samples"][: int(n)]
