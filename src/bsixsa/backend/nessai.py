"""Nested sampling with nessai's importance nested sampler, driven by normalising flows."""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING

import numpy as np
from nessai.flowsampler import FlowSampler
from nessai.livepoint import live_points_to_array
from nessai.model import Model as NessaiModel

from ..solver import FitResults
from .abc import Backend, traced_log_likelihood
from .config import Nessai

if TYPE_CHECKING:
    from ..solver import Solver
    from .tracer import EvaluationTracer


class _NessaiModel(NessaiModel):
    """nessai's view of the solver.

    It holds the solver, not the backend, because nessai may pickle it for its worker pool.
    """

    def __init__(self, solver: Solver, tracer: EvaluationTracer) -> None:
        self.solver = solver
        self._tracer = tracer
        self.names = solver.parameter_names
        self.bounds = dict(zip(self.names, solver.prior.bounds))

    def to_array(self, x: np.ndarray) -> np.ndarray:
        """Live points (a structured array) as a plain array of shape `(n, d)`."""
        return np.atleast_2d(live_points_to_array(x, names=self.names, copy=True))

    def _with_values(self, x: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Copy of *x* with its parameter fields set from the columns of *values*."""
        out = x.copy()
        for j, name in enumerate(self.names):
            out[name] = values[:, j]
        return out

    def from_unit_hypercube(self, x: np.ndarray) -> np.ndarray:
        """Map live points from the unit hypercube to physical parameters."""
        return self._with_values(x, self.solver.prior.from_unit_cube(self.to_array(x)))

    def to_unit_hypercube(self, x: np.ndarray) -> np.ndarray:
        """Map live points from physical parameters to the unit hypercube."""
        return self._with_values(x, self.solver.prior.to_unit_cube(self.to_array(x)))

    def log_prior(self, theta: np.ndarray) -> np.ndarray:
        """Log prior density of the live points."""
        return self.solver.log_prior_fn(self.to_array(theta))

    def log_likelihood(self, theta: np.ndarray) -> np.ndarray:
        """Log-likelihood `-C/2` of the live points, recorded by the tracer."""
        return traced_log_likelihood(self.solver, self._tracer, self.to_array(theta))


class NessaiBackend(Backend):
    """Importance nested sampling with nessai; returns the equal-weight posterior and log Z."""

    config_cls = Nessai

    def __init__(self, *, solver: Solver, config: Nessai) -> None:
        super().__init__(solver=solver, config=config)
        self.model = _NessaiModel(solver, self.tracer)
        output = os.path.join(solver.outputfiles_basename, "nessai_outputs")
        os.makedirs(output, exist_ok=True)
        self._flow_sampler = FlowSampler(
            self.model,
            nlive=config.num_live_points,
            output=output,
            importance_nested_sampler=True,
            **config.engine_kwargs,
        )

    def run(self) -> FitResults:
        """Run nessai and return its equal-weight posterior and log evidence."""
        with warnings.catch_warnings():
            # Harmless numpy warnings raised inside nessai.
            warnings.filterwarnings("ignore", message="divide by zero encountered in log")
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            warnings.filterwarnings("ignore", message="invalid value encountered in scalar subtract")
            warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")
            self._flow_sampler.run()

        nested_sampler = self._flow_sampler.ns
        return FitResults(
            time=float(nested_sampler.current_sampling_time.total_seconds()),
            posterior_samples=self._posterior_dataframe(self.model.to_array(self._flow_sampler.posterior_samples)),
            n_likelihood_evaluations=nested_sampler.total_likelihood_evaluations,
            log_Z=float(self._flow_sampler.log_evidence),
            log_Z_err=float(self._flow_sampler.log_evidence_error),
        )

    def sample(self, n: int) -> np.ndarray:
        """Draw *n* posterior rows with replacement, using nessai's importance weights."""
        rows = self._flow_sampler.ns.draw_posterior_samples(sampling_method="importance_sampling", n=int(n))
        return self.model.to_array(rows)
