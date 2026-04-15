import os
import typing

import numpy as np
import pandas as pd
from nessai.flowsampler import FlowSampler
from nessai.livepoint import live_points_to_array, live_points_to_dict
from nessai.model import Model as NessaiModel
from nessai.posterior import draw_posterior_samples
from ..solver import FitResults
from .abc import Backend
from . import register_backend
from .config import Nessai
import warnings

if typing.TYPE_CHECKING:
    from ..solver import SIXSASolver


class ModelFromSolver(NessaiModel):

    def __init__(self, solver, tracer):
        self.solver = solver
        self._tracer = tracer
        self.names = solver.parameter_names
        self.bounds = {
            self.names[idx]: bound
            for idx, bound in enumerate(solver.prior.bounds)
        }

    def to_array(self, x):
        new_array = live_points_to_array(x, names=self.names, copy=True)

        if new_array.ndim == 1:
            new_array = new_array[None, :]

        return new_array

    def from_unit_hypercube(self, x):
        """Map from the unit hypercube to the physical parameter space."""
        x_out = x.copy()
        params = self.solver.prior.from_unit_cube(self.to_array(x))
        for idx, name in enumerate(self.names):
            x_out[name] = params[:, idx]
        return x_out

    def to_unit_hypercube(self, x):
        """Map from the physical parameter space to the unit hypercube."""
        x_out = x.copy()
        cube = self.solver.prior.to_unit_cube(self.to_array(x))
        for idx, name in enumerate(self.names):
            x_out[name] = cube[:, idx]
        return x_out

    def log_prior(self, theta):
        """Returns log of prior given a live point."""
        theta = self.to_array(theta)
        return self.solver.log_prior_fn(theta, None)

    def log_likelihood(self, theta):
        """Returns log likelihood of given live point."""
        theta = self.to_array(theta)
        ll = self.solver.log_likelihood_fn(theta, None, progress_bar=False, no_pool=False)
        self._tracer.record(theta, -2.0 * np.atleast_1d(ll))
        return ll


@register_backend
class NessaiBackend(Backend):

    name = "nessai"
    config_cls = Nessai

    def __init__(
        self,
        *,
        solver: 'SIXSASolver',
        config: Nessai,
    ):
        super().__init__(solver=solver, config=config)

        flow_kwargs = self.config.engine_kwargs.copy()
        flow_kwargs.setdefault("importance_nested_sampler", True)
        self.model = ModelFromSolver(self.solver, self.tracer)
        self.n_live_points = self.config.num_live_points

        nessai_output = os.path.join(self.solver.outputfiles_basename, "nessai_outputs")
        os.makedirs(nessai_output, exist_ok=True)

        self._flow_sampler = FlowSampler(
            self.model,
            nlive=self.config.num_live_points,
            output=nessai_output,
            **flow_kwargs
        )

    def run(self, **kwargs) -> 'FitResults':
        from ..solver import FitResults

        with warnings.catch_warnings():
            # Ignore a bunch of warnings that are raised by nessai early
            warnings.filterwarnings("ignore", message="divide by zero encountered in log")
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            warnings.filterwarnings("ignore", message="invalid value encountered in scalar subtract")
            warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

            self._flow_sampler.run(
                #compute_initial_posterior=True,
                #redraw_samples=True
            )

        posterior_dict = live_points_to_dict(
            self._flow_sampler.posterior_samples,
            names=self.solver.parameter_names,
        )

        return FitResults(
            time=float(self._flow_sampler.ns.current_sampling_time.total_seconds()),
            posterior_samples=pd.DataFrame.from_dict(posterior_dict),
            n_likelihood_evaluations=self._flow_sampler.ns.total_likelihood_evaluations,
            log_Z=float(self._flow_sampler.log_evidence),
            log_Z_err=float(self._flow_sampler.log_evidence_error),
        )

    def sample(self, n: int) -> np.ndarray:
        if getattr(self._flow_sampler, "nested_samples", None) is not None:
            samples = draw_posterior_samples(
                self._flow_sampler.nested_samples,
                nlive=self.n_live_points,
                n=n,
                method="importance_sampling",
            )
            return self.model.to_array(samples)

        posterior = self.model.to_array(self._flow_sampler.posterior_samples)
        if posterior.shape[0] >= n:
            return posterior[:n]

        idx = np.random.default_rng().choice(posterior.shape[0], size=n, replace=True)
        return posterior[idx]
