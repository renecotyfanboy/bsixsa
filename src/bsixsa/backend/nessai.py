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

if typing.TYPE_CHECKING:
    from ..solver import SIXSASolver


class ModelFromSolver(NessaiModel):

    def __init__(self, solver):
        self.solver = solver
        self.names = solver.parameter_names
        self.bounds = {self.names[idx]: bound for idx, bound in zip(range(len(self.names)), solver.bounds)}

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
        return self.solver.log_likelihood_fn(theta, None, progress_bar=False, no_pool=False)


@register_backend
class NessaiBackend(Backend):

    name = "nessai"

    def __init__(self, *, solver: 'SIXSASolver', n_live_points: int = 1_000, **kwargs):
        super().__init__(solver=solver)

        self.model = ModelFromSolver(self.solver)
        self.n_live_points = n_live_points
        kwargs.setdefault("importance_nested_sampler", True)
        kwargs.setdefault("min_samples", min(500, n_live_points))
        self._flow_sampler = FlowSampler(
            self.model,
            nlive=n_live_points,
            output=self.solver.outputfiles_basename,
            **kwargs
        )

    def run(self, **kwargs) -> 'FitResults':
        from ..solver import FitResults

        self._flow_sampler.run()

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
