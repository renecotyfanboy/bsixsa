import torch
import typing
import numpy as np
from .abc import Sampler
import warnings
import pandas as pd
from nessai.model import Model as NessaiModel
from nessai.flowsampler import FlowSampler
from nessai.livepoint import live_points_to_array, live_points_to_dict
from nessai.posterior import draw_posterior_samples


if typing.TYPE_CHECKING:
    from ..solver import SIXSASolver

class ModelFromSolver(NessaiModel):

    def __init__(self, solver):
        # Names of parameters to sample
        self.solver = solver
        self.names = solver.parameter_names
        # Prior bounds for each parameter
        self.bounds = {self.names[idx]: bound for idx, bound in zip(range(len(self.names)), solver.bounds)}

    def to_array(self, x):
        new_array = live_points_to_array(x, names=self.names, copy=True)

        if new_array.ndim == 1:
            new_array = new_array[None, :]

        return new_array

    def from_unit_hypercube(self, x):
        """
        Map from the unit hypercube to the physical parameter space.
        """
        x = x.copy()
        for name, dist in zip(self.names, self.solver.prior.dists):
            x[name] = dist.ppf(x[name])
        return x

    def to_unit_hypercube(self, x):
        """
        Map from the physical parameter space to the unit hypercube.
        """
        x = x.copy()
        for name, dist in zip(self.names, self.solver.prior.dists):
            x[name] = np.clip(dist.cdf(x[name]), 0.0, 1.0)
        return x

    def log_prior(self, theta):
        """
        Returns log of prior given a live point assuming uniform
        priors on each parameter.
        """
        theta = self.to_array(theta)
        prob = self.solver.log_prior_fn(theta, None)
        return prob

    def log_likelihood(self, theta):
        """
        Returns log likelihood of given live point assuming a Gaussian
        likelihood.
        """
        theta = self.to_array(theta)
        res = self.solver.log_likelihood_fn(theta, None, progress_bar=False, no_pool=False)

        return res


class NessaiSampler(Sampler):

    def __init__(self, *, solver:'SIXSASolver', n_live_points:int=1_000, **kwargs):

        super().__init__(solver=solver)

        self.model = ModelFromSolver(self.solver)
        self.n_live_points = n_live_points
        kwargs.setdefault("importance_nested_sampler", True)
        kwargs.setdefault("min_samples", min(500, n_live_points))
        self.sampler = FlowSampler(
            self.model,
            nlive=n_live_points,
            output=self.solver.outputfiles_basename,
            **kwargs
        )

    def run(self):
        from ..solver import FitResults
        self.sampler.run()

        posterior_dict = live_points_to_dict(self.sampler.posterior_samples, names=self.solver.parameter_names)

        self.solver.fit_result = FitResults(
            time=float(self.sampler.ns.current_sampling_time.total_seconds()),
            posterior_samples=pd.DataFrame.from_dict(posterior_dict),
            n_likelihood_evaluations=self.sampler.ns.total_likelihood_evaluations,
            log_Z=float(self.sampler.log_evidence),
            log_Z_err=float(self.sampler.log_evidence_error),
        )

        return self.sampler

    def sample(self, shape):
        n = int(shape[0])

        if getattr(self.sampler, "nested_samples", None) is not None:

            samples = draw_posterior_samples(
                self.sampler.nested_samples,
                nlive=self.n_live_points,
                n=n,
                method="importance_sampling",
            )
            return self.model.to_array(samples)

        posterior = self.model.to_array(self.sampler.posterior_samples)
        if posterior.shape[0] >= n:
            return posterior[:n]

        idx = np.random.default_rng().choice(posterior.shape[0], size=n, replace=True)
        return posterior[idx]
