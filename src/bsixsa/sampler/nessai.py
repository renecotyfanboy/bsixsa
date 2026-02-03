import torch
import typing
from .abc import Sampler
import warnings
from nessai.model import Model as NessaiModel
from nessai.flowsampler import FlowSampler
from nessai.livepoint import live_points_to_array
from nessai.posterior import draw_posterior_samples

if typing.TYPE_CHECKING:
    from ..solver import SIXSASolver

class ModelFromSolver(NessaiModel):

    def __init__(self, solver):
        # Names of parameters to sample
        self.solver = solver
        self.names = [solver.parameter_names[idx] for idx in solver.indexes]
        # Prior bounds for each parameter
        self.bounds = {solver.parameter_names[idx]: bound for idx, bound in zip(solver.indexes, solver.bounds)}

    def to_array(self, x):
        new_array = live_points_to_array(x, names=self.names, copy=True)

        if new_array.ndim == 1:
            new_array = new_array[None, :]

        return new_array

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
        self.sampler = FlowSampler(
            self.model,
            nlive=n_live_points,
            output=self.solver.outputfiles_basename,
            **kwargs
        )

    def run(self):

        self.sampler.run()
        return self.sampler

    def sample(self, shape):

        samples = draw_posterior_samples(self.sampler.nested_samples, nlive=self.n_live_points, n=int(shape[0]), method="importance_sampling")
        return self.model.to_array(samples)
