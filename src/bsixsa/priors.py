from __future__ import print_function

import numpy as np
import re
from .convenience import XSilence
from scipy.stats import loguniform, norm

__all__ = [
    'build_prior',
    'loguniform',
    'norm',
    'uniform',
]

def uniform(low, high):
    """
    Simple wrapper around scipy.stats.uniform that returns a left-right defined distribution.
    """
    from scipy.stats import uniform
    return uniform(low, high-low)


class MultipleIndependent:
    def __init__(self, dists):
        self.dists = list(dists)
        self.ndim = len(self.dists)

    def sample(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        cols = [dist.rvs(size=size, random_state=rng) for dist in self.dists]
        x = np.stack(cols, axis=-1)  # shape: size + (ndim,)
        return x

    def log_prob(self, x):
        x = np.asarray(x)
        x2 = np.atleast_2d(x)
        if x2.shape[-1] != self.ndim:
            raise ValueError(f"Expected last dim = {self.ndim}, got {x2.shape[-1]}")

        lp = np.zeros(x2.shape[0], dtype=float)
        for j, dist in enumerate(self.dists):
            if hasattr(dist, "logpdf"):
                lp += dist.logpdf(x2[:, j])
            else:
                lp += dist.logpmf(x2[:, j])

        return lp[0] if x.ndim == 1 else lp


def build_prior(xspec_model, define_prior, return_bounds=False):

    parameter_to_set = {}
    list_of_prior = []
    parameters_index = []
    bounds = []

    with XSilence():
        for component, parameter, distribution in define_prior:

            # Handle the weird situation where a component is defined multiple times
            # EG tbabs*(powerlaw + powerlaw) will yield tbabs, powerlaw & powerlaw_3 as component names
            # An insightful user might want to pass "powerlaw_2" & "powerlaw_3" instead of "powerlaw" & "powerlaw_3"
            if bool(re.fullmatch(r'.*_\d+$', component)):
                if component not in xspec_model.componentNames:
                    split_name = component.split('_')[0]
                    if split_name in xspec_model.componentNames:
                        component = split_name
                    else:
                        raise ValueError(f"Component '{component}' or '{split_name}' not in {xspec_model.componentNames}")

            xspec_comp = getattr(xspec_model, component)
            xspec_par = getattr(xspec_comp, parameter)

            xspec_par.prior = "cons" # we handle the prior log_prob instead of XSPEC
            low, high = distribution.support()
            parameter_to_set[xspec_par.index] = (
                f"{np.random.uniform(low, high)},,{low},{low},{high},{high}"
            )

            list_of_prior.append(distribution)
            bounds.append([low, high])
            parameters_index.append(xspec_par.index)

        with XSilence():
            xspec_model.setPars(parameter_to_set)

    prior = MultipleIndependent(list_of_prior)

    if not return_bounds:

        return prior, parameters_index

    else:

        return prior, parameters_index, bounds
