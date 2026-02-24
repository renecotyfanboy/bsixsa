from __future__ import print_function

from collections import defaultdict
import numpy as np
import re
from .convenience import XSilence
from scipy.stats import loguniform, norm
from xspec import AllModels
from .convenience import iter_all_parameters

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
        x = np.stack(cols, axis=-1)
        return x

    def from_unit_cube(self, cube):
        cube = np.asarray(cube, dtype=float)
        params = np.empty_like(cube, dtype=float)
        for j, dist in enumerate(self.dists):
            params[..., j] = dist.ppf(cube[..., j])
        return params

    def to_unit_cube(self, x):
        x = np.asarray(x, dtype=float)
        cube = np.empty_like(x, dtype=float)
        for j, dist in enumerate(self.dists):
            cube[..., j] = dist.cdf(x[..., j])
        return cube

    def log_prob(self, x):
        x = np.asarray(x)
        x2 = np.atleast_2d(x)
        lp = np.zeros(x2.shape[0], dtype=float)

        for j, dist in enumerate(self.dists):
            lp += dist.logpdf(x2[:, j])

        return lp[0] if x.ndim == 1 else lp


def build_prior(define_prior, return_bounds=False):
    model_list = [AllModels(1, modName=name) for name in AllModels.sources.values()]

    is_single_model_prior = all(len(definition) == 3 for definition in define_prior)
    is_multi_model_prior = all(len(definition) == 4 for definition in define_prior)

    # Sanity checks
    if not (is_single_model_prior or is_multi_model_prior):
        raise ValueError("Prior must be defined either as single model or multi-model")

    if (is_single_model_prior and len(model_list) != 1) or (is_multi_model_prior and len(model_list) == 1):
        raise ValueError("Single model prior defined for multiple models (or the opposite)")

    # Let's format the prior to the general multimodel setup
    if is_single_model_prior:

        # We explicitly add the model name at the start of the prior settings
        model_name = model_list[0].name
        define_prior = [(model_name,) + prior_setting for prior_setting in define_prior]

    # We turn the list of prior settings in a nested dictionary indexed by model/components/parameter names
    define_prior_dict = defaultdict(lambda: defaultdict(dict))

    for model_name, component_name, parameter_name, distribution in define_prior:

        model = AllModels(1, modName=model_name)

        # Handle the weird situation where a component is defined multiple times
        # EG tbabs*(powerlaw + powerlaw) will yield tbabs, powerlaw & powerlaw_3 as component names
        # An insightful user might want to pass "powerlaw_2" & "powerlaw_3" instead of "powerlaw" & "powerlaw_3"
        if bool(re.fullmatch(r'.*_\d+$', component_name)):
            if component_name not in model.componentNames:
                split_name = component_name.split('_')[0]
                if split_name in model.componentNames:
                    component_name = split_name
                else:
                    raise ValueError(f"Component '{component_name}' or '{split_name}' not in {model.componentNames}")

        # Nested defaultdicts handle intermediate dictionary creation
        define_prior_dict[model_name][component_name][parameter_name] = distribution

    # Now we build the necessary blocks for inference
    parameter_to_set = defaultdict(dict)  # Will contain {model : {par_index:prior}}
    list_of_prior = []
    model_index = []
    parameters_index = []
    model_names = []
    bounds = []

    for par in iter_all_parameters():

        prior = None

        # Smooth brain syntaxe to check whether a prior exist for a given parameter and extract its value
        match define_prior_dict:
            # Prior is only instantiated in the following line if there is a match
            case {par.model.name: {par.component.name: {par.parameter.name: prior}}}:
                pass

        if prior is None and not (par.parameter.frozen or bool(par.parameter.link)):
            raise ValueError(f"No prior or link for {par.name}. Either pass a prior or freeze explicitly.")

        if prior is not None:

            low, high = prior.support()
            parameter_to_set[par.model][par.parameter.index] = f"{np.random.uniform(low, high)},0.1,{low},{low},{high},{high}"

            list_of_prior.append(prior)
            model_names.append(par.model.name)
            bounds.append((low, high))
            parameters_index.append(par.parameter.index)
            model_index.append(par.model.name)

    # Turn the model / index dict into args for AllModels.set and apply them
    parameter_to_set = sum(((k, v) for k, v in parameter_to_set.items()), ())

    with XSilence():
        AllModels.setPars(*parameter_to_set)

    prior = MultipleIndependent(list_of_prior)

    if not return_bounds:

        return prior, parameters_index, model_index

    else:

        return prior, parameters_index, model_index, bounds
