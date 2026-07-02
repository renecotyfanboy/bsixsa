from collections import defaultdict
import numpy as np
from .convenience import XSilence, has_indexed_suffix
from xspec import AllModels
from .convenience import iter_all_parameters

__all__ = [
    'build_prior',
    'loguniform',
    'norm',
    'uniform',
]

def uniform(low, high):
    """Create a uniform prior on a finite interval.

    Parameters:
        low:
            Lower bound of the interval.
        high:
            Upper bound of the interval.

    Returns:
        A SciPy ``uniform`` distribution defined on ``[low, high]``.
    """
    from scipy.stats import uniform
    return uniform(low, high-low)

def loguniform(low, high):
    """Create a log-uniform prior on a positive interval.

    Parameters:
        low:
            Strictly positive lower bound.
        high:
            Upper bound.

    Returns:
        A SciPy ``loguniform`` distribution defined on ``[low, high]``.
    """
    from scipy.stats import loguniform
    return loguniform(low, high)

def norm(loc, scale, low, high):
    """Create a truncated Gaussian prior.

    Parameters:
        loc:
            Mean of the underlying Gaussian distribution.
        scale:
            Standard deviation of the underlying Gaussian distribution.
        low:
            Lower truncation bound, expressed in SciPy ``truncnorm``
            coordinates.
        high:
            Upper truncation bound, expressed in SciPy ``truncnorm``
            coordinates.

    Returns:
        A SciPy ``truncnorm`` distribution with the requested location,
        scale, and truncation limits.
    """
    from scipy.stats import truncnorm
    a, b = (low - loc) / scale, (high - loc) / scale
    return truncnorm(a, b, loc=loc, scale=scale)


class MultipleIndependent:
    def __init__(self, dists):
        self.dists = list(dists)
        self.ndim = len(self.dists)
        self.bounds = [dist.support() for dist in self.dists]

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

    def from_unit_gaussian(self, z):
        """Map ``z ~ N(0, I)^D`` to physical space via ``ppf(Phi(z))`` per marginal.

        Composes the standard-normal CDF ``Phi`` with each marginal's inverse CDF
        (``from_unit_cube``), so a standard-normal latent is mapped to the prior
        distribution. This is the unbounded counterpart of ``from_unit_cube`` used
        by the SIXSA backend, whose latent space is a unit Gaussian rather than the
        unit cube.
        """
        from scipy.stats import norm
        z = np.asarray(z, dtype=float)
        return self.from_unit_cube(norm.cdf(z))

    def to_unit_gaussian(self, x):
        """Map physical parameters to ``z ~ N(0, I)^D`` via ``Phi^{-1}(cdf(x))`` per marginal.

        Inverse of ``from_unit_gaussian``: the per-marginal CDF (``to_unit_cube``)
        followed by the standard-normal inverse CDF. The unit-cube values are
        clipped away from ``0``/``1`` so the inverse CDF stays finite at the prior
        bounds.
        """
        from scipy.stats import norm
        cube = np.clip(self.to_unit_cube(x), 1e-12, 1.0 - 1e-12)
        return norm.ppf(cube)

    def log_prob(self, x):
        x = np.asarray(x)
        x2 = np.atleast_2d(x)
        lp = np.zeros(x2.shape[0], dtype=float)

        for j, dist in enumerate(self.dists):
            lp += dist.logpdf(x2[:, j])

        return lp[0] if x.ndim == 1 else lp

    def xspec_bayes_contribution(self, x) -> float:
        """XSPEC ``Fit.bayes="on"`` prior contribution to the fit statistic.

        Evaluates ``-2 * Log(prior)`` per axis with XSPEC's unnormalised
        conventions (``Bayes::lPrior``) for the priors we compare against
        XSPEC: ``uniform`` -> ``cons`` contributes ``0``, ``loguniform`` ->
        ``jeffreys`` contributes ``2*ln(x)``. Adding this to the bare cstat
        at ``x`` reproduces XSPEC's ``Fit.statistic`` when ``Fit.bayes`` is
        on for those priors.
        """
        x = np.asarray(x, dtype=float).ravel()
        contribution = 0.0
        for j, dist in enumerate(self.dists):
            name = dist.dist.name
            if name == "uniform":
                continue
            elif name in ("loguniform", "reciprocal"):
                contribution += 2.0 * np.log(x[j])
            else:
                # Only uniform (-> cons) and loguniform (-> jeffreys) are
                # guaranteed to match XSPEC's bayes-on statistic exactly.
                # Everything else uses the scipy log-pdf, which differs
                # from the XSPEC counterpart by normalisation constants
                # (e.g. XSPEC's gauss prior is an untruncated Gaussian
                # while bsixsa's norm() is a truncated one) — acceptable,
                # as only uniform/loguniform fits are cross-checked
                # against XSPEC.
                contribution += -2.0 * float(dist.logpdf(x[j]))
        return float(contribution)

    def max_log_pdf_per_axis(self) -> np.ndarray:
        """Per-axis maximum of ``logpdf`` over the prior support.

        Used by least-squares MAP fits to shift the prior log-density into a
        non-negative range, so that the prior contribution can be encoded as
        extra residuals whose sum of squares equals
        ``-2 * log_prior + const``.

        Closed-form per family. Supported priors are ``uniform``,
        ``loguniform``, and truncated ``norm``.
        """
        out = np.empty(self.ndim, dtype=float)
        for j, dist in enumerate(self.dists):
            lo, hi = (float(b) for b in dist.support())
            name = dist.dist.name
            if name == "uniform":
                out[j] = float(dist.logpdf(0.5 * (lo + hi)))
            elif name in ("loguniform", "reciprocal"):
                out[j] = float(dist.logpdf(lo))
            elif name == "truncnorm":
                loc = float(dist.kwds.get("loc", 0.0))
                out[j] = float(dist.logpdf(min(max(loc, lo), hi)))
            else:
                raise NotImplementedError(
                    f"max_log_pdf is not implemented for {name!r}; "
                    "supported priors are uniform, loguniform, and truncnorm."
                )
        return out


def build_prior(define_prior):
    """Build the joint prior from a list of XSPEC prior specifications.

    For a **single-model** fit, each item must be a 3-tuple:

    `(component_name, parameter_name, distribution)`

    For a **multi-model** fit, each item must be a 4-tuple:

    `(model_name, component_name, parameter_name, distribution)`

    `distribution` can be and must behave like a SciPy continuous
    distribution and provide at least `support()`, `ppf()`,
    `cdf()`, `rvs()`, and `logpdf()`.
    Component and parameter names must match the names exposed by `xspec`.
    Repeated components can be addressed with suffixed names such as `powerlaw_3`.

    Parameters:
        define_prior:
            Sequence describing one prior per free XSPEC parameter.

    Returns:
        `(prior, parameters_index, model_index)`

        `prior` is a [`MultipleIndependent`][bsixsa.priors.MultipleIndependent]
        distribution over all unfrozen, unlinked fitted parameters. Its
        per-parameter support bounds are available on `prior.bounds`.
    """
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
        if has_indexed_suffix(component_name):
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
    for par in iter_all_parameters():

        prior = None

        # Smooth brain syntax to check whether a prior exists for a given parameter and extract its value
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
            parameters_index.append(par.parameter.index)
            model_index.append(par.model.name)

    # Turn the model / index dict into args for AllModels.set and apply them
    parameter_to_set = sum(((k, v) for k, v in parameter_to_set.items()), ())

    with XSilence():
        AllModels.setPars(*parameter_to_set)

    prior = MultipleIndependent(list_of_prior)
    return prior, parameters_index, model_index
