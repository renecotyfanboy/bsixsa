from __future__ import print_function
from math import log10


def create_uniform_prior_for(model, par):
    """Create a uniform prior transformation for location parameters.

    Parameters:
            model: XSPEC model that owns the parameter.
            par: XSPEC parameter whose value should follow a uniform prior.

    Returns:
            dict: Metadata describing how to transform samples into parameter
                    space.
    """
    pval, pdelta, pmin, pbottom, ptop, pmax = par.values
    print("  uniform prior for %s between %f and %f " % (par.name, pmin, pmax))
    # TODO: should we use min/max or bottom/top?
    low = float(pmin)
    spread = float(pmax - pmin)
    if pmin > 0 and pmax / pmin > 100:
        print(
            "   note: this parameter spans several dex. Should it be log-uniform (create_jeffreys_prior_for)?"
        )

    def uniform_transform(x):
        return x * spread + low

    return dict(
        model=model,
        index=par._Parameter__index,
        name=par.name,
        transform=uniform_transform,
        aftertransform=lambda x: x,
    )


def create_jeffreys_prior_for(model, par):
    """Return a log-uniform prior transformation (deprecated wrapper).

    Parameters:
            model: XSPEC model that owns the parameter.
            par: XSPEC parameter to transform.

    Returns:
            dict: Metadata describing how to transform samples into parameter
                    space.
    """
    return create_loguniform_prior_for(model, par)


def create_loguniform_prior_for(model, par):
    """Create a Jeffreys (log-uniform) prior transformation.

    Parameters:
            model: XSPEC model that owns the parameter.
            par: XSPEC parameter whose scale should be log-uniform.

    Returns:
            dict: Metadata describing how to transform samples into parameter
                    space.
    """
    pval, pdelta, pmin, pbottom, ptop, pmax = par.values
    # TODO: should we use min/max or bottom/top?
    # print '  ', par.values
    print("  jeffreys prior for %s between %e and %e " % (par.name, pmin, pmax))
    if pmin == 0:
        raise Exception(
            "You forgot to set reasonable parameter limits on %s" % par.name
        )
    low = log10(pmin)
    spread = log10(pmax) - log10(pmin)
    if spread > 10:
        print(
            "   note: this parameter spans *many* dex. Double-check the limits are reasonable."
        )

    def log_transform(x):
        return x * spread + low

    def log_after_transform(x):
        return 10**x

    return dict(
        model=model,
        index=par._Parameter__index,
        name="log(%s)" % par.name,
        transform=log_transform,
        aftertransform=log_after_transform,
    )


def create_gaussian_prior_for(model, par, mean, std):
    """Create a Gaussian prior transformation for informed parameters.

    Parameters:
            model: XSPEC model that owns the parameter.
            par: XSPEC parameter to transform.
            mean (float): Mean of the Gaussian prior.
            std (float): Standard deviation of the Gaussian prior.

    Returns:
            dict: Metadata describing how to transform samples into parameter
                    space.
    """
    import scipy.stats

    pval, pdelta, pmin, pbottom, ptop, pmax = par.values
    rv = scipy.stats.norm(mean, std)

    def gauss_transform(x):
        return max(pmin, min(pmax, rv.ppf(x)))

    print("  gaussian prior for %s of %f +- %f" % (par.name, mean, std))
    return dict(
        model=model,
        index=par._Parameter__index,
        name=par.name,
        transform=gauss_transform,
        aftertransform=lambda x: x,
    )


def create_custom_prior_for(model, par, transform, aftertransform=lambda x: x):
    """Create a prior transformation using caller-provided functions.

    Parameters:
            model: XSPEC model that owns the parameter.
            par: XSPEC parameter to transform.
            transform (Callable[[float], float]): Function that maps unit-cube
                    samples onto the parameter support.
            aftertransform (Callable[[float], float], optional): Reverse mapping
                    from parameter space back to unit-cube space. Defaults to the
                    identity function.

    Returns:
            dict: Metadata describing how to transform samples into parameter
                    space.
    """
    print("  custom prior for %s" % (par.name))
    return dict(
        model=model,
        index=par._Parameter__index,
        name=par.name,
        transform=transform,
        aftertransform=aftertransform,
    )


def create_prior_function(transformations):
    """Compose prior transformations into a single callable.

    Parameters:
            transformations (list[dict]): Sequence of prior transformation
                    metadata dictionaries as returned by the helper constructors.

    Returns:
            Callable: Function that mutates an in-place unit-cube sample to obey
                    the specified priors.
    """

    def prior(cube, ndim, nparams):
        for i, t in enumerate(transformations):
            transform = t["transform"]
            cube[i] = transform(cube[i])

    return prior
