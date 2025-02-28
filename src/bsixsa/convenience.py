from . import create_uniform_prior_for, create_loguniform_prior_for
import numpy as np

def set_prior_and_build_transform(xspec_model, define_prior, backend="sixsa"):

    transform_build_dict = {
        "uniform": create_uniform_prior_for,
        "loguniform": create_loguniform_prior_for
    }

    parameter_to_set = {}

    for component, parameter, low, high, kind in define_prior:

        assert kind in transform_build_dict.keys(), f"kind must be one of {transform_build_dict.keys()}"

        xspec_comp = getattr(xspec_model, component)
        xspec_par = getattr(xspec_comp, parameter)
        parameter_to_set[xspec_par.index] = f"{np.random.uniform(low, high)},,{low},{low},{high},{high}"

        if kind == "uniform":
            xspec_par.prior = "jeffreys"
        else:
            xspec_par.prior = "CONS"

    xspec_model.setPars(parameter_to_set)

    transformations = []

    for component, parameter, low, high, kind in define_prior:

        xspec_comp = getattr(xspec_model, component)
        xspec_par = getattr(xspec_comp, parameter)
        transformations.append(transform_build_dict[kind](xspec_model, xspec_par))

    return transformations
