from . import create_uniform_prior_for, create_loguniform_prior_for
from .solver import SIXSASolver, XSilence

import numpy as np
import dill
import xspec
import pandas as pd
from random import sample


def set_prior_and_build_transform(xspec_model, define_prior):

    transform_build_dict = {
        "uniform": create_uniform_prior_for,
        "loguniform": create_loguniform_prior_for
    }

    parameter_to_set = {}

    with XSilence():

        for component, parameter, low, high, kind in define_prior:

            assert kind in transform_build_dict.keys(), f"kind must be one of {transform_build_dict.keys()}"

            xspec_comp = getattr(xspec_model, component)
            xspec_par = getattr(xspec_comp, parameter)
            parameter_to_set[xspec_par.index] = f"{np.random.uniform(low, high)},,{low},{low},{high},{high}"

            if kind == "uniform":
                xspec_par.prior = "cons"
            else:
                xspec_par.prior = "jeffreys"

        xspec_model.setPars(parameter_to_set)

        transformations = []

        for component, parameter, low, high, kind in define_prior:

            xspec_comp = getattr(xspec_model, component)
            xspec_par = getattr(xspec_comp, parameter)
            transformations.append(transform_build_dict[kind](xspec_model, xspec_par))

    return transformations


def load_xspec_data(
        path="spectrum_opt.pha",
        low_energy=0.3,
        high_energy=12.,
        lmod=None
):

    xspec.AllData.clear()
    xspec.AllModels.clear()

    with XSilence():

        if lmod is not None:
            xspec.AllModels.lmod(lmod)

        xspec.Xset.restore("model.xcm")
        xspec.Fit.statMethod = "cstat"
        xspec.Fit.bayes = "on"

        xspec_observation = xspec.Spectrum(path)
        xspec_observation.background = None
        xspec_observation.ignore(f"0.0-{low_energy:.1f} {high_energy:.1f}-**")

        xspec_model = xspec.AllModels(1)

    return xspec_model, xspec_observation


def load_solver_from_pickle(transformations, path_pickle)-> SIXSASolver:

    solver = SIXSASolver(
        transformations,
        outputfiles_basename=""
    )

    with open(path_pickle, "rb") as f:
        new_solver = dill.load(f)
        new_solver.transformations = solver.transformations

    return new_solver


def build_dataframe_from_solver(path, solver, n_points=10000) -> pd.DataFrame:

    indexes = np.sort([t['index'] for t in solver.transformations])
    parameter_names = np.asarray(solver.parameter_names_uniques)[indexes - 1]

    posterior = solver.fitted_posteriors[-1]
    samples = posterior.sample((n_points,))
    warped_samples = solver.unit_cube_to_xspec(samples.numpy().T)

    dict_of_params = {}

    for i in indexes:
        name = solver.parameter_names_uniques[i - 1]
        dict_of_params[name] = np.asarray([warped_samples[j][i] for j in range(len(warped_samples))])

    df = pd.DataFrame.from_dict(dict_of_params)
    df.to_csv(path)

    return df


def build_cstat_df(path, solver: SIXSASolver, dataframe:pd.DataFrame, mapping, n_points=3000):

    list_of_params = []

    for pars in dataframe.to_dict(orient="records"):
        param_dict = {indexes: pars[key] for key, indexes in mapping.items()}
        list_of_params.append(param_dict)

    c_stat = solver.simulate(sample(list_of_params, n_points), return_stat=True)

    np.savetxt(path, c_stat)