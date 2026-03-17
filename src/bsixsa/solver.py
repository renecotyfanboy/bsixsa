from __future__ import print_function

import os
import shutil
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xspec
from xspec import AllModels

from .convenience import XSilence
from .priors import build_prior
from .xspec import parallel_folding
import pathos.multiprocessing as multiprocessing  # pathos


@dataclass
class FitResults:
    time: float
    posterior_samples: pd.DataFrame
    n_likelihood_evaluations: int
    log_Z: float
    log_Z_err: float


def set_parameters(values, indexes, model_indexes):
    """Update the active XSPEC parameters using transformed values.

    Parameters:
        values (Sequence[float]): Parameter values in the physical space
    """
    parameter_to_set = {}  # Will contain {model : {par_index:prior}}
    for value, index, model_name in zip(values, indexes, model_indexes):
        parameter_to_set[model_name] = parameter_to_set.get(model_name, {})
        parameter_to_set[model_name][index] = float(value)

    parameter_to_set = sum(
        ((AllModels(1, modName=k), v) for k, v in parameter_to_set.items()), ()
    )
    AllModels.setPars(*parameter_to_set)


class SIXSASolver(object):
    allowed_stats = ["cstat", "pstat"]

    def __init__(
        self,
        prior,
        outputfiles_basename="./sixsa",
        overwrite=False,
        n_jobs=os.cpu_count(),
        sampler="nessai"
    ):

        prior, indexes, model_indexes, bounds = build_prior(prior, return_bounds=True)

        self.indexes = indexes
        self.model_indexes = model_indexes
        sources_models = AllModels.sources
        self.nb_models = len(sources_models)
        self.prior = prior
        self.samplers = {"prior": prior}
        self.bounds = bounds
        self.pool = multiprocessing.Pool(processes=n_jobs)
        self.posterior_samples = None
        self.sampler_kind:str = sampler
        self.sampler = None

        normalized_output_dir = os.path.normpath(outputfiles_basename)

        if not os.path.exists(normalized_output_dir):
            os.makedirs(normalized_output_dir)

        else:
            if not os.path.isdir(normalized_output_dir):
                raise ValueError(
                    f"Output path '{outputfiles_basename}' exists and is not a directory."
                )

            existing_entries = os.listdir(normalized_output_dir)
            if existing_entries:
                if not overwrite:
                    raise FileExistsError(
                        f"Output directory '{normalized_output_dir}' is not empty. "
                        "Pass overwrite=True to clear it."
                    )

                for entry in existing_entries:
                    entry_path = os.path.join(normalized_output_dir, entry)
                    if os.path.isdir(entry_path) and not os.path.islink(entry_path):
                        shutil.rmtree(entry_path)
                    else:
                        os.remove(entry_path)

        if not normalized_output_dir.endswith(os.sep):
            normalized_output_dir = normalized_output_dir + os.sep

        outputfiles_basename = normalized_output_dir
        self.outputfiles_basename = outputfiles_basename

    @property
    def num_parameters(self):
        return len(self.indexes)

    @property
    def observed_spectrum(self):
        """
        Return the observed spectrum read from `xspec`
        """
        rate = np.asarray(xspec.AllData(1).values, dtype=np.float32)
        exposure = xspec.AllData(1).exposure
        return rate * exposure

    def sample_parameters(
        self,
        n_samples: int,
        sampler_name: str = "prior",
    ):
        r"""Draw parameter samples from one of the registered samplers.

        Parameters:
            n_samples (int): Number of draws to generate.
            sampler_name (str, optional): Key of the sampler in
                ``self.samplers``. Defaults to ``"prior"``.

        Returns:
            torch.Tensor | numpy.ndarray: Raw samples returned by the selected
                sampler.
        """

        sampler = self.samplers[sampler_name]
        theta = sampler.sample((n_samples,))  # In the unit cube space
        return theta

    def simulate(self, parameters, return_kind="full_model_counts", progress_bar=True):
        """Fold parameters through XSPEC and stack simulation outputs.

        Parameters:
            parameters: Array-like batch of parameter vectors.
            return_kind (str, optional): One of ``"cstat"``,
                ``"full_model_counts"``, or ``"models_and_components"``.

        Returns:
            dict[str, numpy.ndarray | dict[str, numpy.ndarray]]: Stacked
                outputs returned by :func:`bsixsa.xspec.parallel_folding`.
        """

        return parallel_folding(
            parameters,
            self.indexes,
            self.model_indexes,
            pool=self.pool,
            return_kind=return_kind,
            progress_bar=progress_bar
        )

    def log_prob_fn(self, theta, x_o, from_unit_cube=False):
        r"""
        Return the log-posterior probability, defined as $\mathcal{LL} = -\frac{1}{2} \texttt{Cstat}$. Include the log-likelihood and any prior term defined in `xspec`.

        !!! note "On `x_o` parameter"
            `x_o` is a dummy parameter used for compatibility with `sbi` as the true spectrum is directly extracted from
            `xspec`. `sbi` require this parameter as in normal workflow, this function can be conditioned on any `x_o`.

        Parameters:
            theta (torch.Tensor): Array of samples on the unit cube
            x_o (torch.Tensor): Observed value, pass `None` if needed

        """

        if not from_unit_cube:

            return self.log_likelihood_fn(theta, x_o) + self.log_prior_fn(theta, x_o)

        else:
            theta = self.prior.from_unit_cube(theta)
            return self.log_likelihood_fn(theta, x_o)

    def log_likelihood_fn(self, theta, x_o, progress_bar=True, no_pool=False):

        if no_pool:
            pool = None

        else:
            pool = self.pool

        simulation_outputs = parallel_folding(
            theta,
            self.indexes,
            self.model_indexes,
            desc="Evaluating C_stat - ",
            progress_bar=progress_bar,
            pool=pool,
            return_kind="cstat",
        )

        return -0.5 * simulation_outputs["cstat"]

    def log_prior_fn(self, theta, x_o):

        return self.samplers["prior"].log_prob(theta)


    @property
    def parameter_names(self) -> list[str]:
        """Return unique parameter names aligned with XSPEC ordering.

        Returns:
            (list[str]): Parameter names augmented with component identifiers to
                avoid duplicates.
        """

        from .convenience import iter_thawn_parameters
        return [parameter.name for parameter in iter_thawn_parameters()]


    def build_dataframe(
        self,
        sampler: str = "exact_sampler",
        num_samples=10_000,
    ) -> pd.DataFrame:
        """Build a posterior sample table from the fitted neural network.

        Parameters:
            num_samples (int, optional): Number of samples to draw from the
                posterior sampler. Defaults to 10_000.

        Returns:
            pandas.DataFrame: Table with parameter samples and corresponding
                importance weights.
        """

        sampler_kwargs = {}

        samples = self.samplers[sampler].sample(
            (num_samples,), **sampler_kwargs
        )

        dict_of_params = {
            name: parameters for name, parameters in zip(self.parameter_names, samples.T)
        }

        return pd.DataFrame.from_dict(dict_of_params)


    def get_xspec_best_fit(self):
        """Run an XSPEC fit and return best-fit parameters with covariance.

        Returns:
            (tuple[numpy.ndarray, numpy.ndarray]): Flattened parameter vector and
                covariance matrix estimated by XSPEC.
        """

        with XSilence():
            xspec.Fit.perform()

        def build_covariance_matrix_np(covar_elements):
            covar_elements = np.asarray(covar_elements, dtype=float)

            M = len(covar_elements)
            N = int((np.sqrt(1 + 8 * M) - 1) // 2)

            cov_matrix = np.zeros((N, N), dtype=float)
            i, j = np.tril_indices(N)
            cov_matrix[i, j] = covar_elements
            cov_matrix[j, i] = covar_elements

            return cov_matrix

        sources_models = xspec.AllModels.sources
        best_fit_parameters = []
        for source, model_name in zip(sources_models.keys(), sources_models.values()):

            xspec_model = xspec.AllModels(1,model_name)


            best_fit_parameters.extend(
                [xspec_model(i + 1).values[0] for i in range(xspec_model.nParameters)]
            )
        covariance = build_covariance_matrix_np(xspec.Fit.covariance)
        best_fit_parameters = np.asarray(best_fit_parameters)
        return best_fit_parameters.ravel(), covariance


    def plot_ppc(self, sampler, **kwargs):
        from .analysis.plotting import plot_ppc

        return plot_ppc(self, sampler, **kwargs)

    def run(self, *args, **kwargs):


        # TODO : check for a way to distinguish sampler & samplers
        if self.sampler_kind == "nessai":
            from .sampler.nessai import NessaiSampler
            self.sampler = NessaiSampler(solver=self, **kwargs)

        elif self.sampler_kind == "nautilus":
            from .sampler.nautilus import NautilusSampler
            self.sampler = NautilusSampler(solver=self, **kwargs)

        elif self.sampler_kind == "ultranest":
            from .sampler.ultranest import UltranestSampler
            self.sampler = UltranestSampler(solver=self, **kwargs)

        elif self.sampler_kind == "sixsa":
            from .sampler.sixsa import SIXSASampler
            self.sampler = SIXSASampler(solver=self)

            return self.sampler.run(*args, **kwargs)

        elif self.sampler_kind == "levenberg_marquardt":
            from .sampler.levenberg_marquart import LevenbergMarquardtSampler
            self.sampler = LevenbergMarquardtSampler(solver=self)
            self.sampler.run(*args, **kwargs)
            self.samplers["posterior"] = self.sampler

            n_samples = kwargs.get("n_posterior_samples", 10_000)
            self.posterior_samples = self.build_dataframe(
                sampler="posterior", num_samples=n_samples
            )

            return self.sampler.result

        else:
            raise NotImplementedError()

        if not self.sampler_kind == "sixsa":

            self.posterior_samples = self.sampler.run()
            self.samplers["posterior"] = self.sampler

            return self.sampler.sampler
