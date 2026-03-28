from __future__ import print_function

import os
import shutil
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xspec
from xspec import AllModels

from .priors import build_prior
from .xspec import parallel_folding
from scipy.stats import poisson
import arviz as az
import pathos.multiprocessing as multiprocessing  # pathos


@dataclass
class FitResults:
    time: float
    posterior_samples: pd.DataFrame
    n_likelihood_evaluations: int
    log_Z: float
    log_Z_err: float


def likelihood_per_bin(observed: np.ndarray, model: np.ndarray) -> np.ndarray:
    r"""Per-bin Poisson log-likelihood (Cash statistic).
    For a bin with observed counts $S_i$ and predicted counts $\mathcal{M}_i$:

    $$
    \log \mathcal{L}_i = S_i \ln(\mathcal{M}_i) − \mathcal{M}_i
    $$

    where $\mathcal{M}_i > 0$

    Parameters:
        observed: Observed counts, shape ``(n_bins,)``.
        model: Predicted model counts, shape ``(n_bins,)`` or
            ``(n_samples, n_bins)``.

    Returns:
        Per-bin log-likelihood with the same shape as *model*.
    """

    model = np.asarray(model, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)

    return poisson.logpmf(observed, model)


def _validate_finite(parameters, parameter_names):
    """Raise ``ValueError`` if *parameters* contains NaN or Inf values."""
    bad_mask = ~np.isfinite(parameters)
    if bad_mask.any():
        bad_params = set()
        for col_idx in np.where(bad_mask.any(axis=0))[0]:
            name = parameter_names[col_idx] if col_idx < len(parameter_names) else f"index {col_idx}"
            bad_rows = np.where(bad_mask[:, col_idx])[0]
            bad_params.add(f"{name} (samples {bad_rows.tolist()})")
        raise ValueError(
            f"Non-finite parameter values detected before simulation: "
            f"{', '.join(sorted(bad_params))}"
        )


def _split_kwargs(cls, kwargs):
    """Separate constructor kwargs from run kwargs by inspecting __init__."""
    import inspect
    init_params = inspect.signature(cls.__init__).parameters
    init_kwargs = {}
    run_kwargs = {}
    for key, value in kwargs.items():
        if key in init_params:
            init_kwargs[key] = value
        else:
            run_kwargs[key] = value
    return init_kwargs, run_kwargs


def set_parameters(values, indexes, model_indexes):
    """Update the active XSPEC parameters using transformed values.

    Parameters:
        values: Parameter values in the physical space.
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
        trace=True,
        backend="nessai"
    ):
        r"""Initialise the Bayesian X-ray spectral analysis solver.

        Reads the currently loaded XSPEC model, builds the prior
        distribution from *prior*, and prepares a multiprocessing pool
        for parallel likelihood evaluations.

        Parameters:
            prior: Prior specification forwarded to
                ``build_prior``.  Accepts the same formats as
                :func:`~bsixsa.priors.build_prior`.
            outputfiles_basename: Directory where backend output files
                (chains, checkpoints, traces) are stored.  Created
                automatically if it does not exist.
            overwrite: If ``True``, clear *outputfiles_basename* when it
                already contains files.  Defaults to ``False``.
            n_jobs: Number of worker processes for the multiprocessing
                pool.  Defaults to ``os.cpu_count()``.
            trace: If ``True``, enable run-time trace logging in the
                backend.  Defaults to ``True``.
            backend: Name of the backend to use
                (e.g. ``"nessai"``).  Defaults to ``"nessai"``.
        """

        if xspec.AllData.nSpectra < 1:
            raise ValueError("No spectra loaded in XSPEC")

        if len(xspec.AllModels.sources) < 1:
            raise ValueError("No model loaded in XSPEC")

        prior, indexes, model_indexes, bounds = build_prior(prior, return_bounds=True)

        self.indexes = indexes
        self.model_indexes = model_indexes
        sources_models = AllModels.sources
        self.nb_models = len(sources_models)
        self.prior = prior
        self.distributions = {"prior": prior}
        self.bounds = bounds
        self.pool = multiprocessing.Pool(processes=n_jobs)
        self.posterior_samples = None
        self.backend_name: str = backend
        self.backend = None
        self.fit_result: FitResults | None = None
        self.trace = trace

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
        distribution_name: str = "prior",
    ):
        r"""Draw parameter samples from one of the registered distributions.

        Parameters:
            n_samples: Number of draws to generate.
            distribution_name: Key of the distribution in
                ``self.distributions``. Defaults to ``"prior"``.

        Returns:
            Raw samples returned by the selected distribution.
        """

        distribution = self.distributions[distribution_name]
        theta = distribution.sample((n_samples,))  # In the unit cube space
        return theta

    def simulate(self, parameters, return_kind="full_model_counts", progress_bar=True):
        """Fold parameters through XSPEC and stack simulation outputs.

        Parameters:
            parameters: Array-like batch of parameter vectors.
            return_kind: One of ``"cstat"``, ``"full_model_counts"``, or
                ``"models_and_components"``.

        Returns:
            Stacked outputs returned by ``parallel_folding``.

        Raises:
            ValueError: If any parameter values are NaN or Inf.
        """

        parameters = np.asarray(parameters)
        _validate_finite(parameters, self.parameter_names)

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
            theta: Array of samples on the unit cube.
            x_o: Observed value, pass ``None`` if needed.
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

        return self.distributions["prior"].log_prob(theta)


    @property
    def parameter_names(self) -> list[str]:
        """Return unique parameter names aligned with XSPEC ordering.

        Returns:
            Parameter names augmented with component identifiers to avoid duplicates.
        """

        from .convenience import iter_thawn_parameters
        return [parameter.name for parameter in iter_thawn_parameters()]


    def _sample_from(self, distribution: str, n: int) -> np.ndarray:
        """Draw samples from a named distribution or backend."""
        from .backend.abc import Backend

        dist = self.distributions[distribution]
        if isinstance(dist, Backend):
            return dist.sample(n)
        else:
            samples = dist.sample((n,))
            if hasattr(samples, "numpy"):
                samples = samples.numpy()
            return np.asarray(samples)

    def build_dataframe(
        self,
        distribution: str = "posterior",
        num_samples=10_000,
    ) -> pd.DataFrame:
        """Build a posterior sample table from a named distribution.

        Parameters:
            distribution: Key in ``self.distributions`` to sample from.
                Defaults to ``"posterior"``.
            num_samples: Number of samples to draw. Defaults to 10_000.

        Returns:
            Table with parameter samples.
        """

        samples = self._sample_from(distribution, num_samples)

        dict_of_params = {
            name: parameters for name, parameters in zip(self.parameter_names, samples.T)
        }

        return pd.DataFrame.from_dict(dict_of_params)

    def build_inference_data(
        self,
        distribution: str = "posterior",
        num_samples: int = 10_000,
        include_prior: bool = True,
        include_observed: bool = True,
        include_log_likelihood: bool = True,
    ) -> az.InferenceData:
        """Build an ArviZ InferenceData object from the fitted model.

        Mirrors ``build_dataframe`` but returns a richer container that
        is understood by the ArviZ ecosystem (diagnostics, plotting, I/O,
        and model comparison via ``az.compare``).

        Parameters:
            distribution: Key in ``self.distributions`` to sample posterior
                draws from. Defaults to ``"posterior"``.
            num_samples: Number of posterior draws. Defaults to 10_000.
            include_prior: If ``True``, draw prior samples and attach them as
                the *prior* group. Defaults to ``True``.
            include_observed: If ``True``, attach the observed spectrum as the
                *observed_data* group. Requires an active XSPEC session.
                Defaults to ``True``.
            include_log_likelihood: If ``True``, simulate the posterior
                samples through the instrument response and compute
                per-bin Poisson log-likelihoods. Required for
                ``az.compare``. Defaults to ``True``.

        Returns:
            Inference data with *posterior* (and optionally *prior*,
            *observed_data*, *log_likelihood*) groups.
        """

        samples = self._sample_from(distribution, num_samples)

        posterior_dict = {
            name: samples[:, i][np.newaxis, :]
            for i, name in enumerate(self.parameter_names)
        }

        kwargs: dict = {"posterior": posterior_dict}

        if include_log_likelihood:
            sim = self.simulate(
                samples, return_kind="full_model_counts", progress_bar=True
            )
            observed = self.observed_spectrum
            log_lik = likelihood_per_bin(observed, sim["total_model_counts"])
            kwargs["log_likelihood"] = {
                "spectrum": log_lik[np.newaxis, :, :]
            }

        if include_prior:
            prior_samples = self.distributions["prior"].sample((num_samples,))
            if hasattr(prior_samples, "numpy"):
                prior_samples = prior_samples.numpy()
            prior_samples = np.asarray(prior_samples)

            prior_dict = {
                name: prior_samples[:, i][np.newaxis, :]
                for i, name in enumerate(self.parameter_names)
            }
            kwargs["prior"] = prior_dict

        if include_observed:
            obs = self.observed_spectrum
            kwargs["observed_data"] = {"spectrum": obs}

        if self.fit_result is not None:
            kwargs["attrs"] = {
                "log_Z": self.fit_result.log_Z,
                "log_Z_err": self.fit_result.log_Z_err,
            }

        idata = az.from_dict(**kwargs)

        return idata

    def plot_predictive_coverage(self, distribution="posterior", **kwargs):
        from .analysis.plotting import plot_predictive_coverage

        return plot_predictive_coverage(self, distribution=distribution, **kwargs)

    def run(self, *args, **kwargs) -> FitResults:
        from .backend import get_backend_class

        backend_cls = get_backend_class(self.backend_name)
        init_kwargs, run_kwargs = _split_kwargs(backend_cls, kwargs)

        self.backend = backend_cls(solver=self, trace=self.trace, **init_kwargs)

        self.fit_result = self.backend.run(*args, **run_kwargs)
        self.posterior_samples = self.fit_result.posterior_samples
        self.distributions["posterior"] = self.backend

        self.backend.tracer.plot_progress()
        self.backend.tracer.save()

        return self.fit_result
