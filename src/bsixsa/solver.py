import collections
import contextlib
import importlib.util
import os
import shutil
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pathos.multiprocessing as multiprocessing
import xspec
from scipy.stats import poisson

from .backend.config import BackendConfig
from .convenience import available_cpu_count, iter_free_parameters
from .priors import MultipleIndependent, build_prior
from .xspec import RETURN_KINDS, loaded_spectra, parallel_folding

if TYPE_CHECKING:
    import arviz as az
    from matplotlib.figure import Figure


@dataclass
class FitResults:
    """Outcome of `Solver.run`, also stored as `Solver.fit_result`.

    Attributes:
        time: Wall-clock time of the backend run, in seconds.
        posterior_samples: Posterior draws in physical space, one column per
            fitted parameter, named as `Solver.parameter_names`.
        n_likelihood_evaluations: Number of folds of the backend run (SIXSA's history matching included).
        log_Z: Natural-log evidence, with the likelihood
            $e^{-C/2}$ of XSPEC's C-statistic. It differs from the full Poisson
            evidence by a data-only constant, so compare it only between
            models fitted to the same spectrum and binning. `NaN` for `Emcee`,
            `Iminuit` and `LevenbergMarquardt`.
        log_Z_err: Uncertainty on `log_Z`.
        best_fit: Best-fit parameter vector, if the backend reports one.
        best_fit_stat: cstat plus XSPEC's prior term (as with `Fit.bayes on`) at `best_fit`.
    """

    time: float
    posterior_samples: pd.DataFrame
    n_likelihood_evaluations: int
    log_Z: float
    log_Z_err: float
    best_fit: pd.Series | None = None
    best_fit_stat: float | None = None


def likelihood_per_bin(observed: np.ndarray, model: np.ndarray) -> np.ndarray:
    r"""Per-bin Poisson log-probability of the observed counts.

    For a bin with observed counts $S_i$ and predicted counts $M_i$
    (`scipy.stats.poisson.logpmf`):

    $$
    \log p(S_i \mid M_i) = S_i \ln M_i - M_i - \ln(S_i!)
    $$

    Summed over the bins, this is $-C/2$ ($C$: XSPEC's cstat) plus a
    data-only constant. This per-bin form is what ArviZ's `loo`, `waic` and
    `compare` expect.

    Parameters:
        observed: Observed integer counts, shape `(n_bins,)`.
        model: Predicted counts, shape `(n_bins,)` or `(n_samples, n_bins)`.

    Returns:
        Per-bin log-probability with the same shape as *model*.
    """
    model = np.asarray(model, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    return poisson.logpmf(observed, model)


def _check_finite(parameters: np.ndarray, names: Sequence[str]) -> None:
    """Raise `ValueError` naming the parameters that hold NaN or Inf."""
    bad = ~np.isfinite(parameters)
    if bad.any():
        details = [
            f"{names[column]} (samples {np.flatnonzero(bad[:, column]).tolist()})"
            for column in np.flatnonzero(bad.any(axis=0))
        ]
        raise ValueError(
            f"Cannot fold parameter vectors containing NaN or Inf: {', '.join(details)}. "
            "Check the sampler output or the prior support."
        )


def _check_sampler_installed(config: BackendConfig) -> None:
    """Raise `ImportError` if the sampler package of *config* is not installed.

    Looked up without importing: no torch code may load before the folding pool is forked.
    """
    if config.required_package is None:
        return
    module, pip_name = config.required_package
    if importlib.util.find_spec(module) is None:
        raise ImportError(
            f"The {type(config).__name__} backend needs the '{pip_name}' package, "
            f"which is not installed: pip install {pip_name}"
        )


def _is_or_contains(directory: str, path: str) -> bool:
    """Whether *path* is *directory* or lies inside it (by file identity, robust to case and symlinks)."""
    path = os.path.realpath(path)
    while True:
        if os.path.exists(path) and os.path.samefile(directory, path):
            return True
        parent = os.path.dirname(path)
        if parent == path:  # reached the root
            return False
        path = parent


def _check_output_directory(outputfiles_basename: str, *, overwrite: bool) -> str:
    """Check that the output directory can be used (never the cwd, home, or their parents); return its normalised path."""
    path = os.path.normpath(outputfiles_basename)
    if not os.path.exists(path):
        return path
    if not os.path.isdir(path):
        raise ValueError(f"Output path '{outputfiles_basename}' exists and is not a directory.")
    if not os.listdir(path):
        return path
    if not overwrite:
        raise FileExistsError(
            f"Output directory '{path}' is not empty. Pass overwrite=True to delete its "
            "contents, or choose another directory."
        )
    if any(_is_or_contains(path, protected) for protected in (os.getcwd(), os.path.expanduser("~"))):
        raise ValueError(
            f"Refusing to clear '{outputfiles_basename}': it is, or contains, the current working "
            "directory or your home directory. Use a dedicated output directory, "
            "e.g. outputfiles_basename='fit-results'."
        )
    return path


def _prepare_output_directory(outputfiles_basename: str, *, overwrite: bool) -> str:
    """Create the output directory, emptying it when *overwrite* is `True`; return it with a trailing separator."""
    path = _check_output_directory(outputfiles_basename, overwrite=overwrite)
    os.makedirs(path, exist_ok=True)
    for entry in os.listdir(path):
        entry_path = os.path.join(path, entry)
        if os.path.isdir(entry_path) and not os.path.islink(entry_path):
            shutil.rmtree(entry_path)
        else:
            os.remove(entry_path)
    return os.path.join(path, "")


class Solver:
    """Bayesian fit of the spectra and models currently loaded in PyXspec.

    The solver reads the data and the models from the live XSPEC session,
    builds the prior, and hands the log-likelihood $-C/2$ ($C$: the Cash
    statistic, XSPEC's cstat, summed over the loaded spectra) to a backend: a
    nested sampler, an MCMC sampler, an optimiser or SIXSA.

    Example:
        ```python
        solver = Solver(prior, backend=Nautilus())
        result = solver.run()
        idata = solver.build_inference_data()
        ```

    Attributes:
        prior: The joint prior built from the specification.
        distributions: What `sample_parameters` draws from: `"prior"` and
            `"posterior"`. You may add your own entries (a bsixsa backend or
            a torch/sbi distribution).
        parameter_indices: XSPEC number of each fitted parameter within its model.
        model_names: XSPEC model name of each fitted parameter (`""` for the unnamed model).
        outputfiles_basename: Output directory, with a trailing separator.
        backend_config: The backend configuration passed as `backend`.
        backend: The backend of the last `run`, or `None`.
        fit_result: Results of the last `run`, or `None`.

    Note: Extending the solver
        To fit another forward model, subclass `Solver`, call `_setup(...)`
        from your `__init__` instead of reading XSPEC, and override
        `_fold_batch` (returns a dict shaped like
        [`parallel_folding`][bsixsa.xspec.parallel_folding]), `_get_pool`,
        `_check_xspec_state`, `_xspec_constraints`, `observed_counts` and
        `n_bins` as needed.
    """

    def __init__(
        self,
        prior: Sequence[tuple],
        outputfiles_basename: str = "./sixsa",
        overwrite: bool = False,
        n_jobs: int | None = None,
        *,
        backend: BackendConfig,
    ):
        """Read the XSPEC session and build the prior.

        Parameters:
            prior: Prior entries, see [`build_prior`][bsixsa.priors.build_prior].
            outputfiles_basename: Directory for backend output files (chains, checkpoints, traces).
            overwrite: Delete the content of a non-empty output directory.
            n_jobs: Number of folding worker processes (default: available CPUs).
            backend: Backend configuration, e.g. `Nautilus()` or `SIXSA()`.

        Raises:
            TypeError: If *backend* has the wrong type.
            ImportError: If the sampler package of the backend is not installed.
            ValueError: If no spectrum or model is loaded, or the prior is invalid.
            FileExistsError: If the output directory is not empty and *overwrite* is `False`.
        """
        if isinstance(backend, type):
            raise TypeError(
                f"backend must be a configuration instance: write backend={backend.__name__}() "
                f"(with any options), not backend={backend.__name__}."
            )
        if not isinstance(backend, BackendConfig) or not hasattr(backend, "backend_name"):
            raise TypeError(
                "backend must be a backend configuration such as bsixsa.Nautilus(), Nessai(), "
                f"Ultranest(), SIXSA(), Emcee(), Iminuit() or LevenbergMarquardt(); got {backend!r}."
            )
        _check_sampler_installed(backend)
        if xspec.AllData.nSpectra < 1:
            raise ValueError(
                "No spectrum is loaded in XSPEC: load one (e.g. xspec.Spectrum('source.pha')) "
                "before creating the Solver."
            )
        if len(xspec.AllModels.sources) < 1:
            raise ValueError(
                "No model is defined in XSPEC: define one (e.g. xspec.Model('tbabs*powerlaw')) "
                "before creating the Solver."
            )

        # Checks without side effects first: build_prior edits the XSPEC session.
        _check_output_directory(outputfiles_basename, overwrite=overwrite)

        prior, parameter_indices, model_names = build_prior(prior)
        # XSPEC's free parameters are now exactly the prior axes, in prior order.
        parameter_names = [parameter.name for parameter in iter_free_parameters()]

        self._setup(
            prior=prior,
            parameter_indices=parameter_indices,
            model_names=model_names,
            parameter_names=parameter_names,
            outputfiles_basename=outputfiles_basename,
            overwrite=overwrite,
            n_jobs=n_jobs,
            backend=backend,
        )

    def _setup(
        self,
        *,
        prior: MultipleIndependent,
        parameter_indices: Sequence[int],
        model_names: Sequence[str],
        parameter_names: Sequence[str],
        outputfiles_basename: str,
        overwrite: bool,
        n_jobs: int | None,
        backend: BackendConfig,
    ) -> None:
        """Initialise every attribute that does not need XSPEC (see "Extending the solver").

        Parameters:
            prior: Joint prior, typically a [`MultipleIndependent`][bsixsa.priors.MultipleIndependent].
            parameter_indices: XSPEC parameter numbers, in prior order.
            model_names: Model names, in prior order.
            parameter_names: Parameter names, in prior order.
            outputfiles_basename: Output directory.
            overwrite: Whether a non-empty output directory may be cleared.
            n_jobs: Number of folding worker processes (default: available CPUs).
            backend: Backend configuration.
        """
        if not len(parameter_names) == len(model_names) == len(parameter_indices):
            raise ValueError(
                f"parameter_names ({len(parameter_names)} entries) and model_names "
                f"({len(model_names)}) must have one entry per fitted parameter "
                f"({len(parameter_indices)} in parameter_indices)."
            )
        self.parameter_indices = parameter_indices
        self.model_names = model_names
        self._parameter_names = tuple(parameter_names)
        self.prior = prior
        self.distributions = {"prior": prior}
        self.backend_config = backend
        self.backend = None
        self.fit_result: FitResults | None = None
        self._n_jobs = available_cpu_count() if n_jobs is None else n_jobs
        self.pool = None
        self._pool_depth = 0
        self._fold_counts: collections.Counter[str] = collections.Counter()
        self._fold_phase_name = "backend"
        # Before the output directory is emptied: a configuration error must not clear it.
        self.backend_config.validate_for_solver(self)
        self.outputfiles_basename = _prepare_output_directory(outputfiles_basename, overwrite=overwrite)

    # ------------------------------------------------------------------
    # The problem
    # ------------------------------------------------------------------

    @property
    def parameter_names(self) -> list[str]:
        """Names of the fitted parameters in prior order, e.g. `TBabs_1_nH` or `cluster_bapec_2_kT`.

        The order is models by source number, then XSPEC parameter number,
        not the order of the prior list.
        """
        return list(self._parameter_names)

    @property
    def num_parameters(self) -> int:
        """Number of fitted parameters (dimension of the prior)."""
        return len(self.parameter_indices)

    @property
    def indexes(self) -> Sequence[int]:
        """Old name of `parameter_indices`, kept for existing scripts."""
        return self.parameter_indices

    @property
    def model_indexes(self) -> Sequence[str]:
        """Old name of `model_names`, kept for existing scripts."""
        return self.model_names

    @property
    def backend_name(self) -> str:
        """Name of the configured backend, e.g. `"nautilus"`."""
        return self.backend_config.backend_name

    @property
    def posterior_samples(self) -> pd.DataFrame | None:
        """Posterior draws of the last run (physical units, one column per parameter); `None` before `run`."""
        return None if self.fit_result is None else self.fit_result.posterior_samples

    @property
    def observed_counts(self) -> np.ndarray:
        """Observed counts per bin, rounded to integers (float64, shape `(n_bins,)`).

        The noticed bins of every loaded spectrum, concatenated in spectrum
        order, as the model counts returned by `simulate`.
        """
        return np.rint(np.concatenate([state.observed_counts for state in loaded_spectra()]))

    @property
    def observed_spectrum(self) -> np.ndarray:
        """Same as `observed_counts`; kept for older scripts."""
        return self.observed_counts

    @property
    def n_bins(self) -> int:
        """Number of noticed bins, summed over the loaded spectra."""
        return int(len(self.observed_counts))

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def run(self) -> FitResults:
        """Fit the problem with the configured backend.

        Returns:
            The [`FitResults`][bsixsa.solver.FitResults], also stored as `fit_result`.

        Raises:
            RuntimeError: If `xspec.Fit.bayes` is not `"off"` (the prior would be counted twice).
        """
        from .backend import get_backend_class

        _check_sampler_installed(self.backend_config)  # also when backend_config was replaced
        with self.pool_scope():
            self._check_xspec_state()
            self._get_pool()  # fork the workers before any torch work
            self._fold_phase_name = "backend"
            self.backend = get_backend_class(self.backend_name)(solver=self, config=self.backend_config)
            self.fit_result = self.backend.run()
            self._fold_phase_name = "post"
            self.distributions["posterior"] = self.backend
            self.backend.tracer.plot_progress()
            self.backend.tracer.save()
        return self.fit_result

    # ------------------------------------------------------------------
    # After the run
    # ------------------------------------------------------------------

    def sample_parameters(self, num_samples: int, distribution: str = "prior") -> np.ndarray:
        """Draw parameter vectors from one of the solver's `distributions`.

        Parameters:
            num_samples: Number of draws.
            distribution: `"prior"`, `"posterior"`, or a key you added to
                `distributions`.

        Returns:
            Draws in physical units, shape `(n, num_parameters)`. Backends
            with a finite posterior sample may return fewer rows.

        Raises:
            KeyError: If *distribution* is not available, e.g. `"posterior"` before `run`.
        """
        if distribution not in self.distributions:
            raise KeyError(
                f"No {distribution!r} distribution: the available ones are "
                f"{sorted(self.distributions)}. 'posterior' exists once run() has returned."
            )
        from .backend.abc import Backend

        source = self.distributions[distribution]
        # A bsixsa backend takes a number of draws, other distributions a sample shape.
        draws = source.sample(num_samples) if isinstance(source, Backend) else source.sample((num_samples,))
        return np.asarray(draws)

    def build_dataframe(
        self,
        distribution: str = "posterior",
        num_samples: int = 10_000,
    ) -> pd.DataFrame:
        """Draw parameter vectors into a table with one column per parameter.

        Parameters:
            distribution: Key of `distributions` to draw from (see `sample_parameters`).
            num_samples: Number of draws requested (may return fewer, see `sample_parameters`).

        Returns:
            Table of draws, with columns `parameter_names`.
        """
        return pd.DataFrame(self.sample_parameters(num_samples, distribution), columns=self.parameter_names)

    def export_xspec_chain(
        self,
        path: str | os.PathLike,
        *,
        samples: pd.DataFrame | np.ndarray | None = None,
        fit_statistic: np.ndarray | None = None,
        overwrite: bool = False,
    ) -> str:
        """Write the posterior samples as an XSPEC chain file, for XSPEC's chain-based errors.

        Shortcut for [`export_xspec_chain`][bsixsa.export.export_xspec_chain].

        Example:
            ```python
            path = solver.export_xspec_chain("posterior_chain.fits")
            xspec.AllChains.clear()
            xspec.AllChains += path
            xspec.Fit.error("1-3")  # 90% credible intervals from the samples
            ```

        Parameters:
            path: Output FITS file.
            samples: Parameter values to write instead of the posterior samples.
            fit_statistic: Fit statistic of each sample (default: folded through XSPEC).
            overwrite: Whether an existing file may be replaced.

        Returns:
            Absolute path of the written file, as `xspec.AllChains +=` expects.
        """
        from .export import export_xspec_chain

        return export_xspec_chain(self, path, samples=samples, fit_statistic=fit_statistic, overwrite=overwrite)

    def build_inference_data(
        self,
        distribution: str = "posterior",
        num_samples: int = 10_000,
        include_prior: bool = True,
        include_observed: bool = True,
        include_log_likelihood: bool = True,
    ) -> "az.InferenceData":
        """Build an ArviZ `InferenceData` object, e.g. for `az.loo` and `az.compare`.

        Parameters:
            distribution: Key of `distributions` for the *posterior* group (see `sample_parameters`).
            num_samples: Number of draws requested.
            include_prior: Add a *prior* group drawn from the prior.
            include_observed: Add the observed counts as *observed_data*.
            include_log_likelihood: Fold the draws and store their per-bin
                Poisson log-likelihoods (see
                [`likelihood_per_bin`][bsixsa.solver.likelihood_per_bin]),
                over the bins of all spectra (see `observed_counts`).

        Returns:
            Inference data with *posterior* (and optionally *prior*,
            *observed_data*, *log_likelihood*) groups.
        """
        import arviz as az

        names = self.parameter_names

        def as_chain(samples: np.ndarray) -> dict[str, np.ndarray]:
            # ArviZ arrays are (chain, draw): the draws form one chain.
            return {name: column[np.newaxis] for name, column in zip(names, samples.T)}

        samples = self.sample_parameters(num_samples, distribution)
        contents: dict = {"posterior": as_chain(samples)}
        if include_log_likelihood or include_observed:
            observed = self.observed_counts
        if include_log_likelihood:
            model_counts = self.simulate(samples, return_kind="full_model_counts")["total_model_counts"]
            contents["log_likelihood"] = {"spectrum": likelihood_per_bin(observed, model_counts)[np.newaxis]}
        if include_prior:
            contents["prior"] = as_chain(self.sample_parameters(num_samples, "prior"))
        if include_observed:
            contents["observed_data"] = {"spectrum": observed}
        if self.fit_result is not None:
            contents["attrs"] = {"log_Z": self.fit_result.log_Z, "log_Z_err": self.fit_result.log_Z_err}
        return az.from_dict(**contents)

    def plot_predictive_coverage(self, distribution: str = "posterior", **kwargs: Any) -> "Figure":
        """Compare the observed spectra with spectra simulated from `distribution`.

        A prior or posterior predictive check. Shortcut for
        [`plot_predictive_coverage`][bsixsa.analysis.plotting.plot_predictive_coverage].

        Parameters:
            distribution: Key of `distributions` to draw from (see `sample_parameters`).
            **kwargs: Forwarded to `plot_predictive_coverage`.

        Returns:
            Figure with a spectrum and a residual panel per plotted spectrum.
        """
        from .analysis.plotting import plot_predictive_coverage

        return plot_predictive_coverage(self, distribution=distribution, **kwargs)

    # ------------------------------------------------------------------
    # Likelihood and folding
    # ------------------------------------------------------------------

    def simulate(
        self,
        parameters: np.ndarray,
        return_kind: str = "full_model_counts",
        progress_bar: bool = True,
        no_pool: bool = False,
    ) -> dict:
        """Fold parameter vectors through the model and the instrument response.

        Every fold is counted in `fold_counts` under the current `fold_phase`.

        Parameters:
            parameters: Physical parameter values, shape `(n, num_parameters)`, prior order.
            return_kind: `"cstat"` (C-statistic only), `"full_model_counts"`
                (also predicted counts per bin) or `"models_and_components"`
                (also per model and per additive component).
            progress_bar: Whether to show a progress bar.
            no_pool: Fold serially in this process.

        Returns:
            C-stat and predicted counts, stacked as returned by
            [`parallel_folding`][bsixsa.xspec.parallel_folding].

        Raises:
            ValueError: If *parameters* has the wrong shape or contains NaN or Inf.
        """
        if return_kind not in RETURN_KINDS:
            raise ValueError(f"return_kind must be one of {RETURN_KINDS}, got {return_kind!r}.")
        d = self.num_parameters
        parameters = np.asarray(parameters, dtype=np.float64)
        if parameters.size == 0:
            parameters = parameters.reshape(0, d)
        if parameters.ndim != 2 or parameters.shape[1] != d:
            raise ValueError(
                f"parameters must have shape (n, {d}): one row per parameter vector, one column "
                "per fitted parameter in prior order (see Solver.parameter_names); "
                f"got {parameters.shape}."
            )
        _check_finite(parameters, self.parameter_names)
        if len(parameters) == 0:
            return self._empty_fold_outputs(return_kind)
        self._fold_counts[self._fold_phase_name] += len(parameters)

        desc = "Evaluating C_stat - " if return_kind == "cstat" else ""
        with contextlib.nullcontext() if no_pool else self.pool_scope():
            return self._fold_batch(
                parameters,
                return_kind=return_kind,
                desc=desc,
                progress_bar=progress_bar,
                pool=None if no_pool else self._get_pool(),
            )

    def _empty_fold_outputs(self, return_kind: str) -> dict:
        """Outputs of a fold over zero parameter vectors."""
        outputs = {"cstat": np.empty(0, dtype=np.float64)}
        if return_kind in ("full_model_counts", "models_and_components"):
            outputs["total_model_counts"] = np.empty((0, self.n_bins), dtype=np.float64)
        if return_kind == "models_and_components":
            outputs["model_counts"] = {}
            outputs["component_counts"] = {}
        return outputs

    def log_likelihood_fn(
        self,
        theta: np.ndarray,
        x_o: object = None,
        progress_bar: bool = True,
        no_pool: bool = False,
    ) -> np.ndarray:
        r"""Log-likelihood $\log \mathcal{L}(\theta) = -C(\theta)/2$, with $C$ the Cash statistic (XSPEC's cstat).

        Parameters:
            theta: Physical parameters, shape `(n, num_parameters)`, prior order.
            x_o: Ignored.
            progress_bar: Whether to show a progress bar.
            no_pool: Fold serially in this process.

        Returns:
            Array of shape `(n,)`.
        """
        outputs = self.simulate(theta, return_kind="cstat", progress_bar=progress_bar, no_pool=no_pool)
        return -0.5 * outputs["cstat"]

    def log_prior_fn(self, theta: np.ndarray, x_o: object = None) -> np.ndarray:
        """Log-density of the prior, shape `(n,)`; *x_o* is ignored."""
        return self.prior.log_prob(theta)

    def log_prob_fn(self, theta: np.ndarray, x_o: object = None) -> np.ndarray:
        r"""Unnormalised log-posterior $-C(\theta)/2 + \log \pi(\theta)$, shape `(n,)`; *x_o* is ignored."""
        return self.log_likelihood_fn(theta) + self.log_prior_fn(theta)

    @contextlib.contextmanager
    def fold_phase(self, name: str) -> Iterator[None]:
        """Attribute the folds performed inside the block to phase *name*.

        Folds outside any phase count as `"backend"`, or `"post"` after a `run`.
        """
        if not isinstance(name, str):
            raise TypeError(f"fold phase name must be a str, got {type(name).__name__}.")
        previous = self._fold_phase_name
        self._fold_phase_name = name
        try:
            yield
        finally:
            self._fold_phase_name = previous

    @property
    def fold_counts(self) -> dict[str, int]:
        """Number of folds performed so far, per phase."""
        return dict(self._fold_counts)

    @contextlib.contextmanager
    def pool_scope(self) -> Iterator[None]:
        """Share one folding worker pool across the folds made inside the block.

        Nested scopes reuse the pool, which is terminated when the outermost
        scope exits.

        Example:
            ```python
            with solver.pool_scope():
                idata = solver.build_inference_data()
                fig = solver.plot_predictive_coverage()
            ```
        """
        self._pool_depth += 1
        try:
            yield
        finally:
            self._pool_depth -= 1
            if self._pool_depth <= 0:
                self.close_pool()

    def close_pool(self) -> None:
        """Terminate and drop the worker pool if one is open. Idempotent."""
        pool = self.pool
        self.pool = None
        self._pool_depth = 0
        if pool is not None:
            try:
                pool.terminate()
                pool.join()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Hooks (see "Extending the solver")
    # ------------------------------------------------------------------

    def _get_pool(self):
        """Create the worker pool on first use; call it only inside `pool_scope()`."""
        if self.pool is None:
            self.pool = multiprocessing.Pool(processes=self._n_jobs)
        return self.pool

    def _fold_batch(
        self,
        parameters: np.ndarray,
        *,
        return_kind: str,
        desc: str,
        progress_bar: bool,
        pool: Any,
    ) -> dict:
        """Forward model: fold a checked, non-empty `(n, d)` batch through XSPEC (already counted by `simulate`)."""
        return parallel_folding(
            parameters,
            self.parameter_indices,
            self.model_names,
            desc=desc,
            progress_bar=progress_bar,
            pool=pool,
            return_kind=return_kind,
        )

    def _check_xspec_state(self) -> None:
        """Raise `RuntimeError` if `xspec.Fit.bayes` is on, which would count the prior twice."""
        value = xspec.Fit.bayes
        if value.lower() != "off":
            raise RuntimeError(
                f"xspec.Fit.bayes is {value!r}: the fit statistic would include "
                "XSPEC's prior on top of the bsixsa prior, counting the prior "
                "twice. Set xspec.Fit.bayes = 'off'."
            )

    def _xspec_constraints(self) -> dict:
        """Session facts a backend may restrict: `n_spectra`, `stat_method` and `has_background`."""
        try:
            has_background = xspec.AllData(1).background is not None
        except Exception:  # PyXspec raises "Spectrum has no background."
            has_background = False
        return {
            "n_spectra": int(xspec.AllData.nSpectra),
            "stat_method": xspec.Fit.statMethod,
            "has_background": has_background,
        }
