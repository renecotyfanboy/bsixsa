import contextlib
import hashlib
import json
import os
import shutil
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xspec
from xspec import AllModels

from . import _cube
from .exceptions import BoxEdgeWarning
from .priors import build_prior, validate_finite_support
from .xspec import parallel_folding
from scipy.stats import poisson
import arviz as az
import pathos.multiprocessing as multiprocessing  # pathos


@dataclass
class FitResults:
    """Outcome of `Solver.run`.

    Attributes:
        time: Wall-clock time of the backend run, in seconds.
        posterior_samples: Posterior draws in physical space, one column per
            parameter in prior order.
        n_likelihood_evaluations: Folds spent by the backend, plus the
            restriction folds when this solver computed the restriction for
            this run (not on a re-run that reuses it).
        log_Z: Log evidence under the base prior. For a backend sampling a
            restricted box it already includes `log_Z_restriction_correction`.
        log_Z_err: Uncertainty on `log_Z`.
        best_fit: Best-fit parameter vector, if the backend reports one.
        best_fit_stat: Fit statistic (cstat plus the XSPEC-convention prior
            term of the base prior) at `best_fit`.
        n_restriction_evaluations: Folds spent by the history-matching
            restriction computed for this run (`0` if none, if the
            restriction was injected, or if an earlier run already reported
            it).
        restriction_time: Wall-clock time of that restriction, including its
            bounding box, in seconds.
        log_Z_restriction_correction: Log prior mass of the restricted box
            sampled by the backend (`0.0` without a box); added to `log_Z`
            when that is finite.
    """

    time: float
    posterior_samples: pd.DataFrame
    n_likelihood_evaluations: int
    log_Z: float
    log_Z_err: float
    best_fit: pd.Series | None = None
    best_fit_stat: float | None = None
    n_restriction_evaluations: int = 0
    restriction_time: float = 0.0
    log_Z_restriction_correction: float = 0.0


def likelihood_per_bin(observed: np.ndarray, model: np.ndarray) -> np.ndarray:
    r"""Per-bin Poisson log-likelihood (Cash statistic).
    For a bin with observed counts $S_i$ and predicted counts $\mathcal{M}_i$:

    $$
    \log \mathcal{L}_i = S_i \ln(\mathcal{M}_i) − \mathcal{M}_i
    $$

    where $\mathcal{M}_i > 0$

    Parameters:
        observed: Observed counts, shape `(n_bins,)`.
        model: Predicted model counts, shape `(n_bins,)` or
            `(n_samples, n_bins)`.

    Returns:
        Per-bin log-likelihood with the same shape as *model*.
    """

    model = np.asarray(model, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)

    return poisson.logpmf(observed, model)


def _validate_finite(parameters, parameter_names):
    """Raise `ValueError` if *parameters* contains NaN or Inf values."""
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


_RETURN_KINDS = ("cstat", "full_model_counts", "models_and_components")


def _check_output_directory(outputfiles_basename: str, *, overwrite: bool) -> str:
    """Check that the output directory can be used, without creating or clearing it.

    Run before anything with side effects (e.g. `build_prior`, which sets
    `xspec` parameters) so that an unusable directory fails first.

    Parameters:
        outputfiles_basename: Requested output directory.
        overwrite: Whether a non-empty directory may later be cleared.

    Returns:
        The normalised path, without a trailing separator.

    Raises:
        ValueError: If the path exists and is not a directory.
        FileExistsError: If the directory is not empty and *overwrite* is
            `False`.
    """
    normalized_output_dir = os.path.normpath(outputfiles_basename)
    if os.path.exists(normalized_output_dir):
        if not os.path.isdir(normalized_output_dir):
            raise ValueError(
                f"Output path '{outputfiles_basename}' exists and is not a directory."
            )
        if os.listdir(normalized_output_dir) and not overwrite:
            raise FileExistsError(
                f"Output directory '{normalized_output_dir}' is not empty. "
                "Pass overwrite=True to clear it."
            )
    return normalized_output_dir


def _prepare_output_directory(outputfiles_basename: str, *, overwrite: bool) -> str:
    """Setup the directory for the output of the run."""
    normalized_output_dir = _check_output_directory(outputfiles_basename, overwrite=overwrite)

    if not os.path.exists(normalized_output_dir):
        os.makedirs(normalized_output_dir)
    else:
        existing_entries = os.listdir(normalized_output_dir)
        if existing_entries:
            for entry in existing_entries:
                entry_path = os.path.join(normalized_output_dir, entry)
                if os.path.isdir(entry_path) and not os.path.islink(entry_path):
                    shutil.rmtree(entry_path)
                else:
                    os.remove(entry_path)

    if not normalized_output_dir.endswith(os.sep):
        normalized_output_dir = normalized_output_dir + os.sep

    return normalized_output_dir


def _canonical(value):
    """Map *value* to a JSON-serialisable form that is stable across runs.

    Every real number becomes ``repr(float(x))`` so that ``uniform(1, 3)`` and
    ``uniform(1.0, 3.0)`` (identical distributions whose ``args`` differ only in
    Python type) hash identically.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return repr(float(value))
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        return [_canonical(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, dict):
        return [[str(key), _canonical(value[key])] for key in sorted(value, key=str)]
    text = repr(value)
    if " at 0x" in text:
        # A default object repr embeds a memory address, which would make the
        # hash differ between otherwise identical processes.
        return type(value).__qualname__
    return text


# Quantile levels probed on every prior axis by `Solver.fingerprint`.
_FINGERPRINT_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _quantile_probe(dist) -> list | None:
    """Rounded quantiles of *dist*, a family-agnostic identity check for `fingerprint`.

    Opaque families (e.g. `rv_histogram`, whose histogram is not in `args`)
    are only told apart by their values. Rounding to 12 significant digits
    keeps the hash stable across SciPy versions that differ in the last ulp.

    Parameters:
        dist: Frozen distribution exposing `ppf`.

    Returns:
        List of canonical strings, or `None` if `ppf` cannot be evaluated.
    """
    try:
        values = np.asarray(dist.ppf(np.asarray(_FINGERPRINT_QUANTILES)), dtype=np.float64)
    except Exception:
        return None
    if values.shape != (len(_FINGERPRINT_QUANTILES),):
        return None
    return [_canonical(float(f"{value:.12g}")) for value in values]


def _axis_descriptor(dist) -> dict:
    """Canonical description of one prior marginal, for `Solver.fingerprint`.

    SciPy frozen distributions are normalised through `_parse_args`, so a
    `loc`/`scale` passed positionally or by keyword describes the same axis.
    A [`TruncatedMarginal`][bsixsa.priors.TruncatedMarginal] carries no
    `args`, so its base distribution and unit-cube interval are described
    instead; otherwise two truncations of different families sharing a
    support would hash identically. Every axis also carries a quantile probe,
    which separates families whose parameters are not in `args`.

    Parameters:
        dist: A frozen SciPy distribution or a `TruncatedMarginal`-like object.

    Returns:
        JSON-serialisable dict.
    """
    support = _canonical(tuple(dist.support()))
    quantiles = _quantile_probe(dist)
    base = getattr(dist, "base", None)
    if base is not None and hasattr(dist, "lo_u") and hasattr(dist, "hi_u"):
        return {
            "family": "truncated",
            "base": _axis_descriptor(base),
            "interval_u": _canonical((dist.lo_u, dist.hi_u)),
            "support": support,
            "quantiles": quantiles,
        }

    family = getattr(dist, "dist", None)
    name = getattr(family, "name", None)
    args = tuple(getattr(dist, "args", ()))
    kwds = dict(getattr(dist, "kwds", {}) or {})
    parse_args = getattr(family, "_parse_args", None)
    if callable(parse_args):
        try:
            shapes, loc, scale = parse_args(*args, **kwds)
        except Exception:
            pass
        else:
            args, kwds = tuple(shapes), {"loc": loc, "scale": scale}
    return {
        "family": str(name if name is not None else type(dist).__qualname__),
        "args": _canonical(args),
        "kwds": _canonical(kwds),
        "support": support,
        "quantiles": quantiles,
    }


def _plain(value):
    """Return *value* if it is a plain `str`/`int`/`float`, else `None`.

    Keeps `xspec` reads inert under a mocked module: a `MagicMock` attribute
    is never a plain value. `bool` is rejected so a mock flag is not read as
    a number.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)):
        return value
    return None


def _read_background_file(spectrum):
    """Background file name of an `xspec` spectrum.

    PyXspec raises when `Spectrum.background` is read on a spectrum without a
    background ("Error: Spectrum has no background."), so that exception means
    "no background", while any other failure means "unknown".

    Parameters:
        spectrum: An `xspec.Spectrum`-like object.

    Returns:
        The file name (`str`), `False` when no background is loaded, or
        `None` when it cannot be determined.
    """
    try:
        background = spectrum.background
    except Exception as err:
        return False if "no background" in str(err).lower() else None
    if background is None:
        return False
    file_name = getattr(background, "fileName", None)
    if isinstance(file_name, str):
        return file_name.strip() or False
    return None


def _is_restriction(value) -> bool:
    """Duck-typed test for a precomputed restriction.

    Parameters:
        value: Candidate object.

    Returns:
        `True` if *value* is an instance (not a class) exposing callable
        `truncated_prior` and `check_fingerprint` methods, like
        [`Restriction`][bsixsa.restriction.Restriction].
    """
    # The class itself has both methods too, unbound: `restrict_prior=Restriction`
    # would only fail at `run`, after the output directory was prepared.
    if isinstance(value, type):
        return False
    return callable(getattr(value, "truncated_prior", None)) and callable(
        getattr(value, "check_fingerprint", None)
    )


def _check_restrict_prior_arg(restrict_prior) -> None:
    """Refuse `restrict_prior` values that no restriction step can resolve.

    `None`, a `bool`, a [`HistoryMatching`][bsixsa.restriction.HistoryMatching]
    configuration and a precomputed restriction (duck-typed, see
    `_is_restriction`) are accepted. Anything else is refused here so that a
    typo such as `restrict_prior="yes"` or `restrict_prior=1` fails before the
    output directory or the `xspec` session is touched.

    Parameters:
        restrict_prior: The value passed to `Solver`.

    Raises:
        TypeError: If *restrict_prior* is of any other type.
    """
    if restrict_prior is None or isinstance(restrict_prior, (bool, np.bool_)):
        return
    message = (
        "restrict_prior must be None, a bool, a HistoryMatching configuration "
        f"or a precomputed Restriction, got {type(restrict_prior).__name__} "
        f"{restrict_prior!r}."
    )
    # Plain values are refused before the restriction package is imported.
    if isinstance(
        restrict_prior,
        (int, float, complex, str, bytes, list, tuple, dict, set, frozenset, np.number, np.ndarray),
    ):
        raise TypeError(message)
    if _is_restriction(restrict_prior):
        return
    from .restriction.config import HistoryMatching

    if not isinstance(restrict_prior, HistoryMatching):
        raise TypeError(message)


def _resolve_restrict_prior(restrict_prior, backend, prior) -> tuple:
    """Resolve a checked `restrict_prior` into a restriction request.

    | Value | Restriction | Configuration |
    |---|---|---|
    | `None` | resolved as `type(backend).restrict_prior_default` | |
    | `True` | `None` | `HistoryMatching()` |
    | `False` | `None` | `None` |
    | `HistoryMatching` | `None` | the value |
    | restriction | the value (injected) | its `config` |

    When a restriction is requested and the prior exposes `dists`, every
    axis must have a finite support.

    Parameters:
        restrict_prior: Value already accepted by `_check_restrict_prior_arg`.
        backend: Backend configuration object.
        prior: Base prior of the solver.

    Returns:
        `(restriction, config)`: the injected restriction or `None`, and the
        history-matching configuration or `None`.

    Raises:
        ValueError: If a restriction is requested on a prior with an infinite
            support.
    """
    if restrict_prior is None:
        default = getattr(type(backend), "restrict_prior_default", False)
        # Only a real bool counts: a mocked config class must not opt in.
        restrict_prior = bool(default) if isinstance(default, (bool, np.bool_)) else False

    restriction, config = None, None
    if isinstance(restrict_prior, (bool, np.bool_)):
        if restrict_prior:
            from .restriction.config import HistoryMatching

            config = HistoryMatching()
    elif _is_restriction(restrict_prior):
        restriction = restrict_prior
        config = getattr(restrict_prior, "config", None)
    else:
        config = restrict_prior

    if (restriction is not None or config is not None) and getattr(prior, "dists", None) is not None:
        validate_finite_support(prior)
    return restriction, config


def set_parameters(values, indexes, model_indexes):
    """Update the active `xspec` parameters using transformed values.

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


class Solver(object):
    allowed_stats = ["cstat", "pstat"]

    # Class-level defaults keep instances built with `Solver.__new__` (test
    # fixtures that bypass `__init__`) valid for the restriction and fold
    # accounting code paths. The fold-count dict itself is created lazily per
    # instance: a class-level dict would be shared by every solver.
    restriction = None
    restriction_config = None
    _restriction_injected: bool = False
    _restriction_accounted: bool = False
    _fold_phase_name: str = "backend"
    _restrict_prior_arg = None

    def __init__(
        self,
        prior,
        outputfiles_basename="./sixsa",
        overwrite=False,
        n_jobs=None,
        *,
        backend,
        restrict_prior=None,
    ):
        r"""Initialise the Bayesian X-ray spectral analysis solver.

        Reads the currently loaded `xspec` model, builds the prior
        distribution from *prior*, and prepares a multiprocessing pool
        for parallel likelihood evaluations.

        Parameters:
            prior: Prior specification forwarded to
                [`build_prior`][bsixsa.priors.build_prior].  Accepts the same formats as
                [`build_prior`][bsixsa.priors.build_prior].
            outputfiles_basename: Directory where backend output files
                (chains, checkpoints, traces) are stored.  Created
                automatically if it does not exist.
            overwrite: If `True`, clear *outputfiles_basename* when it
                already contains files.  Defaults to `False`.
            n_jobs: Number of worker processes for the multiprocessing
                pool.  Defaults to the number of CPUs available to this
                process (affinity-aware: respects a SLURM/cgroup cpuset on
                Linux, falling back to the full core count on macOS).
            backend: Backend configuration object. The solver infers the
                backend at construction time from this mandatory config.
            restrict_prior: Prior restriction by history matching. `None`
                uses the backend's `restrict_prior_default` (only SIXSA
                restricts by default), `True` a default
                [`HistoryMatching`][bsixsa.restriction.HistoryMatching],
                `False` none; a `HistoryMatching` sets the configuration,
                and a precomputed
                [`Restriction`][bsixsa.restriction.Restriction] is reused
                as is (checked against `fingerprint()` at `run`, its folds
                are not counted again). The restriction is computed at the
                start of `run`, or earlier by calling `restrict`.

        Raises:
            TypeError: If *backend* does not define a `backend_name`, or if
                *restrict_prior* is not one of the accepted types.
            ValueError: If no spectrum or no model is loaded in `xspec`, if
                the output path exists and is not a directory, or if a
                restriction is requested on a prior with an infinite
                support.
            FileExistsError: If the output directory is not empty and
                *overwrite* is `False`. Raised before `build_prior` touches
                the `xspec` parameters.
        """
        if not hasattr(backend, "backend_name"):
            raise TypeError("`config` must define a backend_name.")

        if xspec.AllData.nSpectra < 1:
            raise ValueError("No spectra loaded in XSPEC")

        if len(xspec.AllModels.sources) < 1:
            raise ValueError("No model loaded in XSPEC")

        # Cheap, side-effect-free checks first: build_prior sets every prior
        # parameter in the xspec session, and the directory is only cleared by
        # _setup once the prior is known to be valid.
        _check_restrict_prior_arg(restrict_prior)
        _check_output_directory(outputfiles_basename, overwrite=overwrite)

        prior, indexes, model_indexes = build_prior(prior)
        self.nb_models = len(AllModels.sources)

        self._setup(
            prior=prior,
            indexes=indexes,
            model_indexes=model_indexes,
            outputfiles_basename=outputfiles_basename,
            overwrite=overwrite,
            n_jobs=n_jobs,
            backend=backend,
            restrict_prior=restrict_prior,
        )

    def _setup(
        self,
        *,
        prior,
        indexes,
        model_indexes,
        outputfiles_basename,
        overwrite,
        n_jobs,
        backend,
        restrict_prior=None,
    ) -> None:
        """Initialise every attribute that does not require `xspec`.

        Split from `__init__` so that subclasses can build a fully functional
        solver around an analytic forward model (see `tests/_analytic.py`).
        The prior is stored as given: its `dists` are only inspected (for a
        finite support) when a restriction is requested.

        *restrict_prior* is resolved (see `_resolve_restrict_prior`) into
        `restriction_config` and, for a precomputed restriction,
        `restriction`, before the output directory is touched.

        Parameters:
            prior: Joint prior, typically a
                [`MultipleIndependent`][bsixsa.priors.MultipleIndependent].
            indexes: Parameter indexes aligned with the prior axes.
            model_indexes: Model names aligned with *indexes*.
            outputfiles_basename: Output directory, prepared (and cleared when
                *overwrite* is `True`) here.
            overwrite: Whether a non-empty output directory may be cleared.
            n_jobs: Worker processes for the folding pool, or `None` for the
                number of available CPUs.
            backend: Backend configuration object.
            restrict_prior: Prior restriction request (see `Solver`).

        Raises:
            TypeError: If *restrict_prior* is not one of the accepted types.
            ValueError: If a restriction is requested on a prior with an
                infinite support, or if the output path exists and is not a
                directory.
            FileExistsError: If the output directory is not empty and
                *overwrite* is `False`.
        """
        from .convenience import available_cpu_count

        _check_restrict_prior_arg(restrict_prior)
        restriction, restriction_config = _resolve_restrict_prior(restrict_prior, backend, prior)
        self.outputfiles_basename = _prepare_output_directory(
            outputfiles_basename,
            overwrite=overwrite,
        )

        self.indexes = indexes
        self.model_indexes = model_indexes
        # `base_prior` is never replaced; `prior` is the working prior that a
        # restriction may later swap for a truncated one.
        self.base_prior = prior
        self.prior = prior
        self.distributions = {"prior": prior}
        self.posterior_samples = None
        self.backend_config = backend
        self.backend_name: str = self.backend_config.backend_name
        self.backend = None
        self.fit_result: FitResults | None = None
        self._n_jobs = available_cpu_count() if n_jobs is None else n_jobs
        self.pool = None
        self._pool_depth = 0
        self._fold_counts: dict[str, int] = {}
        self._fold_phase_name = "backend"
        self._restrict_prior_arg = restrict_prior
        self.restriction = restriction
        self.restriction_config = restriction_config
        self._restriction_injected = restriction is not None
        self._restriction_accounted = False
        # Validated last so that config checks see a fully initialised solver.
        self.backend_config.validate_for_solver(self)

    def _get_pool(self):
        """Materialise the worker pool lazily. Only valid inside a `_pool_scope`."""
        if self.pool is None:
            self.pool = multiprocessing.Pool(processes=self._n_jobs)
        return self.pool

    @contextlib.contextmanager
    def _pool_scope(self):
        """Re-entrant pool scope: nested scopes share one pool, torn down when
        the outermost scope exits. A closed pool cannot be restarted, so each
        outermost scope owns a fresh pool.
        """
        self._pool_depth += 1
        try:
            yield
        finally:
            self._pool_depth -= 1
            if self._pool_depth <= 0:
                self.close_pool()

    def pool_scope(self) -> contextlib.AbstractContextManager[None]:
        """Open a re-entrant scope sharing one folding pool.

        Entering the scope does not create the pool: the first fold inside it
        does, or `restrict` and `run`, which create it before any torch work.
        Nested scopes (including the ones opened internally by `simulate`,
        `log_likelihood_fn`, `restrict` and `run`) reuse that pool, which is
        terminated when the outermost scope exits. Wrap a manual `restrict()`
        followed by `run()` in one scope, so that `run` reuses the pool forked
        before the history matching started torch rather than forking a new
        one afterwards.

        Returns:
            A context manager; see `_pool_scope`.
        """
        return self._pool_scope()

    def close_pool(self):
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

    @property
    def observed_counts(self) -> np.ndarray:
        """Observed counts per bin of the first spectrum, as integers in float64.

        Unlike `observed_spectrum` (float32, rate times exposure), the counts
        are rounded to the nearest integer: rate times exposure carries
        floating-point noise, and a Poisson likelihood needs exact counts.

        Returns:
            Array of shape `(n_bins,)` and dtype `float64`.
        """
        spectrum = xspec.AllData(1)
        rate = np.asarray(spectrum.values, dtype=np.float64)
        return np.rint(rate * float(spectrum.exposure))

    @property
    def n_bins(self) -> int:
        """Number of noticed bins in the observed spectrum."""
        return int(len(self.observed_counts))

    def _check_xspec_state(self) -> None:
        """Refuse an `xspec` session whose fit statistic already includes a prior.

        The backends add the Python-side prior to `-cstat/2`; with
        `Fit.bayes` enabled, `Fit.statistic` would already contain XSPEC's own
        prior term and the prior would be counted twice. Only `str` values are
        compared, so a mocked `xspec` module is inert.

        Raises:
            RuntimeError: If `xspec.Fit.bayes` is a string other than `"off"`.
        """
        value = xspec.Fit.bayes
        if isinstance(value, str) and value.lower() != "off":
            raise RuntimeError(
                f"xspec.Fit.bayes is {value!r}: the fit statistic would include "
                "XSPEC's prior on top of the bsixsa prior, counting the prior "
                "twice. Set xspec.Fit.bayes = 'off' before running the solver."
            )

    def _xspec_constraints(self) -> dict:
        """Read the `xspec` session properties that backends may restrict.

        Each value is read lazily and independently; a value that cannot be
        read or does not have the expected Python type (e.g. under a mocked
        `xspec`) is reported as `None`, meaning "unknown".

        `stat_method` is PyXspec's `Fit.statMethod`, i.e. XSPEC's *default*
        fit statistic: a per-spectrum override such as
        `Fit.statMethod = "chi 1"` is not visible through PyXspec.

        Returns:
            Dict with keys `n_spectra` (`int | None`), `stat_method`
            (`str | None`) and `has_background` (`bool | None`, for the first
            spectrum; `False` when PyXspec reports that it has no background).
        """
        n_spectra = None
        try:
            value = xspec.AllData.nSpectra
            if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                n_spectra = int(value)
        except Exception:
            pass

        stat_method = None
        try:
            value = xspec.Fit.statMethod
            if isinstance(value, str):
                stat_method = value
        except Exception:
            pass

        has_background = None
        try:
            spectrum = xspec.AllData(1)
        except Exception:
            spectrum = None
        if spectrum is not None:
            background_file = _read_background_file(spectrum)
            if background_file is not None:
                has_background = bool(background_file)

        return {
            "n_spectra": n_spectra,
            "stat_method": stat_method,
            "has_background": has_background,
        }

    def _xspec_problem_state(self) -> dict:
        """Read the `xspec` state that shapes the likelihood besides the counts.

        Used by `fingerprint` so that a restriction computed for one problem
        is not accepted for another that shares the observed counts, e.g. the
        same PHA file with another response, exposure, abundance table, or
        frozen parameter value. Values of parameters that are folded (those
        in the prior, and the thawed ones) are excluded, since folding changes
        them; linked parameters contribute their link expression only.

        Every read is guarded and only plain `str`/`int`/`float` values are
        kept, so the hook is inert (all `None` or empty) under a mocked
        `xspec`.

        Returns:
            JSON-serialisable dict with keys `spectra` (per spectrum: file,
            exposure, response and arf file names, background file, noticed
            channels), `models` (per source: name, expression, and the fixed or
            linked parameters of data group 1) and `settings` (`Xset.abund`,
            `Xset.xsect`, `Xset.cosmo`).
        """
        n_spectra = self._xspec_constraints()["n_spectra"]
        spectra = []
        for index in range(1, (n_spectra or 0) + 1):
            entry = {}
            try:
                spectrum = xspec.AllData(index)
            except Exception:
                spectra.append(None)
                continue
            for key, read in (
                ("file", lambda: spectrum.fileName),
                ("exposure", lambda: spectrum.exposure),
                ("rmf", lambda: spectrum.response.rmf),
                ("arf", lambda: spectrum.response.arf),
            ):
                try:
                    entry[key] = _plain(read())
                except Exception:
                    entry[key] = None
            entry["background"] = _read_background_file(spectrum)
            try:
                noticed = list(spectrum.noticed)
                entry["noticed"] = noticed if all(_plain(c) is not None for c in noticed) else None
            except Exception:
                entry["noticed"] = None
            spectra.append(entry)

        folded = {
            (str(model), int(index))
            for model, index in zip(getattr(self, "model_indexes", ()), getattr(self, "indexes", ()))
        }
        models = []
        try:
            sources = xspec.AllModels.sources
            source_items = sorted(sources.items()) if isinstance(sources, dict) else []
        except Exception:
            source_items = []
        for source, name in source_items:
            entry = {"source": _plain(source), "name": _plain(name), "expression": None, "fixed": None}
            try:
                model = xspec.AllModels(1, modName=name)
                entry["expression"] = _plain(model.expression)
                n_parameters = model.nParameters
                if _plain(n_parameters) is not None:
                    fixed = []
                    for index in range(1, int(n_parameters) + 1):
                        parameter = model(index)
                        link = _plain(parameter.link) or ""
                        frozen = parameter.frozen is True
                        if (str(name), index) in folded or not (frozen or link):
                            continue
                        value = _plain(parameter.values[0]) if frozen and not link else None
                        fixed.append([index, value, link])
                    entry["fixed"] = fixed
            except Exception:
                pass
            models.append(entry)

        settings = {}
        for key in ("abund", "xsect", "cosmo"):
            try:
                value = getattr(xspec.Xset, key)
            except Exception:
                value = None
            if isinstance(value, (tuple, list)):
                plain = [_plain(item) for item in value]
                settings[key] = plain if all(item is not None for item in plain) else None
            else:
                settings[key] = _plain(value)

        return {"spectra": spectra, "models": models, "settings": settings}

    @contextlib.contextmanager
    def fold_phase(self, name: str) -> Iterator[None]:
        """Attribute the folds performed inside the block to phase *name*.

        Re-entrant: the previous phase is restored on exit, including when the
        block raises. Folds outside any phase are counted under `"backend"`,
        or under `"post"` once a `run` has returned from its backend
        (predictive checks, `build_inference_data`, ...).

        Parameters:
            name: Phase label, e.g. `"restriction"` or `"post"`.

        Raises:
            TypeError: If *name* is not a string.
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
        """Number of folds performed so far, per phase (a copy).

        This is the single source of fold accounting for every backend: all
        folds go through `_fold`.
        """
        return dict(self._fold_counter())

    def _fold_counter(self) -> dict[str, int]:
        """Return the per-instance fold-count dict, creating it if missing."""
        counts = self.__dict__.get("_fold_counts")
        if counts is None:
            counts = {}
            self._fold_counts = counts
        return counts

    def _count_folds(self, n: int) -> None:
        """Add *n* folds to the current phase.

        Parameters:
            n: Number of parameter vectors folded.
        """
        n = int(n)
        if n <= 0:
            return
        counts = self._fold_counter()
        phase = self._fold_phase_name
        counts[phase] = counts.get(phase, 0) + n

    def _fold(
        self,
        parameters,
        return_kind: str = "cstat",
        progress_bar: bool = True,
        no_pool: bool = False,
    ) -> dict:
        """Fold a batch of parameter vectors; every fold of the solver goes here.

        Both `simulate` and `log_likelihood_fn` delegate to this method, which
        validates the batch, counts the folds and manages the pool scope, then
        hands the batch to the leaf hook `_fold_batch`. Subclasses replace the
        forward model by overriding `_fold_batch` only, so validation and
        accounting stay the production code.

        An empty batch (any array with no elements) is not folded and not
        counted: it returns `{"cstat": (0,)}`, plus `"total_model_counts"` of
        shape `(0, n_bins)` for the two model kinds, plus empty
        `"model_counts"` and `"component_counts"` dicts for
        `"models_and_components"`.

        Parameters:
            parameters: Array-like of shape `(n, d)` in physical space, with
                `d = num_parameters`.
            return_kind: One of `"cstat"`, `"full_model_counts"`, or
                `"models_and_components"`.
            progress_bar: Whether to display a progress bar.
            no_pool: Fold serially in this process instead of in the pool.

        Returns:
            Stacked outputs returned by [`parallel_folding`][bsixsa.xspec.parallel_folding].

        Raises:
            ValueError: If *return_kind* is unknown, if *parameters* is not a
                non-empty `(n, d)` array or an empty one, or if any parameter
                value is NaN or Inf. Nothing is counted in these cases.
        """
        if return_kind not in _RETURN_KINDS:
            raise ValueError(f"return_kind must be one of {_RETURN_KINDS}, got {return_kind!r}.")
        d = self.num_parameters
        parameters = np.asarray(parameters, dtype=np.float64)
        if parameters.size == 0:
            parameters = parameters.reshape(0, d)
        # A 1-D vector would be iterated coordinate by coordinate downstream.
        if parameters.ndim != 2 or parameters.shape[1] != d:
            raise ValueError(f"parameters must have shape (n, {d}), got {parameters.shape}.")
        # Parameter names come from xspec, so only build them to report an error.
        if not np.all(np.isfinite(parameters)):
            _validate_finite(parameters, self.parameter_names)
        if len(parameters) == 0:
            return self._empty_fold_outputs(return_kind)
        self._count_folds(len(parameters))

        desc = "Evaluating C_stat - " if return_kind == "cstat" else ""
        if no_pool:
            return self._fold_batch(
                parameters,
                return_kind=return_kind,
                desc=desc,
                progress_bar=progress_bar,
                pool=None,
            )

        with self._pool_scope():
            return self._fold_batch(
                parameters,
                return_kind=return_kind,
                desc=desc,
                progress_bar=progress_bar,
                pool=self._get_pool(),
            )

    def _fold_batch(
        self,
        parameters: np.ndarray,
        *,
        return_kind: str,
        desc: str,
        progress_bar: bool,
        pool,
    ) -> dict:
        """Leaf folding hook: evaluate a validated batch through `xspec`.

        Called by `_fold` only, with a non-empty float64 `(n, d)` batch that
        has already been counted. Override it (not `_fold`) to substitute the
        forward model; do not count folds here.

        Parameters:
            parameters: Finite physical parameters, shape `(n, d)`, `n >= 1`.
            return_kind: One of `"cstat"`, `"full_model_counts"`, or
                `"models_and_components"`.
            desc: Prefix for the progress-bar label.
            progress_bar: Whether to display a progress bar.
            pool: The folding pool, or `None` to fold serially.

        Returns:
            Stacked outputs, as returned by
            [`parallel_folding`][bsixsa.xspec.parallel_folding].
        """
        return parallel_folding(
            parameters,
            self.indexes,
            self.model_indexes,
            desc=desc,
            progress_bar=progress_bar,
            pool=pool,
            return_kind=return_kind,
        )

    def _empty_fold_outputs(self, return_kind: str) -> dict:
        """Outputs of a fold over zero parameter vectors (see `_fold`)."""
        outputs = {"cstat": np.empty(0, dtype=np.float64)}
        if return_kind in ("full_model_counts", "models_and_components"):
            outputs["total_model_counts"] = np.empty((0, self.n_bins), dtype=np.float64)
        if return_kind == "models_and_components":
            outputs["model_counts"] = {}
            outputs["component_counts"] = {}
        return outputs

    def fingerprint(self) -> str:
        """Hash identifying the inference problem this solver was built for.

        Covers the observed counts, the parameter names, each prior axis
        (family, normalised shape/`loc`/`scale` arguments, support and a
        rounded quantile probe; the base family and unit-cube interval of a
        truncated marginal), the number of bins, the `_xspec_constraints`
        (number of spectra, fit statistic, background) and the
        `_xspec_problem_state` (responses, exposures, noticed channels, model
        expressions, fixed parameter values, abundance and cross-section
        tables). It reads only the overridable hooks, so a subclass can
        replace every `xspec` read. The base prior is used, so the hash does
        not change when a restriction installs a truncated working prior.

        Returns:
            SHA-256 hex digest.

        Raises:
            TypeError: If the prior does not expose per-axis `dists`.
        """
        prior = getattr(self, "base_prior", None)
        if prior is None:
            prior = self.prior
        dists = getattr(prior, "dists", None)
        if dists is None:
            raise TypeError("fingerprint() requires a prior exposing per-axis `dists`.")

        counts = np.ascontiguousarray(self.observed_counts, dtype=np.float64)
        constraints = self._xspec_constraints()
        document = {
            "parameter_names": [str(name) for name in self.parameter_names],
            "axes": [_axis_descriptor(dist) for dist in dists],
            "n_bins": int(self.n_bins),
            "n_spectra": constraints["n_spectra"],
            "stat_method": constraints["stat_method"],
            "has_background": constraints["has_background"],
            "problem_state": _canonical(self._xspec_problem_state()),
        }

        digest = hashlib.sha256()
        digest.update(b"bsixsa.solver.fingerprint.v1\0")
        digest.update(counts.tobytes())
        digest.update(json.dumps(document, sort_keys=True).encode("utf-8"))
        return digest.hexdigest()

    def restrict(
        self,
        config=None,
        *,
        force: bool = False,
        log: Callable[[str], None] | None = print,
    ):
        r"""Restrict the prior by history matching, or return the cached result.

        The history matching runs in the unit cube $u = F(\theta)$ of the
        base prior and only asks for the cstat of each fold. Its folds are
        counted under the `"restriction"` phase of `fold_counts` and do not
        go through the backend's tracer. The result is written to
        `<outputfiles_basename>/restriction/` (the bounding box is added by
        the first `run` of a box-mode backend, which computes it) and stored
        in `restriction`, which `run` then uses even when the solver was built
        with `restrict_prior=False`.

        The pool is forked before the history matching starts torch. Calling
        `restrict()` and then `run()` separately closes that pool in between
        and forks a new one after torch has run; wrap both in
        `with solver.pool_scope():` to share one pool.

        Parameters:
            config: History-matching configuration. `None` uses
                `restriction_config`, or a default
                [`HistoryMatching`][bsixsa.restriction.HistoryMatching].
            force: Recompute even if a restriction is already stored.
            log: Receives one progress line per wave (and figure failures);
                `None` silences the run.

        Returns:
            The [`Restriction`][bsixsa.restriction.Restriction].

        Raises:
            TypeError: If *config* is neither `None` nor a
                [`HistoryMatching`][bsixsa.restriction.HistoryMatching].
            ValueError: If a base prior axis has an infinite support.
            RuntimeError: If the `xspec` session is not usable (see
                `_check_xspec_state`).
            FoldError: If a fold returns a NaN or $-\infty$ cstat.
            TruncationBudgetError: If a wave cannot be drawn from the previous
                region within `HistoryMatching.max_candidates` candidates.
        """
        from .restriction.config import HistoryMatching

        # Checked before the cache, the xspec state and the pool fork: the
        # constructor accepts `True`, so `restrict(True)` is an easy mistake.
        if config is not None and not isinstance(config, HistoryMatching):
            raise TypeError(
                "restrict() expects None or a HistoryMatching configuration, got "
                f"{type(config).__name__} {config!r}."
            )
        if (
            not force
            and self.restriction is not None
            and (config is None or config == getattr(self.restriction, "config", None))
        ):
            return self.restriction

        # Imported only when a restriction is computed: the waves pull in torch.
        from .restriction.waves import run_history_matching

        if config is not None:
            cfg = config
        elif self.restriction_config is not None:
            cfg = self.restriction_config
        else:
            cfg = HistoryMatching()

        validate_finite_support(self.base_prior)
        base_prior = self.base_prior

        def simulate_cstat(u: np.ndarray) -> np.ndarray:
            theta = _cube.to_physical(base_prior, u)
            with self.fold_phase("restriction"):
                outputs = self._fold(theta, return_kind="cstat", progress_bar=False)
            return np.asarray(outputs["cstat"], dtype=np.float64)

        with self._pool_scope():
            self._check_xspec_state()
            # Fork the pool before the regressor spins up torch threads.
            self._get_pool()
            seed_seq = _cube.make_seed_sequence(cfg.seed)
            outdir = os.path.join(self.outputfiles_basename, "restriction")
            result = run_history_matching(
                simulate_cstat,
                d=self.num_parameters,
                cfg=cfg,
                seed_seq=seed_seq,
                outdir=outdir,
                fingerprint=self.fingerprint(),
                param_names=list(self.parameter_names),
                log=log,
            )
            result.save(outdir)

        self.restriction = result
        self.restriction_config = cfg
        self._restriction_injected = False
        self._restriction_accounted = False
        return result

    def sample_parameters(
        self,
        n_samples: int,
        distribution_name: str = "prior",
    ):
        r"""Draw parameter samples from one of the registered distributions.

        Parameters:
            n_samples: Number of draws to generate.
            distribution_name: Key of the distribution in
                `self.distributions`. Defaults to `"prior"`.

        Returns:
            Raw samples returned by the selected distribution.
        """

        distribution = self.distributions[distribution_name]
        theta = distribution.sample((n_samples,))  # In the unit cube space
        return theta

    def simulate(self, parameters, return_kind="full_model_counts", progress_bar=True):
        """Fold parameters through `xspec` and stack simulation outputs.

        Parameters:
            parameters: Array-like batch of parameter vectors.
            return_kind: One of `"cstat"`, `"full_model_counts"`, or `"models_and_components"`.

        Returns:
            Stacked outputs returned by [`parallel_folding`][bsixsa.xspec.parallel_folding].

        Raises:
            ValueError: If any parameter values are NaN or Inf.
        """

        return self._fold(
            parameters,
            return_kind=return_kind,
            progress_bar=progress_bar,
            no_pool=False,
        )

    def log_prob_fn(self, theta, x_o, from_unit_cube=False):
        r"""
        Return the unnormalised log-posterior $\log p(\theta \mid x) = -\frac{1}{2} \texttt{Cstat} + \log \pi(\theta)$, summing the C-stat log-likelihood and the Python-side prior set via [`build_prior`][bsixsa.priors.build_prior].

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
        r"""Return $\log \mathcal{L}(\theta) = -\frac{1}{2}\,\texttt{Cstat}(\theta)$.

        Parameters:
            theta: Array of shape `(n, d)` in physical space.
            x_o: Unused; kept for `sbi` compatibility.
            progress_bar: Whether to display a progress bar.
            no_pool: Fold serially in this process instead of in the pool.

        Returns:
            Array of shape `(n,)`.

        Raises:
            ValueError: If any parameter values are NaN or Inf.
        """

        simulation_outputs = self._fold(
            theta,
            return_kind="cstat",
            progress_bar=progress_bar,
            no_pool=no_pool,
        )

        return -0.5 * simulation_outputs["cstat"]

    def log_prior_fn(self, theta, x_o):
        """Log-density of the working prior `self.prior` at *theta*."""

        return self.prior.log_prob(theta)


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
            distribution: Key in `self.distributions` to sample posterior
                draws from. Defaults to `"posterior"`.
            num_samples: Number of posterior draws. Defaults to 10_000.
            include_prior: If `True`, draw prior samples and attach them as
                the *prior* group. Defaults to `True`.
            include_observed: If `True`, attach the observed spectrum as the
                *observed_data* group. Requires an active XSPEC session.
                Defaults to `True`.
            include_log_likelihood: If `True`, simulate the posterior
                samples through the instrument response and compute
                per-bin Poisson log-likelihoods. Required for
                `az.compare`. Defaults to `True`.

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

    def run(self) -> FitResults:
        """Run the configured backend, restricting the prior first if requested.

        Inside one pool scope (forked before any torch work): the `xspec`
        session is checked, the restriction is computed if configured and not
        yet available, and a stored restriction is checked against
        `fingerprint()`. A backend whose config has
        `restriction_mode == "box"` then samples the truncated bounding box of
        the restriction, installed as `prior` (and as
        `distributions["restricted_prior"]`); its evidence is corrected by the
        box's prior mass and its best-fit statistic is reported with the base
        prior term. A `"region"` backend keeps the base prior. The box is
        always derived from `base_prior`, so re-running never truncates twice,
        and it stays installed after the run so that post-run sampling of
        point-estimate backends stays inside it. `distributions["prior"]`
        is always the base prior.

        Folds made by the backend are counted under `"backend"`; once it
        returns, folds outside an explicit `fold_phase` are counted under
        `"post"`. The results, the posterior and the tracer are stored before
        any [`BoxEdgeWarning`][bsixsa.exceptions.BoxEdgeWarning] is emitted,
        so a warnings filter that turns it into an error does not discard the
        run.

        Returns:
            The backend's [`FitResults`][bsixsa.solver.FitResults], corrected
            for the restriction (see `_finalise_restriction`).

        Raises:
            RuntimeError: If the `xspec` session is not usable; raised before
                the backend is constructed.
            ValueError: If the restriction was computed for another problem
                (the base prior is reinstalled first), or if the backend's
                `x0_physical` lies outside the restricted box.
        """
        from .backend import get_backend_class

        with self._pool_scope():
            self._check_xspec_state()
            # Fork the pool before any torch work (history matching, flows).
            self._get_pool()
            self._fold_phase_name = "backend"

            if self.restriction is None and self.restriction_config is not None:
                self.restrict()
            if self.restriction is not None:
                try:
                    self.restriction.check_fingerprint(self.fingerprint())
                except ValueError:
                    # A box installed by an earlier run belongs to the rejected
                    # restriction too.
                    self._install_base_prior()
                    raise

            mode = getattr(type(self.backend_config), "restriction_mode", "box")
            base_prior = getattr(self, "base_prior", None)
            if self.restriction is not None and mode == "box":
                box_prior, log_mass = self.restriction.truncated_prior(base_prior)
                self.prior = box_prior
                self.distributions["restricted_prior"] = box_prior
                try:
                    self._check_x0_in_prior()
                except ValueError:
                    self._install_base_prior()
                    raise
                if not self._restriction_injected and callable(getattr(self.restriction, "save", None)):
                    # restrict() saved before the box existed.
                    self.restriction.save(os.path.join(self.outputfiles_basename, "restriction"))
            else:
                self._install_base_prior()
                log_mass = 0.0

            backend_cls = get_backend_class(self.backend_name)
            self.backend = backend_cls(solver=self, config=self.backend_config)
            fit = self.backend.run()
            self._fold_phase_name = "post"
            self.fit_result = self._finalise_restriction(fit, mode, log_mass)
            self.posterior_samples = self.fit_result.posterior_samples
            self.distributions["posterior"] = self.backend

            self.backend.tracer.plot_progress()
            self.backend.tracer.save()

            messages = []
            if self.restriction is not None and mode == "box":
                messages = self._box_edge_messages_safe(self.fit_result)

        # Emitted last: a warnings filter set to "error" must not discard the run.
        for message in messages:
            warnings.warn(BoxEdgeWarning(message), stacklevel=2)
        return self.fit_result

    def _install_base_prior(self) -> None:
        """Make the base prior the working prior and drop any restricted box."""
        base_prior = getattr(self, "base_prior", None)
        # Solvers built with `__new__` may only carry `prior`.
        if base_prior is not None:
            self.prior = base_prior
        self.distributions.pop("restricted_prior", None)

    def _check_x0_in_prior(self) -> None:
        """Require the backend's starting point to lie inside the working prior.

        Raises:
            ValueError: If the backend config has an `x0_physical` at which
                the working (restricted) prior has no finite log density,
                naming the offending parameters and the box bounds.
        """
        x0 = getattr(self.backend_config, "x0_physical", None)
        if x0 is None:
            return
        x0 = np.asarray(x0, dtype=np.float64).ravel()
        if np.all(np.isfinite(np.asarray(self.prior.log_prob(x0), dtype=np.float64))):
            return

        names = self._axis_names(x0.size)
        details = []
        for j, dist in enumerate(getattr(self.prior, "dists", ())):
            low, high = (float(bound) for bound in dist.support())
            if j < x0.size and not low <= x0[j] <= high:
                details.append(f"{names[j]}={x0[j]!r} outside [{low!r}, {high!r}]")
        raise ValueError(
            "x0_physical lies outside the restricted prior box installed by the "
            "history-matching restriction"
            + (f": {'; '.join(details)}" if details else "")
            + ". Start from a point inside the box, or disable the restriction "
            "(restrict_prior=False)."
        )

    def _axis_names(self, d: int) -> list[str]:
        """Parameter names for messages, falling back to `axis <j>`."""
        try:
            names = [str(name) for name in self.parameter_names]
        except Exception:
            names = []
        if len(names) != d:
            names = [f"axis {j}" for j in range(d)]
        return names

    def _finalise_restriction(self, fit: FitResults, mode: str, log_mass: float) -> FitResults:
        """Account for the restriction in a backend's results.

        Restriction folds are added to `n_likelihood_evaluations` (and
        reported in `n_restriction_evaluations` / `restriction_time`, which
        includes the bounding-box computation) only when this solver computed
        the restriction, and only in the first run that uses it: a re-run
        spends no restriction fold. For a `"box"` backend, the box's log
        prior mass is added to a finite `log_Z`, and `best_fit_stat` is
        re-expressed with the base prior's XSPEC-convention term (so it is
        comparable with an unrestricted run). Box-edge diagnostics are left
        to `run`, which emits them once the results are stored.

        Parameters:
            fit: Results returned by the backend; modified in place.
            mode: The backend's `restriction_mode`.
            log_mass: Log prior mass of the installed box (`0.0` without one).

        Returns:
            *fit*.
        """
        restriction = self.restriction
        if restriction is None:
            return fit

        if self._restriction_injected or self._restriction_accounted:
            fit.n_restriction_evaluations = 0
            fit.restriction_time = 0.0
        else:
            n_simulations = int(restriction.n_simulations)
            box_info = getattr(restriction, "box_info", None)
            box_seconds = box_info.get("seconds", 0.0) if isinstance(box_info, dict) else 0.0
            fit.n_likelihood_evaluations += n_simulations
            fit.n_restriction_evaluations = n_simulations
            fit.restriction_time = float(restriction.seconds) + float(box_seconds)
            self._restriction_accounted = True

        if mode != "box":
            return fit

        log_mass = float(log_mass)
        if np.isfinite(fit.log_Z):
            fit.log_Z = fit.log_Z + log_mass
        fit.log_Z_restriction_correction = log_mass

        if fit.best_fit is not None and fit.best_fit_stat is not None:
            try:
                x = np.asarray(fit.best_fit, dtype=np.float64)
                # Only a truncated non-uniform family (e.g. truncnorm) changes
                # the term: its density is renormalised by the box.
                fit.best_fit_stat = float(
                    fit.best_fit_stat
                    - self.prior.xspec_bayes_contribution(x)
                    + self.base_prior.xspec_bayes_contribution(x)
                )
            except (AttributeError, NotImplementedError, TypeError, ValueError):
                pass

        return fit

    def _box_edge_messages_safe(self, fit: FitResults) -> list[str]:
        """Diagnose posterior mass crowding an interior face of the restricted box.

        The box has a core (the bounding box of the restriction's region) and
        a padding shell on each side. A face is interior when it is not a face
        of the prior cube. Posterior samples are mapped to the box's own unit
        cube; each interior face whose shell (at least `1e-3` wide) holds a
        fraction of the samples above `max(1e-3, 5 / N_eff)`, with `N_eff`
        the number of distinct rows, yields a message, and so does a
        `best_fit` within `1e-3` of an interior face. `run` emits each one
        as a [`BoxEdgeWarning`][bsixsa.exceptions.BoxEdgeWarning].
        Diagnostics that cannot be computed are skipped: this method never
        raises.

        Parameters:
            fit: Results of a backend that sampled the box.

        Returns:
            The warning messages, possibly empty.
        """
        try:
            return self._box_edge_messages(fit)
        except Exception:
            return []

    def _box_edge_messages(self, fit: FitResults) -> list[str]:
        """Messages for `_box_edge_messages_safe` (see there); may raise on bad inputs."""
        restriction = self.restriction
        lo, hi = (np.asarray(bound, dtype=np.float64).ravel() for bound in restriction.box())
        core_lo, core_hi = (np.asarray(bound, dtype=np.float64).ravel() for bound in restriction.box_core)
        d = lo.size
        lower_interior = lo > 0.0
        upper_interior = hi < 1.0
        if not (lower_interior.any() or upper_interior.any()):
            return []

        width = hi - lo
        shell_lower = np.maximum((core_lo - lo) / width, 1e-3)
        shell_upper = np.maximum((hi - core_hi) / width, 1e-3)
        names = self._axis_names(d)
        messages = []

        samples = fit.posterior_samples
        if samples is not None:
            # Columns are in prior order; names are not used to index them.
            samples = np.asarray(samples, dtype=np.float64)
            if samples.ndim == 2 and samples.shape[1] == d:
                samples = samples[np.all(np.isfinite(samples), axis=1)]
            if samples.ndim == 2 and samples.shape[1] == d and len(samples) > 0:
                u = np.asarray(self.prior.to_unit_cube(samples), dtype=np.float64)
                n_eff = len(np.unique(samples, axis=0))
                threshold = max(1e-3, 5.0 / n_eff)
                for j in range(d):
                    for interior, face, inside in (
                        (lower_interior[j], "lower", u[:, j] <= shell_lower[j]),
                        (upper_interior[j], "upper", u[:, j] >= 1.0 - shell_upper[j]),
                    ):
                        fraction = float(np.mean(inside))
                        if interior and fraction > threshold:
                            messages.append(
                                f"{fraction:.2%} of the posterior samples of {names[j]} lie in "
                                f"the padding shell of the {face} face of the restricted prior "
                                f"box (tolerance {threshold:.2%}). The restriction may have "
                                "cut off posterior mass: widen it (e.g. larger "
                                "HistoryMatching.target_spans or box_padding) or run without it."
                            )

        if fit.best_fit is not None:
            best = np.asarray(fit.best_fit, dtype=np.float64).ravel()
            if best.size == d and np.all(np.isfinite(best)):
                u_best = np.asarray(self.prior.to_unit_cube(best.reshape(1, d)), dtype=np.float64)[0]
                for j in range(d):
                    for interior, face, close in (
                        (lower_interior[j], "lower", u_best[j] <= 1e-3),
                        (upper_interior[j], "upper", u_best[j] >= 1.0 - 1e-3),
                    ):
                        if interior and close:
                            messages.append(
                                f"The best fit of {names[j]} ({best[j]!r}) sits on the {face} "
                                "face of the restricted prior box. The optimum may lie outside "
                                "the box: widen the restriction or run without it."
                            )
        return messages
