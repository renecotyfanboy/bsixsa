import contextlib
import hashlib
import json
import os
import shutil
from collections.abc import Iterator
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
    best_fit: pd.Series | None = None
    best_fit_stat: float | None = None


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


def _check_restrict_prior_arg(restrict_prior) -> None:
    """Refuse `restrict_prior` values that no restriction step can resolve.

    `None`, a `bool`, and configuration or restriction objects are accepted;
    the objects are resolved (and checked precisely) by the restriction step.
    Plain numbers, strings and containers are refused here so that a typo
    such as `restrict_prior="yes"` or `restrict_prior=1` fails before the
    output directory is touched.

    Parameters:
        restrict_prior: The value passed to `Solver`.

    Raises:
        TypeError: If *restrict_prior* is a number (other than a `bool`), a
            string, bytes, or a built-in container.
    """
    if restrict_prior is None or isinstance(restrict_prior, (bool, np.bool_)):
        return
    if isinstance(
        restrict_prior,
        (int, float, complex, str, bytes, list, tuple, dict, set, frozenset, np.number, np.ndarray),
    ):
        raise TypeError(
            "restrict_prior must be None, a bool, a history-matching configuration "
            f"or a precomputed restriction, got {type(restrict_prior).__name__} "
            f"{restrict_prior!r}."
        )


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
            restrict_prior: Prior restriction request (`None`, a `bool`, a
                history-matching configuration or a precomputed restriction).
                Stored as given; it is resolved by the restriction step.

        Raises:
            TypeError: If *backend* does not define a `backend_name`, or if
                *restrict_prior* is a plain number, string or container.
            ValueError: If no spectrum or no model is loaded in `xspec`, or
                if the output path exists and is not a directory.
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
        The prior is stored as given: its `dists` are not inspected here.

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
            restrict_prior: Prior restriction request, stored after a type
                check (see `_check_restrict_prior_arg`).

        Raises:
            TypeError: If *restrict_prior* is a plain number, string or
                container.
            FileExistsError: If the output directory is not empty and
                *overwrite* is `False`.
            ValueError: If the output path exists and is not a directory.
        """
        from .convenience import available_cpu_count

        _check_restrict_prior_arg(restrict_prior)
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

        Nested scopes (including the ones opened internally by `simulate`
        and `log_likelihood_fn`) reuse the same pool, which is terminated when
        the outermost scope exits. Open it around work that mixes folds with
        torch so the pool is forked before torch spins up its threads.

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
        block raises. Folds outside any phase are counted under `"backend"`.

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
        from .backend import get_backend_class

        with self._pool_scope():
            backend_cls = get_backend_class(self.backend_name)
            self.backend = backend_cls(solver=self, config=self.backend_config)
            self.fit_result = self.backend.run()
            self.posterior_samples = self.fit_result.posterior_samples
            self.distributions["posterior"] = self.backend

            self.backend.tracer.plot_progress()
            self.backend.tracer.save()

            return self.fit_result
