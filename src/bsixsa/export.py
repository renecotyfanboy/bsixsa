"""Export a bsixsa posterior as an XSPEC chain file.

Once a chain file is loaded, XSPEC's `error` command reports credible
intervals from the chain rows, and the error options of `flux`, `lumin` and
`eqwidth` draw their parameter sets from it.

[`export_xspec_chain`][bsixsa.export.export_xspec_chain] (or
[`Solver.export_xspec_chain`][bsixsa.solver.Solver.export_xspec_chain])
writes the posterior samples of a [`Solver`][bsixsa.solver.Solver] in the
format of XSPEC's own `chain run`:

```python
import xspec

solver.run()
path = solver.export_xspec_chain("posterior_chain.fits")

xspec.AllChains.clear()
xspec.AllChains += path
xspec.Fit.error("1-3")                     # 90% credible intervals ("cluster:2" for a named model)
xspec.AllModels.calcFlux("0.5 2.0 err")    # flux with its credible interval
```

Clear previously loaded chains first: XSPEC asks for confirmation when chains
of different lengths are loaded together. XSPEC only uses a chain while its
columns match the current free parameters, so load it into the session used
for the run. A session rebuilt from an `.xcm` file may order its models
differently: export the chain again from that session.
"""

from __future__ import annotations

import datetime
import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .convenience import first_model_copy

if TYPE_CHECKING:
    from .solver import Solver

__all__ = ["export_xspec_chain"]

# Names fixed by XSPEC (FITSChain.cxx).
CHAIN_EXTENSION = "CHAIN"
STATISTIC_COLUMN = "FIT_STATISTIC"
# XSPEC's label of the statistic when several statistics are in use.
TOTAL_STATISTIC_NAME = "Total_Stat"


@dataclass(frozen=True)
class ChainParameter:
    """One fitted XSPEC parameter and its column in a chain file.

    Attributes:
        model: XSPEC model name, `""` for the default (unnamed) model.
        index: Parameter number in the model.
        name: XSPEC parameter name, e.g. `"kT"`.
        unit: XSPEC parameter unit, e.g. `"keV"`, or `""`.
        linked: Whether the parameter is now linked to others in XSPEC.
    """

    model: str
    index: int
    name: str
    unit: str
    linked: bool = False

    @property
    def column(self) -> str:
        """Column name in the chain file, e.g. `kT__2` or `kT__cluster__2`."""
        if self.model:
            return f"{self.name}__{self.model}__{self.index}"
        return f"{self.name}__{self.index}"

    @property
    def label(self) -> str:
        """XSPEC's label of the parameter, e.g. `2` or `cluster:2`."""
        if self.model:
            return f"{self.model}:{self.index}"
        return str(self.index)


def export_xspec_chain(
    solver: Solver,
    path: str | os.PathLike,
    *,
    samples: pd.DataFrame | np.ndarray | None = None,
    fit_statistic: np.ndarray | None = None,
    overwrite: bool = False,
) -> str:
    """Write the posterior of a solver as an XSPEC chain file.

    One row per posterior sample, written as a single Metropolis-Hastings
    chain at temperature 1. Columns follow XSPEC's order of free parameters,
    which can differ from the prior order. If the fitted parameters are not
    XSPEC's current free parameters, the file is written with a warning, since
    XSPEC ignores the chain until they match.

    With `LevenbergMarquardt` and `Iminuit` the samples come from a Gaussian
    (Laplace) approximation around the best fit, so XSPEC reports the errors
    of that approximation.

    Example:
        ```python
        path = export_xspec_chain(solver, "posterior_chain.fits")
        xspec.AllChains.clear()
        xspec.AllChains += path
        xspec.Fit.error("1-3")
        ```

    Parameters:
        solver: A [`Solver`][bsixsa.solver.Solver] whose XSPEC session is still loaded.
        path: Output FITS file.
        samples: Physical parameter values, at least 2 rows. Defaults to
            `solver.fit_result.posterior_samples`. A `DataFrame` is matched by
            `solver.parameter_names`; an array must be in prior order.
        fit_statistic: Fit statistic of each sample. By default the samples
            are folded through XSPEC (bare statistic, so `xspec.Fit.bayes`
            must be off). Only `AllChains.best()` and `AllChains.dic()` use it.
        overwrite: Whether an existing file may be replaced.

    Returns:
        Absolute path of the written file, as a `str` (what `xspec.AllChains +=` expects).

    Raises:
        FileExistsError: If *path* exists and *overwrite* is `False`.
        ValueError: If the samples are missing, malformed, fewer than 2 or not finite.
        RuntimeError: If a fitted parameter cannot be read from XSPEC, or `xspec.Fit.bayes` is on.
    """
    path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    _check_destination(path, overwrite=overwrite)

    values = _sample_values(solver, samples)
    parameters = _fitted_parameters(solver)
    xspec_labels = _xspec_free_parameter_labels()
    _warn_if_xspec_will_ignore_chain(parameters, xspec_labels)
    if fit_statistic is None:
        statistic = _folded_statistic(solver, values, parameters)
    else:
        statistic = np.asarray(fit_statistic, dtype=np.float64)
        if statistic.shape != (len(values),):
            raise ValueError(
                f"fit_statistic must have shape ({len(values)},), one value per sample, got {statistic.shape}."
            )
        _refuse_non_finite_rows(statistic, "fit_statistic")

    if samples is None:
        comments = [f"Rows are posterior samples exported by bsixsa (backend: {solver.backend_name})."]
    else:
        comments = ["Rows are samples supplied to bsixsa export_xspec_chain."]
    if fit_statistic is None:
        comments.append(f"{STATISTIC_COLUMN} is XSPEC's fit statistic of each row, without prior term.")
    else:
        comments.append(f"{STATISTIC_COLUMN} was supplied to bsixsa export_xspec_chain.")

    order = _xspec_order(parameters, xspec_labels or [])
    _write_chain_file(
        path,
        [parameters[j] for j in order],
        values[:, order],
        statistic,
        statistic_name=_xspec_statistic_name(),
        comments=comments,
        overwrite=overwrite,
    )
    return path


def _check_destination(path: str, *, overwrite: bool) -> None:
    """Raise if *path* cannot be written."""
    if os.path.isdir(path):
        raise IsADirectoryError(
            f"'{path}' is a directory. Give a file name, e.g. '{os.path.join(path, 'chain.fits')}'."
        )
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"'{path}' already exists. Pass overwrite=True to replace it.")
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")


def _refuse_non_finite_rows(array: np.ndarray, name: str, hint: str = "") -> None:
    """Raise `ValueError` if a row of *array* (1-D or 2-D) holds NaN or an infinity."""
    bad_rows = np.flatnonzero(~np.isfinite(array.reshape(len(array), -1)).all(axis=1))
    if bad_rows.size:
        raise ValueError(
            f"{name} has NaN or infinite values in {bad_rows.size} of {len(array)} rows "
            f"(first: row {bad_rows[0]}).{hint}"
        )


def _sample_values(solver: Solver, samples: pd.DataFrame | np.ndarray | None) -> np.ndarray:
    """Return the samples as a finite float array of shape `(n_samples, d)` in prior order."""
    if samples is None:
        if solver.fit_result is None:
            raise ValueError("The solver has no posterior samples yet: call solver.run() first, or pass samples=.")
        samples = solver.fit_result.posterior_samples.to_numpy()  # columns in prior order
    elif isinstance(samples, pd.DataFrame):
        names = solver.parameter_names
        missing = [name for name in names if name not in samples.columns]
        if missing:
            raise ValueError(
                f"samples has no column for {missing}. Expected the solver's parameter "
                f"names {names}, or pass an array with its columns in prior order."
            )
        samples = samples[names]

    values = np.asarray(samples, dtype=np.float64)
    d = solver.num_parameters
    if values.ndim != 2 or values.shape[1] != d:
        raise ValueError(f"samples must have shape (n_samples, {d}), got {values.shape}.")
    if len(values) < 2:
        raise ValueError(f"samples has {len(values)} rows, but XSPEC needs at least 2 to compute errors from a chain.")
    _refuse_non_finite_rows(
        values, "samples", " Keep only the finite rows, e.g. samples=samples[np.isfinite(samples).all(axis=1)]."
    )
    return values


def _folded_statistic(solver: Solver, values: np.ndarray, parameters: list[ChainParameter]) -> np.ndarray:
    """Fold the samples through XSPEC and return the fit statistic of each."""
    linked = [parameter.label for parameter in parameters if parameter.linked]
    if linked:
        raise ValueError(
            f"Cannot fold the samples: parameters fitted by bsixsa are now linked in XSPEC "
            f"({', '.join(linked)}). Untie them (parameter.untie()) or pass fit_statistic=."
        )
    solver._check_xspec_state()  # the statistic must not include XSPEC's prior term

    # Resampled posteriors repeat rows: fold each distinct row once.
    distinct_rows, row_of_sample = np.unique(values, axis=0, return_inverse=True)
    with solver.fold_phase("post"):
        distinct_statistic = solver.simulate(distinct_rows, return_kind="cstat")["cstat"]
    statistic = np.asarray(distinct_statistic, dtype=np.float64)[row_of_sample.reshape(-1)]
    _refuse_non_finite_rows(statistic, "XSPEC's fit statistic", " Pass fit_statistic= to supply your own.")
    return statistic


def _fitted_parameters(solver: Solver) -> list[ChainParameter]:
    """Name, unit and link state in XSPEC of each fitted parameter, in prior order."""
    parameters = []
    for model, index in zip(solver.model_names, solver.parameter_indices):
        model, index = str(model), int(index)
        try:
            xspec_parameter = first_model_copy(model)(index)
            name, unit = str(xspec_parameter.name), str(xspec_parameter.unit).strip()
            linked = bool(str(xspec_parameter.link).strip())
        except Exception as error:
            label = f"{model}:{index}" if model else str(index)
            raise RuntimeError(
                f"Cannot read parameter {label} from XSPEC. Export the chain while the "
                "models used for the run are loaded."
            ) from error
        parameters.append(ChainParameter(model, index, name, unit, linked))
    return parameters


def _xspec_free_parameter_labels() -> list[str] | None:
    """Labels (`"2"`, `"cluster:2"`) of XSPEC's free parameters in chain order, or `None` if unavailable.

    PyXspec has no public accessor for this list; it comes from the chain manager.
    """
    try:
        from xspec import _pyXspec

        labels = _pyXspec.getChainManagerInfo()[1]
    except Exception:
        return None
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        return None
    return labels


def _xspec_statistic_name() -> str:
    """Name of XSPEC's fit statistic (e.g. `"C-Statistic"`), the unit of `FIT_STATISTIC`; `"Total_Stat"` as fallback."""
    try:
        from xspec import _pyXspec

        name = _pyXspec.doTclout(["statmethod"])
    except Exception:
        return TOTAL_STATISTIC_NAME
    if isinstance(name, str) and name.strip():
        return name.strip()
    return TOTAL_STATISTIC_NAME


def _warn_if_xspec_will_ignore_chain(parameters: list[ChainParameter], xspec_labels: list[str] | None) -> None:
    """Warn if XSPEC will not use the chain, because its free parameters differ from the fitted ones."""
    if xspec_labels is None:
        warnings.warn(
            "Could not read XSPEC's list of free parameters, so the chain columns are "
            "written in prior order. XSPEC expects this order only if the models were "
            "defined in the order of their source numbers.",
            UserWarning,
            stacklevel=3,
        )
        return

    labels = [parameter.label for parameter in parameters]
    not_fitted = [label for label in xspec_labels if label not in labels]
    not_free = [label for label in labels if label not in xspec_labels]
    reasons = []
    if not_fitted:
        reasons.append(f"free in XSPEC but not fitted by bsixsa: {', '.join(not_fitted)}")
    if not_free:
        reasons.append(
            f"fitted by bsixsa but not free in XSPEC (frozen, linked, or in an inactive model): {', '.join(not_free)}"
        )
    if reasons:
        warnings.warn(
            "XSPEC will ignore this chain (for error, flux errors, ...) until its free "
            f"parameters match the chain columns. Parameters {'; '.join(reasons)}. Make the "
            "same parameters free as during the run, then load the chain.",
            UserWarning,
            stacklevel=3,
        )


def _xspec_order(parameters: list[ChainParameter], xspec_labels: list[str]) -> list[int]:
    """Positions of *parameters* sorted in XSPEC's order: by model definition order, then parameter number."""
    model_rank: dict[str, int] = {}
    for label in xspec_labels:
        model, _, number = label.rpartition(":")
        if number.isdigit():  # skip response parameters, labelled "r1", "2:r1", ...
            model_rank.setdefault(model, len(model_rank))
    for parameter in parameters:
        model_rank.setdefault(parameter.model, len(model_rank))
    return sorted(range(len(parameters)), key=lambda j: (model_rank[parameters[j].model], parameters[j].index))


def _write_chain_file(
    path: str,
    parameters: list[ChainParameter],
    values: np.ndarray,
    statistic: np.ndarray,
    *,
    statistic_name: str,
    comments: list[str],
    overwrite: bool,
) -> None:
    """Write the FITS chain file, with the header keywords XSPEC reads."""
    from astropy.io import fits

    columns = [
        # Without the unit XSPEC loads the chain but ignores it; XSPEC writes no TUNIT for a unitless parameter.
        fits.Column(name=parameter.column, format="D", unit=parameter.unit or None, array=values[:, j])
        for j, parameter in enumerate(parameters)
    ]
    columns.append(fits.Column(name=STATISTIC_COLUMN, format="D", unit=statistic_name, array=statistic))
    table = fits.BinTableHDU.from_columns(columns, name=CHAIN_EXTENSION)

    header = table.header
    # One Metropolis-Hastings run at temperature 1, starting at row 1: XSPEC
    # refuses a chain without at least one TEMPR/STROW pair.
    header["TEMPR001"] = (1.0, "chain temperature")
    header["STROW001"] = (1, "first row at this temperature")
    header["CHAINTYP"] = "MetropolisHastings"
    created = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    header["DATE"] = (created, "file creation date (UTC)")
    header.add_history("Created by bsixsa export_xspec_chain")
    for comment in comments:
        header.add_comment(comment)

    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path, overwrite=overwrite)
