"""Priors for bsixsa fits.

- [`uniform`][bsixsa.priors.uniform], [`loguniform`][bsixsa.priors.loguniform]
  and [`norm`][bsixsa.priors.norm] build 1-D priors (frozen SciPy distributions).
- [`MultipleIndependent`][bsixsa.priors.MultipleIndependent] is the joint
  prior of independent marginals, with its unit-cube maps.
- [`build_prior`][bsixsa.priors.build_prior] matches a prior list to the
  parameters of the loaded XSPEC model(s).
"""

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats
from xspec import AllModels

from .convenience import (
    XSilence,
    XspecParameter,
    base_component_name,
    first_model_copy,
    iter_all_parameters,
    model_data_groups,
)
from .xspec import set_parameters

__all__ = [
    'MultipleIndependent',
    'build_prior',
    'loguniform',
    'norm',
    'uniform',
    'validate_finite_support',
]

# A frozen SciPy continuous distribution.
Distribution = Any


def uniform(low: float, high: float) -> Distribution:
    """Uniform prior on ``[low, high]``, like XSPEC's ``cons`` prior.

    Parameters:
        low: Lower bound.
        high: Upper bound.

    Returns:
        ``scipy.stats.uniform(loc=low, scale=high - low)``.
    """
    if not (np.isfinite(low) and np.isfinite(high) and low < high):
        raise ValueError(
            f"uniform(low, high) needs finite bounds with low < high, got low={low}, high={high}."
        )
    return stats.uniform(low, high - low)


def loguniform(low: float, high: float) -> Distribution:
    """Log-uniform prior on ``[low, high]``, like XSPEC's ``jeffreys`` prior.

    The density is proportional to ``1/x``: every decade gets the same prior
    mass. Use it for scale parameters such as normalisations.

    Parameters:
        low: Lower bound, strictly positive.
        high: Upper bound.

    Returns:
        ``scipy.stats.loguniform(low, high)``.
    """
    if not (0 < low < high < np.inf):
        raise ValueError(
            f"loguniform(low, high) needs 0 < low < high < inf, got low={low}, high={high}."
        )
    return stats.loguniform(low, high)


def norm(loc: float, scale: float, low: float, high: float) -> Distribution:
    """Gaussian prior of mean ``loc`` and standard deviation ``scale``, truncated to ``[low, high]``.

    Parameters:
        loc: Mean of the underlying Gaussian.
        scale: Standard deviation of the underlying Gaussian.
        low: Lower truncation bound.
        high: Upper truncation bound.

    Returns:
        A ``scipy.stats.truncnorm`` distribution.
    """
    if not (scale > 0 and low < high):
        raise ValueError(
            f"norm(loc, scale, low, high) needs scale > 0 and low < high, got scale={scale}, low={low}, high={high}."
        )
    # SciPy's truncnorm takes the bounds in units of standard deviations from loc.
    a, b = (low - loc) / scale, (high - loc) / scale
    return stats.truncnorm(a, b, loc=loc, scale=scale)


def validate_finite_support(prior: "MultipleIndependent") -> None:
    """Require every axis of a prior to have a finite support.

    Raises:
        ValueError: Naming every axis with an infinite or ``nan`` bound.
    """
    offending = []
    for j, dist in enumerate(prior.dists):
        low, high = (float(bound) for bound in dist.support())
        if not (np.isfinite(low) and np.isfinite(high)):
            offending.append(f"axis {j} has support [{low}, {high}]")
    if offending:
        raise ValueError(
            "Prior support must be finite on every axis: " + "; ".join(offending) + "."
        )


class MultipleIndependent:
    r"""Joint prior made of independent 1-D marginals, one per fitted parameter.

    The unit-cube coordinate of axis $j$ is $u_j = F_j(\theta_j)$, where
    $F_j$ is the CDF of marginal $j$; under the prior, $u$ is uniform on
    $[0, 1]^d$ (see
    [`from_unit_cube`][bsixsa.priors.MultipleIndependent.from_unit_cube] and
    [`to_unit_cube`][bsixsa.priors.MultipleIndependent.to_unit_cube]).

    Attributes:
        dists: The 1-D marginals, in parameter order.
        ndim: Number of parameters.
        bounds: Support ``(low, high)`` of each marginal.
    """

    def __init__(self, dists: Sequence[Distribution]):
        """Combine 1-D marginals (frozen SciPy-like distributions, in parameter order) into a joint prior."""
        self.dists = list(dists)
        self.ndim = len(self.dists)
        self.bounds = [dist.support() for dist in self.dists]

    def _as_points(self, x: Any, *, batch_only: bool) -> np.ndarray:
        """Return ``x`` as a float array whose last axis has ``ndim`` entries (at most 2-D with ``batch_only``)."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 0 or x.shape[-1] != self.ndim or (batch_only and x.ndim > 2):
            expected = f"({self.ndim},) or (n, {self.ndim})" if batch_only else f"(..., {self.ndim})"
            raise ValueError(f"Expected points of shape {expected}, got {x.shape}.")
        return x

    def sample(
        self,
        size: int | tuple[int, ...] = 1,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw independent samples from the prior, shape ``(*size, ndim)``."""
        rng = np.random.default_rng(random_state)
        return np.stack([dist.rvs(size=size, random_state=rng) for dist in self.dists], axis=-1)

    def from_unit_cube(self, cube: Any) -> np.ndarray:
        r"""Map unit-cube points to physical parameters, $\theta_j = F_j^{-1}(u_j)$ (last axis of size ``ndim``)."""
        cube = self._as_points(cube, batch_only=False)
        theta = np.empty_like(cube)
        for j, dist in enumerate(self.dists):
            theta[..., j] = dist.ppf(cube[..., j])
        return theta

    def to_unit_cube(self, x: Any) -> np.ndarray:
        r"""Map physical parameters to the unit cube, $u_j = F_j(\theta_j)$ (last axis of size ``ndim``)."""
        x = self._as_points(x, batch_only=False)
        cube = np.empty_like(x)
        for j, dist in enumerate(self.dists):
            cube[..., j] = dist.cdf(x[..., j])
        return cube

    def log_prob(self, x: Any) -> float | np.ndarray:
        """Log prior density (sum of the marginal log-densities), ``-inf`` outside the support.

        Parameters:
            x: One point ``(ndim,)`` or a batch ``(n, ndim)``.

        Returns:
            A float for one point, an array of shape ``(n,)`` for a batch.
        """
        x = self._as_points(x, batch_only=True)
        rows = np.atleast_2d(x)
        log_density = np.zeros(len(rows))
        for j, dist in enumerate(self.dists):
            log_density += dist.logpdf(rows[:, j])
        return log_density[0] if x.ndim == 1 else log_density

    def xspec_bayes_contribution(self, x: Any) -> float | np.ndarray:
        """Prior term that XSPEC adds to the fit statistic when ``Fit.bayes = "on"``.

        Bare C-stat plus this term reproduces XSPEC's ``Fit.statistic`` with
        bayes on. Per axis XSPEC adds ``-2 ln(prior)``, unnormalised:
        ``uniform`` (``cons``) gives 0, ``loguniform`` (``jeffreys``) gives
        ``2 ln(x)``, other families give ``-2 logpdf(x)`` (equal to XSPEC only
        up to a constant).

        Parameters:
            x: One point ``(ndim,)`` or a batch ``(n, ndim)``.

        Returns:
            A float for one point, an array of shape ``(n,)`` for a batch.
        """
        x = self._as_points(x, batch_only=True)
        rows = np.atleast_2d(x)
        contribution = np.zeros(len(rows))
        for j, dist in enumerate(self.dists):
            name = dist.dist.name
            if name in ("loguniform", "reciprocal"):
                contribution += 2.0 * np.log(rows[:, j])
            elif name != "uniform":
                contribution -= 2.0 * np.asarray(dist.logpdf(rows[:, j]), dtype=float)
        return float(contribution[0]) if x.ndim == 1 else contribution

    def max_log_pdf_per_axis(self) -> np.ndarray:
        """Per-axis maximum of ``logpdf`` over the support (``uniform``, ``loguniform`` and ``norm`` only).

        Least-squares MAP fits use it to write the prior as extra residuals
        whose sum of squares is ``-2 * log_prior + const``.
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
                # _parse_args also resolves a loc passed positionally, which
                # dist.kwds would miss (silently clamping the mode to 0).
                _, loc, _ = dist.dist._parse_args(*dist.args, **dist.kwds)
                out[j] = float(dist.logpdf(min(max(float(loc), lo), hi)))
            else:
                raise NotImplementedError(
                    f"max_log_pdf is not implemented for {name!r}; "
                    "supported priors are uniform, loguniform, and truncnorm."
                )
        return out


def _model_label(model_name: str) -> str:
    """Name of a model in error messages."""
    return f"model '{model_name}'" if model_name else "the model"


def _model_names_label(model_names: Sequence[str]) -> str:
    """List of model names in error messages."""
    return ", ".join(repr(name) if name else "'' (the unnamed model)" for name in model_names)


def _resolve_component(model_name: str, name: str) -> str:
    """Return the XSPEC name of a component given by XSPEC name or as ``<base>_<position>``."""
    names = list(first_model_copy(model_name).componentNames)
    if name in names:
        return name
    suffixed = re.fullmatch(r"(.+)_(\d+)", name)
    if suffixed:
        position = int(suffixed.group(2))
        if 1 <= position <= len(names) and base_component_name(names[position - 1]) == suffixed.group(1):
            return names[position - 1]
    by_position = [f"{base_component_name(other)}_{k}" for k, other in enumerate(names, start=1)]
    raise ValueError(
        f"Unknown component '{name}' in {_model_label(model_name)}. Its components are "
        f"{', '.join(names)} (or, by position, {', '.join(by_position)})."
    )


def _prior_entries(define_prior: Sequence[tuple], loaded_models: list[str]) -> dict[tuple[str, str, str], tuple]:
    """Map ``(model, XSPEC component, parameter)`` to ``(entry label, distribution)``."""
    lengths = sorted({len(entry) for entry in define_prior})
    if lengths and lengths not in ([3], [4]):
        raise ValueError(
            "Each prior entry must be (component, parameter, distribution), or "
            "(model_name, component, parameter, distribution) to name the model, and all "
            f"entries must have the same form; got entries with {lengths} elements."
        )
    if lengths == [3] and len(loaded_models) > 1:
        raise ValueError(
            f"{len(loaded_models)} models are loaded in XSPEC ({_model_names_label(loaded_models)}): "
            "start each prior entry with the model name, e.g. "
            f"({loaded_models[0]!r}, 'powerlaw', 'PhoIndex', uniform(1, 3))."
        )

    entries = {}
    for entry in define_prior:
        label, distribution = tuple(entry[:-1]), entry[-1]
        model_name, component_name, parameter_name = label if len(label) == 3 else (loaded_models[0], *label)
        if model_name not in loaded_models:
            single_model_hint = (
                " With a single model, the model name can be left out (3-element entries)."
                if len(loaded_models) == 1
                else ""
            )
            raise ValueError(
                f"Prior entry {label}: no model named {model_name!r} is loaded in XSPEC; "
                f"the loaded models are {_model_names_label(loaded_models)}.{single_model_hint}"
            )
        key = (model_name, _resolve_component(model_name, component_name), parameter_name)
        if key in entries:
            raise ValueError(
                f"Two prior entries address the same XSPEC parameter: {entries[key][0]} and {label}."
            )
        entries[key] = (label, distribution)
    return entries


def _as_prior_entry(par: XspecParameter, with_model: bool) -> str:
    """Write a parameter as a prior entry, without the distribution."""
    names = (par.component.name, par.parameter.name)
    return repr((par.model.name, *names) if with_model else names)


def _free_in_later_copies() -> list[str]:
    """XSPEC labels (`"17 (factor)"`, `"bkg:3 (norm)"`) of the free parameters of model copies after the first."""
    labels = []
    for model_name in AllModels.sources.values():
        for group in model_data_groups(model_name)[1:]:
            model = AllModels(group, model_name)
            for number in range(1, model.nParameters + 1):
                parameter = model(number)
                if not (parameter.frozen or parameter.link):
                    index = model.startParIndex + number - 1
                    label = f"{model_name}:{index}" if model_name else str(index)
                    labels.append(f"{label} ({parameter.name})")
    return labels


def build_prior(define_prior: Sequence[tuple]) -> tuple["MultipleIndependent", list[int], list[str]]:
    """Build the joint prior from a list of XSPEC prior specifications.

    Each entry gives the prior of one XSPEC parameter:

    - `(component, parameter, distribution)` when a single model is loaded;
    - `(model_name, component, parameter, distribution)` with any number of
      loaded models (required when several are loaded, e.g. a source and a
      background model).

    Component and parameter names are XSPEC's (case-sensitive). A repeated
    component can also be named by its position in the model expression: in
    `tbabs*(powerlaw + powerlaw)` the first powerlaw is `powerlaw` or
    `powerlaw_2`, as in `Solver.parameter_names`. Each distribution needs a
    finite support, e.g. [`uniform`][bsixsa.priors.uniform],
    [`loguniform`][bsixsa.priors.loguniform] or [`norm`][bsixsa.priors.norm].

    Every parameter given a prior is thawed, unlinked, set to the prior median
    and given the prior support as XSPEC limits. Every other parameter must be
    frozen or linked, so the free XSPEC parameters are exactly the prior axes,
    ordered by model (source number) then parameter number. With several data
    groups, the entries address the copy of each model in its first data group
    (data group 1 for a model on every spectrum); the copies in later data
    groups must be frozen or linked.

    Example:
        ```python
        from bsixsa.priors import build_prior, loguniform, uniform

        prior, indices, models = build_prior([
            ("powerlaw", "PhoIndex", uniform(1, 3)),
            ("powerlaw", "norm", loguniform(1e-5, 1e-1)),
        ])
        ```

    Parameters:
        define_prior: One entry per free XSPEC parameter.

    Returns:
        The joint [`MultipleIndependent`][bsixsa.priors.MultipleIndependent]
        prior, the XSPEC index of each parameter, and its model name (`''` for
        the unnamed model).

    Raises:
        ValueError: If an entry is malformed or unknown, a support is not
            finite, a free parameter has no prior, or a parameter is free in a
            later data group.
    """
    loaded_models = list(AllModels.sources.values())
    if not loaded_models:
        raise ValueError("No XSPEC model is loaded: define the model before building the prior.")
    entries = _prior_entries(define_prior, loaded_models)

    distributions, parameter_indices, model_names, xspec_settings = [], [], [], []
    problems, missing = [], []
    for par in iter_all_parameters():
        key = (par.model.name, par.component.name, par.parameter.name)
        if key not in entries:
            if not (par.parameter.frozen or par.parameter.link):
                missing.append(par)
            continue

        label, distribution = entries.pop(key)
        low, high = (float(bound) for bound in distribution.support())
        if not (np.isfinite(low) and np.isfinite(high) and low < high):
            problems.append(
                f"the prior of {label} has support [{low}, {high}], but XSPEC needs finite parameter "
                "limits with low < high. Give the prior finite bounds (for a Gaussian prior: "
                "bsixsa.priors.norm(loc, scale, low, high) with finite low and high)."
            )
            continue
        # Range-proportional fit delta (used by XSPEC only when Fit.delta == 0).
        delta = 0.01 * (high - low)
        start = float(distribution.ppf(0.5))
        xspec_settings.append(f"{start},{delta},{low},{low},{high},{high}")
        distributions.append(distribution)
        parameter_indices.append(par.parameter.index)
        model_names.append(par.model.name)

    for (model_name, component_name, _), (label, _) in entries.items():
        component = getattr(first_model_copy(model_name), component_name)
        problems.append(
            f"the prior entry {label} matches no parameter; the parameters of {component_name} in "
            f"{_model_label(model_name)} are {', '.join(component.parameterNames)} (names are case-sensitive)."
        )
    if missing:
        with_model = len(loaded_models) > 1 or any(len(entry) == 4 for entry in define_prior)
        example = missing[0]
        name = example.model.name
        group = model_data_groups(name)[0]
        model_handle = f"xspec.AllModels({group}, {name!r})" if name else f"xspec.AllModels({group})"
        problems.append(
            "these free XSPEC parameters have no prior: "
            + ", ".join(_as_prior_entry(par, with_model=with_model) for par in missing)
            + ". Give each one a prior, or freeze it (e.g. "
            f"{model_handle}.{example.component.name}.{example.parameter.name}.frozen = True) "
            "or link it before building the Solver."
        )
    later_free = _free_in_later_copies()
    if later_free:
        problems.append(
            "bsixsa fits the parameters of the first data group of each model, but these are free in "
            f"a later data group: {', '.join(later_free)}. Link them to the first data group or freeze them."
        )
    if problems:
        raise ValueError("Cannot build the prior:\n" + "\n".join(f"  - {problem}" for problem in problems))
    if not distributions:
        raise ValueError("Nothing to fit: the prior list is empty and every XSPEC parameter is frozen or linked.")

    with XSilence():
        set_parameters(xspec_settings, parameter_indices, model_names)
    return MultipleIndependent(distributions), parameter_indices, model_names
