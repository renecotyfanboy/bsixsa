"""Priors for bsixsa fits.

- [`uniform`][bsixsa.priors.uniform], [`loguniform`][bsixsa.priors.loguniform]
  and [`norm`][bsixsa.priors.norm] build 1-D priors (frozen SciPy distributions).
- [`MultipleIndependent`][bsixsa.priors.MultipleIndependent] is the joint
  prior of independent marginals, with its unit-cube maps and truncation.
- [`build_prior`][bsixsa.priors.build_prior] matches a prior list to the
  parameters of the loaded XSPEC model(s).
"""

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
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
    'TruncatedMarginal',
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


def _scalar_or_array(values: np.ndarray) -> np.ndarray | float:
    """Return a NumPy scalar for 0-d input and the array otherwise (SciPy's convention)."""
    return values[()]


class _TruncatedFamily:
    """Family descriptor of a ``TruncatedMarginal``: ``dist.dist.name == "truncated"``."""

    name = "truncated"

    def __repr__(self) -> str:
        return "_TruncatedFamily()"


class TruncatedMarginal:
    r"""Generic truncation of a 1-D distribution to a unit-cube interval.

    Given a base distribution with CDF $F$ and an interval
    $[u_{lo}, u_{hi}] \subseteq [0, 1]$ of its unit cube $u = F(\theta)$, this
    is the base restricted to $[a, b] = [F^{-1}(u_{lo}), F^{-1}(u_{hi})]$ and
    renormalised by the mass $u_{hi} - u_{lo}$:

    $$\pi_B(\theta) = \frac{\pi(\theta)}{u_{hi} - u_{lo}}, \qquad
    F_B(\theta) = \frac{F(\theta) - u_{lo}}{u_{hi} - u_{lo}}.$$

    Used by
    [`MultipleIndependent.truncated`][bsixsa.priors.MultipleIndependent.truncated]
    for families without a closed-form truncation. It behaves like a frozen
    SciPy distribution.

    Attributes:
        base: Wrapped distribution.
        lo_u: Lower bound of the interval in the base unit cube.
        hi_u: Upper bound of the interval in the base unit cube.
        a: Lower physical bound, ``base.ppf(lo_u)``.
        b: Upper physical bound, ``base.ppf(hi_u)``.
    """

    def __init__(self, base: Distribution, lo_u: float, hi_u: float):
        """Wrap ``base`` restricted to ``[base.ppf(lo_u), base.ppf(hi_u)]``.

        Raises:
            ValueError: If ``0 <= lo_u < hi_u <= 1`` fails or the interval collapses to a point.
        """
        lo_u, hi_u = float(lo_u), float(hi_u)
        if not (0.0 <= lo_u < hi_u <= 1.0):
            raise ValueError(
                f"TruncatedMarginal needs 0 <= lo_u < hi_u <= 1, got [{lo_u}, {hi_u}]."
            )
        a, b = float(base.ppf(lo_u)), float(base.ppf(hi_u))
        support = getattr(base, "support", None)
        if callable(support):
            # An interior ppf can round an ulp past the base support; the
            # truncated support must never leave the original one.
            base_low, base_high = (float(bound) for bound in support())
            a = min(max(a, base_low), base_high)
            b = min(max(b, base_low), base_high)
        if not a < b:
            raise ValueError(_collapse_message(base, lo_u, hi_u, a, b))
        self.base = base
        self.lo_u = lo_u
        self.hi_u = hi_u
        self.a = a
        self.b = b
        self.dist = _TruncatedFamily()
        self.args = ()
        self.kwds = {}

    @property
    def _width(self) -> float:
        return self.hi_u - self.lo_u

    def __repr__(self) -> str:
        return f"TruncatedMarginal({self.base!r}, lo_u={self.lo_u!r}, hi_u={self.hi_u!r})"

    def support(self) -> tuple[float, float]:
        """Physical support ``(a, b)`` of the truncated distribution."""
        return self.a, self.b

    def cdf(self, x: ArrayLike) -> np.ndarray | float:
        """Cumulative distribution function $(F(x) - u_{lo}) / (u_{hi} - u_{lo})$, clipped to ``[0, 1]``."""
        base_cdf = np.asarray(self.base.cdf(np.asarray(x, dtype=float)), dtype=float)
        return _scalar_or_array(np.clip((base_cdf - self.lo_u) / self._width, 0.0, 1.0))

    def sf(self, x: ArrayLike) -> np.ndarray | float:
        """Survival function $(u_{hi} - F(x)) / (u_{hi} - u_{lo})$, clipped to ``[0, 1]``."""
        base_cdf = np.asarray(self.base.cdf(np.asarray(x, dtype=float)), dtype=float)
        return _scalar_or_array(np.clip((self.hi_u - base_cdf) / self._width, 0.0, 1.0))

    def ppf(self, q: ArrayLike) -> np.ndarray | float:
        """Inverse CDF, ``base.ppf(lo_u + q * (hi_u - lo_u))``; ``nan`` outside ``[0, 1]``."""
        q = np.asarray(q, dtype=float)
        inside = (q >= 0.0) & (q <= 1.0)
        # Clip in both spaces: roundoff in lo + q*width must never step outside
        # the box, otherwise ppf(1) could land an ulp beyond b.
        u = np.clip(self.lo_u + np.where(inside, q, 0.0) * self._width, self.lo_u, self.hi_u)
        x = np.clip(np.asarray(self.base.ppf(u), dtype=float), self.a, self.b)
        return _scalar_or_array(np.where(inside, x, np.nan))

    def isf(self, q: ArrayLike) -> np.ndarray | float:
        """Inverse survival function, ``ppf(1 - q)``."""
        return self.ppf(1.0 - np.asarray(q, dtype=float))

    def logpdf(self, x: ArrayLike) -> np.ndarray | float:
        """Log-density ``base.logpdf(x) - log(hi_u - lo_u)`` on ``[a, b]``, ``-inf`` outside."""
        x = np.asarray(x, dtype=float)
        inside = (x >= self.a) & (x <= self.b)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_pdf = np.asarray(self.base.logpdf(x), dtype=float) - np.log(self._width)
        out = np.where(inside, log_pdf, -np.inf)
        return _scalar_or_array(np.where(np.isnan(x), np.nan, out))

    def pdf(self, x: ArrayLike) -> np.ndarray | float:
        """Probability density, ``exp(logpdf(x))``."""
        return _scalar_or_array(np.exp(np.asarray(self.logpdf(x), dtype=float)))

    def rvs(
        self,
        size: int | tuple[int, ...] | None = None,
        random_state: int | np.random.Generator | np.random.RandomState | None = None,
    ) -> np.ndarray | float:
        """Draw samples by inverse-CDF sampling."""
        if isinstance(random_state, np.random.RandomState):
            u = random_state.uniform(size=size)
        else:
            u = np.random.default_rng(random_state).uniform(size=size)
        return self.ppf(u)


def _validate_unit_box(lower_u, upper_u, ndim: int, min_width: float) -> tuple[np.ndarray, np.ndarray]:
    """Check a unit-cube box and return its bounds as float arrays of shape ``(ndim,)``."""
    lo = np.asarray(lower_u, dtype=float)
    hi = np.asarray(upper_u, dtype=float)
    if lo.shape != (ndim,) or hi.shape != (ndim,):
        raise ValueError(
            f"Box bounds must have shape ({ndim},), got {lo.shape} and {hi.shape}."
        )
    problems = []
    for j in range(ndim):
        if not (np.isfinite(lo[j]) and np.isfinite(hi[j])):
            problems.append(f"axis {j}: non-finite bounds [{lo[j]}, {hi[j]}]")
        elif not (0.0 <= lo[j] < hi[j] <= 1.0):
            problems.append(f"axis {j}: [{lo[j]}, {hi[j]}] violates 0 <= lo < hi <= 1")
        elif hi[j] - lo[j] < min_width:
            problems.append(f"axis {j}: width {hi[j] - lo[j]:.3e} < min_width {min_width:.3e}")
    if problems:
        raise ValueError("Invalid unit-cube box: " + "; ".join(problems) + ".")
    return lo, hi


def _collapse_message(dist, lo_u: float, hi_u: float, low: float, high: float) -> str:
    """Error message for a unit-cube interval with an empty physical image."""
    return (
        f"Unit-cube interval [{lo_u}, {hi_u}] of {dist!r} collapses to "
        f"[{low}, {high}] in physical space; widen the box."
    )


def _rebuild_inside(make, lo_param: float, hi_param: float, outer_low: float, outer_high: float):
    """Build ``make(lo_param, hi_param)`` with its support inside ``[outer_low, outer_high]``.

    Rebuilding a family rounds its bounds again, so a face can land one ulp
    outside the original support, where the prior density is ``-inf``. Such a
    bound is nudged inward by doubling ulp steps.
    """
    first_step = float(np.spacing(max(abs(lo_param), abs(hi_param))))
    step_lo = step_hi = 0.0
    for _ in range(64):
        new = make(lo_param, hi_param)
        new_low, new_high = (float(bound) for bound in new.support())
        low_ok, high_ok = new_low >= outer_low, new_high <= outer_high
        if low_ok and high_ok:
            if new_low < new_high:
                return new
            break
        if not low_ok:
            step_lo = max(2.0 * step_lo, first_step)
            lo_param += step_lo
        if not high_ok:
            step_hi = max(2.0 * step_hi, first_step)
            hi_param -= step_hi
        if not lo_param < hi_param:
            break
    raise ValueError(
        f"Cannot rebuild a truncated {new.dist.name} inside the original support "
        f"[{outer_low}, {outer_high}]: the box is too narrow, widen it."
    )


def _truncate_marginal(dist, lo_u: float, hi_u: float):
    """Restrict one marginal to ``[dist.ppf(lo_u), dist.ppf(hi_u)]``, in the same family when possible."""
    if isinstance(dist, TruncatedMarginal):
        # Compose in the base cube instead of nesting wrappers. An untouched
        # face keeps its exact bound: lo + 1.0 * width can round past hi_u.
        width = dist.hi_u - dist.lo_u
        new_lo = dist.lo_u if lo_u == 0.0 else min(dist.lo_u + lo_u * width, dist.hi_u)
        new_hi = dist.hi_u if hi_u == 1.0 else min(dist.lo_u + hi_u * width, dist.hi_u)
        return TruncatedMarginal(dist.base, new_lo, new_hi)

    family = getattr(dist, "dist", None)
    name = getattr(family, "name", None)
    if name not in ("uniform", "loguniform", "reciprocal", "truncnorm"):
        return TruncatedMarginal(dist, lo_u, hi_u)

    low, high = float(dist.ppf(lo_u)), float(dist.ppf(hi_u))
    if not low < high:
        raise ValueError(_collapse_message(dist, lo_u, hi_u, low, high))
    outer_low, outer_high = (float(bound) for bound in dist.support())
    # _parse_args normalises positional and keyword loc/scale alike.
    shapes, loc, scale = family._parse_args(*dist.args, **dist.kwds)
    if name == "uniform":
        return _rebuild_inside(uniform, low, high, outer_low, outer_high)
    if name == "truncnorm":
        loc, scale = float(loc), float(scale)
        # Standardised bounds, as in norm(); an untouched face reuses the
        # original one so that it stays bit-identical to the original support.
        a = float(shapes[0]) if lo_u == 0.0 else (low - loc) / scale
        b = float(shapes[1]) if hi_u == 1.0 else (high - loc) / scale
        return _rebuild_inside(
            lambda a_, b_: stats.truncnorm(a_, b_, loc=loc, scale=scale), a, b, outer_low, outer_high
        )
    if loc == 0:
        # A pure rescale of a log-uniform is still log-uniform on [low, high].
        return _rebuild_inside(loguniform, low, high, outer_low, outer_high)
    # A shifted log-uniform has no closed form in this family.
    return TruncatedMarginal(dist, lo_u, hi_u)


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

    def truncated(
        self, lower_u: np.ndarray, upper_u: np.ndarray, *, min_width: float = 1e-9
    ) -> tuple["MultipleIndependent", float]:
        r"""Restrict the prior to a box of its unit cube $u = F(\theta)$.

        Each marginal is restricted to
        $[F_j^{-1}(u^{lo}_j), F_j^{-1}(u^{hi}_j)]$ and renormalised, so the
        returned prior $\pi_B$ satisfies
        $\pi_B(\theta) = \pi(\theta) / M_B$ inside the box, with prior mass
        $M_B = \prod_j (u^{hi}_j - u^{lo}_j)$, and its unit cube is the affine
        rescaling $F_{B,j}(\theta_j) = (F_j(\theta_j) - u^{lo}_j) / (u^{hi}_j - u^{lo}_j)$.
        An evidence computed under $\pi_B$ is therefore corrected with
        $\log Z = \log Z_B + \log M_B$.

        ``uniform``, ``loguniform`` and ``norm`` marginals stay in their SciPy
        family; others are wrapped in a
        [`TruncatedMarginal`][bsixsa.priors.TruncatedMarginal]. Truncated
        supports always lie inside the original ones.

        Parameters:
            lower_u: Lower box bounds in the unit cube, shape ``(ndim,)``.
            upper_u: Upper box bounds in the unit cube, shape ``(ndim,)``.
            min_width: Smallest accepted width on any axis.

        Returns:
            ``(prior, log_mass)``: the truncated prior and $\log M_B$.

        Raises:
            ValueError: If the box is invalid, too narrow, or collapses to a point.
        """
        lo, hi = _validate_unit_box(lower_u, upper_u, self.ndim, min_width)
        dists = [
            dist if (lo[j] == 0.0 and hi[j] == 1.0)
            else _truncate_marginal(dist, float(lo[j]), float(hi[j]))
            for j, dist in enumerate(self.dists)
        ]
        # Under the prior each u_j is independent U(0, 1): the box mass is exact.
        log_mass = float(np.sum(np.log(hi - lo)))
        return MultipleIndependent(dists), log_mass

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
            elif name == "truncated":
                raise NotImplementedError(
                    f"max_log_pdf is not implemented for axis {j}, a TruncatedMarginal "
                    f"over {dist.base!r}: its maximum depends on the wrapped family. "
                    "Only uniform, loguniform, and truncnorm priors (truncated or "
                    "not) are supported."
                )
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
