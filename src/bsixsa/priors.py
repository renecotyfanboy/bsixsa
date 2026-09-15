from collections import defaultdict
import numpy as np
from .convenience import XSilence, has_indexed_suffix
from xspec import AllModels
from .convenience import iter_all_parameters

__all__ = [
    'TruncatedMarginal',
    'build_prior',
    'loguniform',
    'norm',
    'uniform',
    'validate_finite_support',
]

def uniform(low, high):
    """Create a uniform prior on a finite interval.

    Parameters:
        low:
            Lower bound of the interval.
        high:
            Upper bound of the interval.

    Returns:
        A SciPy ``uniform`` distribution defined on ``[low, high]``.
    """
    from scipy.stats import uniform
    return uniform(low, high-low)

def loguniform(low, high):
    """Create a log-uniform prior on a positive interval.

    Parameters:
        low:
            Strictly positive lower bound.
        high:
            Upper bound.

    Returns:
        A SciPy ``loguniform`` distribution defined on ``[low, high]``.
    """
    from scipy.stats import loguniform
    return loguniform(low, high)

def norm(loc, scale, low, high):
    """Create a truncated Gaussian prior.

    Parameters:
        loc:
            Mean of the underlying Gaussian distribution.
        scale:
            Standard deviation of the underlying Gaussian distribution.
        low:
            Lower truncation bound, expressed in SciPy ``truncnorm``
            coordinates.
        high:
            Upper truncation bound, expressed in SciPy ``truncnorm``
            coordinates.

    Returns:
        A SciPy ``truncnorm`` distribution with the requested location,
        scale, and truncation limits.
    """
    from scipy.stats import truncnorm
    a, b = (low - loc) / scale, (high - loc) / scale
    return truncnorm(a, b, loc=loc, scale=scale)


def _scalar_or_array(values: np.ndarray):
    """Return a NumPy scalar for 0-d input and the array otherwise (SciPy's convention)."""
    return values[()]


class _TruncatedFamily:
    """Stand-in for a SciPy ``rv_continuous`` that only exposes the family name.

    Code dispatching on ``dist.dist.name`` (``xspec_bayes_contribution``,
    ``max_log_pdf_per_axis``, prior fingerprints) reads ``"truncated"`` for a
    [`TruncatedMarginal`][bsixsa.priors.TruncatedMarginal].
    """

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

    It is the fallback of
    [`MultipleIndependent.truncated`][bsixsa.priors.MultipleIndependent.truncated]
    for families without a closed-form truncation. It mimics the frozen SciPy
    interface used across bsixsa (``support``, ``cdf``, ``sf``, ``ppf``,
    ``isf`` -- required by nautilus --, ``logpdf``, ``pdf``, ``rvs``) and is
    picklable, so it can be shipped to worker processes.

    Attributes:
        base: Wrapped distribution.
        lo_u: Lower bound of the interval in the base unit cube.
        hi_u: Upper bound of the interval in the base unit cube.
        a: Lower physical bound, ``base.ppf(lo_u)``.
        b: Upper physical bound, ``base.ppf(hi_u)``.
        dist: Family descriptor whose ``name`` is ``"truncated"``.
        args: Always ``()``.
        kwds: Always ``{}``.
    """

    def __init__(self, base, lo_u: float, hi_u: float):
        """Wrap ``base`` restricted to ``[base.ppf(lo_u), base.ppf(hi_u)]``.

        Parameters:
            base:
                Distribution behaving like a frozen SciPy continuous
                distribution (at least ``cdf``, ``ppf`` and ``logpdf``).
            lo_u:
                Lower bound in the base unit cube.
            hi_u:
                Upper bound in the base unit cube.

        Raises:
            ValueError: If ``0 <= lo_u < hi_u <= 1`` does not hold, or if the
                interval collapses to a single physical value
                (``base.ppf(lo_u) >= base.ppf(hi_u)``), which would leave a
                non-normalisable density.
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

    def cdf(self, x):
        """Cumulative distribution function $(F(x) - u_{lo}) / (u_{hi} - u_{lo})$, clipped to ``[0, 1]``."""
        base_cdf = np.asarray(self.base.cdf(np.asarray(x, dtype=float)), dtype=float)
        return _scalar_or_array(np.clip((base_cdf - self.lo_u) / self._width, 0.0, 1.0))

    def sf(self, x):
        """Survival function $(u_{hi} - F(x)) / (u_{hi} - u_{lo})$, clipped to ``[0, 1]``."""
        base_cdf = np.asarray(self.base.cdf(np.asarray(x, dtype=float)), dtype=float)
        return _scalar_or_array(np.clip((self.hi_u - base_cdf) / self._width, 0.0, 1.0))

    def ppf(self, q):
        """Inverse CDF, ``base.ppf(lo_u + q * (hi_u - lo_u))``; ``nan`` outside ``[0, 1]``."""
        q = np.asarray(q, dtype=float)
        inside = (q >= 0.0) & (q <= 1.0)
        # Clip in both spaces: roundoff in lo + q*width must never step outside
        # the box, otherwise ppf(1) could land an ulp beyond b.
        u = np.clip(self.lo_u + np.where(inside, q, 0.0) * self._width, self.lo_u, self.hi_u)
        x = np.clip(np.asarray(self.base.ppf(u), dtype=float), self.a, self.b)
        return _scalar_or_array(np.where(inside, x, np.nan))

    def isf(self, q):
        """Inverse survival function, ``ppf(1 - q)``."""
        return self.ppf(1.0 - np.asarray(q, dtype=float))

    def logpdf(self, x):
        """Log-density ``base.logpdf(x) - log(hi_u - lo_u)`` on ``[a, b]``, ``-inf`` outside."""
        x = np.asarray(x, dtype=float)
        inside = (x >= self.a) & (x <= self.b)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_pdf = np.asarray(self.base.logpdf(x), dtype=float) - np.log(self._width)
        out = np.where(inside, log_pdf, -np.inf)
        return _scalar_or_array(np.where(np.isnan(x), np.nan, out))

    def pdf(self, x):
        """Probability density, ``exp(logpdf(x))``."""
        return _scalar_or_array(np.exp(np.asarray(self.logpdf(x), dtype=float)))

    def rvs(self, size=None, random_state=None):
        """Draw samples by inverse-CDF sampling.

        Parameters:
            size:
                Output shape, as in SciPy (``None`` draws a single scalar).
            random_state:
                Seed, ``numpy.random.Generator`` or legacy
                ``numpy.random.RandomState``.

        Returns:
            Samples within ``[a, b]``.
        """
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

    Rebuilding a family from its truncation bounds rounds them again (e.g.
    ``uniform`` stores ``scale = high - low`` and reports ``low + scale``), so a
    face can land one ulp outside the original support, where the original
    prior density is ``-inf``. Each offending bound is moved inward by a step
    that starts at one ulp of the bounds' magnitude and doubles, so the loop
    terminates quickly and moves a face by a few ulps at most in practice.

    Parameters:
        make:
            Callable ``(lo_param, hi_param) -> frozen distribution``.
        lo_param:
            Lower bound parameter passed to ``make``.
        hi_param:
            Upper bound parameter passed to ``make``.
        outer_low:
            Lower bound of the original support.
        outer_high:
            Upper bound of the original support.

    Returns:
        The rebuilt distribution, whose support is non-empty and inside the
        original one.

    Raises:
        ValueError: If no nudged pair of bounds fits (a box a few ulps wide).
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
    """Restrict one marginal to ``[dist.ppf(lo_u), dist.ppf(hi_u)]``, renormalised.

    ``uniform``, ``loguniform``/``reciprocal`` and ``truncnorm`` are rebuilt in
    closed form in the same SciPy family, so family-dispatching code
    (``xspec_bayes_contribution``, ``max_log_pdf_per_axis``) keeps working on
    the result. Everything else is wrapped in a ``TruncatedMarginal``. In
    every case the truncated support lies inside ``dist.support()``.
    """
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
        from scipy.stats import truncnorm

        loc, scale = float(loc), float(scale)
        # Standardised bounds, as in norm(); an untouched face reuses the
        # original one so that it stays bit-identical to the original support.
        a = float(shapes[0]) if lo_u == 0.0 else (low - loc) / scale
        b = float(shapes[1]) if hi_u == 1.0 else (high - loc) / scale
        return _rebuild_inside(
            lambda a_, b_: truncnorm(a_, b_, loc=loc, scale=scale), a, b, outer_low, outer_high
        )
    if loc == 0:
        # A pure rescale of a log-uniform is still log-uniform on [low, high].
        return _rebuild_inside(loguniform, low, high, outer_low, outer_high)
    # A shifted log-uniform has no closed form in this family.
    return TruncatedMarginal(dist, lo_u, hi_u)


def validate_finite_support(prior) -> None:
    """Require every axis of a prior to have a finite support.

    Prior restriction works in the unit cube and maps boxes back through
    ``ppf``, which only yields a usable physical box when both bounds are
    finite.

    Parameters:
        prior:
            A [`MultipleIndependent`][bsixsa.priors.MultipleIndependent] prior.

    Raises:
        ValueError: Naming every axis whose ``support()`` has an infinite (or
            ``nan``) bound.
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
    def __init__(self, dists):
        self.dists = list(dists)
        self.ndim = len(self.dists)
        self.bounds = [dist.support() for dist in self.dists]

    def sample(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        cols = [dist.rvs(size=size, random_state=rng) for dist in self.dists]
        x = np.stack(cols, axis=-1)
        return x

    def from_unit_cube(self, cube):
        cube = np.asarray(cube, dtype=float)
        params = np.empty_like(cube, dtype=float)
        for j, dist in enumerate(self.dists):
            params[..., j] = dist.ppf(cube[..., j])
        return params

    def to_unit_cube(self, x):
        x = np.asarray(x, dtype=float)
        cube = np.empty_like(x, dtype=float)
        for j, dist in enumerate(self.dists):
            cube[..., j] = dist.cdf(x[..., j])
        return cube

    def from_unit_gaussian(self, z):
        """Map ``z ~ N(0, I)^D`` to physical space via ``ppf(Phi(z))`` per marginal.

        Composes the standard-normal CDF ``Phi`` with each marginal's inverse CDF
        (``from_unit_cube``), so a standard-normal latent is mapped to the prior
        distribution. This is the unbounded counterpart of ``from_unit_cube`` used
        by the SIXSA backend, whose latent space is a unit Gaussian rather than the
        unit cube.
        """
        from scipy.stats import norm
        z = np.asarray(z, dtype=float)
        return self.from_unit_cube(norm.cdf(z))

    def to_unit_gaussian(self, x):
        """Map physical parameters to ``z ~ N(0, I)^D`` via ``Phi^{-1}(cdf(x))`` per marginal.

        Inverse of ``from_unit_gaussian``: the per-marginal CDF (``to_unit_cube``)
        followed by the standard-normal inverse CDF. The unit-cube values are
        clipped away from ``0``/``1`` so the inverse CDF stays finite at the prior
        bounds.
        """
        from scipy.stats import norm
        cube = np.clip(self.to_unit_cube(x), 1e-12, 1.0 - 1e-12)
        return norm.ppf(cube)

    def log_prob(self, x):
        x = np.asarray(x)
        x2 = np.atleast_2d(x)
        lp = np.zeros(x2.shape[0], dtype=float)

        for j, dist in enumerate(self.dists):
            lp += dist.logpdf(x2[:, j])

        return lp[0] if x.ndim == 1 else lp

    def xspec_bayes_contribution(self, x) -> float:
        """XSPEC ``Fit.bayes="on"`` prior contribution to the fit statistic.

        Evaluates ``-2 * Log(prior)`` per axis with XSPEC's unnormalised
        conventions (``Bayes::lPrior``) for the priors we compare against
        XSPEC: ``uniform`` -> ``cons`` contributes ``0``, ``loguniform`` ->
        ``jeffreys`` contributes ``2*ln(x)``. Adding this to the bare cstat
        at ``x`` reproduces XSPEC's ``Fit.statistic`` when ``Fit.bayes`` is
        on for those priors.
        """
        x = np.asarray(x, dtype=float).ravel()
        return float(self.xspec_bayes_contribution_batch(x.reshape(1, -1))[0])

    def xspec_bayes_contribution_batch(self, x) -> np.ndarray:
        """Vectorised [`xspec_bayes_contribution`][bsixsa.priors.MultipleIndependent.xspec_bayes_contribution].

        Parameters:
            x:
                Physical parameters of shape ``(n, ndim)``; a single point of
                shape ``(ndim,)`` is treated as ``n = 1``.

        Returns:
            Array of shape ``(n,)`` whose row ``i`` is
            ``xspec_bayes_contribution(x[i])``.

        Raises:
            ValueError: If ``x`` does not have ``ndim`` columns (extra
                columns would otherwise be silently ignored).
        """
        x = np.asarray(x, dtype=float)
        if x.ndim > 2 or np.atleast_2d(x).shape[1] != self.ndim:
            raise ValueError(
                f"x must have shape ({self.ndim},) or (n, {self.ndim}), got {x.shape}."
            )
        x = np.atleast_2d(x)
        contribution = np.zeros(x.shape[0], dtype=float)
        for j, dist in enumerate(self.dists):
            name = dist.dist.name
            if name == "uniform":
                continue
            elif name in ("loguniform", "reciprocal"):
                contribution += 2.0 * np.log(x[:, j])
            else:
                # Only uniform (-> cons) and loguniform (-> jeffreys) are
                # guaranteed to match XSPEC's bayes-on statistic exactly.
                # Everything else uses the scipy log-pdf, which differs
                # from the XSPEC counterpart by normalisation constants
                # (e.g. XSPEC's gauss prior is an untruncated Gaussian
                # while bsixsa's norm() is a truncated one) — acceptable,
                # as only uniform/loguniform fits are cross-checked
                # against XSPEC.
                contribution += -2.0 * np.asarray(dist.logpdf(x[:, j]), dtype=float)
        return contribution

    def truncated(
        self, lower_u, upper_u, *, min_width: float = 1e-9
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

        ``uniform``, ``loguniform`` and truncated ``norm`` marginals are
        rebuilt in the same SciPy family (so ``xspec_bayes_contribution`` and
        ``max_log_pdf_per_axis`` keep working); other families are wrapped in a
        [`TruncatedMarginal`][bsixsa.priors.TruncatedMarginal]. Axes spanning
        the whole ``[0, 1]`` keep their original distribution object. Every
        truncated support lies inside the original one, even when rebuilding
        a family would round a face an ulp past it (a face on the original
        support boundary is moved inward by a few ulps), so the original prior
        density stays finite on the whole truncated support.

        Parameters:
            lower_u:
                Lower box bounds in the unit cube, shape ``(ndim,)``.
            upper_u:
                Upper box bounds in the unit cube, shape ``(ndim,)``.
            min_width:
                Smallest accepted ``upper_u - lower_u`` on any axis.

        Returns:
            ``(prior, log_mass)``: the truncated prior and $\log M_B$.

        Raises:
            ValueError: If the bounds have the wrong shape, are not finite,
                violate ``0 <= lower_u < upper_u <= 1``, are narrower than
                ``min_width``, or collapse to a single physical value (for
                every family, including the ``TruncatedMarginal`` fallback).
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
        """Per-axis maximum of ``logpdf`` over the prior support.

        Used by least-squares MAP fits to shift the prior log-density into a
        non-negative range, so that the prior contribution can be encoded as
        extra residuals whose sum of squares equals
        ``-2 * log_prior + const``.

        Closed-form per family. Supported priors are ``uniform``,
        ``loguniform``, and truncated ``norm``.
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


def build_prior(define_prior):
    """Build the joint prior from a list of XSPEC prior specifications.

    For a **single-model** fit, each item must be a 3-tuple:

    `(component_name, parameter_name, distribution)`

    For a **multi-model** fit, each item must be a 4-tuple:

    `(model_name, component_name, parameter_name, distribution)`

    `distribution` can be and must behave like a SciPy continuous
    distribution and provide at least `support()`, `ppf()`,
    `cdf()`, `rvs()`, and `logpdf()`.
    Component and parameter names must match the names exposed by `xspec`.
    Repeated components can be addressed with suffixed names such as `powerlaw_3`.

    Parameters:
        define_prior:
            Sequence describing one prior per free XSPEC parameter.

    Returns:
        `(prior, parameters_index, model_index)`

        `prior` is a [`MultipleIndependent`][bsixsa.priors.MultipleIndependent]
        distribution over all unfrozen, unlinked fitted parameters. Its
        per-parameter support bounds are available on `prior.bounds`.
    """
    model_list = [AllModels(1, modName=name) for name in AllModels.sources.values()]

    is_single_model_prior = all(len(definition) == 3 for definition in define_prior)
    is_multi_model_prior = all(len(definition) == 4 for definition in define_prior)

    # Sanity checks
    if not (is_single_model_prior or is_multi_model_prior):
        raise ValueError("Prior must be defined either as single model or multi-model")

    if (is_single_model_prior and len(model_list) != 1) or (is_multi_model_prior and len(model_list) == 1):
        raise ValueError("Single model prior defined for multiple models (or the opposite)")

    # Let's format the prior to the general multimodel setup
    if is_single_model_prior:

        # We explicitly add the model name at the start of the prior settings
        model_name = model_list[0].name
        define_prior = [(model_name,) + prior_setting for prior_setting in define_prior]

    # We turn the list of prior settings in a nested dictionary indexed by model/components/parameter names
    define_prior_dict = defaultdict(lambda: defaultdict(dict))

    for model_name, component_name, parameter_name, distribution in define_prior:

        model = AllModels(1, modName=model_name)

        # Handle the weird situation where a component is defined multiple times
        # EG tbabs*(powerlaw + powerlaw) will yield tbabs, powerlaw & powerlaw_3 as component names
        # An insightful user might want to pass "powerlaw_2" & "powerlaw_3" instead of "powerlaw" & "powerlaw_3"
        if has_indexed_suffix(component_name):
            if component_name not in model.componentNames:
                split_name = component_name.split('_')[0]
                if split_name in model.componentNames:
                    component_name = split_name
                else:
                    raise ValueError(f"Component '{component_name}' or '{split_name}' not in {model.componentNames}")

        # Nested defaultdicts handle intermediate dictionary creation
        define_prior_dict[model_name][component_name][parameter_name] = distribution

    # Now we build the necessary blocks for inference
    parameter_to_set = defaultdict(dict)  # Will contain {model : {par_index:prior}}
    list_of_prior = []
    model_index = []
    parameters_index = []
    for par in iter_all_parameters():

        prior = None

        # Smooth brain syntax to check whether a prior exists for a given parameter and extract its value
        match define_prior_dict:
            # Prior is only instantiated in the following line if there is a match
            case {par.model.name: {par.component.name: {par.parameter.name: prior}}}:
                pass

        if prior is None and not (par.parameter.frozen or bool(par.parameter.link)):
            raise ValueError(f"No prior or link for {par.name}. Either pass a prior or freeze explicitly.")

        if prior is not None:

            low, high = prior.support()
            # Range-proportional fit delta: XSPEC only uses these per-parameter
            # deltas when Fit.delta == 0 (PyXspec defaults to proportional
            # deltas of 0.01), but a fixed 0.1 here is a landmine — it exceeds
            # e.g. a whole redshift range and is ~1e5 times a typical norm.
            delta = 0.01 * (high - low)
            parameter_to_set[par.model][par.parameter.index] = f"{np.random.uniform(low, high)},{delta},{low},{low},{high},{high}"

            list_of_prior.append(prior)
            parameters_index.append(par.parameter.index)
            model_index.append(par.model.name)

    # Turn the model / index dict into args for AllModels.set and apply them
    parameter_to_set = sum(((k, v) for k, v in parameter_to_set.items()), ())

    with XSilence():
        AllModels.setPars(*parameter_to_set)

    prior = MultipleIndependent(list_of_prior)
    return prior, parameters_index, model_index
