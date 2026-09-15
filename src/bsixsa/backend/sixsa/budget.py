r"""Simulation budgets of the SIXSA rounds.

Three pure rules decide how many folds SIXSA spends:

- [`training_set_size`][bsixsa.backend.sixsa.budget.training_set_size]: the
  size of the first training set, from the number of free parameters and the
  length of the spectrum;
- [`next_training_size`][bsixsa.backend.sixsa.budget.next_training_size]: the
  geometric ramp between rounds;
- [`final_draws`][bsixsa.backend.sixsa.budget.final_draws]: the number of raw
  proposal draws of the final importance sample, fixed in advance from a fresh
  certificate so that the estimate carries no stopping bias.

The module depends on the standard library only (no numpy, torch or XSPEC).
"""

from __future__ import annotations

import math
import operator

__all__ = ["final_draws", "next_training_size", "training_set_size"]


def _positive_int(value, name: str, *, minimum: int = 1) -> int:
    """Return *value* as an ``int`` not smaller than *minimum*.

    Parameters:
        value:
            Integer-like value.
        name:
            Argument name used in error messages.
        minimum:
            Smallest accepted value.

    Returns:
        The value as a Python ``int``.

    Raises:
        TypeError: If *value* is not an integer.
        ValueError: If *value* is smaller than *minimum*.
    """
    try:
        out = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.") from None
    if out < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {out}.")
    return out


def training_set_size(
    d: int,
    n_bins: int,
    *,
    base: float = 2000.0,
    per_param_min: int = 750,
    ref_d: float = 6.0,
    exponent: float = 0.6,
    bins_ref: float = 2500.0,
    bins_exponent: float = 0.25,
    round_to: int = 250,
) -> int:
    r"""Number of simulated spectra in the first training set.

    The count per free parameter is

    $$
    p = \mathrm{base} \left(\frac{d}{d_{ref}}\right)^{\mathrm{exponent}}
        \operatorname{clip}\left(
            \left(\frac{n_{bins}}{n_{ref}}\right)^{\mathrm{bins\_exponent}},
            0.5, 3\right),
    $$

    rounded to the nearest multiple of *round_to* and raised to at least
    *per_param_min*; the set holds $p \cdot d$ spectra. The anchor is 2000 per
    parameter at $d = 6$ on a 2500-bin spectrum. A longer spectrum needs more
    spectra, because the compressor's first layer is *n_bins* wide. This is
    the v2 heuristic (a calibration, not a measurement), without the crash its
    logging hit when ``per_param_min`` was left at ``None``.

    Parameters:
        d:
            Number of free parameters, at least 1.
        n_bins:
            Number of channels of the spectrum. The binning factor is skipped
            (taken as 1) when it is 0 or *bins_exponent* is 0.
        base:
            Spectra per parameter at the anchor.
        per_param_min:
            Floor of the per-parameter count.
        ref_d:
            Dimension of the anchor.
        exponent:
            Growth exponent of the per-parameter count with *d*.
        bins_ref:
            Channel count of the anchor.
        bins_exponent:
            Growth exponent with *n_bins*.
        round_to:
            Rounding step of the per-parameter count (at least 1 is used).

    Returns:
        The training-set size, a multiple of *d*.

    Raises:
        TypeError: If *d* or *n_bins* is not an integer.
        ValueError: If *d* is smaller than 1 or *n_bins* is negative.
    """
    d = _positive_int(d, "d")
    n_bins = _positive_int(n_bins, "n_bins", minimum=0)
    per_param = float(base) * (d / max(float(ref_d), 1e-9)) ** float(exponent)
    if float(bins_exponent) != 0.0 and n_bins > 0:
        bins_factor = (n_bins / max(float(bins_ref), 1e-9)) ** float(bins_exponent)
        per_param *= min(max(bins_factor, 0.5), 3.0)
    step = max(1, int(round_to))
    per_param = max(int(round(per_param / step)) * step, int(per_param_min))
    return int(per_param) * d


def next_training_size(n: int, *, ramp: float, n_max: int | None) -> int:
    """Size of the next round's training set.

    Parameters:
        n:
            Size of the current training set, at least 1.
        ramp:
            Growth factor per round, at least 1.
        n_max:
            Cap on the size, or ``None`` for no cap.

    Returns:
        ``min(round(n * ramp), n_max)``.

    Raises:
        TypeError: If *n* or *n_max* is not an integer.
        ValueError: If *n* or *n_max* is smaller than 1, or *ramp* is smaller
            than 1 or not finite.
    """
    n = _positive_int(n, "n")
    ramp = float(ramp)
    if not math.isfinite(ramp) or ramp < 1.0:
        raise ValueError(f"ramp must be a finite number >= 1, got {ramp}.")
    size = int(round(n * ramp))
    if n_max is None:
        return size
    return min(size, _positive_int(n_max, "n_max"))


def final_draws(
    *,
    target_ess: float,
    ess_fresh: float,
    n_fresh: int,
    max_draws: int,
    chunk: int = 5000,
    min_draws: int = 5000,
) -> int:
    r"""Number of raw draws of the final importance sample, fixed in advance.

    The efficiency of the fresh certificate, $\eta = \mathrm{ESS}/N$, predicts
    the draws needed for *target_ess* with a 25 % margin,
    $N = \lceil 1.25 \, \mathrm{target\_ess} / \eta \rceil$. $N$ is clipped
    to ``[min_draws, max_draws]`` and rounded up to a multiple of *chunk*, but
    never above *max_draws*: when *max_draws* is not a multiple of *chunk*
    and the rounding would overshoot it, *max_draws* itself is returned. A
    non-finite or non-positive efficiency is treated as $10^{-12}$, which
    requests *max_draws*.

    Parameters:
        target_ess:
            Effective sample size wanted from the final sample, positive.
        ess_fresh:
            Effective sample size of the fresh certificate.
        n_fresh:
            Number of raw draws of that certificate.
        max_draws:
            Hard cap on the result, at least 1; it wins over *min_draws*.
        chunk:
            Rounding step, at least 1.
        min_draws:
            Floor of the result before rounding.

    Returns:
        The number of raw draws, in ``[1, max_draws]``.

    Raises:
        TypeError: If an integer argument is not an integer.
        ValueError: If *target_ess* is not a positive finite number, or
            *max_draws* or *chunk* is smaller than 1.
    """
    target_ess = float(target_ess)
    if not math.isfinite(target_ess) or target_ess <= 0.0:
        raise ValueError(f"target_ess must be a positive finite number, got {target_ess}.")
    max_draws = _positive_int(max_draws, "max_draws")
    chunk = _positive_int(chunk, "chunk")
    min_draws = _positive_int(min_draws, "min_draws", minimum=0)
    n_fresh = _positive_int(n_fresh, "n_fresh", minimum=0)

    efficiency = float(ess_fresh) / max(n_fresh, 1)
    if not math.isfinite(efficiency) or efficiency < 1e-12:
        efficiency = 1e-12
    wanted = 1.25 * target_ess / efficiency
    # A float above max_draws needs no exact ceiling, and converting a huge
    # float first would only risk an overflow.
    if wanted >= max_draws:
        return max_draws
    n = max(math.ceil(wanted), min_draws)
    n = min(n, max_draws)
    n = -(-n // chunk) * chunk
    return min(n, max_draws)
