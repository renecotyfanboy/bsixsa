"""Warnings and errors raised by the restriction and SIXSA machinery.

The module itself is deliberately dependency-free (no XSPEC, no torch). Note
that importing it as ``bsixsa.exceptions`` still executes the package
``bsixsa/__init__``, which imports the solver and therefore XSPEC.
"""

__all__ = [
    "BoxEdgeWarning",
    "FoldError",
    "TruncationBudgetError",
]


class BoxEdgeWarning(UserWarning):
    """Posterior mass crowds an interior face of a restricted prior box.

    Raised when posterior samples pile up against a face of the bounding box
    that was introduced by a restriction (not a face of the original prior
    support). The restriction likely cut off posterior mass, so the box or
    the history-matching settings should be widened.
    """


class FoldError(RuntimeError):
    r"""A fold returned an invalid fit statistic.

    Raised when a simulated cstat is ``NaN`` or $-\infty$. A value of
    $+\infty$ is a legitimate zero-likelihood point and is not an error.
    """


class TruncationBudgetError(RuntimeError):
    r"""The exact truncated sampler exceeded its candidate budget.

    Raised when drawing from the truncated region
    $A = [0, 1]^d \cap R \cap \{\log q > t\}$ would require more candidate
    draws than allowed. There is no silent fallback to an approximate sampler.
    """
