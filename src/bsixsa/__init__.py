from .backend import (
    Emcee,
    Iminuit,
    LevenbergMarquardt,
    Nautilus,
    Nessai,
    BackendConfig,
    SIXSA,
    Ultranest,
)
from .exceptions import BoxEdgeWarning, FoldError, TruncationBudgetError
from .solver import FitResults, Solver

# Public alias: `SIXSASolver` is the SIXSA-facing name for the solver entry point
# (used by tests and some docstrings); `Solver` is the generic name used in the
# docs, notebooks, and internal type hints. Both refer to the same class.
SIXSASolver = Solver

__all__ = [
    "BoxEdgeWarning",
    "Emcee",
    "FitResults",
    "FoldError",
    "HistoryMatching",
    "Iminuit",
    "LevenbergMarquardt",
    "Nautilus",
    "Nessai",
    "Solver",
    "SIXSASolver",
    "BackendConfig",
    "SIXSA",
    "TruncationBudgetError",
    "Ultranest",
]


def __getattr__(name: str):
    """Resolve the restriction exports on first access.

    Parameters:
        name: Attribute requested on the package.

    Returns:
        The requested object.

    Raises:
        AttributeError: If *name* is not a lazy export.
    """
    # Loaded lazily so that `import bsixsa` never pays for the restriction
    # package (whose regressor needs torch).
    if name == "HistoryMatching":
        from .restriction.config import HistoryMatching

        return HistoryMatching
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List the package attributes, including the lazy exports.

    Returns:
        Sorted attribute names.
    """
    return sorted(set(globals()) | set(__all__))
