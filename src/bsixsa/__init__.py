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
from .solver import FitResults, Solver

# Public alias: `SIXSASolver` is the SIXSA-facing name for the solver entry point
# (used by tests and some docstrings); `Solver` is the generic name used in the
# docs, notebooks, and internal type hints. Both refer to the same class.
SIXSASolver = Solver

__all__ = [
    "Emcee",
    "FitResults",
    "Iminuit",
    "LevenbergMarquardt",
    "Nautilus",
    "Nessai",
    "Solver",
    "SIXSASolver",
    "BackendConfig",
    "SIXSA",
    "Ultranest",
]
