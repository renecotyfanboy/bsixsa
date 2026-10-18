"""Bayesian X-ray spectral fitting on top of PyXspec.

Load a spectrum and a model in PyXspec, build a [`Solver`][bsixsa.solver.Solver]
from a prior (see [`bsixsa.priors`][bsixsa.priors]) and a backend configuration
(`Nautilus`, `Nessai`, `Ultranest`, `Emcee`, `Iminuit`, `LevenbergMarquardt` or
`SIXSA`), and call `run()`. [`HistoryMatching`][bsixsa.restriction.HistoryMatching]
configures the optional prior restriction, and
[`export_xspec_chain`][bsixsa.export.export_xspec_chain] writes the posterior as
an XSPEC chain file.
"""

from .backend import (
    SIXSA,
    BackendConfig,
    Emcee,
    Iminuit,
    LevenbergMarquardt,
    Nautilus,
    Nessai,
    Ultranest,
)
from .backend.config import CompressorConfig, NDEConfig, TrainingConfig
from .export import export_xspec_chain
from .restriction import HistoryMatching
from .solver import FitResults, Solver

__all__ = [
    "SIXSA",
    "BackendConfig",
    "CompressorConfig",
    "Emcee",
    "FitResults",
    "HistoryMatching",
    "Iminuit",
    "LevenbergMarquardt",
    "NDEConfig",
    "Nautilus",
    "Nessai",
    "Solver",
    "TrainingConfig",
    "Ultranest",
    "export_xspec_chain",
]
