"""The backends: the fitting methods a [`Solver`][bsixsa.solver.Solver] can run.

Choose one by passing its configuration as `Solver(backend=...)`:

| configuration | method | `FitResults` holds |
|---|---|---|
| `Nautilus`, `Nessai`, `Ultranest` | nested sampling | posterior samples and log Z |
| `SIXSA` | simulation-based inference, certified by importance sampling | posterior samples and log Z |
| `Emcee` | ensemble MCMC | posterior samples (correlated chain) |
| `LevenbergMarquardt`, `Iminuit` | optimisation | best fit and a Gaussian approximation of the posterior |

Only the sampler you use needs to be installed.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .config import (
    SIXSA,
    BackendConfig,
    Emcee,
    Iminuit,
    LevenbergMarquardt,
    Nautilus,
    Nessai,
    Ultranest,
)

if TYPE_CHECKING:
    from .abc import Backend

__all__ = [
    "BACKEND_REGISTRY",
    "SIXSA",
    "BackendConfig",
    "Emcee",
    "Iminuit",
    "LevenbergMarquardt",
    "Nautilus",
    "Nessai",
    "Ultranest",
    "get_backend_class",
]

# `backend_name` of a configuration -> "module:Class" (or class) of the backend that runs it.
BACKEND_REGISTRY: dict[str, str | type[Backend]] = {
    "nautilus": "bsixsa.backend.nautilus:NautilusBackend",
    "nessai": "bsixsa.backend.nessai:NessaiBackend",
    "ultranest": "bsixsa.backend.ultranest:UltranestBackend",
    "levenberg_marquardt": "bsixsa.backend.levenberg_marquardt:LevenbergMarquardtBackend",
    "iminuit": "bsixsa.backend.iminuit:IminuitBackend",
    "emcee": "bsixsa.backend.emcee:EmceeBackend",
    "sixsa": "bsixsa.backend.sixsa._sixsa:SIXSABackend",
}


def get_backend_class(name: str) -> type[Backend]:
    """Return the backend class that runs configurations with this `backend_name`.

    Parameters:
        name: The configuration's `backend_name`, e.g. `"nautilus"`.

    Returns:
        The backend class.

    Raises:
        ValueError: If no backend has this name.
    """
    if name not in BACKEND_REGISTRY:
        raise ValueError(f"Unknown backend {name!r}. Available backends: {', '.join(sorted(BACKEND_REGISTRY))}.")
    entry = BACKEND_REGISTRY[name]
    if isinstance(entry, str):
        module, class_name = entry.split(":")
        entry = getattr(importlib.import_module(module), class_name)
    return entry

