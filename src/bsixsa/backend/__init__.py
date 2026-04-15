from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .abc import Backend
    from .cmaes import CMAESBackend
    from .config import (
        Emcee,
        Iminuit,
        LevenbergMarquardt,
        Nautilus,
        Nessai,
        BackendConfig,
        Ultranest,
    )
    from .emcee import EmceeBackend
    from .iminuit import IminuitBackend
    from .levenberg_marquart import LevenbergMarquardtBackend
    from .nautilus import NautilusBackend
    from .nessai import NessaiBackend
    from .sixsa import SIXSABackend
    from .tracer import EvaluationTracer
    from .ultranest import UltranestBackend

__all__ = [
    "Backend",
    "CMAESBackend",
    "EvaluationTracer",
    "EmceeBackend",
    "Emcee",
    "IminuitBackend",
    "Iminuit",
    "LevenbergMarquardtBackend",
    "LevenbergMarquardt",
    "NautilusBackend",
    "Nautilus",
    "NessaiBackend",
    "Nessai",
    "SIXSABackend",
    "BackendConfig",
    "UltranestBackend",
    "Ultranest",
    "BACKEND_REGISTRY",
    "get_backend_class",
    "register_backend",
]

from .config import (
    Emcee,
    Iminuit,
    LevenbergMarquardt,
    Nautilus,
    Nessai,
    BackendConfig,
    Ultranest,
)

BACKEND_REGISTRY: dict[str, type[Backend]] = {}


def register_backend(cls: type[Backend]) -> type[Backend]:
    """Register a :class:`Backend` subclass by its ``name`` attribute."""
    if not hasattr(cls, "name"):
        raise ValueError(f"{cls.__name__} must define a class-level `name` attribute.")
    if cls.name in BACKEND_REGISTRY:
        raise ValueError(
            f"Backend name '{cls.name}' is already registered "
            f"by {BACKEND_REGISTRY[cls.name].__name__}."
        )
    BACKEND_REGISTRY[cls.name] = cls
    return cls


def get_backend_class(name: str) -> type[Backend]:
    """Look up a backend class by name, importing lazily if needed."""
    if name not in BACKEND_REGISTRY:
        # Trigger lazy imports so decorators run
        _lazy_import(name)
    if name not in BACKEND_REGISTRY:
        available = ", ".join(sorted(BACKEND_REGISTRY)) or "(none)"
        raise ValueError(
            f"Unknown backend '{name}'. Available backends: {available}"
        )
    return BACKEND_REGISTRY[name]


def _lazy_import(name: str) -> None:
    """Import backend modules on demand to populate the registry."""
    _module_map = {
        "nessai": ".nessai",
        "nautilus": ".nautilus",
        "ultranest": ".ultranest",
        "levenberg_marquardt": ".levenberg_marquart",
        "sixsa": ".sixsa",
        "cmaes": ".cmaes",
        "iminuit": ".iminuit",
        "emcee": ".emcee"
    }
    if name in _module_map:
        import importlib
        importlib.import_module(_module_map[name], package=__package__)
