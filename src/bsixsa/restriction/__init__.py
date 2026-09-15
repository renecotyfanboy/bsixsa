"""History-matching restriction of the prior, shared by every backend.

Waves of exact folds train a C-stat surrogate on the prior's unit cube and cut
it down to the region worth sampling. The configuration and the result
classes import without torch; `run_history_matching` (which trains the torch
regressor) is loaded on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import HistoryMatching
from .region import Region, Restriction

if TYPE_CHECKING:
    from .waves import run_history_matching

__all__ = ["HistoryMatching", "Region", "Restriction", "run_history_matching"]

_LAZY = {"run_history_matching": ".waves"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from importlib import import_module

        value = getattr(import_module(_LAZY[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
