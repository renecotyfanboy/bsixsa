from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from .tracer import EvaluationTracer

if TYPE_CHECKING:
    from ..solver import FitResults, SIXSASolver


class Backend(ABC):
    """Base class for all backends.

    Subclasses must set a class-level ``name`` attribute (used as the
    registry key) and implement :meth:`run` and :meth:`sample`.
    """

    name: str  # registry key, e.g. "nessai"

    def __init__(self, *, solver: SIXSASolver, plot_every: int = 50, flush_every: int = 200, **kwargs):
        self.solver = solver
        self.tracer = EvaluationTracer(
            output_dir=solver.outputfiles_basename,
            parameter_names=solver.parameter_names,
            plot_every=plot_every,
            flush_every=flush_every,
        )

    @abstractmethod
    def run(self, **kwargs) -> FitResults:
        """Execute the backend and return results."""
        ...

    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        """Draw *n* posterior samples after a completed run."""
        ...