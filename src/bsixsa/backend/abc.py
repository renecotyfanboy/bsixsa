from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from .tracer import EvaluationTracer, DummyTracer

if TYPE_CHECKING:
    from ..solver import FitResults, SIXSASolver


class Backend(ABC):
    """Base class for all backends.

    Subclasses must set a class-level ``name`` attribute (used as the
    registry key) and implement :meth:`run` and :meth:`sample`.
    """

    name: str  # registry key, e.g. "nessai"

    def __init__(self, *, solver: SIXSASolver, trace=True, plot_every: int = 50, flush_every: int = 200, **kwargs):
        self.solver = solver

        if trace:
            self.tracer = EvaluationTracer(
                output_dir=solver.outputfiles_basename,
                parameter_names=solver.parameter_names,
                plot_every=plot_every,
                flush_every=flush_every,
            )
        else:
            self.tracer = DummyTracer(
                output_dir=solver.outputfiles_basename,
                parameter_names=solver.parameter_names,
                plot_every=plot_every,
                flush_every=flush_every,
            )

    def _cube_to_physical(self, cube: np.ndarray) -> np.ndarray:
        """Map unit-cube parameters to physical space."""
        return self.solver.prior.from_unit_cube(np.atleast_2d(cube))

    def _physical_to_cube(self, physical: np.ndarray) -> np.ndarray:
        """Map physical-space parameters to the unit cube."""
        return self.solver.prior.to_unit_cube(
            np.atleast_2d(physical)
        ).ravel()

    @abstractmethod
    def run(self, **kwargs) -> FitResults:
        """Execute the backend and return results."""
        ...

    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        """Draw *n* posterior samples after a completed run."""
        ...