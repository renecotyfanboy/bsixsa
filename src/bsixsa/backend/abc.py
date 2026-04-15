from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel
from scipy.special import expit as sigmoid

if TYPE_CHECKING:
    from ..solver import FitResults, Solver
    from .config import BackendConfig


class Backend(ABC):
    """Base class for all backends.

    Subclasses must set a class-level ``name`` attribute (used as the
    registry key) and implement :meth:`run` and :meth:`sample`.
    """

    name: str  # registry key, e.g. "nessai"
    config_cls: type[BaseModel] | None = None

    def __init__(
        self,
        *,
        solver: Solver,
        config: BackendConfig,
        **kwargs,
    ):

        self.validate_config(solver=solver, config=config)
        self.solver = solver
        self.config = config
        self.tracer = config.create_tracer(
            output_dir=solver.outputfiles_basename,
            parameter_names=solver.parameter_names,
        )

    @classmethod
    def validate_config(cls, *, solver: Solver, config: BackendConfig) -> None:
        expected_config_cls = cls.config_cls
        if expected_config_cls is None:
            if config is not None:
                raise TypeError(f"Backend '{cls.name}' does not accept a `config` object.")
            return
        if not isinstance(config, expected_config_cls):
            raise TypeError(
                f"`config` must be an instance of {expected_config_cls.__name__} for backend "
                f"'{cls.name}'."
            )
        config.validate_for_solver(solver)

    def _cube_to_physical(self, cube: np.ndarray) -> np.ndarray:
        """Map unit-cube parameters to physical space."""
        return self.solver.prior.from_unit_cube(np.atleast_2d(cube))

    def _physical_to_cube(self, physical: np.ndarray) -> np.ndarray:
        """Map physical-space parameters to the unit cube."""
        return self.solver.prior.to_unit_cube(
            np.atleast_2d(physical)
        ).ravel()

    def _unconstrained_to_physical(self, unconstrained: np.ndarray) -> np.ndarray:
        """Map unconstrained-space parameters to physical space (sigmoid then inverse CDF)."""
        unit_cube = sigmoid(np.atleast_2d(unconstrained))
        return self.solver.prior.from_unit_cube(unit_cube)

    def _physical_to_unconstrained(self, physical: np.ndarray) -> np.ndarray:
        """Map physical-space parameters to unconstrained space (CDF then logit)."""
        unit_cube = self.solver.prior.to_unit_cube(
            np.atleast_2d(physical)
        ).ravel()
        unit_cube = np.clip(unit_cube, 1e-25, 1.0 - 1e-25)
        return np.log(unit_cube / (1.0 - unit_cube))

    @abstractmethod
    def run(self, **kwargs) -> FitResults:
        """Execute the backend and return results."""
        ...

    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        """Draw *n* posterior samples after a completed run."""
        ...
