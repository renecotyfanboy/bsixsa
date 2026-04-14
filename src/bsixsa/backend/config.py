from __future__ import annotations

"""Pydantic-backed configuration models for bsixsa backends."""

from typing import Annotated, Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
PositiveFloat = Annotated[float, Field(strict=True, gt=0)]
PlotStepPercent = Annotated[int, Field(strict=True, gt=0, le=100)]
MODEL_CONFIG = ConfigDict(
    arbitrary_types_allowed=True,
    extra="forbid",
    validate_default=True,
)


class TraceConfig(BaseModel):
    """Shared tracing options for backend config models."""

    model_config = MODEL_CONFIG

    trace: StrictBool | None = Field(
        default=None,
        description="Override the solver trace flag. Use None to inherit solver.trace.",
    )
    plot_every: PositiveInt = Field(
        default=50,
        description=(
            "Write the first progress plot after this many evaluations. "
            "Later plots follow `plot_step_percent` on a log-scaled schedule."
        ),
    )
    plot_step_percent: PlotStepPercent = Field(
        default=10,
        description=(
            "Relative plotting cadence after the first plot. "
            "The default of 10 reproduces the current 10%-per-decade schedule."
        ),
    )


class NautilusConfig(TraceConfig):
    """Configuration for :class:`bsixsa.backend.nautilus.NautilusBackend`."""

    backend_name: ClassVar[str] = "nautilus"

    num_live_points: PositiveInt = Field(
        default=1_000,
        description="Number of live points used by the nested sampler.",
    )


class NessaiConfig(TraceConfig):
    """Configuration for :class:`bsixsa.backend.nessai.NessaiBackend`."""

    backend_name: ClassVar[str] = "nessai"
    _protected_engine_keys: ClassVar[set[str]] = {
        "nlive",
        "output",
        "importance_nested_sampler",
    }

    num_live_points: PositiveInt = Field(
        default=1_000,
        description="Number of live points used by the nested sampler.",
    )
    engine_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional keyword arguments forwarded to nessai's FlowSampler. "
            "Reserved keys such as 'nlive', 'output', and "
            "'importance_nested_sampler' are not allowed."
        ),
    )

    @model_validator(mode="after")
    def _validate_engine_key_conflicts(self) -> NessaiConfig:
        conflicts = sorted(self._protected_engine_keys.intersection(self.engine_kwargs))
        if conflicts:
            raise ValueError(
                "`engine_kwargs` cannot override fixed backend options: "
                f"{', '.join(conflicts)}."
            )
        return self


class UltranestConfig(TraceConfig):
    """Configuration for :class:`bsixsa.backend.ultranest.UltranestBackend`."""

    backend_name: ClassVar[str] = "ultranest"

    num_live_points: PositiveInt = Field(
        default=1_000,
        description="Number of live points used by the nested sampler.",
    )
    resume: StrictBool = Field(
        default=True,
        description="Resume a previous UltraNest run if checkpoint files exist.",
    )
    use_step_sampler: StrictBool = Field(
        default=False,
        description="Enable UltraNest's population step sampler.",
    )


class IminuitConfig(TraceConfig):
    """Configuration for :class:`bsixsa.backend.iminuit.IminuitBackend`."""

    backend_name: ClassVar[str] = "iminuit"

    x0_physical: np.ndarray | None = Field(
        default=None,
        description=(
            "Initial parameter vector in physical parameter space. "
            "Must contain one value per free parameter."
        ),
    )
    init_from_prior: StrictBool = Field(
        default=False,
        description="Initialise the optimizer from a draw from the prior.",
    )
    run_hesse: StrictBool = Field(
        default=True,
        description="Run HESSE after MIGRAD to estimate parameter covariance.",
    )
    engine_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments forwarded to Minuit.migrad().",
    )

    @field_validator("x0_physical", mode="before")
    @classmethod
    def _validate_x0_physical(cls, value: Any) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            raise ValueError("`x0_physical` must be a one-dimensional parameter vector.")
        vector = array.ravel()
        if vector.size == 0:
            raise ValueError("`x0_physical` cannot be empty.")
        if not np.all(np.isfinite(vector)):
            raise ValueError("`x0_physical` must contain only finite values.")
        return vector

    @model_validator(mode="after")
    def _validate_initialisation_strategy(self) -> IminuitConfig:
        if self.x0_physical is not None and self.init_from_prior:
            raise ValueError(
                "`x0_physical` and `init_from_prior=True` are mutually exclusive."
            )
        return self


class LevenbergMarquardtConfig(TraceConfig):
    """Configuration for Levenberg-Marquardt optimisation."""

    backend_name: ClassVar[str] = "levenberg_marquardt"

    x0_physical: np.ndarray | None = Field(
        default=None,
        description=(
            "Initial parameter vector in physical parameter space. "
            "Must contain one value per free parameter."
        ),
    )
    init_from_prior: StrictBool = Field(
        default=False,
        description="Initialise the optimizer from a draw from the prior.",
    )
    engine_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments forwarded to scipy.optimize.least_squares().",
    )

    @field_validator("x0_physical", mode="before")
    @classmethod
    def _validate_x0_physical(cls, value: Any) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            raise ValueError("`x0_physical` must be a one-dimensional parameter vector.")
        vector = array.ravel()
        if vector.size == 0:
            raise ValueError("`x0_physical` cannot be empty.")
        if not np.all(np.isfinite(vector)):
            raise ValueError("`x0_physical` must contain only finite values.")
        return vector

    @model_validator(mode="after")
    def _validate_initialisation_strategy(self) -> LevenbergMarquardtConfig:
        if self.x0_physical is not None and self.init_from_prior:
            raise ValueError(
                "`x0_physical` and `init_from_prior=True` are mutually exclusive."
            )
        return self


class EmceeConfig(TraceConfig):
    """Configuration for :class:`bsixsa.backend.emcee.EmceeBackend`."""

    backend_name: ClassVar[str] = "emcee"

    x0_physical: np.ndarray | None = Field(
        default=None,
        description=(
            "Initial parameter vector in physical parameter space. "
            "Must contain one value per free parameter."
        ),
    )
    init_from_prior: StrictBool = Field(
        default=False,
        description="Initialise the optimizer from a draw from the prior.",
    )
    num_warmup: PositiveInt = Field(
        default=100,
        description="Number of warmup or burn-in steps to discard.",
    )
    num_samples: PositiveInt = Field(
        default=10_000,
        description="Number of production samples to draw after warmup.",
    )
    num_walkers: PositiveInt = Field(
        default=32,
        description="Number of walkers in the ensemble sampler.",
    )
    init_spread: PositiveFloat = Field(
        default=1e-3,
        description=(
            "Relative Gaussian spread applied to the initial point to generate "
            "the walker ensemble."
        ),
    )

    @field_validator("x0_physical", mode="before")
    @classmethod
    def _validate_x0_physical(cls, value: Any) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            raise ValueError("`x0_physical` must be a one-dimensional parameter vector.")
        vector = array.ravel()
        if vector.size == 0:
            raise ValueError("`x0_physical` cannot be empty.")
        if not np.all(np.isfinite(vector)):
            raise ValueError("`x0_physical` must contain only finite values.")
        return vector

    @model_validator(mode="after")
    def _validate_initialisation_strategy(self) -> EmceeConfig:
        if self.x0_physical is not None and self.init_from_prior:
            raise ValueError(
                "`x0_physical` and `init_from_prior=True` are mutually exclusive."
            )
        if self.x0_physical is None and not self.init_from_prior:
            raise ValueError(
                "EmceeConfig requires either `x0_physical` or `init_from_prior=True`."
            )
        return self
