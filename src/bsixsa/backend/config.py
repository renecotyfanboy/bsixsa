from __future__ import annotations

"""Pydantic-backed configuration models for bsixsa backends."""

from typing import TYPE_CHECKING, Annotated, Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

if TYPE_CHECKING:
    from ..solver import Solver

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
PositiveFloat = Annotated[float, Field(strict=True, gt=0)]
PlotStepPercent = Annotated[int, Field(strict=True, gt=0, le=100)]
MODEL_CONFIG = ConfigDict(
    arbitrary_types_allowed=True,
    extra="forbid",
    validate_default=True,
)


class BackendConfig(BaseModel):
    """Backend config models with shared tracing options."""

    model_config = MODEL_CONFIG

    trace: StrictBool = Field(
        default=True,
        description="Whether to actively trace the run or not.",
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

    def validate_for_solver(self, solver: "Solver") -> None:
        """Validate solver-dependent config constraints."""

    def create_tracer(
        self,
        *,
        output_dir: str,
        parameter_names: list[str],
    ):
        from .tracer import DummyTracer, EvaluationTracer

        tracer_cls = EvaluationTracer if self.trace else DummyTracer
        return tracer_cls(
            output_dir=output_dir,
            parameter_names=parameter_names,
            plot_every=self.plot_every,
            plot_step_percent=self.plot_step_percent,
        )


def _validate_x0_length_for_solver(
    x0_physical: np.ndarray | None,
    solver: "Solver",
) -> None:
    if x0_physical is not None and x0_physical.size != solver.num_parameters:
        raise ValueError(
            "`x0_physical` must have one value per free parameter "
            f"({solver.num_parameters} expected, got {x0_physical.size})."
        )


class Nautilus(BackendConfig):
    """Configuration for the [`nautilus`](https://nautilus-sampler.readthedocs.io/en/latest/) backend."""

    backend_name: ClassVar[str] = "nautilus"

    num_live_points: PositiveInt = Field(
        default=3_000,
        description="Number of live points used by the nested sampler.",
    )

    n_batch: PositiveInt = Field(
        default=100,
        description="Minimal number of likelihood evaluation performed at once.",
    )


class Nessai(BackendConfig):
    """Configuration for the [`nessai`](https://nessai.readthedocs.io/en/latest/) backend."""

    backend_name: ClassVar[str] = "nessai"
    _protected_engine_keys: ClassVar[set[str]] = {
        "nlive",
        "output",
        "importance_nested_sampler",
    }

    num_live_points: PositiveInt = Field(
        default=3_000,
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
    def _validate_engine_key_conflicts(self) -> Nessai:
        conflicts = sorted(self._protected_engine_keys.intersection(self.engine_kwargs))
        if conflicts:
            raise ValueError(
                "`engine_kwargs` cannot override fixed backend options: "
                f"{', '.join(conflicts)}."
            )
        return self


class Ultranest(BackendConfig):
    """Configuration for the [`ultranest`](https://johannesbuchner.github.io/UltraNest/index.html) backend."""

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


class Iminuit(BackendConfig):
    """Configuration for the [`iminuit`](https://scikit-hep.org/iminuit/) backend."""

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
    strategy: Annotated[int, Field(strict=True, ge=0, le=2)] = Field(
        default=1,
        description=(
            "MIGRAD strategy level (0, 1, or 2). Higher values trade more "
            "function evaluations for more accurate gradient and Hessian "
            "estimates. ``2`` is recommended for noisy or strongly "
            "correlated likelihood surfaces."
        ),
    )
    tol: PositiveFloat | None = Field(
        default=None,
        description=(
            "MIGRAD EDM stopping tolerance. The actual stop threshold is "
            "``0.002 * tol * errordef`` (so iminuit's default ``tol=0.1`` "
            "with ``errordef=1`` gives an EDM threshold of ``2e-4``). Set "
            "to ``None`` to keep iminuit's default."
        ),
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
    def _validate_initialisation_strategy(self) -> Iminuit:
        if self.x0_physical is not None and self.init_from_prior:
            raise ValueError(
                "`x0_physical` and `init_from_prior=True` are mutually exclusive."
            )
        return self

    def validate_for_solver(self, solver: "Solver") -> None:
        _validate_x0_length_for_solver(self.x0_physical, solver)


class LevenbergMarquardt(BackendConfig):
    """Configuration for the [Levenberg-Marquardt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) backend."""

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
    reparametrise: StrictBool = Field(
        default=False,
        description=(
            "If True, search in unconstrained logit-of-CDF space. If False "
            "(default), search in physical space with explicit prior bounds "
            "via scipy's `trf` method — matches XSPEC's `leven`."
        ),
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
    def _validate_initialisation_strategy(self) -> LevenbergMarquardt:
        if self.x0_physical is not None and self.init_from_prior:
            raise ValueError(
                "`x0_physical` and `init_from_prior=True` are mutually exclusive."
            )
        return self

    def validate_for_solver(self, solver: "Solver") -> None:
        _validate_x0_length_for_solver(self.x0_physical, solver)


class Emcee(BackendConfig):
    """Configuration for the [`emcee`](https://emcee.readthedocs.io/en/stable/index.html) backend."""

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
    def _validate_initialisation_strategy(self) -> Emcee:
        if self.x0_physical is not None and self.init_from_prior:
            raise ValueError(
                "`x0_physical` and `init_from_prior=True` are mutually exclusive."
            )
        if self.x0_physical is None and not self.init_from_prior:
            raise ValueError(
                "EmceeConfig requires either `x0_physical` or `init_from_prior=True`."
            )
        return self

    def validate_for_solver(self, solver: "Solver") -> None:
        _validate_x0_length_for_solver(self.x0_physical, solver)
        if self.num_walkers < 2 * solver.num_parameters:
            raise ValueError(
                "`num_walkers` must be at least twice the number of free "
                f"parameters ({2 * solver.num_parameters} minimum)."
            )


Backend = (
        Nautilus
        | Nessai
        | Ultranest
        | Iminuit
        | LevenbergMarquardt
        | Emcee
)
