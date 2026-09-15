"""Pydantic-backed configuration models for bsixsa backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

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
    """Backend config models with shared tracing options.

    Attributes:
        restriction_mode: Class-level. How a prior restriction computed by the
            solver (see `Solver(restrict_prior=...)`) reaches the backend:
            `"box"` samples the truncated bounding-box prior, and the evidence
            is corrected by the box's prior mass; `"region"` keeps the full
            prior and consumes the exact region itself (SIXSA).
        restrict_prior_default: Class-level. What `Solver(restrict_prior=None)`
            resolves to for this backend: `True` restricts with a default
            [`HistoryMatching`][bsixsa.restriction.HistoryMatching]. Only SIXSA
            restricts by default.
    """

    model_config = MODEL_CONFIG

    restriction_mode: ClassVar[Literal["box", "region"]] = "box"
    # Resolution of `Solver(restrict_prior=None)` for this backend.
    restrict_prior_default: ClassVar[bool] = False

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


def _ensure_init_not_overspecified(
    x0_physical: np.ndarray | None,
    init_from_prior: bool,
) -> None:
    if x0_physical is not None and init_from_prior:
        raise ValueError(
            "`x0_physical` and `init_from_prior=True` are mutually exclusive."
        )


class _OptimizerInitConfig(BackendConfig):
    """Shared starting-point options for optimiser-style backends.

    Holds the ``x0_physical`` / ``init_from_prior`` fields and their validation,
    deduplicated across the iminuit, Levenberg-Marquardt and emcee configs.
    """

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
    def _validate_initialisation_strategy(self) -> _OptimizerInitConfig:
        _ensure_init_not_overspecified(self.x0_physical, self.init_from_prior)
        return self

    def validate_for_solver(self, solver: "Solver") -> None:
        _validate_x0_length_for_solver(self.x0_physical, solver)


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

    n_networks: PositiveInt = Field(
        default=4,
        description=(
            "Number of neural-network emulators nautilus trains per bound "
            "(nautilus `n_networks`). Default 4 matches the library."
        ),
    )

    n_pool: PositiveInt = Field(
        default=4,
        description=(
            "Worker processes for nautilus's sampler / NN-training pool. "
            "Bounded so it does not add a full node-wide fleet on top of the "
            "solver's likelihood pool. Set to 1 for serial."
        ),
    )

    seed: Annotated[int, Field(strict=True, ge=0)] | None = Field(
        default=None,
        description=(
            "Seed forwarded to the nautilus sampler and used to shuffle the "
            "equal-weight posterior returned by `sample`. `None` draws fresh "
            "OS entropy, so runs are not reproducible."
        ),
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
        description=(
            "Resume a previous UltraNest run if checkpoint files exist; `False` starts over. "
            "A run whose prior transform changed since (a restriction installed, recomputed "
            "with `restrict(force=True)` or another configuration, or removed) cannot resume: "
            "UltraNest refuses it, after any restriction folds were spent, so set `False` or "
            "use a fresh `outputfiles_basename` in that case."
        ),
    )
    use_step_sampler: StrictBool = Field(
        default=False,
        description="Enable UltraNest's population step sampler.",
    )


class Iminuit(_OptimizerInitConfig):
    """Configuration for the [`iminuit`](https://scikit-hep.org/iminuit/) backend."""

    backend_name: ClassVar[str] = "iminuit"

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
    precision: PositiveFloat | None = Field(
        default=None,
        description=(
            "Declared precision of the objective function "
            "(``Minuit.precision``)."
        ),
    )
    engine_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments forwarded to Minuit.migrad().",
    )


class LevenbergMarquardt(_OptimizerInitConfig):
    """Configuration for the [Levenberg-Marquardt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) backend."""

    backend_name: ClassVar[str] = "levenberg_marquardt"

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


class Emcee(_OptimizerInitConfig):
    """Configuration for the [`emcee`](https://emcee.readthedocs.io/en/stable/index.html) backend."""

    backend_name: ClassVar[str] = "emcee"

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

    @model_validator(mode="after")
    def _validate_initialisation_strategy(self) -> Emcee:
        _ensure_init_not_overspecified(self.x0_physical, self.init_from_prior)
        if self.x0_physical is None and not self.init_from_prior:
            raise ValueError(
                "EmceeConfig requires either `x0_physical` or `init_from_prior=True`."
            )
        return self

    def validate_for_solver(self, solver: "Solver") -> None:
        super().validate_for_solver(solver)
        if self.num_walkers < 2 * solver.num_parameters:
            raise ValueError(
                "`num_walkers` must be at least twice the number of free "
                f"parameters ({2 * solver.num_parameters} minimum)."
            )



class NDEConfig(BaseModel):
    """Neural density estimator architecture (sbi posterior_nn)."""

    model_config = MODEL_CONFIG
    model: Literal["zuko_maf", "zuko_nsf", "maf", "nsf"] = "zuko_maf"
    hidden_features: PositiveInt = 50
    num_transforms: PositiveInt = 5
    num_bins: PositiveInt = 5
    z_score_theta: Literal["independent", "structured", "none"] = "independent"
    z_score_x: Literal["independent", "structured", "none"] = "independent"


class TrainingConfig(BaseModel):
    """Per-member NPE training budget; the loss is plain maximum likelihood."""

    model_config = MODEL_CONFIG
    learning_rate: PositiveFloat = 1e-3
    training_batch_size: PositiveInt = 1024
    validation_fraction: Annotated[float, Field(gt=0, lt=1)] = 0.2
    stop_after_epochs: PositiveInt = 10
    max_num_epochs: PositiveInt = 25
    clip_max_norm: PositiveFloat = 1.0


class CompressorConfig(BaseModel):
    """Cash-loss autoencoder, fitted before each round's NPE ensemble."""

    model_config = MODEL_CONFIG
    latent_dim: PositiveInt | None = None
    hidden_dims: list[PositiveInt] | None = None
    n_layers: PositiveInt = 3
    epochs: PositiveInt = 1000
    batch_size: PositiveInt | Literal["auto"] = 512
    lr: PositiveFloat = 1e-3
    patience: PositiveInt = 20
    val_fraction: Annotated[float, Field(gt=0, lt=1)] = 0.2
    norm: Literal["batch", "layer", "none"] = "batch"
    dropout_rate: Annotated[float, Field(ge=0, lt=1)] = 0.0
    warm_start: StrictBool = True
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    torch_threads: PositiveInt = 1


class SIXSA(BackendConfig):
    """SIXSA v2: history matching, autoencoder and certified parallel NPEs.

    The Solver enables history matching by default. Certification and final
    importance sampling always target its full base prior. Only one spectrum,
    C-stat, no background and finite independent priors are supported.
    Old ensemble/buffer/emulator configuration fields are intentionally rejected.
    """

    backend_name: ClassVar[str] = "sixsa"
    restriction_mode: ClassVar[Literal["box", "region"]] = "region"
    restrict_prior_default: ClassVar[bool] = True
    n_nde: PositiveInt = 5
    n_rounds: PositiveInt = 20
    khat_threshold: Annotated[float, Field(ge=0, le=1)] = 0.7
    khat_safe: Annotated[float, Field(gt=0, le=1)] = 0.5
    screen_tolerance: Annotated[float, Field(ge=0)] = 0.05
    min_efficiency: Annotated[float, Field(ge=0, le=100)] = 5.0
    check_draws: Annotated[int, Field(strict=True, ge=100)] = 1000
    screen_members: StrictBool = True
    truncated_quantile: Annotated[float, Field(gt=0, lt=1)] = 1e-3
    truncated_support_draws: Annotated[int, Field(strict=True, ge=100)] = 10_000
    truncated_sampling: Literal["sir", "auto", "ellipsoid", "flow"] = "sir"
    truncated_max_candidates: PositiveInt = 20_000_000
    train_ramp: Annotated[float, Field(ge=1)] = 1.25
    train_max: PositiveInt = 50_000
    n_training: Annotated[int, Field(strict=True, ge=20)] | None = None
    training_set_base: PositiveFloat = 2000.0
    training_set_min_per_param: PositiveInt = 750
    refresh_compressor: StrictBool | Annotated[int, Field(strict=True, ge=0)] = True
    nde: NDEConfig = Field(default_factory=NDEConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    compressor: CompressorConfig = Field(default_factory=CompressorConfig)
    target_ess: PositiveFloat = 2000.0
    num_posterior_samples: PositiveInt = 10_000
    max_sampling_draws: PositiveInt = 200_000
    fold_chunk: PositiveInt = 10_000
    training_n_jobs: PositiveInt | None = None
    threads_per_member: PositiveInt = 1
    seed: Annotated[int, Field(strict=True, ge=0)] | None = None
    plot_diagnostics: StrictBool = True

    @model_validator(mode="after")
    def _validate_budgets(self):
        if self.n_training is not None and self.n_training > self.train_max:
            raise ValueError("n_training must not exceed train_max.")
        return self

    def validate_for_solver(self, solver: "Solver") -> None:
        from ..priors import validate_finite_support

        validate_finite_support(solver.base_prior)
        constraints = solver._xspec_constraints()
        if constraints["n_spectra"] != 1:
            raise ValueError("SIXSA requires exactly one spectrum.")
        if constraints["has_background"]:
            raise ValueError("SIXSA does not support a loaded background.")
        if constraints["stat_method"] != "cstat":
            raise ValueError("SIXSA requires statMethod='cstat'.")


Backend = (
        Nautilus
        | Nessai
        | Ultranest
        | Iminuit
        | LevenbergMarquardt
        | Emcee
        | SIXSA
)
