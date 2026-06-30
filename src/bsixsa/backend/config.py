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


class SIXSA(BackendConfig):
    """Configuration for the SIXSA simulation-based inference (SBI) backend.

    SIXSA fits spectra with neural posterior estimation (NPE) over several
    sequential rounds: each round draws parameters from the current proposal,
    simulates the corresponding spectra, and trains an ensemble of neural
    density estimators to refine the posterior. Importance-sampling
    diagnostics computed each round are used to select the best round and can
    trigger early stopping. See the
    [SBI-with-NPE paper series](https://ui.adsabs.harvard.edu/abs/2025A%26A...699A.179D/abstract)
    for the underlying methodology.
    """

    backend_name: ClassVar[str] = "sixsa"

    num_simulations_per_round: list[PositiveInt] = Field(
        ...,
        min_length=1,
        description=(
            "Number of simulations to run in each sequential inference round; the "
            "list length sets the number of rounds. Round 1 trains on all of them; "
            "later rounds keep the top `1 - is_filter_fraction` by importance "
            "weight. A front-loaded schedule like `[10_000, 1000, 1000, 1000]` "
            "trains a solid base flow, then refines cheaply. Note this is the "
            "*simulated* (pool) count per round: `_sixsa_dev` instead controls the "
            "*kept* count and inflates the pool, so to reproduce a reference kept "
            "count `K` set the list entry to `ceil(K / (1 - is_filter_fraction))`."
        ),
    )
    n_ensemble: PositiveInt = Field(
        default=8,
        description="Number of neural posterior estimators trained per round.",
    )
    first_round_sampling: str = Field(
        default="qmc",
        description=(
            "Round-1 design in the unit-Gaussian latent N(0, I): 'qmc' draws a "
            "Sobol-based quasi-Monte-Carlo design (`MultivariateNormalQMC`) for even "
            "space-filling coverage at a fixed simulation budget; 'prior' draws i.i.d. "
            "standard-normal samples."
        ),
    )
    embedding: Any = Field(
        default=None,
        description=(
            "Embedding network(s) used to compress spectra. Accepts a single "
            "`Embedding`, a list of one `Embedding` per round, or `None` to use "
            "the identity embedding."
        ),
    )
    embedding_net: Any = Field(
        default=None,
        description=(
            "Embedding network trained jointly with the NPE via sbi's `embedding_net` "
            "(an `EmbeddingNet`, a list of one per round, or `None`). `None` uses the "
            "built-in default `FCEmbeddingNet()` (an `FCEmbeddingAPD` geometric pyramid "
            "compressing the spectrum to `output_dim=32`, linear)."
        ),
    )
    training_kwargs: dict[str, Any] | list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Keyword arguments forwarded to the NPE training step, either shared "
            "across rounds (single dict) or per round (list). `None` uses the "
            "backend defaults."
        ),
    )
    nde_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Keyword arguments forwarded to sbi's `posterior_nn` building the "
            "density estimator (the normalizing flow). `None` uses the reference "
            "defaults: a MAF with `hidden_features=100`, `num_transforms=10`, and "
            "independent z-scoring of theta and x."
        ),
    )
    max_num_epochs: PositiveInt = Field(
        default=20,
        description=(
            "Maximum number of training epochs for each neural density estimator. "
            "Forwarded to sbi's `train(max_num_epochs=...)`. An explicit "
            "`max_num_epochs` inside `training_kwargs` takes precedence."
        ),
    )
    plot_diagnostics: StrictBool = Field(
        default=True,
        description=(
            "Write SIXSA diagnostic plots: a per-round coverage band and an "
            "ensemble-posterior corner under each `round_<n>/` directory, plus a "
            "`training_history.pdf` summary at the end of the run."
        ),
    )
    ensemble_corner_samples: PositiveInt = Field(
        default=2_000,
        description=(
            "Samples drawn from each NDE posterior when building the per-round "
            "ensemble-posterior corner plot."
        ),
    )
    device: str = Field(
        default="cpu",
        description="Torch device used for the restricted proposal.",
    )
    proposal_mode: str = Field(
        default="truncated",
        description=(
            "Strategy for the next round's proposal: 'truncated' (the `_sixsa_dev` "
            "default) wraps the ensemble posterior in a density-thresholded "
            "RestrictedPrior over the clean prior; 'posterior' draws directly from "
            "the ensemble posterior."
        ),
    )
    is_filter_fraction: float = Field(
        default=0.5,
        strict=True,
        ge=0.0,
        lt=1.0,
        description=(
            "Fraction of each round's simulations discarded by importance-sampling "
            "filtering before training; the top `1 - is_filter_fraction` by weight "
            "are kept. `0.0` disables filtering."
        ),
    )
    khat_threshold: float | None = Field(
        default=0.5,
        description=(
            "Pareto k-hat threshold for early stopping. A round whose IS "
            "diagnostics fall below this value triggers convergence. `None` "
            "disables the early-stopping check."
        ),
    )
    force_last_round: StrictBool = Field(
        default=True,
        description="Run one extra round after the k-hat threshold is crossed.",
    )
    truncated_quantile: PositiveFloat = Field(
        default=1e-4,
        description="Density quantile used to truncate the restricted proposal.",
    )
    truncated_num_samples_to_estimate_support: PositiveInt = Field(
        default=10_000,
        description="Number of samples used to estimate the restricted-proposal support.",
    )
    truncated_sampling_method: str = Field(
        default="sir",
        description=(
            "Sampling method for the restricted proposal ('sir' or 'rejection'). "
            "SIXSA importance-weights each draw with the ensemble posterior density as "
            "the proposal density (log_q_mix), so the proposal must approximate the "
            "posterior: 'sir' draws from the posterior and rejects out-of-HPR points "
            "(approx. posterior) and is the correct choice. 'rejection' samples the "
            "truncated PRIOR instead, which is inconsistent with the IS weighting and "
            "biases the k-hat/ESS/efficiency diagnostics and the resampled posterior."
        ),
    )
    num_posterior_samples: PositiveInt = Field(
        default=10_000,
        description="Number of posterior samples drawn into the returned results.",
    )


Backend = (
        Nautilus
        | Nessai
        | Ultranest
        | Iminuit
        | LevenbergMarquardt
        | Emcee
        | SIXSA
)
