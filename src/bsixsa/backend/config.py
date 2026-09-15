from __future__ import annotations

"""Pydantic-backed configuration models for bsixsa backends."""

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


class SIXSA(BackendConfig):
    """Configuration for the SIXSA simulation-based inference (SBI) backend.

    SIXSA fits spectra with neural posterior estimation over sequential rounds.
    It maintains a persistent training set ``T`` refined by a monotone
    replace-worst nested move, trains an ensemble of NPE_C flows with the plain
    maximum-likelihood loss, and gates convergence with defensive-mixture
    importance sampling (``q_def = alpha*prior + (1-alpha)*q_flow``). The
    headline posterior / khat / ESS / log_Z come from a banked pooled
    multiple-IS estimator, followed by an optional emulator pass. Everything
    runs in the unit-Gaussian latent ``N(0, I)``; log-uniform parameters are a
    per-parameter `loguniform` prior-object choice, mapped to physical space by
    ``solver.prior.from_unit_gaussian``. See the
    [SBI-with-NPE paper series](https://ui.adsabs.harvard.edu/abs/2025A%26A...699A.179D/abstract)
    for the underlying methodology.

    Implements the SIXSA specification (code version 1.11.1). Budgets scale
    with the number of free parameters ``d``.
    """

    backend_name: ClassVar[str] = "sixsa"

    # ----- budgets (per free parameter, scaled by d at run time) -----
    n_initial_per_param: PositiveInt = Field(
        default=5_000,
        description=(
            "Round-1 training-set size is `n_initial_per_param * d` LHS/QMC draws "
            "from the prior box (simulated exactly). Spec production default: 5000."
        ),
    )
    n_picked_per_param: PositiveInt = Field(
        default=2_500,
        description=(
            "Per-round batch base is `n_batch or n_picked_per_param * d`. Spec "
            "production default: 2500."
        ),
    )
    n_batch: PositiveInt | None = Field(
        default=None,
        description="Absolute override for the per-round batch base `n_batch_base`.",
    )
    n_sub_batch: PositiveInt | None = Field(
        default=None,
        description=(
            "Per-NDE sub-batch size. `None` -> `ceil(n_batch_base / n_ensemble)`. "
            "Each ensemble member draws `n_sub * ramp(r)` flat samples from its "
            "truncated proposal and replace-worst-merges them into `T`."
        ),
    )
    n_ensemble: PositiveInt = Field(
        default=10,
        description=(
            "Number of NPE_C members trained per round (spec production: 10; "
            "5 in experiments)."
        ),
    )
    n_rounds: PositiveInt = Field(
        default=10,
        description="Maximum number of sequential inference rounds.",
    )
    n_simulate_ramp: float = Field(
        default=1.0,
        ge=1.0,
        description=(
            "Geometric ramp base on the per-round sub-batch size: "
            "`ramp(r) = min(n_simulate_ramp**(r-1), n_simulate_ramp_max)`. "
            "1.0 keeps the sub-batch constant across rounds."
        ),
    )
    n_simulate_ramp_max: PositiveFloat | None = Field(
        default=4.0,
        description="Cap on `ramp(r)`. `None` leaves the ramp uncapped.",
    )

    # ----- end-of-round training-set update -----
    merge_mode: Literal["merge", "fresh"] = Field(
        default="merge",
        description=(
            "Between-round update of `T`. 'merge' (default): draw "
            "`n_sub * ramp(r+1)` flat samples from the ensemble truncated "
            "proposal and replace-worst-merge them into `T` (size held). 'fresh': "
            "replace `T` wholesale with `n_batch_base * ramp(r+1)` fresh draws."
        ),
    )
    n_train_keep: PositiveInt | None = Field(
        default=None,
        description=(
            "Held size of the training buffer `T`. `None` -> `n_initial_per_param "
            "* d` (the round-1 size)."
        ),
    )
    last_member_sub_batch: StrictBool = Field(
        default=False,
        description=(
            "If True, the last ensemble member also performs a sub-batch "
            "replace-worst move (by default only members `1..n_ensemble-1` do)."
        ),
    )

    # ----- truncated proposal (region construction + flat sampling) -----
    first_round_sampling: Literal["qmc", "lhs", "prior"] = Field(
        default="qmc",
        description=(
            "Round-1 design in the unit-Gaussian latent N(0, I): 'qmc' draws a "
            "Sobol-based quasi-Monte-Carlo design for even coverage; 'lhs' draws a "
            "Latin-hypercube design pushed through the normal quantile function "
            "(matching `_sixsa_dev`'s box LHS); 'prior' draws i.i.d. N(0, I)."
        ),
    )
    truncated_quantile: PositiveFloat = Field(
        default=1e-3,
        description=(
            "Density quantile of the flow over its own draws used to threshold "
            "the highest-density region for the restricted proposal (spec: 1e-3)."
        ),
    )
    truncated_num_samples_to_estimate_support: PositiveInt = Field(
        default=10_000,
        description="Number of samples used to estimate the restricted-proposal support.",
    )
    truncated_oversampling_factor: PositiveInt = Field(
        default=2,
        description=(
            "SIR candidates per accepted sample for the restricted proposal "
            "(spec value: 2). CAVEAT -- currently INERT: `RestrictedPrior.sample` "
            "accepts `oversampling_factor` but forwards it into "
            "`sampling_importance_resampling`, which never reads it (it is "
            "absorbed by `**kwargs`); the effective candidate count is sbi's "
            "hardcoded `num_candidate_samples=32`. Verified in sbi 0.25.0 and "
            "0.26.1. Since 32 > 2, SIR is *more* thorough than configured (only "
            "slower, not less correct), so the value is kept for spec parity. "
            "Passing `num_candidate_samples` instead is not possible: "
            "`RestrictedPrior.sample` has no such parameter and no `**kwargs`."
        ),
    )
    truncated_restrict_from_prior: StrictBool = Field(
        default=True,
        description=(
            "Restrict the CLEAN prior each round (no stacking): an early "
            "over-restriction never persists. False restricts the previous "
            "proposal (legacy stacking)."
        ),
    )
    truncated_max_sampling_batch_size: PositiveInt = Field(
        default=1_000,
        description=(
            "Batch size for SIR sampling, forwarded to "
            "`RestrictedPrior.sample(max_sampling_batch_size=...)`. Purely a "
            "performance/memory knob; does not change the sampled distribution."
        ),
    )

    # ----- convergence gates / checks (defensive IS) -----
    khat_from_round: PositiveInt = Field(
        default=3,
        description=(
            "Per-member CHECKs and the ROUND GATE only run from this round on "
            "(earlier rounds are too under-trained to gate on)."
        ),
    )
    khat_threshold: float | None = Field(
        default=0.7,
        description=(
            "PSIS k-hat PASS threshold for checks/gates (the standard reliability "
            "limit). `None` disables the k-hat gate."
        ),
    )
    early_member_n: PositiveInt = Field(
        default=2_500,
        description="Defensive-IS draws for a per-member CHECK.",
    )
    early_member_min_eff: float = Field(
        default=1.0,
        ge=0.0,
        description="Minimum IS efficiency (percent) for a CHECK/GATE to PASS.",
    )
    gate_n: PositiveInt = Field(
        default=10_000,
        description="Defensive-IS draws for the FINAL GATE (full ensemble).",
    )
    round_gate_n: PositiveInt | None = Field(
        default=None,
        description="Defensive-IS draws for the per-round ROUND GATE. `None` -> `gate_n`.",
    )
    gate_alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description=(
            "Defensive prior-floor weight alpha in `q_def = alpha*prior + "
            "(1-alpha)*q_flow`. Bounds the IS weights by `L/alpha` -> finite "
            "variance by construction."
        ),
    )
    confirm_exit: StrictBool = Field(
        default=True,
        description=(
            "On the first passing CHECK (and if not the last member), train the "
            "next member and run its check purely as confirmation (recorded, not "
            "deciding), then exit."
        ),
    )
    member_selection: Literal["report", "apply"] = Field(
        default="apply",
        description=(
            "GATE behaviour. 'apply' promotes the best-scoring self-consistent "
            "banked variant as the returned weighting; 'report' only logs. "
            "Per-member CHECKs always use 'report'."
        ),
    )
    ess_mode: Literal["off", "floor"] = Field(
        default="off",
        description=(
            "If 'floor' (and `ess_target` set), a PASS additionally requires the "
            "pooled ESS to reach `ess_target`."
        ),
    )
    ess_target: PositiveFloat | None = Field(
        default=None,
        description="Pooled-ESS floor used when `ess_mode='floor'`.",
    )

    # ----- networks / training -----
    embedding: Any = Field(
        default=None,
        description=(
            "Diagnostic-only numpy pre-processor embedding, or `None` for the "
            "identity embedding. Its output never feeds the NPE."
        ),
    )
    embedding_net: Any = Field(
        default=None,
        description=(
            "Embedding network trained jointly with the NPE via sbi's "
            "`embedding_net`. `None` uses the default `FCEmbeddingNet()` "
            "(an `FCEmbeddingAPD` linear pyramid compressing to `output_dim=32`)."
        ),
    )
    training_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Keyword arguments forwarded to NPE_C `train`. `None` uses the spec "
            "defaults (plain MLE loss via `force_first_round_loss=True`, "
            "`use_combined_loss=False`, fresh weights per member, batch 512, "
            "lr 1e-3, val fraction 0.2, patience 10, clip 1.0)."
        ),
    )
    nde_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Keyword arguments forwarded to sbi's `posterior_nn`. User keys are "
            "MERGED over the reference defaults (MAF, `hidden_features=100`, "
            "`num_transforms=10`, independent z-scoring of theta and x)."
        ),
    )
    max_num_epochs: PositiveInt = Field(
        default=100,
        description=(
            "Maximum training epochs per NDE (spec production: 100). An explicit "
            "`max_num_epochs` inside `training_kwargs` takes precedence."
        ),
    )
    device: str = Field(
        default="cpu",
        description="Torch device used for the flows and the restricted proposal.",
    )

    # ----- emulator pass (spec section 10) -----
    run_emulator_pass: StrictBool = Field(
        default=True,
        description=(
            "After the loop, draw true surrogate samples from the best ensemble, "
            "simulate them exactly, and run a plain flow-IS pass (no defensive "
            "floor). Returns weighted samples and extends the (theta, cstat) pool."
        ),
    )
    emulator_min_per_param: PositiveInt = Field(
        default=5_000,
        description="Surrogate draws for the emulator pass = `emulator_min_per_param * d`.",
    )
    train_likelihood_emulator: StrictBool = Field(
        default=False,
        description=(
            "Train a PyTorch MLP likelihood emulator (`theta_log -> -0.5*cstat`) "
            "on the emulator-pass samples only. Uses a pure-torch training loop "
            "(no tensorflow); `pytorch-lightning` is used only if installed and "
            "requested via `emulator_kwargs['use_lightning']`."
        ),
    )
    emulator_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Likelihood-emulator hyperparameters: `hidden_units`, "
            "`theta_scaler_type`, `cstat_scaler_type`, `percentage_to_remove`, "
            "`num_epochs`, `learning_rate`, `patience`, `batch_size`, "
            "`use_lightning`."
        ),
    )

    # ----- diagnostics / output -----
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
    num_posterior_samples: PositiveInt = Field(
        default=10_000,
        description="Number of posterior samples drawn into the returned results.",
    )
    posterior_source: Literal["pool", "best_flow", "emulator_pass"] = Field(
        default="pool",
        description=(
            "Where the returned posterior samples come from -- a trade-off "
            "between statistical exactness and sample diversity. "
            "'pool' (default): multinomial resample of the banked pooled "
            "multiple-IS draws. The most exact (exact-likelihood IS with the "
            "defensive prior floor), but the number of DISTINCT rows is capped "
            "by the banked draws and their pooled ESS, so a large "
            "`num_posterior_samples` repeats rows. "
            "'emulator_pass': resample the emulator pass's flow-IS weighted "
            "draws -- fresh surrogate samples from the best flow, still "
            "importance-corrected, usually far more distinct rows (requires "
            "`run_emulator_pass=True`). "
            "'best_flow': draw FRESH from the best flow -- maximum diversity "
            "and no duplicates, but NOT importance-corrected, i.e. the flow's "
            "own approximation of the posterior. "
            "Each choice falls back to the next available source, so a run "
            "always returns a posterior. The reported `log_Z` always comes "
            "from the banked pool regardless of this setting."
        ),
    )

    @model_validator(mode="after")
    def _validate_gates(self):
        if self.ess_mode == "floor" and self.ess_target is None:
            raise ValueError(
                "`ess_target` must be set when `ess_mode='floor'`."
            )
        if (
            self.n_simulate_ramp_max is not None
            and self.n_simulate_ramp_max < 1.0
        ):
            raise ValueError("`n_simulate_ramp_max` must be >= 1.0.")
        return self


Backend = (
        Nautilus
        | Nessai
        | Ultranest
        | Iminuit
        | LevenbergMarquardt
        | Emcee
        | SIXSA
)
