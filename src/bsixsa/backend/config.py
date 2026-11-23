"""Backend configurations: pass one of them as `Solver(backend=...)` to choose how to fit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from .._config_types import Integer, NonNegativeInt, PositiveFloat, PositiveInt
from ..restriction.config import HistoryMatching

if TYPE_CHECKING:
    from ..solver import Solver

# `extra="forbid"` turns a misspelled option into an error.
MODEL_CONFIG = ConfigDict(
    arbitrary_types_allowed=True,
    extra="forbid",
    validate_default=True,
)


class BackendConfig(BaseModel):
    """Settings shared by every backend: the evaluation trace and its progress plot.

    Pass one of its subclasses ([`Nautilus`][bsixsa.Nautilus],
    [`SIXSA`][bsixsa.SIXSA], ...) as `Solver(backend=...)`.

    Attributes:
        backend_name: Class-level. Key of the backend in
            `bsixsa.backend.BACKEND_REGISTRY`, e.g. `"nautilus"`.
        required_package: Class-level. `(module, pip name)` of the sampler
            package the backend needs, or `None`.
    """

    model_config = MODEL_CONFIG

    backend_name: ClassVar[str]
    required_package: ClassVar[tuple[str, str] | None] = None

    trace: StrictBool = Field(
        default=True,
        description=(
            "Record every likelihood evaluation in `evaluation_trace.npz` and redraw "
            "`progress.png` in the output directory during the run."
        ),
    )
    plot_every: PositiveInt = Field(
        default=50,
        description="Evaluations before the first progress plot.",
    )
    plot_step_percent: PositiveInt = Field(
        default=10,
        le=100,
        description=(
            "Spacing of the later progress plots, in percent of the next power of ten "
            "(with 10: at 60, 70, ..., 100, 200, ... evaluations)."
        ),
    )

    def validate_for_solver(self, solver: Solver) -> None:
        """Check the options that depend on the solver.

        Raises:
            ValueError: If an option does not fit the solver.
        """


class _StartingPointConfig(BackendConfig):
    """Starting point: `x0_physical`, a prior draw, or (optimisers only) the prior median."""

    x0_physical: np.ndarray | None = Field(
        default=None,
        description="Starting point in physical units, one value per free parameter.",
    )
    init_from_prior: StrictBool = Field(
        default=False,
        description="Start from a random draw from the prior (unseeded) instead.",
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
    def _validate_initialisation_strategy(self) -> _StartingPointConfig:
        if self.x0_physical is not None and self.init_from_prior:
            raise ValueError("`x0_physical` and `init_from_prior=True` are mutually exclusive.")
        return self

    def validate_for_solver(self, solver: Solver) -> None:
        """Check that `x0_physical` has one value per free parameter."""
        if self.x0_physical is not None and self.x0_physical.size != solver.num_parameters:
            raise ValueError(
                "`x0_physical` must have one value per free parameter "
                f"({solver.num_parameters} expected, got {self.x0_physical.size})."
            )


class Nautilus(BackendConfig):
    """Nested sampling with [`nautilus`](https://nautilus-sampler.readthedocs.io/en/latest/), boosted by neural networks.

    `FitResults` holds equal-weight posterior samples and `log_Z`. Needs the
    `nautilus-sampler` package.
    """

    backend_name: ClassVar[str] = "nautilus"
    required_package: ClassVar[tuple[str, str] | None] = ("nautilus", "nautilus-sampler")

    num_live_points: PositiveInt = Field(
        default=3_000,
        description="Number of live points used by the nested sampler.",
    )
    n_batch: PositiveInt = Field(
        default=100,
        description="Points proposed and folded together on the solver's pool; keep it above `n_jobs`.",
    )
    n_networks: PositiveInt = Field(
        default=4,
        description="Number of neural networks trained for each bound.",
    )
    n_pool: PositiveInt = Field(
        default=4,
        description=(
            "Processes for nautilus' own work (likelihoods run on the solver's pool); "
            "on macOS and Windows a `.py` script must guard its code with "
            "`if __name__ == \"__main__\":` or set `n_pool=1`, otherwise the run hangs."
        ),
    )
    seed: NonNegativeInt | None = Field(
        default=None,
        description="Random seed of the sampler and of `sample`; `None` is not reproducible.",
    )


class Nessai(BackendConfig):
    """Nested sampling with [`nessai`](https://nessai.readthedocs.io/en/latest/), driven by normalising flows.

    Runs nessai's importance nested sampler. `FitResults` holds equal-weight
    posterior samples and `log_Z`. Needs the `nessai` package.
    """

    backend_name: ClassVar[str] = "nessai"
    required_package: ClassVar[tuple[str, str] | None] = ("nessai", "nessai")
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
            "Extra keyword arguments for nessai's `FlowSampler`, e.g. `{'seed': 1}`, "
            "except `nlive`, `output` and `importance_nested_sampler`."
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
    """Nested sampling with [`ultranest`](https://johannesbuchner.github.io/UltraNest/index.html) (MLFriends regions).

    `FitResults` holds equal-weight posterior samples and `log_Z`. Plots and
    checkpoints go to `<outputfiles_basename>/ultranest_outputs`. Needs the
    `ultranest` package.
    """

    backend_name: ClassVar[str] = "ultranest"
    required_package: ClassVar[tuple[str, str] | None] = ("ultranest", "ultranest")

    num_live_points: PositiveInt = Field(
        default=1_000,
        description="Minimum number of live points; UltraNest may add more.",
    )
    resume: StrictBool = Field(
        default=True,
        description="Resume a previous run from its checkpoints.",
    )
    use_step_sampler: StrictBool = Field(
        default=False,
        description=(
            "Use a population slice sampler instead of region sampling, usually faster with many "
            "parameters. Its directions alternate between a principal axis of the live points and the "
            "difference of two live points, the best proposal in Buchner (2022, arXiv:2211.09426)."
        ),
    )
    nsteps: PositiveInt | None = Field(
        default=None,
        description=(
            "Slice-sampling steps per new live point, with `use_step_sampler`. `None` means 2 x the "
            "number of parameters, the minimum Buchner (2022) finds necessary for these directions."
        ),
    )
    popsize: PositiveInt = Field(
        default=128,
        description=(
            "Walkers advanced together, with `use_step_sampler`. Each likelihood call folds this many "
            "points, so a multiple of the solver's `n_jobs` keeps every worker busy."
        ),
    )


class Iminuit(_StartingPointConfig):
    """Optimiser: MINUIT's MIGRAD algorithm through [`iminuit`](https://scikit-hep.org/iminuit/).

    Finds the maximum a posteriori parameters (`best_fit`, and `best_fit_stat`
    as XSPEC's statistic with `Fit.bayes = "on"`). `posterior_samples` are
    draws from a Gaussian (Laplace) approximation around it, and `log_Z` is NaN.
    Needs the `iminuit` package.
    """

    backend_name: ClassVar[str] = "iminuit"
    required_package: ClassVar[tuple[str, str] | None] = ("iminuit", "iminuit")

    run_hesse: StrictBool = Field(
        default=True,
        description="Run HESSE after MIGRAD to estimate the covariance.",
    )
    strategy: Integer = Field(
        default=1,
        ge=0, le=2,
        description="MIGRAD strategy (0, 1 or 2); higher is slower but more accurate, 2 suits noisy or correlated fits.",
    )
    tol: PositiveFloat | None = Field(
        default=None,
        description="MIGRAD stopping tolerance (EDM below `0.002 * tol`); `None` keeps iminuit's default.",
    )
    precision: PositiveFloat | None = Field(
        default=None,
        description=(
            "Relative precision of the C-stat, e.g. `1e-5` for noisy table models; "
            "`None` keeps iminuit's default."
        ),
    )
    engine_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra keyword arguments for `Minuit.migrad()`.",
    )


class LevenbergMarquardt(_StartingPointConfig):
    """Optimiser: least squares on the Poisson deviance with [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html), like XSPEC's `fit`.

    Finds the maximum a posteriori parameters (`best_fit`, and `best_fit_stat`
    as XSPEC's statistic with `Fit.bayes = "on"`). `posterior_samples` are
    draws from a Gaussian (Laplace) approximation around it, and `log_Z` is NaN.
    """

    backend_name: ClassVar[str] = "levenberg_marquardt"

    reparametrise: StrictBool = Field(
        default=False,
        description=(
            "Search the unbounded logit-of-CDF space with SciPy's `lm` instead of "
            "physical units within the prior bounds with `trf` (like XSPEC's `leven`)."
        ),
    )
    engine_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra keyword arguments for `scipy.optimize.least_squares()`, overriding bsixsa's defaults.",
    )


class Emcee(_StartingPointConfig):
    """Ensemble MCMC with [`emcee`](https://emcee.readthedocs.io/en/stable/index.html).

    `posterior_samples` is the chain after burn-in (correlated draws); `log_Z`
    is NaN. Requires `x0_physical` or `init_from_prior=True`, and the `emcee`
    package.
    """

    backend_name: ClassVar[str] = "emcee"
    required_package: ClassVar[tuple[str, str] | None] = ("emcee", "emcee")

    num_warmup: PositiveInt = Field(
        default=100,
        description="Burn-in steps per walker, discarded.",
    )
    num_samples: PositiveInt = Field(
        default=10_000,
        description="Recorded steps per walker; the posterior holds `num_samples * num_walkers` draws.",
    )
    num_walkers: PositiveInt = Field(
        default=32,
        description="Number of walkers in the ensemble, at least twice the number of free parameters.",
    )
    init_spread: PositiveFloat = Field(
        default=1e-3,
        description="Relative scatter of the walkers around the start, `x0 * (1 + init_spread * N(0, 1))`.",
    )

    @model_validator(mode="after")
    def _require_a_starting_point(self) -> Emcee:
        if self.x0_physical is None and not self.init_from_prior:
            raise ValueError(
                "Emcee requires either `x0_physical` (one value per free parameter) or "
                "`init_from_prior=True` to place its walkers."
            )
        return self

    def validate_for_solver(self, solver: Solver) -> None:
        """Check `x0_physical` and that `num_walkers` is at least twice the number of free parameters."""
        super().validate_for_solver(solver)
        if self.num_walkers < 2 * solver.num_parameters:
            raise ValueError(
                "`num_walkers` must be at least twice the number of free "
                f"parameters ({2 * solver.num_parameters} minimum), got {self.num_walkers}."
            )


class FlowConfig(BaseModel):
    """Architecture and training budget of one SIXSA flow, trained with sbi.

    The defaults are the first member of the reference ensemble
    ([`reference_flow_ensemble`][bsixsa.backend.config.reference_flow_ensemble]).
    """

    model_config = ConfigDict(**MODEL_CONFIG, allow_inf_nan=False)

    model: Literal["zuko_nsf", "zuko_maf"] = Field(
        default="zuko_nsf",
        description="`zuko_nsf` (neural spline flow) or `zuko_maf` (masked autoregressive flow).",
    )
    hidden_features: PositiveInt = Field(default=30, description="Width of the hidden layers.")
    num_transforms: PositiveInt = Field(default=3, description="Number of stacked transforms.")
    num_bins: PositiveInt = Field(
        default=8,
        description="Spline segments per transform; used by `zuko_nsf` only.",
    )
    training_batch_size: PositiveInt = Field(default=1024, description="Training pairs per gradient step.")
    max_num_epochs: PositiveInt = Field(
        default=500,
        description="Maximum number of epochs; early stopping (`stop_after_epochs`) normally ends training first.",
    )
    stop_after_epochs: PositiveInt = Field(
        default=10,
        description="Epochs without validation improvement before stopping (patience).",
    )


def reference_flow_ensemble() -> list[FlowConfig]:
    """The flows of the reference SIXSA implementation: five spline flows and two autoregressive flows.

    A mixture of different architectures has heavier tails than any single
    one, and no architecture was best on every spectrum.
    """
    return [
        FlowConfig(model="zuko_nsf", hidden_features=30, num_transforms=3, num_bins=8),
        FlowConfig(model="zuko_nsf", hidden_features=40, num_transforms=3, num_bins=8),
        FlowConfig(model="zuko_nsf", hidden_features=30, num_transforms=4, num_bins=8),
        FlowConfig(model="zuko_nsf", hidden_features=40, num_transforms=4, num_bins=8),
        FlowConfig(model="zuko_nsf", hidden_features=30, num_transforms=3, num_bins=10),
        FlowConfig(model="zuko_maf", hidden_features=30, num_transforms=3),
        FlowConfig(model="zuko_maf", hidden_features=40, num_transforms=4),
    ]


# Rows `(d_below, value)`: the value applies to every number of free
# parameters d < d_below; the last row, with d_below = None, to the rest.
DimensionSchedule = list[tuple[PositiveInt | None, float]]


def schedule_value(schedule: DimensionSchedule, d: int) -> float:
    """The value of a `(d_below, value)` schedule for `d` free parameters: the first row with `d < d_below`."""
    for d_below, value in schedule:
        if d_below is None or int(d) < int(d_below):
            return float(value)
    raise ValueError(f"the schedule {schedule} does not cover d = {d}; end it with a (None, value) row.")


def _check_schedule(name: str, schedule: DimensionSchedule) -> None:
    """Thresholds strictly increasing and a final `(None, value)` row, so that every `d` gets a value."""
    if not schedule:
        raise ValueError(f"`{name}` must have at least one row.")
    thresholds = [d_below for d_below, _ in schedule]
    if thresholds[-1] is not None or any(d_below is None for d_below in thresholds[:-1]):
        raise ValueError(f"`{name}` must end with one `(None, value)` row, got {schedule}.")
    if any(later <= earlier for earlier, later in zip(thresholds[:-2], thresholds[1:-1])):
        raise ValueError(f"the thresholds of `{name}` must increase strictly, got {thresholds[:-1]}.")


class ScoreCompressorConfig(BaseModel):
    r"""Score compression, SIXSA's default: one code per parameter, from exact derivatives at a fiducial point.

    The code of a spectrum $k$ is $t = u_\star + (F + 12\,I)^{-1} \nabla_u \log L(k \mid u_\star)$,
    with $F$ the Fisher matrix at the fiducial $u_\star$ (Alsing & Wandelt 2018) and 12
    the precision of the uniform prior in the cube, which keeps poorly constrained
    directions at the prior's scale; the derivatives cost a few folds per
    parameter, and nothing is trained.
    """

    model_config = ConfigDict(**MODEL_CONFIG, allow_inf_nan=False)

    fiducial: Literal["fit", "best_fold"] = Field(
        default="fit",
        description=(
            "Where the derivatives are taken in round 1: `fit` moves the best fold of history matching "
            "and of the first training set to the maximum likelihood by Fisher scoring; `best_fold` takes that "
            "fold as it is."
        ),
    )
    recentre: StrictBool = Field(
        default=False,
        description=(
            "Rebuild the map at the start of a later round whenever a fold has beaten the fiducial by more than "
            "0.1 in C-stat, from that fold (fitted again with `fiducial='fit'`). For fits that stop in a local "
            "optimum, such as a narrow line the round-1 fit missed."
        ),
    )


class AutoencoderConfig(BaseModel):
    """Cash-loss autoencoder that compresses each spectrum before the SIXSA flows."""

    model_config = ConfigDict(**MODEL_CONFIG, allow_inf_nan=False)

    refresh: StrictBool | NonNegativeInt = Field(
        default=True,
        description="Refit the autoencoder every round (`True`), only in round 1 (`False`), or in rounds 1 to `k`.",
    )
    latent_dim: PositiveInt | None = Field(
        default=None,
        description="Size of the compressed spectrum; `None` picks it from the number of parameters.",
    )
    hidden_dims: list[PositiveInt] | None = Field(
        default=None,
        description="Encoder hidden widths (mirrored in the decoder); `None` picks them from the number of channels.",
    )
    n_layers: PositiveInt = Field(default=3, description="Hidden layers when `hidden_dims` is `None`.")
    epochs: PositiveInt = Field(default=1000, description="Maximum number of epochs per fit.")
    batch_size: PositiveInt = Field(default=512, description="Spectra per gradient step.")
    learning_rate: PositiveFloat = Field(default=1e-3, description="Adam learning rate.")
    patience: PositiveInt = Field(default=20, description="Epochs without validation improvement before stopping.")
    validation_fraction: float = Field(
        default=0.2,
        gt=0, lt=1,
        description="Fraction of the spectra held out for early stopping.",
    )
    warm_start: StrictBool = Field(default=True, description="Start each refit from the previous weights.")
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Device of the autoencoder; `auto` picks cuda, then mps, then cpu.",
    )


class SIXSA(BackendConfig):
    """SIXSA: history matching, score compression and certified NPE proposals.

    History matching first restricts the prior to the region worth
    sampling. Each spectrum is then compressed to one code per parameter,
    by score compression at the maximum likelihood of the observation
    (`compressor`; an autoencoder is the alternative). Each round trains
    `n_nde` flows on the codes and certifies their
    equal-weight mixture by importance sampling on exact folds (Pareto k-hat
    below `khat_threshold`, efficiency at least `min_efficiency`). Until one
    passes, the next training set is drawn inside the proposal's
    highest-density region, intersected with the history-matching region.
    SIXSA fits one spectrum with the C-statistic, without background, and
    finite priors.

    The defaults follow the reference SIXSA implementation: seven flows of
    two kinds, an efficiency gate and a final effective sample size that
    decrease with the number of free parameters `d`, and a first training
    set that grows as `d^1.6`. The score compression is bsixsa's own
    variant (`ScoreCompressorConfig`).

    `FitResults` holds equal-weight posterior samples, `log_Z`, and the best
    fold as `best_fit`.
    """

    model_config = ConfigDict(**MODEL_CONFIG, allow_inf_nan=False)

    backend_name: ClassVar[str] = "sixsa"

    history_matching: HistoryMatching | None = Field(
        default_factory=HistoryMatching,
        description=(
            "History matching run before the first round; the training sets are drawn in its "
            "region. `None` skips it."
        ),
    )
    n_nde: PositiveInt = Field(
        default=7,
        description=(
            "Flows trained per round from different seeds; their equal-weight mixture is the round's "
            "proposal. Flow `k` uses `flow[k % len(flow)]`."
        ),
    )
    n_rounds: PositiveInt = Field(
        default=20,
        description="Maximum number of rounds; the run stops once a proposal is certified.",
    )
    check_draws: Integer = Field(
        default=2000,
        ge=100,
        description="Proposal draws (one exact fold each) that certify the proposal of each round.",
    )
    khat_threshold: float = Field(
        default=0.7,
        ge=0, le=1,
        description="Largest Pareto k-hat of the importance weights for certification (0.7 is the usual PSIS limit).",
    )
    min_efficiency: Annotated[float, Field(ge=0, le=100)] | None = Field(
        default=None,
        description=(
            "Smallest importance-sampling efficiency (effective sample size per fold) for certification, in "
            "percent; `None` takes it from `efficiency_schedule` for the number of free parameters."
        ),
    )
    efficiency_schedule: DimensionSchedule = Field(
        default=[(5, 10.0), (10, 5.0), (15, 2.5), (None, 1.0)],
        description=(
            "`min_efficiency` by number of free parameters `d` when it is `None`: rows `(d_below, percent)`, "
            "the first row with `d < d_below` applies and the last row, `(None, percent)`, covers the rest. "
            "The efficiency of importance sampling falls with `d`."
        ),
    )
    truncated_quantile: float = Field(
        default=1e-3,
        gt=0, lt=1,
        description="The next training set is drawn in the proposal's `1 - truncated_quantile` highest-density region.",
    )
    truncated_support_draws: Integer = Field(
        default=10_000,
        ge=100,
        description="Proposal draws used to place that density threshold.",
    )
    truncated_oversampling: PositiveInt = Field(
        default=32,
        description="Flow candidates per training draw when sampling that region; more gives more uniform draws.",
    )
    n_training: Annotated[Integer, Field(ge=20)] | None = Field(
        default=None,
        description="Size of the first training set; `None` picks it from the number of free parameters.",
    )
    train_ramp: float = Field(
        default=1.1,
        ge=1,
        description="Growth factor of the training set per round.",
    )
    train_max: PositiveInt = Field(default=100_000, description="Largest training set.")
    flow: Annotated[list[FlowConfig], Field(min_length=1)] = Field(
        default_factory=reference_flow_ensemble,
        description=(
            "Architectures and training budgets of the flows: flow `k` of a round uses `flow[k % len(flow)]`. "
            "One `FlowConfig` gives every flow the same settings. The default is the reference ensemble."
        ),
    )
    flow_torch_threads: PositiveInt | None = Field(
        default=None,
        description=(
            "Torch threads while a flow trains; `None` takes one per 10 free parameters (at least one), "
            "within the solver's `n_jobs` and the available CPUs."
        ),
    )
    compressor: ScoreCompressorConfig | AutoencoderConfig = Field(
        default_factory=ScoreCompressorConfig,
        description="How each spectrum is compressed before the flows: score compression or an autoencoder.",
    )
    target_ess: PositiveFloat | None = Field(
        default=None,
        description=(
            "Effective sample size at which the final importance sample stops; `None` takes it from "
            "`ess_schedule` for the number of free parameters."
        ),
    )
    ess_schedule: DimensionSchedule = Field(
        default=[(5, 5000.0), (10, 3000.0), (15, 2000.0), (None, 1000.0)],
        description=(
            "`target_ess` by number of free parameters `d` when it is `None`, as rows `(d_below, ESS)` "
            "read like `efficiency_schedule`."
        ),
    )
    max_sampling_draws: PositiveInt = Field(
        default=200_000,
        description="Largest number of draws of the final importance sample.",
    )
    num_posterior_samples: PositiveInt = Field(
        default=10_000,
        description="Equal-weight rows in `FitResults.posterior_samples`.",
    )
    seed: NonNegativeInt | None = Field(
        default=None,
        description="Seed of every random draw of the run, except history matching's; `None` is not reproducible.",
    )

    @field_validator("flow", mode="before")
    @classmethod
    def _one_flow_for_every_member(cls, value: Any) -> Any:
        """A single `FlowConfig` (or dict) stands for a one-entry list."""
        if isinstance(value, (FlowConfig, dict)):
            return [value]
        return value

    @model_validator(mode="after")
    def _validate_budgets(self) -> SIXSA:
        if self.n_training is not None and self.n_training > self.train_max:
            raise ValueError(
                f"n_training ({self.n_training}) must not exceed train_max ({self.train_max})."
            )
        _check_schedule("efficiency_schedule", self.efficiency_schedule)
        _check_schedule("ess_schedule", self.ess_schedule)
        if not all(0.0 <= value <= 100.0 for _, value in self.efficiency_schedule):
            raise ValueError(f"`efficiency_schedule` values are percentages in [0, 100], got {self.efficiency_schedule}.")
        if not all(value > 0.0 for _, value in self.ess_schedule):
            raise ValueError(f"`ess_schedule` values must be positive, got {self.ess_schedule}.")
        return self

    def flow_members(self) -> list[FlowConfig]:
        """The settings of the `n_nde` flows of a round, cycling through `flow`."""
        return [self.flow[k % len(self.flow)] for k in range(self.n_nde)]

    def gate_efficiency(self, d: int) -> float:
        """`min_efficiency`, or its value in `efficiency_schedule` for `d` free parameters."""
        return float(self.min_efficiency) if self.min_efficiency is not None else schedule_value(
            self.efficiency_schedule, d)

    def final_ess(self, d: int) -> float:
        """`target_ess`, or its value in `ess_schedule` for `d` free parameters."""
        return float(self.target_ess) if self.target_ess is not None else schedule_value(self.ess_schedule, d)

    def validate_for_solver(self, solver: Solver) -> None:
        """Check that the prior is finite and XSPEC holds one spectrum, no background, and uses cstat.

        Raises:
            ValueError: If SIXSA cannot fit this solver.
        """
        from ..priors import validate_finite_support

        validate_finite_support(solver.prior)
        constraints = solver._xspec_constraints()
        n_spectra = constraints["n_spectra"]
        if n_spectra is None:
            raise ValueError(
                "SIXSA fits exactly one spectrum, but the number of loaded spectra "
                "could not be read from XSPEC."
            )
        if n_spectra != 1:
            raise ValueError(
                f"SIXSA fits exactly one spectrum, but {n_spectra} are loaded; use another "
                "backend (e.g. Nautilus) for joint fits."
            )
        if constraints["has_background"]:
            raise ValueError(
                "SIXSA does not model a background, and with one loaded XSPEC's cstat "
                "becomes W-stat: remove it (xspec.AllData(1).background = None) and model "
                "the background instead, or use another backend."
            )
        stat_method = constraints["stat_method"]
        if stat_method is None:
            raise ValueError(
                "SIXSA needs the C-statistic, but xspec.Fit.statMethod could not be read; "
                "set xspec.Fit.statMethod = 'cstat'."
            )
        if stat_method != "cstat":
            raise ValueError(
                f"SIXSA needs the C-statistic, but xspec.Fit.statMethod is {stat_method!r}: "
                "set xspec.Fit.statMethod = 'cstat'."
            )
