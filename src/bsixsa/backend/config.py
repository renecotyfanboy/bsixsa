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
        description="Use a population slice sampler instead of region sampling, usually faster with many parameters.",
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
    """Size and training budget of each SIXSA flow (a masked autoregressive flow trained with sbi)."""

    model_config = ConfigDict(**MODEL_CONFIG, allow_inf_nan=False)

    hidden_features: PositiveInt = Field(default=50, description="Width of the hidden layers.")
    num_transforms: PositiveInt = Field(default=5, description="Number of stacked transforms.")
    training_batch_size: PositiveInt = Field(default=1024, description="Training pairs per gradient step.")
    max_num_epochs: PositiveInt = Field(default=25, description="Maximum number of epochs.")
    stop_after_epochs: PositiveInt = Field(
        default=10,
        description="Epochs without validation improvement before stopping (patience).",
    )


class CompressorConfig(BaseModel):
    """Cash-loss autoencoder that compresses each spectrum before the SIXSA flows."""

    model_config = ConfigDict(**MODEL_CONFIG, allow_inf_nan=False)

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
    """SIXSA: history matching, a Cash-loss autoencoder and certified NPE proposals.

    History matching first restricts the prior to the region worth
    sampling. Each round then trains `n_nde` flows and certifies their
    equal-weight mixture by importance sampling on exact folds (Pareto k-hat
    below `khat_threshold`, efficiency at least `min_efficiency`). Until one
    passes, the next training set is drawn inside the proposal's
    highest-density region, intersected with the history-matching region.
    SIXSA fits one spectrum with the C-statistic, without background, and
    finite priors.

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
        default=5,
        description="Flows trained per round from different seeds; their equal-weight mixture is the round's proposal.",
    )
    n_rounds: PositiveInt = Field(
        default=20,
        description="Maximum number of rounds; the run stops once a proposal is certified.",
    )
    check_draws: Integer = Field(
        default=1000,
        ge=100,
        description="Proposal draws (one exact fold each) that certify the proposal of each round.",
    )
    khat_threshold: float = Field(
        default=0.7,
        ge=0, le=1,
        description="Largest Pareto k-hat of the importance weights for certification (0.7 is the usual PSIS limit).",
    )
    min_efficiency: float = Field(
        default=5.0,
        ge=0, le=100,
        description="Smallest importance-sampling efficiency (effective sample size per fold) for certification, in percent.",
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
        description="Size of the first training set; `None` picks it from the number of parameters and channels.",
    )
    train_ramp: float = Field(
        default=1.25,
        ge=1,
        description="Growth factor of the training set per round.",
    )
    train_max: PositiveInt = Field(default=50_000, description="Largest training set.")
    refresh_compressor: StrictBool | NonNegativeInt = Field(
        default=True,
        description="Refit the autoencoder every round (`True`), only in round 1 (`False`), or in rounds 1 to `k`.",
    )
    flow: FlowConfig = Field(default_factory=FlowConfig, description="Size and training budget of the flows.")
    compressor: CompressorConfig = Field(default_factory=CompressorConfig, description="Spectrum autoencoder.")
    target_ess: PositiveFloat = Field(
        default=2000.0,
        description="Effective sample size at which the final importance sample stops.",
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

    @model_validator(mode="after")
    def _validate_budgets(self) -> SIXSA:
        if self.n_training is not None and self.n_training > self.train_max:
            raise ValueError(
                f"n_training ({self.n_training}) must not exceed train_max ({self.train_max})."
            )
        return self

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
