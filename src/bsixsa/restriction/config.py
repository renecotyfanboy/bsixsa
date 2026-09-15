"""Configuration of the history-matching prior restriction.

The model only depends on pydantic, so it can be imported (and validated) in
processes that never load torch.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictBool

__all__ = ["HistoryMatching"]

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveFloat = Annotated[float, Field(strict=True, gt=0)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0)]
OpenUnitFloat = Annotated[float, Field(strict=True, gt=0, lt=1)]

MODEL_CONFIG = ConfigDict(
    arbitrary_types_allowed=True,
    extra="forbid",
    validate_default=True,
)


class HistoryMatching(BaseModel):
    r"""Settings of the history-matching restriction of the prior.

    Waves of exact folds train a regressor $u \mapsto C(u)$ on the unit cube
    $u = F(\theta)$ of the prior. The region $\{u : \hat C(u) \le t\}$ is kept
    and the next wave is drawn uniformly inside it, until the region stops
    shrinking, becomes too small to sample, or the wave budget is spent.

    The threshold of a wave is
    $t = \max(\hat C_{best} + s\,\Delta,\; \hat C_{best} + \Delta + k\,\sigma_C,\; t_{cost})$,
    where $\Delta = \chi^2_d(0.999)$ is the posterior span, $s$ is
    ``target_spans``, $\sigma_C$ is the regressor's C-stat error near the
    boundary, $k$ is ``min_margin_sigma`` and $t_{cost}$ is the smallest
    region that ``max_candidates`` uniform candidates can still fill. It never
    increases from one wave to the next, except through the recall check,
    which only widens the region.
    """

    model_config = MODEL_CONFIG

    # ----- waves -----
    n_waves: PositiveInt = Field(
        default=5,
        description="Maximum number of history-matching waves.",
    )
    simulations_per_parameter: PositiveInt = Field(
        default=500,
        description="Exact folds per wave, per free parameter (a wave folds `simulations_per_parameter * d`).",
    )

    # ----- threshold -----
    target_spans: PositiveFloat = Field(
        default=3.0,
        description=(
            "Requested region width, in posterior spans `chi2_d(0.999)` above the best C-stat."
        ),
    )
    min_margin_sigma: NonNegativeFloat = Field(
        default=3.0,
        description=(
            "Safety floor: the threshold stays at least this many regressor C-stat RMSEs "
            "above `best + span`."
        ),
    )

    # ----- stopping -----
    stop_when_acceptance: OpenUnitFloat = Field(
        default=5e-3,
        description="Stop once the region's prior mass falls below this fraction of the cube.",
    )
    stop_when_threshold: NonNegativeFloat = Field(
        default=0.01,
        description=(
            "Stop once the threshold falls by less than this relative amount between two waves "
            "(ignored on a wave widened by the recall check). The gain is relative to the "
            "absolute C-stat, as in SIXSA v2, so spectra with a large best C-stat (many bins) "
            "stop sooner, with a wider region; lower it for those."
        ),
    )

    # ----- regressor -----
    hidden: PositiveInt = Field(
        default=128,
        description="Width of the first hidden layer; later layers halve it (never below 8).",
    )
    blocks: PositiveInt = Field(
        default=3,
        description="Number of hidden Linear+GELU layers of the regressor.",
    )
    epochs: PositiveInt = Field(
        default=1000,
        description="Maximum number of training epochs per wave.",
    )
    patience: PositiveInt = Field(
        default=20,
        description="Early-stopping patience, in epochs without a validation improvement.",
    )
    learning_rate: PositiveFloat = Field(
        default=1e-3,
        description="Adam learning rate.",
    )
    batch_size: PositiveInt = Field(
        default=256,
        description="Mini-batch size.",
    )
    validation_fraction: OpenUnitFloat = Field(
        default=0.2,
        description="Fraction of the accumulated folds held out for early stopping and the RMSE.",
    )
    focus_weight: PositiveFloat = Field(
        default=10.0,
        description="Loss weight of the folds close to the previous threshold.",
    )
    focus_factor: PositiveFloat = Field(
        default=3.0,
        description=(
            "A fold is close to the boundary when its C-stat is at most "
            "`focus_factor * previous threshold`."
        ),
    )

    # ----- region sampling -----
    mass_probe: PositiveInt = Field(
        default=50_000,
        description=(
            "Uniform cube points used to place the cost floor and, grown fourfold until 100 hits, "
            "to measure the region's prior mass. The growth stops at `50 * mass_probe` points, or "
            "at the `400 / stop_when_acceptance` points that resolve the acceptance stop if more, "
            "and never beyond `max_candidates`."
        ),
    )
    box_samples: NonNegativeInt = Field(
        default=50_000,
        description=(
            "Region draws added to the folded points when computing the bounding box "
            "consumed by box-restricted backends. At most `50 * max(mass_probe, box_samples)` "
            "uniform candidates (and `max_candidates`) are spent on them; a region too small "
            "to fill them keeps the draws that budget yields."
        ),
    )
    box_padding: NonNegativeFloat = Field(
        default=0.1,
        description="Minimum padding of the bounding box on each side, as a fraction of its width.",
    )
    max_candidates: PositiveInt = Field(
        default=2_000_000_000,
        description=(
            "Ceiling on the uniform candidates spent to fill one region sample; beyond it "
            "sampling raises `TruncationBudgetError`."
        ),
    )

    # ----- execution -----
    torch_threads: PositiveInt | None = Field(
        default=1,
        description="Intra-op torch threads while training and predicting; `None` keeps torch's setting.",
    )
    seed: NonNegativeInt | None = Field(
        default=None,
        description="Seed of every random stream of the restriction; `None` draws fresh entropy.",
    )
    plot: StrictBool = Field(
        default=True,
        description="Write one diagnostic PDF per wave when an output directory is given.",
    )
