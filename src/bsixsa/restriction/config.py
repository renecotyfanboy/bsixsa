"""Configuration of the history-matching prior restriction."""

from __future__ import annotations

import math
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictBool

from .._config_types import NonNegativeInt, PositiveFloat, PositiveInt

__all__ = ["HistoryMatching"]

NonNegativeFloat = Annotated[float, Field(strict=True, ge=0)]
OpenUnitFloat = Annotated[float, Field(strict=True, gt=0, lt=1)]


class HistoryMatching(BaseModel):
    r"""Settings of history matching, which shrinks the prior to the region worth sampling.

    Most users set only `seed`, `n_waves`, `simulations_per_parameter` and
    `target_spans`; the other fields tune the surrogate and the sampling budgets.

    The threshold of a wave is
    $t = \max(C_{best} + s\,\Delta,\; C_{best} + \Delta + k\,\sigma_C,\; t_{cost})$,
    where $C_{best}$ is the best exact C-stat so far, $\Delta = \chi^2_d(0.999)$
    the posterior span, $s$ is `target_spans`, $k$ is `min_margin_sigma`,
    $\sigma_C$ the surrogate's held-out C-stat RMSE near the region and
    $t_{cost}$ the lowest threshold one wave of `max_candidates` can still fill.
    The threshold never increases between waves, except through the recall check.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    # ----- waves -----
    n_waves: PositiveInt = Field(
        default=5,
        description="Maximum number of history-matching waves.",
    )
    simulations_per_parameter: PositiveInt = Field(
        default=500,
        description="Exact folds per wave and per free parameter.",
    )

    # ----- threshold -----
    target_spans: PositiveFloat = Field(
        default=3.0,
        description="Width of the region, in posterior spans above the best C-stat.",
    )
    min_margin_sigma: NonNegativeFloat = Field(
        default=3.0,
        description="Keep the threshold at least this many surrogate C-stat errors above `best + span`.",
    )

    # ----- stopping -----
    stop_when_acceptance: OpenUnitFloat = Field(
        default=5e-3,
        description="Stop once the region holds less than this fraction of the prior.",
    )
    stop_when_threshold: NonNegativeFloat = Field(
        default=0.01,
        description=(
            "Stop once the threshold falls by less than this relative amount between two waves; "
            "lower it for spectra with a large best C-stat."
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
        description="Fraction of each wave's folds held out for validation and the recall check.",
    )
    focus_weight: PositiveFloat = Field(
        default=10.0,
        description="Loss weight of folds with C-stat below `focus_factor` times the previous threshold.",
    )
    focus_factor: PositiveFloat = Field(
        default=3.0,
        description="Multiple of the previous threshold below which folds get `focus_weight` and set the RMSE.",
    )

    # ----- region sampling -----
    mass_probe: PositiveInt = Field(
        default=50_000,
        description="Size of the uniform prior samples used to measure the region's prior mass.",
    )
    box_samples: NonNegativeInt = Field(
        default=50_000,
        description="Region draws added to the folds when computing the bounding box (0 skips them).",
    )
    box_padding: NonNegativeFloat = Field(
        default=0.1,
        description="Minimum padding of the bounding box on each side, as a fraction of its width.",
    )
    max_candidates: PositiveInt = Field(
        default=2_000_000_000,
        description="Maximum number of uniform trial points per rejection draw from the region.",
    )

    # ----- execution -----
    torch_threads: PositiveInt | None = Field(
        default=1,
        description="Torch threads for the surrogate; `None` keeps torch's setting.",
    )
    seed: NonNegativeInt | None = Field(
        default=None,
        description="Random seed for reproducible runs; `None` draws fresh entropy.",
    )
    plot: StrictBool = Field(
        default=True,
        description="Write one diagnostic PDF per wave when an output directory is given.",
    )

    @property
    def mass_probe_budget(self) -> int:
        """Largest number of uniform points one prior-mass measurement may use."""
        resolve_stop = math.ceil(400.0 / self.stop_when_acceptance)
        return min(self.max_candidates, max(50 * self.mass_probe, resolve_stop))

    @property
    def box_candidate_budget(self) -> int:
        """Largest number of uniform candidates spent on the bounding-box draws."""
        return min(self.max_candidates, 50 * max(self.mass_probe, self.box_samples))
