r"""History matching: SIXSA's first step, which shrinks the prior to the region worth sampling.

- **fold**: one exact C-stat evaluation at a parameter point (the expensive step).
- **surrogate**: a small neural network trained on the folds to predict the C-stat.
- **wave**: fold a batch of points, retrain the surrogate, and keep the points whose
  predicted C-stat is below a threshold; the next wave draws inside what was kept.
- **posterior span** $\Delta = \chi^2_d(0.999)$: the C-stat rise above the best fit that
  holds about 99.9 % of the posterior ($d$ free parameters). The threshold sits
  `HistoryMatching.target_spans` spans above the best fit.
- **recall check**: the threshold is raised if held-out folds that belong to the region
  would be discarded.
- **region**: the parameter values whose predicted C-stat is below the threshold
  ([`Region`][bsixsa.restriction.Region]); SIXSA draws its training data in it.

Everything is done on the unit cube $u = F(\theta)$ (prior CDF per parameter), so the
volume of a region is its prior mass.
"""

from .config import HistoryMatching
from .region import Region, Restriction
from .waves import run_history_matching

__all__ = ["HistoryMatching", "Region", "Restriction", "run_history_matching"]
