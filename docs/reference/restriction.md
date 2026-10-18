# History matching

SIXSA starts with history matching, which shrinks the prior to the region
worth sampling. Pass `SIXSA(history_matching=HistoryMatching(...))` for
explicit settings, or `history_matching=None` to skip it.

```python
from bsixsa import SIXSA, HistoryMatching, Solver

solver = Solver(
    prior,
    backend=SIXSA(seed=42, history_matching=HistoryMatching(seed=42, target_spans=3.0)),
    outputfiles_basename="fit-sixsa",
)
result = solver.run()
print(solver.backend.restriction.report())
```

::: bsixsa.restriction
    options:
      show_root_heading: false
      members: false

## How the result is used

History matching returns a [`Restriction`][bsixsa.restriction.Restriction],
kept in memory as `solver.backend.restriction`; its folds and per-wave history
are written to `<outputfiles_basename>/sixsa/restriction/`. The surrogate itself
is not saved, so a restriction cannot be reloaded from disk.

SIXSA draws its training data inside the region and keeps the full prior for
its importance weights. Its folds count in `FitResults.n_likelihood_evaluations`,
under the `"restriction"` phase of `solver.fold_counts`.

## Configuration and results

::: bsixsa.restriction.HistoryMatching
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: bsixsa.restriction.Restriction
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: bsixsa.restriction.Region
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: bsixsa.restriction.run_history_matching
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## How the region is decided

::: bsixsa.restriction.rules
    options:
      show_root_heading: false
      heading_level: 3
      members:
      - posterior_span
      - threshold_rule
      - recall_check
      - stop_reason
