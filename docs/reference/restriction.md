# History matching

`Solver(..., restrict_prior=True)` enables history matching for any backend.
Pass `HistoryMatching(...)` for explicit settings, or `False` to disable it.
With `None` (the default), only SIXSA enables it.

```python
from bsixsa import HistoryMatching, Nautilus, Solver

solver = Solver(
    prior, backend=Nautilus(seed=42),
    restrict_prior=HistoryMatching(seed=42, target_spans=3.0),
    outputfiles_basename="fit-nautilus",
)
with solver.pool_scope():  # share the XSPEC worker pool across both steps
    restriction = solver.restrict()  # optional: run() computes it if needed
    result = solver.run()
print(restriction.report())
```

::: bsixsa.restriction
    options:
      show_root_heading: false
      members: false

## How the result is used

History matching returns a [`Restriction`][bsixsa.restriction.Restriction],
kept in memory as `solver.restriction`; its folds and per-wave history are
written to `<outputfiles_basename>/restriction/`. The surrogate itself is not saved,
so a restriction cannot be reloaded from disk; `restrict(force=True)`
recomputes it.

SIXSA draws its training data inside the region and keeps the full prior for
its importance weights. The other backends sample the padded bounding box as
a truncated prior, which keeps the family of each marginal where possible
(`solver.prior` and `distributions["restricted_prior"]`; the original prior
stays at `solver.base_prior` and `distributions["prior"]`). The Solver adds
the log prior mass of the box to their evidence (`log_Z_restriction_correction`).
**This corrects the prior normalisation, not excluded posterior mass**: the
result equals the full evidence only when the posterior mass outside the box
is negligible. A `UserWarning` flags posterior points crowding an edge of the
box; the absence of this warning cannot exclude a missed separate mode.

A computed `Restriction` can be passed as `restrict_prior=` to another Solver
for the same problem, e.g. to compare backends on the same region; the reused
folds are not counted again. It is not checked: reuse it only with the same
spectrum, responses, model and priors, and recompute it after any change. For UltraNest, use a fresh output directory when recomputing
the region: a run cannot resume with a different prior box.

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
      - pad_box
