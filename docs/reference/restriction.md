# Shared history matching

`Solver(..., restrict_prior=True)` enables history matching for any backend.
Pass `HistoryMatching(...)` for explicit settings, or `False` to disable it.
With `None` (the default), only SIXSA enables restriction.

```python
from bsixsa import Solver, Nautilus, HistoryMatching

solver = Solver(
    prior, backend=Nautilus(seed=42),
    restrict_prior=HistoryMatching(seed=42, target_spans=3.0),
    outputfiles_basename="fit-nautilus",
)
with solver.pool_scope():  # share the XSPEC pool across both steps
    region = solver.restrict()  # optional: run() computes it if needed
    result = solver.run()
```

Each wave folds exact C-stat values, fits a regressor, and tightens a level-set
region. A validation recall check may widen that region. Results are cached in
memory and written as numeric diagnostics; loading a model from disk is not
supported. `restrict(force=True)` recomputes it.

SIXSA uses the region to generate training data and keeps the full prior for its
importance weights. Other backends use a padded bounding box, preserving the
original marginal families where supported. The Solver adds `log(M_box)` to
their evidence. **This corrects the prior normalisation, not excluded posterior
mass**: the result equals the full evidence only when excluded mass is negligible.
A `BoxEdgeWarning` flags posterior points crowding an artificial boundary;
absence of this warning cannot exclude a missed disjoint mode.

The original prior stays at `solver.base_prior` and `distributions["prior"]`.
The installed box is `solver.prior` and `distributions["restricted_prior"]`.

A computed `Restriction` can be passed to another Solver for the same problem.
A fingerprint checks priors, data, responses and model state. Reused folds are not
charged again. Restriction diagnostics record their original simulation count.
Changing the XSPEC problem invalidates the cached restriction. For UltraNest,
use a fresh output directory when recomputing the region; resuming a run with a
different prior box is unsupported.

::: bsixsa.restriction
    options:
      show_root_heading: false
      members:
      - HistoryMatching
      - Restriction
      - Region
      - run_history_matching
