# SIXSA v2 (SBI 0.27)

SIXSA compresses simulated counts with a Cash-loss autoencoder, trains independent
NPEs in parallel, and checks their proposals against exact XSPEC likelihoods.
History matching is enabled by default through the Solver.

```python
import xspec
from bsixsa import Solver, SIXSA, HistoryMatching

xspec.Fit.statMethod = "cstat"
xspec.Fit.bayes = "off"
# Load one spectrum, its response and model, then define your prior as usual.
solver = Solver(
    prior,
    backend=SIXSA(
        seed=42,
        n_nde=5,
        training_n_jobs=4,
        training={"max_num_epochs": 25},
        compressor={"epochs": 1000},
    ),
    restrict_prior=HistoryMatching(seed=42),
    outputfiles_basename="fit-sixsa",
    n_jobs=4,  # exact XSPEC folds; distinct from NPE training workers
)
result = solver.run()
print(result.log_Z, result.log_Z_err)
print(solver.backend.posterior.certified, solver.backend.posterior.k_hat)
```

## What the run does

1. History matching learns a generous plausible region from exact C-stat evaluations.
2. A fresh training set is simulated in that region (or on the full prior with
   `restrict_prior=False`).
3. An autoencoder maps counts to latent codes. It is warm-refitted each round;
   every flow keeps its own observation code and compressor generation.
4. NPE members train in separate loky processes. Their proposals and mixture are
   screened with exact folds; the mixture reuses the members' draws.
5. The selected proposal is checked on **fresh** draws. The default stopping gate
   is k-hat < 0.7 and PSIS ESS / exact folds >= 5%.
6. If needed, a fully replaced training set is drawn for the next round.
7. A planned final importance sample supplies posterior weights and log evidence.

A passed gate is a finite-sample diagnostic, not a proof that every posterior mode
was found. A low k-hat cannot diagnose a mode absent from the proposed draws.

## Evidence and posterior outputs

All certification targets the **base prior**, even with history matching.
Coordinates are transformed through each marginal CDF to the unit cube, where the
prior density is one. Raw flow draws outside that cube receive zero weight and
remain in the evidence denominator. No estimated leakage correction is needed.

`target_ess` controls the planned draw budget; `num_posterior_samples` controls the
number of equal-weight output rows. These are different: resampling can duplicate
points. Inspect `posterior.ess`, `n_unique`, `shortfall` and `k_hat`.

An optional, single ESS-driven top-up enriches posterior resampling. The reported
evidence and its error use **only the initial planned batch**
(`evidence_n_raw`, `evidence_k_hat`), because including an adaptively sized top-up
would introduce stopping bias. Above evidence k-hat 0.5, the delta-method error is
doubled as a heuristic and flagged `log_z_err_reliable=False`; this does not
establish calibrated uncertainty. A run exhausting `n_rounds` returns an explicitly
uncertified result and a warning.

`posterior.certified` means the selected proposal passed the fresh round gate.
Inspect the **final** `posterior.k_hat` too; a later failed diagnostic still emits
a warning. The evidence uses XSPEC's `exp(-C/2)` convention, rather than an
absolute Poisson likelihood with its data-dependent constant.

`best_fit_stat` follows the other bsixsa backends: exact C-stat plus the
base prior's XSPEC-convention contribution. It is not bare C-stat; importance
weights still use the likelihood alone.

```python
points, weights = solver.backend.weighted_samples()  # physical coordinates
more = solver.backend.sample(1000)
idata = solver.build_inference_data(num_samples=1000)
```

Files under `fit-sixsa/sixsa/` contain `summary.json`, weighted `posterior.npz`,
and convergence, compressor and weight diagnostics. Restriction diagnostics live
in `fit-sixsa/restriction/`. Posterior predictive evaluations after the fit are
counted separately in `solver.fold_counts["post"]`.

## Truncation

`truncated_sampling="sir"` keeps the reference v2 behavior. SBI 0.27's
`RestrictedPrior` SIR path uses 32 candidates and interprets its Boolean mask as
a log potential. Consequently training draws are **not uniform** on the contour
and some may leave the history-matching region. The measured fractions are saved.
This affects the learned proposal; final weights still use the full target.

Optional methods:

- `"flow"`: exact rejection sampling on the density contour intersected with
  the history-matching region.
- `"ellipsoid"`: uniform sampling on that region intersected with a fitted
  ellipsoid. It covers the whole contour only if the ellipsoid contains it;
  the empirical coverage and shell checks cannot prove this.
- `"auto"`: chooses between the ellipsoid and flow approaches based on cost.

Candidate budgets raise `TruncationBudgetError` on exhaustion. Flow rejection can
increase the contour quantile up to 0.01, recorded in its diagnostics. No exact
method silently switches to SIR.

## Migration from 0.1.x

The legacy buffer, Gaussian latent, emulator and embedding API has been replaced.
Use `n_nde` instead of `n_ensemble`, and nested `nde`, `training`, `compressor`
configuration. Old fields are rejected. Historical notebooks can be replayed from
the Git tag `sixsa-legacy-spec-snapshot`.

SIXSA currently requires one spectrum, C-stat, no loaded background and finite
independent prior supports. Other likelihood-only samplers retain their capabilities.

## Configuration reference

::: bsixsa
    options:
      show_root_heading: false
      members:
      - BackendConfig
      - SIXSA

::: bsixsa.backend.config
    options:
      show_root_heading: false
      members:
      - NDEConfig
      - TrainingConfig
      - CompressorConfig

## Other backends

::: bsixsa
    options:
      show_root_heading: false
      members:
      - Nautilus
      - Nessai
      - Ultranest
      - LevenbergMarquardt
      - Iminuit
      - Emcee
