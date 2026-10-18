# Backends

The backend is the algorithm that fits: choose it by passing its configuration
as `Solver(backend=...)`, with any options.

```python
from bsixsa import Nautilus, Solver

solver = Solver(prior, backend=Nautilus(num_live_points=2000, seed=1))
```

- **Nested samplers** ([`Nautilus`][bsixsa.backend.config.Nautilus],
  [`Nessai`][bsixsa.backend.config.Nessai],
  [`Ultranest`][bsixsa.backend.config.Ultranest]): posterior samples and the
  evidence `log_Z`.
- **MCMC** ([`Emcee`][bsixsa.backend.config.Emcee]): posterior samples, no
  evidence.
- **Optimisers** ([`LevenbergMarquardt`][bsixsa.backend.config.LevenbergMarquardt],
  [`Iminuit`][bsixsa.backend.config.Iminuit]): the best fit and a Gaussian
  approximation of the posterior around it, no evidence. They warn when a
  parameter ends on a prior bound, or much closer to it than its
  uncertainty: that approximation is then meaningless.
- **Simulation-based inference** ([`SIXSA`][bsixsa.backend.config.SIXSA],
  below): posterior samples and the evidence, checked against exact folds.

Each configuration checks its options when it is built, so a misspelled option
or an invalid value fails at once. Integer options also accept whole numbers
written as floats, such as `1e4`. The options that depend on the data (e.g.
`x0_physical` must have one value per free parameter) are checked when the
Solver is built. The configuration names the sampler package the backend
needs; `Solver(...)` raises `ImportError`, with the `pip install` command,
when it is missing.

## SIXSA

SIXSA (simulation-based inference for X-ray spectral analysis) fits a spectrum
without running a sampler on the likelihood. It learns a **proposal** from
simulated spectra (an autoencoder compresses the counts, neural posterior
estimators map the codes to parameters), then corrects that proposal by
importance sampling against XSPEC's own C-statistic. The networks only decide
where to look; the answer comes from exact folds, so a poor network costs
efficiency and cannot cost correctness, and how poor it is gets measured every
round.

```python
import xspec
from bsixsa import SIXSA, HistoryMatching, Solver

xspec.Fit.statMethod = "cstat"
xspec.Fit.bayes = "off"
# Load one spectrum, its response and model, then define your prior as usual.
solver = Solver(
    prior,
    backend=SIXSA(seed=42, n_nde=5, history_matching=HistoryMatching(seed=42)),
    outputfiles_basename="fit-sixsa",
)
result = solver.run()
print(result.log_Z, result.log_Z_err)
print(solver.backend.posterior.passed, solver.backend.posterior.k_hat)
```

The backend logs each round through the `bsixsa` logger: attach a handler (or
`logging.basicConfig(level=logging.INFO)`) to follow the history-matching
waves, the compressor fits, the k-hat trajectory and the truncation.

SIXSA requires one spectrum, the C-statistic, no loaded background and finite
priors; `Solver(...)` checks this.

!!! info "Terms"

    - **fold**: one exact evaluation of the model through XSPEC, giving the
      C-stat $C$ of a parameter vector.
    - **unit cube**: each parameter mapped through its prior CDF,
      $u = F(\theta)$, so the prior is uniform on $[0, 1]^d$.
    - **proposal**: a density $q$ over the parameters, the equal-weight
      mixture of the round's flows at the observed spectrum, cheap to sample
      and to evaluate.
    - **importance weight**: $w = e^{-C/2} / q$ for a draw of the proposal; the
      weighted draws are a posterior sample, and the mean weight estimates the
      evidence.
    - **k-hat** ($\hat k$): the shape of a Pareto fit to the largest weights
      (Pareto-smoothed importance sampling, PSIS). Below 0.5 the weights have
      a finite variance; above 0.7 a few draws dominate and the estimate is not
      reliable.
    - **ESS** and **efficiency**: the effective sample size of the weighted
      draws, and the ESS per fold in percent.
    - **certificate**: the draws, weights and diagnostics of one check of a
      proposal; a proposal passes the **gate** when its k-hat is below
      `khat_threshold`, its efficiency is at least `min_efficiency`, and
      more than ⌈3√n⌉ + 4 of its n weights are positive, which is enough for
      the Pareto tail fit (draws outside the unit cube weigh zero).
    - **SIR** (sampling importance resampling): drawing many candidates from
      a proposal and resampling them by weight, here to draw uniformly inside
      a region.

### What the run does

1. [History matching](restriction.md) learns a generous plausible region from
   exact C-stat evaluations; `history_matching=None` skips it.
2. A training set is simulated flat on that region (or on a Latin hypercube of
   the full prior without history matching). Parameters live in the prior's
   unit cube throughout, where the prior density is one.
3. A Cash-loss autoencoder maps counts to latent codes. It is refitted at the
   start of each round, warm-started from the previous fit (see
   `refresh_compressor` to refit only in the first rounds); the training
   spectra and the observation are then both encoded with the current fit.
4. `n_nde` flows are trained from different seeds on the (parameters, code)
   pairs with sbi's `NPE_C` and the plain maximum-likelihood loss; their
   equal-weight mixture is the round's proposal.
5. The proposal is certified on `check_draws` draws (each one inside the prior
   bounds costs an exact fold). The gate
   is k-hat below `khat_threshold` (0.7) and an efficiency of at least
   `min_efficiency` (5%), with more than ⌈3√n⌉ + 4 positive weights (at least
   100 of the default 1000 draws) so that k-hat can be fitted at all.
6. Unless the gate passed or the rounds ran out, the next training set (`train_ramp` times larger,
   capped at `train_max`) is drawn approximately uniformly inside the proposal's
   highest-density region $\{u : \log q(u) > t\}$, which holds
   1 - `truncated_quantile` of its in-cube mass (99.9% by default), intersected
   with the history-matching region, by sampling importance resampling with
   `truncated_oversampling` flow candidates per draw. This happens whatever the
   round's k-hat: a poor certificate says the proposal's tails cannot carry
   importance weights, not that its high-density region misses the posterior,
   and shrinking the training region round after round is what makes the loop
   converge (truncated sequential NPE, Deistler et al. 2022).
7. The proposal of the best gate certificate is importance-sampled in batches
   of 5000 draws (each draw inside the prior bounds costs one exact fold) until
   the effective sample size reaches `target_ess` or the draws reach
   `max_sampling_draws`; the weighted draws give the posterior and the log
   evidence.

A passed gate is a finite-sample diagnostic, not a proof that every posterior
mode was found. A low k-hat cannot diagnose a mode absent from the proposed
draws, and no certificate can see posterior mass excluded by the
history-matching region; the backend logs a warning when 99.9% of the
posterior weight comes within one posterior span ($\chi^2_d(0.999)$) of the
region's threshold.

A run that exhausts `n_rounds` without passing the gate warns
(`RuntimeWarning`) and caps its final sample at
`min(max_sampling_draws, 5 * check_draws)` draws. The final sample is then
certified on its own: whenever its k-hat or efficiency fails the gate, the
backend warns again, and its posterior and `log_Z` should be treated as
approximate.

### Outputs

`solver.run()` returns the usual [`FitResults`][bsixsa.solver.FitResults]:
`num_posterior_samples` equal-weight rows resampled from the weighted draws,
the log evidence with its delta-method error, and the best fold with the
XSPEC-convention prior term. `solver.backend` keeps:

- `posterior`, the final certificate: raw draws `u` in the unit cube, `cstat`,
  `log_w`, PSIS-smoothed `log_w_smoothed`, `k_hat`, `ess`, `efficiency`,
  `log_z`, `log_z_err` and `passed`;
- `history`, one record per round: the gate certificate's summary scalars
  plus the certificate itself under `certificate` (left out of
  `summary.json`), the compressor fit, the flows' training records and the
  truncation diagnostics;
- `certificates`, the gate certificate of every round;
- `restriction`, the history-matching
  [`Restriction`][bsixsa.restriction.Restriction], or `None` without history
  matching;
- `best_round` and `exit_reason` (`"gate"` or `"n_rounds"`).

```python
points, weights = solver.backend.weighted_samples()  # physical coordinates
more = solver.backend.sample(1000)
idata = solver.build_inference_data(num_samples=1000)
```

Under `fit-sixsa/sixsa/`: `summary.json` (the scalars above, per round, plus
the configuration), `posterior.npz` (the weighted draws), one TensorBoard
directory per trained flow in `round_<r>/member_<k>/`, and the history-matching
folds and per-wave history in `restriction/`.

The evidence uses XSPEC's `exp(-C/2)` convention. Raw flow draws outside the
unit cube receive zero weight and stay in the denominator, so certification
targets the full prior without any leakage estimate.

### Certificates

A certificate weighs `n` raw draws of a proposal against the exact statistic,
`log w = -C/2 - log q`, and reports the Pareto-smoothed importance sampling
diagnostics (Vehtari et al.): k-hat, the effective sample size, the efficiency
and `log Z = logsumexp(log w) - log n`. When no round passes the gate, the
best-ranked certificate gives the final proposal. k-hat gates and efficiency
ranks: a passing certificate beats the others; among the rest, those with k-hat below
0.5 (finite weight variance) compare on efficiency, and otherwise the lowest
k-hat (within 0.05) decides, again on efficiency. A certificate whose weights
cannot support a tail fit is never selected over one that can.

### Implementation

Every fold goes through `Solver.simulate`, the parameters through the prior's
inverse CDF, clipped to its support (the same map as history matching), and
the evaluations through the backend's tracer, like the other backends.
Certification and truncation take a C-stat callback and run without XSPEC or
torch in the tests.

| module | what it owns |
|---|---|
| `_sixsa` | the rounds, the training-set law and the outputs |
| `compressor` | the Cash-loss autoencoder and its generation counter |
| `flows` | the flows frozen at the observation, their mixture, sbi training |
| `certify` | importance weights, PSIS, k-hat and the ranking |
| `truncation` | the SIR draws of the next training set |
| `sampling` | the final importance sample and its equal-weight resample |

The compressor is the one large network (its first layer is `n_bins` wide);
`compressor=CompressorConfig(device="auto")` runs it on cuda or mps when
available.

### Configuration reference

::: bsixsa.backend.config.SIXSA
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

::: bsixsa.backend.config.FlowConfig
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

::: bsixsa.backend.config.CompressorConfig
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

## Other backends

::: bsixsa.backend.config.Nautilus
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: bsixsa.backend.config.Nessai
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: bsixsa.backend.config.Ultranest
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: bsixsa.backend.config.Iminuit
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: bsixsa.backend.config.LevenbergMarquardt
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: bsixsa.backend.config.Emcee
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Options shared by every backend

::: bsixsa.backend.config.BackendConfig
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
