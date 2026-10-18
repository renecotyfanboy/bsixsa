# Solver and results

A [`Solver`][bsixsa.solver.Solver] fits the spectrum and the model loaded in
PyXspec. It reads the data and the model from the live session, builds the
prior from your list of priors (see [Priors](priors.md)), and hands the
likelihood to the backend you choose (see [Backends](backend.md)).

```python
import xspec
from bsixsa import Nautilus, Solver
from bsixsa.priors import loguniform, uniform

xspec.Fit.statMethod = "cstat"
xspec.Fit.bayes = "off"  # bsixsa applies the priors itself
xspec.Spectrum("source.pha")
xspec.Model("tbabs*powerlaw")

prior = [
    ("TBabs", "nH", uniform(0.01, 1.0)),
    ("powerlaw", "PhoIndex", uniform(1.0, 3.0)),
    ("powerlaw", "norm", loguniform(1e-5, 1e-1)),
]
solver = Solver(prior, outputfiles_basename="fit-nautilus", backend=Nautilus(seed=1))
result = solver.run()
print(result.posterior_samples.describe())
print(result.log_Z, result.log_Z_err)
```

Every likelihood evaluation is one **fold**: XSPEC evaluates the model at one
parameter vector, folds it through the instrument response and returns the
C-statistic $C$ (XSPEC's cstat). The likelihood is $e^{-C/2}$. Folds are the
expensive step, so the solver runs them in a pool of worker processes
(`n_jobs`) and counts them (`FitResults.n_likelihood_evaluations`).

## Before running

- Set `xspec.Fit.bayes = "off"`. bsixsa applies the priors itself, and
  `run()` refuses a session where XSPEC would add them a second time.
- The output directory `outputfiles_basename` is created if needed. If it is
  not empty, `Solver(...)` raises `FileExistsError` unless `overwrite=True`,
  which deletes its whole content. The current working directory, your home
  directory and their parents are never cleared.
- If the backend needs a sampler package that is not installed, `Solver(...)`
  raises `ImportError` with the `pip install` command, before anything is
  written.
- `Solver(...)` changes the prior parameters in the XSPEC session (start
  values, limits, frozen and linked states); see
  [what the Solver changes in XSPEC](priors.md#what-the-solver-changes-in-xspec).

## Reading the results

`solver.run()` returns a [`FitResults`][bsixsa.solver.FitResults], also stored
as `solver.fit_result`.

- `posterior_samples` is a table with one column per fitted parameter, in
  physical units, named as in `solver.parameter_names` (see
  [parameter names](priors.md#parameter-names-in-the-results)). Nautilus,
  Nessai, UltraNest and SIXSA give equal-weight posterior draws; Emcee gives
  the walker chain after the burn-in (correlated draws). LevenbergMarquardt
  and Iminuit give 10 000 draws from a Gaussian approximation around the best
  fit (built in the logit of the prior CDF, so that they stay inside the
  prior): a quick estimate, not a sampled posterior.
- `log_Z` is the natural logarithm of the evidence (the marginal likelihood),
  with the likelihood $e^{-C/2}$, and `log_Z_err` its uncertainty. $C$
  differs from $-2$ times the full Poisson log-likelihood by a constant that
  depends only on the data, so `log_Z` compares models fitted to the same
  spectrum with the same binning: the difference of two `log_Z` is the log
  Bayes factor. It is NaN for Emcee, Iminuit and LevenbergMarquardt.
- `best_fit` is the best parameter vector (optimisers and SIXSA; `None` for
  the nested samplers and Emcee). `best_fit_stat` is its C-stat plus the
  prior term XSPEC adds when `Fit.bayes = "on"` (0 for a uniform prior,
  $2 \ln x$ for a log-uniform one): compare it with `xspec.Fit.statistic` of a
  bayes-on XSPEC fit with `cons` and `jeffreys` priors, not with a bare cstat
  fit.
- `n_likelihood_evaluations` counts the folds of the backend (for SIXSA,
  history matching included); `time` is the wall-clock time of the backend
  run, in seconds.

After the run, `build_dataframe` and `build_inference_data` draw from the
prior or the posterior (an ArviZ `InferenceData` for diagnostics and model
comparison), `plot_predictive_coverage` checks the fit against the data (see
[Plotting](plotter.md)), and `export_xspec_chain` writes the posterior as an
XSPEC chain file, so that XSPEC's `error` and flux commands use it (see
[XSPEC chain export](export.md)).

`indexes`, `model_indexes` and `observed_spectrum` are older names of
`parameter_indices`, `model_names` and `observed_counts`, kept for existing
scripts.

::: bsixsa.solver.Solver
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
      - parameter_names
      - num_parameters
      - observed_counts
      - n_bins
      - posterior_samples
      - backend_name
      - run
      - sample_parameters
      - build_dataframe
      - export_xspec_chain
      - build_inference_data
      - plot_predictive_coverage

::: bsixsa.solver
    options:
      show_root_heading: false
      members:
      - FitResults
      - likelihood_per_bin

## Advanced: folds, pools and custom models

These `Solver` members are for writing a backend, calling the likelihood
directly, or replacing XSPEC by another forward model (see "Extending the
solver" above).

::: bsixsa.solver.Solver.simulate
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: bsixsa.solver.Solver.log_likelihood_fn
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: bsixsa.solver.Solver.log_prior_fn
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: bsixsa.solver.Solver.log_prob_fn
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: bsixsa.solver.Solver.fold_counts
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: bsixsa.solver.Solver.fold_phase
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: bsixsa.solver.Solver.pool_scope
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: bsixsa.solver.Solver.close_pool
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
