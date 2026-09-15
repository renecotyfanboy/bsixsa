# Solver and results

Set `restrict_prior=True` or pass `HistoryMatching(...)` to share prior restriction
between samplers. SIXSA enables it by default. See [history matching](restriction.md).

Set `xspec.Fit.bayes = "off"` before running: the Solver checks this for every
backend to prevent XSPEC and the sampler from counting priors twice.

`FitResults.n_likelihood_evaluations` counts the backend folds and any newly
computed restriction. `n_restriction_evaluations`, `restriction_time` and
`log_Z_restriction_correction` make that accounting explicit.

::: bsixsa.solver
    options:
      show_root_heading: false
      members:
      - Solver
      - FitResults
      - likelihood_per_bin
