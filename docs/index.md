---
icon: lucide/rocket
---

# Welcome

`bsixsa` does Bayesian X-ray spectral fitting on top of PyXspec. You load your
spectrum and define your model in PyXspec as usual, give each free parameter a
prior, and choose how to fit: nested sampling, MCMC, an optimiser, or SIXSA
(Simulation-based Inference for X-ray Spectral Analysis), the method of the
"SBI with NPE applied to X-ray spectral fitting" paper series[^1][^2].

It started as a fork of the [Bayesian X-ray Analysis toolkit (`BXA`)](https://johannesbuchner.github.io/BXA/index.html)
and keeps a similar syntax. It has since grown into a general-purpose wrapper
around `xspec` for a wide range of fitting and sampling methods.

[^1]: [Barret & Dupourqué, 2024, Astronomy & Astrophysics, Volume 686, id.A133, 13 pp.](https://ui.adsabs.harvard.edu/abs/2024A%26A...686A.133B/abstract)
[^2]: [Dupourqué & Barret, 2025, Astronomy & Astrophysics, Volume 699, id.A179, 16 pp.](https://ui.adsabs.harvard.edu/abs/2025A%26A...699A.179D/abstract)

## Installation

bsixsa 0.2 is not on PyPI yet (the `bsixsa` package there is an older,
incompatible release), so install it from GitHub. First create an environment
with XSPEC, from HEASARC's conda channel:

```bash
conda create -n bsixsa python=3.12 xspec xspec-data -c https://heasarc.gsfc.nasa.gov/FTP/software/conda/ -c conda-forge
conda activate bsixsa
```

Then install bsixsa with the samplers you want, given as extras in brackets:

```bash
pip install "bsixsa[nautilus] @ git+https://github.com/renecotyfanboy/bsixsa"
```

Each sampler is an extra named after its backend: `nautilus`, `nessai`,
`ultranest`, `emcee` and `iminuit` (see [Choosing a backend](#choosing-a-backend)).
Give several separated by commas, e.g. `bsixsa[nautilus,emcee]`, or all of them
with `bsixsa[all]`. SIXSA and LevenbergMarquardt need no extra. If the sampler of
the backend you chose is missing, `Solver(...)` stops with an `ImportError` that
gives the `pip install` command.

- PyTorch comes with bsixsa (SIXSA uses it). On a Linux machine without a GPU,
  install the smaller CPU build first:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- In Jupyter, the progress bars need `ipywidgets`: `pip install jupyterlab ipywidgets`.

## A fit in a few lines

```python
import xspec
from bsixsa import Solver, Nautilus
from bsixsa.priors import uniform, loguniform

xspec.Fit.statMethod = "cstat"  # Poisson likelihood (C-statistic)
xspec.Fit.bayes = "off"         # bsixsa applies the priors itself
xspec.Spectrum("source.pha")    # with its responses, as in any PyXspec session
xspec.Model("tbabs*powerlaw")

prior = [
    ("TBabs", "nH", uniform(0.01, 1.0)),
    ("powerlaw", "PhoIndex", uniform(1.0, 3.0)),
    ("powerlaw", "norm", loguniform(1e-5, 1e-1)),
]
solver = Solver(prior, outputfiles_basename="fit_nautilus/", backend=Nautilus())

solver.plot_predictive_coverage("prior")      # can the prior produce spectra like the data?
result = solver.run()
solver.plot_predictive_coverage("posterior")  # does the fit explain the data?
print(result.posterior_samples.describe())
print(result.log_Z, result.log_Z_err)
```

- The `Solver` reads the data and the model from the live PyXspec session, so set
  them up first. It turns XSPEC's fit statistic $C$ into the log-likelihood
  $\ln L = -C/2$, so choose the statistic you would use in XSPEC: `cstat` for
  Poisson counts.
- **Keep `xspec.Fit.bayes = "off"`.** bsixsa applies the priors itself. With
  bayes on, XSPEC would add its own prior term to the statistic and the prior
  would count twice, so `solver.run()` refuses to start.
- The backend writes its files (chains, checkpoints, traces) in
  `outputfiles_basename`. Pass `overwrite=True` to reuse a directory that is not
  empty: its content is deleted.
- **In a `.py` script on macOS or Windows, put the fit under
  `if __name__ == "__main__":`.** Nautilus starts its own worker processes,
  which re-import the script; without this guard the run hangs.
  `Nautilus(n_pool=1)` also avoids it. Jupyter is not affected.

The [simple fit](examples/demo.md) example goes through these steps on a real
case.

## Writing the prior

Each entry gives the prior of one XSPEC parameter:

- `(component, parameter, distribution)` when one model is loaded;
- `(model_name, component, parameter, distribution)` when several models are
  loaded. XSPEC requires a name for every model on source 2 or higher
  (`xspec.Model(..., sourceNum=2, modName="nxb")`). The model on source 1 may
  stay unnamed, and its model name is then `""` (see the
  [NXB example](examples/nxb.md)).

All entries use the same form. Component and parameter names are PyXspec's
(`model.componentNames`, `model.TBabs.parameterNames`) and are case-sensitive:
the expression `tbabs*powerlaw` gives the component `TBabs`, and `apec`'s
abundance is `Abundanc`.

**Repeated components.** XSPEC gives the first copy of a component its plain name
and the others a suffix with their position in the model expression: in
`tbabs*(vnei + vnei)` the components are `TBabs`, `vnei` and `vnei_3`. bsixsa
also accepts `<component>_<position>` for any component: `vnei_2` is the first
`vnei` and `vnei_3` the second. This positional form is unambiguous, and it is
the one used in the results.

**Distributions.** [`bsixsa.priors`](reference/priors.md) provides:

| helper | density | XSPEC prior |
|---|---|---|
| `uniform(low, high)` | flat between the bounds | `cons` |
| `loguniform(low, high)` | flat in $\log x$: each decade gets the same mass; for normalisations and other scale parameters | `jeffreys` |
| `norm(loc, scale, low, high)` | Gaussian of mean `loc` and standard deviation `scale`, truncated to the bounds | `gauss` (not truncated) |

Any frozen SciPy continuous distribution with finite bounds also works, except
with `LevenbergMarquardt`, which accepts only `uniform`, `loguniform` and `norm`
(SciPy's `uniform`, `loguniform` and `truncnorm`). Every prior needs finite
bounds: they become the parameter's limits in XSPEC.

**Parameters without a prior** must be frozen or linked in XSPEC. A prior on a
frozen parameter thaws it, and a prior on a linked parameter removes its link.
When an entry matches no parameter, or a free parameter has no prior,
`Solver(...)` reports all such problems together in one `ValueError`.

**What the `Solver` changes in XSPEC.** For each parameter with a prior, it sets
the value to the prior median, the per-parameter fit delta to 1% of the prior
range (XSPEC uses it only after `xspec.Fit.delta = 0`; by default it uses 1% of
the parameter value), and the soft and hard limits to the prior bounds, which
can be wider than XSPEC's defaults.

## Choosing a backend

The backend is the configuration object passed as `Solver(..., backend=...)`. Its
options are listed in the [backend reference](reference/backend.md).

| backend | method | posterior | `log_Z` | pip extra |
|---|---|---|---|---|
| `Nautilus()` | nested sampling, guided by neural networks | samples | yes | `nautilus` |
| `Nessai()` | nested sampling, guided by normalising flows | samples | yes | `nessai` |
| `Ultranest()` | nested sampling (MLFriends regions) | samples | yes | `ultranest` |
| `Emcee()` | ensemble MCMC | correlated chain | NaN | `emcee` |
| `LevenbergMarquardt()` | optimiser, like XSPEC's `fit` | Gaussian approximation | NaN | none |
| `Iminuit()` | optimiser (MINUIT's MIGRAD) | Gaussian approximation | NaN | `iminuit` |
| `SIXSA()` | simulation-based inference, checked by importance sampling | samples | yes | none |

- **Posterior and evidence: a nested sampler.** It returns posterior samples and
  the evidence needed to compare models. The [simple fit](examples/demo.md) uses
  Nautilus, the [NXB example](examples/nxb.md) Nessai.
- **A quick best fit: an optimiser.** `LevenbergMarquardt` and `Iminuit` return
  the maximum a posteriori parameters (`best_fit`). Their `posterior_samples` are
  10,000 draws from a Gaussian approximation around the best fit, not a sampled
  posterior. Use a sampler when the posterior may be skewed or a parameter sits
  near a prior bound (bsixsa warns in that case). `LevenbergMarquardt` fits the
  first spectrum only (see [Current limitations](#current-limitations)).
- **MCMC: `Emcee`** samples the posterior, without an evidence. It needs a
  starting point: `x0_physical` (one value per free parameter) or
  `init_from_prior=True`.
- **SIXSA** trains neural networks on simulated spectra to propose parameters,
  then weights the proposals with XSPEC's exact C-statistic: the networks only
  decide where to look, and a k-hat test (see the [glossary](#glossary)) checks
  the result. It fits one spectrum with `cstat` and no background, and by
  default first runs [history matching](reference/restriction.md) to narrow the
  region where it simulates spectra. See
  [SIXSA](reference/backend.md) and its [validation](reference/sixsa-validation.md).

## Reading the results

`solver.run()` returns a [`FitResults`](reference/solver.md), also stored as
`solver.fit_result`:

| field | content |
|---|---|
| `posterior_samples` | `pandas.DataFrame`: one row per draw, one column per parameter, in physical units |
| `log_Z`, `log_Z_err` | log evidence and its uncertainty (NaN for the optimisers and Emcee) |
| `best_fit`, `best_fit_stat` | best parameters and their statistic (optimisers and SIXSA; `None` otherwise) |
| `n_likelihood_evaluations`, `time` | number of [folds](#glossary) and run time in seconds |

**Column names** are `<component>_<position>_<parameter>`, preceded by the model
name for a named model: `TBabs_1_nH`, `vnei_2_kT`, `cluster_bapec_2_kT`. The
position counts every component of the expression from 1, so the single
`powerlaw` of `tbabs*powerlaw` is `powerlaw_2`. `solver.parameter_names` lists
the names. The columns follow the parameter numbers of each model, models in
source-number order, not the order of your prior list: select them by name.

**`log_Z`** is the natural logarithm of the Bayesian evidence,
$Z = \int L(\theta)\, \pi(\theta)\, d\theta$, with $L = e^{-C/2}$ as in XSPEC.
This $-C/2$ differs from the full Poisson log-likelihood by a constant that
depends only on the data, so `log_Z` values can be compared only between models fitted
to the same spectrum, with the same binning and energy range. Their difference is
the log Bayes factor of the two models.

**`best_fit_stat`** is the C-statistic at `best_fit` plus the prior term XSPEC
adds with `Fit.bayes = "on"` (0 for `uniform`, $2 \ln x$ for `loguniform`).
Compare it with XSPEC's `Fit.statistic` under the same priors (`cons`,
`jeffreys`), not with a plain C-stat fit.

To go further:

- `solver.plot_predictive_coverage("posterior")` compares the data with spectra
  simulated from the posterior (see the [plotter](reference/plotter.md));
- `solver.build_inference_data()` returns an ArviZ `InferenceData`, for summaries,
  plots and model comparison with `az.loo` and `az.compare`;
- `solver.export_xspec_chain("chain.fits")` writes the posterior as an XSPEC chain
  file, from which XSPEC's `error`, `flux`, `lumin` and `eqwidth` commands compute
  credible intervals (see [XSPEC chain export](reference/export.md) and the last
  step of the [simple fit](examples/demo.md#errors-and-fluxes-with-xspec)).

## Glossary

- **Fold**: one evaluation of the model through the instrument response, which
  gives the predicted counts and the fit statistic: one likelihood evaluation.
  Folds take most of the run time; `n_likelihood_evaluations` counts them.
- **Prior / posterior predictive check**: spectra simulated from parameters drawn
  from the prior (before the fit) or from the posterior (after it), compared with
  the data by `solver.plot_predictive_coverage`.
- **Evidence** (`log_Z`): the probability of the data under the model, averaged
  over the prior; see [Reading the results](#reading-the-results).
- **Unit cube**: the parameters mapped to $[0, 1]$ by the cumulative distribution
  of their priors, where the prior is uniform. Samplers and SIXSA work there;
  results are always given in physical units.
- **History matching**: SIXSA's first step, which folds parameter values spread
  over the prior and rules out the regions where the model cannot fit the data,
  so that SIXSA only learns in the plausible region. See
  [History matching](reference/restriction.md).
- **Importance sampling**: drawing parameters from an approximate distribution
  (the proposal) and giving each draw a weight, its posterior density over its
  proposal density, so that the weighted draws follow the posterior.
- **ESS** (effective sample size): the number of independent, equal-weight draws
  that carry as much information as a set of weighted draws.
- **Efficiency**: the ESS divided by the number of folded draws (those inside the
  prior bounds), in percent. It measures how close the proposal is to the
  posterior.
- **k-hat**: the Pareto $\hat{k}$ diagnostic of importance weights
  (Pareto-smoothed importance sampling, Vehtari et al.). Below 0.7 the weighted
  results are reliable; above it, a few draws dominate them.

## Current limitations

- **Several spectra and backgrounds** are not supported explicitly. The nested
  samplers, Emcee and Iminuit use XSPEC's total fit statistic, so they fit every
  loaded spectrum and account for a loaded background as XSPEC does (C-stat then
  becomes W-stat), but this is not tested systematically. `LevenbergMarquardt`
  fits the counts of the first spectrum only, without its loaded background: do
  not use it with several spectra or a background file (a background modelled
  as a second source, as in the [NXB example](examples/nxb.md), is fine).
  Priors apply to the models of data group 1: keep the parameters of other data
  groups linked to them (XSPEC's default).
- **SIXSA** fits one spectrum, with `cstat`, no loaded background, and
  independent priors with finite bounds.
- **Plots** show the first spectrum only, without its background. The observed
  data and the per-bin likelihood of `build_inference_data` also come from the
  first spectrum only.
- **Cross-calibration** is not supported through the standard `xspec` interface
  (a free constant per data group), but it can be built into the spectral model.
- **Fluxes** are not computed by bsixsa itself: export the posterior as an XSPEC
  chain and use XSPEC's `flux` command with errors, as in the
  [simple fit](examples/demo.md#errors-and-fluxes-with-xspec).
