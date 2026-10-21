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
