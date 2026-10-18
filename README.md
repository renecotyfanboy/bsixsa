# bsixsa

> :warning: **Work in progress**: expect bugs and breaking API changes.

Bayesian X-ray spectral fitting on top of PyXspec: nested sampling (Nautilus,
Nessai, UltraNest), MCMC (emcee), optimisers (Levenberg-Marquardt, MINUIT) and
SIXSA, Simulation-based Inference for X-ray Spectral Analysis.

Documentation: https://bsixsa.readthedocs.io/en/latest/

## Installation

bsixsa needs XSPEC. Create an environment with it from HEASARC's conda channel,
then install bsixsa from GitHub with the samplers you want as extras:

```bash
conda create -n bsixsa python=3.12 xspec xspec-data -c https://heasarc.gsfc.nasa.gov/FTP/software/conda/ -c conda-forge
conda activate bsixsa
pip install "bsixsa[nautilus] @ git+https://github.com/renecotyfanboy/bsixsa"
```

The extras are `nautilus`, `nessai`, `ultranest`, `emcee`, `iminuit`, or `all`.
SIXSA and Levenberg-Marquardt need none. The `bsixsa` package on PyPI is an
older, incompatible release.

## Quick start

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
result = solver.run()
print(result.posterior_samples.describe())
print(result.log_Z, result.log_Z_err)
```

In a `.py` script on macOS or Windows, put the fit under
`if __name__ == "__main__":`: Nautilus starts its own worker processes, which
re-import the script, and without this guard the run hangs. Jupyter is not
affected.

The documentation explains how to write priors, which backend to choose, and how
to read the results.
