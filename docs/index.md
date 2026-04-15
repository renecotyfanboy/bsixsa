---
icon: lucide/rocket
---

# Welcome 

## Welcome to (B)SIXSA's documentation!

`(B)SIXSA` is a friendly implementation of the simulation-based inference methodology developed in the "SBI with NPE applied to X-ray spectral fitting" paper serie[^1][^2].
In originally started as a fork the [Bayesian X-ray Analysis toolkit (`BXA`)](https://johannesbuchner.github.io/BXA/index.html) and implements a similar syntax using `xspec` as a backend.
[^1]: [Barret & Dupourqué, 2024, Astronomy & Astrophysics, Volume 686, id.A133, 13 pp.](https://ui.adsabs.harvard.edu/abs/2024A%26A...686A.133B/abstract)
[^2]: [Dupourqué & Barret, 2025, Astronomy & Astrophysics, Volume 699, id.A179, 16 pp.](https://ui.adsabs.harvard.edu/abs/2025A%26A...699A.179D/abstract)

## Installation

The package is not yet deployed on PyPI. However, it should be easy to install it from the source code. 
The following procedure is recommended.

First create a clean environment with `xspec` installed.

```
mamba create -n bsixsa python=3.12 xspec xspec-data -c https://heasarc.gsfc.nasa.gov/FTP/software/conda/
conda activate bsixsa # Activate the environment
```

Then install the package from the source code.

```
pip install uv # uv is recommended as it is faster than pip and better handles dependencies
uv pip install git+https://github.com/renecotyfanboy/bsixsa
```

You can now install the extra samplers you want to use. 

```
uv pip install jupyterlab ipywidgets nessai nautilus-sampler # Install nautilus and nessai 
```

## Current limitations

While we do not explicitly support multi-observation fitting and background, solvers using only the likelihood should work correctly. 

- Plotting : the plotters currently support the first spectrum. Also, there is no background support for plotting yet. 
- Flux : flux computation is not yet supported.