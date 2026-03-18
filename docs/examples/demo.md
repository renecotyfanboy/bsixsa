---
icon: lucide/rocket
---

# Example fit

## Data loading with PyXSPEC

``` python title="Mandatory imports and clearing"
import os
import xspec

xspec.AllData.clear()
xspec.AllModels.clear()

# Some settings for XSPEC
xspec.Fit.statMethod = "cstat"
xspec.Fit.bayes = "off"
xspec.Xset.xsect = "vern"
xspec.Xset.abund = "lpgs"
```

``` python title="Load the data with OGIP files"
xspec_observation = xspec.Spectrum(
    "fakeit_tycho_raw.pha",
    respFile="reg_64_10093.rmf",
    arfFile="reg_64_10093.arf",
)

# Removing bad channels
xspec.AllData.ignore("bad") 

# Set the energy band
low_energy, high_energy = 0.5, 10.
xspec_observation.ignore(f"0.0-{low_energy:.1f} {high_energy:.1f}-**") 
``` 


``` python title="Setup the XSPEC model"
xspec.Xset.addModelString("NEIAPECROOT","3.1.3") # Extra model variable
xspec_model = xspec.Model("tbabs*(vnei + vnei)")
xspec_model.show()
```

## Prior distributions

``` python title="Define prior distributions"
from bsixsa.priors import uniform, loguniform
from bsixsa.convenience import XSilence

with XSilence():
    
    xspec_model.vnei_3.Redshift.link = xspec_model.vnei.Redshift # Linking parameters
    xspec_model.TBabs.nH = '0.7,-1,0,0,9999,9999' # Fixing a value

prior = [
    ("TBabs_1", "nH", uniform(0.4, 0.8)),
    ("vnei_2", "kT", uniform(0.4, 10.)),
    ("vnei_2", "Mg", loguniform(0.1, 500.)),
    ("vnei_2", "Si", loguniform(50., 5_000.)),
    ("vnei_2", "S", loguniform(50., 5_000.)),
    ("vnei_2", "Ar", loguniform(50., 5_000.)),
    ("vnei_2", "Ca", loguniform(50., 5_000.)),
    ("vnei_2", "Tau", loguniform(5e8, 5e11)),
    ("vnei_2", "Redshift", uniform(-6e-2, +6e-2)),
    ("vnei_2", "norm", loguniform(1e-7, 1e-3)),
    ("vnei_3", "kT", uniform(5., 20.)),
    ("vnei_3", "Fe", loguniform(50., 5_000.)),
    ("vnei_3", "Tau", loguniform(5e8, 5e11)),
    ("vnei_3", "norm", loguniform(1e-8, 1e-4)),
]
```


## Solver 

``` python title="Define the solver"
from bsixsa.solver import SIXSASolver

# Instantiate solver the BXA's way
solver = SIXSASolver(
    prior,
    outputfiles_basename="result_nessai/",
    overwrite=True,
    backend="nessai"
)
```

``` python title="Prior predictive check"
solver.plot_ppc("prior");
```

``` python title="Run the solver"
sampler = solver.run(n_live_points=2_000)
```

``` python title="Posterior predictive check"
solver.plot_ppc("posterior");
```