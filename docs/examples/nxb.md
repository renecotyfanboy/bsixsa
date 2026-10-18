---
icon: lucide/satellite
---

# Including `NXB` with multi-response

The non-X-ray background (`NXB`) of the detector can be modelled with the [`multiresponse`](https://heasarc.gsfc.nasa.gov/docs/software/xspec/python/html/spectrum.html) interface of `PyXSPEC`, as a second source next to the astrophysical one. We explore this using a mock X-IFU observation of a galaxy cluster.

## Data loading with PyXSPEC

Let's first load the data and define our observational configuration.

``` python title="Mandatory imports and data loading"
import xspec

xspec.AllData.clear()
xspec.AllModels.clear()

# Some settings for XSPEC
xspec.Fit.statMethod = "cstat"  # Poisson likelihood (C-statistic)
xspec.Fit.bayes = "off"         # bsixsa applies the priors itself
xspec.Xset.xsect = "vern"
xspec.Xset.abund = "wilm"

xspec_observation = xspec.Spectrum(
    "fakeit_A85.pha",
    respFile="ABELL85_CORE_60k_60k_Hp_allpix_L.rmf",
    arfFile="xa201099010rsl_p0px1000_ptsrc.arf",
)

# Removing bad channels
xspec.AllData.ignore("bad")

# Set the energy band
low_energy, high_energy = 1.5, 8.
xspec_observation.ignore(f"0.0-{low_energy:.1f} {high_energy:.1f}-**")
```

## Models

Now, we define the model for the cluster as a simple absorbed and broadened APEC model. With several sources, each model needs its own source number, and every model on source 2 or higher needs a name (we name the cluster model too, but it could stay unnamed and be written `""` in the prior).

``` python title="Cluster model"
cluster_model = xspec.Model("tbabs*bapec", sourceNum=1, modName="cluster")
cluster_model.show()
```

Now, we can define the multi-response for the NXB. As the NXB is contributed by the instrument solely, it doesn't go through the effective area: its response is the RMF alone. Hence, we define the multi-response as follows.

``` python title="Multi-response and NXB model setting"
xspec_observation.multiresponse[1] = xspec_observation.multiresponse[0].rmf
nxb_model = xspec.Model("pow", sourceNum=2, modName="nxb", setPars={1: 0, 2: 1e-5})
nxb_model.show()
```

## Prior distributions

With several models loaded, each prior entry must say which model its parameter belongs to: the prior tuples are in the form `(modelName, componentName, parameterName, distribution)` instead of `(componentName, parameterName, distribution)`. The parameters without a prior are frozen first, through the model handles defined above.

``` python title="Define prior distributions"
from bsixsa.priors import uniform, loguniform

# 'value,delta,hard_low,soft_low,soft_high,hard_high': a negative delta freezes the parameter.
cluster_model.TBabs.nH = "0.01, -1, 0, 0, 1, 1"
nxb_model.powerlaw.PhoIndex = "0., -1, 0, 0, 1, 1"  # a flat NXB spectrum

prior = [
    ("cluster", "bapec", "kT", uniform(0, 7)),
    ("cluster", "bapec", "Abundanc", uniform(0, 4)),
    ("cluster", "bapec", "Redshift", uniform(0.05, 0.3)),
    ("cluster", "bapec", "Velocity", uniform(0.5, 1000)),
    ("cluster", "bapec", "norm", loguniform(1e-5, 1e-2)),
    ("nxb", "powerlaw", "norm", loguniform(1e-4, 1e0)),
]
```

## Solver

Once this is set, we can fit the data, here with the Nessai nested sampler.

``` python title="Define the solver"
from bsixsa import Solver, Nessai

solver = Solver(
    prior,
    outputfiles_basename="result_nessai_nxb/",
    overwrite=True,
    backend=Nessai(),
)
```

As in the other fits, we should first check if the priors are properly chosen.

``` python title="Prior predictive check"
solver.plot_predictive_coverage(
    "prior",
    residual_kind="sigma",
    grouping=30,
    figsize=(7,5),
    y_lim=(2, 5e7)
);
```

![Prior predictive](nxb_plots/prior_ppc_nxb.png){ width="600"}
/// caption
Prior predictive coverage for the configured model and priors.
///

Let's run the sampler.

``` python title="Run the solver"
fit_result = solver.run()
```

Now, the fit should be good! Let's check the posterior predictive.

``` python title="Posterior predictive check"
solver.plot_predictive_coverage(
    "posterior",
    residual_kind="sigma",
    grouping=30,
    figsize=(7,5),
    y_lim=(1e2, 3e4)
);
```

![Posterior predictive](nxb_plots/posterior_ppc_nxb.png){ width="600"}
/// caption
Posterior predictive coverage after running the nested sampler.
///

## Reading the results

The result columns start with the model name, e.g. `cluster_bapec_2_kT` and `nxb_powerlaw_1_norm`. In XSPEC, the parameters of a named model are written `<model>:<number>`.

``` python title="Results and XSPEC errors"
print(fit_result.posterior_samples.describe())
print(fit_result.log_Z, fit_result.log_Z_err)

path = solver.export_xspec_chain("posterior_chain_nxb.fits")
xspec.AllChains.clear()
xspec.AllChains += path
xspec.Fit.error("cluster:2")  # 90% credible interval of the cluster temperature
```

Fluxes work as in the [simple fit](demo.md#errors-and-fluxes-with-xspec); XSPEC then reports one flux per source.
