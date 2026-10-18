# XSPEC helpers

bsixsa reads the data and the model from the live PyXspec session. These
helpers put a fitted point back into XSPEC, read a loaded spectrum and its
model in counts, and keep XSPEC quiet.

```python
import xspec
from bsixsa.xspec import SpectrumState, set_parameters

result = solver.run()
# Load the posterior median into XSPEC: its plot, flux and eqwidth commands now use it.
set_parameters(result.posterior_samples.median(), solver.parameter_indices, solver.model_names)
xspec.AllModels.calcFlux("0.5 2.0")
print(xspec.AllData(1).flux[0])  # erg/cm^2/s

state = SpectrumState(1)  # spectrum 1, in counts per channel
residuals = state.observed_counts - state.total_model_counts
```

To get XSPEC's errors and fluxes with credible intervals from the posterior,
export it as a chain (see [XSPEC chain export](export.md)).

::: bsixsa.xspec.set_parameters
    options:
      show_root_heading: true
      show_root_full_path: false

::: bsixsa.xspec.SpectrumState
    options:
      show_root_heading: true
      show_root_full_path: false

::: bsixsa.xspec.loaded_spectra
    options:
      show_root_heading: true
      show_root_full_path: false

::: bsixsa.convenience.XSilence
    options:
      show_root_heading: true
      show_root_full_path: false

::: bsixsa.convenience.iter_free_parameters
    options:
      show_root_heading: true
      show_root_full_path: false

::: bsixsa.convenience.XspecParameter
    options:
      show_root_heading: true
      show_root_full_path: false

## Several spectra and models

bsixsa fits every loaded spectrum: the likelihood is XSPEC's fit statistic
summed over the spectra, and the folded counts (`Solver.simulate`,
`Solver.observed_counts`) concatenate the noticed channels of the spectra in
XSPEC index order. XSPEC keeps one copy of a model per data group whose
spectra have a response for its source, so a background model attached to
the third spectrum only exists in that spectrum's data group. Prior entries
address the copy of each model in its first data group; the parameters of
the later copies must be frozen or linked (XSPEC links them to the first
data group by default).

::: bsixsa.convenience.first_model_copy
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: bsixsa.convenience.model_data_groups
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Folding parameter vectors

Every fold of a [`Solver`][bsixsa.solver.Solver] goes through this function;
call it directly only to fold outside a solver.

::: bsixsa.xspec.parallel_folding
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
