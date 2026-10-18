# Priors

The prior is a list with one entry per parameter to fit. Each entry names an
XSPEC parameter and gives its prior distribution, built with
[`uniform`][bsixsa.priors.uniform], [`loguniform`][bsixsa.priors.loguniform]
or [`norm`][bsixsa.priors.norm]:

```python
import xspec
from bsixsa import Nautilus, Solver
from bsixsa.priors import loguniform, norm, uniform

xspec.Model("tbabs*(powerlaw + powerlaw)")  # with the spectrum already loaded
prior = [
    ("TBabs", "nH", norm(0.3, 0.05, 0.0, 1.0)),       # truncated Gaussian
    ("powerlaw", "PhoIndex", uniform(1.0, 3.0)),      # first powerlaw
    ("powerlaw", "norm", loguniform(1e-5, 1e-1)),     # same prior mass per decade
    ("powerlaw_3", "PhoIndex", uniform(1.0, 3.0)),    # second powerlaw
    ("powerlaw_3", "norm", loguniform(1e-5, 1e-1)),
]
solver = Solver(prior, backend=Nautilus())
```

## Writing the entries

- `(component, parameter, distribution)` when a single model is loaded;
- `(model_name, component, parameter, distribution)` with any number of
  loaded models, and required when several are loaded (e.g. a source model
  and a background model). The unnamed model is `""`. All the entries of a
  list must have the same form.

Component and parameter names are XSPEC's, and case-sensitive: `TBabs`, not
`tbabs` (`model.componentNames` and `component.parameterNames` list them).
XSPEC names a repeated component with its position in the model expression:
in `tbabs*(powerlaw + powerlaw)` the components are `TBabs`, `powerlaw` and
`powerlaw_3`. The first powerlaw can also be written `powerlaw_2`, its
position, as in the result columns below.

Every free parameter needs a prior; frozen and linked parameters need none.
A prior on a frozen parameter thaws it, and a prior on a linked parameter
removes the link. `Solver(...)` refuses a list with a misspelled name, two
entries for the same parameter, a free parameter without a prior or a prior
without finite bounds, and its error says how to fix it.

## What the Solver changes in XSPEC

For each parameter with a prior, `Solver(...)` (through
[`build_prior`][bsixsa.priors.build_prior]) sets in the XSPEC session:

- the value to the prior median;
- the per-parameter fit delta to 1% of the prior range (XSPEC uses it only
  after `xspec.Fit.delta = 0`; with the default `xspec.Fit.delta = 0.01` it
  uses 1% of the parameter value);
- the soft and hard limits to the prior bounds, which may be wider than
  XSPEC's defaults. Keep the bounds where the model is defined: a temperature
  prior starting at 0, for example, lets the sampler evaluate the model below
  the smallest temperature XSPEC allows for it.

Every prior needs finite bounds, because the limits of an XSPEC parameter are
finite: `norm(loc, scale, low, high)` is a Gaussian truncated to
`[low, high]`.

## Parameter names in the results

`solver.parameter_names`, the columns of `posterior_samples` and the
variables of `build_inference_data` name each parameter
`<component>_<position>_<parameter>`, prefixed by `<model_name>_` for a named
model. The position is the 1-based position of the component in the model
expression, so that names are unique: `TBabs_1_nH`, `powerlaw_2_PhoIndex`
(the first powerlaw of `tbabs*(powerlaw + powerlaw)`, which PyXspec calls
`model.powerlaw`), `powerlaw_3_PhoIndex`, or `cluster_bapec_2_kT` for a model
named `cluster`.

The parameters are ordered model by model, in source-number order, then by
XSPEC parameter number, whatever the order of your list; this is the "prior
order" of the arrays in this reference. It matches XSPEC's own order of free
parameters only when the models were defined in source-number order (XSPEC
follows the order of definition), so select columns by name. The names are
read once, when the Solver is built. Plot legends name components as XSPEC
does (`powerlaw_3`), prefixed by the model name for a named model.

## Building priors

::: bsixsa.priors.uniform
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: bsixsa.priors.loguniform
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: bsixsa.priors.norm
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: bsixsa.priors.build_prior
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## The joint prior

`solver.prior` is a
[`MultipleIndependent`][bsixsa.priors.MultipleIndependent] prior: one
independent marginal per fitted parameter, with the maps to and from the unit
cube that samplers use.

::: bsixsa.priors.MultipleIndependent
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: bsixsa.priors.validate_finite_support
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
