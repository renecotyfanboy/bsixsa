# Plotting

```python
from bsixsa.analysis import plot_data

fig = plot_data(min_counts=20)  # the data and the current XSPEC model, before fitting
fig = solver.plot_predictive_coverage("prior")  # prior predictive check
result = solver.run()
fig = solver.plot_predictive_coverage()  # posterior predictive check
```

With several spectra loaded (e.g. MOS1, MOS2 and pn in three data groups),
each figure has one column per spectrum, showing the models folded into that
spectrum; ignored energy ranges are left empty. `spectra` selects some of the
spectra, and a list in `plot_components` draws the components of the listed
models only, which keeps the figure readable next to a background model with
many lines:

```python
fig = solver.plot_predictive_coverage(spectra=[1, 3], plot_components=[""])  # components of the unnamed model
```

::: bsixsa.analysis.plotting
    options:
      show_root_heading: false
      members:
      - plot_data
      - plot_predictive_coverage
      - poisson_error_bars
