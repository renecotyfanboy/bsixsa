
## Generic backend config

::: bsixsa
    options:
      show_root_heading: false
      members:
      - BackendConfig

## SIXSA backend

The simulation-based inference backend that gives the package its name. Its
parameters control the sequential inference rounds, the ensemble of neural
density estimators, and the importance-sampling diagnostics.

::: bsixsa
    options:
      show_root_heading: false
      members:
      - SIXSA

## Other backends

::: bsixsa
    options:
      show_root_heading: false
      members:
      - Nautilus
      - Nessai
      - Ultranest
      - LevenbergMarquardt
      - Iminuit
      - Emcee
