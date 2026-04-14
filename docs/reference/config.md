# Config

Backend configuration objects are Pydantic-backed models used explicitly via `solver.run(config=...)`. Legacy backend kwargs on `solver.run(...)` are no longer supported.

## Usage

```python
from bsixsa import NessaiConfig

result = solver.run(config=NessaiConfig(num_live_points=2_000))
```

The current config reference covers:

- `TraceConfig`
- `NautilusConfig`
- `NessaiConfig`
- `UltranestConfig`
- `EmceeConfig`
- `LevenbergMarquardtConfig`
- `IminuitConfig`

`TraceConfig` documents the shared tracing controls inherited by all backend configs: `trace`, `plot_every`, and `plot_step_percent`. `flush_every` has been removed from the public config API and is now handled internally by the tracer.

::: bsixsa.backend.config
    options:
      show_root_heading: false
      members:
      - TraceConfig
      - NautilusConfig
      - NessaiConfig
      - UltranestConfig
      - EmceeConfig
      - LevenbergMarquardtConfig
      - IminuitConfig
