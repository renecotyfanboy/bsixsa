# SIXSA v2 validation — 2026-09-16

Validated with Python 3.12, SBI 0.27.0, PyTorch 2.5.1 and ArviZ 0.23.4.
The implementation follows the new SIXSA reference, including SIR as the default
training-set sampler; it does not claim SIR draws are exactly uniform.

## Automated checks

- Full activated-HEASoft suite: **748 passed, 1 skipped**, 85.45 seconds.
- A benchmark-convention regression test checks the prior term in the best-fit
  comparison.
- Coverage includes closed-cube boundaries, prior transforms, exact evidence
  on analytic targets, history matching, parallel/serial NPE parity, generation
  safety, raw importance weights, mixture allocations, SIR/exact truncation,
  planned evidence versus adaptive posterior top-ups, and end-to-end runs.
- End-to-end variants cover HPD mass at 68/95%, deterministic replay, three
  parameters without restriction, tracing, exhausted budgets and diagnostics.
- Real XSPEC tests check Cash/count consistency and restriction through the
  existing folding pool. The optional low-resolution fixture is skipped because
  its referenced background file is not bundled.
- Scoped Ruff and whitespace checks pass. Documentation builds successfully
  (existing missing-type-annotation warnings remain).

The deliberately small unrestricted three-parameter test reports ESS 1397.6
against a target of 1500 after its 30000-draw budget. That warning is expected;
the posterior/evidence checks still pass. Multiprocess also emits its existing
macOS multithreaded-fork warning. The XSPEC folding architecture was not changed.

## Real XSPEC benchmark

Synthetic PN spectrum, `tbabs*powerlaw`, 50 ks, seed 42, 1942 noticed channels.
Three unrestricted Nautilus runs use 3000 live points and seeds 0/1/2.
Nautilus and SIXSA reuse the same history-matching result: 4500 exact folds,
three waves. SIXSA uses the production defaults, SIR, seed 0, two NPE workers
and four XSPEC folding workers.

| Run | log evidence | Backend folds | Backend time |
|---|---:|---:|---:|
| Nautilus full, seed 0 | −1040.3094 | 106500 | 138.0 s |
| Nautilus full, seed 1 | −1040.3066 | 106800 | 136.3 s |
| Nautilus full, seed 2 | −1040.3193 | 105800 | 150.7 s |
| Nautilus restricted | −1040.3088 | 86300 | 123.4 s |
| SIXSA restricted | −1040.3148 ± 0.0131 | 70779 | 279.5 s |

Restriction cost is shared and excluded from the backend figures above.
SIXSA uses fewer folds on this case, but is **not faster**: autoencoder fitting
takes 164.4 s. These timings are one local run, not a general speed comparison.

SIXSA exits through the gate at round 5: fresh k-hat −0.309, efficiency 38.5%.
Its final k-hat is −0.427, ESS 3686, from 10000 planned raw draws.
Against the combined Nautilus reference:

- Absolute log-evidence difference: 0.0030.
- Largest mean difference: 0.0147 reference standard deviations.
- Largest relative standard-deviation difference: 1.18%.
- Largest correlation difference: 0.00515.
- Truth lies inside the 99.7% intervals; every fold counter reconciles.
- No artificial-box-edge warning from restricted Nautilus.

All corrected G1–G7 checks pass. **Audit note:** the first strict invocation
returned status 1 only because G6 compared a prior-inclusive `best_fit_stat`
with bare C-stat. The benchmark now refines the reference posterior centre
using the same prior-inclusive statistic. Rechecking only G6 on the saved
spectrum required 399 additional folds: reference 2033.76105, SIXSA 2033.76585
(difference 0.00480, threshold 3). No posterior fit or other gate was changed.
The machine-readable report retains the original erroneous G6 under `audit`
in `benchmark/sixsa_v2_acceptance_20260916.json`.

## Reproduce

```bash
conda activate bsixsa
python -m pytest -q
python benchmark/sixsa_vs_nautilus_synthetic.py --repeats 3 --strict
```

The benchmark refuses to overwrite an existing output directory.
`--exact` adds the optional auto-truncation comparison; this acceptance run
used the default SIR path. A passed tail diagnostic does not rule out missed
posterior modes. See the [inference caveats](backend.md) and
[restriction semantics](restriction.md).
