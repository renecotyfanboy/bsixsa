# SIXSA validation

## Automated checks

`python -m pytest -q` runs the suite without XSPEC: the analytic solver of
`tests/_analytic.py` folds a 64-channel absorbed power law, so every SIXSA
module is exercised end to end against grid quadrature (evidence within its
error bar, posterior moments within a few percent, fold accounting to the
unit), and the certificates, the SIR truncation and the importance sampler are
checked on Gaussian targets whose evidences and leakages are known in closed
form. `tests/integration_xspec/` adds the checks that need a real PyXspec
session (`conda activate bsixsa`).

## SIXSA against Nautilus on a simulated spectrum

`benchmark/sixsa_vs_nautilus_synthetic.py` simulates an XMM-Newton EPIC-pn
spectrum of an absorbed power law (`tbabs*powerlaw`, 3 free parameters, 50 ks,
with the responses of `tests/low_res_data/`), fits it with Nautilus on the full
prior (three seeds, the reference), then with Nautilus and SIXSA on the same
history-matching restriction. It checks that:

- SIXSA's log Z agrees with the reference within three combined standard
  errors (its own and the reference's), the restricted Nautilus run's within
  3√2 reference standard errors, and neither by more than 0.25 (the reference
  error is the scatter of the three seeds divided by √3, at least 0.02);
- the restricted Nautilus run raises no warning about the edges of the
  restricted prior box;
- for both runs, the posterior means agree within 0.1 reference standard
  deviations, the standard deviations within 10% and the correlations within
  0.05;
- SIXSA passed its certification gate, and the central 99.7% of its posterior
  contains the simulated parameters;
- SIXSA's best-fit statistic is within 3 of the reference one, and every run
  reports exactly the folds it made.

`benchmark/sixsa_v2_acceptance_20260916.json` is the report of a full run
(16 September 2026): every check passed (the best-fit check was recomputed
afterwards on the saved spectrum, once a statistic-convention bug in the
benchmark itself was fixed; no fit was rerun, see the file's `audit` section),
SIXSA's log Z differed from the reference by 0.003, and SIXSA used 70 779
folds (plus the 4 500 of the shared history matching) against about 106 000
for each full-prior Nautilus run.

## Perseus with X-IFU

This run is not distributed with bsixsa: it comes from the author's analysis
notebooks. A simulated 100 ks X-IFU spectrum of the Perseus cluster,
`tbabs*bvapec` with 11 free parameters (nH, kT, six abundances, redshift,
velocity broadening and a log-uniform normalisation) on 3197 channels, was fitted
with the default configuration (`SIXSA(n_nde=5, seed=0)` and
`HistoryMatching(seed=0)`). The reference is a nessai importance nested
sampling run (log Z −1531.599 ± 0.010).

| round | training set | k-hat | efficiency |
|---|---:|---:|---:|
| 1 | 33 000 | 4.43 | 0.4% |
| 2 | 41 250 | 0.54 | 8.9% (gate passed) |

The final importance sample reached an ESS of 2138 on 25 000 draws (k-hat
0.21). The run spent about 105 000 folds in the backend (33 000 harvest,
41 250 training draws, 9 300 certification, 21 400 final sampling) plus 11 000
in the two history-matching waves, 116 000 in all, in 11 minutes on a 16-core
laptop, a third of it in the two compressor fits on the CPU. Against nessai:

- log Z −1531.585 ± 0.021, a difference of 0.013;
- every posterior mean within 0.06 standard deviations of the reference;
- standard-deviation ratios between 0.95 and 1.02.

The k-hat trajectory is the diagnosis to read: it falls as the training set
shrinks inside the flow's highest-density region round after round. A loop
that only truncates once k-hat is already low never leaves round 1 on this
problem.

## Reproduce

```bash
conda activate bsixsa
python -m pytest -q
python benchmark/sixsa_vs_nautilus_synthetic.py --quick   # smaller settings, for a quick check
python benchmark/sixsa_vs_nautilus_synthetic.py --strict  # full settings; exit status 1 if a check fails
```

The benchmark writes its report (`summary.json`) and a figure comparing the
marginal posteriors to a new directory under `benchmark/`. The round-by-round
log of a SIXSA run appears through the `bsixsa` logger, and
`<outputfiles_basename>/sixsa/summary.json` holds its numbers.
