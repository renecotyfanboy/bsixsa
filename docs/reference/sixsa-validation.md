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

## Perseus with X-IFU

This run is not distributed with bsixsa: it comes from the author's analysis
notebooks. A simulated 100 ks X-IFU spectrum of the Perseus cluster,
`tbabs*bvapec` with 11 free parameters (nH, kT, six abundances, redshift,
velocity broadening and a log-uniform normalisation) on 3197 channels, was fitted
with the default configuration
(`SIXSA(n_nde=5, seed=0, history_matching=HistoryMatching(seed=0))`). The reference is a nessai importance nested
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
shrinks inside the proposal's highest-density region round after round. A loop
that only truncates once k-hat is already low never leaves round 1 on this
problem.
