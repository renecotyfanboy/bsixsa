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
with a previous version of SIXSA, which certified every flow of a round and
their mixture before gating the best of them, and ran history matching before
the backend; it used the default settings of that version (five masked
autoregressive flows trained for at most 25 epochs, a 5% efficiency gate on
1000 draws, a training set scaled with the number of channels), `n_nde=5`,
`seed=0` and `HistoryMatching(seed=0)`. The defaults now follow the reference
implementation and this run has not been repeated with them. The reference is a nessai importance nested
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

## Score compression against the autoencoder

Score compression replaced the autoencoder as SIXSA's default compressor in
October 2026. The two were compared on two simulated X-IFU spectra, against
nested-sampling references, with paired seeds: for a given seed every variant
shares the same history matching and the same first training set, so the
variants differ only by how the spectra are compressed.

- **Perseus**, `tbabs*bvapec`, 11 free parameters on 2373 channels (the
  special-issue benchmark); the reference pools 10 Nautilus and 10 Nessai runs
  (log Z −1139.442).
- **Power law and one Gaussian line**, `tbabs(powerlaw + zgauss)`, 6 free
  parameters on 2606 channels: a 13 eV wide line with a 1 keV wide prior on
  its energy. The reference is a Nautilus run (log Z −795.640).

Default settings of the time throughout (`n_nde=5`, flows capped at 25
epochs, 100 since: see the next section), seeds
`SIXSA(seed=s, history_matching=HistoryMatching(seed=s))`. Folds count every
exact evaluation of a run, history matching included; ΔlogZ is the mean
difference to the reference with its standard error over the runs. In every
run of the table the posterior means agree with the reference within 0.06
standard deviations, and the standard deviations within 5%.

| target | compressor | runs | rounds | folds | ΔlogZ |
|---|---|---:|---:|---:|---:|
| Perseus | score, fitted fiducial (default) | 5 | 1.4 (1–2) | 80 000 | −0.004 ± 0.003 |
| Perseus | score, best fold as fiducial | 5 | 1.4 (1–2) | 74 000 | −0.008 ± 0.009 |
| Perseus | autoencoder | 5 | 2.8 (2–4) | 176 000 | −0.008 ± 0.001 |
| line | score, fitted fiducial (default) | 2 | 2 | 49 500 | +0.001 ± 0.000 |
| line | score, best fold as fiducial | 2 | 2 | 49 500 | −0.016 ± 0.012 |
| line | autoencoder | 2 | 2 | 49 500 | −0.030 ± 0.006 |

What the runs say:

- **Fewer rounds on Perseus.** With score compression three seeds out of
  five passed the gate in round 1 and the other two in round 2; on the same
  seeds the autoencoder needed two to four rounds, as in the ten runs of the
  special-issue campaign, and 1.3 to 3.9 times more folds on every seed. A
  run took 4–6 minutes instead of 11–20 on a 16-core laptop. On the line
  model the rounds and the folds are the same (the training sets dominate the
  budget), and the run is about twice as fast, since no network is trained
  before the flows.
- **The fiducial: fit or best fold.** History matching alone leaves a poor
  best fold (C-stat 2332–2608 on Perseus and 4199–4898 on the line model,
  against maxima of the likelihood at 2216.55 and 1535.35); the first
  training set brings it to 2238–2310 and 1644–1866. From there, Fisher
  scoring reaches the maximum likelihood in three or four iterations, 70 to
  140 folds, whatever the seed. Both fiducials cost the same; the fitted one
  gave the more accurate and less scattered evidences, which is suggestive
  on seven runs, not conclusive. The fit is the default: it is nearly free,
  and it makes the observation's code the fiducial itself.
- **Re-centring the map on the best fold of each round** gained nothing
  (73 500 folds on the five Perseus seeds, against 74 300 without), so it is
  off by default (`ScoreCompressorConfig(recentre=True)`). It is meant for
  fits that stop in a local optimum. On the emission-line benchmark of the
  special issue, the round-1 fit left narrow lines out (their normalisation
  at the prior floor) in four of five runs with three lines and in every run
  with six to twelve. None of those runs passed the gate in 20 rounds,
  although their proposals did find the missing lines.
- **The prior precision in the codes** changes nothing on these two models,
  whose parameters the data all constrain well (three unregularised Perseus
  runs: 87 000 folds, ΔlogZ +0.005 ± 0.008). It matters for degenerate
  models: on the Tycho benchmark (16 parameters) the Fisher correlation matrix
  at the maximum likelihood has a condition number of 5 × 10⁶, through the
  degeneracy of the second component's normalisation with its iron
  abundance, which `F⁻¹` alone would turn into Poisson noise amplified about
  2000 times in the codes of that direction. The Tycho runs of the next
  section use these codes.

## Flow training budget

Each flow trains until `stop_after_epochs` epochs pass without a validation
improvement, or until `max_num_epochs`. The cap was 25 until October 2026;
in the 25-epoch Perseus runs below every flow reached it while its
validation loss was still improving. The special-issue campaign ran SIXSA
with score compression on 32-core nodes, ten runs per target at 100 epochs,
ten Perseus runs and four Tycho runs at 25, next to ten Nautilus and ten
Nessai runs of the same likelihoods:

- **Perseus**, the special-issue model of the previous section; the
  reference pools the twenty nested-sampling runs (log Z −1139.442).
- **Tycho**, `TBabs(bvnei + bvnei + powerlaw)`, 16 free parameters on 2235
  channels, with the degeneracy described above. The reference is Nautilus
  (log Z −872.862 ± 0.003), for the reason given below.

Folds and ΔlogZ are counted as above; the time is the wall time of a run.

| target | epochs | runs | rounds | folds | time | ΔlogZ |
|---|---:|---:|---:|---:|---:|---:|
| Perseus | 25 | 10 | 1.7 (1–2) | 80 000 | 7.4 min | −0.006 ± 0.007 |
| Perseus | 100 | 10 | 1.3 (1–2) | 63 000 | 13 min | −0.001 ± 0.006 |
| Tycho | 100 | 10 | 3.1 (3–4) | 210 000 | 2.0 h | +0.010 ± 0.008 |

What the runs say:

- **Perseus: the same posterior, an unbiased evidence.** At 100 epochs seven
  runs of ten passed the gate in round 1 (three at 25 epochs), every final
  sample was certified, and the evidence bias fell from −0.006 to −0.001.
  Pooled over the ten runs, the posterior means stay within 0.02 standard
  deviations of the reference at both caps. The longer training costs about
  six minutes per run, more than the 17 000 folds it saves; Nautilus and
  Nessai need 279 000 and 339 000 folds, in 16.5 minutes.
- **Tycho converges at 100 epochs.** The four runs started at 25 epochs were
  stopped after five or six rounds without passing the gate: two still had a
  k-hat above 1, the other two had reached k-hat 0.59–0.77 at 2.6–3.9%
  efficiency, under the 5% floor. At 100 epochs nine runs of ten passed in
  round 3, and the tenth, whose fiducial fit stopped 2.7 in C-stat above the
  maximum likelihood, in round 4. That is 3.6 times fewer folds than Nautilus
  (765 000, in 7.1 hours) and 2.7 times fewer than Nessai (562 000, in 3.9
  hours).
- **The cap still binds.** In round 1, 84% of the Perseus flows and all the
  Tycho flows train for the full 100 epochs; in later rounds 40 to 92% do,
  and the others stop early after 55 epochs or more.
- **Certificates.** Nine Tycho runs of ten end with a certified sample (k-hat
  0.41–0.56, efficiency 5.6–14%). The tenth reached its ESS at 4.4%
  efficiency although its round had passed the gate at 10.8%: on 1000 draws
  the gate overestimates the final efficiency by a factor of 1.2 to 2.5 on
  this model. Its evidence and posterior are as accurate as the others'.
- **Which reference for Tycho.** The two nested samplers disagree. Nessai's
  log Z (−872.881 ± 0.006) lies 0.019 below Nautilus's, and its posterior is
  11% narrower on the degenerate normalisation. Over its ten runs, the
  evidence of a run follows the 99% quantile of that normalisation
  (correlation 0.91), the signature of a sampler that misses part of the
  valley. SIXSA's evidence lies 0.010 above Nautilus's (1.2 standard errors)
  and its width on the normalisation within 2% of Nautilus's. Pooled over
  its ten runs, its means lie within 0.05 standard deviations of Nautilus's
  on both parameters of the degeneracy, and within 0.01 of the pooled
  references on the other fourteen.
