"""SIXSA v2 versus Nautilus on a reproducible XSPEC fakeit spectrum.

Activate the bsixsa environment, then run:
    python benchmark/sixsa_vs_nautilus_synthetic.py --quick
    python benchmark/sixsa_vs_nautilus_synthetic.py --repeats 3 --strict

Each invocation writes to a fresh directory unless --output is provided.
--exact adds a report-only comparison using the optional auto truncation.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "tests/low_res_data"
TRUTH = np.array([0.2, 1.8, 1e-3])


def setup_xspec(outdir):
    import xspec
    xspec.AllData.clear()
    xspec.AllModels.clear()
    xspec.Xset.chatter = 0
    xspec.Xset.logChatter = 0
    xspec.Fit.statMethod = "cstat"
    xspec.Fit.bayes = "off"
    xspec.Xset.xsect = "vern"
    xspec.Xset.abund = "wilm"
    xspec.Xset.seed = 42
    xspec.Model("tbabs*powerlaw", setPars={i + 1: v for i, v in enumerate(TRUTH)})
    settings = xspec.FakeitSettings(
        response=str(DATA_DIR / "PN.rmf"), arf=str(DATA_DIR / "PN.arf"),
        exposure=50000.0, fileName=str(outdir / "synthetic_powerlaw.fak"),
    )
    xspec.AllData.fakeit(1, [settings], applyStats=True)
    xspec.AllData.ignore("bad")
    xspec.AllData(1).ignore("0.0-0.3 10.0-**")


def prior_spec():
    from bsixsa.priors import loguniform, uniform
    return [("TBabs", "nH", uniform(0, 1)),
            ("powerlaw", "PhoIndex", uniform(1, 3)),
            ("powerlaw", "norm", loguniform(1e-5, 1e-1))]


def summary(solver, fit):
    values = np.asarray(fit.posterior_samples)
    out = dict(log_Z=fit.log_Z, log_Z_err=fit.log_Z_err,
               mean=values.mean(0).tolist(), std=values.std(0).tolist(),
               corr=np.corrcoef(values.T).tolist(), time=fit.time,
               n_likelihood_evaluations=fit.n_likelihood_evaluations,
               fold_counts=solver.fold_counts, best_fit_stat=fit.best_fit_stat,
               restriction_time=fit.restriction_time,
               log_Z_restriction_correction=fit.log_Z_restriction_correction)
    if solver.backend_name == "sixsa":
        b, p = solver.backend, solver.backend.posterior
        out.update(exit_reason=b.exit_reason, certified=p.certified, k_hat=p.k_hat,
                   ess=p.ess, n_unique=p.n_unique, best_round=b.best_round,
                   log_Z_err_reliable=p.log_z_err_reliable,
                   history=[{k: row[k] for k in ("round", "k_hat", "efficiency", "n_train",
                                                "generation", "truncation_source", "truncation")}
                            for row in b.history])
    return out


def reference_best_fit_stat(solver, reference_mean):
    """Refine a reference-posterior centre in the same statistic convention."""
    from scipy.optimize import minimize

    prior = solver.base_prior
    initial = prior.to_unit_cube(np.atleast_2d(reference_mean))[0]

    def objective(u):
        theta = prior.from_unit_cube(np.asarray(u)[None, :])
        cstat = solver.simulate(theta, return_kind="cstat", progress_bar=False)["cstat"]
        return float((cstat + prior.xspec_bayes_contribution_batch(theta))[0])

    result = minimize(objective, initial, method="Powell",
                      bounds=[(0.0, 1.0)] * solver.num_parameters,
                      options=dict(xtol=1e-7, ftol=1e-9, maxiter=100))
    if not result.success:
        raise RuntimeError(f"Reference best-fit refinement failed: {result.message}")
    return float(result.fun)


def run(args):
    from bsixsa import HistoryMatching, Nautilus, SIXSA, Solver
    import torch
    torch.set_num_threads(1)
    outdir = args.output.resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    setup_xspec(outdir)
    summaries, draws = {}, {}
    restriction = None
    for seed in range(args.repeats):
        key = f"nautilus_full_{seed}"
        solver = Solver(prior_spec(), outputfiles_basename=str(outdir / key), n_jobs=args.jobs,
                        backend=Nautilus(trace=False, seed=seed, n_pool=1,
                                         num_live_points=500 if args.quick else 3000))
        fit = solver.run()
        summaries[key], draws[key] = summary(solver, fit), np.asarray(fit.posterior_samples)
        print(f"{key}: log Z={fit.log_Z:.4f}; {fit.n_likelihood_evaluations} folds", flush=True)

    hm = HistoryMatching(seed=0, plot=True)
    if args.quick:
        hm = HistoryMatching(seed=0, plot=True, n_waves=3, simulations_per_parameter=200,
                             hidden=64, epochs=300, patience=15)
    owner = Solver(prior_spec(), outputfiles_basename=str(outdir / "restriction"),
                   n_jobs=args.jobs, backend=Nautilus(trace=False), restrict_prior=hm)
    restriction = owner.restrict()
    summaries["restriction"] = dict(n_likelihood_evaluations=restriction.n_simulations,
                                    seconds=restriction.seconds, prior_mass=restriction.prior_mass,
                                    threshold=restriction.threshold, stop_reason=restriction.stop_reason)

    def infer(key, config):
        import warnings
        from bsixsa import BoxEdgeWarning
        solver = Solver(prior_spec(), outputfiles_basename=str(outdir / key),
                        n_jobs=args.jobs, backend=config, restrict_prior=restriction)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = solver.run()
        summaries[key] = summary(solver, fit)
        summaries[key]["box_edge_warning"] = any(issubclass(w.category, BoxEdgeWarning) for w in caught)
        summaries[key]["warnings"] = [str(w.message) for w in caught]
        draws[key] = np.asarray(fit.posterior_samples)
        print(f"{key}: log Z={fit.log_Z:.4f}; {fit.n_likelihood_evaluations} folds", flush=True)
        return solver

    infer("nautilus_restrict", Nautilus(trace=False, seed=0, n_pool=1,
                                       num_live_points=500 if args.quick else 3000))
    sixsa_kw = dict(seed=0, trace=False, training_n_jobs=args.training_jobs)
    if args.quick:
        sixsa_kw.update(n_nde=2, n_rounds=6, n_training=3000, train_max=12000,
                        training=dict(max_num_epochs=60, training_batch_size=256),
                        compressor=dict(epochs=150, patience=15),
                        target_ess=1000, num_posterior_samples=5000, max_sampling_draws=30000)
    solver = infer("sixsa_restrict", SIXSA(**sixsa_kw))
    if args.exact:
        infer("sixsa_exact", SIXSA(**sixsa_kw, truncated_sampling="auto"))

    full = [summaries[f"nautilus_full_{i}"] for i in range(args.repeats)]
    reference = np.concatenate([draws[f"nautilus_full_{i}"] for i in range(args.repeats)])
    logz_ref = float(np.mean([s["log_Z"] for s in full]))
    error_ref = max(float(np.std([s["log_Z"] for s in full], ddof=1) / np.sqrt(args.repeats))
                    if args.repeats > 1 else 0.0, 0.02)
    ref_mean, ref_std = reference.mean(0), reference.std(0)
    gates = {}
    def check(name, value, bound):
        gates[name] = dict(value=float(value), threshold=float(bound), passed=bool(value <= bound))

    six = summaries["sixsa_restrict"]
    box = summaries["nautilus_restrict"]
    check("G1_logZ_sixsa", abs(six["log_Z"] - logz_ref),
          min(0.25, 3 * np.hypot(six["log_Z_err"], error_ref)))
    check("G2_logZ_box", abs(box["log_Z"] - logz_ref), min(0.25, 3 * np.hypot(error_ref, error_ref)))
    check("G2_box_edge", box["box_edge_warning"], 0)
    for key in ("sixsa_restrict", "nautilus_restrict"):
        v = draws[key]
        check(f"G3_{key}_mean", np.max(np.abs(v.mean(0) - ref_mean) / ref_std), 0.10)
        check(f"G3_{key}_std", np.max(np.abs(v.std(0) / ref_std - 1)), 0.10)
        check(f"G3_{key}_corr", np.max(np.abs(np.corrcoef(v.T) - np.corrcoef(reference.T))), 0.05)
    last = six["history"][-1]
    check("G4_certified", not (six["certified"] and six["exit_reason"] == "gate"
                             and last["k_hat"] < 0.7 and last["efficiency"] >= 5
                             and six["k_hat"] < 0.7), 0)
    low, high = np.quantile(draws["sixsa_restrict"], [0.0015, 0.9985], axis=0)
    check("G5_truth", not np.all((low <= TRUTH) & (TRUTH <= high)), 0)
    # Refine the reference centre with the same prior term as best_fit_stat.
    reference_stat = reference_best_fit_stat(solver, ref_mean)
    check("G6_best_fit", abs(six["best_fit_stat"] - reference_stat), 3.0)
    for key, row in summaries.items():
        if "fold_counts" in row:
            check(f"G7_{key}_folds", abs(row["n_likelihood_evaluations"] - sum(row["fold_counts"].values())), 0)

    report = dict(runs=summaries, reference=dict(log_Z=logz_ref, log_Z_err=error_ref, best_fit_stat=reference_stat,
                                               error_method="replicate scatter / sqrt(R), floor 0.02"),
                  gates=gates, passed=all(g["passed"] for g in gates.values()))
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.generic):
            return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x
    (outdir / "summary.json").write_text(json.dumps(clean(report), indent=2))
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(9, 3), layout="constrained")
    FigureCanvasAgg(fig)
    for j, ax in enumerate(fig.subplots(1, 3)):
        for key in ("nautilus_full_0", "nautilus_restrict", "sixsa_restrict"):
            ax.hist(draws[key][:, j], bins=40, density=True, histtype="step", label=key)
        ax.axvline(TRUTH[j], color="black", linestyle=":")
        ax.set_xlabel(["nH", "PhoIndex", "norm"][j])
    fig.axes[0].legend(fontsize=7)
    fig.savefig(outdir / "posterior_comparison.pdf")
    print(json.dumps(gates, indent=2), flush=True)
    print(f"Report: {outdir / 'summary.json'}", flush=True)
    return 1 if args.strict and not report["passed"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--training-jobs", type=int, default=2)
    parser.add_argument("--output", type=Path, default=REPO / "benchmark" /
                        ("sixsa_v2_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    logging.basicConfig(level=logging.INFO)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
