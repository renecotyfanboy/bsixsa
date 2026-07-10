"""SIXSA vs nautilus on a synthetic spectrum with known truth.

Fakes an absorbed power law (tbabs*powerlaw, KNOWN parameters) through the PN
response from tests/low_res_data, then runs the nautilus backend (ground-truth
posterior + evidence) and the SIXSA backend (posterior-mode proposals + final
importance pass, i.e. the new defaults) on the SAME Solver problem.

Reported acceptance metrics:
  * |delta log_Z| vs the combined error (should be within a few sigma),
  * final-pass mean k_hat (< 0.7) and pooled ESS (>= fallback target),
  * per-parameter posterior mean/std vs nautilus AND vs the injected truth,
  * posterior-sample cstat distributions (mean/min) for both backends,
  * an overlaid corner plot.

A cheap truncated-mode arm can be added with --truncated to smoke-test the
legacy proposal path end-to-end.

Usage (needs the activated bsixsa conda env for HEASoft/xspec):
    conda activate bsixsa
    python benchmark/sixsa_vs_nautilus_synthetic.py [--truncated] [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "tests" / "low_res_data"
OUT_DIR = REPO / "benchmark" / "synthetic_result"

TRUTH = {"nH": 0.2, "PhoIndex": 1.8, "norm": 1e-3}
EXPOSURE = 50_000.0  # s
FAKEIT_SEED = 42


TRUTH_BY_POSITION = [TRUTH["nH"], TRUTH["PhoIndex"], TRUTH["norm"]]
PARAM_LABELS = ["nH", "PhoIndex", "norm"]


def setup_xspec():
    """Fakeit a tbabs*powerlaw spectrum with the TRUTH parameters."""
    import xspec

    xspec.AllData.clear()
    xspec.AllModels.clear()
    xspec.Xset.chatter = 0
    xspec.Xset.logChatter = 0
    xspec.Fit.statMethod = "cstat"
    xspec.Fit.bayes = "off"
    xspec.Xset.xsect = "vern"
    xspec.Xset.abund = "wilm"
    xspec.Xset.seed = FAKEIT_SEED  # reproducible fakeit realization

    os.chdir(DATA_DIR)
    xspec.Model(
        "tbabs*powerlaw",
        setPars={
            1: TRUTH["nH"],
            2: TRUTH["PhoIndex"],
            3: TRUTH["norm"],
        },
    )
    settings = xspec.FakeitSettings(
        response="PN.rmf",
        arf="PN.arf",
        exposure=EXPOSURE,
        fileName=str(OUT_DIR / "synthetic_powerlaw.fak"),
    )
    xspec.AllData.fakeit(1, [settings], applyStats=True)
    xspec.AllData.ignore("bad")
    xspec.AllData(1).ignore("0.0-0.3 10.0-**")


def prior_spec():
    from bsixsa.priors import loguniform, uniform

    return [
        ("TBabs", "nH", uniform(0.0, 1.0)),
        ("powerlaw", "PhoIndex", uniform(1.0, 3.0)),
        ("powerlaw", "norm", loguniform(1e-5, 1e-1)),
    ]


def posterior_cstats(solver, samples, n=500, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(samples), size=min(n, len(samples)), replace=False)
    subset = np.asarray(samples)[idx]
    out = solver.simulate(subset, return_kind="cstat", progress_bar=False)
    return np.asarray(out["cstat"], dtype=np.float64)


def run_nautilus(quick):
    from bsixsa import Nautilus, Solver

    solver = Solver(
        prior_spec(),
        outputfiles_basename=str(OUT_DIR / "nautilus"),
        overwrite=True,
        backend=Nautilus(
            trace=False, num_live_points=(500 if quick else 2000)
        ),
    )
    result = solver.run()
    return solver, result


def run_sixsa(proposal_mode, quick):
    from bsixsa import SIXSA, Solver

    if quick:
        rounds = [1024] * 3
        extra = dict(final_is_target_ess=2000.0, final_is_fallback_ess=500.0)
    else:
        # Reference-faithful schedule for 3 free parameters: a large LHS-scale
        # first round pre-filtered to its best-cstat half (the dev comparison
        # harness's filter_round0_fraction=0.5), then pools sized so the kept
        # count per round ~ dev's 500/param.
        rounds = [16384, 3072, 3072, 3072, 3072]
        extra = dict(filter_round0_fraction=0.5)
    solver = Solver(
        prior_spec(),
        outputfiles_basename=str(OUT_DIR / f"sixsa_{proposal_mode}"),
        overwrite=True,
        backend=SIXSA(
            num_simulations_per_round=rounds,
            proposal_mode=proposal_mode,
            plot_diagnostics=False,
            **extra,
        ),
    )
    result = solver.run()
    return solver, result


def summarize(name, result, samples, cstats, extra=None):
    # The backends label posterior columns differently, so moments are keyed
    # positionally by PARAM_LABELS.
    values = np.asarray(samples)
    stats = {
        "backend": name,
        "log_Z": float(result.log_Z),
        "log_Z_err": float(result.log_Z_err),
        "n_likelihood_evaluations": int(result.n_likelihood_evaluations),
        "time_s": float(result.time),
        "posterior_mean": {
            label: float(values[:, i].mean()) for i, label in enumerate(PARAM_LABELS)
        },
        "posterior_std": {
            label: float(values[:, i].std()) for i, label in enumerate(PARAM_LABELS)
        },
        "cstat_mean": float(cstats.mean()),
        "cstat_min": float(cstats.min()),
        "cstat_p95": float(np.percentile(cstats, 95)),
    }
    if extra:
        stats.update(extra)
    return stats


def corner_plot(chains, path):
    try:
        import pandas as pd
        from chainconsumer import Chain, ChainConsumer

        consumer = ChainConsumer()
        for label, samples in chains:
            frame = pd.DataFrame(np.asarray(samples), columns=PARAM_LABELS)
            consumer.add_chain(Chain(samples=frame, name=label))
        fig = consumer.plotter.plot()
        fig.savefig(path, bbox_inches="tight")
        print(f"corner plot -> {path}")
    except Exception as exc:  # plotting must never sink the metrics
        print(f"corner plot failed: {exc!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="reduced budgets")
    parser.add_argument(
        "--truncated",
        action="store_true",
        help="also smoke-test the legacy truncated proposal mode",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_xspec()

    all_stats = []

    print("=== nautilus (ground truth) ===", flush=True)
    nautilus_solver, nautilus_result = run_nautilus(args.quick)
    nautilus_samples = nautilus_result.posterior_samples
    nautilus_cstats = posterior_cstats(nautilus_solver, nautilus_samples)
    stats = summarize("nautilus", nautilus_result, nautilus_samples, nautilus_cstats)
    print(json.dumps(stats, indent=2), flush=True)
    all_stats.append(stats)

    print("=== SIXSA (posterior mode, new defaults) ===", flush=True)
    sixsa_solver, sixsa_result = run_sixsa("posterior", args.quick)
    sixsa_samples = sixsa_result.posterior_samples
    sixsa_cstats = posterior_cstats(sixsa_solver, sixsa_samples)
    final_info = sixsa_solver.backend.final_is_info or {}
    stats = summarize(
        "sixsa_posterior",
        sixsa_result,
        sixsa_samples,
        sixsa_cstats,
        extra={
            "final_is_mean_khat": final_info.get("mean_khat"),
            "final_is_ess": final_info.get("ess"),
            "final_is_rounds": final_info.get("n_rounds"),
            "final_is_reached_target": final_info.get("reached_target_ess"),
            "best_round": sixsa_solver.backend.best_round,
            "in_loop_khat_per_round": [
                r["k_hat"] for r in sixsa_solver.backend.history
            ],
        },
    )
    print(json.dumps(stats, indent=2), flush=True)
    all_stats.append(stats)

    if args.truncated:
        print("=== SIXSA (truncated mode, legacy smoke test) ===", flush=True)
        trunc_solver, trunc_result = run_sixsa("truncated", True)
        trunc_samples = trunc_result.posterior_samples
        trunc_cstats = posterior_cstats(trunc_solver, trunc_samples)
        info = trunc_solver.backend.final_is_info or {}
        stats = summarize(
            "sixsa_truncated",
            trunc_result,
            trunc_samples,
            trunc_cstats,
            extra={
                "final_is_mean_khat": info.get("mean_khat"),
                "final_is_ess": info.get("ess"),
            },
        )
        print(json.dumps(stats, indent=2), flush=True)
        all_stats.append(stats)

    # ---- acceptance summary ----
    delta_logz = all_stats[1]["log_Z"] - all_stats[0]["log_Z"]
    sigma = float(
        np.hypot(all_stats[0]["log_Z_err"], all_stats[1]["log_Z_err"])
    )
    print("\n=== acceptance summary ===")
    print(
        f"delta log_Z (SIXSA - nautilus) = {delta_logz:+.3f} "
        f"(combined err {sigma:.3f} -> {abs(delta_logz) / sigma if sigma > 0 else float('inf'):.1f} sigma)"
    )
    print(
        f"final-IS mean k_hat = {all_stats[1]['final_is_mean_khat']}, "
        f"pooled ESS = {all_stats[1]['final_is_ess']}"
    )
    for i, label in enumerate(PARAM_LABELS):
        m_ref = all_stats[0]["posterior_mean"][label]
        s_ref = all_stats[0]["posterior_std"][label]
        m = all_stats[1]["posterior_mean"][label]
        s = all_stats[1]["posterior_std"][label]
        true_value = TRUTH_BY_POSITION[i]
        print(
            f"  {label:<12} mean {m:+.4g} vs {m_ref:+.4g} "
            f"({(m - m_ref) / s_ref if s_ref > 0 else float('nan'):+.2f} sigma_ref), "
            f"std ratio {s / s_ref if s_ref > 0 else float('nan'):.2f}, "
            f"truth {true_value:+.4g} "
            f"({(m - true_value) / s if s > 0 else float('nan'):+.2f} sigma_sixsa)"
        )
    print(
        f"cstat mean/min: SIXSA {all_stats[1]['cstat_mean']:.1f}/"
        f"{all_stats[1]['cstat_min']:.1f} vs nautilus "
        f"{all_stats[0]['cstat_mean']:.1f}/{all_stats[0]['cstat_min']:.1f}"
    )

    with open(OUT_DIR / "summary.json", "w") as file:
        json.dump(all_stats, file, indent=2, default=float)
    print(f"metrics -> {OUT_DIR / 'summary.json'}")

    corner_plot(
        [("nautilus", nautilus_samples), ("SIXSA", sixsa_samples)],
        OUT_DIR / "corner_overlay.pdf",
    )


if __name__ == "__main__":
    main()
