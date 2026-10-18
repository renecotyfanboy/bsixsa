r"""SIXSA: simulation-based inference whose answer is checked against exact folds.

History matching first shrinks the prior to the region worth sampling (see
[`bsixsa.restriction`][bsixsa.restriction]). Then each round has two parts.

1. **Learning where to look.** Spectra are simulated (folded through XSPEC,
   with Poisson noise) at parameters spread over the plausible region. An
   autoencoder compresses each spectrum into a short code, and `n_nde`
   normalising flows learn $p(\text{parameters} \mid \text{code})$ from these
   pairs. Evaluated at the code of the observed spectrum, the equal-weight
   mixture of the flows is the round's *proposal*: a cheap guess of the
   posterior.
2. **Checking the guess exactly.** Draws from the proposal are folded
   through XSPEC and weighted by $e^{-C/2} / q$, the exact likelihood over the
   proposal density. The Pareto k-hat of these weights says whether the guess
   is good enough for importance sampling. If it is not, the next round
   simulates only where the proposal puts its mass, with a larger training
   set.

Once a proposal passes, or the rounds run out, the posterior is an importance
sample from the best one. Parameters live in the prior's unit cube, where the
prior density is 1.

Glossary:

- *fold*: one exact evaluation of the model through XSPEC.
- *gate*: certifying the round's proposal on `check_draws` draws; it passes
  when k-hat is below `khat_threshold` and the efficiency is at least
  `min_efficiency`.
- *certificate*: the draws, weights and diagnostics of one such check
  ([`Certificate`][bsixsa.backend.sixsa.certify.Certificate]).
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..abc import Backend
from ..config import SIXSA
from ..._cube import in_cube, latin_hypercube, to_physical
from ...restriction import Restriction, run_history_matching
from ...restriction.region import _to_jsonable
from ...solver import FitResults, Solver
from .certify import Certificate, Proposal, certify, rank
from .compressor import Compressor
from .flows import FlowProposal, MixtureProposal, train_flow
from .sampling import importance_sample, resample
from .truncation import sample_truncated

_LOG = logging.getLogger(__name__)
# Proposal draws per batch of the final importance sample.
_SAMPLING_BATCH = 5000


def training_set_size(d: int, n_bins: int) -> int:
    """Size of the first training set: `d` times a per-parameter budget.

    The budget is 2000 simulations per parameter for `d = 6` on 2500 channels,
    and grows with the number of parameters and of channels.
    """
    dimension_factor = (d / 6.0) ** 0.6
    channel_factor = min(max((n_bins / 2500.0) ** 0.25, 0.5), 3.0)
    per_param = 2000.0 * dimension_factor * channel_factor
    return int(max(round(per_param / 250) * 250, 750)) * int(d)


class SIXSABackend(Backend):
    """Train flow proposals on simulated spectra and certify them against exact folds.

    After `solver.run()`, `solver.backend` holds the attributes below.

    Attributes:
        restriction: The history-matching
            [`Restriction`][bsixsa.restriction.Restriction], or `None` without
            history matching.
        posterior: The final weighted sample
            ([`Certificate`][bsixsa.backend.sixsa.certify.Certificate]).
        n_unique: Number of distinct rows in the equal-weight posterior samples.
        history: One dict per round: the gate certificate's diagnostics plus
            the training, compressor and truncation records.
        certificates: The gate certificate of every round, in order.
        compressor: The spectrum autoencoder
            ([`Compressor`][bsixsa.backend.sixsa.compressor.Compressor]).
        best_round: Round whose gate certificate gave the final proposal.
        exit_reason: `"gate"` (a proposal passed) or `"n_rounds"` (the round
            budget ran out).
        best_fit: Physical parameters of the best fold.
        best_fit_stat: The best C-stat plus the prior term XSPEC's `bayes`
            would add.
        outdir: Output directory, `<outputfiles_basename>/sixsa`.
        rng: NumPy generator seeded from `SIXSA.seed`.
    """

    config_cls = SIXSA

    def __init__(self, *, solver: Solver, config: SIXSA) -> None:
        super().__init__(solver=solver, config=config)
        self.rng = np.random.default_rng(config.seed)
        self.outdir = Path(solver.outputfiles_basename) / "sixsa"
        self.restriction: Restriction | None = None
        self.history: list[dict] = []
        self.certificates: list[Certificate] = []
        self.posterior: Certificate | None = None
        self.n_unique: int | None = None
        self.compressor: Compressor | None = None
        self.best_round: int | None = None
        self.exit_reason = ""
        self.best_fit: np.ndarray | None = None
        self.best_fit_stat = np.inf

    def _to_physical(self, u: np.ndarray) -> np.ndarray:
        """Physical parameters of cube points."""
        return to_physical(self.solver.prior, u)

    def _record(self, theta: np.ndarray, cstat: np.ndarray) -> None:
        """Feed the tracer and track the best fold."""
        self.tracer.record(theta, cstat)
        stat = cstat + self.solver.prior.xspec_bayes_contribution(theta)
        if np.isfinite(stat).any() and np.nanmin(stat) < self.best_fit_stat:
            best = int(np.nanargmin(stat))
            self.best_fit, self.best_fit_stat = theta[best].copy(), float(stat[best])

    def _cstat(self, u: np.ndarray) -> np.ndarray:
        """Exact C-stat of cube points, `+inf` outside the closed cube."""
        u = np.asarray(u, dtype=np.float64)
        out = np.full(len(u), np.inf)
        inside = in_cube(u)
        if inside.any():
            theta = self._to_physical(u[inside])
            out[inside] = self.solver.simulate(theta, return_kind="cstat", progress_bar=False)["cstat"]
            self._record(theta, out[inside])
        return out

    def _training_data(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Exact C-stat and one Poisson realisation of the model counts at each cube point."""
        theta = self._to_physical(u)
        sim = self.solver.simulate(theta, return_kind="full_model_counts", progress_bar=False)
        cstat = np.asarray(sim["cstat"], dtype=np.float64)
        self._record(theta, cstat)
        # Model counts can dip below zero (e.g. negative-normalisation lines).
        counts = self.rng.poisson(np.clip(sim["total_model_counts"], 0, None)).astype(np.float32)
        return cstat, counts

    def _draw_region(self, n: int) -> np.ndarray:
        """Flat draws on the history-matching region, or a Latin hypercube of the cube without one."""
        restriction = self.restriction
        if restriction is None:
            return latin_hypercube(self.solver.num_parameters, n, self.rng)
        return restriction.region.sample(n, self.rng, max_candidates=restriction.config.max_candidates)

    def _refits_compressor(self, r: int) -> bool:
        """Whether round `r` refits the compressor (`refresh_compressor`: every round, never again, or up to round k)."""
        refresh = self.config.refresh_compressor
        if isinstance(refresh, bool):
            return r == 1 or refresh
        return r <= max(refresh, 1)

    def _encode(self, r: int, cstat: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Refit the compressor if due; return the training codes and the observation's code."""
        restriction = self.restriction
        if self._refits_compressor(r):
            # Round 1 trains on the spectra that resemble the observation.
            cut = restriction.threshold if r == 1 and restriction is not None else None
            self.compressor.fit(counts, cstat=cstat, max_cstat=cut)
        z = self.compressor.encode(counts)
        z_obs = self.compressor.encode(self.solver.observed_counts)[0]
        if np.any((z_obs < z.min(axis=0)) | (z_obs > z.max(axis=0))):
            _LOG.warning(
                "Round %d: the observed spectrum's code lies outside the range of the training spectra, so the "
                "flows must extrapolate. If this persists over rounds, the model or prior may not reproduce the "
                "data.", r,
            )
        return z, z_obs

    def _train_flows(self, r: int, u: np.ndarray, z: np.ndarray,
                     z_obs: np.ndarray) -> tuple[list[FlowProposal], list[dict]]:
        """Train `n_nde` flows on the `(u, z)` pairs, each frozen at the observation's code."""
        flows, infos = [], []
        for k in range(self.config.n_nde):
            estimator, info = train_flow(
                u, z, **self.config.flow.model_dump(), seed=int(self.rng.integers(2**31 - 1)),
                log_dir=self.outdir / f"round_{r}" / f"member_{k}",
            )
            flows.append(FlowProposal(estimator, z_obs, name=f"round{r}/member{k}"))
            infos.append(info)
        return flows, infos

    def _warn_if_region_cuts_posterior(self, r: int, cert: Certificate) -> None:
        """Warn when 99.9% of the posterior weight reaches within one posterior span of the region's threshold.

        The flows only see the history-matching region, so posterior mass
        beyond it would go unnoticed.
        """
        restriction = self.restriction
        if restriction is None or not np.isfinite(cert.log_w_smoothed).any():
            return
        order = np.argsort(cert.cstat)
        cumulative = np.cumsum(np.exp(cert.log_w_smoothed[order]))
        reach = float(cert.cstat[order][np.searchsorted(cumulative, 0.999)])
        margin = restriction.threshold - reach
        if margin < restriction.span:
            _LOG.warning(
                "Round %d: 99.9%% of the posterior weight lies below C-stat %.1f, only %.1f below the region "
                "threshold %.1f (less than one posterior span, %.1f); the history-matching region may cut into "
                "the posterior. Rerun with a wider restriction (larger HistoryMatching(target_spans=...)) or "
                "SIXSA(history_matching=None) to check.",
                r, reach, margin, restriction.threshold, restriction.span,
            )

    def _round(self, r: int, u: np.ndarray, cstat: np.ndarray, counts: np.ndarray) -> dict:
        """One round: encode, train the flows, and certify their equal-weight mixture."""
        cfg = self.config
        z, z_obs = self._encode(r, cstat, counts)
        flows, infos = self._train_flows(r, u, z, z_obs)
        proposal = flows[0] if len(flows) == 1 else MixtureProposal(flows, name=f"round{r}/mixture")
        with self.solver.fold_phase("gate"):
            cert = certify(proposal, cfg.check_draws, self._cstat, rng=self.rng, round_index=r,
                           khat_threshold=cfg.khat_threshold, min_efficiency=cfg.min_efficiency)
        self.certificates.append(cert)
        _LOG.info("SIXSA round %d: %s k_hat=%.3f, efficiency=%.1f%%, ESS=%.0f, best fold %.1f",
                  r, cert.name, cert.k_hat, cert.efficiency, cert.ess, float(np.min(cert.cstat)))
        self._warn_if_region_cuts_posterior(r, cert)
        return dict(cert.summary(), round=r, n_train=len(u), certificate=cert,
                    compressor=self.compressor.history[-1], nde=infos, truncation=None)

    def _truncated_training_set(self, record: dict) -> tuple[np.ndarray, dict]:
        """Draw the next, larger training set inside the round proposal's highest-density region.

        Done whatever the proposal's k-hat: a poor k-hat condemns the
        proposal's tails, not its high-density region.
        """
        cfg, restriction = self.config, self.restriction
        n = min(int(round(record["n_train"] * cfg.train_ramp)), cfg.train_max)
        u, info = sample_truncated(
            record["certificate"].proposal, n=n, quantile=cfg.truncated_quantile,
            n_support=cfg.truncated_support_draws, oversampling=cfg.truncated_oversampling,
            rng=self.rng, region=None if restriction is None else restriction.region,
        )
        _LOG.info("SIXSA round %d truncation: %s", record["round"], info)
        return u, info

    def _final_sample(self, proposal: Proposal, max_draws: int) -> np.ndarray:
        """Importance-sample `proposal` into `self.posterior`; return equal-weight cube rows."""
        cfg = self.config
        with self.solver.fold_phase("sampling"):
            self.posterior = importance_sample(
                proposal, self._cstat, target_ess=cfg.target_ess, max_draws=max_draws, batch=_SAMPLING_BATCH,
                rng=self.rng, khat_threshold=cfg.khat_threshold, min_efficiency=cfg.min_efficiency,
            )
        rows, self.n_unique = resample(self.posterior.u, self.posterior.log_w_smoothed,
                                       cfg.num_posterior_samples, self.rng)
        return rows

    def run(self) -> FitResults:
        """Run the SIXSA rounds, then importance-sample the best proposal.

        Returns:
            A [`FitResults`][bsixsa.solver.FitResults] with the posterior
            samples, the log evidence and the best fit.

        Warns:
            RuntimeWarning: If no round passed the gate, or if the final
                importance sample is not certified.
        """
        started = time.perf_counter()
        cfg, solver = self.config, self.solver
        folds_before = sum(solver.fold_counts.values())
        if cfg.history_matching is not None:
            with solver.fold_phase("restriction"):
                self.restriction = run_history_matching(
                    self._cstat, solver.num_parameters, cfg.history_matching,
                    param_names=solver.parameter_names, log=_LOG.info,
                )
            self.restriction.save(self.outdir / "restriction")
        self.compressor = Compressor(solver.n_bins, solver.num_parameters, **cfg.compressor.model_dump(),
                                     seed=cfg.seed, log=_LOG.info)

        n = min(cfg.n_training or training_set_size(solver.num_parameters, solver.n_bins), cfg.train_max)
        u = self._draw_region(n)
        with solver.fold_phase("harvest"):
            cstat, counts = self._training_data(u)
        for r in range(1, cfg.n_rounds + 1):
            record = self._round(r, u, cstat, counts)
            self.history.append(record)
            if record["passed"] or r == cfg.n_rounds:
                break
            u, record["truncation"] = self._truncated_training_set(record)
            with solver.fold_phase("training"):
                cstat, counts = self._training_data(u)
        self.exit_reason = "gate" if self.history[-1]["passed"] else "n_rounds"

        gates = [record["certificate"] for record in self.history]
        best = gates[rank(gates)]
        self.best_round = best.round_index
        max_draws = cfg.max_sampling_draws
        if self.exit_reason != "gate":
            # Without a certified proposal, the final sample gets a small budget.
            max_draws = min(max_draws, 5 * cfg.check_draws)
            warnings.warn(
                f"SIXSA used all n_rounds={cfg.n_rounds} rounds without a proposal passing certification (best "
                f"round {self.best_round}: k-hat {best.k_hat:.2f}, efficiency {best.efficiency:.1f}%; the gate "
                f"needs k-hat < {cfg.khat_threshold} and efficiency >= {cfg.min_efficiency}%), so the final "
                f"importance sample is capped at {max_draws} draws (min(max_sampling_draws, "
                "5 x check_draws)). Increase n_rounds or train_max, check the priors and the model, and check "
                f"the per-round k_hat in {self.outdir / 'summary.json'}.",
                RuntimeWarning, stacklevel=3,  # points at the caller of Solver.run
            )
        rows = self._final_sample(best.proposal, max_draws)
        post = self.posterior
        if not post.passed:
            if self.exit_reason == "gate":
                advice = (f"the gate passed on {cfg.check_draws} draws but the final sample did not; raise "
                          "check_draws for a stricter gate, or cross-check with a nested sampler.")
            else:
                advice = "increase n_rounds, check the priors and the model, or cross-check with a nested sampler."
            warnings.warn(
                f"SIXSA final importance sample not certified: k-hat {post.k_hat:.2f} (needs < "
                f"{cfg.khat_threshold}), efficiency {post.efficiency:.1f}% (needs >= {cfg.min_efficiency}%), "
                f"ESS {post.ess:.0f} from {post.n_raw} draws. Treat the posterior and log Z as approximate: "
                + advice,
                RuntimeWarning, stacklevel=3,
            )

        result = FitResults(
            time=time.perf_counter() - started,
            posterior_samples=self._posterior_dataframe(self._to_physical(rows)),
            n_likelihood_evaluations=sum(solver.fold_counts.values()) - folds_before,
            log_Z=post.log_z, log_Z_err=post.log_z_err,
            best_fit=pd.Series(self.best_fit, index=solver.parameter_names), best_fit_stat=self.best_fit_stat,
        )
        self._save(result)
        return result

    def sample(self, n: int) -> np.ndarray:
        """Resample the final weighted posterior, returning physical parameters."""
        if self.posterior is None:
            raise RuntimeError("No SIXSA posterior yet: call solver.run() first.")
        rows, _ = resample(self.posterior.u, self.posterior.log_w_smoothed, n, self.rng)
        return self._to_physical(rows)

    def weighted_samples(self) -> tuple[np.ndarray, np.ndarray]:
        """Physical posterior points and their normalised PSIS weights."""
        if self.posterior is None:
            raise RuntimeError("No SIXSA posterior yet: call solver.run() first.")
        keep = np.isfinite(self.posterior.log_w_smoothed)
        return self._to_physical(self.posterior.u[keep]), np.exp(self.posterior.log_w_smoothed[keep])

    def _save(self, result: FitResults) -> None:
        """Write `summary.json` and the weighted posterior draws under the output directory."""
        self.outdir.mkdir(parents=True, exist_ok=True)
        post = self.posterior
        summary = {
            "exit_reason": self.exit_reason, "best_round": self.best_round, "time": result.time,
            "n_likelihood_evaluations": result.n_likelihood_evaluations, "fold_counts": self.solver.fold_counts,
            "posterior": dict(post.summary(), n_unique=self.n_unique),
            "rounds": [{k: v for k, v in record.items() if k != "certificate"} for record in self.history],
            "config": self.config.model_dump(mode="json"),
        }
        # Non-finite values (e.g. k-hat = inf without a Pareto fit) become null: standard JSON.
        (self.outdir / "summary.json").write_text(json.dumps(_to_jsonable(summary), indent=2))
        np.savez_compressed(self.outdir / "posterior.npz", u=post.u, cstat=post.cstat, log_w=post.log_w,
                            log_w_smoothed=post.log_w_smoothed)
