r"""SIXSA: simulation-based inference whose answer is checked against exact folds.

History matching first shrinks the prior to the region worth sampling (see
[`bsixsa.restriction`][bsixsa.restriction]). Then each round has two parts.

1. **Learning where to look.** Spectra are simulated (folded through XSPEC,
   with Poisson noise) at parameters spread over the plausible region. Each
   spectrum is compressed into one code per parameter by score compression
   at the maximum likelihood of the observation (or, as an alternative, into
   a short code by an autoencoder), and `n_nde` normalising flows learn
   $p(\text{parameters} \mid \text{code})$ from these pairs. Evaluated at the
   code of the observed spectrum, the equal-weight mixture of the flows is
   the round's *proposal*: a cheap guess of the posterior.
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
from ..config import SIXSA, AutoencoderConfig, FlowConfig
from ..._cube import in_cube, latin_hypercube, to_physical
from ..._torch_utils import torch_threads
from ...restriction import Restriction, run_history_matching
from ...restriction.region import _to_jsonable
from ...solver import FitResults, Solver
from .certify import Certificate, Proposal, certify, rank
from .compressor import Compressor
from .flows import FlowProposal, MixtureProposal, train_flow
from .sampling import importance_sample, resample
from .score import ScoreCompressor
from .truncation import sample_truncated

_LOG = logging.getLogger(__name__)
# Proposal draws per batch of the final importance sample.
_SAMPLING_BATCH = 5000
# C-stat a fold must gain over the fiducial before `recentre` rebuilds the score map (the fit's own stopping gain).
_RECENTRE_GAIN = 0.1


# First training set: _TRAINING_PER_PARAMETER simulations per free parameter
# at d = 6, scaled by (d / 6)^0.6, rounded to 250 and floored at
# _TRAINING_MIN_PER_PARAMETER, as in the reference implementation.
_TRAINING_PER_PARAMETER = 1000.0
_TRAINING_MIN_PER_PARAMETER = 750


def training_set_size(d: int) -> int:
    """Size of the first training set: `d` times a per-parameter budget.

    The budget is 1000 simulations per parameter at `d = 6`, times
    `(d / 6)^0.6`, rounded to 250 and at least 750. The number of channels
    does not enter: the flows learn `p(parameters | code)`, whose size is set
    by `d`, and a short spectrum given fewer simulations converged worse.
    """
    per_param = _TRAINING_PER_PARAMETER * (int(d) / 6.0) ** 0.6
    return int(max(round(per_param / 250) * 250, _TRAINING_MIN_PER_PARAMETER)) * int(d)


def _architecture(flow: FlowConfig) -> str:
    """Short label of a flow's architecture for the logs, e.g. `zuko_nsf h30 t3 b8`."""
    label = f"{flow.model} h{flow.hidden_features} t{flow.num_transforms}"
    return label + (f" b{flow.num_bins}" if flow.model == "zuko_nsf" else "")


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
        compressor: The spectrum compressor: a
            [`ScoreCompressor`][bsixsa.backend.sixsa.score.ScoreCompressor] (the
            default) or the autoencoder
            ([`Compressor`][bsixsa.backend.sixsa.compressor.Compressor]).
        best_round: Round whose gate certificate gave the final proposal.
        exit_reason: `"gate"` (a proposal passed) or `"n_rounds"` (the round
            budget ran out).
        best_fit: Physical parameters of the best fold.
        best_fit_stat: The best C-stat plus the prior term XSPEC's `bayes`
            would add.
        outdir: Output directory, `<outputfiles_basename>/sixsa`.
        rng: NumPy generator seeded from `SIXSA.seed`.
        min_efficiency: The efficiency gate of this run, in percent
            (`SIXSA.min_efficiency`, or its schedule value for the number
            of free parameters).
        target_ess: The effective sample size the final importance sample
            aims for (`SIXSA.target_ess`, or its schedule value).
    """

    config_cls = SIXSA

    def __init__(self, *, solver: Solver, config: SIXSA) -> None:
        super().__init__(solver=solver, config=config)
        self.rng = np.random.default_rng(config.seed)
        self.min_efficiency = config.gate_efficiency(solver.num_parameters)
        self.target_ess = config.final_ess(solver.num_parameters)
        self.outdir = Path(solver.outputfiles_basename) / "sixsa"
        self.restriction: Restriction | None = None
        self.history: list[dict] = []
        self.certificates: list[Certificate] = []
        self.posterior: Certificate | None = None
        self.n_unique: int | None = None
        self.compressor: ScoreCompressor | Compressor | None = None
        self.best_round: int | None = None
        self.exit_reason = ""
        self.best_fit: np.ndarray | None = None
        self.best_fit_stat = np.inf
        # Lowest C-stat folded so far, in the cube: the score compressor's fiducial.
        self._best_u: np.ndarray | None = None
        self._best_cstat = np.inf

    def _to_physical(self, u: np.ndarray) -> np.ndarray:
        """Physical parameters of cube points."""
        return to_physical(self.solver.prior, u)

    def _record(self, u: np.ndarray, theta: np.ndarray, cstat: np.ndarray) -> None:
        """Feed the tracer and track the best folds (with XSPEC's prior term, and bare C-stat)."""
        self.tracer.record(theta, cstat)
        stat = cstat + self.solver.prior.xspec_bayes_contribution(theta)
        if np.isfinite(stat).any() and np.nanmin(stat) < self.best_fit_stat:
            best = int(np.nanargmin(stat))
            self.best_fit, self.best_fit_stat = theta[best].copy(), float(stat[best])
        if np.isfinite(cstat).any() and np.nanmin(cstat) < self._best_cstat:
            best = int(np.nanargmin(cstat))
            self._best_u, self._best_cstat = np.array(u[best], dtype=np.float64), float(cstat[best])

    def _cstat(self, u: np.ndarray) -> np.ndarray:
        """Exact C-stat of cube points, `+inf` outside the closed cube."""
        u = np.asarray(u, dtype=np.float64)
        out = np.full(len(u), np.inf)
        inside = in_cube(u)
        if inside.any():
            theta = self._to_physical(u[inside])
            out[inside] = self.solver.simulate(theta, return_kind="cstat", progress_bar=False)["cstat"]
            self._record(u[inside], theta, out[inside])
        return out

    def _model_counts(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Exact C-stat and expected counts of cube points."""
        theta = self._to_physical(u)
        sim = self.solver.simulate(theta, return_kind="full_model_counts", progress_bar=False)
        cstat = np.asarray(sim["cstat"], dtype=np.float64)
        self._record(u, theta, cstat)
        return cstat, np.asarray(sim["total_model_counts"], dtype=np.float64)

    def _training_data(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Exact C-stat and one Poisson realisation of the model counts at each cube point."""
        cstat, mu = self._model_counts(u)
        # Model counts can dip below zero (e.g. negative-normalisation lines).
        counts = self.rng.poisson(np.clip(mu, 0, None)).astype(np.float32)
        return cstat, counts

    def _draw_region(self, n: int) -> np.ndarray:
        """Flat draws on the history-matching region, or a Latin hypercube of the cube without one."""
        restriction = self.restriction
        if restriction is None:
            return latin_hypercube(self.solver.num_parameters, n, self.rng)
        return restriction.region.sample(n, self.rng, max_candidates=restriction.config.max_candidates)

    def _refits_compressor(self, r: int) -> bool:
        """Whether round `r` fits the compressor: the score map in round 1; the autoencoder per its `refresh`.

        With `recentre`, the score map is rebuilt in a later round once a fold
        has beaten its fiducial by more than `_RECENTRE_GAIN` in C-stat.
        `refresh` refits the autoencoder every round, never again, or up to
        round k.
        """
        settings = self.config.compressor
        if r == 1:
            return True
        if isinstance(settings, AutoencoderConfig):
            if isinstance(settings.refresh, bool):
                return settings.refresh
            return r <= max(settings.refresh, 1)
        return settings.recentre and self._best_cstat < self.compressor.cstat_star - _RECENTRE_GAIN

    def _refit_compressor(self, r: int, cstat: np.ndarray, counts: np.ndarray) -> None:
        """Build the score map at the best fold so far, or refit the autoencoder on the round's spectra."""
        compressor = self.compressor
        if isinstance(compressor, ScoreCompressor):
            if self._best_u is None:
                raise RuntimeError(
                    "SIXSA found no parameter vector with a finite C-stat to build the score compression at: "
                    "the model predicts zero counts where counts were observed everywhere it was folded. "
                    "Check the prior ranges and the model set-up."
                )
            with self.solver.fold_phase("compressor"):
                if self.config.compressor.fiducial == "fit":
                    compressor.fit(self._best_u)
                else:
                    compressor.build(self._best_u)
            self._log_score_map(r)
            return
        restriction = self.restriction
        # Round 1 trains on the spectra that resemble the observation.
        cut = restriction.threshold if r == 1 and restriction is not None else None
        compressor.fit(counts, cstat=cstat, max_cstat=cut)

    def _log_score_map(self, r: int) -> None:
        """Name the parameters the data barely constrain at the fiducial, and the unsettled derivatives.

        A parameter whose marginal Fisher sigma exceeds the prior's own,
        `1/sqrt(12)` in the cube, is prior-dominated there: its code carries
        little, and the flows learn its posterior from the prior's edges.
        """
        compressor, names = self.compressor, np.array(self.solver.parameter_names)
        dominated = names[compressor.sigma > 1.0 / np.sqrt(12.0)].tolist()
        unsettled = names[compressor.unstable].tolist()
        _LOG.info(
            "SIXSA round %d score map: fiducial C-stat %.2f, marginal Fisher sigmas in the cube %s; "
            "prior-dominated: %s; derivative unsettled: %s.",
            r, compressor.cstat_star, " ".join(f"{s:.2g}" for s in compressor.sigma),
            ", ".join(dominated) or "none", ", ".join(unsettled) or "none",
        )

    def _encode(self, r: int, cstat: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Refit the compressor if due; return the training codes and the observation's code."""
        if self._refits_compressor(r):
            self._refit_compressor(r, cstat, counts)
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
        """Train the `n_nde` flows on the `(u, z)` pairs, each frozen at the observation's code."""
        flows, infos = [], []
        with torch_threads(self.config.flow_torch_threads):
            for k, member in enumerate(self.config.flow_members()):
                estimator, info = train_flow(
                    u, z, **member.model_dump(), seed=int(self.rng.integers(2**31 - 1)),
                    log_dir=self.outdir / f"round_{r}" / f"member_{k}",
                )
                flows.append(FlowProposal(estimator, z_obs, name=f"round{r}/member{k}"))
                infos.append(info)
                _LOG.info("SIXSA round %d flow %d/%d (%s): %d epochs, best validation loss %.4f%s",
                          r, k + 1, self.config.n_nde, _architecture(member), info["epochs_trained"],
                          info["best_validation_loss"],
                          "" if info["attempts"] == 1 else f" after {info['attempts'] - 1} diverged attempt(s)")
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
                           khat_threshold=cfg.khat_threshold, min_efficiency=self.min_efficiency)
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
                proposal, self._cstat, target_ess=self.target_ess, max_draws=max_draws, batch=_SAMPLING_BATCH,
                rng=self.rng, khat_threshold=cfg.khat_threshold, min_efficiency=self.min_efficiency,
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
        _LOG.info("SIXSA gate for d=%d: k-hat < %g and efficiency >= %g%% on %d draws; final effective sample "
                  "size %g; %d flows per round.", solver.num_parameters, cfg.khat_threshold, self.min_efficiency,
                  cfg.check_draws, self.target_ess, cfg.n_nde)
        if cfg.history_matching is not None:
            with solver.fold_phase("restriction"):
                self.restriction = run_history_matching(
                    self._cstat, solver.num_parameters, cfg.history_matching,
                    param_names=solver.parameter_names, log=_LOG.info,
                )
            self.restriction.save(self.outdir / "restriction")
        if isinstance(cfg.compressor, AutoencoderConfig):
            self.compressor = Compressor(solver.n_bins, solver.num_parameters,
                                         **cfg.compressor.model_dump(exclude={"refresh"}), seed=cfg.seed, log=_LOG.info)
        else:
            self.compressor = ScoreCompressor(self._model_counts, solver.observed_counts, log=_LOG.info)

        n = min(cfg.n_training or training_set_size(solver.num_parameters), cfg.train_max)
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
                f"needs k-hat < {cfg.khat_threshold} and efficiency >= {self.min_efficiency:g}%), so the final "
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
                f"{cfg.khat_threshold}), efficiency {post.efficiency:.1f}% (needs >= {self.min_efficiency:g}%), "
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
            "gate": {"khat_threshold": self.config.khat_threshold, "min_efficiency": self.min_efficiency,
                     "check_draws": self.config.check_draws, "target_ess": self.target_ess},
            "rounds": [{k: v for k, v in record.items() if k != "certificate"} for record in self.history],
            "config": self.config.model_dump(mode="json"),
        }
        # Non-finite values (e.g. k-hat = inf without a Pareto fit) become null: standard JSON.
        (self.outdir / "summary.json").write_text(json.dumps(_to_jsonable(summary), indent=2))
        np.savez_compressed(self.outdir / "posterior.npz", u=post.u, cstat=post.cstat, log_w=post.log_w,
                            log_w_smoothed=post.log_w_smoothed)
