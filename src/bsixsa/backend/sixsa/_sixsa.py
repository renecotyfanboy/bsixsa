"""SIXSA v2 orchestration over the Solver's exact folding interface."""

from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .. import register_backend
from ..abc import Backend
from ..config import SIXSA
from ..._cube import child_rng, child_seed, latin_hypercube, make_seed_sequence, to_physical
from ..._torch_utils import torch_threads
from ...solver import FitResults
from . import budget
from ._simulator import CubeSimulator
from .certify import certify_fresh, certify_many, rank_candidates, tier
from .compressor import Compressor
from .flows import FlowProposal, MixtureProposal, member_seeds, train_members
from .sampling import resample, sample_final
from .truncation import sample_truncated

_LOG = logging.getLogger(__name__)


class _BestFitRecorder:
    """Track the best exact fold even when evaluation tracing is disabled."""

    def __init__(self, prior, tracer):
        self.prior, self.tracer = prior, tracer
        self.theta = None
        self.stat = np.inf

    def record(self, theta, cstat, *, trace=True):
        theta = np.asarray(theta, dtype=np.float64)
        values = np.asarray(cstat, dtype=np.float64)
        stats = values + self.prior.xspec_bayes_contribution_batch(theta)
        finite = np.isfinite(stats)
        if finite.any():
            index = int(np.argmin(np.where(finite, stats, np.inf)))
            if stats[index] < self.stat:
                self.stat = float(stats[index])
                self.theta = theta[index].copy()
        if trace and self.tracer is not None:
            self.tracer.record(theta, values)


@register_backend
class SIXSABackend(Backend):
    """Train proposals in parallel and certify them against the exact likelihood."""

    name = "sixsa"
    config_cls = SIXSA

    def __init__(self, *, solver, config):
        super().__init__(solver=solver, config=config)
        self.restriction = solver.restriction
        self.history = []
        self.certificates = []
        self.posterior = None
        self.best_round = None
        self.exit_reason = ""
        self.flows = []
        self.flow = None
        self.compressor = None
        self.fold_counts = {}
        self.timers = {}
        self._seed_seq = make_seed_sequence(config.seed)
        self._sample_rng = child_rng(self._seed_seq, 41)

    def run(self) -> FitResults:
        # Small networks and density evaluations are substantially faster with
        # one intra-op thread; restore the caller's setting even on failure.
        with torch_threads(self.config.threads_per_member):
            return self._run()

    def _run(self) -> FitResults:
        started = time.perf_counter()
        cfg, solver, ss = self.config, self.solver, self._seed_seq
        solver._check_xspec_state()
        before = solver.fold_counts
        recorder = _BestFitRecorder(solver.base_prior, self.tracer)
        sim = CubeSimulator(solver, seed_seq=ss, fold_chunk=cfg.fold_chunk, tracer=recorder)
        region = self.restriction.region if self.restriction is not None else None
        if self.restriction is not None:
            recorder.record(to_physical(solver.base_prior, self.restriction.u), self.restriction.cstat, trace=False)

        n = cfg.n_training or budget.training_set_size(
            sim.d, sim.n_bins, base=cfg.training_set_base,
            per_param_min=cfg.training_set_min_per_param,
        )
        n = min(n, cfg.train_max)
        u = self._draw_region(n, child_rng(ss, 1))
        cstat, counts = sim.counts(u, phase="harvest", key=(2, 0))
        self.compressor = Compressor(
            sim.n_bins, sim.d, **cfg.compressor.model_dump(),
            seed=child_seed(ss, 3), log=_LOG.info,
        )
        gate = dict(khat_threshold=cfg.khat_threshold, khat_safe=cfg.khat_safe,
                    min_efficiency=cfg.min_efficiency)
        fresh_all = []
        best = None
        for r in range(1, cfg.n_rounds + 1):
            t0 = time.perf_counter()
            refresh = cfg.refresh_compressor
            if r == 1 or refresh is True or (type(refresh) is int and r <= refresh):
                cut = self.restriction.threshold if r == 1 and region is not None else None
                self.compressor.fit(counts, cstat=cstat, max_cstat=cut)
            z, generation = self.compressor.encode_with_generation(counts)
            z_obs, obs_generation = self.compressor.encode_with_generation(sim.x_obs)
            self.compressor.check_generation(generation, "training codes")
            self.compressor.check_generation(obs_generation, "observation")
            outside = np.any((z_obs[0] < z.min(axis=0)) | (z_obs[0] > z.max(axis=0)))
            if outside:
                _LOG.warning("Round %s: observation code outside the training range.", r)
            self._time("compressor", t0)

            t0 = time.perf_counter()
            estimators, infos = train_members(
                u, z, nde=cfg.nde.model_dump(), training=cfg.training.model_dump(),
                seeds=member_seeds(ss, r, cfg.n_nde), n_jobs=cfg.training_n_jobs,
                threads_per_member=cfg.threads_per_member, log=_LOG.info,
            )
            self.flows = [
                FlowProposal(est, z_obs[0], name=f"round{r}/member{k}", generation=generation)
                for k, est in enumerate(estimators)
            ]
            self._time("nde", t0)

            t0 = time.perf_counter()
            if len(self.flows) == 1:
                proposal = self.flows[0]
            else:
                mixture = MixtureProposal(self.flows, name=f"round{r}/mixture")
                if cfg.screen_members:
                    screens = certify_many(
                        self.flows, cfg.check_draws, sim, mixture=mixture,
                        rng=child_rng(ss, 20, r), phase="screen", round_index=r, **gate,
                    )
                    selected = rank_candidates(screens, khat_safe=cfg.khat_safe,
                                               tolerance=cfg.screen_tolerance)
                    proposal = screens[selected].proposal
                    self.certificates.extend(c.without_proposal() for c in screens)
                else:
                    proposal = mixture
            fresh = certify_fresh(
                proposal, cfg.check_draws, sim, rng=child_rng(ss, 21, r),
                phase="gate", round_index=r, **gate,
            )
            fresh_all.append(fresh)
            self.certificates.append(fresh.without_proposal())
            best = fresh_all[rank_candidates(fresh_all, khat_safe=cfg.khat_safe,
                                            tolerance=cfg.screen_tolerance)]
            self._time("certify", t0)
            record = dict(fresh.summary(), round=r, n_train=len(u),
                          generation=generation, observation_outside_range=bool(outside),
                          nde_stats=infos, truncation_source=None, truncation=None)
            self.history.append(record)
            _LOG.info("SIXSA round %d: k_hat=%.3f, efficiency=%.1f%%, ESS=%.0f",
                      r, fresh.k_hat, fresh.efficiency, fresh.ess)
            if fresh.passed:
                self.exit_reason = "gate"
                break
            if r == cfg.n_rounds:
                self.exit_reason = "n_rounds"
                break

            t0 = time.perf_counter()
            n = budget.next_training_size(len(u), ramp=cfg.train_ramp, n_max=cfg.train_max)
            source = fresh if tier(fresh, khat_safe=cfg.khat_safe) <= 1 else best
            if tier(source, khat_safe=cfg.khat_safe) <= 1:
                u, info = sample_truncated(
                    source.proposal, n=n, quantile=cfg.truncated_quantile,
                    n_support=cfg.truncated_support_draws, region=region,
                    method=cfg.truncated_sampling, rng=child_rng(ss, 30, r),
                    max_candidates=cfg.truncated_max_candidates, log=_LOG.info,
                )
                record["truncation_source"] = source.name
                record["truncation"] = info
            else:
                u = self._draw_region(n, child_rng(ss, 31, r))
                record["truncation_source"] = "region" if region is not None else "prior"
            self._time("propose", t0)
            cstat, counts = sim.counts(u, phase="training", key=(2, r))

        self.best_round = best.round_index
        self.flow = best.proposal  # Keeps the original conditioning code/generation.
        limit = cfg.max_sampling_draws
        if self.exit_reason != "gate":
            limit = min(limit, 5 * cfg.check_draws)
            warnings.warn("SIXSA reached n_rounds without passing certification.",
                          RuntimeWarning, stacklevel=2)
        n_planned = budget.final_draws(target_ess=cfg.target_ess, ess_fresh=best.ess,
                                      n_fresh=best.n_raw, max_draws=limit)
        t0 = time.perf_counter()
        self.posterior = sample_final(
            self.flow, sim, n_planned=n_planned, target_ess=cfg.target_ess,
            max_draws=limit, rng=child_rng(ss, 40), certified=best.passed,
            khat_threshold=cfg.khat_threshold,
        )
        self._time("sampling", t0)
        rows, n_unique = resample(self.posterior.u, self.posterior.log_w_smoothed,
                                 cfg.num_posterior_samples, self._sample_rng)
        self.posterior.n_unique = n_unique
        samples = self._posterior_dataframe(to_physical(solver.base_prior, rows))
        self.fold_counts = {k: v - before.get(k, 0) for k, v in solver.fold_counts.items()
                            if v != before.get(k, 0)}
        self.timers["folds"] = dict(sim.seconds)
        self.timers["total"] = time.perf_counter() - started
        result = FitResults(
            time=self.timers["total"], posterior_samples=samples,
            n_likelihood_evaluations=sum(self.fold_counts.values()),
            log_Z=self.posterior.log_z, log_Z_err=self.posterior.log_z_err,
            best_fit=pd.Series(recorder.theta, index=solver.parameter_names),
            best_fit_stat=recorder.stat,
        )
        self._save()
        if cfg.plot_diagnostics:
            self.plot_diagnostics()
        for message in self.posterior.warnings:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        return result

    def _draw_region(self, n, rng):
        if self.restriction is None:
            return latin_hypercube(self.solver.num_parameters, n, rng)
        return self.restriction.region.sample(
            n, rng, max_candidates=self.restriction.config.max_candidates,
        )

    def _time(self, name, started):
        self.timers[name] = self.timers.get(name, 0.0) + time.perf_counter() - started

    def sample(self, n: int) -> np.ndarray:
        """Resample the final weighted posterior, returning physical parameters."""
        if self.posterior is None:
            raise RuntimeError("Run SIXSA before drawing posterior samples.")
        rows, _ = resample(self.posterior.u, self.posterior.log_w_smoothed, n, self._sample_rng)
        return to_physical(self.solver.base_prior, rows)

    def weighted_samples(self):
        """Return physical posterior points and normalised PSIS weights."""
        if self.posterior is None:
            raise RuntimeError("Run SIXSA before requesting weighted samples.")
        keep = np.isfinite(self.posterior.log_w_smoothed)
        return (to_physical(self.solver.base_prior, self.posterior.u[keep]),
                np.exp(self.posterior.log_w_smoothed[keep]))

    def _save(self):
        directory = Path(self.solver.outputfiles_basename) / "sixsa"
        directory.mkdir(parents=True, exist_ok=True)
        post = self.posterior
        summary = dict(exit_reason=self.exit_reason, best_round=self.best_round,
                       k_hat=post.k_hat, ess=post.ess, n_unique=post.n_unique,
                       certified=post.certified, log_Z=post.log_z, log_Z_err=post.log_z_err,
                       log_Z_err_reliable=post.log_z_err_reliable,
                       n_raw=post.n_raw, shortfall=post.shortfall,
                       evidence_n_raw=post.evidence_n_raw, evidence_k_hat=post.evidence_k_hat,
                       fold_counts=self.fold_counts, timers=self.timers,
                       history=self.history, config=self.config.model_dump(mode="json"))
        def clean(value):
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(v) for v in value]
            if isinstance(value, np.ndarray):
                return clean(value.tolist())
            if isinstance(value, np.generic):
                return clean(value.item())
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value
        (directory / "summary.json").write_text(json.dumps(clean(summary), indent=2))
        np.savez_compressed(directory / "posterior.npz", u=post.u, cstat=post.cstat,
                            log_w=post.log_w, log_w_smoothed=post.log_w_smoothed)

    def plot_diagnostics(self):
        """Write diagnostic figures; plotting failures do not discard inference."""
        try:
            from .plots import plot_diagnostics
            return plot_diagnostics(self)
        except Exception as exc:
            warnings.warn(f"SIXSA diagnostic plotting failed: {exc}", RuntimeWarning, stacklevel=2)
            return []
