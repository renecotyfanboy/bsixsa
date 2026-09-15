"""SIXSA backend: persistent replace-worst loop with defensive-IS gates.

Implements the SIXSA specification (code version 1.11.1) on the `bsixsa`
Solver/prior/XSPEC boundaries, in the unit-Gaussian latent ``N(0, I)``.

Each round trains an ensemble of NPE_C flows SEQUENTIALLY on a single persistent
training set ``T``: member ``k`` trains on ``T``, draws a flat sub-batch from its
truncated proposal, and replace-worst-merges it into ``T`` before member ``k+1``
trains -- so members differ by initialization AND by their (progressively
refined) training set. Convergence is gated by defensive-mixture importance
sampling (see ``_importance``); the headline posterior / khat / ESS / log_Z come
from the banked pooled multiple-IS estimator, followed by an optional emulator
pass. The flows are never trusted directly -- the defensive-IS khat is the
arbiter.
"""

import contextlib
import functools
import math
import os
import time
import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch
from sbi.inference import EnsemblePosterior, NPE_C
from sbi.neural_nets import posterior_nn

from .. import register_backend
from ..abc import Backend
from ..config import SIXSA
from ...convenience import available_cpu_count
from . import plots
from ._buffer import TrainingBuffer
from ._emulator import emulator_importance_pass, train_likelihood_emulator
from ._importance import DefensiveISBank, defensive_is_diagnostics, run_check, run_gate
from ._proposal import build_truncated_proposal, sample_flat
from .embedding.nets import EmbeddingNet, FCEmbeddingNet

if TYPE_CHECKING:
    from ...solver import FitResults, Solver

# Code version reported on the first log line (spec logging contract).
SIXSA_VERSION = "1.11.1"

# Density-estimator defaults mirroring the reference nde_kwargs: a MAF with wide
# transforms and independent z-scoring of theta and x.
DEFAULT_NDE_KWARGS = {
    "model": "maf",
    "hidden_features": 100,
    "num_transforms": 10,
    "z_score_theta": "independent",
    "z_score_x": "independent",
}


def merge_nde_kwargs(user_kwargs):
    """Merge user `nde_kwargs` over `DEFAULT_NDE_KWARGS` (user keys win)."""
    return {**DEFAULT_NDE_KWARGS, **(user_kwargs or {})}


def _unit_gaussian_prior(ndim):
    """Standard-normal prior ``N(0, I)`` over the SIXSA latent space.

    Latent draws map to physical parameters by ``solver.prior.from_unit_gaussian``
    (``ppf(Phi(z))`` per marginal); log-uniform parameters are a per-parameter
    `loguniform` prior-object choice handled there.
    """
    return torch.distributions.MultivariateNormal(torch.zeros(ndim), torch.eye(ndim))


def patch_sample_no_pbar(posterior):
    """Default the posterior's `sample` to no progress bar."""
    original = posterior.sample

    @functools.wraps(original)
    def sample(*args, _original=original, **kwargs):
        kwargs.setdefault("show_progress_bars", False)
        return _original(*args, **kwargs)

    posterior.sample = sample
    return posterior


def work_quiet(func):
    """Run a function with stdout redirected to devnull."""

    def quiet_func(*args, **kwargs):
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            return func(*args, **kwargs)

    return quiet_func


@work_quiet
def training_job(
    number,
    theta,
    x,
    *,
    embedding_net,
    round_number,
    training_kwargs=None,
    nde_kwargs=None,
    num_threads=None,
):
    """Train one ensemble member (NPE_C, plain MLE loss, fresh weights).

    Trains on the STORED Poisson realization ``x`` (no per-member Poisson --
    the single realization is drawn once per theta at simulation time). Returns
    ``(posterior, stats)``.
    """
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))

    theta_t = torch.as_tensor(np.asarray(theta), dtype=torch.float32)
    x_t = torch.as_tensor(np.asarray(x), dtype=torch.float32)

    if embedding_net is not None:
        net = embedding_net.build(x_t.shape[1])
    else:
        net = FCEmbeddingNet().build(x_t.shape[1])
    prior_sbi = _unit_gaussian_prior(theta_t.shape[-1])
    build_fun = posterior_nn(embedding_net=net, **(nde_kwargs or DEFAULT_NDE_KWARGS))
    inference = NPE_C(prior=prior_sbi, density_estimator=build_fun, device="cpu")

    training_kwargs = {} if training_kwargs is None else dict(training_kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        t0 = time.time()
        inference.append_simulations(theta_t, x_t).train(**training_kwargs)
        training_time = time.time() - t0
        posterior = inference.build_posterior()

    summary = getattr(inference, "summary", {}) or {}
    val = summary.get("validation_loss", []) or []
    trn = summary.get("training_loss", []) or []
    epochs = summary.get("epochs_trained", []) or []
    stats = {
        "nde_index": number,
        "round": round_number,
        "training_time_s": float(training_time),
        "best_val_loss": float(np.min(val)) if len(val) else float("nan"),
        "best_train_loss": float(np.min(trn)) if len(trn) else float("nan"),
        "n_epochs": int(epochs[-1]) if len(epochs) else (len(val) if len(val) else 0),
    }
    return posterior, stats


@register_backend
class SIXSABackend(Backend):

    name = "sixsa"
    config_cls = SIXSA

    def __init__(self, *, solver: "Solver", config: SIXSA, **kwargs):
        super().__init__(solver=solver, config=config)
        self.prior = _unit_gaussian_prior(solver.prior.ndim)
        self.history = []
        self.bank = None
        self.pool_info = None
        self.emulator_pass_info = None
        self.likelihood_emulator = None
        self.best_ensemble = None
        self.best_round = 0
        # Which `posterior_source` actually produced the returned samples
        # (may differ from the config when the preferred source is missing).
        self.posterior_source_used = None
        self.all_theta_log = None
        self.all_cstat = None
        self._buffer_sims = 0
        self._sim_theta = []
        self._sim_cstat = []

    # ------------------------------------------------------------------ #
    # Public sampling boundary (latent -> physical)
    # ------------------------------------------------------------------ #
    def sample(self, n: int) -> np.ndarray:
        latent = self._sample_latent(n)
        if hasattr(latent, "detach"):
            latent = latent.detach().cpu().numpy()
        return self._gaussian_to_physical(np.asarray(latent))

    # Fallback order per `posterior_source`: the chosen source first, then the
    # remaining ones, so a run always returns a posterior even when its
    # preferred source is unavailable (e.g. the emulator pass was disabled).
    _POSTERIOR_SOURCE_ORDER = {
        "pool": ("pool", "emulator_pass", "best_flow"),
        "emulator_pass": ("emulator_pass", "pool", "best_flow"),
        "best_flow": ("best_flow", "pool", "emulator_pass"),
    }

    def _sample_latent(self, n: int):
        """Draw ``n`` latent posterior samples from the configured source.

        Exactness vs diversity (see ``SIXSA.posterior_source``):

        * ``"pool"`` -- resample the banked pooled multiple-IS draws. Most
          exact, but the DISTINCT-row count is capped by the banked draws and
          their pooled ESS, so a large ``num_posterior_samples`` repeats rows.
        * ``"emulator_pass"`` -- resample the emulator pass's flow-IS weighted
          draws: fresh surrogate samples, still importance-corrected.
        * ``"best_flow"`` -- draw fresh from the best flow: no duplicates, but
          no importance correction (the flow's own approximation).
        """
        order = self._POSTERIOR_SOURCE_ORDER[self.config.posterior_source]
        for source in order:
            latent = self._sample_from_source(source, n)
            if latent is not None:
                self.posterior_source_used = source
                return latent
        self.posterior_source_used = "prior"
        return self.prior.sample((n,))

    def _sample_from_source(self, source: str, n: int):
        """Sample from one named source, or return ``None`` if unavailable."""
        if source == "pool":
            if self.bank is None or not len(self.bank):
                return None
            try:
                z, w = self.bank.pooled_weighted(self.prior)
            except Exception:
                return None
            return z[torch.multinomial(w, n, replacement=True)]

        if source == "emulator_pass":
            emu = self.emulator_pass_info
            if emu is None or emu.get("weighted") is None:
                return None
            z, w = emu["weighted"]
            return z[torch.multinomial(w, n, replacement=True)]

        if source == "best_flow":
            if self.best_ensemble is None:
                return None
            try:
                return self.best_ensemble.sample((n,)).detach()
            except Exception:
                return None

        return None

    # ------------------------------------------------------------------ #
    # Simulation helpers
    # ------------------------------------------------------------------ #
    def _observation(self):
        return torch.as_tensor(self.solver.observed_spectrum, dtype=torch.float32)

    def _log_likelihood_latent(self, z):
        """Exact log-likelihood ``-0.5*cstat`` for latent draws ``z``."""
        physical = self._gaussian_to_physical(
            z.detach().cpu().numpy() if hasattr(z, "detach") else np.asarray(z)
        )
        out = self.solver.simulate(physical, return_kind="cstat", progress_bar=False)
        return torch.as_tensor(-0.5 * np.asarray(out["cstat"], dtype=np.float64))

    def _simulate_latent(self, z):
        """Fold ``z``, draw the single Poisson realization, return (x, cstat)."""
        z_np = z.detach().cpu().numpy() if hasattr(z, "detach") else np.asarray(z)
        physical = self._gaussian_to_physical(z_np)
        res = self.solver.simulate(physical, return_kind="full_model_counts")
        mean = np.asarray(res["total_model_counts"], dtype=np.float64)
        cstat = np.asarray(res["cstat"], dtype=np.float64)
        x_poisson = np.random.poisson(np.clip(mean, 0.0, None)).astype(np.float32)
        self._buffer_sims += len(z_np)
        self._sim_theta.append(np.asarray(z_np, dtype=np.float32))
        self._sim_cstat.append(cstat)
        return x_poisson, cstat

    def _init_buffer(self, n_initial, capacity):
        z = self._initial_latent(n_initial)
        x, cstat = self._simulate_latent(z)
        return TrainingBuffer(
            z.detach().cpu().numpy(), x, cstat, capacity=capacity
        )

    def _initial_latent(self, n: int) -> torch.Tensor:
        """Round-1 design in the unit-Gaussian latent (QMC / LHS / i.i.d. prior)."""
        if self.config.first_round_sampling == "qmc":
            from scipy.stats import qmc

            z = qmc.MultivariateNormalQMC(
                mean=np.zeros(self.solver.prior.ndim)
            ).random(n)
            return torch.as_tensor(z, dtype=torch.float32)
        if self.config.first_round_sampling == "lhs":
            from scipy.stats import norm, qmc

            u = qmc.LatinHypercube(d=self.solver.prior.ndim).random(n)
            z = norm.ppf(np.clip(u, 1e-12, 1.0 - 1e-12))
            return torch.as_tensor(z, dtype=torch.float32)
        return self.prior.sample((n,))

    # ------------------------------------------------------------------ #
    # Training and proposals
    # ------------------------------------------------------------------ #
    def _train_member(self, buffer, member, round_number):
        theta, x = buffer.training_arrays()
        posterior, stats = training_job(
            member,
            theta,
            x,
            embedding_net=self.config.embedding_net,
            round_number=round_number,
            training_kwargs=self._training_kwargs,
            nde_kwargs=merge_nde_kwargs(self.config.nde_kwargs),
            num_threads=available_cpu_count(),
        )
        stats["cstat_ceiling"] = float(buffer.ceiling)
        self._member_stats.append(stats)
        posterior = patch_sample_no_pbar(posterior)
        posterior.set_default_x(self._observation())
        return posterior

    def _truncated(self, flow):
        return build_truncated_proposal(
            flow,
            self.prior,
            quantile=self.config.truncated_quantile,
            n_support=self.config.truncated_num_samples_to_estimate_support,
            device=self.config.device,
            restrict_from_prior=self.config.truncated_restrict_from_prior,
        )

    def _flat_draw(self, proposal, n):
        return sample_flat(
            proposal,
            int(n),
            oversampling_factor=self.config.truncated_oversampling_factor,
            max_sampling_batch_size=self.config.truncated_max_sampling_batch_size,
        )

    def _sub_batch_move(self, member_flow, buffer, n_draw):
        proposal = self._truncated(member_flow)
        z = self._flat_draw(proposal, n_draw)
        x, cstat = self._simulate_latent(z)
        replaced = buffer.replace_worst(z.detach().cpu().numpy(), x, cstat)
        return len(z), replaced

    def _ensemble(self, members):
        ens = EnsemblePosterior([m for m in members])
        ens.set_default_x(self._observation())
        return patch_sample_no_pbar(ens)

    # ------------------------------------------------------------------ #
    # The round loop
    # ------------------------------------------------------------------ #
    def run(self) -> "FitResults":
        from ...solver import FitResults
        from ...convenience import catchtime

        cfg = self.config
        d = self.solver.prior.ndim
        n_initial = cfg.n_initial_per_param * d
        n_batch_base = cfg.n_batch or cfg.n_picked_per_param * d
        n_sub = cfg.n_sub_batch or int(math.ceil(n_batch_base / cfg.n_ensemble))
        n_train_keep = cfg.n_train_keep or n_initial
        round_gate_n = cfg.round_gate_n or cfg.gate_n

        def ramp(r):
            v = cfg.n_simulate_ramp ** (r - 1)
            if cfg.n_simulate_ramp_max is not None:
                v = min(v, cfg.n_simulate_ramp_max)
            return v

        self._training_kwargs = self._build_training_kwargs()

        self._log(f"SIXSA version {SIXSA_VERSION} | d={d} n_initial={n_initial} "
                  f"n_batch_base={n_batch_base} n_sub={n_sub} n_ensemble={cfg.n_ensemble} "
                  f"n_rounds={cfg.n_rounds} merge_mode={cfg.merge_mode}")

        self.history = []
        self._buffer_sims = 0
        self._sim_theta = []
        self._sim_cstat = []
        bank = DefensiveISBank()
        self.bank = bank
        exited = None
        last_members = []

        with catchtime("Running SIXSA inference", print_time=False) as run_time:
            with self._phase(0, "initial simulation"):
                buffer = self._init_buffer(n_initial, capacity=n_train_keep)

            for r in range(1, cfg.n_rounds + 1):
                self._maybe_plot_predictive(buffer, r)
                self._member_stats = []
                members = []
                passed = False

                for k in range(1, cfg.n_ensemble + 1):
                    with self._phase(r, f"train member {k}"):
                        m = self._train_member(buffer, k, r)
                    members.append(m)

                    if k < cfg.n_ensemble or cfg.last_member_sub_batch:
                        with self._phase(r, f"sub-batch move (member {k})"):
                            n_draw = int(round(n_sub * ramp(r)))
                            self._sub_batch_move(m, buffer, n_draw)

                    if r >= cfg.khat_from_round:
                        with self._phase(r, f"check (member {k})"):
                            passed, diag = self._run_check(bank, m, r, k)
                        if passed:
                            if cfg.confirm_exit and k < cfg.n_ensemble:
                                with self._phase(r, "confirmation member"):
                                    m2 = self._train_member(buffer, k + 1, r)
                                    members.append(m2)
                                    self._run_check(bank, m2, r, k + 1)
                            exited = ("check", r, k)
                            break

                last_members = members
                if exited:
                    self._append_round_info(r, members, buffer, bank, gated="check")
                    break

                if r >= cfg.khat_from_round and not passed:
                    ens = self._ensemble(members)
                    with self._phase(r, "round gate"):
                        gate_ok, _ = self._run_gate(bank, ens, round_gate_n, "round_gate", r)
                    if gate_ok:
                        exited = ("round_gate", r)
                        self._append_round_info(r, members, buffer, bank, gated="round_gate")
                        break

                self._append_round_info(r, members, buffer, bank, gated=None)

                if r < cfg.n_rounds:
                    ens = self._ensemble(members)
                    self._maybe_plot_ensemble(members, ens, r, buffer)
                    with self._phase(r, "between-round update"):
                        buffer = self._between_round_update(
                            ens, buffer, n_sub, n_batch_base, ramp(r + 1),
                            n_train_keep,
                        )

            # FINAL GATE (skip when a round gate already covered >= gate_n draws).
            skip_final = (
                exited is not None
                and exited[0] == "round_gate"
                and round_gate_n >= cfg.gate_n
            )
            if not skip_final and last_members:
                ens = self._ensemble(last_members)
                with self._phase(0, "final gate"):
                    self._run_gate(bank, ens, cfg.gate_n, "final_gate", exited[1] if exited else cfg.n_rounds)

            self._finalize(bank, last_members)

            emu_sims = 0
            if cfg.run_emulator_pass and self.best_ensemble is not None:
                with self._phase(0, "emulator pass"):
                    emu_sims = self._run_emulator_pass(d)

        n_evaluations = self._buffer_sims + int(sum(len(b.z) for b in bank.batches)) + emu_sims
        self._maybe_plot_history()

        log_Z = self.pool_info["log_Z"] if self.pool_info else float("nan")
        log_Z_err = self.pool_info["log_Z_err"] if self.pool_info else float("nan")
        best_fit, best_fit_stat = self._best_fit()

        posterior = self._posterior_dataframe(
            np.asarray(self.sample(cfg.num_posterior_samples))
        )
        return FitResults(
            time=float(run_time()),
            posterior_samples=posterior,
            n_likelihood_evaluations=int(n_evaluations),
            log_Z=log_Z,
            log_Z_err=log_Z_err,
            best_fit=best_fit,
            best_fit_stat=best_fit_stat,
        )

    # ------------------------------------------------------------------ #
    # Round-loop sub-steps
    # ------------------------------------------------------------------ #
    def _build_training_kwargs(self):
        cfg = self.config
        base = cfg.training_kwargs
        if base is None:
            base = dict(
                force_first_round_loss=True,
                use_combined_loss=False,
                retrain_from_scratch=True,
                discard_prior_samples=True,
                learning_rate=1e-3,
                stop_after_epochs=10,
                training_batch_size=512,
                validation_fraction=0.2,
                clip_max_norm=1.0,
            )
        return {"max_num_epochs": cfg.max_num_epochs, **dict(base)}

    def _run_check(self, bank, member_flow, round_, member):
        cfg = self.config
        return run_check(
            bank,
            flow=member_flow,
            prior=self.prior,
            log_likelihood_fn=self._log_likelihood_latent,
            n=cfg.early_member_n,
            alpha=cfg.gate_alpha,
            khat_threshold=cfg.khat_threshold,
            min_eff=cfg.early_member_min_eff,
            ess_mode=cfg.ess_mode,
            ess_target=cfg.ess_target,
            round_=round_,
            member=member,
        )

    def _run_gate(self, bank, ensemble_flow, n, kind, round_):
        cfg = self.config
        return run_gate(
            bank,
            flow=ensemble_flow,
            prior=self.prior,
            log_likelihood_fn=self._log_likelihood_latent,
            n=n,
            alpha=cfg.gate_alpha,
            khat_threshold=cfg.khat_threshold,
            min_eff=cfg.early_member_min_eff,
            ess_mode=cfg.ess_mode,
            ess_target=cfg.ess_target,
            kind=kind,
            round_=round_,
            member_selection=cfg.member_selection,
        )

    def _between_round_update(self, ens, buffer, n_sub, n_batch_base, ramp_next, n_train_keep):
        proposal = self._truncated(ens)
        if self.config.merge_mode == "merge":
            n_draw = int(round(n_sub * ramp_next))
            z = self._flat_draw(proposal, n_draw)
            x, cstat = self._simulate_latent(z)
            buffer.replace_worst(z.detach().cpu().numpy(), x, cstat)
            return buffer
        # fresh: wholesale replacement
        n_draw = int(round(n_batch_base * ramp_next))
        z = self._flat_draw(proposal, n_draw)
        x, cstat = self._simulate_latent(z)
        return TrainingBuffer(
            z.detach().cpu().numpy(), x, cstat, capacity=n_train_keep
        )

    def _finalize(self, bank, last_members):
        cfg = self.config
        if len(bank):
            self.pool_info = bank.pooled_estimate(self.prior, balance_heuristic=True)
            best_batch = bank.best_batch(cfg.khat_threshold, cfg.early_member_min_eff)
            if best_batch is not None:
                self.best_ensemble = best_batch.flow
                self.best_round = best_batch.round_
        if self.best_ensemble is None and last_members:
            self.best_ensemble = self._ensemble(last_members)

    def _run_emulator_pass(self, d):
        cfg = self.config
        emu = emulator_importance_pass(
            best_flow=self.best_ensemble,
            prior=self.prior,
            log_likelihood_fn=self._log_likelihood_latent,
            n_draws=cfg.emulator_min_per_param * d,
            num_posterior_samples=cfg.num_posterior_samples,
        )
        self.emulator_pass_info = emu
        if emu is None:
            return 0
        # Extend the run's (theta, cstat) pool with the surrogate draws.
        self._sim_theta.append(emu["z"].detach().cpu().numpy())
        self._sim_cstat.append(np.asarray(emu["cstat"]))
        self._assemble_pool()
        if cfg.train_likelihood_emulator:
            with self._phase(0, "train likelihood emulator"):
                self.likelihood_emulator = train_likelihood_emulator(
                    emu["z"].detach().cpu().numpy(),
                    np.asarray(emu["cstat"]),
                    emulator_kwargs=cfg.emulator_kwargs,
                    path_results=self.solver.outputfiles_basename,
                )
        return int(emu["n_draws"])

    def _assemble_pool(self):
        thetas = list(self._sim_theta)
        cstats = list(self._sim_cstat)
        if self.bank is not None:
            thetas += [b.z.detach().cpu().numpy() for b in self.bank.batches]
            cstats += [(-2.0 * b.log_lik).detach().cpu().numpy() for b in self.bank.batches]
        if thetas:
            self.all_theta_log = np.concatenate(thetas, axis=0)
            self.all_cstat = np.concatenate([np.asarray(c).ravel() for c in cstats])

    def _best_fit(self):
        self._assemble_pool()
        if self.all_cstat is None or len(self.all_cstat) == 0:
            return None, None
        i = int(np.argmin(self.all_cstat))
        physical = self._gaussian_to_physical(self.all_theta_log[i][None, :])[0]
        best_fit = self._posterior_dataframe(np.asarray(physical)[None, :]).iloc[0]
        return best_fit, float(self.all_cstat[i])

    def _append_round_info(self, r, members, buffer, bank, gated):
        cstat = buffer.cstat
        # Round headline diagnostics: the round's most recent banked batch (gate
        # if it ran this round, else the best member check this round).
        round_batches = [b for b in bank.batches if b.round_ == r]
        khat = eff = log_Z = log_Z_err = float("nan")
        if round_batches:
            diags = [defensive_is_diagnostics(b.log_w.cpu().numpy(), n_total=len(b.z))
                     for b in round_batches]
            best = min(diags, key=lambda dd: dd["khat"] if np.isfinite(dd["khat"]) else float("inf"))
            khat, eff = best["khat"], best["efficiency"]
            log_Z, log_Z_err = best["log_Z"], best["log_Z_err"]
        self.history.append({
            "round": r,
            "nde_stats": list(self._member_stats),
            "n_simulations_total": int(self._buffer_sims),
            "cstat_min": float(cstat.min()),
            "cstat_median": float(np.median(cstat)),
            "cstat_max": float(cstat.max()),
            "cstat_ceiling": float(buffer.ceiling),
            "k_hat": khat,
            "efficiency": eff,
            "log_Z": log_Z,
            "log_Z_err": log_Z_err,
            "gated": gated,
            "n_members": len(members),
        })

    # ------------------------------------------------------------------ #
    # Diagnostics / plotting / logging
    # ------------------------------------------------------------------ #
    def _log(self, msg):
        if self.config.trace:
            print(f"[SIXSA] {msg}", flush=True)

    @contextlib.contextmanager
    def _phase(self, round_number, name):
        if not self.config.trace:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            print(f"[SIXSA r{round_number}] {name}: {time.perf_counter() - t0:.2f} s",
                  flush=True)

    def _maybe_plot_predictive(self, buffer, round_number):
        if not self.config.plot_diagnostics:
            return
        try:
            from ...xspec import SpectrumState

            state = SpectrumState(1)
            round_path = os.path.join(
                self.solver.outputfiles_basename, f"round_{round_number}"
            )
            os.makedirs(round_path, exist_ok=True)
            plots.plot_round_coverage(
                buffer.x, state.observed_counts, state.bin_edges_1d,
                round_index=round_number, outdir=round_path,
            )
        except Exception:
            pass

    def _maybe_plot_ensemble(self, members, ensemble, round_number, buffer):
        if not self.config.plot_diagnostics:
            return
        try:
            import matplotlib.pyplot as plt

            round_path = os.path.join(
                self.solver.outputfiles_basename, f"round_{round_number}"
            )
            os.makedirs(round_path, exist_ok=True)
            plots.plot_ensemble_posteriors(
                members, ensemble, self.solver.observed_spectrum,
                self.solver.prior, self.solver.parameter_names,
                n_samples=self.config.ensemble_corner_samples,
                outdir=round_path, round_index=round_number,
            )
            plt.close("all")
        except Exception:
            pass

    def _maybe_plot_history(self):
        if not (self.config.plot_diagnostics and self.history):
            return
        try:
            import matplotlib.pyplot as plt

            result = plots.plot_training_history(
                self.history,
                khat_threshold=self.config.khat_threshold,
                outdir=self.solver.outputfiles_basename,
            )
            if result is not None:
                plt.close(result[0])
        except Exception:
            pass

    def plot_training_history(self, save=False, outdir=None, **kwargs):
        """Build the six-panel training-history figure from ``self.history``."""
        if save and outdir is None:
            outdir = self.solver.outputfiles_basename
        result = plots.plot_training_history(
            getattr(self, "history", []),
            khat_threshold=self.config.khat_threshold,
            outdir=outdir,
            save=save,
            **kwargs,
        )
        return result[0] if result is not None else None
