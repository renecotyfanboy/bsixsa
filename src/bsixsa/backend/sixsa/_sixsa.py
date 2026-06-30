import contextlib
import functools
import os
import time
import warnings
from typing import TYPE_CHECKING

import dill
import numpy as np
import torch
from joblib import Parallel, delayed
from sbi.inference import EnsemblePosterior, NPE
from sbi.neural_nets import posterior_nn
from sbi.utils import RestrictedPrior, get_density_thresholder

from .. import register_backend
from ..abc import Backend
from ..config import SIXSA
from . import plots
from .embedding import IdentityEmbedding
from .embedding.abc import Embedding
from .embedding.nets import EmbeddingNet, FCEmbeddingNet

if TYPE_CHECKING:
    from ...solver import FitResults, Solver

# Density-estimator defaults mirroring `_sixsa_dev` (sixsa_default.yml nde_kwargs):
# a MAF with wide transforms and independent z-scoring of theta and x.
DEFAULT_NDE_KWARGS = {
    "model": "maf",
    "hidden_features": 100,
    "num_transforms": 10,
    "z_score_theta": "independent",
    "z_score_x": "independent",
}

# Minimum effective sample size below which the Pareto k_hat diagnostic is not
# trustworthy: with too few effective samples the generalized-Pareto tail fit is
# uninformative and psislw reports a spuriously LOW k_hat on a proposal that is
# actually catastrophic. Mirrors `_sixsa_dev` (sixsa_main.py); below this we
# report k_hat = inf so such rounds are never selected as best.
_KHAT_MIN_ESS = 10.0


def _unit_gaussian_prior(ndim):
    """Standard-normal prior ``N(0, I)`` over the SIXSA latent space.

    SIXSA works in a unit-Gaussian latent space (rather than the unit cube): the
    unbounded support lets the flow and the rejection sampler avoid the hard box
    edges that previously caused mass leakage and inefficient sampling. Latent
    draws are mapped to physical parameters by ``solver.prior.from_unit_gaussian``
    (``ppf(Phi(z))`` per marginal).
    """
    return torch.distributions.MultivariateNormal(
        torch.zeros(ndim), torch.eye(ndim)
    )


def patch_sample_no_pbar(posterior):
    """Patch the sample method of a posterior to remove the progress bar."""
    original = posterior.sample
    posterior._original_sample = original

    @functools.wraps(original)
    def sample(*args, _original=original, **kwargs):
        kwargs.setdefault("show_progress_bars", False)
        return _original(*args, **kwargs)

    posterior.sample = sample
    return posterior


def work_quiet(func):
    """Decorator to run a function without printing anything to stdout."""

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
    output_dir,
    proposal=None,
    training_kwargs=None,
    nde_kwargs=None,
    num_threads=None,
):
    # Each ensemble member trains in its own process; cap torch's intra-op
    # threads so the `n_ensemble` parallel workers fill the cores instead of
    # oversubscribing them (the default is one full machine's worth per worker).
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))

    current_round_dir = os.path.join(output_dir, f"round_{round_number}")
    current_artifacts_dir = os.path.join(current_round_dir, "artifacts")
    os.makedirs(current_artifacts_dir, exist_ok=True)

    # Each ensemble member sees its own Poisson realization of the model counts,
    # which diversifies the ensemble and stabilises training and generation.
    x = torch.as_tensor(np.random.poisson(x).astype(np.float32))

    if (round_number == 1) or training_kwargs.get("retrain_from_scratch", False):
        # Embedding network trained jointly with the NPE. A user-provided
        # `EmbeddingNet` spec wins; otherwise use the default FC pyramid, which
        # mirrors how `_sixsa_dev` embeds spectra.
        if embedding_net is not None:
            net = embedding_net.build(x.shape[1])
        else:
            net = FCEmbeddingNet().build(x.shape[1])
        prior_sbi = _unit_gaussian_prior(theta.shape[-1])
        build_fun = posterior_nn(
            embedding_net=net, **(nde_kwargs or DEFAULT_NDE_KWARGS)
        )
        inference = NPE(prior=prior_sbi, density_estimator=build_fun, device="cpu")
    else:
        previous_round_dir = os.path.join(output_dir, f"round_{round_number - 1}")
        previous_artifacts_dir = os.path.join(previous_round_dir, "artifacts")
        previous_inference_path = os.path.join(
            previous_artifacts_dir,
            f"inference_{number}.pkl",
        )
        if not os.path.exists(previous_inference_path):
            raise FileNotFoundError(
                f"Missing previous round state for net {number}: {previous_inference_path}"
            )
        with open(previous_inference_path, "rb") as file:
            inference = dill.load(file)

    training_kwargs = {} if training_kwargs is None else training_kwargs

    with warnings.catch_warnings():
        # Catches warning about pickling NPE / Restricted prior not pointing to wrong address
        warnings.simplefilter("ignore", category=UserWarning)

        training_start = time.time()
        density_estimator = inference.append_simulations(
            torch.from_numpy(theta.copy()),
            x,
            proposal=proposal,
        ).train(**training_kwargs)
        training_time = time.time() - training_start

        posterior = inference.build_posterior()

        with open(
            os.path.join(current_artifacts_dir, f"inference_{number}.pkl"),
            "wb",
        ) as file:
            dill.dump(inference, file, recurse=True)

        with open(
            os.path.join(current_artifacts_dir, f"density_estimator_{number}.pkl"),
            "wb",
        ) as file:
            dill.dump(density_estimator, file, recurse=True)

        with open(
            os.path.join(current_artifacts_dir, f"posterior_{number}.pkl"),
            "wb",
        ) as file:
            dill.dump(posterior, file, recurse=True)

    # Per-NDE training summary, consumed by the diagnostics history.
    summary = getattr(inference, "summary", {}) or {}
    val = summary.get("validation_loss", []) or []
    trn = summary.get("training_loss", []) or []
    epochs = summary.get("epochs_trained", []) or []
    stats = {
        "nde_index": number,
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

    def __init__(
        self,
        *,
        solver: "Solver",
        config: SIXSA,
        **kwargs,
    ):
        super().__init__(solver=solver, config=config)
        self.proposals = []
        self.history = []
        self.n_nets = config.n_ensemble
        self.prior = _unit_gaussian_prior(solver.prior.ndim)
        self.best_ensemble = None
        self.best_round = 0
        self.best_efficiency = -float("inf")
        self.best_khat = float("inf")
        self.best_log_Z = float("nan")
        self.best_log_Z_err = float("nan")
        # Importance-resampling posterior: (theta_latent, normalised_weights)
        # for the best round, and the most recent round's set.
        self.best_weighted = None
        self._last_weighted = None

    def sample(self, n: int) -> np.ndarray:
        # The backend works in the unit-Gaussian latent internally (flow, IS
        # weights, resampling); convert to physical here, at the public boundary,
        # so the returned posterior matches the prior and the other backends and
        # can be fed straight to `solver.simulate()` / the predictive-coverage plots.
        if self.best_weighted is not None:
            # Preferred: importance-resample the best round's simulated points
            # (the reference's posterior). These are real prior draws, so the
            # physical samples lie within the prior support (no out-of-distribution
            # events) and the flow is never sampled.
            theta, weights = self.best_weighted
            idx = torch.multinomial(weights, n, replacement=True)
            latent = theta[idx]
        elif self.best_ensemble is not None:
            latent = self.best_ensemble.sample((n,))
        else:
            sampler = self.proposals[-1] if self.proposals else self.prior
            latent = sampler.sample((n,))

        if hasattr(latent, "detach"):
            latent = latent.detach().cpu().numpy()
        return self._gaussian_to_physical(np.asarray(latent))

    def _compute_is_diagnostics(self, new_log, cstats, log_w):
        import torch
        import numpy as np
        from arviz import psislw

        finite = torch.isfinite(log_w)
        log_w_safe = torch.where(finite, log_w, torch.full_like(log_w, -np.inf))
        log_w_fin = log_w[finite]

        diag = {
            "ess": float("nan"),
            "efficiency": float("nan"),
            "k_hat": float("nan"),
            "log_Z": float("nan"),
            "log_Z_err": float("nan"),
            "log_w_safe": log_w_safe,
            "finite": finite,
            "w_norm": None,
        }

        if len(log_w_fin) > 0:
            w = torch.exp(log_w_fin - log_w_fin.max())
            w_norm = w / w.sum()
            ess = float(1.0 / (w_norm ** 2).sum().item())
            n_total = len(log_w)

            # Normalised importance weights over the finite samples, used to
            # importance-resample the posterior (the returned, flow-free posterior).
            diag["w_norm"] = w_norm
            diag["ess"] = ess
            diag["efficiency"] = 100.0 * ess / len(log_w_fin)
            diag["log_Z"] = float((torch.logsumexp(log_w_fin, 0) - np.log(n_total)).item())
            diag["log_Z_err"] = float(np.sqrt((n_total - ess) / (n_total * ess)))

            # k_hat via arviz psislw, matching `_sixsa_dev` (which abandoned sbi's
            # gpdfit because it reports a systematically different k_hat on
            # identical weights, making in-loop and final k_hat non-comparable).
            # The min-ESS guard returns k_hat=inf when there are too few effective
            # samples for a meaningful generalized-Pareto tail fit (see _KHAT_MIN_ESS).
            try:
                if not np.isfinite(ess) or ess < _KHAT_MIN_ESS:
                    diag["k_hat"] = float("inf")
                else:
                    _, khat_val = psislw(log_w_fin.detach().cpu().numpy().copy())
                    diag["k_hat"] = float(khat_val)
            except Exception:
                diag["k_hat"] = float("nan")

        return diag

    def _initial_latent(self, n: int) -> torch.Tensor:
        """Draw the first round's parameters in the unit-Gaussian latent ``N(0, I)``.

        With ``first_round_sampling="qmc"`` (default) this returns a Sobol-based
        quasi-Monte-Carlo design (``MultivariateNormalQMC``), which fills the
        latent far more evenly than i.i.d. normal draws at the same simulation
        budget; ``"prior"`` restores the plain i.i.d. ``N(0, I)`` draw. The
        samples are mapped to physical space downstream by
        ``solver.prior.from_unit_gaussian`` (``ppf(Phi(z))`` per marginal).
        """
        if self.config.first_round_sampling == "qmc":
            from scipy.stats import qmc

            z = qmc.MultivariateNormalQMC(
                mean=np.zeros(self.solver.prior.ndim)
            ).random(n)
            return torch.as_tensor(z, dtype=torch.float32)
        return self.prior.sample((n,))

    def inference_round(
        self,
        n_simulations,
        *,
        embedding: Embedding,
        embedding_net=None,
        proposal=None,
        round_number=0,
        training_kwargs=None,
        is_filter_fraction=0.0,
        proposal_mode="truncated",
        previous_posteriors_list=None,
    ):
        # `n_simulations` is the per-round simulation budget: we draw and simulate
        # exactly this many, then keep the top (1 - is_filter_fraction) fraction by
        # importance weight for training (round 1 has no weights and keeps all).
        n_pool = n_simulations
        if proposal is None:
            parameters_latent = self._initial_latent(n_pool)
        else:
            parameters_latent = proposal.sample((n_pool,))

        parameters = self.solver.prior.from_unit_gaussian(parameters_latent.numpy())
        simulations_results = self.solver.simulate(parameters, return_kind="full_model_counts")
        spectra = simulations_results["total_model_counts"]
        cstats = torch.as_tensor(simulations_results["cstat"], dtype=torch.float32)

        diag = None
        selected_mask = None
        kept_theta_latent = parameters_latent
        kept_spectra = spectra
        kept_cstats = cstats
        self._last_weighted = None

        if proposal is not None and previous_posteriors_list is not None:
            observation = torch.from_numpy(self.solver.observed_spectrum)

            # Evaluate log_q under the previous ensemble of NPE posteriors
            log_q = torch.stack(
                [pk.log_prob(parameters_latent, x=observation)
                 for pk in previous_posteriors_list],
                0,
            )
            log_q_mix = torch.logsumexp(log_q, dim=0) - np.log(len(previous_posteriors_list))

            # log_q_mix is the proposal density in the IS weights below, so this is
            # only valid when the proposal approximates the posterior
            # (proposal_mode="posterior", or "truncated" with sample_with="sir").
            # Rejection sampling draws the truncated prior instead and would bias
            # these weights -- see `run()` for the guard.

            # Prior density (standard-normal prior log_prob on the latent samples)
            log_prior = self.prior.log_prob(parameters_latent)

            # Raw importance log-weights: -0.5 cstat + log prior - log proposal
            log_w = -0.5 * cstats + log_prior - log_q_mix

            diag = self._compute_is_diagnostics(parameters_latent, cstats, log_w)

            # Stash the importance-weighted simulated points (all finite samples)
            # so the run can return an IS-resampled posterior from the best round.
            if diag["w_norm"] is not None:
                self._last_weighted = (
                    parameters_latent[diag["finite"]].detach(),
                    diag["w_norm"].detach(),
                )
            else:
                self._last_weighted = None

            # Keep the top-weight fraction as the training batch (sample weights).
            if is_filter_fraction > 0.0:
                n_keep = max(
                    1,
                    int(round((1.0 - is_filter_fraction) * len(parameters_latent))),
                )
                keep_idx = torch.argsort(diag["log_w_safe"], descending=True)[:n_keep]
                kept_theta_latent = parameters_latent[keep_idx]
                kept_spectra = spectra[keep_idx]
                kept_cstats = cstats[keep_idx]
                selected_mask = np.zeros(len(parameters_latent), dtype=bool)
                selected_mask[keep_idx.numpy()] = True

        # Fit the (optional) pre-processor embedding on the kept spectra. The NPEs
        # train on the raw kept spectra with their own jointly-trained embedding
        # net, so this only matters for a trainable pre-processor embedding.
        noisy_spectra = np.random.poisson(kept_spectra).astype(np.float32)
        round_path = os.path.join(self.solver.outputfiles_basename, f"round_{round_number}")
        os.makedirs(round_path, exist_ok=True)
        embedding.fit(
            noisy_spectra,
            metrics_path=os.path.join(round_path, "embedding_training.pdf"),
        )

        # Train the ensemble of NPEs on the kept (filtered) parameters. Each job
        # returns (posterior, training_stats). Split the cores evenly across the
        # parallel workers so torch does not oversubscribe (default: a full
        # machine's worth of threads per worker -> n_nets x cores contention).
        nde_kwargs = self.config.nde_kwargs or DEFAULT_NDE_KWARGS
        num_threads = max(1, (os.cpu_count() or 1) // max(1, self.n_nets))
        kept_theta_latent_np = kept_theta_latent.numpy().copy()
        results = Parallel(n_jobs=-1)(
            delayed(training_job)(
                i,
                kept_theta_latent_np,
                kept_spectra,
                embedding_net=embedding_net,
                round_number=round_number,
                output_dir=self.solver.outputfiles_basename,
                proposal=proposal,
                training_kwargs=training_kwargs,
                nde_kwargs=nde_kwargs,
                num_threads=num_threads,
            )
            for i in range(self.n_nets)
        )

        # Gather the posterior ensemble + per-NDE training stats
        posteriors = [patch_sample_no_pbar(result[0]) for result in results]
        nde_stats = [result[1] for result in results]
        ensemble = EnsemblePosterior(posteriors)
        ensemble.set_default_x(torch.from_numpy(self.solver.observed_spectrum))

        # Per-round summary consumed by the training-history diagnostic.
        def _diag_value(key):
            return diag[key] if diag is not None else float("nan")

        round_info = {
            "round": round_number,
            "n_simulations_round": n_pool,
            "n_kept": int(len(kept_theta_latent)),
            "nde_stats": nde_stats,
            "cstat_min": float(kept_cstats.min().item()),
            "cstat_median": float(kept_cstats.median().item()),
            "cstat_max": float(kept_cstats.max().item()),
            "ess": _diag_value("ess"),
            "efficiency": _diag_value("efficiency"),
            "k_hat": _diag_value("k_hat"),
            "log_Z": _diag_value("log_Z"),
            "log_Z_err": _diag_value("log_Z_err"),
        }

        # Optional per-round diagnostic plots (coverage band + ensemble corner).
        if self.config.plot_diagnostics:
            self._plot_round_diagnostics(
                spectra, selected_mask, posteriors, ensemble,
                round_number, round_path,
            )

        return ensemble, posteriors, round_info

    def _plot_round_diagnostics(
        self, spectra, selected_mask, posteriors, ensemble, round_number, round_path
    ):
        """Write the per-round coverage band and ensemble-posterior corner (best-effort)."""
        import matplotlib.pyplot as plt

        try:
            from ...xspec import SpectrumState

            state = SpectrumState(1)
            x_sim = np.random.poisson(np.asarray(spectra)).astype(float)
            plots.plot_round_coverage(
                x_sim,
                state.observed_counts,
                state.bin_edges_1d,
                selected_mask=selected_mask,
                round_index=round_number,
                outdir=round_path,
            )
        except Exception:
            pass

        try:
            plots.plot_ensemble_posteriors(
                posteriors,
                ensemble,
                self.solver.observed_spectrum,
                self.solver.prior,
                self.solver.parameter_names,
                n_samples=self.config.ensemble_corner_samples,
                outdir=round_path,
                round_index=round_number,
            )
        except Exception:
            pass

        plt.close("all")

    def run(self) -> 'FitResults':
        from ...solver import FitResults
        from ...convenience import catchtime

        # All run-time options are sourced from the SIXSA config so the backend
        # can be driven by `Solver.run()`, which calls `run()` with no arguments.
        num_simulations_per_round = self.config.num_simulations_per_round
        embedding = self.config.embedding
        embedding_net = self.config.embedding_net
        training_kwargs = self.config.training_kwargs
        max_num_epochs = self.config.max_num_epochs
        device = self.config.device
        proposal_mode = self.config.proposal_mode
        is_filter_fraction = self.config.is_filter_fraction
        khat_threshold = self.config.khat_threshold
        force_last_round = self.config.force_last_round
        truncated_quantile = self.config.truncated_quantile
        truncated_num_samples_to_estimate_support = (
            self.config.truncated_num_samples_to_estimate_support
        )
        truncated_sampling_method = self.config.truncated_sampling_method

        self.n_nets = self.config.n_ensemble
        num_rounds = len(num_simulations_per_round)

        if training_kwargs is None:
            # Defaults matching `_sixsa_dev`'s effective regime. Two key knobs:
            #
            # * `force_first_round_loss=True` keeps the simple (non-atomic) MLE
            #   loss every round instead of the atomic SNPE-C loss. The loss is
            #   applied to the importance-filtered current batch, which approximates
            #   the posterior. (The unit-Gaussian latent is unbounded, so the flow
            #   no longer needs to respect a prior box.)
            # * `retrain_from_scratch=True` trains a FRESH net + FRESH embedding
            #   each round on that round's kept batch ONLY. The reference does
            #   this (a new NPE_C per round); warm-starting + accumulating earlier
            #   rounds instead fits a blend of all proposals, so the posterior
            #   never tightens and the Pareto k-hat will not drop.
            training_kwargs = dict(
                force_first_round_loss=True,
                use_combined_loss=False,
                retrain_from_scratch=True,   # fresh net + embedding per round (current batch only)
                discard_prior_samples=True,
                learning_rate=1e-3,
                stop_after_epochs=10,
                training_batch_size=512,
                validation_fraction=0.2,
                clip_max_norm=1.0,
            )

        if isinstance(training_kwargs, list):
            if len(training_kwargs) != num_rounds:
                raise ValueError(
                    "`training_kwargs` length must match `num_simulations_per_round`."
                )
            training_kwargs_list = training_kwargs
        else:
            training_kwargs_list = [training_kwargs] * num_rounds

        # Cap each NDE's training length with the dedicated `max_num_epochs`
        # config knob, while letting an explicit per-round value win. This also
        # copies each dict so rounds don't share a mutable reference.
        training_kwargs_list = [
            {"max_num_epochs": max_num_epochs, **dict(round_kwargs)}
            for round_kwargs in training_kwargs_list
        ]

        if embedding is None:
            identity_embedding = IdentityEmbedding()
            embedding_list = [identity_embedding] * num_rounds
        elif isinstance(embedding, Embedding):
            embedding_list = [embedding] * num_rounds
        elif isinstance(embedding, list):
            if len(embedding) != num_rounds:
                raise ValueError("`embedding` list length must match number of rounds.")
            embedding_list = embedding
        else:
            raise TypeError("`embedding` must be an Embedding, list[Embedding], or None.")

        # `embedding_net` is the network trained jointly with the NPE (sbi
        # `embedding_net`); `None` lets `training_job` use its built-in default.
        if embedding_net is None or isinstance(embedding_net, EmbeddingNet):
            embedding_net_list = [embedding_net] * num_rounds
        elif isinstance(embedding_net, list):
            if len(embedding_net) != num_rounds:
                raise ValueError("`embedding_net` list length must match number of rounds.")
            embedding_net_list = embedding_net
        else:
            raise TypeError(
                "`embedding_net` must be an EmbeddingNet, list[EmbeddingNet], or None."
            )

        with catchtime("Running SBI inference", print_time=False) as run_time:
            proposal = None
            self.proposals = []
            previous_posteriors_list = None

            self.best_ensemble = None
            self.best_round = 0
            self.best_efficiency = -float("inf")
            self.best_khat = float("inf")
            self.best_log_Z = float("nan")
            self.best_log_Z_err = float("nan")
            self.best_weighted = None

            self.history = []
            n_sim_total = 0
            threshold_crossed = False
            extra_round_done = False

            # Multiple round inference loop
            for round_number, (
                num_simulations,
                current_embedding,
                current_embedding_net,
                current_training_kwargs,
            ) in enumerate(
                zip(num_simulations_per_round, embedding_list, embedding_net_list, training_kwargs_list),
                start=1,
            ):
                proposal, previous_posteriors_list, round_info = self.inference_round(
                    num_simulations,
                    embedding=current_embedding,
                    embedding_net=current_embedding_net,
                    proposal=proposal,
                    round_number=round_number,
                    training_kwargs=current_training_kwargs,
                    is_filter_fraction=is_filter_fraction,
                    proposal_mode=proposal_mode,
                    previous_posteriors_list=previous_posteriors_list,
                )
                self.proposals.append(proposal)

                # Record the round in the run history (drives the diagnostic plots).
                n_sim_total += round_info["n_simulations_round"]
                round_info["n_simulations_total"] = n_sim_total
                self.history.append(round_info)

                # Best-round tracking (hierarchical: reliability gate, then quality),
                # mirroring `_sixsa_dev`. k_hat (PSIS) is a RELIABILITY gate -- can
                # IS be trusted here at all? -- while efficiency/ESS is the QUALITY
                # score GIVEN reliability. A round is "reliable" iff k_hat is finite
                # AND below khat_threshold; among reliable rounds pick the highest
                # efficiency; if none is reliable yet, prefer the lowest finite k_hat
                # (least-bad), falling back to efficiency only when no finite k_hat
                # exists. Without this gate a heavy-tailed round with a spuriously
                # high efficiency could be selected and returned as the posterior.
                khat = round_info["k_hat"]
                eff = round_info["efficiency"]
                thr = khat_threshold if khat_threshold is not None else float("inf")
                this_reliable = np.isfinite(khat) and khat < thr
                best_reliable = np.isfinite(self.best_khat) and self.best_khat < thr

                if this_reliable and best_reliable:
                    take = np.isfinite(eff) and eff > self.best_efficiency
                elif this_reliable and not best_reliable:
                    take = True
                elif (not this_reliable) and best_reliable:
                    take = False
                elif np.isfinite(khat):
                    take = khat < self.best_khat
                else:
                    take = (
                        not np.isfinite(self.best_khat)
                        and np.isfinite(eff)
                        and eff > self.best_efficiency
                    )

                if take:
                    self.best_khat = khat if np.isfinite(khat) else self.best_khat
                    self.best_efficiency = (
                        eff if np.isfinite(eff) else self.best_efficiency
                    )
                    self.best_ensemble = proposal
                    self.best_log_Z = round_info["log_Z"]
                    self.best_log_Z_err = round_info["log_Z_err"]
                    self.best_round = round_number
                    self.best_weighted = self._last_weighted

                # Early stopping based on the Pareto k-hat threshold.
                if khat_threshold is not None and np.isfinite(round_info["k_hat"]):
                    if not threshold_crossed and round_info["k_hat"] < khat_threshold:
                        threshold_crossed = True
                        if not force_last_round:
                            break
                    elif threshold_crossed and not extra_round_done:
                        extra_round_done = True
                        break

                # We train a restricted proposal if not in the last round
                if round_number < num_rounds:
                    if proposal_mode == "truncated":
                        if truncated_sampling_method == "rejection":
                            warnings.warn(
                                "proposal_mode='truncated' with "
                                "truncated_sampling_method='rejection' samples the "
                                "truncated prior, which is inconsistent with SIXSA's "
                                "importance weighting (it uses the posterior density "
                                "as the proposal density) and biases the k-hat/ESS/"
                                "efficiency diagnostics and the resampled posterior. "
                                "Use truncated_sampling_method='sir'.",
                                stacklevel=2,
                            )
                        accept_reject_fn = get_density_thresholder(
                            proposal,
                            num_samples_to_estimate_support=truncated_num_samples_to_estimate_support,
                            quantile=truncated_quantile,
                        )
                        proposal = RestrictedPrior(
                            self.prior,
                            accept_reject_fn,
                            posterior=proposal,
                            sample_with=truncated_sampling_method,
                            device=device,
                        )

            # Fallback to the final proposal if best_ensemble is not populated
            if self.best_ensemble is None and self.proposals:
                self.best_ensemble = self.proposals[-1]

        # End-of-run six-panel training-history summary.
        if self.config.plot_diagnostics and self.history:
            try:
                import matplotlib.pyplot as plt

                result = plots.plot_training_history(
                    self.history,
                    khat_threshold=khat_threshold,
                    outdir=self.solver.outputfiles_basename,
                )
                if result is not None:
                    plt.close(result[0])
            except Exception:
                pass

        # Build posterior samples from the best proposal/ensemble. `sample()`
        # already returns physical-space parameters.
        n_posterior = self.config.num_posterior_samples
        posterior = self._posterior_dataframe(np.asarray(self.sample(n_posterior)))

        return FitResults(
            time=float(run_time()),
            posterior_samples=posterior,
            n_likelihood_evaluations=0,
            log_Z=self.best_log_Z,
            log_Z_err=self.best_log_Z_err,
        )

    def plot_training_history(self, save=False, outdir=None, **kwargs):
        """Build the six-panel training-history figure from ``self.history``.

        Convenience for inline display after ``solver.run()``. By default the
        figure is returned without writing a file; pass ``save=True`` (and
        optionally ``outdir=...``) to also write ``training_history.pdf``.

        Returns:
            The matplotlib ``Figure``, or ``None`` if there is no history yet.
        """
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

