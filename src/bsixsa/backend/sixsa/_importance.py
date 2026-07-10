"""Final iterative importance-sampling refinement for the SIXSA backend.

Port of `_sixsa_dev/src/importance_sampling.py::is_on_given_posterior` (the
exact-simulator path; the optional Keras likelihood emulator is not ported).
The in-loop SIXSA rounds locate the posterior; this pass quantifies it: it
importance-samples the best round's ensemble posterior against the exact
likelihood until the pooled effective sample size reaches a target, then the
pooled Pareto-smoothed weights become the returned posterior and the pooled
RAW weights the evidence estimate.

The sampling loop is hand-rolled (`proposal.sample` + `proposal.log_prob` +
explicit weights) rather than delegated to sbi's
`ImportanceSamplingPosterior(method="importance")`. The two are mathematically
identical (see `sbi/samplers/importance/importance_sampling.py`); hand-rolling
avoids the `(theta, x_o)` potential-wrapper signature constraint and keeps the
per-sample statistics accessible.

Weight conventions (mirroring the reference):
  * RAW log-weights `log_lik + log_prior - log_q` feed the pooled ESS and the
    evidence: `log_Z = logsumexp(log_w) - log(N)` with the error and negative
    finite-sample bias of Dax et al. (2022), arXiv:2210.05686, Suppl. Eqs.
    (3)/(6)/(8). N counts ALL draws (a non-finite weight is a zero-weight
    draw, not a missing one).
  * PSIS-smoothed log-weights (arviz `psislw`, per round) feed the returned
    posterior resample only -- psislw normalizes its output, so smoothed
    weights carry no evidence information.
  * Non-finite raw weights are dropped TOGETHER with their samples before
    smoothing/pooling, so pooled samples and pooled weights always align
    (the reference filtered weights but indexed the unfiltered sample stack).
"""

from __future__ import annotations

import time

import numpy as np
import torch
from arviz import psislw
from scipy.special import logsumexp


def ess_from_log_weights(log_w) -> tuple[float, float]:
    """Effective sample size and efficiency (%) of a 1D array of log-weights."""
    log_w = np.asarray(log_w, dtype=np.float64)
    log_w = log_w[np.isfinite(log_w)]
    if log_w.size == 0:
        return 0.0, 0.0
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    ess = float(1.0 / np.sum(w**2))
    return ess, float(100.0 * ess / len(log_w))


def combined_pool_evidence(log_w_raw_list) -> tuple[float, float, float, float]:
    """Pooled ESS and log-evidence from per-round RAW log-weights.

    Returns ``(ess, log_Z, log_Z_err, log_Z_bias)``. Pass RAW importance
    weights (the psislw *input*), never the smoothed ones: psislw normalizes
    its output, so pooling smoothed weights collapses `logsumexp(.) - log(N)`
    to a constant `~ -log(N)` instead of an evidence.

    `log_Z = logsumexp(log_w_finite) - log(N_total)` where N_total counts all
    draws including non-finite-weight ones (they are zero-weight draws from
    the proposal). Error and bias follow Dax et al. (2022), Suppl. Eqs.
    (6)/(8): `err = sqrt((N - ess)/(N ess))`, `bias = -err^2/2` (negative --
    log_Z is underestimated; assumes a mass-covering proposal, which the
    per-round k_hat monitors).
    """
    log_w_all = np.concatenate(
        [np.asarray(lw, dtype=np.float64).ravel() for lw in log_w_raw_list]
    )
    n_total = len(log_w_all)
    finite = log_w_all[np.isfinite(log_w_all)]
    if finite.size == 0 or n_total == 0:
        return 0.0, float("nan"), float("nan"), float("nan")
    w = np.exp(finite - finite.max())
    w_norm = w / w.sum()
    ess = float(1.0 / np.sum(w_norm**2))
    log_z = float(logsumexp(finite) - np.log(n_total))
    log_z_err = float(np.sqrt((n_total - ess) / (n_total * ess)))
    log_z_bias = float(-0.5 * (n_total - ess) / (n_total * ess))
    return ess, log_z, log_z_err, log_z_bias


def resample_from_log_weights(
    samples: torch.Tensor, log_w, n: int, rng: np.random.Generator | None = None
) -> torch.Tensor:
    """Multinomial-resample `n` rows of `samples` with probabilities ~ exp(log_w).

    Non-finite log-weights are dropped together with their samples, so the
    returned rows always come from finite-weight draws.
    """
    if rng is None:
        rng = np.random.default_rng()
    log_w = np.asarray(log_w, dtype=np.float64).ravel()
    if len(log_w) != len(samples):
        raise ValueError(
            f"log_w has {len(log_w)} entries for {len(samples)} samples."
        )
    finite = np.isfinite(log_w)
    if not np.any(finite):
        raise ValueError("No finite log-weights to resample from.")
    samples = samples[torch.as_tensor(finite)]
    log_w = log_w[finite]
    p = np.exp(log_w - log_w.max())
    p /= p.sum()
    idx = rng.choice(len(p), size=n, replace=True, p=p)
    return samples[torch.as_tensor(idx)]


def final_importance_pass(
    *,
    proposal,
    latent_prior,
    log_likelihood_fn,
    n_per_round: int,
    target_ess: float,
    fallback_target_ess: float | None = None,
    max_rounds: int = 10,
    max_rounds_extended: int | None = None,
    hopeless_khat: float = 2.0,
    trace: bool = False,
) -> dict:
    """Iterative importance sampling on `proposal` until the pooled ESS target.

    Args:
        proposal: the best SIXSA ensemble posterior. Must expose
            ``sample(sample_shape)`` and ``log_prob(theta)`` evaluated at its
            default conditioning observation -- for an sbi
            ``EnsemblePosterior`` with ``set_default_x`` applied, ``log_prob``
            is exactly the mixture density that ``sample`` draws from, so the
            weights are consistent by construction.
        latent_prior: prior over the latent space (``log_prob`` used).
        log_likelihood_fn: callable ``theta_latent -> log-likelihood tensor``
            (for SIXSA, ``-0.5 * cstat`` via the exact XSPEC folding).
        n_per_round: simulations per IS round.
        target_ess: main-loop pooled-ESS target.
        fallback_target_ess: graceful-degradation floor. If the main loop hits
            `max_rounds` short of `target_ess`, keep running until this lower
            target, stopping early when the ESS gain plateaus (two consecutive
            rounds gaining < 1% of the fallback target). None disables.
        max_rounds: main-loop round budget.
        max_rounds_extended: hard ceiling on total rounds (main + extension);
            None -> ``5 * max_rounds``.
        hopeless_khat: if after two rounds every k_hat is >= this value the
            proposal does not cover the posterior and the pass aborts.
        trace: print per-round progress.

    Returns:
        dict with ``pooled_weighted`` (``(samples, normalized_weights)`` over
        the pooled finite draws, PSIS-smoothed -- ``None`` if hopeless),
        ``log_Z``/``log_Z_err``/``log_Z_bias``/``log_Z_corrected``, ``ess``,
        ``mean_khat``, per-round diagnostics, round/simulation counts, the
        ``reached_target_ess``/``reached_fallback_ess``/``hopeless`` flags and
        ``time_s``.
    """
    t_start = time.time()

    samples_per_round: list[torch.Tensor] = []
    log_w_smoothed_per_round: list[np.ndarray] = []
    log_w_raw_per_round: list[np.ndarray] = []
    khat_list: list[float] = []
    eff_list: list[float] = []
    n_simulations = 0
    cumulative_ess = 0.0
    log_z = log_z_err = log_z_bias = float("nan")
    hopeless = False
    i_round = 0

    def _print(msg):
        if trace:
            print(f"[SIXSA final IS] {msg}", flush=True)

    def _run_one_round():
        nonlocal n_simulations, cumulative_ess, log_z, log_z_err, log_z_bias
        t0 = time.time()
        with torch.no_grad():
            z = proposal.sample((n_per_round,)).detach()
            log_q = proposal.log_prob(z).detach().double()
        log_lik = torch.as_tensor(log_likelihood_fn(z)).double()
        log_prior = latent_prior.log_prob(z).double()
        log_w_raw = (log_lik + log_prior - log_q).cpu().numpy()
        n_simulations += int(len(z))

        # Drop non-finite weights together with their samples so the pooled
        # stacks stay aligned; the dropped draws still count in N for log_Z.
        finite = np.isfinite(log_w_raw)
        z_finite = z[torch.as_tensor(finite)].cpu()
        log_w_finite = log_w_raw[finite]

        if len(log_w_finite) > 0:
            try:
                log_w_smoothed, k_hat = psislw(log_w_finite.copy())
                k_hat = float(k_hat)
            except Exception:
                log_w_smoothed, k_hat = log_w_finite, float("nan")
        else:
            log_w_smoothed, k_hat = log_w_finite, float("inf")
        khat_list.append(k_hat)

        ess_round, eff_round = ess_from_log_weights(log_w_smoothed)
        eff_list.append(eff_round)

        samples_per_round.append(z_finite)
        log_w_smoothed_per_round.append(np.asarray(log_w_smoothed, dtype=np.float64))
        log_w_raw_per_round.append(log_w_raw)

        cumulative_ess, log_z, log_z_err, log_z_bias = combined_pool_evidence(
            log_w_raw_per_round
        )
        return k_hat, ess_round, eff_round, time.time() - t0

    _print(
        f"max_rounds={max_rounds}, target_ess={target_ess:.0f}, "
        f"n_per_round={n_per_round}"
    )

    # ---- main loop: run until target_ess or max_rounds ----
    while cumulative_ess < target_ess and i_round < max_rounds:
        i_round += 1
        k_hat, ess_round, eff_round, dt = _run_one_round()
        _print(
            f"round {i_round}/{max_rounds} ({dt:.1f}s): k_hat={k_hat:.3f}, "
            f"round ESS={ess_round:.1f} ({eff_round:.1f}%), cumulative "
            f"ESS={cumulative_ess:.1f}, logZ={log_z:.3f}+/-{log_z_err:.3f}"
        )
        # Early exit if the proposal is hopeless (k_hat stuck very high).
        if i_round == 2 and np.min(khat_list) >= hopeless_khat:
            _print(f"early exit: k_hat hopeless {khat_list}.")
            hopeless = True
            break

    # ---- extension: fall back to a lower ESS target, with plateau detection ----
    if (
        fallback_target_ess is not None
        and not hopeless
        and cumulative_ess < target_ess
        and cumulative_ess < fallback_target_ess
    ):
        hard_ceiling = (
            max_rounds_extended if max_rounds_extended is not None else 5 * max_rounds
        )
        _print(
            f"target ESS={target_ess:.0f} not reached in {max_rounds} rounds "
            f"(ESS={cumulative_ess:.1f}); extending toward fallback "
            f"ESS={fallback_target_ess:.0f} (hard cap {hard_ceiling} rounds)."
        )
        min_gain = 0.01 * fallback_target_ess
        stalls = 0
        while cumulative_ess < fallback_target_ess and i_round < hard_ceiling:
            i_round += 1
            ess_before = cumulative_ess
            k_hat, ess_round, eff_round, dt = _run_one_round()
            gain = cumulative_ess - ess_before
            _print(
                f"round {i_round}/{hard_ceiling} [extension] ({dt:.1f}s): "
                f"k_hat={k_hat:.3f}, cumulative ESS={cumulative_ess:.1f} "
                f"(+{gain:.1f})"
            )
            if gain < min_gain:
                stalls += 1
                if stalls >= 2:
                    _print(
                        f"extension stopped: ESS plateaued at "
                        f"{cumulative_ess:.1f}; more rounds of this proposal "
                        f"will not help."
                    )
                    break
            else:
                stalls = 0

    mean_khat = float(np.mean(khat_list)) if khat_list else float("nan")
    if trace:
        if mean_khat > 1.0:
            _print(
                f"PSIS k_hat = {mean_khat:.2f} >> 0.7: proposal does not "
                f"cover the posterior; ESS is meaningless."
            )
        elif mean_khat > 0.7:
            _print(f"PSIS k_hat = {mean_khat:.2f} > 0.7: IS results unreliable.")
        elif mean_khat > 0.5:
            _print(f"PSIS k_hat = {mean_khat:.2f} in [0.5, 0.7]: usable but degraded.")
        else:
            _print(f"PSIS k_hat = {mean_khat:.2f}: IS reliable.")

    result = {
        "pooled_weighted": None,
        "log_Z": log_z,
        "log_Z_err": log_z_err,
        # Negative finite-sample bias (Dax et al. 2022, Suppl. Eq. 8);
        # log_Z_corrected subtracts it.
        "log_Z_bias": log_z_bias,
        "log_Z_corrected": (
            log_z - log_z_bias if np.isfinite(log_z_bias) else log_z
        ),
        "ess": cumulative_ess,
        "mean_khat": mean_khat,
        "mean_efficiency": float(np.mean(eff_list)) if eff_list else float("nan"),
        "khat_per_round": khat_list,
        "efficiency_per_round": eff_list,
        "n_rounds": i_round,
        "n_simulations": n_simulations,
        "target_ess": target_ess,
        "fallback_target_ess": fallback_target_ess,
        "reached_target_ess": bool(cumulative_ess >= target_ess),
        "reached_fallback_ess": (
            None
            if fallback_target_ess is None
            else bool(cumulative_ess >= fallback_target_ess)
        ),
        "hopeless": hopeless,
        "time_s": time.time() - t_start,
    }

    if hopeless or not samples_per_round:
        return result

    # ---- pool the PSIS-smoothed weights once; the resample happens on demand ----
    samples_stacked = torch.cat(samples_per_round, dim=0)
    log_w_stacked = np.concatenate(log_w_smoothed_per_round)
    if len(log_w_stacked) == 0:
        return result
    w = np.exp(log_w_stacked - log_w_stacked.max())
    w_norm = torch.as_tensor(w / w.sum(), dtype=torch.float64)
    result["pooled_weighted"] = (samples_stacked, w_norm)
    return result
