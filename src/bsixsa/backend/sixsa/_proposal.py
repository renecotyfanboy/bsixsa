"""Truncated proposals and flat region sampling for the SIXSA round loop.

Given a trained flow ``q`` (a member ``DirectPosterior`` or an
``EnsemblePosterior``, with ``set_default_x`` applied), the region is the
highest-density set ``{z : q(z|x_obs) >= t}`` with ``t`` the ``truncated_quantile``
quantile of ``q`` over its own draws (sbi ``get_density_thresholder``). The
proposal is the PRIOR restricted to that region (sbi ``RestrictedPrior`` with
``sample_with="sir"``): each round restricts the CLEAN prior afresh, so an early
over-restriction never persists.

Unit-Gaussian-latent note: the prior here is unbounded ``N(0, I)``, so the spec's
"flat draws on region ∩ box" become SIR draws from ``N(0, I)`` restricted to the
region -- posterior-tilted, not literally uniform. The training set is therefore
an approximation of the exact level set; the defensive-IS k_hat (see
``_importance``) is the sole arbiter of correctness, never the flow.
"""

from __future__ import annotations

import torch
from sbi.utils import RestrictedPrior, get_density_thresholder


def build_truncated_proposal(
    flow,
    prior,
    *,
    quantile: float,
    n_support: int,
    device: str = "cpu",
    restrict_from_prior: bool = True,
    previous_proposal=None,
):
    """A density-thresholded ``RestrictedPrior`` over the clean prior.

    Args:
        flow: trained posterior with ``set_default_x`` applied (member or ensemble).
        prior: the clean latent prior (``N(0, I)``) to restrict.
        quantile: highest-density-region quantile for the thresholder.
        n_support: samples used to estimate the region support.
        restrict_from_prior: restrict the clean ``prior`` (default, no stacking);
            if False, restrict ``previous_proposal`` when given (legacy stacking).
    """
    accept_reject_fn = get_density_thresholder(
        flow, quantile=quantile, num_samples_to_estimate_support=n_support
    )
    base = prior
    if not restrict_from_prior and previous_proposal is not None:
        base = previous_proposal
    return RestrictedPrior(
        base, accept_reject_fn, flow, sample_with="sir", device=device
    )


def sample_flat(
    proposal,
    n: int,
    *,
    oversampling_factor: int = 2,
    max_sampling_batch_size: int = 1000,
) -> torch.Tensor:
    """Draw ``n`` flat (SIR) samples from a truncated ``proposal``.

    ``max_sampling_batch_size`` caps peak memory/compute (performance only).

    ``oversampling_factor`` is forwarded for spec parity but is INERT in sbi:
    ``RestrictedPrior.sample`` passes it to ``sampling_importance_resampling``,
    which never reads it, so the effective candidates-per-accepted-sample is
    sbi's hardcoded ``num_candidate_samples=32`` (verified in sbi 0.25.0/0.26.1).
    32 > the spec's 2, so SIR is more thorough than configured, never less.
    """
    z = proposal.sample(
        (n,),
        oversampling_factor=oversampling_factor,
        max_sampling_batch_size=max_sampling_batch_size,
        show_progress_bars=False,
    )
    return z.detach()
