"""Wiring tests for SIXSA truncated proposals (`_proposal.py`).

The heavy behaviour (SIR from a trained flow) is exercised end-to-end by the
benchmark; here we only pin the wiring: the clean prior is restricted, the
thresholder gets the configured quantile/support, and `sample_flat` forwards the
oversampling/batch knobs.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.modules.setdefault("xspec", MagicMock())


def test_build_truncated_proposal_restricts_clean_prior():
    from bsixsa.backend.sixsa import _proposal

    prior = MagicMock(name="prior")
    flow = MagicMock(name="flow")
    fn = MagicMock(name="accept_reject_fn")

    with patch.object(_proposal, "get_density_thresholder", return_value=fn) as thr, \
         patch.object(_proposal, "RestrictedPrior") as rp:
        _proposal.build_truncated_proposal(
            flow, prior, quantile=1e-3, n_support=10_000, device="cpu"
        )

    thr.assert_called_once_with(
        flow, quantile=1e-3, num_samples_to_estimate_support=10_000
    )
    args, kwargs = rp.call_args
    # RestrictedPrior(base=prior, accept_reject_fn, flow, sample_with="sir", ...)
    assert args[0] is prior
    assert args[1] is fn
    assert args[2] is flow
    assert kwargs["sample_with"] == "sir"


def test_build_truncated_proposal_legacy_stacking():
    from bsixsa.backend.sixsa import _proposal

    prior = MagicMock(name="prior")
    prev = MagicMock(name="previous_proposal")
    flow = MagicMock(name="flow")

    with patch.object(_proposal, "get_density_thresholder", return_value=MagicMock()), \
         patch.object(_proposal, "RestrictedPrior") as rp:
        _proposal.build_truncated_proposal(
            flow, prior, quantile=1e-3, n_support=100,
            restrict_from_prior=False, previous_proposal=prev,
        )
    assert rp.call_args.args[0] is prev  # restricts the previous proposal


def test_sample_flat_forwards_knobs():
    from bsixsa.backend.sixsa import _proposal

    proposal = MagicMock()
    proposal.sample.return_value = MagicMock(detach=lambda: "z")
    out = _proposal.sample_flat(
        proposal, 128, oversampling_factor=2, max_sampling_batch_size=64
    )
    assert out == "z"
    _, kwargs = proposal.sample.call_args
    assert kwargs["oversampling_factor"] == 2
    assert kwargs["max_sampling_batch_size"] == 64
    assert kwargs["show_progress_bars"] is False
    assert proposal.sample.call_args.args[0] == (128,)


def test_oversampling_factor_is_inert_in_sbi():
    """Pin the known sbi quirk documented on `truncated_oversampling_factor`.

    `RestrictedPrior.sample` accepts `oversampling_factor` but hands it to
    `sampling_importance_resampling`, which never reads it -- the effective
    candidates-per-accepted-sample is sbi's hardcoded `num_candidate_samples`
    (32). If a future sbi starts honouring the knob, this test fails and the
    config docstring must be updated.
    """
    import inspect

    pytest.importorskip("sbi")
    from sbi.samplers.importance.sir import sampling_importance_resampling as sir
    from sbi.utils import RestrictedPrior

    params = inspect.signature(sir).parameters
    assert "oversampling_factor" not in params
    assert params["num_candidate_samples"].default == 32
    assert "oversampling_factor" not in inspect.getsource(sir)
    # RestrictedPrior still advertises the (ineffective) kwarg.
    assert "oversampling_factor" in inspect.signature(RestrictedPrior.sample).parameters
