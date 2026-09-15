"""Regression tests for the fakeit-time clamp against XSPEC hard limits.

XSPEC serialises a parameter's hard limits into the ``.xcm`` template at only ~6
significant figures, i.e. a hair tighter than the full-precision scipy prior support the
limits were derived from. A sample sitting on the support edge can therefore fall just
below the stored hard-min and be rejected by ``Xset.restore`` with
``"... is outside hard range ..."``. ``local_xcm_path`` now clamps each value against the
bounds parsed from the ``.xcm`` line before writing it, which removes the crash.

The clamp logic is exercised without XSPEC (``local_xcm_path`` is pure text I/O); the
end-to-end fakeit replay is guarded by the ``xspec_session`` fixture chain and auto-skips
where HEASoft is unavailable.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Mirror the other test modules: make ``import xspec`` resolve even without HEASoft so
# importing ``bsixsa.xspec`` (which imports ``xspec`` at module load) succeeds.
sys.modules.setdefault("xspec", MagicMock())

from bsixsa.xspec import get_model_block, local_xcm_path  # noqa: E402

# Hard limits as XSPEC would serialise them in a saved model block (~6 sig figs).
HARD_MIN = 1.39238
HARD_MAX = 1.58491


def _write_base_xcm(path):
    """Write a minimal ``.xcm`` whose model block holds a single free parameter.

    The value field occupies the first 15 columns (matching the contract assumed by
    ``local_xcm_path``); the trailing fields are ``delta hardmin softmin softmax hardmax``.
    """
    param_line = (
        f"{1.45:.8g}".rjust(15)
        + f"      0.1   {HARD_MIN}   {HARD_MIN}   {HARD_MAX}   {HARD_MAX}\n"
    )
    contents = (
        "statistic cstat\n"
        "model  tbabs*powerlaw\n"
        + param_line
        + "bayes off\n"
    )
    path.write_text(contents)
    return str(path)


def _read_written_value(xcm_path):
    """Return the value written for parameter index 1 of the (unnamed) model."""
    with open(xcm_path) as f:
        lines = f.readlines()
    line, _ = get_model_block(lines)[("", 1)]
    return float(line.split()[0])


@pytest.mark.parametrize(
    "raw_value",
    [
        np.nextafter(HARD_MIN, -np.inf),  # a hair below hard-min -> the crashing case
        np.nextafter(HARD_MAX, np.inf),   # a hair above hard-max
        0.5 * (HARD_MIN + HARD_MAX),      # comfortably in range -> untouched
    ],
)
def test_local_xcm_path_clamps_value_into_hard_range(tmp_path, raw_value):
    base_xcm = _write_base_xcm(tmp_path / "base.xcm")

    with local_xcm_path(
        [raw_value], [1], [""], base_xcm, tmp_dir=str(tmp_path)
    ) as produced:
        written = _read_written_value(produced)

    assert HARD_MIN <= written <= HARD_MAX


def test_local_xcm_path_leaves_in_range_value_untouched(tmp_path):
    base_xcm = _write_base_xcm(tmp_path / "base.xcm")
    in_range = 0.5 * (HARD_MIN + HARD_MAX)

    with local_xcm_path(
        [in_range], [1], [""], base_xcm, tmp_dir=str(tmp_path)
    ) as produced:
        written = _read_written_value(produced)

    assert written == pytest.approx(in_range)


def test_prior_edge_quantiles_stay_in_support():
    """First barrier: ``from_unit_cube`` keeps edge quantiles inside the scipy support."""
    from bsixsa.priors import MultipleIndependent, loguniform, norm, uniform

    prior = MultipleIndependent(
        [
            uniform(0.0, 0.5),
            loguniform(1e-5, 1e-2),
            norm(loc=0.3, scale=1.0, low=-2.0, high=2.0),
        ]
    )
    ndim = prior.ndim

    # Extreme quantiles, including the 0/1 endpoints, must map inside the support.
    edge_cube = np.array(
        [
            np.zeros(ndim),
            np.ones(ndim),
            np.full(ndim, 1e-300),
            np.full(ndim, 1.0 - 1e-16),
        ]
    )
    edge_params = prior.from_unit_cube(edge_cube)

    for j, (low, high) in enumerate(prior.bounds):
        # scipy ppf at the exact 0/1 quantile can land ~1 ULP outside its own support
        # (e.g. loguniform.ppf(0) < support()[0]); that benign rounding is precisely the
        # discrepancy the XSPEC-side clamp absorbs, so allow a tiny relative tolerance.
        tol = 1e-9 * max(abs(low), abs(high)) + 1e-300
        assert np.all(edge_params[:, j] >= low - tol)
        assert np.all(edge_params[:, j] <= high + tol)

    # Strictly-interior near-edge quantiles must have finite log-density. (At the exact
    # 0/1 endpoints scipy's own logpdf can be -inf at the support boundary, which is the
    # very fuzz the XSPEC clamp guards against, so those are excluded here.)
    interior = prior.from_unit_cube(
        np.array([np.full(ndim, 1e-6), np.full(ndim, 1.0 - 1e-6)])
    )
    assert np.all(np.isfinite(prior.log_prob(interior)))


def test_xset_restore_accepts_edge_value_clamped_to_saved_hard_range(xspec_session):
    """Real-XSPEC replay of the exact crash site (skips without HEASoft).

    Pushes a full-precision support edge into a model's hard range, lets XSPEC serialise
    it to ~6 significant figures (here the bound rounds *up*, so the saved hard-min
    exceeds the sampled edge value -- the intermittent case), then folds the edge value
    through ``local_xcm_path`` -> ``Xset.restore``. Before the clamp this aborted with
    ``"... is outside hard range ..."``; the clamp makes restore accept the value.

    The test only ever drives the clamped (succeeding) restore, so it cannot drop XSPEC
    into the interactive error prompt.
    """
    import os
    import tempfile

    xspec = xspec_session
    xspec.AllModels.clear()
    model = xspec.Model("powerlaw")

    # low_full rounds up to 1.39239 at 6 sig figs under XSPEC's model-save format.
    low_full, high_full = 1.3923857, 1.5849123
    mid = 0.5 * (low_full + high_full)
    model.powerlaw.PhoIndex = f"{mid},0.1,{low_full},{low_full},{high_full},{high_full}"

    with tempfile.TemporaryDirectory() as tmp:
        template = os.path.join(tmp, "template.xcm")
        xspec.Xset.save(template, info="m")

        with open(template) as f:
            line, _ = get_model_block(f.readlines())[("", 1)]
        saved_hard_min = float(line.split()[2])

        if not saved_hard_min > low_full:
            pytest.skip(
                "XSPEC did not round the saved hard-min above the edge value on this "
                "build, so the crashing configuration is not reproduced here."
            )

        # Fold the edge value (which sits below the saved hard-min) through the real
        # injection path. Without the clamp this is the row that aborted Xset.restore.
        with local_xcm_path([low_full], [1], [""], template, tmp_dir=tmp) as xcm:
            xspec.Xset.restore(xcm)

        restored = xspec.AllModels(1).powerlaw.PhoIndex.values[0]

    assert saved_hard_min <= restored <= high_full
