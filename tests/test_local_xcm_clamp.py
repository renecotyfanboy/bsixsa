"""Tests for how ``local_xcm_path`` writes parameter values into the ``.xcm`` template.

Two properties are checked:

- the clamp against XSPEC hard limits: XSPEC saves a parameter's hard limits in the
  ``.xcm`` template with ~6 significant figures, which can put them a hair inside the
  full-precision scipy prior support they come from. A sample on the support edge can fall just
  below the saved hard-min, and ``Xset.restore`` rejects it ("... is outside hard
  range ..."). ``local_xcm_path`` clamps each value into the limits read from its line;
- full precision: values are written so that they read back as the same float64, so a
  fold sees exactly the requested parameters.

The text-level checks run without XSPEC (``local_xcm_path`` is pure text I/O); the
real-XSPEC checks use the ``xspec_session`` fixture and skip where HEASoft is unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Mirror the other test modules: make ``import xspec`` resolve even without HEASoft so
# importing ``bsixsa.xspec`` (which imports ``xspec`` at module load) succeeds.
sys.modules.setdefault("xspec", MagicMock())

from bsixsa.xspec import get_model_block, local_xcm_path, parallel_folding  # noqa: E402

# Hard limits as XSPEC would serialise them in a saved model block (~6 sig figs).
HARD_MIN = 1.39238
HARD_MAX = 1.58491


def _write_base_xcm(path):
    """Write a minimal ``.xcm`` whose model block holds a single free parameter.

    The value is right-aligned in the first 15 columns, as XSPEC writes it; the trailing
    fields are ``delta hardmin softmin softmax hardmax``.
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


@pytest.mark.parametrize(
    "in_range",
    [0.5 * (HARD_MIN + HARD_MAX), 1.4523456789012345, np.nextafter(HARD_MAX, -np.inf), np.float32(1.5)],
)
def test_local_xcm_path_writes_in_range_values_at_full_precision(tmp_path, in_range):
    """Regression: values used to be written with 8 significant digits."""
    base_xcm = _write_base_xcm(tmp_path / "base.xcm")

    with local_xcm_path(
        [in_range], [1], [""], base_xcm, tmp_dir=str(tmp_path)
    ) as produced:
        with open(produced) as f:
            line, _ = get_model_block(f.readlines())[("", 1)]

    fields = line.split()
    assert float(fields[0]) == float(in_range)
    # The other fields are kept, each separated from the value.
    assert [float(field) for field in fields[1:]] == [0.1, HARD_MIN, HARD_MIN, HARD_MAX, HARD_MAX]


def test_local_xcm_path_keeps_the_rest_of_the_template_and_deletes_the_copy(tmp_path):
    base_xcm = _write_base_xcm(tmp_path / "base.xcm")
    with local_xcm_path([1.5], [1], [""], base_xcm, tmp_dir=str(tmp_path)) as produced:
        copy = Path(produced).read_text().splitlines()
    template = Path(base_xcm).read_text().splitlines()

    assert len(copy) == len(template)
    assert copy[:2] + copy[3:] == template[:2] + template[3:]
    assert not Path(produced).exists()


def test_get_model_block_maps_named_models_and_links():
    lines = [
        "statistic cstat\n",
        "model  1:src TBabs*powerlaw\n",
        "           0.01         -1          0          0     100000      1e+06\n",
        "              2       0.01         -3         -2          9         10\n",
        "\n",
        "              1       0.01          0          0      1e+20      1e+24\n",
        "model  2:bkg powerlaw\n",
        "= src:p2\n",
        "          1e-05       0.01          0          0      1e+20      1e+24\n",
        "bayes off\n",
        "           1.23       0.01          0          0          5          5\n",
    ]
    block = get_model_block(lines)
    assert set(block) == {("src", 1), ("src", 2), ("src", 3), ("bkg", 1), ("bkg", 2)}
    assert block[("src", 3)] == (lines[5], 5)
    assert block[("bkg", 1)] == (lines[7], 7)
    assert get_model_block(["model  powerlaw\n", lines[3], "bayes off\n"]) == {("", 1): (lines[3], 1)}


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
    through ``local_xcm_path`` -> ``Xset.restore``. Without the clamp this aborts with
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
        # injection path. Without the clamp, Xset.restore would refuse this row.
        with local_xcm_path([low_full], [1], [""], template, tmp_dir=tmp) as xcm:
            xspec.Xset.restore(xcm)

        restored = xspec.AllModels(1).powerlaw.PhoIndex.values[0]

    assert saved_hard_min <= restored <= high_full


def test_fold_uses_the_parameter_values_at_full_precision(xspec_session, high_res_data_dir, monkeypatch):
    """Regression: folds used to see the values rounded to 8 significant digits.

    A fold through ``parallel_folding`` must give the same statistic as setting the same
    values directly in XSPEC, otherwise finite-difference Jacobians (LM, iminuit) see the
    rounding instead of the model.
    """
    xspec = xspec_session
    monkeypatch.chdir(high_res_data_dir)
    spectrum = xspec.Spectrum("spec_170.pha")
    xspec.AllData.ignore("bad")
    spectrum.ignore("0.0-1.5 8.0-**")
    model = xspec.Model("tbabs*(bbodyrad + powerlaw)")
    model.TBabs.nH = "0.01,-1"
    # Free parameters 2-6: bbodyrad kT, norm, powerlaw PhoIndex, norm (nH frozen).
    indices = [2, 3, 4, 5]
    limits = [(0.1, 5.0), (1e-3, 1e3), (0.5, 4.0), (1e-6, 1e-1)]
    for index, (low, high) in zip(indices, limits):
        model(index).values = f"{np.sqrt(low * high)},{0.01 * (high - low)},{low},{low},{high},{high}"
    values = np.array([1.2345678901234567, 12.345678901234567, 1.8765432109876543, 1.2345678901234567e-3])

    folded = parallel_folding(values[None, :], indices, [""] * 4, progress_bar=False, pool=None)
    folded_cstat = float(folded["cstat"][0])

    def direct_cstat(parameter_values):
        xspec.AllModels(1).setPars({i: float(v) for i, v in zip(indices, parameter_values)})
        return float(xspec.Fit.statistic)

    full_precision = direct_cstat(values)
    eight_digits = direct_cstat([float(f"{v:.8g}") for v in values])

    # The check is sensitive: 8-digit rounding moves the statistic far more than the tolerance.
    assert abs(eight_digits - full_precision) > 1e-10 * abs(full_precision)
    assert folded_cstat == pytest.approx(full_precision, rel=1e-12, abs=0)
