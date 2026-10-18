"""Tests for the torch regressor and the history-matching waves.

The simulator is the analytic quadratic C-stat of `tests/_analytic.QuadCstat`
(posterior sigma ~0.02 in the cube), so the true posterior and its level sets
are known exactly. The budget is kept small so that each run takes a few
seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault("xspec", MagicMock())
torch = pytest.importorskip("torch")

from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _analytic import QuadCstat  # noqa: E402

import bsixsa.restriction as restriction_pkg  # noqa: E402
import bsixsa.restriction.region as region_module  # noqa: E402
import bsixsa.restriction.regressor as regressor_module  # noqa: E402
import bsixsa.restriction.waves as waves_module  # noqa: E402
from bsixsa.restriction import HistoryMatching, Region, Restriction  # noqa: E402
from bsixsa.restriction.regressor import CstatRegressor, train_regressor  # noqa: E402
from bsixsa.restriction.rules import posterior_span  # noqa: E402
from bsixsa.restriction.waves import run_history_matching  # noqa: E402

_POSTERIOR_COV = np.array([[4e-4, 2e-4], [2e-4, 4e-4]])
_HISTORY_KEYS = {
    "wave",
    "n",
    "threshold",
    "best",
    "prior_mass",
    "delta",
    "spans",
    "purity",
    "recall",
    "margin_sigma",
    "rmse_log",
    "rmse_cstat",
    "wall",
    "recall_info",
    "seconds",
}


def _held_out(cstat: np.ndarray, seed: int = 0, fraction: float = 0.2) -> np.ndarray:
    """The waves' own validation split of a set of folds."""
    return waves_module._hold_out(cstat, fraction, np.random.default_rng(seed))


def _small_config(**overrides) -> HistoryMatching:
    kwargs = dict(
        n_waves=3,
        simulations_per_parameter=150,
        epochs=150,
        hidden=32,
        blocks=2,
        mass_probe=20_000,
        box_samples=5_000,
        plot=False,
    )
    kwargs.update(overrides)
    return HistoryMatching(**kwargs)


@pytest.fixture(scope="module")
def quad() -> QuadCstat:
    return QuadCstat([0.4, 0.6], np.linalg.inv(_POSTERIOR_COV), c0=100.0)


@pytest.fixture(scope="module")
def run(quad) -> Restriction:
    return run_history_matching(quad.cstat, 2, _small_config(seed=2026))


# ------------------------------------------------------------ regressor


def test_regressor_widths_taper_and_never_fall_below_eight():
    assert CstatRegressor(3, hidden=128, blocks=3).widths == [128, 64, 32]
    assert CstatRegressor(3, hidden=20, blocks=4).widths == [20, 10, 8, 8]
    linears = [m for m in CstatRegressor(3, hidden=32, blocks=2).net if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [(3, 32), (32, 16), (16, 1)]


def test_regressor_predict_inverts_the_delta_target():
    model = CstatRegressor(2, hidden=16, blocks=1, c_ref=1234.5678)
    with torch.no_grad():
        for p in model.net.parameters():
            p.zero_()
        model.net[-1].bias.fill_(np.log(2.0))
    pred = model.predict_cstat(np.random.default_rng(0).random((7, 2)))
    assert pred.dtype == np.float64 and pred.shape == (7,)
    np.testing.assert_allclose(pred, 1234.5678 + 2.0, rtol=0, atol=1e-5)
    assert model.c_ref == 1234.5678 and isinstance(model.c_ref, float)
    np.testing.assert_allclose(model.to_target([1236.5678]), [np.log(2.0)], atol=1e-9)

    plain = CstatRegressor(2, hidden=16, blocks=1)
    assert plain.c_ref is None
    np.testing.assert_allclose(plain.to_target([np.e, 0.0]), [1.0, np.log(1e-6)])


def test_regressor_predict_is_batched_and_accepts_tensors():
    model = CstatRegressor(2, hidden=16, blocks=2)
    u = np.random.default_rng(1).random((1001, 2))
    np.testing.assert_allclose(model.predict_cstat(u, batch=100), model.predict_cstat(u), rtol=1e-6)
    np.testing.assert_allclose(model.predict_cstat(torch.as_tensor(u)), model.predict_cstat(u), rtol=1e-6)
    assert model.predict_cstat(np.empty((0, 2))).shape == (0,)
    with pytest.raises(ValueError):
        model.predict_cstat(np.zeros((3, 5)))


def test_train_regressor_is_reproducible_and_leaves_global_torch_state_alone(quad):
    rng = np.random.default_rng(0)
    u = rng.random((400, 2))
    cstat = quad.cstat(u)
    held = _held_out(cstat)
    cfg = _small_config(epochs=30, torch_threads=1)

    threads = torch.get_num_threads()
    torch.manual_seed(99)
    expected_next = torch.rand(3)
    torch.manual_seed(99)
    model_a, info_a = train_regressor(u, cstat, cfg=cfg, held_out=held, seed=7)
    assert torch.equal(torch.rand(3), expected_next)
    assert torch.get_num_threads() == threads

    model_b, info_b = train_regressor(u, cstat, cfg=cfg, held_out=held, seed=7)
    model_c, _ = train_regressor(u, cstat, cfg=cfg, held_out=held, seed=8)
    probe = rng.random((100, 2))
    np.testing.assert_array_equal(model_a.predict_cstat(probe), model_b.predict_cstat(probe))
    assert info_a == info_b
    assert not np.array_equal(model_a.predict_cstat(probe), model_c.predict_cstat(probe))


def test_train_regressor_split_drops_infinite_rows(quad):
    rng = np.random.default_rng(3)
    u = rng.random((500, 2))
    cstat = quad.cstat(u)
    cstat[:50] = np.inf
    held = _held_out(cstat, seed=1)
    assert held.sum() == 90 and not held[:50].any()  # 20 % of the 450 finite folds
    model, info = train_regressor(u, cstat, cfg=_small_config(epochs=5), held_out=held, c_ref=90.0, seed=1)

    assert info["n_dropped"] == 50
    assert info["n_val"] == 90 and info["n_train"] == 360
    assert np.isfinite(info["rmse_log"]) and np.isfinite(info["rmse_cstat"])
    assert info["epochs"] <= 5

    # Infinite rows marked as held out are dropped too, not validated on.
    held[:10] = True
    _, info = train_regressor(u, cstat, cfg=_small_config(epochs=2), held_out=held, c_ref=90.0, seed=1)
    assert info["n_val"] == 90 and info["n_train"] == 360 and info["n_dropped"] == 50


def test_train_regressor_learns_the_quadratic_bowl(quad):
    rng = np.random.default_rng(4)
    u = rng.random((1500, 2))
    cstat = quad.cstat(u)
    best = float(cstat.min())
    model, info = train_regressor(
        u,
        cstat,
        cfg=_small_config(epochs=400, hidden=64),
        held_out=_held_out(cstat, seed=2),
        c_ref=best - 7.0,
        focus_below=best + 40.0,
        seed=2,
    )
    test = rng.random((2000, 2))
    exact = quad.cstat(test)
    pred = model.predict_cstat(test)
    # it must rank, not reproduce: Spearman correlation close to 1
    ranks = lambda x: np.argsort(np.argsort(x))  # noqa: E731
    assert np.corrcoef(ranks(pred), ranks(exact))[0, 1] > 0.99
    assert info["rmse_log"] < 0.5


def test_train_regressor_holds_out_the_given_rows(quad):
    rng = np.random.default_rng(6)
    u = rng.random((300, 2))
    cstat = quad.cstat(u)
    cstat[:10] = np.inf
    held = np.zeros(300, dtype=bool)
    held[[0, 3, 50, 51, 52, 299]] = True  # two of them non-finite

    model, info = train_regressor(u, cstat, cfg=_small_config(epochs=3), held_out=held, seed=4)

    assert info["n_val"] == 4 and info["n_train"] == 300 - 10 - 4 and info["n_dropped"] == 10
    # The validation C-stat RMSE is computed on exactly the finite held-out rows.
    val = [50, 51, 52, 299]
    rmse = np.sqrt(np.mean((model.predict_cstat(u[val]) - cstat[val]) ** 2))
    assert info["rmse_cstat"] == pytest.approx(rmse, rel=1e-12)


def test_train_regressor_refuses_a_degenerate_held_out_mask(quad):
    u = np.random.default_rng(7).random((20, 2))
    cstat = quad.cstat(u)
    cfg = _small_config(epochs=2)
    with pytest.raises(ValueError, match="validation"):
        train_regressor(u, cstat, cfg=cfg, held_out=np.zeros(20, dtype=bool), seed=0)
    with pytest.raises(ValueError, match="training"):
        train_regressor(u, cstat, cfg=cfg, held_out=np.ones(20, dtype=bool), seed=0)
    # Indices instead of a mask, or a mask of the wrong length, are refused.
    with pytest.raises(ValueError, match="boolean mask"):
        train_regressor(u, cstat, cfg=cfg, held_out=np.array([0, 1, 2]), seed=0)
    with pytest.raises(ValueError, match="boolean mask"):
        train_regressor(u, cstat, cfg=cfg, held_out=np.zeros(21, dtype=bool), seed=0)


def test_train_regressor_refuses_too_few_finite_rows():
    with pytest.raises(ValueError, match="finite"):
        train_regressor(
            np.zeros((3, 2)), np.array([1.0, np.inf, np.inf]), cfg=_small_config(), held_out=np.zeros(3, bool), seed=0
        )
    with pytest.raises(ValueError, match="shape"):
        train_regressor(np.zeros((3, 2)), np.zeros(4), cfg=_small_config(), held_out=np.zeros(4, bool), seed=0)


# ------------------------------------------------------------ waves


def test_package_exposes_run_history_matching():
    assert restriction_pkg.run_history_matching is run_history_matching
    assert "run_history_matching" in dir(restriction_pkg)
    with pytest.raises(AttributeError):
        restriction_pkg.not_a_name  # noqa: B018


def test_final_region_contains_the_posterior(run, quad):
    draws = quad.posterior_sample(20_000, rng=np.random.default_rng(1))
    assert run.region.accepts(draws).mean() >= 0.995


def test_run_bookkeeping(run):
    cfg = run.config
    assert isinstance(run, Restriction) and isinstance(run.region, Region)
    assert 1 <= len(run.history) <= cfg.n_waves
    assert run.stop_reason in {"acceptance", "threshold", "n_waves"}
    if run.stop_reason == "n_waves":
        assert len(run.history) == cfg.n_waves
    assert run.threshold >= run.best
    assert run.threshold == run.history[-1]["threshold"] == run.region.threshold
    assert run.best == pytest.approx(run.cstat.min())
    assert run.span == pytest.approx(posterior_span(2))
    assert run.prior_mass == run.history[-1]["prior_mass"]
    assert run.rmse_cstat == run.history[-1]["rmse_cstat"]

    per_wave = [row["n"] for row in run.history]
    assert run.n_simulations == len(run.cstat) == len(run.u) == sum(per_wave)
    assert per_wave == [cfg.simulations_per_parameter * 2] * len(run.history)
    np.testing.assert_array_equal(np.bincount(run.wave)[1:], per_wave)
    assert np.all(np.diff(run.wave) >= 0)

    for i, row in enumerate(run.history, start=1):
        assert _HISTORY_KEYS <= set(row)
        assert row["wave"] == i
        assert row["threshold"] >= row["best"]
        assert 0.0 < row["prior_mass"] <= 1.0
    assert run.history[0]["purity"] is None
    assert all(row["purity"] is not None for row in run.history[1:])
    # the threshold never rises except on a recall wave
    for prev, row in zip(run.history, run.history[1:]):
        assert row["threshold"] <= prev["threshold"] or row["wall"] == "recall"

    assert run.u.dtype == np.float64 and np.all((run.u >= 0.0) & (run.u <= 1.0))


def test_validation_split_is_sticky_and_the_recall_check_reads_held_out_folds(quad, monkeypatch):
    fits: list[dict] = []
    checks: list[dict] = []
    real_train, real_recall = regressor_module.train_regressor, waves_module.recall_check

    def train(u, cstat, **kwargs):
        model, info = real_train(u, cstat, **kwargs)
        held = np.asarray(kwargs["held_out"])
        val = np.flatnonzero(held & np.isfinite(cstat))
        train_rows = np.flatnonzero(~held & np.isfinite(cstat))
        assert info["n_val"] == val.size and info["n_train"] == train_rows.size
        fits.append({"n": len(cstat), "val": val, "train": train_rows, "model": model, "u": np.array(u)})
        return model, info

    def recall(**kwargs):
        checks.append(kwargs)
        return real_recall(**kwargs)

    monkeypatch.setattr(regressor_module, "train_regressor", train)
    monkeypatch.setattr(waves_module, "recall_check", recall)
    cfg = _small_config(n_waves=3, epochs=40, stop_when_threshold=0.0, seed=9)
    res = run_history_matching(quad.cstat, 2, cfg)

    assert len(fits) == len(checks) == len(res.history) == 3
    validated = np.empty(0, dtype=int)
    for w, (fit, check, row) in enumerate(zip(fits, checks, res.history), start=1):
        # Every row ever validated stays out of this and every later fit.
        assert np.intersect1d(validated, fit["train"]).size == 0
        assert np.all(np.isin(validated, fit["val"]))
        # Each wave holds out validation_fraction of its own folds.
        new = fit["val"][res.wave[fit["val"]] == w]
        assert new.size == round(cfg.validation_fraction * row["n"])
        validated = fit["val"]
        # The recall check reads exactly the held-out folds, with this fit's predictions.
        np.testing.assert_array_equal(check["cstat_val"], res.cstat[fit["val"]])
        np.testing.assert_allclose(check["pred_val"], fit["model"].predict_cstat(fit["u"][fit["val"]]))
    assert validated.size == sum(round(cfg.validation_fraction * row["n"]) for row in res.history)
    assert [row["n_validation"] for row in res.history] == [fit["val"].size for fit in fits]


def test_first_wave_is_an_exact_latin_hypercube(run):
    first = run.u[run.wave == 1]
    n = len(first)
    for j in range(first.shape[1]):
        strata = np.floor(first[:, j] * n).astype(int)
        assert sorted(strata.tolist()) == list(range(n))


def test_later_waves_are_drawn_inside_the_previous_region(run):
    if len(run.history) < 2:
        pytest.skip("the run stopped after one wave")
    # purity is the fraction of a wave's folds below the previous threshold
    for w, row in enumerate(run.history[1:], start=2):
        prev_thr = run.history[w - 2]["threshold"]
        assert row["purity"] == pytest.approx(np.mean(run.cstat[run.wave == w] <= prev_thr))


def test_box_contains_the_posterior_level_set(run, quad):
    lo, hi = run.box()
    draws = quad.posterior_sample(50_000, rng=np.random.default_rng(2))
    level = quad.cstat(draws) <= quad.c0 + chi2.ppf(0.999, 2)
    inner = draws[level]
    assert np.all(inner >= lo) and np.all(inner <= hi)
    assert run.box_info["sampling"] == "ok"
    assert np.all((lo >= 0.0) & (hi <= 1.0) & (hi > lo))


def test_run_is_deterministic_given_the_seed(run, quad):
    again = run_history_matching(quad.cstat, 2, _small_config(seed=2026))
    assert again.threshold == run.threshold
    np.testing.assert_array_equal(again.u, run.u)
    np.testing.assert_array_equal(again.cstat, run.cstat)
    np.testing.assert_array_equal(again.box()[0], run.box()[0])
    assert [row["wall"] for row in again.history] == [row["wall"] for row in run.history]


def test_plots_are_written_per_wave(quad, tmp_path):
    lines: list[str] = []
    res = run_history_matching(
        quad.cstat,
        2,
        _small_config(n_waves=2, plot=True, seed=5),
        outdir=str(tmp_path),
        param_names=["nH", "norm"],
        log=lines.append,
    )
    for w in range(1, len(res.history) + 1):
        path = tmp_path / f"restriction_wave_{w}.pdf"
        assert path.is_file() and path.stat().st_size > 1000
    assert not any("failed" in line for line in lines)
    assert sum(line.startswith(f"[restriction] wave {w}/") for w in (1, 2) for line in lines) == len(res.history)
    assert res.param_names == ["nH", "norm"]


def test_no_plots_when_plot_is_disabled(quad, tmp_path):
    run_history_matching(
        quad.cstat, 2, _small_config(n_waves=1, seed=1), outdir=str(tmp_path)
    )
    assert list(tmp_path.iterdir()) == []


def test_infinite_cstat_is_a_valid_zero_likelihood(quad):
    def simulate(u):
        cstat = quad.cstat(u)
        cstat[u[:, 0] > 0.9] = np.inf
        return cstat

    res = run_history_matching(simulate, 2, _small_config(n_waves=1, seed=3))
    assert np.isinf(res.cstat).any()
    assert np.isfinite(res.threshold)


@pytest.mark.parametrize("bad_value", [np.inf, -np.inf, np.nan])
def test_a_first_wave_without_two_finite_cstat_values_is_explained(bad_value):
    # One finite fold cannot train the surrogate: the error names the cause
    # (a model predicting no counts) and the way out, not an internal function.
    def simulate(u):
        cstat = np.full(len(u), bad_value)
        cstat[0] = 10.0
        return cstat

    with pytest.raises(ValueError, match="first wave") as err:
        run_history_matching(simulate, 2, _small_config(n_waves=1, seed=0))
    assert "only 1 of 300 folds" in str(err.value) and "restrict_prior=False" in str(err.value)


def test_argument_validation(quad):
    with pytest.raises(ValueError, match="d >= 1"):
        run_history_matching(quad.cstat, 0, _small_config(seed=0))
    with pytest.raises(ValueError, match="param_names"):
        run_history_matching(
            quad.cstat, 2, _small_config(seed=0), param_names=["a"]
        )
    with pytest.raises(ValueError, match="returned"):
        run_history_matching(lambda u: np.zeros(3), 2, _small_config(seed=0))
