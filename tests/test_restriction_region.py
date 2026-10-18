"""Tests for `bsixsa.restriction.region` (and its configuration) with analytic surrogates.

The surrogate is replaced by the exact quadratic C-stat of `tests/_analytic.QuadCstat`,
so the region is an ellipse whose volume, moments and bounding box are known in
closed form. No torch is needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.modules.setdefault("xspec", MagicMock())

from pydantic import ValidationError
from scipy.stats import kstest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _analytic import QuadCstat  # noqa: E402

from bsixsa.priors import MultipleIndependent, loguniform, uniform  # noqa: E402
from bsixsa.restriction import HistoryMatching, Region, Restriction  # noqa: E402
from bsixsa.restriction.rules import pad_box, posterior_span  # noqa: E402

_COV = np.array([[0.01, 0.004], [0.004, 0.02]])
_MU = np.array([0.45, 0.55])
_C0 = 10.0


class QuadModel:
    """Exact surrogate: `predict_cstat` is the quadratic C-stat itself."""

    def __init__(self, quad: QuadCstat, with_d: bool = True):
        self.quad = quad
        if with_d:
            self.d = quad.d
        self.rows_seen = 0

    def predict_cstat(self, u, batch: int = 200_000) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        self.rows_seen += len(u)
        return self.quad.cstat(u)


class ConstantModel:
    def __init__(self, value: float, d: int = 2):
        self.value = float(value)
        self.d = d
        self.inputs: list[np.ndarray] = []

    def predict_cstat(self, u, batch: int = 200_000) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        self.inputs.append(u.copy())
        return np.full(len(u), self.value)


@pytest.fixture
def quad() -> QuadCstat:
    return QuadCstat(_MU, np.linalg.inv(_COV), c0=_C0)


def _ellipse_region(quad: QuadCstat, r2: float) -> Region:
    return Region(QuadModel(quad), quad.c0 + r2)


# ------------------------------------------------------------ HistoryMatching


def test_history_matching_defaults():
    cfg = HistoryMatching()
    assert cfg.n_waves == 5
    assert cfg.simulations_per_parameter == 500
    assert cfg.target_spans == 3.0
    assert cfg.min_margin_sigma == 3.0
    assert cfg.stop_when_acceptance == 5e-3
    assert cfg.stop_when_threshold == 0.01
    assert (cfg.hidden, cfg.blocks, cfg.epochs, cfg.patience) == (128, 3, 1000, 20)
    assert (cfg.learning_rate, cfg.batch_size, cfg.validation_fraction) == (1e-3, 256, 0.2)
    assert (cfg.focus_weight, cfg.focus_factor) == (10.0, 3.0)
    assert (cfg.mass_probe, cfg.box_samples, cfg.box_padding) == (50_000, 50_000, 0.1)
    assert cfg.max_candidates == 2_000_000_000
    assert cfg.torch_threads == 1
    assert cfg.seed is None
    assert cfg.plot is True


def test_history_matching_compares_by_value():
    assert HistoryMatching(n_waves=3, seed=1) == HistoryMatching(n_waves=3, seed=1)
    assert HistoryMatching(n_waves=3) != HistoryMatching(n_waves=4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_waves": 0},
        {"simulations_per_parameter": -1},
        {"target_spans": 0.0},
        {"min_margin_sigma": -0.1},
        {"stop_when_acceptance": 0.0},
        {"stop_when_acceptance": 1.0},
        {"validation_fraction": 1.0},
        {"box_samples": -1},
        {"max_candidates": 2.5e9 + 0.5},  # whole numbers written as floats (2e9) are accepted
        {"torch_threads": 0},
        {"plot": "yes"},
        {"n_waves": True},
        {"unknown": 1},
    ],
)
def test_history_matching_refuses_invalid_values(kwargs):
    with pytest.raises(ValidationError):
        HistoryMatching(**kwargs)


def test_history_matching_budgets_are_derived_not_stored():
    cfg = HistoryMatching()
    dumped = cfg.model_dump()
    # history.json stores model_dump(), and HistoryMatching(**dumped) must round-trip.
    assert "mass_probe_budget" not in dumped and "box_candidate_budget" not in dumped
    assert HistoryMatching(**dumped) == cfg


def test_history_matching_accepts_none_threads_and_zero_box_samples():
    cfg = HistoryMatching(torch_threads=None, box_samples=0, seed=7, target_spans=2)
    assert cfg.torch_threads is None and cfg.box_samples == 0 and cfg.target_spans == 2.0


# ------------------------------------------------------------ Region.accepts


def test_accepts_masks_points_outside_the_closed_cube_even_when_predicted_low():
    model = ConstantModel(-1e9)
    region = Region(model, threshold=0.0)
    u = np.array([[1.0 + 1e-9, 0.5], [0.5, -1e-9], [0.5, 0.5], [np.nan, 0.5], [0.0, 1.0]])

    np.testing.assert_array_equal(region.accepts(u), [False, False, True, False, True])
    # the surrogate never sees a point outside the cube
    seen = np.concatenate(model.inputs)
    assert np.all((seen >= 0.0) & (seen <= 1.0))


def test_accepts_applies_the_threshold_inclusively(quad):
    point = np.array([0.5, 0.7])
    region = Region(QuadModel(quad), threshold=float(quad.cstat(point)))
    np.testing.assert_array_equal(region.accepts(np.vstack([point, _MU, [0.99, 0.01]])), [True, True, False])
    # a single 1-D point is one row
    assert Region(QuadModel(quad), threshold=_C0).accepts(_MU).tolist() == [True]


def test_accepts_without_any_point_in_the_cube_skips_the_model():
    model = ConstantModel(0.0)
    assert Region(model, 1.0).accepts(np.full((3, 2), 2.0)).tolist() == [False] * 3
    assert model.inputs == []


def test_region_dimension_is_read_from_the_model_or_given(quad):
    assert Region(QuadModel(quad), 1.0).d == 2
    assert Region(QuadModel(quad, with_d=False), 1.0, d=2).d == 2
    assert Region(QuadModel(quad), 1.0, d=np.int64(2)).d == 2


def test_region_with_unknown_dimension_is_refused_at_construction(quad):
    with pytest.raises(ValueError, match="dimension"):
        Region(QuadModel(quad, with_d=False), _C0 + 1.0)


# ------------------------------------------------------------ Region.sample


def test_sample_returns_exactly_n_accepted_rows(quad):
    region = _ellipse_region(quad, 1.0)
    draws = region.sample(1234, np.random.default_rng(0), max_candidates=10_000_000, batch=5_000)

    assert draws.shape == (1234, 2)
    assert draws.dtype == np.float64
    assert region.accepts(draws).all()


def test_sample_is_uniform_in_the_ellipse(quad):
    r2, n = 1.0, 20_000
    region = _ellipse_region(quad, r2)
    draws = region.sample(n, np.random.default_rng(1), max_candidates=10_000_000)

    # Uniform in {x^T P x <= r2} in d = 2: mean mu, covariance r2 * P^-1 / (d + 2).
    mean = draws.mean(axis=0)
    expected_cov = r2 * _COV / 4.0
    se_mean = np.sqrt(np.diag(expected_cov) / n)
    assert np.all(np.abs(mean - _MU) < 4.0 * se_mean)

    centred = draws - mean
    for i in range(2):
        for j in range(2):
            products = centred[:, i] * centred[:, j]
            se = products.std(ddof=1) / np.sqrt(n)
            assert abs(products.mean() - expected_cov[i, j]) < 4.0 * se

    # Radius: (r / R)^d ~ U(0, 1); whitened angle ~ U(-pi, pi).
    delta = draws - _MU
    radius2 = np.einsum("ni,ij,nj->n", delta, quad.P, delta) / r2
    assert kstest(radius2, "uniform").pvalue > 1e-3
    white = np.linalg.solve(np.linalg.cholesky(_COV), delta.T).T
    angle = (np.arctan2(white[:, 1], white[:, 0]) + np.pi) / (2 * np.pi)
    assert kstest(angle, "uniform").pvalue > 1e-3


def test_sample_is_deterministic_given_the_generator(quad):
    region = _ellipse_region(quad, 1.0)
    a = region.sample(500, np.random.default_rng(42), max_candidates=10_000_000)
    b = region.sample(500, np.random.default_rng(42), max_candidates=10_000_000)
    c = region.sample(500, np.random.default_rng(43), max_candidates=10_000_000)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_sample_raises_on_a_tiny_region_with_a_tiny_budget(quad):
    region = _ellipse_region(quad, 1e-8)  # prior mass ~ 4e-10
    with pytest.raises(RuntimeError, match="budget") as err:
        region.sample(10, np.random.default_rng(0), max_candidates=100_000, batch=10_000)
    assert type(err.value) is RuntimeError
    assert "HistoryMatching.max_candidates" in str(err.value)
    # It gives up once even an optimistic acceptance rate cannot fill the
    # draws: after 60 000 hitless candidates, before the budget is spent.
    assert region.model.rows_seen == 60_000


def test_sample_never_exceeds_the_candidate_budget(quad):
    # mass ~ 0.043: 1000 draws need ~23k candidates, far over 5k.
    region = _ellipse_region(quad, 1.0)
    with pytest.raises(RuntimeError):
        region.sample(1000, np.random.default_rng(0), max_candidates=5_000, batch=3_000)
    assert region.model.rows_seen <= 5_000


def test_sample_succeeds_when_the_budget_is_ample(quad):
    region = _ellipse_region(quad, 1.0)
    draws = region.sample(1000, np.random.default_rng(0), max_candidates=200_000, batch=1_000)
    assert draws.shape == (1000, 2)


def test_sample_zero_and_negative(quad):
    region = _ellipse_region(quad, 1.0)
    assert region.sample(0, np.random.default_rng(0), max_candidates=1).shape == (0, 2)
    assert region.model.rows_seen == 0
    with pytest.raises(ValueError):
        region.sample(-1, np.random.default_rng(0), max_candidates=1)
    with pytest.raises(RuntimeError, match="over the budget of -5"):
        region.sample(1, np.random.default_rng(0), max_candidates=-5)


# ------------------------------------------------------------ Region.prior_mass


def test_prior_mass_matches_the_analytic_ellipse_area(quad):
    r2 = 1.0
    region = _ellipse_region(quad, r2)
    truth = quad.level_volume(quad.c0 + r2)
    mass = region.prior_mass(np.random.default_rng(3), n_probe=200_000, max_candidates=10_000_000)

    sigma = np.sqrt(truth * (1.0 - truth) / region.last_probe_size)
    assert abs(mass - truth) < 3.0 * sigma
    assert region.last_probe_size == 200_000  # enough hits on the first probe
    assert region.last_probe_hits == round(mass * 200_000)


def test_prior_mass_grows_the_probe_until_enough_hits(quad):
    r2 = 0.0025
    region = _ellipse_region(quad, r2)
    truth = quad.level_volume(quad.c0 + r2)  # ~1e-4
    mass = region.prior_mass(np.random.default_rng(4), n_probe=10_000, max_candidates=50_000_000)

    assert region.last_probe_hits >= 100
    size = region.last_probe_size
    assert size > 10_000
    assert np.log(size / 10_000) / np.log(4) == pytest.approx(round(np.log(size / 10_000) / np.log(4)))
    sigma = np.sqrt(truth / size)
    assert abs(mass - truth) < 4.0 * sigma


def test_prior_mass_stops_at_the_budget_and_reports_zero_without_hits():
    region = Region(ConstantModel(np.inf), threshold=1.0)
    mass = region.prior_mass(np.random.default_rng(0), n_probe=1_000, max_candidates=50_000)
    assert mass == 0.0
    assert region.last_probe_size == 50_000
    assert region.last_probe_hits == 0


def test_prior_mass_probe_never_exceeds_the_budget():
    region = Region(ConstantModel(np.inf), threshold=1.0)
    region.prior_mass(np.random.default_rng(0), n_probe=100_000, max_candidates=30_000)
    assert region.last_probe_size == 30_000


def test_mass_probe_budget_resolves_the_acceptance_stop():
    # Default: 50 * mass_probe (2.5M) already exceeds 400 / 5e-3 = 80k.
    assert HistoryMatching().mass_probe_budget == 2_500_000
    # A lowered acceptance stop grows the probe until it can resolve it (the
    # fixed 50 * mass_probe cap reported a 1e-6 region as exactly 0).
    assert HistoryMatching(stop_when_acceptance=1e-7).mass_probe_budget == 2_000_000_000
    assert HistoryMatching(stop_when_acceptance=1e-6).mass_probe_budget == 400_000_000
    # ... but never beyond the candidate budget.
    assert HistoryMatching(stop_when_acceptance=1e-6, max_candidates=10_000_000).mass_probe_budget == 10_000_000
    assert isinstance(HistoryMatching().mass_probe_budget, int)


def test_prior_mass_measures_a_region_below_the_old_fixed_cap(quad):
    # mass ~ 2e-5 with mass_probe = 100: a fixed 50 * mass_probe probe (5 000
    # points) reports it as exactly 0; the budget that resolves a 1e-4
    # acceptance stop (4M points) measures it.
    r2 = 5e-4
    region = _ellipse_region(quad, r2)
    truth = quad.level_volume(quad.c0 + r2)
    cfg = HistoryMatching(mass_probe=100, stop_when_acceptance=1e-4)
    fixed = region.prior_mass(np.random.default_rng(0), n_probe=cfg.mass_probe, max_candidates=50 * cfg.mass_probe)
    assert fixed == 0.0
    budget = cfg.mass_probe_budget
    assert budget == 4_000_000
    mass = region.prior_mass(np.random.default_rng(0), n_probe=cfg.mass_probe, max_candidates=budget)
    assert region.last_probe_hits > 30
    assert abs(mass - truth) < 4.0 * np.sqrt(truth / region.last_probe_size)


# ------------------------------------------------------------ Restriction


def _restriction(quad, *, r2=4.0, n=3000, seed=123, rmse=0.5, **cfg_kwargs):
    rng = np.random.default_rng(11)
    u = rng.random((n, 2))
    cstat = quad.cstat(u)
    threshold = quad.c0 + r2
    cfg = HistoryMatching(box_samples=cfg_kwargs.pop("box_samples", 5_000), seed=seed, **cfg_kwargs)
    history = [
        {
            "wave": 1,
            "n": n,
            "threshold": threshold,
            "best": float(cstat.min()),
            "prior_mass": 0.17,
            "delta": threshold - float(cstat.min()),
            "spans": 0.3,
            "purity": None,
            "recall": 1.0,
            "rmse_cstat": rmse,
            "wall": "request",
            "recall_info": {"skipped": True, "n_core": 3},
        }
    ]
    return Restriction(
        region=Region(QuadModel(quad), threshold),
        threshold=threshold,
        best=float(cstat.min()),
        span=posterior_span(2),
        prior_mass=0.17,
        rmse_cstat=rmse,
        u=u,
        cstat=cstat,
        wave=np.ones(n, dtype=int),
        history=history,
        stop_reason="n_waves",
        n_simulations=n,
        seconds=1.5,
        config=cfg,
        param_names=["nH", "norm"],
    )


def test_restriction_inside_points(quad):
    res = _restriction(quad)
    inside = res.cstat <= res.threshold
    np.testing.assert_array_equal(res.u_in, res.u[inside])
    np.testing.assert_array_equal(res.cstat_in, res.cstat[inside])
    assert np.all(res.cstat_in <= res.threshold)
    assert res.d == 2


def test_restriction_validates_shapes(quad):
    with pytest.raises(ValueError, match="shape"):
        Restriction(
            region=Region(QuadModel(quad), 1.0),
            threshold=1.0,
            best=0.0,
            span=1.0,
            prior_mass=0.1,
            rmse_cstat=0.1,
            u=np.zeros((5, 2)),
            cstat=np.zeros(4),
            wave=np.ones(5),
            history=[],
            stop_reason="n_waves",
            n_simulations=5,
            seconds=0.0,
            config=HistoryMatching(),
        )


def test_restriction_optional_fields_default_to_none(quad):
    rng = np.random.default_rng(0)
    u = rng.random((10, 2))
    res = Restriction(
        region=Region(QuadModel(quad), 20.0),
        threshold=20.0,
        best=10.0,
        span=1.0,
        prior_mass=0.1,
        rmse_cstat=0.1,
        u=u,
        cstat=quad.cstat(u),
        wave=np.ones(10),
        history=[],
        stop_reason="n_waves",
        n_simulations=10,
        seconds=0.0,
        config=HistoryMatching(box_samples=10),
    )
    assert res.param_names is None


def test_box_contains_every_inside_point_and_is_padded(quad):
    res = _restriction(quad)
    lo, hi = res.box()
    core_lo, core_hi = res.box_core

    assert np.all(lo <= res.u_in.min(axis=0)) and np.all(hi >= res.u_in.max(axis=0))
    assert np.all(core_lo <= res.u_in.min(axis=0)) and np.all(core_hi >= res.u_in.max(axis=0))
    width = core_hi - core_lo
    np.testing.assert_allclose(core_lo - lo, 0.1 * width)
    np.testing.assert_allclose(hi - core_hi, 0.1 * width)
    assert np.all((lo >= 0.0) & (hi <= 1.0))
    assert res.box_info["sampling"] == "ok"
    assert res.box_info["n_region_samples"] == 5_000


def test_box_contains_the_analytic_level_set(quad):
    res = _restriction(quad)
    lo, hi = res.box()
    true_lo, true_hi = quad.aabb(res.threshold)
    assert np.all(lo <= true_lo) and np.all(hi >= true_hi)
    # and the unpadded core converges to it from inside
    core_lo, core_hi = res.box_core
    np.testing.assert_allclose(core_lo, true_lo, atol=0.01)
    np.testing.assert_allclose(core_hi, true_hi, atol=0.01)


def test_box_padding_follows_the_surrogate_error(quad):
    res = _restriction(quad, rmse=1.0)  # 3 * 1 / (2 * ~4) * w / 2 ~ 0.19 w > 0.1 w
    lo, hi = res.box()
    core_lo, core_hi = res.box_core
    delta = res.threshold - res.best
    expected_lo, expected_hi = pad_box(core_lo, core_hi, box_padding=0.1, rmse_cstat=1.0, delta=delta)
    np.testing.assert_array_equal(lo, expected_lo)
    np.testing.assert_array_equal(hi, expected_hi)
    width = core_hi - core_lo
    np.testing.assert_allclose(core_lo - lo, 3.0 / (2.0 * delta) * width / 2.0)
    assert np.all(core_lo - lo > 0.1 * width)


def test_box_is_cached_and_returns_copies(quad):
    res = _restriction(quad)
    with patch.object(res.region, "_sample_within", wraps=res.region._sample_within) as spy:
        first = res.box()
        first[0][:] = -1.0
        second = res.box()
        _ = res.box_core
        assert spy.call_count == 1
    assert np.all(second[0] >= 0.0)


def test_box_is_deterministic_given_the_seed(quad):
    a = _restriction(quad, seed=5).box()
    b = _restriction(quad, seed=5).box()
    c = _restriction(quad, seed=6).box()
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert not np.array_equal(a[0], c[0])


def test_box_keeps_the_draws_its_candidate_budget_yields(quad):
    res = _restriction(quad, max_candidates=10)
    lo, hi = res.box()
    assert res.box_info["sampling"] == "partial"
    assert res.box_info["n_candidates"] == res.box_info["candidate_budget"] == 10
    assert res.box_info["n_region_samples"] <= 10
    assert np.all(lo <= res.u_in.min(axis=0)) and np.all(hi >= res.u_in.max(axis=0))
    assert "candidate budget" in res.report()


def test_box_candidate_budget_is_about_one_mass_probe():
    assert HistoryMatching().box_candidate_budget == 2_500_000
    assert HistoryMatching(mass_probe=1_000, box_samples=4_000).box_candidate_budget == 200_000
    assert HistoryMatching(max_candidates=123).box_candidate_budget == 123


def test_box_of_a_small_region_spends_its_budget_not_max_candidates(quad):
    # mass ~ 1e-3: 5 000 draws would need ~5M candidates, as many surrogate
    # evaluations as max_candidates allows; the box stops at 50 * box_samples.
    res = _restriction(quad, r2=0.025, box_samples=5_000, mass_probe=1_000)
    budget = res.config.box_candidate_budget
    assert budget == 250_000 < res.config.max_candidates
    res.region.model.rows_seen = 0
    lo, hi = res.box()
    info = res.box_info
    assert info["sampling"] == "partial" and info["n_candidates"] == budget
    assert 0 < info["n_region_samples"] < 5_000
    # the folded-point acceptance test plus the draws: never max_candidates evaluations
    assert res.region.model.rows_seen == len(res.u) + budget
    assert info["seconds"] >= 0.0
    true_lo, true_hi = quad.aabb(res.threshold)
    assert np.all(lo <= true_lo) and np.all(hi >= true_hi)


def test_sample_within_returns_partial_uniform_draws(quad):
    region = _ellipse_region(quad, 1.0)  # mass ~ 0.043
    draws, spent = region._sample_within(10_000, np.random.default_rng(0), max_candidates=20_000, batch=3_000)
    assert spent == 20_000 and region.model.rows_seen == 20_000
    assert 0 < len(draws) < 10_000 and region.accepts(draws).all()
    full, spent = region._sample_within(100, np.random.default_rng(0), max_candidates=10_000_000, batch=1_000)
    assert full.shape == (100, 2) and spent < 10_000_000
    empty, spent = Region(ConstantModel(np.inf), 1.0)._sample_within(5, np.random.default_rng(0), max_candidates=0)
    assert empty.shape == (0, 2) and spent == 0


def test_box_skips_region_draws_when_box_samples_is_zero(quad):
    res = _restriction(quad, box_samples=0)
    with patch.object(res.region, "_sample_within") as spy:
        res.box()
    spy.assert_not_called()
    assert res.box_info["sampling"] == "skipped"


def test_box_is_the_full_cube_with_fewer_than_two_points(quad):
    res = _restriction(quad, box_samples=0)
    res.threshold = quad.c0 - 1.0  # nothing inside, and the region accepts nothing
    res.region = Region(QuadModel(quad), quad.c0 - 1.0)
    lo, hi = res.box()
    np.testing.assert_array_equal(lo, [0.0, 0.0])
    np.testing.assert_array_equal(hi, [1.0, 1.0])
    assert res.box_info["full_cube"] is True


def test_truncated_prior_uses_base_prior_truncated_and_caches(quad):
    res = _restriction(quad)
    base = MagicMock()
    sentinel = (object(), -1.25)
    base.truncated.return_value = sentinel

    assert res.truncated_prior(base) is sentinel
    assert res.truncated_prior(base) is sentinel
    base.truncated.assert_called_once()
    args = base.truncated.call_args.args
    lo, hi = res.box()
    np.testing.assert_array_equal(args[0], lo)
    np.testing.assert_array_equal(args[1], hi)

    other = MagicMock()
    other.truncated.return_value = (object(), 0.0)
    assert res.truncated_prior(other) is other.truncated.return_value
    other.truncated.assert_called_once()


def test_truncated_prior_with_a_real_prior(quad):
    res = _restriction(quad)
    base = MultipleIndependent([uniform(0.0, 0.5), loguniform(1e-5, 1e-2)])
    prior, log_mass = res.truncated_prior(base)
    lo, hi = res.box()
    assert log_mass == pytest.approx(np.sum(np.log(hi - lo)))
    np.testing.assert_allclose(prior.to_unit_cube(base.from_unit_cube(lo[None]))[0], [0.0, 0.0], atol=1e-7)
    assert res.truncated_prior(base)[0] is prior


def test_save_writes_the_arrays_and_the_history(quad, tmp_path):
    res = _restriction(quad)
    target = tmp_path / "nested" / "restriction"
    with patch.object(res.region, "_sample_within", wraps=res.region._sample_within) as spy:
        res.save(str(target))
    # Saving never spends region draws: the box is recorded once computed.
    spy.assert_not_called()
    with np.load(target / "restriction.npz") as data:
        assert "box_lo" not in data.files and "box_core_hi" not in data.files
    assert json.loads((target / "history.json").read_text())["box_info"] == {}

    lo, hi = res.box()
    res.save(str(target))
    with np.load(target / "restriction.npz") as data:
        np.testing.assert_array_equal(data["u"], res.u)
        np.testing.assert_array_equal(data["cstat"], res.cstat)
        np.testing.assert_array_equal(data["wave"], res.wave)
        assert float(data["threshold"]) == res.threshold
        assert float(data["best"]) == res.best
        assert float(data["span"]) == res.span
        assert float(data["prior_mass"]) == res.prior_mass
        np.testing.assert_array_equal(data["box_lo"], lo)
        np.testing.assert_array_equal(data["box_hi"], hi)
        core_lo, core_hi = res.box_core
        np.testing.assert_array_equal(data["box_core_lo"], core_lo)
        np.testing.assert_array_equal(data["box_core_hi"], core_hi)

    payload = json.loads((target / "history.json").read_text())
    assert payload["stop_reason"] == "n_waves"
    assert payload["n_simulations"] == res.n_simulations
    assert payload["seconds"] == res.seconds
    assert HistoryMatching(**payload["config"]) == res.config
    assert payload["history"][0]["purity"] is None
    assert payload["history"][0]["recall_info"] == {"skipped": True, "n_core": 3}
    assert payload["param_names"] == ["nH", "norm"]


def test_report_summarises_the_waves(quad):
    res = _restriction(quad)
    text = res.report()
    assert "'n_waves'" in text
    assert "request" in text
    assert "Bounding box" not in text  # the report never spends region draws
    res.box()
    text = res.report()
    assert "Bounding box" in text and "nH" in text and "norm" in text
    assert "Restriction(d=2" in repr(res)
