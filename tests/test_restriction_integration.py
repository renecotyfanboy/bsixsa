"""End-to-end tests of the prior restriction with the real `bsixsa.restriction` package.

The history matching runs for real (torch regressor, waves, region, bounding
box) through `Solver.restrict` / `Solver.run` on the d=2
`tests/_analytic.AnalyticSolver`, and is checked against the tensor-grid
quadrature of `grid_truth`:

- the region keeps the posterior and the box encloses it;
- an oracle backend that integrates the working prior exactly shows that the
  box-mode evidence correction recovers the full-prior evidence, and that a
  box cutting posterior mass is both flagged and biased by exactly the mass it
  cut;
- a region-mode backend keeps the base prior;
- the real nautilus sampler reaches the grid evidence with and without the
  restriction (slow).
"""

from __future__ import annotations

import dataclasses
import json
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from scipy import special

pytest.importorskip("torch")

from _analytic import AnalyticSolver, AnalyticSpectrum, grid_truth, marginal_cdf  # noqa: E402

import bsixsa.backend as backend_package  # noqa: E402
from bsixsa.backend.abc import Backend  # noqa: E402
from bsixsa.backend.config import BackendConfig, Nautilus  # noqa: E402
from bsixsa.exceptions import BoxEdgeWarning  # noqa: E402
from bsixsa.restriction import HistoryMatching, Restriction  # noqa: E402
from bsixsa.solver import FitResults  # noqa: E402

SPECTRUM_SEED = 12345


def _small_config() -> HistoryMatching:
    return HistoryMatching(
        n_waves=3,
        simulations_per_parameter=200,
        hidden=32,
        blocks=2,
        epochs=200,
        patience=10,
        mass_probe=20_000,
        box_samples=20_000,
        seed=0,
        plot=False,
    )


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


def _tensor_grid(lo, hi, n_grid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nodes, per-axis spacing and log trapezoid weights of a tensor grid.

    Returns:
        `(u, h, log_w)`: nodes of shape `(n_grid**d, d)` in C order, spacings
        of shape `(d,)` and log weights of shape `(n_grid**d,)`.
    """
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    axes, log_weights = [], []
    for a, b in zip(lo, hi):
        weights = np.full(n_grid, (b - a) / (n_grid - 1))
        weights[[0, -1]] *= 0.5
        axes.append(np.linspace(a, b, n_grid))
        log_weights.append(np.log(weights))
    mesh = np.meshgrid(*axes, indexing="ij")
    u = np.stack([m.ravel() for m in mesh], axis=1)
    log_w = sum(np.meshgrid(*log_weights, indexing="ij")).ravel()
    return u, (hi - lo) / (n_grid - 1), log_w


def _draw_from_grid(u, h, log_mass, n: int, rng, lo, hi) -> np.ndarray:
    """Inverse-CDF draws of grid nodes by mass, jittered uniformly within their cell."""
    cdf = np.cumsum(np.exp(log_mass - special.logsumexp(log_mass)))
    index = np.minimum(np.searchsorted(cdf, rng.random(n) * cdf[-1]), len(cdf) - 1)
    jitter = (rng.random((n, u.shape[1])) - 0.5) * h
    return np.clip(u[index] + jitter, lo, hi)


# ---------------------------------------------------------------------------
# Oracle backends
# ---------------------------------------------------------------------------


class _OracleBoxConfig(BackendConfig):
    backend_name: ClassVar[str] = "oracle_box"

    n_grid: int = 512
    n_samples: int = 10_000
    seed: int = 0


class _OracleRegionConfig(_OracleBoxConfig):
    backend_name: ClassVar[str] = "oracle_region"
    restriction_mode: ClassVar[str] = "region"


class _OracleBackend(Backend):
    r"""Evidence and posterior of the working prior by dense quadrature on its unit cube.

    With $v = F_\pi(\theta)$ the unit cube of the working prior $\pi$ (the
    base prior, or the truncated box prior installed by the solver), the
    prior density is 1 in $v$, so $Z_\pi = \int_{[0,1]^d} e^{-C(\theta(v))/2}\,dv$.
    The likelihood goes through `Solver.log_likelihood_fn`, so the grid is
    counted in `fold_counts["backend"]`. Posterior draws are grid nodes drawn
    by mass and jittered within their cell.
    """

    name = "oracle"
    config_cls = _OracleBoxConfig

    def __init__(self, *, solver, config):
        super().__init__(solver=solver, config=config)
        self.tracer = SimpleNamespace(plot_progress=lambda: None, save=lambda: None)
        self.prior_at_construction = solver.prior
        self.saw_base_prior = solver.prior is solver.base_prior

    def run(self, **kwargs) -> FitResults:
        start = time.perf_counter()
        solver, config = self.solver, self.config
        d = solver.num_parameters
        v, h, log_w = _tensor_grid(np.zeros(d), np.ones(d), config.n_grid)
        theta = solver.prior.from_unit_cube(v)
        chunk = 1 << 16
        log_like = np.concatenate(
            [
                solver.log_likelihood_fn(theta[i : i + chunk], None, progress_bar=False)
                for i in range(0, len(theta), chunk)
            ]
        )
        log_mass = log_like + log_w
        log_z = float(special.logsumexp(log_mass))
        rng = np.random.default_rng(config.seed)
        draws = _draw_from_grid(v, h, log_mass, config.n_samples, rng, 0.0, 1.0)
        self._samples = solver.prior.from_unit_cube(draws)
        return FitResults(
            time=time.perf_counter() - start,
            posterior_samples=self._posterior_dataframe(self._samples),
            n_likelihood_evaluations=len(v),
            log_Z=log_z,
            log_Z_err=0.0,
        )

    def sample(self, n: int) -> np.ndarray:
        return self._samples[: int(n)]


@pytest.fixture
def oracle_backends(monkeypatch):
    for name, config_cls in (("oracle_box", _OracleBoxConfig), ("oracle_region", _OracleRegionConfig)):
        backend_cls = type(f"_Oracle_{name}", (_OracleBackend,), {"name": name, "config_cls": config_cls})
        monkeypatch.setitem(backend_package.BACKEND_REGISTRY, name, backend_cls)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spectrum() -> AnalyticSpectrum:
    return AnalyticSpectrum(d=2, seed=SPECTRUM_SEED)


@pytest.fixture(scope="module")
def truth(spectrum) -> dict:
    return grid_truth(spectrum)


@pytest.fixture(scope="module")
def posterior_u(spectrum, truth) -> np.ndarray:
    """20 000 posterior draws on the grid of `grid_truth`, mapped back to the cube by the prior."""
    lo, hi = truth["window_lo"], truth["window_hi"]
    u, h, log_w = _tensor_grid(lo, hi, truth["n_grid"])
    log_mass = -0.5 * spectrum.cstat_u(u) + log_w
    assert special.logsumexp(log_mass) == pytest.approx(truth["log_z"], abs=1e-9)
    draws = _draw_from_grid(u, h, log_mass, 20_000, np.random.default_rng(7), lo, hi)
    prior = spectrum.prior()
    return prior.to_unit_cube(prior.from_unit_cube(draws))


@pytest.fixture(scope="module")
def restricted(tmp_path_factory, spectrum) -> SimpleNamespace:
    """A solver that ran `restrict()` once with the small configuration."""
    solver = AnalyticSolver(
        spectrum,
        backend=_OracleBoxConfig(trace=False),
        outputfiles_basename=tmp_path_factory.mktemp("restricted") / "out",
        restrict_prior=_small_config(),
    )
    restriction = solver.restrict()
    return SimpleNamespace(solver=solver, restriction=restriction)


def _box_edge_warnings(caught) -> list[str]:
    return [str(w.message) for w in caught if issubclass(w.category, BoxEdgeWarning)]


# ---------------------------------------------------------------------------
# restrict()
# ---------------------------------------------------------------------------


def test_restrict_keeps_the_posterior_and_writes_its_outputs(restricted, truth, posterior_u):
    solver, restriction = restricted.solver, restricted.restriction

    assert isinstance(restriction, Restriction)
    assert solver.restriction is restriction
    assert restriction.config == _small_config()
    restriction.check_fingerprint(solver.fingerprint())

    # Posterior coverage of the region and of its bounding box.
    accepted = restriction.region.accepts(posterior_u)
    assert accepted.mean() >= 0.99
    lo, hi = restriction.box()
    assert np.all((posterior_u >= lo) & (posterior_u <= hi))
    assert np.all((truth["map_u"] >= lo) & (truth["map_u"] <= hi))
    # The box encloses the region, so its volume exceeds the region's mass.
    assert 0.0 < restriction.prior_mass < float(np.prod(hi - lo)) < 1.0
    assert restriction.best == pytest.approx(truth["c_min"], abs=restriction.span)

    # Every restriction fold is counted, once, under its own phase.
    assert solver.fold_counts == {"restriction": restriction.n_simulations}
    assert restriction.n_simulations == len(restriction.cstat) == sum(row["n"] for row in restriction.history)
    assert {(entry["phase"], entry["return_kind"]) for entry in solver.fold_log} == {("restriction", "cstat")}
    np.testing.assert_allclose(restriction.cstat, solver.spectrum.cstat_u(restriction.u), rtol=1e-12)

    outdir = Path(solver.outputfiles_basename) / "restriction"
    with np.load(outdir / "restriction.npz") as saved:
        np.testing.assert_array_equal(saved["u"], restriction.u)
        # restrict() never pays for the box: a region-mode backend never uses it.
        assert "box_lo" not in saved.files
    history = json.loads((outdir / "history.json").read_text())
    assert history["n_simulations"] == restriction.n_simulations
    assert history["fingerprint"] == solver.fingerprint()
    assert len(history["history"]) == len(restriction.history)
    assert not list(outdir.glob("*.pdf"))

    # A second call is served from the cache.
    assert solver.restrict() is restriction
    assert solver.fold_counts == {"restriction": restriction.n_simulations}


# ---------------------------------------------------------------------------
# run(): evidence correction against an exact oracle
# ---------------------------------------------------------------------------


def test_box_mode_evidence_correction_matches_the_grid(tmp_path, spectrum, truth, restricted, oracle_backends):
    solver = AnalyticSolver(
        spectrum,
        backend=_OracleBoxConfig(trace=False),
        outputfiles_basename=tmp_path / "out",
        restrict_prior=_small_config(),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = solver.run()

    restriction = solver.restriction
    box_prior, log_mass = restriction.truncated_prior(solver.base_prior)
    assert _box_edge_warnings(caught) == []
    assert solver.backend.prior_at_construction is box_prior and not solver.backend.saw_base_prior
    assert solver.distributions["prior"] is solver.base_prior
    assert solver.distributions["restricted_prior"] is box_prior
    # Same seed, same problem: the restriction computed inside run() is the fixture's.
    for mine, fixture in zip(restriction.box(), restricted.restriction.box()):
        np.testing.assert_array_equal(mine, fixture)

    lo, hi = restriction.box()
    assert log_mass == pytest.approx(float(np.sum(np.log(hi - lo))), rel=1e-12)
    assert log_mass < -1.0
    assert fit.log_Z_restriction_correction == log_mass
    assert abs(fit.log_Z - truth["log_z"]) <= 0.02

    counts = solver.fold_counts
    assert counts["restriction"] == restriction.n_simulations == fit.n_restriction_evaluations
    assert counts["backend"] == _OracleBoxConfig().n_grid ** 2
    assert fit.n_likelihood_evaluations == sum(counts.values())
    assert restriction.seconds > 0.0 and restriction.box_info["seconds"] > 0.0
    assert fit.restriction_time == pytest.approx(restriction.seconds + restriction.box_info["seconds"])
    # The box draws stay within about one mass probe of candidates.
    assert restriction.box_info["n_candidates"] <= 50 * max(restriction.config.mass_probe, restriction.config.box_samples)
    # The run recorded the box it computed next to the restriction.
    with np.load(Path(solver.outputfiles_basename) / "restriction" / "restriction.npz") as saved:
        np.testing.assert_array_equal(saved["box_lo"], lo)
        np.testing.assert_array_equal(saved["box_hi"], hi)

    samples = fit.posterior_samples.to_numpy()
    np.testing.assert_allclose(samples.mean(axis=0), truth["mean_theta"], atol=0.05 * truth["std_theta"].max())
    np.testing.assert_allclose(samples.std(axis=0), truth["std_theta"], rtol=0.05)


def test_a_box_cutting_the_posterior_warns_and_loses_its_mass(tmp_path, spectrum, truth, restricted, oracle_backends):
    mean, std = truth["mean_u"], truth["std_u"]
    # Gamma is cut at +-1 sigma; norm keeps +-12 sigma. The core is what a
    # `box_padding` of 0.1 would have padded into that box.
    lo = np.array([mean[0] - std[0], mean[1] - 12.0 * std[1]])
    hi = np.array([mean[0] + std[0], mean[1] + 12.0 * std[1]])
    half_core = np.array([std[0] / 1.2, 10.0 * std[1]])
    tight = dataclasses.replace(restricted.restriction)
    tight._box = (lo, hi)
    tight._box_core = (mean - half_core, mean + half_core)

    solver = AnalyticSolver(
        spectrum,
        backend=_OracleBoxConfig(trace=False),
        outputfiles_basename=tmp_path / "out",
        restrict_prior=tight,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = solver.run()

    messages = _box_edge_warnings(caught)
    assert messages
    assert all("Gamma" in message for message in messages)
    assert any("lower" in message for message in messages)
    assert any("upper" in message for message in messages)

    log_box_mass = float(np.log(marginal_cdf(truth, 0, hi[0]) - marginal_cdf(truth, 0, lo[0])))
    assert log_box_mass == pytest.approx(np.log(special.erf(1.0 / np.sqrt(2.0))), abs=0.05)
    assert fit.log_Z < truth["log_z"] - 0.2
    assert fit.log_Z == pytest.approx(truth["log_z"] + log_box_mass, abs=0.1)
    assert fit.log_Z_restriction_correction == pytest.approx(float(np.sum(np.log(hi - lo))), rel=1e-12)
    # An injected restriction is not recounted.
    assert "restriction" not in solver.fold_counts
    assert fit.n_restriction_evaluations == 0
    assert fit.n_likelihood_evaluations == solver.fold_counts["backend"]


def test_region_mode_backend_keeps_the_base_prior(tmp_path, spectrum, truth, restricted, oracle_backends):
    # A copy with no box computed yet: the shared fixture's box may already exist.
    fresh = dataclasses.replace(restricted.restriction)
    assert fresh._box is None and fresh._box_core is None
    solver = AnalyticSolver(
        spectrum,
        backend=_OracleRegionConfig(trace=False),
        outputfiles_basename=tmp_path / "out",
        restrict_prior=fresh,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = solver.run()

    assert solver.backend.saw_base_prior
    assert solver.prior is solver.base_prior
    assert solver.backend.prior_at_construction is solver.base_prior
    assert "restricted_prior" not in solver.distributions
    assert _box_edge_warnings(caught) == []
    assert fit.log_Z_restriction_correction == 0.0
    assert abs(fit.log_Z - truth["log_z"]) <= 0.02
    # The exact region a region-mode backend consumes holds its posterior.
    u = solver.base_prior.to_unit_cube(fit.posterior_samples.to_numpy())
    assert solver.restriction.region.accepts(u).mean() >= 0.99
    assert fit.n_restriction_evaluations == 0 and "restriction" not in solver.fold_counts
    # The region-mode path never pays for the bounding box's region draws.
    assert fresh._box is None and fresh.box_info == {}


# ---------------------------------------------------------------------------
# Real nautilus
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_nautilus_with_and_without_restriction_matches_the_grid(tmp_path, spectrum, truth, capsys):
    fits = {}
    for restricted_run in (False, True):
        solver = AnalyticSolver(
            spectrum,
            backend=Nautilus(num_live_points=500, n_networks=1, n_batch=100, n_pool=1, seed=0, trace=False),
            outputfiles_basename=tmp_path / f"restricted_{restricted_run}",
            restrict_prior=_small_config() if restricted_run else False,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = solver.run()
        assert _box_edge_warnings(caught) == []
        assert fit.n_likelihood_evaluations == sum(solver.fold_counts.values())
        fits[restricted_run] = (solver, fit)

    unrestricted, restricted_fit = fits[False][1], fits[True][1]
    restricted_solver = fits[True][0]
    backend_calls = {
        key: fit.n_likelihood_evaluations - fit.n_restriction_evaluations for key, (_, fit) in fits.items()
    }
    summary = (
        f"grid log Z {truth['log_z']:.4f}; nautilus log Z {unrestricted.log_Z:.4f} "
        f"({backend_calls[False]} calls) unrestricted, {restricted_fit.log_Z:.4f} "
        f"({backend_calls[True]} calls + {restricted_fit.n_restriction_evaluations} restriction folds, "
        f"log M_B {restricted_fit.log_Z_restriction_correction:.4f}) restricted"
    )
    with capsys.disabled():
        print(f"\n[nautilus] {summary}")

    for fit in (unrestricted, restricted_fit):
        assert abs(fit.log_Z - truth["log_z"]) <= 0.15, summary
        mean = fit.posterior_samples.mean().to_numpy()
        np.testing.assert_allclose(mean, truth["mean_theta"], atol=0.25 * truth["std_theta"].max())
    assert unrestricted.n_restriction_evaluations == 0
    assert restricted_fit.n_restriction_evaluations == restricted_solver.restriction.n_simulations
    assert restricted_solver.fold_counts["restriction"] == restricted_solver.restriction.n_simulations
    assert restricted_fit.log_Z_restriction_correction < -1.0
    assert restricted_solver.prior is not restricted_solver.base_prior
    # The box spares the sampler the prior volume it removed.
    assert backend_calls[True] < backend_calls[False], summary
    # Post-run draws stay inside the installed box.
    bounds = np.asarray([dist.support() for dist in restricted_solver.prior.dists])
    draws = restricted_solver.backend.sample(2000)
    assert np.all((draws >= bounds[:, 0]) & (draws <= bounds[:, 1]))
