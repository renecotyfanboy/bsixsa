"""Integration checks using real XSPEC and the bundled PN response."""
import numpy as np
import pytest

from bsixsa import HistoryMatching, Nautilus, Solver
from bsixsa.priors import loguniform, uniform

pytestmark = pytest.mark.xspec


@pytest.fixture
def fake_solver(xspec_session, low_res_data_dir, tmp_path):
    xs = xspec_session
    xs.Xset.seed = 42
    xs.Model("powerlaw", setPars={1: 1.8, 2: 1e-3})
    xs.AllData.fakeit(1, [xs.FakeitSettings(response=str(low_res_data_dir / "PN.rmf"),
                         arf=str(low_res_data_dir / "PN.arf"), exposure=5000,
                         fileName=str(tmp_path / "synthetic.fak"))], applyStats=True)
    xs.AllData.ignore("bad")
    xs.AllData(1).ignore("0.0-0.3 10.0-**")
    return Solver([("powerlaw", "PhoIndex", uniform(1, 3)),
                   ("powerlaw", "norm", loguniform(1e-4, 1e-2))],
                  outputfiles_basename=str(tmp_path / "run"), n_jobs=2,
                  backend=Nautilus(trace=False))


def test_real_fold_counts_and_cash_are_consistent(fake_solver):
    solver = fake_solver
    points = solver.prior.sample(8, random_state=0)
    with solver.pool_scope():
        out = solver.simulate(points, progress_bar=False)
        logl = solver.log_likelihood_fn(points, None, progress_bar=False)
    np.testing.assert_allclose(logl, -0.5 * out["cstat"])
    mu, obs = out["total_model_counts"], solver.observed_counts
    from scipy.special import xlogy
    cash = 2 * np.sum(mu - obs + xlogy(obs, obs) - xlogy(obs, mu), axis=1)
    np.testing.assert_allclose(cash, out["cstat"], rtol=2e-5, atol=1e-3)
    assert sum(solver.fold_counts.values()) == 16
    assert solver.pool is None


def test_real_history_matching_uses_solver_folds(fake_solver):
    from bsixsa._cube import to_physical
    from bsixsa.restriction import run_history_matching

    solver = fake_solver
    cfg = HistoryMatching(n_waves=2, simulations_per_parameter=100, hidden=32,
                          blocks=2, epochs=100, patience=10, mass_probe=2000, seed=0)

    def cstat(u):
        return solver.simulate(to_physical(solver.prior, u), return_kind="cstat", progress_bar=False)["cstat"]

    with solver.pool_scope():
        solver._get_pool()  # fork the workers before the surrogate loads torch
        restriction = run_history_matching(cstat, solver.num_parameters, cfg)
    assert restriction.n_simulations == sum(solver.fold_counts.values())
    assert restriction.region.accepts(restriction.u[np.argmin(restriction.cstat)]).item()
