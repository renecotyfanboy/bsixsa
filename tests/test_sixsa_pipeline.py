"""A mini SIXSA round wired by hand from the modules, on the analytic 64-channel spectrum (d = 2).

Latin-hypercube harvest through `Solver.simulate`, compressor, two flows,
screen with the pooled mixture, ranking, gate certificate on fresh draws, SIR
training draws inside the winner's highest-density region, a warm compressor
refit, a second round, and the final importance sample checked against grid
quadrature. It runs once per module (a few seconds) and several tests read its
artefacts.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault("xspec", MagicMock())

torch = pytest.importorskip("torch")
pytest.importorskip("sbi")

from _analytic import AnalyticSolver, AnalyticSpectrum, grid_truth  # noqa: E402
from scipy.stats import qmc  # noqa: E402

from bsixsa.backend.config import Nautilus  # noqa: E402
from bsixsa._cube import in_cube  # noqa: E402
from bsixsa.backend.sixsa.certify import certify, rank  # noqa: E402
from bsixsa.backend.sixsa.compressor import Compressor  # noqa: E402
from bsixsa.backend.sixsa.flows import FlowProposal, MixtureProposal, train_flow  # noqa: E402
from bsixsa.backend.sixsa.sampling import importance_sample, resample  # noqa: E402
from bsixsa.backend.sixsa.truncation import sample_truncated  # noqa: E402

GATE = {"khat_threshold": 0.7, "min_efficiency": 5.0}
CHECK_DRAWS = 1000
N_MEMBERS = 2
N_HARVEST = 1500
N_TRUNCATED = 2000
NDE = {"model": "zuko_maf", "hidden_features": 32, "num_transforms": 3}
TRAINING = {"max_num_epochs": 20, "training_batch_size": 256, "stop_after_epochs": 5}


def _round(state, r: int, u: np.ndarray, counts: np.ndarray, workdir) -> SimpleNamespace:
    """Encode, train the members, screen with the pooled mixture, rank, gate."""
    solver, comp, rng = state.solver, state.compressor, state.rng
    generation = comp.generation
    z_train = comp.encode(counts)
    z_obs = comp.encode(solver.observed_counts)[0]

    flows, infos = [], []
    for k in range(N_MEMBERS):
        estimator, info = train_flow(u, z_train, nde=NDE, training=TRAINING, seed=int(rng.integers(2**31)),
                                     log_dir=workdir / f"round_{r}" / f"member_{k}")
        flows.append(FlowProposal(estimator, z_obs, name=f"round{r}/member{k}"))
        infos.append(info)
    mixture = MixtureProposal(flows, name=f"round{r}/mixture")

    before = solver.fold_counts.get("screen", 0)
    with solver.fold_phase("screen"):
        screens = certify(flows, CHECK_DRAWS, state.cstat, rng=rng, round_index=r, mixture=mixture, **GATE)
    screen_folds = solver.fold_counts["screen"] - before
    selected = rank(screens)
    before = solver.fold_counts.get("gate", 0)
    with solver.fold_phase("gate"):
        gate = certify([screens[selected].proposal], CHECK_DRAWS, state.cstat, rng=rng, round_index=r, **GATE)[0]
    gate_folds = solver.fold_counts["gate"] - before
    return SimpleNamespace(z_train=z_train, z_obs=z_obs, generation=generation, infos=infos, flows=flows,
                           mixture=mixture, screens=screens, screen_folds=screen_folds, selected=selected,
                           gate=gate, gate_folds=gate_folds)


def run_mini_round(seed: int, workdir) -> SimpleNamespace:
    spectrum = AnalyticSpectrum(d=2)
    solver = AnalyticSolver(spectrum, backend=Nautilus(trace=False), outputfiles_basename=workdir / "out")
    rng = np.random.default_rng(seed)
    state = SimpleNamespace(spectrum=spectrum, solver=solver, rng=rng)

    def cstat(u):
        u = np.asarray(u, dtype=np.float64)
        out = np.full(len(u), np.inf)
        inside = in_cube(u)
        if inside.any():
            out[inside] = solver.simulate(solver.prior.from_unit_cube(u[inside]), return_kind="cstat",
                                          progress_bar=False)["cstat"]
        return out

    def training_data(u):
        sim = solver.simulate(solver.prior.from_unit_cube(u), return_kind="full_model_counts", progress_bar=False)
        return sim["cstat"], rng.poisson(sim["total_model_counts"]).astype(np.float32)

    state.cstat = cstat
    state.u_harvest = qmc.LatinHypercube(2, rng=rng).random(N_HARVEST)
    with solver.fold_phase("harvest"):
        state.cstat_harvest, state.counts_harvest = training_data(state.u_harvest)
    state.compressor = Compressor(n_bins=solver.n_bins, d=2, latent_dim=8, n_layers=2, epochs=80, patience=8,
                                  batch_size=256, seed=seed)
    state.compressor.fit(state.counts_harvest)

    state.round0 = _round(state, 0, state.u_harvest, state.counts_harvest, workdir)
    state.winner0 = state.round0.screens[state.round0.selected].proposal
    state.u_train1, state.info_truncation = sample_truncated(
        state.winner0, n=N_TRUNCATED, quantile=1e-3, n_support=10_000, oversampling=16, rng=rng,
    )
    with solver.fold_phase("training"):
        _, state.counts_train1 = training_data(state.u_train1)
    state.refit = state.compressor.fit(state.counts_train1)
    state.round1 = _round(state, 1, state.u_train1, state.counts_train1, workdir)

    gates = [state.round0.gate, state.round1.gate]
    state.best = gates[rank(gates)]
    with solver.fold_phase("sampling"):
        state.posterior = importance_sample(state.best.proposal, cstat, target_ess=1500, max_draws=30_000,
                                            batch=5000, rng=rng, **GATE)
    rows, state.n_unique = resample(state.posterior.u, state.posterior.log_w_smoothed, 3000, rng)
    state.theta = solver.prior.from_unit_cube(rows)
    return state


@pytest.fixture(scope="module")
def mini_round(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("sixsa_pipeline")
    state = run_mini_round(0, workdir)
    state.workdir = workdir
    state.truth = grid_truth(state.spectrum)
    return state


def test_harvest_feeds_the_compressor_and_the_generations(mini_round):
    state = mini_round
    assert state.cstat_harvest.shape == (N_HARVEST,)
    assert state.counts_harvest.shape == (N_HARVEST, 64) and state.counts_harvest.dtype == np.float32
    assert state.solver.fold_counts["harvest"] == N_HARVEST
    assert state.round0.generation == 1 and state.round1.generation == 2
    assert state.refit["warm"] is True
    assert state.round0.z_train.shape == (N_HARVEST, 8) and state.round0.z_obs.shape == (8,)


def test_members_are_real_flows_logged_per_round(mini_round):
    for r, rnd in enumerate((mini_round.round0, mini_round.round1)):
        assert all(flow.d == 2 for flow in rnd.flows)
        assert rnd.mixture.members == rnd.flows
        for k, info in enumerate(rnd.infos):
            assert info["log_dir"].endswith(f"round_{r}/member_{k}")
            assert any(p.name.startswith("events.out.tfevents")
                       for p in (mini_round.workdir / f"round_{r}" / f"member_{k}").iterdir())


def test_screen_pools_member_draws_into_the_mixture_without_extra_folds(mini_round):
    for rnd in (mini_round.round0, mini_round.round1):
        *members, mix = rnd.screens
        assert [c.proposal for c in members] == rnd.flows and mix.proposal is rnd.mixture
        assert [c.n_raw for c in rnd.screens] == [CHECK_DRAWS, CHECK_DRAWS, N_MEMBERS * CHECK_DRAWS]
        np.testing.assert_array_equal(mix.u, np.concatenate([c.u for c in members]))
        np.testing.assert_array_equal(mix.cstat, np.concatenate([c.cstat for c in members]))
        assert rnd.screen_folds == sum(c.n_in for c in members) == mix.n_in
        assert rnd.gate_folds == rnd.gate.n_in and rnd.gate.n_raw == CHECK_DRAWS
        assert rnd.gate.proposal is rnd.screens[rnd.selected].proposal
        assert np.isfinite(rnd.gate.log_z) and np.isfinite(rnd.gate.log_z_err)


def test_truncated_draws_are_foldable_training_points(mini_round):
    state = mini_round
    u, info = state.u_train1, state.info_truncation
    assert u.shape == (N_TRUNCATED, 2) and u.dtype == np.float64 and np.all(in_cube(u))
    assert np.all(state.winner0.log_prob_raw(u) > info["threshold"])
    assert info["n_candidates"] == 16 * N_TRUNCATED
    assert 0.0 < info["acceptance"] <= 1.0 and 1 < info["n_unique"] <= N_TRUNCATED
    assert state.solver.fold_counts["training"] == N_TRUNCATED
    assert state.counts_train1.shape == (N_TRUNCATED, 64)


def test_final_sample_recovers_the_grid_truth(mini_round):
    state = mini_round
    post, truth = state.posterior, state.truth
    assert state.solver.fold_counts["sampling"] == post.n_in
    assert np.isfinite(post.k_hat)
    assert abs(post.log_z - truth["log_z"]) <= max(4.0 * post.log_z_err, 0.15)
    deviation = np.abs(state.theta.mean(axis=0) - truth["mean_theta"]) / truth["std_theta"]
    assert np.all(deviation < 0.25), deviation
    assert 1 < state.n_unique <= 3000


def test_fold_accounting(mini_round):
    state = mini_round
    expected = (N_HARVEST + sum(rnd.screen_folds + rnd.gate_folds for rnd in (state.round0, state.round1))
                + N_TRUNCATED + state.posterior.n_in)
    assert sum(state.solver.fold_counts.values()) == expected
    assert set(state.solver.fold_counts) == {"harvest", "screen", "gate", "training", "sampling"}
