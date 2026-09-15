"""Integration of the Phase 3 SIXSA modules, wired by hand without the orchestrator.

Two groups of tests:

- **Contracts between modules.** A `MixtureProposal` is stratified: its draws
  come from a deterministic allocation, and its equal-weight `log_prob_raw` is
  the density of that design only when the number of members divides the
  number of draws. `certify.draw_raw` weighs it with the realised allocation,
  and both `certify_many`/`certify_fresh` and `sample_final` go through it.
- **A mini SIXSA round** on the analytic 64-channel spectrum (d = 2): LHS
  harvest, compressor, two NPE members, screen with the pooled mixture, tiered
  selection, fresh gate certificate, truncated training draws from both the
  default ``"sir"`` sampler and the exact ``"flow"`` sampler, a warm compressor
  refit, a second round, and the planned final importance sample checked
  against grid quadrature. It runs once per module (a few seconds) and several
  tests read its artefacts.

The round is wired exactly as far as the modules go: it forces truncation on
the round-0 winner even though the orchestrator's tier rule would fall back to
the region at that tier, so that the samplers see a real flow.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault("xspec", MagicMock())

torch = pytest.importorskip("torch")
pytest.importorskip("sbi")

from _analytic import AnalyticSolver, AnalyticSpectrum, GaussianProposal, QuadCstat, grid_truth  # noqa: E402

from bsixsa._cube import child_rng, in_cube, latin_hypercube, to_physical  # noqa: E402
from bsixsa.backend.config import Nautilus  # noqa: E402
from bsixsa.backend.sixsa import budget  # noqa: E402
from bsixsa.backend.sixsa._simulator import CubeSimulator  # noqa: E402
from bsixsa.backend.sixsa.certify import certify_fresh, certify_many, draw_raw, rank_candidates  # noqa: E402
from bsixsa.backend.sixsa.compressor import Compressor  # noqa: E402
from bsixsa.backend.sixsa.flows import FlowProposal, MixtureProposal, member_seeds, train_members  # noqa: E402
from bsixsa.backend.sixsa.sampling import resample, sample_final  # noqa: E402
from bsixsa.backend.sixsa.truncation import sample_truncated  # noqa: E402

GATE = {"khat_threshold": 0.7, "khat_safe": 0.5, "min_efficiency": 5.0}
SCREEN_TOLERANCE = 0.05
CHECK_DRAWS = 1000
N_MEMBERS = 2
N_TRUNCATED = 2000
TRUNCATED_QUANTILE = 1e-3
TRUNCATED_SUPPORT = 10_000
TARGET_ESS = 1500
MAX_SAMPLING_DRAWS = 30_000
N_POSTERIOR = 3000
NDE = {"model": "zuko_maf", "hidden_features": 32, "num_transforms": 3}
TRAINING = {"max_num_epochs": 20, "training_batch_size": 256, "stop_after_epochs": 5}

# Stream purposes under the run's SeedSequence (member seeds use 10).
_LHS, _COUNTS, _SCREEN, _FRESH, _SIR, _FLOW, _FINAL, _RESAMPLE = 1, 2, 20, 21, 30, 31, 40, 41


# ---------------------------------------------------------------------------
# Contracts: stratified mixtures in certify and sampling
# ---------------------------------------------------------------------------


class _QuadSimulator:
    """Cube-simulator contract (`cstat` only) on a quadratic statistic."""

    def __init__(self, quad: QuadCstat):
        self.quad = quad
        self.calls = 0

    def cstat(self, u, *, phase: str) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        self.calls += 1
        out = np.full(len(u), np.inf)
        inside = in_cube(u)
        out[inside] = self.quad.cstat(u[inside])
        return out


def _three_members(sd: float = 0.03) -> list[GaussianProposal]:
    cov = np.eye(2) * sd**2
    return [GaussianProposal([x, 0.5], cov, name=f"g{k}") for k, x in enumerate((0.2, 0.5, 0.8))]


def test_draw_raw_weighs_a_mixture_with_its_realised_allocation():
    mixture = MixtureProposal(_three_members(), name="mix")

    u, log_q = draw_raw(mixture, 1000, np.random.default_rng(0))

    np.testing.assert_array_equal(u, mixture.sample_raw(1000, np.random.default_rng(0)))
    assert mixture.allocation(1000) == [334, 333, 333]
    np.testing.assert_array_equal(log_q, mixture.log_prob_stratified(u, [334, 333, 333]))
    # 1000 is not a multiple of 3: the equal-weight density is another design.
    assert not np.allclose(log_q, mixture.log_prob_raw(u), rtol=0.0, atol=1e-6)


def test_draw_raw_leaves_plain_proposals_on_their_raw_density():
    proposal = GaussianProposal([0.5, 0.5], np.eye(2) * 0.01)

    u, log_q = draw_raw(proposal, 7, np.random.default_rng(1))

    np.testing.assert_array_equal(u, proposal.sample_raw(7, np.random.default_rng(1)))
    np.testing.assert_array_equal(log_q, proposal.log_prob_raw(u))


def test_mixture_evidence_is_unbiased_when_the_members_do_not_divide_the_draws():
    # Well-separated members and a target under member 0 only; 4 draws are
    # allocated [2, 1, 1]. Weighing them with the equal-weight density would
    # estimate K * n_0 / N * Z = 1.5 Z; the stratified density estimates Z.
    members = _three_members()
    mixture = MixtureProposal(members, name="mix")
    quad = QuadCstat([0.2, 0.5], np.eye(2) / 0.02**2, c0=0.0)
    sim = _QuadSimulator(quad)
    z_true = np.exp(quad.log_evidence_unbounded())
    rng = np.random.default_rng(2)

    ratios = []
    naive = []
    for _ in range(3000):
        cert = certify_fresh(mixture, 4, sim, rng=rng, phase="gate", round_index=0, **GATE)
        ratios.append(np.exp(cert.log_z) / z_true)
        log_w_naive = -0.5 * cert.cstat - mixture.log_prob_raw(cert.u)
        naive.append(np.mean(np.exp(log_w_naive)) / z_true)
    ratios = np.asarray(ratios)

    standard_error = ratios.std(ddof=1) / np.sqrt(ratios.size)
    assert abs(ratios.mean() - 1.0) < 4.0 * standard_error
    assert standard_error < 0.02
    assert np.mean(naive) == pytest.approx(1.5, abs=0.06)
    assert sim.calls == 3000


def test_sample_final_weighs_each_mixture_chunk_with_its_allocation():
    mixture = MixtureProposal(_three_members(sd=0.05), name="mix")
    quad = QuadCstat([0.5, 0.5], np.eye(2) / 0.08**2, c0=3.0)

    post = sample_final(
        mixture,
        _QuadSimulator(quad),
        n_planned=2500,
        target_ess=1.0,
        max_draws=2500,
        rng=np.random.default_rng(3),
        certified=True,
        chunk=1000,
    )

    rng = np.random.default_rng(3)
    u_parts, log_q_parts = [], []
    for m in (1000, 1000, 500):
        counts = mixture.allocation(m)
        u = mixture.sample_stratified(counts, rng)
        u_parts.append(u)
        log_q_parts.append(mixture.log_prob_stratified(u, counts))
    u = np.concatenate(u_parts)
    log_q = np.concatenate(log_q_parts)
    inside = in_cube(u)

    assert post.n_top_up == 0
    np.testing.assert_array_equal(post.u, u)
    np.testing.assert_allclose(post.log_w[inside], -0.5 * quad.cstat(u[inside]) - log_q[inside], rtol=0, atol=1e-12)
    assert np.all(np.isneginf(post.log_w[~inside]))


# ---------------------------------------------------------------------------
# Mini SIXSA round on the analytic solver
# ---------------------------------------------------------------------------


def _fold_total(solver) -> int:
    return int(sum(solver.fold_counts.values()))


def _round(state, r: int, u: np.ndarray, counts: np.ndarray) -> SimpleNamespace:
    """Encode, train the members, screen with the pooled mixture, select, gate."""
    sim, comp, ss, timings = state.sim, state.compressor, state.seed_seq, state.timings
    solver = state.solver

    t0 = time.perf_counter()
    z_train, generation = comp.encode_with_generation(counts)
    z_obs = comp.encode(sim.x_obs)
    comp.check_generation(generation, "z_train")
    timings[f"encode_{r}"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    seeds = member_seeds(ss, r, N_MEMBERS)
    estimators, infos = train_members(u, z_train, nde=NDE, training=TRAINING, seeds=seeds, n_jobs=1)
    timings[f"train_{r}"] = time.perf_counter() - t0
    flows = [
        FlowProposal(est, z_obs[0], name=f"round{r}/member{k}", generation=generation)
        for k, est in enumerate(estimators)
    ]
    mixture = MixtureProposal(flows, name=f"round{r}/mixture")

    t0 = time.perf_counter()
    before = solver.fold_counts.get("screen", 0)
    screens = certify_many(
        flows,
        CHECK_DRAWS,
        sim,
        rng=child_rng(ss, _SCREEN, r),
        phase="screen",
        round_index=r,
        mixture=mixture,
        **GATE,
    )
    screen_folds = solver.fold_counts["screen"] - before
    selected = rank_candidates(screens, khat_safe=GATE["khat_safe"], tolerance=SCREEN_TOLERANCE)
    timings[f"screen_{r}"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    before = solver.fold_counts.get("gate", 0)
    fresh = certify_fresh(
        screens[selected].proposal,
        CHECK_DRAWS,
        sim,
        rng=child_rng(ss, _FRESH, r),
        phase="gate",
        round_index=r,
        **GATE,
    )
    gate_folds = solver.fold_counts["gate"] - before
    timings[f"fresh_{r}"] = time.perf_counter() - t0

    return SimpleNamespace(
        z_train=z_train,
        z_obs=z_obs,
        generation=generation,
        seeds=seeds,
        infos=infos,
        flows=flows,
        mixture=mixture,
        screens=screens,
        screen_folds=screen_folds,
        selected=selected,
        fresh=fresh,
        gate_folds=gate_folds,
    )


def run_mini_round(seed: int, workdir: Path) -> SimpleNamespace:
    """Two hand-wired SIXSA rounds and the final sample; returns every artefact."""
    timings: dict[str, float] = {}
    spectrum = AnalyticSpectrum(d=2)
    solver = AnalyticSolver(spectrum, backend=Nautilus(trace=False), outputfiles_basename=workdir / "out")
    ss = np.random.SeedSequence(seed)
    sim = CubeSimulator(solver, seed_seq=ss)
    state = SimpleNamespace(
        spectrum=spectrum, solver=solver, seed_seq=ss, sim=sim, timings=timings, torch_threads=torch.get_num_threads()
    )

    # (a) harvest
    t0 = time.perf_counter()
    state.n_training = budget.training_set_size(spectrum.d, sim.n_bins)
    state.u_harvest = latin_hypercube(spectrum.d, state.n_training, child_rng(ss, _LHS))
    state.cstat_harvest, state.counts_harvest = sim.counts(state.u_harvest, phase="harvest", key=(_COUNTS, 0))
    timings["harvest"] = time.perf_counter() - t0

    # (b) compressor
    t0 = time.perf_counter()
    state.compressor = Compressor(
        n_bins=sim.n_bins, d=spectrum.d, latent_dim=8, n_layers=2, epochs=80, patience=8, batch_size=256, seed=seed
    )
    state.compressor.fit(state.counts_harvest)
    timings["compressor_0"] = time.perf_counter() - t0

    # (c, d) round 0
    state.round0 = _round(state, 0, state.u_harvest, state.counts_harvest)
    winner = state.round0.screens[state.round0.selected].proposal

    # (e) truncated training draws, both samplers, folded with counts
    t0 = time.perf_counter()
    state.u_sir, state.info_sir = sample_truncated(
        winner,
        n=N_TRUNCATED,
        quantile=TRUNCATED_QUANTILE,
        n_support=TRUNCATED_SUPPORT,
        method="sir",
        rng=child_rng(ss, _SIR, 0),
    )
    timings["truncate_sir"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    state.u_flow, state.info_flow = sample_truncated(
        winner,
        n=N_TRUNCATED,
        quantile=TRUNCATED_QUANTILE,
        n_support=TRUNCATED_SUPPORT,
        method="flow",
        rng=child_rng(ss, _FLOW, 0),
    )
    timings["truncate_flow"] = time.perf_counter() - t0
    state.winner0 = winner

    t0 = time.perf_counter()
    _, counts_sir = sim.counts(state.u_sir, phase="training", key=(_COUNTS, 1))
    _, counts_flow = sim.counts(state.u_flow, phase="training", key=(_COUNTS, 2))
    state.u_train1 = np.concatenate([state.u_sir, state.u_flow])
    state.counts_train1 = np.concatenate([counts_sir, counts_flow])
    timings["fold_training_1"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    state.refit = state.compressor.fit(state.counts_train1)
    timings["compressor_1"] = time.perf_counter() - t0

    # round 1: retrain and certify
    state.round1 = _round(state, 1, state.u_train1, state.counts_train1)

    # (f) best round over every fresh certificate, planned final sample
    fresh_all = [state.round0.fresh, state.round1.fresh]
    state.best = fresh_all[rank_candidates(fresh_all, khat_safe=GATE["khat_safe"], tolerance=SCREEN_TOLERANCE)]
    state.n_planned = budget.final_draws(
        target_ess=TARGET_ESS, ess_fresh=state.best.ess, n_fresh=state.best.n_raw, max_draws=MAX_SAMPLING_DRAWS
    )
    t0 = time.perf_counter()
    before = solver.fold_counts.get("sampling", 0)
    state.posterior = sample_final(
        state.best.proposal,
        sim,
        n_planned=state.n_planned,
        target_ess=TARGET_ESS,
        max_draws=MAX_SAMPLING_DRAWS,
        rng=child_rng(ss, _FINAL),
        certified=state.best.passed,
        khat_threshold=GATE["khat_threshold"],
    )
    state.sampling_folds = solver.fold_counts["sampling"] - before
    rows, state.n_unique = resample(
        state.posterior.u, state.posterior.log_w_smoothed, N_POSTERIOR, child_rng(ss, _RESAMPLE)
    )
    state.theta = to_physical(solver.base_prior, rows)
    timings["final"] = time.perf_counter() - t0
    timings["total"] = sum(timings.values())
    return state


@pytest.fixture(scope="module")
def mini_round(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("sixsa_pipeline")
    previous = Path.cwd()
    os.chdir(workdir)
    try:
        state = run_mini_round(0, workdir)
    finally:
        os.chdir(previous)
    state.workdir = workdir
    state.truth = grid_truth(state.spectrum)
    return state


def test_harvest_feeds_the_compressor_and_the_generation_contract(mini_round):
    state = mini_round
    n = state.n_training

    assert n == 1500
    assert state.cstat_harvest.shape == (n,)
    assert state.counts_harvest.shape == (n, 64) and state.counts_harvest.dtype == np.float32
    assert state.solver.fold_counts["harvest"] == n
    assert state.round0.generation == 1 and state.round1.generation == 2
    assert state.refit["warm"] is True
    assert state.round0.z_train.shape == (n, 8) and state.round0.z_obs.shape == (1, 8)
    assert [flow.generation for flow in state.round1.flows] == [2, 2]
    with pytest.raises(RuntimeError, match="generation 1"):
        state.compressor.check_generation(state.round0.generation, "round-0 z_obs")


def test_members_are_real_flows_with_their_seeds(mini_round):
    for r, rnd in enumerate((mini_round.round0, mini_round.round1)):
        assert rnd.seeds == member_seeds(mini_round.seed_seq, r, N_MEMBERS)
        assert [info["seed"] for info in rnd.infos] == rnd.seeds
        assert {info["executor"] for info in rnd.infos} == {"in_process"}
        assert all(flow.d == 2 for flow in rnd.flows)
        assert rnd.mixture.members == rnd.flows


def test_screen_pools_member_draws_into_the_mixture_without_extra_folds(mini_round):
    for rnd in (mini_round.round0, mini_round.round1):
        *members, mix = rnd.screens

        assert [c.proposal for c in members] == rnd.flows and mix.proposal is rnd.mixture
        assert [c.n_raw for c in rnd.screens] == [CHECK_DRAWS, CHECK_DRAWS, N_MEMBERS * CHECK_DRAWS]
        np.testing.assert_array_equal(mix.u, np.concatenate([c.u for c in members]))
        np.testing.assert_array_equal(mix.cstat, np.concatenate([c.cstat for c in members]))
        # One fold per in-cube member draw; the mixture reuses them.
        assert rnd.screen_folds == sum(c.n_in for c in members) == mix.n_in
        assert rnd.gate_folds == rnd.fresh.n_in
        assert rnd.fresh.kind == "fresh" and rnd.fresh.n_raw == CHECK_DRAWS
        assert rnd.fresh.proposal is rnd.screens[rnd.selected].proposal
        assert np.isfinite(rnd.fresh.log_z) and np.isfinite(rnd.fresh.log_z_err)


def test_truncated_draws_are_foldable_training_points(mini_round):
    state = mini_round

    for u in (state.u_sir, state.u_flow):
        assert u.shape == (N_TRUNCATED, 2) and u.dtype == np.float64
        assert np.all(in_cube(u))
    assert state.info_sir["method_used"] == "sir"
    # The kept v2 quirk: sbi weighs 32 candidates per draw.
    assert state.info_sir["n_candidates"] == 32 * N_TRUNCATED
    assert 0.0 <= state.info_sir["fraction_outside_contour"] < 0.1
    assert state.info_flow["method_used"] == "flow" and state.info_flow["fallback"] is None
    assert np.all(state.winner0.log_prob_raw(state.u_flow) > state.info_flow["threshold"])
    assert state.solver.fold_counts["training"] == 2 * N_TRUNCATED
    assert state.counts_train1.shape == (2 * N_TRUNCATED, 64)


def test_final_sample_recovers_the_grid_truth(mini_round):
    state = mini_round
    post, truth = state.posterior, state.truth

    assert 5000 <= state.n_planned <= MAX_SAMPLING_DRAWS
    assert post.n_planned == state.n_planned and post.n_raw == state.n_planned + post.n_top_up
    assert state.sampling_folds == post.n_in
    assert np.isfinite(post.k_hat)
    assert post.certified == state.best.passed
    if not state.best.passed:
        assert post.log_z_err_reliable is False
    assert abs(post.log_z - truth["log_z"]) <= max(4.0 * post.log_z_err, 0.15)
    deviation = np.abs(state.theta.mean(axis=0) - truth["mean_theta"]) / truth["std_theta"]
    assert np.all(deviation < 0.25), deviation
    assert 1 < state.n_unique <= N_POSTERIOR


def test_fold_accounting_and_process_state(mini_round):
    state = mini_round
    expected = (
        state.n_training
        + sum(rnd.screen_folds + rnd.gate_folds for rnd in (state.round0, state.round1))
        + 2 * N_TRUNCATED
        + state.sampling_folds
    )

    assert _fold_total(state.solver) == expected
    assert set(state.sim.seconds) == {"harvest", "screen", "gate", "training", "sampling"}
    assert torch.get_num_threads() == state.torch_threads
    assert not (state.workdir / "sbi-logs").exists()
    assert set(state.timings) >= {"harvest", "compressor_0", "train_0", "truncate_sir", "final", "total"}
