"""The public SIXSA backend, including real training and exact grid checks."""
import json

import numpy as np
import pytest
from pydantic import ValidationError

from _analytic import AnalyticSolver, AnalyticSpectrum, grid_truth
from bsixsa import HistoryMatching, SIXSA
from bsixsa.backend.config import CompressorConfig, NDEConfig, TrainingConfig


def small_config(**changes):
    values = dict(
        seed=0, n_nde=2, n_rounds=4, check_draws=1000,
        n_training=1500, train_max=3000, trace=False,
        nde=dict(hidden_features=32, num_transforms=3),
        training=dict(max_num_epochs=20, training_batch_size=256, stop_after_epochs=5),
        compressor=dict(latent_dim=8, n_layers=2, epochs=80, patience=8, batch_size=256),
        target_ess=1500, num_posterior_samples=3000, max_sampling_draws=30000,
        training_n_jobs=1, plot_diagnostics=False,
    )
    values.update(changes)
    return SIXSA(**values)


def small_restriction():
    return HistoryMatching(n_waves=3, simulations_per_parameter=200,
                           hidden=32, blocks=2, epochs=200, patience=10,
                           mass_probe=20000, box_samples=20000, seed=0, plot=False)


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    spectrum = AnalyticSpectrum(d=2)
    solver = AnalyticSolver(spectrum, backend=small_config(), restrict_prior=small_restriction(),
                            outputfiles_basename=tmp_path_factory.mktemp("sixsa_e2e"))
    result = solver.run()
    return solver, result, grid_truth(spectrum)


@pytest.mark.e2e
def test_public_run_recovers_grid_and_accounts_for_every_fold(completed):
    solver, fit, truth = completed
    backend = solver.backend
    assert backend.exit_reason == "gate"
    assert backend.best_round == backend.history[-1]["round"]
    assert backend.posterior.certified
    assert abs(fit.log_Z - truth["log_z"]) <= max(4 * fit.log_Z_err, 0.10)
    values = np.asarray(fit.posterior_samples)
    delta = np.abs(values.mean(axis=0) - truth["mean_theta"]) / truth["std_theta"]
    assert np.all(delta < 0.1 + 4 / np.sqrt(backend.posterior.ess)), delta
    ratio = values.std(axis=0) / truth["std_theta"]
    assert np.all((0.85 < ratio) & (ratio < 1.15)), ratio
    assert fit.n_likelihood_evaluations == sum(solver.fold_counts.values())
    assert fit.n_restriction_evaluations == solver.restriction.n_simulations
    assert backend.fold_counts["harvest"] == 1500
    assert fit.best_fit_stat >= truth["c_min"] - 0.05
    assert fit.best_fit_stat <= truth["c_min"] + 2
    assert list(fit.posterior_samples.columns) == solver.parameter_names


def test_public_sampling_and_saved_diagnostics(completed):
    solver, fit, _ = completed
    backend = solver.backend
    assert backend.sample(37).shape == (37, 2)
    theta, weights = backend.weighted_samples()
    assert theta.shape[0] == weights.size
    assert weights.sum() == pytest.approx(1)
    assert np.isfinite(solver.base_prior.log_prob(theta)).all()
    assert solver.build_dataframe("posterior", 20).shape == (20, 2)
    info = solver.build_inference_data(num_samples=20, include_log_likelihood=True)
    assert "posterior" in info.groups()
    assert fit.n_likelihood_evaluations < sum(solver.fold_counts.values())
    from pathlib import Path
    summary = json.loads((Path(solver.outputfiles_basename) / "sixsa/summary.json").read_text())
    assert summary["best_round"] == backend.best_round
    assert len(backend.plot_diagnostics()) == 3


@pytest.mark.parametrize("field,value", [
    ("n_nde", 0), ("check_draws", 10), ("truncated_sampling", "unknown"),
    ("refresh_compressor", "True"), ("training_n_jobs", 0),
    ("n_ensemble", 2), ("run_emulator_pass", True),
])
def test_config_rejects_invalid_and_retired_fields(field, value):
    with pytest.raises(ValidationError):
        SIXSA(**{field: value})


def test_config_nested_defaults_and_restriction_mode():
    cfg = SIXSA(nde={"hidden_features": 32})
    assert cfg.nde.num_transforms == 5
    assert cfg.truncated_sampling == "sir"
    assert cfg.restriction_mode == "region" and cfg.restrict_prior_default
    assert isinstance(cfg.nde, NDEConfig)
    assert isinstance(cfg.training, TrainingConfig)
    assert isinstance(cfg.compressor, CompressorConfig)
    assert SIXSA.model_validate_json(cfg.model_dump_json()) == cfg


@pytest.mark.e2e
def test_budget_exhaustion_returns_uncertified_single_member(tmp_path):
    solver = AnalyticSolver(
        AnalyticSpectrum(), outputfiles_basename=tmp_path, restrict_prior=False,
        backend=small_config(n_nde=1, n_rounds=1, khat_threshold=0,
                             max_sampling_draws=1000, target_ess=100),
    )
    with pytest.warns(RuntimeWarning) as emitted:
        result = solver.run()
    assert any("certif" in str(w.message) for w in emitted)
    assert solver.backend.exit_reason == "n_rounds"
    assert not solver.backend.posterior.certified
    assert not solver.backend.posterior.log_z_err_reliable
    assert "screen" not in solver.fold_counts
    assert np.isfinite(result.log_Z)
    assert result.n_likelihood_evaluations == sum(solver.fold_counts.values())


@pytest.mark.e2e
def test_posterior_mass_in_grid_hpd_regions(completed):
    solver, _, truth = completed
    post = solver.backend.posterior
    weights = np.exp(post.log_w_smoothed)
    for level, threshold in truth["hpd_cstat"].items():
        mass = weights[post.cstat <= threshold].sum()
        assert abs(mass - level) < 0.02 + 4 * np.sqrt(level * (1 - level) / post.ess)


@pytest.mark.slow
def test_public_run_is_reproducible(completed, tmp_path):
    first, _, _ = completed
    repeat = AnalyticSolver(AnalyticSpectrum(d=2), backend=small_config(),
                            restrict_prior=small_restriction(), outputfiles_basename=tmp_path)
    repeat.run()
    a, b = first.backend.posterior, repeat.backend.posterior
    np.testing.assert_array_equal(a.u, b.u)
    np.testing.assert_array_equal(a.log_w, b.log_w)
    assert a.log_z == b.log_z
    assert first.backend.best_round == repeat.backend.best_round


@pytest.mark.slow
@pytest.mark.e2e
def test_three_parameters_without_restriction(tmp_path):
    spectrum = AnalyticSpectrum(d=3)
    solver = AnalyticSolver(
        spectrum, restrict_prior=False, outputfiles_basename=tmp_path,
        backend=small_config(n_training=4000, train_max=8000, n_rounds=6, trace=True,
                             training=dict(max_num_epochs=50, training_batch_size=256,
                                           stop_after_epochs=10)),
    )
    fit = solver.run()
    truth = grid_truth(spectrum)
    assert solver.restriction is None
    assert solver.backend.posterior.certified
    assert abs(fit.log_Z - truth["log_z"]) < max(4 * fit.log_Z_err, 0.1)
    values = np.asarray(fit.posterior_samples)
    error = np.abs(values.mean(0) - truth["mean_theta"]) / truth["std_theta"]
    assert np.all(error < 0.1 + 4 / np.sqrt(solver.backend.posterior.ess)), error
    assert fit.n_likelihood_evaluations == sum(solver.fold_counts.values())
    assert solver.backend.tracer.n_evals == fit.n_likelihood_evaluations
