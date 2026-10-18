"""The public SIXSA backend, including real training and exact grid checks."""
import json
import logging
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.stats import chi2

from _analytic import AnalyticSolver, AnalyticSpectrum, grid_truth
from bsixsa import HistoryMatching, SIXSA
from bsixsa.backend.config import CompressorConfig, FlowConfig
from bsixsa.backend.sixsa._sixsa import SIXSABackend
from bsixsa.priors import MultipleIndependent, loguniform

_SUMMARY_SCALARS = {"name", "round_index", "n_raw", "n_in", "k_hat", "ess", "efficiency", "log_z", "log_z_err",
                    "passed"}


def small_restriction():
    return HistoryMatching(n_waves=3, simulations_per_parameter=200,
                           hidden=32, blocks=2, epochs=200, patience=10,
                           mass_probe=20000, seed=0)


def small_config(**changes):
    values = dict(
        seed=0, n_nde=2, n_rounds=4, check_draws=1000,
        n_training=1500, train_max=3000, trace=False,
        history_matching=small_restriction(),
        flow=dict(hidden_features=32, num_transforms=3, max_num_epochs=20, training_batch_size=256,
                  stop_after_epochs=5),
        compressor=dict(latent_dim=8, n_layers=2, epochs=80, patience=8, batch_size=256),
        target_ess=1500, num_posterior_samples=3000, max_sampling_draws=30000,
    )
    values.update(changes)
    return SIXSA(**values)


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    spectrum = AnalyticSpectrum(d=2)
    solver = AnalyticSolver(spectrum, backend=small_config(),
                            outputfiles_basename=tmp_path_factory.mktemp("sixsa_e2e"))
    result = solver.run()
    return solver, result, grid_truth(spectrum)


@pytest.mark.e2e
def test_public_run_recovers_grid_and_accounts_for_every_fold(completed):
    solver, fit, truth = completed
    backend = solver.backend
    assert backend.exit_reason == "gate"
    assert backend.best_round == backend.history[-1]["round"]
    assert backend.posterior.passed
    assert abs(fit.log_Z - truth["log_z"]) <= max(4 * fit.log_Z_err, 0.10)
    values = np.asarray(fit.posterior_samples)
    delta = np.abs(values.mean(axis=0) - truth["mean_theta"]) / truth["std_theta"]
    assert np.all(delta < 0.1 + 4 / np.sqrt(backend.posterior.ess)), delta
    ratio = values.std(axis=0) / truth["std_theta"]
    assert np.all((0.85 < ratio) & (ratio < 1.15)), ratio
    assert fit.n_likelihood_evaluations == sum(solver.fold_counts.values())
    assert solver.fold_counts["restriction"] == backend.restriction.n_simulations
    # History matching folds inside run()'s pool scope, with the solver's parameter names.
    assert all(entry["pool_depth"] >= 1 for entry in solver.fold_log if entry["phase"] == "restriction")
    assert backend.restriction.param_names == solver.parameter_names
    # Each round certifies the mixture of its flows, once.
    assert all(record["name"] == f"round{record['round']}/mixture" for record in backend.history)
    assert solver.fold_counts["gate"] == sum(record["n_in"] for record in backend.history)
    assert solver.fold_counts["harvest"] == 1500
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
    assert np.isfinite(solver.prior.log_prob(theta)).all()
    assert solver.build_dataframe("posterior", 20).shape == (20, 2)
    info = solver.build_inference_data(num_samples=20, include_log_likelihood=True)
    assert "posterior" in info.groups()
    assert fit.n_likelihood_evaluations < sum(solver.fold_counts.values())
    # Standard JSON: non-finite values are written as null, never as NaN or Infinity.
    summary = json.loads((Path(solver.outputfiles_basename) / "sixsa/summary.json").read_text(),
                         parse_constant=lambda name: pytest.fail(f"summary.json holds {name}"))
    assert summary["best_round"] == backend.best_round
    assert summary["posterior"]["passed"] and summary["rounds"][-1]["passed"]
    # The persisted keys, which notebooks read.
    assert set(summary) == {"exit_reason", "best_round", "time", "n_likelihood_evaluations", "fold_counts",
                            "posterior", "rounds", "config"}
    assert set(summary["posterior"]) == _SUMMARY_SCALARS | {"n_unique"}
    round_keys = _SUMMARY_SCALARS | {"round", "n_train", "compressor", "nde", "truncation"}
    assert all(set(record) == round_keys for record in summary["rounds"])
    assert all(set(record) == round_keys | {"certificate"} for record in backend.history)
    assert backend.history[-1]["certificate"] is backend.certificates[-1]
    assert backend.n_unique == summary["posterior"]["n_unique"]
    assert all((Path(solver.outputfiles_basename) / "sixsa" / f"round_{r + 1}" / "member_0").is_dir()
               for r in range(len(backend.history)))
    with np.load(Path(solver.outputfiles_basename) / "sixsa/restriction/restriction.npz") as saved:
        np.testing.assert_array_equal(saved["u"], backend.restriction.u)


@pytest.mark.parametrize("field,value", [
    ("n_nde", 0), ("check_draws", 10), ("truncated_sampling", "flow"), ("fold_chunk", 1000),
    ("training_n_jobs", 2), ("khat_safe", 0.5), ("refresh_compressor", "True"),
    ("n_ensemble", 2), ("run_emulator_pass", True),
])
def test_config_rejects_invalid_and_retired_fields(field, value):
    with pytest.raises(ValidationError):
        SIXSA(**{field: value})



@pytest.mark.parametrize("field,value", [
    ("n_nde", 0), ("n_rounds", -1), ("train_max", 0),
    ("n_training", 19), ("truncated_oversampling", 0), ("truncated_quantile", 0.0),
    ("truncated_quantile", 1.0), ("truncated_support_draws", 0), ("train_ramp", 0.5),
    ("train_ramp", float("inf")), ("target_ess", 0.0),
    ("target_ess", float("inf")), ("target_ess", float("nan")),
    ("max_sampling_draws", -1), ("num_posterior_samples", 0), ("seed", -1),
])
def test_internal_budgets_are_validated_at_configuration_boundary(field, value):
    with pytest.raises(ValidationError) as exc:
        SIXSA(**{field: value})
    assert exc.value.errors()[0]["loc"] == (field,)


@pytest.mark.parametrize("section,field,value", [
    ("flow", "hidden_features", 0), ("flow", "num_transforms", 0),
    ("flow", "training_batch_size", 0), ("flow", "max_num_epochs", 0),
    ("flow", "stop_after_epochs", 0), ("flow", "model", "nsf"),
    ("history_matching", "n_waves", 0),
    ("compressor", "latent_dim", 0), ("compressor", "hidden_dims", [16, 0]),
    ("compressor", "batch_size", "auto"), ("compressor", "batch_size", 0), ("compressor", "epochs", 0),
    ("compressor", "learning_rate", 0.0), ("compressor", "learning_rate", float("inf")),
    ("compressor", "patience", 0), ("compressor", "validation_fraction", 1.0),
    ("compressor", "device", "tpu"),
])
def test_network_options_are_validated_at_configuration_boundary(section, field, value):
    with pytest.raises(ValidationError) as exc:
        SIXSA(**{section: {field: value}})
    assert exc.value.errors()[0]["loc"][:2] == (section, field)


def test_reference_training_defaults_are_preserved():
    from bsixsa.backend.sixsa import flows

    cfg = SIXSA()
    assert cfg.n_nde == 5
    assert cfg.flow.model_dump() == dict(
        hidden_features=50, num_transforms=5, training_batch_size=1024, max_num_epochs=25, stop_after_epochs=10,
    )
    assert (flows._LEARNING_RATE, flows._VALIDATION_FRACTION, flows._CLIP_MAX_NORM) == (1e-3, 0.2, 1.0)


def test_config_nested_defaults():
    cfg = SIXSA(flow={"hidden_features": 32})
    assert cfg.flow.num_transforms == 5
    assert cfg.truncated_oversampling == 32
    assert cfg.compressor.device == "cpu" and cfg.compressor.batch_size == 512
    assert isinstance(cfg.flow, FlowConfig)
    assert isinstance(cfg.compressor, CompressorConfig)
    assert cfg.history_matching == HistoryMatching()
    assert SIXSA(history_matching=None).history_matching is None
    assert SIXSA.model_validate_json(cfg.model_dump_json()) == cfg


def _run_recording_warnings(solver, monkeypatch):
    """Run `solver`; return SIXSA's RuntimeWarnings, each with the folds spent (per phase) when SIXSA raised it.

    SIXSA issues its RuntimeWarnings while the backend runs, so the folds
    are read at the moment SIXSA calls `warnings.warn`; each warning must
    point at the caller of `solver.run()` (this file).
    """
    from bsixsa.backend.sixsa import _sixsa

    emitted = []

    def warn(message, category, stacklevel):
        emitted.append((str(message), dict(solver.fold_counts)))
        warnings.warn(message, category, stacklevel=stacklevel + 1)

    monkeypatch.setattr(_sixsa, "warnings", SimpleNamespace(warn=warn))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = solver.run()
    shown = [(str(w.message), w.filename) for w in caught if str(w.message).startswith("SIXSA")]
    assert shown == [(message, __file__) for message, _ in emitted]
    return result, emitted


@pytest.mark.e2e
def test_budget_exhaustion_returns_uncertified_single_member(tmp_path, monkeypatch):
    solver = AnalyticSolver(
        AnalyticSpectrum(), outputfiles_basename=tmp_path,
        backend=small_config(n_nde=1, n_rounds=1, khat_threshold=0, history_matching=None,
                             max_sampling_draws=1000, target_ess=100),
    )
    result, emitted = _run_recording_warnings(solver, monkeypatch)
    # Both warnings name the budget and the next steps; the budget warning
    # comes before the final sample spends any fold.
    (budget_folds,) = [folds for m, folds in emitted if "n_rounds=1" in m and "capped at 1000 draws" in m]
    assert "sampling" not in budget_folds
    # The budget warning points at the per-round k-hat values in summary.json.
    (budget_message,) = [m for m, _ in emitted if "capped at 1000 draws" in m]
    summary = solver.backend.outdir / "summary.json"
    assert str(summary) in budget_message and ".pdf" not in budget_message
    assert summary.exists()
    assert any("final importance sample not certified" in m and "increase n_rounds" in m and "nested sampler" in m
               for m, _ in emitted)
    assert solver.backend.exit_reason == "n_rounds"
    assert not solver.backend.posterior.passed
    assert np.isfinite(result.log_Z)
    assert result.n_likelihood_evaluations == sum(solver.fold_counts.values())


@pytest.mark.e2e
def test_uncertified_final_sample_after_a_passed_gate_gives_gate_advice(tmp_path, monkeypatch):
    from bsixsa.backend.sixsa import _sixsa

    solver = AnalyticSolver(
        AnalyticSpectrum(), outputfiles_basename=tmp_path,
        backend=small_config(n_nde=1, n_rounds=1, history_matching=None, max_sampling_draws=1000, target_ess=100),
    )
    # Force the round-1 gate to pass; its proposal is still far from the
    # posterior, so the final sample fails certification.
    real = _sixsa.certify

    def gate_passes(*args, **kwargs):
        cert = real(*args, **kwargs)
        cert.passed = True
        return cert

    monkeypatch.setattr(_sixsa, "certify", gate_passes)
    _, emitted = _run_recording_warnings(solver, monkeypatch)
    assert solver.backend.exit_reason == "gate" and not solver.backend.posterior.passed
    (message,) = [m for m, _ in emitted if "final importance sample not certified" in m]
    assert "the gate passed on 1000 draws" in message and "check_draws" in message
    assert "n_rounds" not in message
    assert not any("capped at" in m for m, _ in emitted)


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
    repeat = AnalyticSolver(AnalyticSpectrum(d=2), backend=small_config(), outputfiles_basename=tmp_path)
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
        spectrum, outputfiles_basename=tmp_path,
        backend=small_config(n_training=4000, train_max=8000, n_rounds=6, trace=True, history_matching=None,
                             flow=dict(hidden_features=32, num_transforms=3, max_num_epochs=50,
                                       training_batch_size=256, stop_after_epochs=10)),
    )
    fit = solver.run()
    truth = grid_truth(spectrum)
    assert solver.backend.restriction is None
    assert solver.backend.posterior.passed
    assert abs(fit.log_Z - truth["log_z"]) < max(4 * fit.log_Z_err, 0.1)
    values = np.asarray(fit.posterior_samples)
    error = np.abs(values.mean(0) - truth["mean_theta"]) / truth["std_theta"]
    assert np.all(error < 0.1 + 4 / np.sqrt(solver.backend.posterior.ess)), error
    assert fit.n_likelihood_evaluations == sum(solver.fold_counts.values())
    assert solver.backend.tracer.n_evals == fit.n_likelihood_evaluations


_REFERENCE_TABLE = {
    2500: [2000, 3750, 12000, 27500, 45500],
    1000: [1500, 3000, 9000, 22500, 38500],
    5000: [2500, 4500, 15000, 32500, 56000],
}


@pytest.mark.parametrize("n_bins", sorted(_REFERENCE_TABLE))
@pytest.mark.parametrize("column, d", list(enumerate([2, 3, 6, 10, 14])))
def test_training_set_size_matches_the_reference_table(n_bins, column, d):
    from bsixsa.backend.sixsa._sixsa import training_set_size

    assert training_set_size(d, n_bins) == _REFERENCE_TABLE[n_bins][column]


def test_training_set_size_floors():
    from bsixsa.backend.sixsa._sixsa import training_set_size

    assert training_set_size(2, 64) == 1500  # binning factor clipped at 0.5, 750 per parameter
    assert training_set_size(6, 10**9) == training_set_size(6, 2500 * 81) == 36000


def test_sampling_before_run_says_what_to_do(tmp_path):
    solver = AnalyticSolver(AnalyticSpectrum(), outputfiles_basename=tmp_path,
                            backend=small_config(history_matching=None))
    backend = SIXSABackend(solver=solver, config=solver.backend_config)
    assert backend.n_unique is None and backend.posterior is None
    with pytest.raises(RuntimeError, match=r"call solver.run\(\) first"):
        backend.sample(3)
    with pytest.raises(RuntimeError, match=r"call solver.run\(\) first"):
        backend.weighted_samples()


def test_cube_points_map_inside_the_prior_support():
    prior = MultipleIndependent([loguniform(662.5966289294636, 704.2139698603843)])
    u = np.array([[0.9999999999999929]])
    # The plain inverse CDF rounds this point an ulp above the upper bound.
    assert np.isneginf(prior.log_prob(prior.from_unit_cube(u))).all()
    backend = SimpleNamespace(solver=SimpleNamespace(prior=prior))
    theta = SIXSABackend._to_physical(backend, u)
    assert theta[0, 0] == pytest.approx(704.2139698603843, rel=1e-15)
    assert np.isfinite(prior.log_prob(theta)).all()


def _region_warnings(caplog, *, best, spans_to_threshold, d=6):
    """Region-reach warnings for a posterior spread flat over one span above `best`.

    History matching places the threshold `spans_to_threshold` posterior spans
    above the best C-stat; 99.9% of this posterior's weight then lies below
    about `best + span`.
    """
    span = float(chi2.ppf(0.999, d))
    restriction = SimpleNamespace(threshold=best + spans_to_threshold * span, span=span)
    cstat = best + span * np.linspace(0.0, 1.0, 1001)
    cert = SimpleNamespace(cstat=cstat, log_w_smoothed=np.full(cstat.size, -np.log(cstat.size)))
    backend = SimpleNamespace(restriction=restriction)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="bsixsa.backend.sixsa._sixsa"):
        SIXSABackend._warn_if_region_cuts_posterior(backend, 1, cert)
    return [r.getMessage() for r in caplog.records if "region may cut" in r.getMessage()]


@pytest.mark.parametrize("best", [100.0, 3000.0, 30000.0])
def test_region_reach_warning_does_not_depend_on_the_size_of_the_cstat(caplog, best):
    # A healthy run: three spans of margin, whatever the number of channels.
    # A threshold fraction (2%) would warn here for C-stat in the thousands; one span does not.
    assert _region_warnings(caplog, best=best, spans_to_threshold=3.0) == []
    # Less than one span of margin left: the region may cut the posterior.
    (message,) = _region_warnings(caplog, best=best, spans_to_threshold=1.5)
    assert "less than one posterior span" in message and "target_spans" in message


@pytest.mark.parametrize(("refresh", "rounds"), [
    (True, [1, 2, 3, 4, 5, 6]), (False, [1]), (0, [1]), (1, [1]), (2, [1, 2]), (5, [1, 2, 3, 4, 5]),
])
def test_refresh_compressor_schedule(refresh, rounds):
    backend = SimpleNamespace(config=SimpleNamespace(refresh_compressor=refresh))
    assert [r for r in range(1, 7) if SIXSABackend._refits_compressor(backend, r)] == rounds
