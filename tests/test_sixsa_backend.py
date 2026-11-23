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
from bsixsa.backend.config import AutoencoderConfig, FlowConfig, ScoreCompressorConfig
from bsixsa.backend.sixsa._sixsa import SIXSABackend
from bsixsa.backend.sixsa.compressor import Compressor
from bsixsa.backend.sixsa.score import ScoreCompressor
from bsixsa.priors import MultipleIndependent, loguniform

_SUMMARY_SCALARS = {"name", "round_index", "n_raw", "n_in", "k_hat", "ess", "efficiency", "log_z", "log_z_err",
                    "passed"}


def small_restriction():
    return HistoryMatching(n_waves=3, simulations_per_parameter=200,
                           hidden=32, blocks=2, epochs=200, patience=10,
                           mass_probe=20000, seed=0)


# A small autoencoder, for the runs that check the alternative compressor.
SMALL_AUTOENCODER = dict(latent_dim=8, n_layers=2, epochs=80, patience=8, batch_size=256)


def small_config(**changes):
    values = dict(
        seed=0, n_nde=2, n_rounds=4, check_draws=1000,
        n_training=1500, train_max=3000, trace=False,
        history_matching=small_restriction(),
        flow=dict(hidden_features=32, num_transforms=3, max_num_epochs=20, training_batch_size=256,
                  stop_after_epochs=5),
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
    # The score map is built once, at the fitted fiducial.
    assert isinstance(backend.compressor, ScoreCompressor) and backend.compressor.u_star is not None
    assert solver.fold_counts["compressor"] == backend.compressor.n_folds
    assert backend.compressor.cstat_star <= truth["c_min"] + 0.5
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
                            "posterior", "gate", "rounds", "config"}
    # small_config sets check_draws and target_ess; the efficiency gate comes from the schedule (d = 2).
    assert summary["gate"] == {"khat_threshold": 0.7, "min_efficiency": 10.0, "check_draws": 1000,
                               "target_ess": 1500.0}
    assert (backend.min_efficiency, backend.target_ess) == (10.0, 1500.0)
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
    ("training_n_jobs", 2), ("khat_safe", 0.5), ("refresh_compressor", True),
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
])
def test_network_options_are_validated_at_configuration_boundary(section, field, value):
    with pytest.raises(ValidationError) as exc:
        SIXSA(**{section: {field: value}})
    # One FlowConfig stands for a one-entry list, so its errors carry the index 0.
    expected = (section, 0, field) if section == "flow" else (section, field)
    assert exc.value.errors()[0]["loc"][:len(expected)] == expected


@pytest.mark.parametrize("member,field,value", [
    ("AutoencoderConfig", "latent_dim", 0), ("AutoencoderConfig", "hidden_dims", [16, 0]),
    ("AutoencoderConfig", "batch_size", "auto"), ("AutoencoderConfig", "batch_size", 0),
    ("AutoencoderConfig", "epochs", 0), ("AutoencoderConfig", "learning_rate", 0.0),
    ("AutoencoderConfig", "learning_rate", float("inf")), ("AutoencoderConfig", "patience", 0),
    ("AutoencoderConfig", "validation_fraction", 1.0), ("AutoencoderConfig", "device", "tpu"),
    ("AutoencoderConfig", "refresh", "True"), ("ScoreCompressorConfig", "fiducial", "mle"),
    ("ScoreCompressorConfig", "recentre", "yes"),
])
def test_compressor_options_are_validated_by_the_compressor_they_belong_to(member, field, value):
    with pytest.raises(ValidationError) as exc:
        SIXSA(compressor={field: value})
    # Each compressor reports on the dictionary; the one owning the field says what is wrong with the value.
    owner = [e for e in exc.value.errors() if e["loc"][:3] == ("compressor", member, field)]
    assert owner and owner[0]["type"] != "extra_forbidden"


def test_reference_training_defaults_are_preserved():
    from bsixsa.backend.sixsa import flows

    cfg = SIXSA()
    assert cfg.n_nde == 7 and cfg.flow_torch_threads is None
    budget = dict(training_batch_size=1024, max_num_epochs=500, stop_after_epochs=10)
    assert [member.model_dump() for member in cfg.flow_members()] == [
        dict(model="zuko_nsf", hidden_features=30, num_transforms=3, num_bins=8, **budget),
        dict(model="zuko_nsf", hidden_features=40, num_transforms=3, num_bins=8, **budget),
        dict(model="zuko_nsf", hidden_features=30, num_transforms=4, num_bins=8, **budget),
        dict(model="zuko_nsf", hidden_features=40, num_transforms=4, num_bins=8, **budget),
        dict(model="zuko_nsf", hidden_features=30, num_transforms=3, num_bins=10, **budget),
        dict(model="zuko_maf", hidden_features=30, num_transforms=3, num_bins=8, **budget),
        dict(model="zuko_maf", hidden_features=40, num_transforms=4, num_bins=8, **budget),
    ]
    assert (cfg.check_draws, cfg.train_ramp, cfg.train_max) == (2000, 1.1, 100_000)
    assert cfg.min_efficiency is None and cfg.target_ess is None
    assert (flows._LEARNING_RATE, flows._VALIDATION_FRACTION, flows._CLIP_MAX_NORM) == (1e-3, 0.2, 1.0)


@pytest.mark.parametrize("d, n_jobs, cpus, explicit, expected", [
    (5, 16, 16, None, 1), (19, 16, 16, None, 1), (20, 16, 16, None, 2), (41, 16, 16, None, 4),
    (41, 2, 16, None, 2), (41, 16, 3, None, 3), (5, 16, 16, 6, 6),
])
def test_flows_train_on_threads_set_by_dimension_within_the_fit_cores(
        d, n_jobs, cpus, explicit, expected, monkeypatch, tmp_path):
    import torch

    seen = []

    def fake_train(*args, **kwargs):
        seen.append(torch.get_num_threads())
        return object(), dict(epochs_trained=1, best_validation_loss=0.0, attempts=1)

    monkeypatch.setattr("bsixsa.backend.sixsa._sixsa.train_flow", fake_train)
    monkeypatch.setattr("bsixsa.backend.sixsa._sixsa.FlowProposal", lambda *args, **kwargs: object())
    monkeypatch.setattr("bsixsa.backend.sixsa._sixsa.available_cpu_count", lambda: cpus)
    backend = SimpleNamespace(config=SIXSA(n_nde=2, flow_torch_threads=explicit), rng=np.random.default_rng(0),
                              solver=SimpleNamespace(num_parameters=d, _n_jobs=n_jobs), outdir=tmp_path)
    before = torch.get_num_threads()
    SIXSABackend._train_flows(backend, 1, None, None, None)
    assert seen == [expected, expected] and torch.get_num_threads() == before


@pytest.mark.parametrize("d, efficiency, ess", [
    (1, 10.0, 5000), (4, 10.0, 5000), (5, 5.0, 3000), (9, 5.0, 3000), (10, 2.5, 2000),
    (14, 2.5, 2000), (15, 1.0, 1000), (24, 1.0, 1000),
])
def test_gate_and_final_ess_follow_the_reference_schedules(d, efficiency, ess):
    cfg = SIXSA()
    assert cfg.gate_efficiency(d) == efficiency and cfg.final_ess(d) == ess
    explicit = SIXSA(min_efficiency=7.5, target_ess=1234)
    assert explicit.gate_efficiency(d) == 7.5 and explicit.final_ess(d) == 1234


def test_custom_schedules_and_their_validation():
    cfg = SIXSA(efficiency_schedule=[(3, 20.0), (None, 4.0)], ess_schedule=[[8, 900], [None, 300]])
    assert (cfg.gate_efficiency(2), cfg.gate_efficiency(3)) == (20.0, 4.0)
    assert (cfg.final_ess(7), cfg.final_ess(8)) == (900.0, 300.0)
    assert SIXSA.model_validate_json(cfg.model_dump_json()) == cfg
    for schedule in ([], [(5, 10.0)], [(None, 1.0), (5, 10.0)], [(10, 5.0), (5, 2.0), (None, 1.0)],
                     [(5, 101.0), (None, 1.0)], [(5, float("nan")), (None, 1.0)]):
        with pytest.raises(ValidationError):
            SIXSA(efficiency_schedule=schedule)
    with pytest.raises(ValidationError):
        SIXSA(ess_schedule=[(5, 0.0), (None, 100.0)])
    with pytest.raises(ValidationError):
        SIXSA(min_efficiency=150.0)


def test_flow_members_cycle_through_the_flow_list():
    cfg = SIXSA(n_nde=3, flow=[{"model": "zuko_maf"}, {"hidden_features": 64}])
    assert [(m.model, m.hidden_features) for m in cfg.flow_members()] == [
        ("zuko_maf", 30), ("zuko_nsf", 64), ("zuko_maf", 30)]
    assert [m.hidden_features for m in SIXSA(n_nde=2).flow_members()] == [30, 40]
    single = SIXSA(n_nde=4, flow=FlowConfig(model="zuko_maf", hidden_features=50, num_transforms=5))
    assert len(single.flow) == 1 and len(single.flow_members()) == 4
    assert all(m == single.flow[0] for m in single.flow_members())
    with pytest.raises(ValidationError):
        SIXSA(flow=[])


def test_config_nested_defaults():
    cfg = SIXSA(flow={"hidden_features": 32})
    assert len(cfg.flow) == 1 and cfg.flow[0].num_transforms == 3 and cfg.flow[0].model == "zuko_nsf"
    assert cfg.truncated_oversampling == 32
    assert isinstance(cfg.flow[0], FlowConfig)
    assert cfg.compressor == ScoreCompressorConfig(fiducial="fit")
    autoencoder = AutoencoderConfig()
    assert autoencoder.device == "cpu" and autoencoder.batch_size == 512 and autoencoder.refresh is True
    assert cfg.history_matching == HistoryMatching()
    assert SIXSA(history_matching=None).history_matching is None
    assert SIXSA.model_validate_json(cfg.model_dump_json()) == cfg


@pytest.mark.parametrize("compressor,expected", [
    ({}, ScoreCompressorConfig()),
    ({"fiducial": "best_fold"}, ScoreCompressorConfig(fiducial="best_fold")),
    ({"recentre": True}, ScoreCompressorConfig(recentre=True)),
    ({"latent_dim": 8}, AutoencoderConfig(latent_dim=8)),
    ({"refresh": 2}, AutoencoderConfig(refresh=2)),
    (AutoencoderConfig(device="auto"), AutoencoderConfig(device="auto")),
])
def test_compressor_dictionaries_select_the_compressor_by_their_fields(compressor, expected):
    cfg = SIXSA(compressor=compressor)
    assert type(cfg.compressor) is type(expected) and cfg.compressor == expected
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
        backend=small_config(n_nde=1, n_rounds=1, history_matching=None, max_sampling_draws=1000, target_ess=100,
                             min_efficiency=100),
    )
    # Force the round-1 gate to pass; no sample reaches 100% efficiency, so the
    # final sample fails certification.
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
def test_autoencoder_run_recovers_grid(tmp_path):
    spectrum = AnalyticSpectrum(d=2)
    solver = AnalyticSolver(spectrum, backend=small_config(compressor=SMALL_AUTOENCODER), outputfiles_basename=tmp_path)
    fit = solver.run()
    truth = grid_truth(spectrum)
    backend = solver.backend
    assert isinstance(backend.compressor, Compressor) and backend.compressor.latent_dim == 8
    assert backend.posterior.passed
    assert abs(fit.log_Z - truth["log_z"]) <= max(4 * fit.log_Z_err, 0.10)
    values = np.asarray(fit.posterior_samples)
    delta = np.abs(values.mean(axis=0) - truth["mean_theta"]) / truth["std_theta"]
    assert np.all(delta < 0.1 + 4 / np.sqrt(backend.posterior.ess)), delta
    # The autoencoder trains on the spectra and folds nothing of its own.
    assert "compressor" not in solver.fold_counts
    assert all(record["compressor"]["generation"] == record["round"] for record in backend.history)


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


# Reference implementation: 1000 per parameter at d = 6, times (d / 6)^0.6, rounded to 250, at least 750.
_REFERENCE_TABLE = {2: 1500, 3: 2250, 6: 6000, 10: 12500, 11: 16500, 14: 24500, 18: 36000, 24: 54000}


@pytest.mark.parametrize("d", sorted(_REFERENCE_TABLE))
def test_training_set_size_matches_the_reference_table(d):
    from bsixsa.backend.sixsa._sixsa import training_set_size

    assert training_set_size(d) == _REFERENCE_TABLE[d]


def test_training_set_size_floors():
    from bsixsa.backend.sixsa._sixsa import training_set_size

    assert training_set_size(1) == 750 and training_set_size(2) == 1500  # 750 per parameter


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
def test_autoencoder_refresh_schedule(refresh, rounds):
    backend = SimpleNamespace(config=SimpleNamespace(compressor=AutoencoderConfig(refresh=refresh)))
    assert [r for r in range(1, 7) if SIXSABackend._refits_compressor(backend, r)] == rounds


@pytest.mark.parametrize("fiducial", ["fit", "best_fold"])
def test_score_map_is_built_once_in_round_1(fiducial):
    backend = SimpleNamespace(config=SimpleNamespace(compressor=ScoreCompressorConfig(fiducial=fiducial)))
    assert [r for r in range(1, 7) if SIXSABackend._refits_compressor(backend, r)] == [1]


@pytest.mark.parametrize(("best", "rounds"), [(2499.95, [1]), (2499.0, [1, 2, 3, 4, 5, 6])])
def test_recentre_rebuilds_the_score_map_once_a_fold_beats_its_fiducial(best, rounds):
    backend = SimpleNamespace(config=SimpleNamespace(compressor=ScoreCompressorConfig(recentre=True)),
                              compressor=SimpleNamespace(cstat_star=2500.0), _best_cstat=best)
    assert [r for r in range(1, 7) if SIXSABackend._refits_compressor(backend, r)] == rounds
