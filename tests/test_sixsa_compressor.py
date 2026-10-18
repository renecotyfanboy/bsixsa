"""Tests for `bsixsa.backend.sixsa.compressor`."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault("xspec", MagicMock())

torch = pytest.importorskip("torch")

from _analytic import AnalyticSpectrum  # noqa: E402

from bsixsa.backend.sixsa.compressor import (  # noqa: E402
    Compressor,
    cstat_loss,
    default_hidden_dims,
    default_latent_dim,
)

N_BINS = 64


@pytest.fixture(scope="module")
def spectra():
    """Poisson spectra of the analytic model over a central box of the prior, with their C-stat."""
    spectrum = AnalyticSpectrum(d=2)
    prior = spectrum.prior()
    rng = np.random.default_rng(0)
    theta = prior.from_unit_cube(0.3 + 0.4 * rng.random((600, 2)))
    mu = spectrum.model_counts(theta)
    return {"counts": rng.poisson(mu).astype(np.float32), "mu": mu, "cstat": spectrum.cstat(theta)}


def _compressor(**kwargs):
    settings = dict(d=2, epochs=40, batch_size=32, learning_rate=1e-2, patience=20, seed=0)
    settings.update(kwargs)
    return Compressor(N_BINS, **settings)


def _numpy_cash(mu, k, eps=1e-6):
    mu = np.clip(np.asarray(mu, dtype=np.float64), eps, None)
    k = np.asarray(k, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(k > 0, k * np.log(np.where(k > 0, k, 1.0) / mu), 0.0)
    return float(np.mean(2.0 * np.sum(mu - k + term, axis=1)))


# ---------------------------------------------------------------------------
# statistics and defaults
# ---------------------------------------------------------------------------


def test_cstat_loss_matches_the_numpy_cash_formula():
    rng = np.random.default_rng(1)
    mu = rng.uniform(0.05, 30.0, size=(5, 17))
    k = rng.poisson(mu).astype(np.float64)
    loss = cstat_loss(torch.as_tensor(mu), torch.as_tensor(k))
    assert loss.ndim == 0
    np.testing.assert_allclose(float(loss), _numpy_cash(mu, k), rtol=1e-12)


def test_cstat_loss_handles_zero_counts_in_value_and_gradient():
    mu = torch.tensor([[0.5, 2.0, 1e-9], [3.0, 0.1, 4.0]], dtype=torch.float64, requires_grad=True)
    k = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 2.0]], dtype=torch.float64)
    loss = cstat_loss(mu, k)
    loss.backward()
    assert torch.isfinite(loss) and torch.all(torch.isfinite(mu.grad))
    np.testing.assert_allclose(float(loss), _numpy_cash(mu.detach().numpy(), k.numpy()), rtol=1e-12)
    # A spectrum of zeros against its model costs 2 * sum(mu).
    np.testing.assert_allclose(float(cstat_loss(mu[:1].detach(), k[:1])), 2.0 * (0.5 + 2.0 + 1e-6), rtol=1e-12)


@pytest.mark.parametrize(("d", "latent"), [(1, 8), (3, 8), (5, 8), (7, 16), (14, 28), (16, 32), (40, 32)])
def test_default_latent_dim_values(d, latent):
    assert default_latent_dim(d) == latent


def test_default_hidden_dims_values():
    assert default_hidden_dims(2502, 28, n_layers=3) == [625, 156, 56]
    assert default_hidden_dims(2502, 28) == [625, 156]
    assert default_hidden_dims(64, 8, n_layers=3) == [32, 16, 16]


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def test_constructor_defaults():
    comp = Compressor(2502, d=14)
    assert comp.latent_dim == 28 and comp.hidden_dims == [625, 156, 56]
    assert comp.generation == 0 and comp.model is None
    assert comp.batch_size == 512 and comp.device == torch.device("cpu")
    assert comp.history == [] and comp.curves == []
    assert Compressor(64, latent_dim=12).latent_dim == 12


def test_architecture(spectra):
    comp = _compressor(epochs=1)
    comp.fit(spectra["counts"])
    encoder, decoder = list(comp.model.encoder), list(comp.model.decoder)
    assert [type(m) for m in encoder[:3]] == [torch.nn.Linear, torch.nn.BatchNorm1d, torch.nn.GELU]
    assert [m.out_features for m in encoder if isinstance(m, torch.nn.Linear)] == [32, 16, 16, 8]
    assert [m.out_features for m in decoder if isinstance(m, torch.nn.Linear)] == [16, 16, 32, N_BINS]
    assert isinstance(encoder[-1], torch.nn.Linear) and isinstance(decoder[-1], torch.nn.Linear)


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


def test_fit_reduces_the_validation_cstat(spectra):
    comp = _compressor()
    record = comp.fit(spectra["counts"])
    curve = comp.curves[-1]
    assert record["val_cstat"] < 0.2 * curve["val"][0]
    # Better than predicting the mean spectrum for every input.
    counts = spectra["counts"]
    mean = np.repeat(counts.mean(axis=0, keepdims=True), len(counts), axis=0)
    assert record["val_cstat"] < 0.5 * _numpy_cash(mean, counts)


def test_fit_record_contents(spectra):
    comp = _compressor(epochs=5, patience=0)
    record = comp.fit(spectra["counts"])
    assert record["generation"] == 1 and record["epochs"] == 5
    assert record["n_train"] == 480 and record["n_val"] == 120
    assert record["warm"] is False and record["seconds"] > 0
    assert np.isfinite(record["val_cstat"])
    assert comp.history == [record]
    record["generation"] = 99
    assert comp.history[0]["generation"] == 1
    assert comp.curves[0]["epoch"] == [1, 2, 3, 4, 5]


def test_generation_increments_on_every_fit_including_warm_refits(spectra):
    comp = _compressor(epochs=3)
    comp.fit(spectra["counts"])
    first_weight = comp.model.encoder[0].weight.detach().clone()
    scaler = comp.model.log_mean.detach().clone()
    comp.fit(spectra["counts"][:300])
    comp.fit(spectra["counts"], epochs=2)
    assert comp.generation == 3
    assert [r["generation"] for r in comp.history] == [1, 2, 3]
    assert [r["warm"] for r in comp.history] == [False, True, True]
    assert comp.history[2]["epochs"] == 2
    # Warm refits keep the scaler and continue from the previous weights.
    torch.testing.assert_close(comp.model.log_mean, scaler)
    assert not torch.equal(comp.model.encoder[0].weight, first_weight)


def test_a_failed_warm_refit_changes_nothing(spectra, monkeypatch):
    comp = _compressor(epochs=2)
    comp.fit(spectra["counts"])
    codes = comp.encode(spectra["counts"])
    before = {name: t.clone() for name, t in comp.model.state_dict().items()}
    seen = {}

    def failing(model, *args, **kwargs):
        seen.update({name: t.clone() for name, t in model.state_dict().items()})
        with torch.no_grad():
            model.encoder[0].weight.add_(1.0)
        raise RuntimeError("interrupted")

    monkeypatch.setattr(comp, "_train", failing)
    with pytest.raises(RuntimeError, match="interrupted"):
        comp.fit(spectra["counts"])
    assert seen.keys() == before.keys() and all(torch.equal(seen[k], before[k]) for k in before)
    assert comp.generation == 1 and len(comp.history) == 1
    np.testing.assert_array_equal(comp.encode(spectra["counts"]), codes)


def test_cold_refits_rebuild_the_model(spectra):
    comp = _compressor(epochs=2, warm_start=False)
    comp.fit(spectra["counts"])
    model = comp.model
    comp.fit(spectra["counts"][:300])
    assert comp.model is not model
    assert [r["warm"] for r in comp.history] == [False, False] and comp.generation == 2


def test_encode_shape_dtype_and_determinism(spectra):
    comp = _compressor(epochs=3)
    comp.fit(spectra["counts"])
    counts = spectra["counts"]
    codes = comp.encode(counts)
    assert codes.dtype == np.float32 and codes.shape == (len(counts), comp.latent_dim)
    np.testing.assert_array_equal(codes, comp.encode(counts))
    np.testing.assert_allclose(comp.encode(counts, batch=7), codes, rtol=1e-5, atol=1e-5)
    single = comp.encode(counts[3])
    assert single.shape == (1, comp.latent_dim)
    np.testing.assert_allclose(single[0], codes[3], rtol=1e-6, atol=1e-6)
    # Evaluation mode is set by encode itself, not inherited.
    comp.model.train()
    np.testing.assert_array_equal(comp.encode(counts), codes)


def test_encode_before_fit_raises(spectra):
    with pytest.raises(RuntimeError, match="not been fitted"):
        _compressor().encode(spectra["counts"])


def test_fit_is_deterministic_for_a_seed(spectra):
    counts = spectra["counts"]
    a, b = _compressor(epochs=8), _compressor(epochs=8)
    rec_a, rec_b = a.fit(counts), b.fit(counts)
    np.testing.assert_array_equal(a.encode(counts), b.encode(counts))
    assert rec_a["val_cstat"] == rec_b["val_cstat"]
    a.fit(counts)
    b.fit(counts)
    np.testing.assert_array_equal(a.encode(counts), b.encode(counts))
    other = _compressor(epochs=8, seed=1)
    other.fit(counts)
    assert not np.array_equal(other.encode(counts), a.encode(counts))


def test_seed_is_combined_with_the_generation(spectra):
    counts = spectra["counts"]
    cold = _compressor(epochs=3, warm_start=False)
    cold.fit(counts)
    first = cold.encode(counts)
    cold.fit(counts)
    second = cold.encode(counts)
    replay = _compressor(epochs=3, warm_start=False)
    replay.generation = 1
    replay.fit(counts)
    assert not np.array_equal(first, second)
    np.testing.assert_array_equal(replay.encode(counts), second)


def test_max_cstat_selection_needs_two_hundred_spectra(spectra):
    counts, cstat = spectra["counts"], spectra["cstat"]
    threshold = float(np.quantile(cstat, 0.5))
    n_keep = int(np.sum(cstat <= threshold))
    assert n_keep >= 200

    messages = []
    kept = _compressor(epochs=1, log=messages.append).fit(counts, cstat=cstat, max_cstat=threshold)
    assert kept["n_train"] + kept["n_val"] == n_keep
    assert any("training on the" in m for m in messages)

    everything = _compressor(epochs=1).fit(counts, cstat=cstat, max_cstat=float(np.quantile(cstat, 0.1)))
    assert everything["n_train"] + everything["n_val"] == len(counts)
    ignored = _compressor(epochs=1).fit(counts, cstat=cstat)
    assert ignored["n_train"] + ignored["n_val"] == len(counts)


def test_fit_validates_inputs(spectra):
    comp = _compressor(epochs=1)
    counts = spectra["counts"]
    bad = counts.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        comp.fit(bad)
    with pytest.raises(ValueError, match="training spectra"):
        comp.fit(counts[:2])
    assert comp.generation == 0
