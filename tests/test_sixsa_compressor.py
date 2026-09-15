"""Tests for `bsixsa.backend.sixsa.compressor`."""

from __future__ import annotations

import copy
import pickle
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault("xspec", MagicMock())

torch = pytest.importorskip("torch")

from _analytic import AnalyticSpectrum

from bsixsa._cube import to_physical
from bsixsa.backend.sixsa.compressor import (
    Compressor,
    cstat_loss,
    cstat_moments,
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
    u = 0.3 + 0.4 * rng.random((600, 2))
    theta = to_physical(prior, u)
    mu = spectrum.model_counts(theta)
    counts = rng.poisson(mu).astype(np.float32)
    cstat = spectrum.cstat(theta)
    return {"counts": counts, "mu": mu, "cstat": cstat}


@pytest.fixture
def restore_torch():
    threads = torch.get_num_threads()
    state = torch.get_rng_state()
    yield
    torch.set_num_threads(threads)
    torch.set_rng_state(state)


def _compressor(**kwargs):
    settings = dict(d=2, epochs=40, batch_size=32, lr=1e-2, patience=20, seed=0, torch_threads=1)
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

    assert torch.isfinite(loss)
    assert torch.all(torch.isfinite(mu.grad))
    np.testing.assert_allclose(float(loss), _numpy_cash(mu.detach().numpy(), k.numpy()), rtol=1e-12)
    # A spectrum of zeros against its model costs 2 * sum(mu).
    zero = cstat_loss(mu[:1].detach(), k[:1])
    np.testing.assert_allclose(float(zero), 2.0 * (0.5 + 2.0 + 1e-6), rtol=1e-12)


def test_cstat_moments_high_and_low_count_limits():
    e_high, v_high = cstat_moments(np.full(10, 200.0))
    assert e_high == pytest.approx(10.0, rel=0.02)
    assert v_high == pytest.approx(20.0, rel=0.05)

    e_low, _ = cstat_moments([0.1])
    assert e_low == pytest.approx(0.47, abs=0.02)
    assert cstat_moments([0.0, -1.0]) == (0.0, 0.0)


def test_cstat_moments_agree_with_monte_carlo():
    rng = np.random.default_rng(2)
    mu = np.array([0.3, 1.5, 4.0, 12.0])
    k = rng.poisson(mu, size=(200_000, 4)).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(k > 0, k * np.log(np.where(k > 0, k, 1.0) / mu), 0.0)
    c = 2.0 * np.sum(mu - k + term, axis=1)

    expectation, variance = cstat_moments(mu)

    assert expectation == pytest.approx(c.mean(), abs=5 * c.std() / np.sqrt(len(c)))
    assert variance == pytest.approx(c.var(), rel=0.03)


@pytest.mark.parametrize(("d", "latent"), [(1, 8), (3, 8), (5, 8), (7, 16), (14, 28), (16, 32), (40, 32)])
def test_default_latent_dim_v2_values(d, latent):
    assert default_latent_dim(d) == latent


def test_default_hidden_dims_v2_values():
    assert default_hidden_dims(2502, 28, n_layers=3) == [625, 156, 56]
    assert default_hidden_dims(2502, 28) == [625, 156]
    assert default_hidden_dims(64, 8, n_layers=3) == [32, 16, 16]


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def test_constructor_defaults():
    comp = Compressor(2502, d=14)

    assert comp.latent_dim == 28
    assert comp.hidden_dims == [625, 156, 56]
    assert comp.generation == 0
    assert comp.model is None
    assert comp.batch_size == 512
    assert comp.device == torch.device("cpu")
    assert comp.history == [] and comp.curves == []


def test_constructor_validation():
    with pytest.raises(ValueError, match="latent_dim"):
        Compressor(64)
    with pytest.raises(ValueError, match="norm"):
        Compressor(64, d=2, norm="group")
    with pytest.raises(ValueError, match="batch_size"):
        Compressor(64, d=2, batch_size="big")
    with pytest.raises(ValueError, match="batch_size"):
        Compressor(64, d=2, batch_size=0)
    with pytest.raises(ValueError, match="val_fraction"):
        Compressor(64, d=2, val_fraction=1.0)
    assert Compressor(64, latent_dim=12).latent_dim == 12
    assert Compressor(64, d=2, batch_size=None).batch_size == "auto"
    assert Compressor(64, d=2, norm="none").norm is None


@pytest.mark.parametrize(("norm", "kind"), [("batch", torch.nn.BatchNorm1d), ("layer", torch.nn.LayerNorm)])
def test_architecture_mirrors_v2(spectra, norm, kind):
    comp = _compressor(norm=norm, epochs=1, dropout_rate=0.1)
    comp.fit(spectra["counts"])

    encoder = list(comp.model.encoder)
    decoder = list(comp.model.decoder)
    # Linear -> norm -> GELU -> Dropout per hidden layer, then a linear latent.
    assert [type(m) for m in encoder[:4]] == [torch.nn.Linear, kind, torch.nn.GELU, torch.nn.Dropout]
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
    reconstruction = comp.reconstruct(counts)
    assert reconstruction.dtype == np.float32 and reconstruction.shape == counts.shape
    assert np.all(reconstruction > 0)
    assert _numpy_cash(reconstruction, counts) < 0.5 * _numpy_cash(mean, counts)


def test_fit_record_contents(spectra):
    comp = _compressor(epochs=5, patience=0)

    record = comp.fit(spectra["counts"])

    assert record["generation"] == 1
    assert record["epochs"] == 5
    assert record["n_train"] == 480 and record["n_val"] == 120
    assert record["warm"] is False
    assert record["batch_size"] == 32
    assert record["seconds"] > 0
    assert np.isfinite(record["val_cstat"])
    assert comp.history == [record]
    record["generation"] = 99
    assert comp.history[0]["generation"] == 1
    assert comp.curves[0]["epoch"] == [1, 2, 3, 4, 5]


def test_generation_increments_on_every_fit_including_warm_refits(spectra):
    comp = _compressor(epochs=3)

    comp.fit(spectra["counts"])
    first_weight = comp.model.encoder[0].weight.detach().clone()
    scaler = comp.model.mu.detach().clone()
    comp.fit(spectra["counts"][:300])
    comp.fit(spectra["counts"], epochs=2)

    assert comp.generation == 3
    assert [r["generation"] for r in comp.history] == [1, 2, 3]
    assert [r["warm"] for r in comp.history] == [False, True, True]
    assert comp.history[2]["epochs"] == 2
    # Warm refits keep the scaler and continue from the previous weights.
    torch.testing.assert_close(comp.model.mu, scaler)
    assert not torch.equal(comp.model.encoder[0].weight, first_weight)


def test_warm_refit_starts_from_the_current_weights_and_a_failed_fit_changes_nothing(spectra, monkeypatch):
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

    assert seen.keys() == before.keys()
    assert all(torch.equal(seen[name], before[name]) for name in before)
    assert comp.generation == 1 and len(comp.history) == 1
    np.testing.assert_array_equal(comp.encode(spectra["counts"]), codes)


def test_cold_refits_rebuild_the_model(spectra):
    comp = _compressor(epochs=2, warm_start=False)

    comp.fit(spectra["counts"])
    model = comp.model
    comp.fit(spectra["counts"][:300])

    assert comp.model is not model
    assert [r["warm"] for r in comp.history] == [False, False]
    assert comp.generation == 2


def test_check_generation_raises_on_stale_codes(spectra):
    comp = _compressor(epochs=2)
    comp.fit(spectra["counts"])
    codes, generation = comp.encode_with_generation(spectra["counts"][:5])

    comp.check_generation(generation)
    comp.fit(spectra["counts"])

    with pytest.raises(RuntimeError, match="z_obs was encoded at compressor generation 1"):
        comp.check_generation(generation, what="z_obs")
    with pytest.raises(RuntimeError, match="generation 1"):
        comp.check_generation(generation)
    comp.check_generation(2)
    assert codes.shape == (5, 8)


def test_encode_shape_dtype_and_determinism(spectra):
    comp = _compressor(epochs=3, dropout_rate=0.2)
    comp.fit(spectra["counts"])
    counts = spectra["counts"]

    codes = comp.encode(counts)
    again = comp.encode(counts)
    batched = comp.encode(counts, batch=7)
    single = comp.encode(counts[3])

    assert codes.dtype == np.float32 and codes.shape == (len(counts), comp.latent_dim)
    np.testing.assert_array_equal(codes, again)
    # Batching only changes the floating-point kernels, not the result.
    np.testing.assert_allclose(batched, codes, rtol=1e-5, atol=1e-5)
    assert single.shape == (1, comp.latent_dim)
    np.testing.assert_allclose(single[0], codes[3], rtol=1e-6, atol=1e-6)
    # Evaluation mode is set by encode itself, not inherited.
    comp.model.train()
    np.testing.assert_array_equal(comp.encode(counts), codes)


def test_encode_before_fit_and_wrong_width_raise(spectra):
    comp = _compressor()

    with pytest.raises(RuntimeError, match="not been fitted"):
        comp.encode(spectra["counts"])
    comp.fit(spectra["counts"], epochs=1)
    with pytest.raises(ValueError, match="shape"):
        comp.encode(np.zeros((2, N_BINS + 1)))


def test_batch_size_auto_is_resolved_per_fit_and_never_written_back(spectra):
    comp = _compressor(batch_size="auto", epochs=2)

    first = comp.fit(spectra["counts"])
    second = comp.fit(spectra["counts"][:200])
    many = np.tile(spectra["counts"], (20, 1))
    third = _compressor(batch_size="auto", epochs=1).fit(many)

    assert first["batch_size"] == max(16, min(512, 480 // 20)) == 24
    assert second["batch_size"] == 16
    assert third["batch_size"] == 480
    assert comp.batch_size == "auto"


def test_fit_is_deterministic_for_a_seed(spectra, restore_torch):
    counts = spectra["counts"]

    a = _compressor(epochs=8)
    b = _compressor(epochs=8)
    c = _compressor(epochs=8, seed=None)
    rec_a, rec_b = a.fit(counts), b.fit(counts)
    rec_c = c.fit(counts, seed=0)

    np.testing.assert_array_equal(a.encode(counts), b.encode(counts))
    np.testing.assert_array_equal(a.encode(counts), c.encode(counts))
    assert rec_a["val_cstat"] == rec_b["val_cstat"] == rec_c["val_cstat"]

    a.fit(counts)
    b.fit(counts)
    np.testing.assert_array_equal(a.encode(counts), b.encode(counts))
    other = _compressor(epochs=8, seed=1)
    other.fit(counts)
    assert not np.array_equal(other.encode(counts), c.encode(counts))


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


def test_max_cstat_selection_honours_min_kept(spectra):
    counts, cstat = spectra["counts"], spectra["cstat"]
    threshold = float(np.quantile(cstat, 0.5))
    n_keep = int(np.sum(cstat <= threshold))

    kept = _compressor(epochs=1).fit(counts, cstat=cstat, max_cstat=threshold, min_kept=200)
    assert kept["n_train"] + kept["n_val"] == n_keep

    messages = []
    everything = _compressor(epochs=1, log=messages.append).fit(
        counts, cstat=cstat, max_cstat=threshold, min_kept=n_keep + 1
    )
    assert everything["n_train"] + everything["n_val"] == len(counts)
    assert any("training on all" in m for m in messages)

    ignored = _compressor(epochs=1).fit(counts, cstat=cstat)
    assert ignored["n_train"] + ignored["n_val"] == len(counts)


def test_fit_validates_inputs(spectra):
    comp = _compressor(epochs=1)
    counts = spectra["counts"]

    with pytest.raises(ValueError, match="shape"):
        comp.fit(counts[:, :10])
    with pytest.raises(ValueError, match="cstat has"):
        comp.fit(counts, cstat=np.zeros(3), max_cstat=1.0)
    bad = counts.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        comp.fit(bad)
    with pytest.raises(ValueError, match="training spectra"):
        comp.fit(counts[:2])
    assert comp.generation == 0


def test_fit_leaves_torch_global_state_alone(spectra, restore_torch):
    torch.manual_seed(1234)
    state = torch.get_rng_state()
    numpy_state = np.random.get_state()[1].copy()
    threads = torch.get_num_threads()
    target = 1 if threads != 1 else 2

    comp = _compressor(epochs=3, torch_threads=target, dropout_rate=0.2)
    comp.fit(spectra["counts"])
    comp.encode(spectra["counts"])

    assert torch.equal(torch.get_rng_state(), state)
    assert torch.get_num_threads() == threads
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state)


def test_fit_runs_with_the_requested_threads(spectra, monkeypatch):
    seen = []
    comp = _compressor(epochs=1, torch_threads=1)
    original = comp._train

    def spy(*args, **kwargs):
        seen.append(torch.get_num_threads())
        return original(*args, **kwargs)

    monkeypatch.setattr(comp, "_train", spy)
    comp.fit(spectra["counts"])

    assert seen == [1]


# ---------------------------------------------------------------------------
# copies
# ---------------------------------------------------------------------------


class _UncopyableLog:
    def __init__(self):
        self.messages = []

    def __call__(self, message):
        self.messages.append(message)

    def __deepcopy__(self, memo):
        raise TypeError("a logger must not be copied")

    def __reduce__(self):
        raise TypeError("a logger must not be pickled")


def test_deepcopy_is_independent_and_shares_the_logger(spectra):
    log = _UncopyableLog()
    comp = _compressor(epochs=2, log=log)
    comp.fit(spectra["counts"])
    codes = comp.encode(spectra["counts"])

    clone = copy.deepcopy(comp)
    assert clone._log is log
    assert clone.model is not comp.model
    np.testing.assert_array_equal(clone.encode(spectra["counts"]), codes)

    clone.fit(spectra["counts"])

    assert clone.generation == 2 and comp.generation == 1
    assert len(comp.history) == 1 and len(clone.history) == 2
    np.testing.assert_array_equal(comp.encode(spectra["counts"]), codes)
    assert not np.array_equal(clone.encode(spectra["counts"]), codes)


def test_pickle_drops_the_logger(spectra):
    comp = _compressor(epochs=1, log=_UncopyableLog())
    comp.fit(spectra["counts"])

    restored = pickle.loads(pickle.dumps(comp))

    assert restored._log is None
    assert restored.generation == 1
    np.testing.assert_array_equal(restored.encode(spectra["counts"]), comp.encode(spectra["counts"]))
    restored.fit(spectra["counts"], epochs=1)
    assert restored.generation == 2
