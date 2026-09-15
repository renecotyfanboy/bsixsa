"""Tests for the SIXSA `posterior_source` knob.

Trades statistical exactness against sample diversity:
  * "pool"          -- resample the banked pooled multiple-IS draws (exact, duplicates)
  * "emulator_pass" -- resample the emulator pass's flow-IS draws (fresh + IS-corrected)
  * "best_flow"     -- draw fresh from the best flow (no duplicates, no IS correction)
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

sys.modules.setdefault("xspec", MagicMock())

from bsixsa.backend.config import SIXSA
from bsixsa.backend.sixsa import SIXSABackend

D = 2


def _backend(source, tmp_path):
    solver = MagicMock()
    solver.prior.ndim = D
    solver.prior.from_unit_gaussian = lambda z: np.asarray(z)
    solver.outputfiles_basename = str(tmp_path)
    solver.parameter_names = [f"p{i}" for i in range(D)]
    return SIXSABackend(
        solver=solver,
        config=SIXSA(posterior_source=source, plot_diagnostics=False, trace=False),
    )


class _Bank:
    """Stub bank exposing the pooled_weighted surface `_sample_latent` uses."""

    def __init__(self, n=64):
        self.n = n
        self._z = torch.arange(n, dtype=torch.float32).repeat(D, 1).T

    def __len__(self):
        return self.n

    def pooled_weighted(self, prior=None):
        return self._z, torch.full((self.n,), 1.0 / self.n, dtype=torch.float64)


class _Flow:
    """Stub best-flow producing distinct continuous draws."""

    def sample(self, shape, *a, **k):
        n = shape[0] if isinstance(shape, (tuple, list, torch.Size)) else shape
        return torch.randn(int(n), D)


def _emulator_info(n=128):
    z = torch.arange(1000, 1000 + n, dtype=torch.float32).repeat(D, 1).T
    w = torch.full((n,), 1.0 / n, dtype=torch.float64)
    return {"weighted": (z, w), "n_draws": n}


def _fully_populated(backend):
    backend.bank = _Bank()
    backend.emulator_pass_info = _emulator_info()
    backend.best_ensemble = _Flow()
    return backend


@pytest.mark.parametrize(
    "source,expected",
    [("pool", "pool"), ("emulator_pass", "emulator_pass"), ("best_flow", "best_flow")],
)
def test_posterior_source_selects_its_own_source(source, expected, tmp_path):
    """With every source available, each setting uses exactly the one it names."""
    backend = _fully_populated(_backend(source, tmp_path))
    z = backend._sample_latent(32)

    assert backend.posterior_source_used == expected
    assert tuple(z.shape) == (32, D)
    if expected == "pool":
        assert float(z.max()) < 1000  # bank rows are 0..63
    elif expected == "emulator_pass":
        assert float(z.min()) >= 1000  # emulator rows are 1000..1127


def test_default_is_pool(tmp_path):
    assert SIXSA().posterior_source == "pool"
    backend = _fully_populated(_backend("pool", tmp_path))
    backend._sample_latent(8)
    assert backend.posterior_source_used == "pool"


def test_best_flow_yields_distinct_rows_pool_duplicates(tmp_path):
    """The whole point of the knob: 'best_flow' does not duplicate, 'pool' does."""
    n = 500

    pooled = _fully_populated(_backend("pool", tmp_path))._sample_latent(n)
    fresh = _fully_populated(_backend("best_flow", tmp_path))._sample_latent(n)

    # Bank holds only 64 distinct rows, so 500 draws must repeat heavily.
    assert len({tuple(r) for r in pooled.tolist()} ) <= 64
    # Fresh continuous draws are essentially all distinct.
    assert len({tuple(r) for r in fresh.tolist()}) == n


def test_falls_back_when_preferred_source_missing(tmp_path):
    """Each source degrades to the next available one, never failing."""
    # emulator_pass requested but the pass never ran -> falls back to the bank.
    backend = _backend("emulator_pass", tmp_path)
    backend.bank = _Bank()
    backend.emulator_pass_info = None
    backend._sample_latent(8)
    assert backend.posterior_source_used == "pool"

    # best_flow requested, nothing trained -> falls back to the bank.
    backend = _backend("best_flow", tmp_path)
    backend.bank = _Bank()
    backend.best_ensemble = None
    backend._sample_latent(8)
    assert backend.posterior_source_used == "pool"

    # Nothing at all -> the prior, still returning the requested shape.
    backend = _backend("pool", tmp_path)
    z = backend._sample_latent(8)
    assert backend.posterior_source_used == "prior"
    assert tuple(z.shape) == (8, D)


def test_invalid_source_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="posterior_source"):
        SIXSA(posterior_source="bogus")
