"""Tests for `bsixsa._torch_utils.torch_threads`."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("xspec", MagicMock())

torch = pytest.importorskip("torch")

from bsixsa._torch_utils import torch_threads  # noqa: E402


@pytest.fixture
def restore_threads():
    previous = torch.get_num_threads()
    yield previous
    torch.set_num_threads(previous)


def _other_count(current: int) -> int:
    return 1 if current != 1 else 2


def test_torch_threads_sets_and_restores(restore_threads):
    before = restore_threads
    target = _other_count(before)

    with torch_threads(target):
        assert torch.get_num_threads() == target

    assert torch.get_num_threads() == before


def test_torch_threads_restores_after_an_exception(restore_threads):
    before = restore_threads
    target = _other_count(before)

    with pytest.raises(RuntimeError, match="boom"):
        with torch_threads(target):
            assert torch.get_num_threads() == target
            raise RuntimeError("boom")

    assert torch.get_num_threads() == before


def test_torch_threads_nests(restore_threads):
    before = restore_threads
    with torch_threads(2):
        with torch_threads(1):
            assert torch.get_num_threads() == 1
        assert torch.get_num_threads() == 2
    assert torch.get_num_threads() == before


def test_torch_threads_none_is_a_no_op(restore_threads):
    before = restore_threads
    with torch_threads(None):
        assert torch.get_num_threads() == before
    assert torch.get_num_threads() == before


def test_torch_threads_rejects_non_positive_counts(restore_threads):
    before = restore_threads
    with pytest.raises(ValueError):
        with torch_threads(0):
            pass  # pragma: no cover - never entered
    assert torch.get_num_threads() == before
