"""Torch helpers."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

__all__ = ["torch_threads"]


@contextlib.contextmanager
def torch_threads(n: int | None) -> Iterator[None]:
    """Temporarily set the number of intra-op torch threads; ``None`` leaves it unchanged."""
    if n is None:
        yield
        return

    n = int(n)
    if n < 1:
        raise ValueError(f"torch_threads expects n >= 1 or None, got {n}.")

    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(previous)
