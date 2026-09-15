"""Small torch helpers that must not import torch at module import time."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

__all__ = ["torch_threads"]


@contextlib.contextmanager
def torch_threads(n: int | None) -> Iterator[None]:
    """Temporarily set the number of intra-op torch threads.

    The previous value of ``torch.get_num_threads()`` is restored on exit,
    including when the body raises. torch is imported lazily, and only when a
    thread count is requested, so ``torch_threads(None)`` works without torch.

    Parameters:
        n:
            Number of threads to use inside the block. ``None`` leaves the
            torch configuration untouched.

    Yields:
        ``None``.

    Raises:
        ValueError: If ``n`` is not ``None`` and smaller than 1.
    """
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
