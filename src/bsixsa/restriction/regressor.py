r"""The C-stat surrogate of the history-matching waves: a small MLP on the unit cube."""

from __future__ import annotations

import math

import numpy as np
import torch

from .._torch_utils import torch_threads
from .config import HistoryMatching

__all__ = ["CstatRegressor", "train_regressor"]

_MIN_DELTA = 1e-6


class CstatRegressor(torch.nn.Module):
    r"""MLP surrogate $u \mapsto \hat C(u)$ on the unit cube.

    The network predicts $\log(C - c_{ref})$, or $\log C$ without a reference.
    The delta target keeps the surrogate's relative error small against the
    posterior span when the best C-stat is large.

    Parameters:
        d: Input dimension.
        hidden: Width of the first hidden layer; later layers halve it (never below 8).
        blocks: Number of hidden Linear+GELU layers.
        c_ref: Reference C-stat of the delta target, or ``None`` for $\log C$.

    Raises:
        ValueError: If *d*, *hidden* or *blocks* is < 1, or *c_ref* is not finite.
    """

    def __init__(self, d: int, hidden: int = 128, blocks: int = 3, c_ref: float | None = None):
        super().__init__()
        d, hidden, blocks = int(d), int(hidden), int(blocks)
        if d < 1 or hidden < 1 or blocks < 1:
            raise ValueError(
                f"CstatRegressor expects d, hidden and blocks >= 1, got {d}, {hidden}, {blocks}."
            )
        if c_ref is not None and not math.isfinite(float(c_ref)):
            raise ValueError(f"c_ref must be finite or None, got {c_ref}.")
        self.d = d
        self.register_buffer("t_mu", torch.zeros(1, d))
        self.register_buffer("t_sd", torch.ones(1, d))
        self.register_buffer("y_mu", torch.zeros(1))
        self.register_buffer("y_sd", torch.ones(1))
        self.c_ref = None if c_ref is None else float(c_ref)
        self.widths = [max(8, hidden // 2**k) for k in range(blocks)]
        layers: list[torch.nn.Module] = []
        previous = d
        for width in self.widths:
            layers += [torch.nn.Linear(previous, width), torch.nn.GELU()]
            previous = width
        layers.append(torch.nn.Linear(previous, 1))
        self.net = torch.nn.Sequential(*layers)

    def to_target(self, cstat: np.ndarray) -> np.ndarray:
        r"""Map C-stat values to the target $\log(\max(C - c_{ref}, 10^{-6}))$, in float64."""
        c = np.asarray(cstat, dtype=np.float64).reshape(-1)
        shifted = c if self.c_ref is None else c - self.c_ref
        with np.errstate(invalid="ignore"):
            return np.log(np.clip(shifted, _MIN_DELTA, None))

    def forward(self, u: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Predict the target for unit-cube points of shape ``(n, d)``."""
        z = (torch.as_tensor(u, dtype=torch.float32) - self.t_mu) / self.t_sd
        return self.net(z).reshape(-1) * self.y_sd + self.y_mu

    def predict_cstat(self, u: np.ndarray | torch.Tensor, batch: int = 200_000) -> np.ndarray:
        """Predict the C-stat of unit-cube points of shape ``(n, d)``, batched and without gradients."""
        if isinstance(u, torch.Tensor):
            u = u.detach().to(torch.float32)
        else:
            u = torch.as_tensor(np.asarray(u, dtype=np.float32))
        if u.ndim != 2 or u.shape[1] != self.d:
            raise ValueError(f"predict_cstat expects shape (n, {self.d}), got {tuple(u.shape)}.")
        n = u.shape[0]
        out = np.empty(n, dtype=np.float64)
        batch = max(1, int(batch))
        with torch.no_grad():
            for start in range(0, n, batch):
                y = self.forward(u[start : start + batch]).double().numpy()
                # Inverted in float64: float32 would round a C-stat of 1e5 to 0.01.
                with np.errstate(over="ignore"):
                    values = np.exp(y)
                out[start : start + len(y)] = values if self.c_ref is None else self.c_ref + values
        return out


def _standardise(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column mean and floored unbiased standard deviation (1 for a single row)."""
    mean = np.asarray(values.mean(axis=0))
    sd = np.asarray(values.std(axis=0, ddof=1)) if len(values) > 1 else np.ones_like(mean)
    return mean, np.maximum(sd, 1e-6)


def train_regressor(
    u: np.ndarray,
    cstat: np.ndarray,
    *,
    cfg: HistoryMatching,
    held_out: np.ndarray,
    c_ref: float | None = None,
    focus_below: float | None = None,
    seed: int | None = None,
) -> tuple[CstatRegressor, dict]:
    r"""Fit a [`CstatRegressor`][bsixsa.restriction.regressor.CstatRegressor].

    Rows with an infinite C-stat are dropped. The loss is a weighted MSE in
    target space, with weight ``cfg.focus_weight`` on the rows with
    $C \le$ ``focus_below * cfg.focus_factor``; the weights with the best
    validation loss are restored.

    Parameters:
        u: Unit-cube points, shape ``(n, d)``.
        cstat: Exact C-stat of the points, shape ``(n,)``.
        cfg: History-matching settings.
        held_out: Boolean mask of shape ``(n,)`` marking the validation rows.
        c_ref: Reference of the delta target, ``None`` for $\log C$.
        focus_below: Previous threshold; ``None`` disables the focus weighting.
        seed: Seed of the initialisation and mini-batch order.

    Returns:
        ``(model, info)``, *info* holding ``rmse_log``, ``rmse_cstat`` (on the
        focus rows when more than 5 are held out), ``epochs``, ``n_train``,
        ``n_val`` and ``n_dropped``.

    Raises:
        ValueError: On inconsistent shapes, fewer than 2 finite rows, or an
            empty training or validation set.
    """
    u = np.asarray(u, dtype=np.float64)
    cstat = np.asarray(cstat, dtype=np.float64).reshape(-1)
    if u.ndim != 2 or u.shape[0] != cstat.shape[0]:
        raise ValueError(
            f"train_regressor expects u of shape (n, d) and cstat of shape (n,), got {u.shape} and {cstat.shape}."
        )
    held_out = np.asarray(held_out)
    if held_out.dtype != bool or held_out.shape != cstat.shape:
        raise ValueError(
            f"held_out must be a boolean mask of shape {cstat.shape}, got {held_out.dtype} of shape {held_out.shape}."
        )
    d = u.shape[1]

    with torch_threads(cfg.torch_threads), torch.random.fork_rng(devices=[]):
        if seed is not None:
            torch.manual_seed(int(seed))
        model = CstatRegressor(d, hidden=cfg.hidden, blocks=cfg.blocks, c_ref=c_ref)
        target = model.to_target(cstat)
        finite = np.isfinite(target)
        n_ok = int(finite.sum())
        if n_ok < 2:
            raise ValueError(
                f"train_regressor needs at least 2 rows with a finite C-stat, got {n_ok} of {cstat.size}."
            )
        train_idx = np.flatnonzero(finite & ~held_out)
        val_idx = np.flatnonzero(finite & held_out)
        if val_idx.size == 0 or train_idx.size == 0:
            raise ValueError(
                f"held_out leaves {train_idx.size} finite training and {val_idx.size} finite "
                "validation rows; both need at least one."
            )

        weights = np.ones(cstat.size, dtype=np.float64)
        focus = None
        if focus_below is not None:
            focus = cstat <= float(focus_below) * cfg.focus_factor
            weights[focus] = cfg.focus_weight

        t_mu, t_sd = _standardise(u[train_idx])
        y_mu, y_sd = _standardise(target[train_idx])
        model.t_mu.copy_(torch.as_tensor(t_mu, dtype=torch.float32).reshape(1, d))
        model.t_sd.copy_(torch.as_tensor(t_sd, dtype=torch.float32).reshape(1, d))
        model.y_mu.copy_(torch.as_tensor(y_mu, dtype=torch.float32).reshape(1))
        model.y_sd.copy_(torch.as_tensor(y_sd, dtype=torch.float32).reshape(1))

        def tensors(idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return (
                torch.as_tensor(u[idx], dtype=torch.float32),
                torch.as_tensor(target[idx], dtype=torch.float32),
                torch.as_tensor(weights[idx], dtype=torch.float32),
            )

        x_tr, y_tr, w_tr = tensors(train_idx)
        x_va, y_va, w_va = tensors(val_idx)
        w_va_sum = w_va.sum().clamp_min(1e-9)
        optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

        best_loss, best_state, since, used = math.inf, None, 0, 0
        n_train = len(train_idx)
        for epoch in range(cfg.epochs):
            model.train()
            perm = torch.randperm(n_train)
            for start in range(0, n_train, cfg.batch_size):
                b = perm[start : start + cfg.batch_size]
                optimiser.zero_grad()
                wb = w_tr[b]
                loss = (((model(x_tr[b]) - y_tr[b]) ** 2) * wb).sum() / wb.sum().clamp_min(1e-9)
                loss.backward()
                optimiser.step()
            model.eval()
            with torch.no_grad():
                # Same weighting as the training loss.
                val_loss = float((((model(x_va) - y_va) ** 2) * w_va).sum() / w_va_sum)
            used = epoch + 1
            if val_loss < best_loss - 1e-6:
                best_loss, since = val_loss, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                since += 1
                if since >= cfg.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            res_log = (model(x_va) - y_va).double().numpy()
        res_cstat = model.predict_cstat(u[val_idx]) - cstat[val_idx]

    rmse_log = float(np.sqrt(np.mean(res_log**2)))
    rmse_cstat = float(np.sqrt(np.mean(res_cstat**2)))
    if focus is not None:
        focus_val = focus[val_idx]
        # A global RMSE would be dominated by the far, high-C-stat folds.
        if int(focus_val.sum()) > 5:
            rmse_cstat = float(np.sqrt(np.mean(res_cstat[focus_val] ** 2)))
    info = {
        "rmse_log": rmse_log,
        "rmse_cstat": rmse_cstat,
        "epochs": used,
        "n_train": int(n_train),
        "n_val": int(val_idx.size),
        "n_dropped": int(cstat.size - n_ok),
    }
    return model, info
