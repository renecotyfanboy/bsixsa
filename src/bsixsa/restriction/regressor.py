r"""The C-stat surrogate of the history-matching waves.

A small MLP regresses $\log C$ (or $\log(C - c_{ref})$) on the unit cube. The
region only needs the surrogate to rank points correctly near the boundary,
which is a far weaker requirement than reproducing $C$ everywhere, so a few
thousand folds suffice.
"""

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

    The network predicts the standardised target $y = \log(C - c_{ref})$, or
    $\log C$ when no reference is given. The delta target matters because the
    surrogate error is relative while the posterior width is absolute: at
    $C \sim 3000$ a 3 % error is $\pm 90$ against a posterior spanning a few
    tens, whereas on a delta of a few hundred it is a few units.

    Widths taper as ``max(8, hidden // 2**k)`` for ``k < blocks``: the output
    is a scalar, so capacity is spent where the input still has structure.
    The standardisation constants and $c_{ref}$ are buffers, so a saved model
    cannot be paired with the wrong scalers.

    Parameters:
        d:
            Input dimension.
        hidden:
            Width of the first hidden layer.
        blocks:
            Number of hidden Linear+GELU layers.
        c_ref:
            Reference C-stat of the delta target, or ``None`` for $\log C$.

    Raises:
        ValueError: If *d*, *hidden* or *blocks* is smaller than 1, or *c_ref*
            is not finite.
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
        # float64 so that the inverse map does not round a C-stat of 1e5 to
        # the nearest 0.01; NaN encodes "no reference".
        self.register_buffer(
            "c_ref",
            torch.tensor([math.nan if c_ref is None else float(c_ref)], dtype=torch.float64),
        )
        self.widths = [max(8, hidden // 2**k) for k in range(blocks)]
        layers: list[torch.nn.Module] = []
        previous = d
        for width in self.widths:
            layers += [torch.nn.Linear(previous, width), torch.nn.GELU()]
            previous = width
        layers.append(torch.nn.Linear(previous, 1))
        self.net = torch.nn.Sequential(*layers)

    @property
    def reference(self) -> float | None:
        """Reference C-stat of the delta target, or ``None``."""
        value = float(self.c_ref[0])
        return None if math.isnan(value) else value

    def to_target(self, cstat) -> np.ndarray:
        r"""Map C-stat values to the unstandardised target, in float64.

        Parameters:
            cstat:
                C-stat values, shape ``(n,)``.

        Returns:
            $\log(\max(C - c_{ref}, 10^{-6}))$, or $\log(\max(C, 10^{-6}))$
            without a reference. $+\infty$ stays $+\infty$.
        """
        c = np.asarray(cstat, dtype=np.float64).reshape(-1)
        ref = self.reference
        shifted = c if ref is None else c - ref
        with np.errstate(invalid="ignore"):
            return np.log(np.clip(shifted, _MIN_DELTA, None))

    def forward(self, u) -> torch.Tensor:
        """Predict the unstandardised target.

        Parameters:
            u:
                Unit-cube points, shape ``(n, d)``.

        Returns:
            float32 tensor of shape ``(n,)``.
        """
        z = (torch.as_tensor(u, dtype=torch.float32) - self.t_mu) / self.t_sd
        return self.net(z).reshape(-1) * self.y_sd + self.y_mu

    def predict_cstat(self, u, batch: int = 200_000) -> np.ndarray:
        """Predict the C-stat, batched and without gradients.

        Parameters:
            u:
                Unit-cube points, numpy array or tensor of shape ``(n, d)``.
            batch:
                Rows per forward pass.

        Returns:
            float64 array of shape ``(n,)``.

        Raises:
            ValueError: If *u* does not have ``d`` columns.
        """
        if isinstance(u, torch.Tensor):
            u = u.detach().to(torch.float32)
        else:
            u = torch.as_tensor(np.asarray(u, dtype=np.float32))
        if u.ndim != 2 or u.shape[1] != self.d:
            raise ValueError(f"predict_cstat expects shape (n, {self.d}), got {tuple(u.shape)}.")
        n = u.shape[0]
        out = np.empty(n, dtype=np.float64)
        ref = self.reference
        batch = max(1, int(batch))
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                for start in range(0, n, batch):
                    y = self.forward(u[start : start + batch]).double().numpy()
                    # Inverted in float64: float32 keeps ~7 significant digits,
                    # which rounds a C-stat of 1e5 to the nearest 0.01.
                    with np.errstate(over="ignore"):
                        values = np.exp(y)
                    out[start : start + len(y)] = values if ref is None else ref + values
        finally:
            self.train(was_training)
        return out


def _standardise(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column mean and floored unbiased standard deviation (1 for a single row)."""
    mean = np.asarray(values.mean(axis=0))
    sd = np.asarray(values.std(axis=0, ddof=1)) if len(values) > 1 else np.ones_like(mean)
    return mean, np.maximum(sd, 1e-6)


def train_regressor(
    u,
    cstat,
    *,
    cfg: HistoryMatching,
    c_ref: float | None = None,
    focus_below: float | None = None,
    seed: int | None = None,
    val_idx: np.ndarray | None = None,
) -> tuple[CstatRegressor, dict]:
    r"""Fit a [`CstatRegressor`][bsixsa.restriction.regressor.CstatRegressor].

    Rows whose target is not finite (a $+\infty$ C-stat) are dropped. The
    rows *val_idx* are held out, or a random ``cfg.validation_fraction`` of
    the finite rows when it is ``None``; the training loss and the
    early-stopping loss are the same weighted MSE in target space, with
    weight ``cfg.focus_weight`` on the rows with
    $C \le$ ``focus_below * cfg.focus_factor``. The weights with the best
    validation loss are restored.

    Training is reproducible: torch is seeded inside ``torch.random.fork_rng``
    (the caller's global torch stream is left untouched), the random split
    uses a numpy generator with the same seed, and the thread count is
    ``cfg.torch_threads``.

    Parameters:
        u:
            Unit-cube points, shape ``(n, d)``.
        cstat:
            Exact C-stat of the points, shape ``(n,)``.
        cfg:
            History-matching settings (architecture, optimiser, focus).
        c_ref:
            Reference of the delta target, ``None`` for $\log C$.
        focus_below:
            Previous threshold; ``None`` disables the focus weighting.
        seed:
            Seed of the initialisation, the random split and the mini-batch
            order; ``None`` draws one from fresh entropy.
        val_idx:
            Indices of the rows to hold out, e.g. a split kept across calls
            so that a row validated once is never trained on; rows with a
            non-finite target are ignored. ``None`` draws a random split.

    Returns:
        ``(model, info)`` with *info* holding ``rmse_log`` (validation RMSE in
        target space), ``rmse_cstat`` (validation C-stat RMSE, restricted to
        the focus rows when more than 5 of them are held out), ``epochs``,
        ``n_train``, ``n_val``, ``n_dropped`` and ``val_idx`` (sorted indices
        into the input rows of the finite held-out rows).

    Raises:
        ValueError: If the shapes disagree, fewer than 2 rows have a finite
            target, or *val_idx* is out of range or leaves no finite row for
            training or for validation.
    """
    u = np.asarray(u, dtype=np.float64)
    cstat = np.asarray(cstat, dtype=np.float64).reshape(-1)
    if u.ndim != 2 or u.shape[0] != cstat.shape[0]:
        raise ValueError(
            f"train_regressor expects u of shape (n, d) and cstat of shape (n,), got {u.shape} and {cstat.shape}."
        )
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    seed = int(seed)
    d = u.shape[1]

    with torch_threads(cfg.torch_threads), torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = CstatRegressor(d, hidden=cfg.hidden, blocks=cfg.blocks, c_ref=c_ref)
        target = model.to_target(cstat)
        finite = np.flatnonzero(np.isfinite(target))
        n_ok = finite.size
        if n_ok < 2:
            raise ValueError(
                f"train_regressor needs at least 2 rows with a finite C-stat, got {n_ok} of {cstat.size}."
            )
        if val_idx is None:
            n_val = min(max(1, int(round(cfg.validation_fraction * n_ok))), n_ok - 1)
            order = np.random.default_rng(seed).permutation(n_ok)
            val_idx = np.sort(finite[order[:n_val]])
            train_idx = finite[order[n_val:]]
        else:
            requested = np.asarray(val_idx, dtype=np.int64).reshape(-1)
            if requested.size and (requested.min() < 0 or requested.max() >= cstat.size):
                raise ValueError(f"val_idx must index rows in [0, {cstat.size}).")
            held = np.zeros(cstat.size, dtype=bool)
            held[requested] = True
            is_finite = np.isfinite(target)
            val_idx = np.flatnonzero(held & is_finite)
            train_idx = np.flatnonzero(~held & is_finite)
            if val_idx.size == 0 or train_idx.size == 0:
                raise ValueError(
                    f"val_idx leaves {train_idx.size} finite training and {val_idx.size} finite "
                    "validation rows; both need at least one."
                )
            n_val = int(val_idx.size)

        weights = np.ones(cstat.size, dtype=np.float64)
        near = None
        if focus_below is not None:
            near = cstat <= float(focus_below) * cfg.focus_factor
            weights[near] = cfg.focus_weight

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
                # Same weighting as the training loss, or early stopping would
                # judge the fit on a criterion it is not minimising.
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
    if near is not None:
        near_val = near[val_idx]
        # Only the boundary decides where the region falls; a global RMSE is
        # dominated by the far, high-C-stat folds.
        if int(near_val.sum()) > 5:
            rmse_cstat = float(np.sqrt(np.mean(res_cstat[near_val] ** 2)))
    info = {
        "rmse_log": rmse_log,
        "rmse_cstat": rmse_cstat,
        "epochs": used,
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_dropped": int(cstat.size - n_ok),
        "val_idx": val_idx,
    }
    return model, info
