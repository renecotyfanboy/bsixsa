"""Emulator preparation pass and PyTorch likelihood emulator (spec section 10).

Two independent pieces; the round loop hard-depends on neither.

(a) :func:`emulator_importance_pass` -- draw ``emulator_min_per_param * d`` TRUE
    surrogate samples from the best ensemble posterior (NOT the truncated/flat
    proposal), simulate them exactly, and run a PLAIN flow-IS pass (no defensive
    floor): ``log w = log prior + log L - log q_mix``. Reports PSIS k_hat, ESS
    from the RAW weights and log Z; returns the weighted samples and the
    ``(theta, cstat)`` pairs that extend the run's pool.

(b) :class:`TorchLikelihoodEmulator` + :func:`train_likelihood_emulator` -- a
    pure-PyTorch MLP regressing ``theta_latent -> -0.5*cstat`` (the exact
    log-likelihood), trained ON THE PASS SAMPLES ONLY (their cstat span covers
    the posterior neighbourhood; training on the full prior pool stretches the
    target standardization and destroys accuracy at the mode). No tensorflow;
    ``pytorch-lightning`` is used only if installed and requested.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
from arviz import psislw
from torch import nn

from ._importance import compute_log_evidence


def _sample(dist, n):
    try:
        s = dist.sample((n,), show_progress_bars=False)
    except TypeError:
        s = dist.sample((n,))
    return s.detach()


def emulator_importance_pass(
    *,
    best_flow,
    prior,
    log_likelihood_fn,
    n_draws: int,
    num_posterior_samples: int = 10_000,
) -> dict | None:
    """Plain flow-IS on true surrogate draws from the best ensemble posterior."""
    if best_flow is None or n_draws <= 0:
        return None

    z = _sample(best_flow, n_draws).to(torch.float32)
    log_q = best_flow.log_prob(z).double()
    log_prior = prior.log_prob(z).double()
    log_lik = torch.as_tensor(log_likelihood_fn(z)).double().reshape(-1)
    log_w = (log_prior + log_lik - log_q)

    log_w_np = log_w.cpu().numpy()
    finite = np.isfinite(log_w_np)
    n_total = len(log_w_np)

    log_z, log_z_err, log_z_bias = compute_log_evidence(log_w_np)
    if finite.any():
        try:
            log_w_smoothed, khat = psislw(log_w_np[finite].copy())
            khat = float(khat)
        except Exception:
            log_w_smoothed, khat = log_w_np[finite], float("nan")
        w = np.exp(log_w_np[finite] - log_w_np[finite].max())
        w /= w.sum()
        ess = float(1.0 / np.sum(w**2))
        # resample-ready smoothed weights over finite draws
        ws = np.exp(log_w_smoothed - log_w_smoothed.max())
        weighted = (
            z[torch.as_tensor(finite)].detach(),
            torch.as_tensor(ws / ws.sum(), dtype=torch.float64),
        )
    else:
        khat, ess, weighted = float("inf"), 0.0, None

    cstat = (-2.0 * log_lik).cpu().numpy()
    return {
        "z": z.detach(),
        "cstat": cstat,
        "log_w": log_w.detach(),
        "weighted": weighted,
        "khat": khat,
        "ess": ess,
        "efficiency": 100.0 * ess / n_total if n_total else 0.0,
        "log_Z": log_z,
        "log_Z_err": log_z_err,
        "log_Z_bias": log_z_bias,
        "n_draws": int(n_total),
    }


# --------------------------------------------------------------------------- #
# PyTorch likelihood emulator
# --------------------------------------------------------------------------- #
class TorchLikelihoodEmulator(nn.Module):
    """MLP ``theta_latent -> -0.5*cstat`` with built-in standardization.

    Standardization statistics (theta and target) are stored as buffers so
    :meth:`predict` is fully self-contained after training.
    """

    def __init__(self, input_dim: int, hidden_units=(128, 64, 32), activation=nn.GELU):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_units:
            layers += [nn.Linear(prev, h), activation()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        d = input_dim
        self.register_buffer("theta_mean", torch.zeros(d))
        self.register_buffer("theta_std", torch.ones(d))
        self.register_buffer("y_mean", torch.zeros(1))
        self.register_buffer("y_std", torch.ones(1))

    def forward(self, theta_std: torch.Tensor) -> torch.Tensor:
        return self.net(theta_std).squeeze(-1)

    def set_standardizers(self, theta_mean, theta_std, y_mean, y_std):
        self.theta_mean.copy_(torch.as_tensor(theta_mean, dtype=torch.float32))
        self.theta_std.copy_(torch.as_tensor(theta_std, dtype=torch.float32))
        self.y_mean.copy_(torch.as_tensor([float(y_mean)], dtype=torch.float32))
        self.y_std.copy_(torch.as_tensor([float(y_std)], dtype=torch.float32))

    @torch.no_grad()
    def predict(self, theta_latent) -> torch.Tensor:
        """Return the emulated log-likelihood ``-0.5*cstat`` for ``theta_latent``."""
        theta = torch.as_tensor(np.asarray(theta_latent), dtype=torch.float32)
        theta_std = (theta - self.theta_mean) / self.theta_std
        y_std = self.forward(theta_std)
        return y_std * self.y_std.squeeze() + self.y_mean.squeeze()


def _standardize(x: np.ndarray):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return mean, std


def train_likelihood_emulator(
    theta_latent,
    cstat,
    *,
    emulator_kwargs: dict | None = None,
    path_results: str | None = None,
    logger=None,
) -> dict:
    """Train the pure-torch likelihood emulator on the emulator-pass samples.

    ``theta_latent -> y = -0.5*cstat``. The worst ``percentage_to_remove%`` of
    rows by cstat are dropped, theta and ``y`` are standardized on the training
    split, and the MLP is trained with MSE + early stopping. Returns a dict with
    the fitted ``emulator`` and validation diagnostics.
    """
    cfg = dict(emulator_kwargs or {})
    hidden_units = tuple(cfg.get("hidden_units", (128, 64, 32)))
    lr = float(cfg.get("learning_rate", 1e-3))
    num_epochs = int(cfg.get("num_epochs", 1000))
    batch_size = int(cfg.get("batch_size", 128))
    patience = int(cfg.get("patience", 50))
    test_size = float(cfg.get("test_size", 0.2))
    pct_remove = float(cfg.get("percentage_to_remove", 1.0))
    seed = int(cfg.get("seed", 0))

    theta = np.asarray(theta_latent, dtype=np.float32)
    cstat = np.asarray(cstat, dtype=np.float64).reshape(-1)
    if theta.ndim != 2 or len(theta) != len(cstat):
        raise ValueError("theta_latent must be (N, d) aligned with cstat (N,).")

    # Drop the worst percentage_to_remove% by cstat (highest cstat = worst fit).
    if 0.0 < pct_remove < 100.0 and len(cstat) > 10:
        thresh = np.quantile(cstat, 1.0 - pct_remove / 100.0)
        keep = cstat <= thresh
        theta, cstat = theta[keep], cstat[keep]
    cstat_min, cstat_med, cstat_max = (
        float(cstat.min()), float(np.median(cstat)), float(cstat.max())
    )

    y = -0.5 * cstat  # the exact log-likelihood target

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(theta))
    n_val = max(1, int(round(test_size * len(theta))))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    theta_mean, theta_std = _standardize(theta[train_idx])
    y_mean, y_std = float(y[train_idx].mean()), float(y[train_idx].std() or 1.0)

    def _prep(idx):
        ts = (theta[idx] - theta_mean) / theta_std
        ys = (y[idx] - y_mean) / y_std
        return (
            torch.as_tensor(ts, dtype=torch.float32),
            torch.as_tensor(ys, dtype=torch.float32),
        )

    xtr, ytr = _prep(train_idx)
    xva, yva = _prep(val_idx)

    torch.manual_seed(seed)
    model = TorchLikelihoodEmulator(theta.shape[1], hidden_units=hidden_units)
    model.set_standardizers(theta_mean, theta_std, y_mean, y_std)

    use_lightning = bool(cfg.get("use_lightning", False))
    if use_lightning:
        try:  # optional; not a hard dependency
            import lightning  # noqa: F401
        except Exception:
            use_lightning = False
            if logger is not None:
                logger.info("pytorch-lightning not available; using pure-torch loop.")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    stalls = 0
    n = len(xtr)
    for epoch in range(num_epochs):
        model.train()
        order = torch.randperm(n)
        for start in range(0, n, batch_size):
            sel = order[start:start + batch_size]
            opt.zero_grad()
            pred = model(xtr[sel])
            loss = loss_fn(pred, ytr[sel])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(xva), yva))
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stalls = 0
        else:
            stalls += 1
            if stalls >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # Validation diagnostics in log-likelihood units.
    with torch.no_grad():
        pred_ll = model.predict(theta[val_idx])
    true_ll = torch.as_tensor(y[val_idx], dtype=torch.float32)
    dev = (pred_ll - true_ll).cpu().numpy()

    path_model = None
    if path_results is not None:
        try:
            os.makedirs(path_results, exist_ok=True)
            path_model = os.path.join(path_results, "likelihood_emulator.pt")
            torch.save(model.state_dict(), path_model)
        except Exception:
            path_model = None

    return {
        "emulator": model,
        "path_model": path_model,
        "val_mse": float(best_val),
        "mean_dev": float(np.mean(dev)),
        "std_dev": float(np.std(dev)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "train_cstat_min": cstat_min,
        "train_cstat_median": cstat_med,
        "train_cstat_max": cstat_max,
    }
