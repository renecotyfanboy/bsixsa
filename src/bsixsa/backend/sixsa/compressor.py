"""Spectrum compressor: an autoencoder that turns thousands of channels into a short code.

The encoder reads standardised `log(1 + counts)`; the decoder returns expected
counts. Training minimises the Cash statistic of the reconstruction, the same
Poisson likelihood as the fit, so the code keeps what matters for the C-stat.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from time import perf_counter
from typing import Any

import numpy as np
import torch

__all__ = ["Compressor", "cstat_loss", "default_hidden_dims", "default_latent_dim"]

# Smallest validation C-stat drop that resets early stopping.
_MIN_IMPROVEMENT = 1e-4
# Apply the max_cstat cut only if it leaves this many spectra.
_MIN_SPECTRA_AFTER_CUT = 200
# Each hidden layer is this many times narrower than the previous one.
_HIDDEN_SHRINK = 4.0


def cstat_loss(model_counts: torch.Tensor, data_counts: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Cash reconstruction loss: sum over channels, mean over spectra; 0 log 0 = 0."""
    mu = model_counts.clamp_min(eps)
    k = data_counts
    term = torch.where(k > 0, k * torch.log(k.clamp_min(eps) / mu), torch.zeros_like(k))
    return 2.0 * (mu - k + term).sum(dim=1).mean()


def default_latent_dim(d: int) -> int:
    """Code size ``clip(2 d, 8, 32)`` rounded to a multiple of 4."""
    return 4 * round(min(max(2 * int(d), 8), 32) / 4)


def default_hidden_dims(n_bins: int, latent_dim: int, n_layers: int = 2) -> list[int]:
    """Hidden widths tapering geometrically from ``n_bins / 4`` towards the code."""
    n_bins, latent_dim = int(n_bins), int(latent_dim)
    dims = []
    width = max(int(n_bins / _HIDDEN_SHRINK), latent_dim * 4)
    for _ in range(int(n_layers)):
        dims.append(int(max(width, latent_dim * 2)))
        width = int(width / _HIDDEN_SHRINK)
    return dims


def _resolve_device(device: str = "cpu") -> torch.device:
    """``"auto"`` picks cuda, then mps, then cpu."""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _AutoEncoder(torch.nn.Module):
    """Counts -> code -> expected counts."""

    def __init__(self, n_bins: int, latent_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        self.register_buffer("log_mean", torch.zeros(1, int(n_bins)))
        self.register_buffer("log_std", torch.ones(1, int(n_bins)))

        def network(widths):
            layers = []
            for n_in, n_out in zip(widths[:-2], widths[1:-1]):
                layers += [torch.nn.Linear(n_in, n_out), torch.nn.BatchNorm1d(n_out), torch.nn.GELU()]
            layers.append(torch.nn.Linear(widths[-2], widths[-1]))
            return torch.nn.Sequential(*layers)

        widths = [int(n_bins), *map(int, hidden_dims), int(latent_dim)]
        self.encoder = network(widths)
        self.decoder = network(widths[::-1])

    def fit_scaler(self, x: torch.Tensor) -> None:
        log_counts = torch.log1p(x.clamp_min(0.0))
        self.log_mean.copy_(log_counts.mean(0, keepdim=True))
        self.log_std.copy_(log_counts.std(0, keepdim=True).clamp_min(1e-3))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder((torch.log1p(x.clamp_min(0.0)) - self.log_mean) / self.log_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.decoder(self.encode(x)))


class Compressor:
    """Spectrum autoencoder, refitted in place; ``generation`` counts the fits.

    ``patience=0`` disables early stopping and ``device="auto"`` picks cuda,
    then mps, then cpu. ``history`` holds one record per fit and ``curves``
    the per-epoch losses.
    """

    def __init__(
        self,
        n_bins: int,
        d: int | None = None,
        *,
        latent_dim: int | None = None,
        hidden_dims: list[int] | None = None,
        n_layers: int = 3,
        epochs: int = 1000,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        patience: int = 20,
        validation_fraction: float = 0.2,
        warm_start: bool = True,
        device: str = "cpu",
        seed: int | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.n_bins = int(n_bins)
        self.latent_dim = int(default_latent_dim(int(d)) if latent_dim is None else latent_dim)
        if hidden_dims is None:
            hidden_dims = default_hidden_dims(self.n_bins, self.latent_dim, n_layers=int(n_layers))
        self.hidden_dims = [int(width) for width in hidden_dims]
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.patience = int(patience)
        self.validation_fraction = float(validation_fraction)
        self.warm_start = bool(warm_start)
        self.device = _resolve_device(device)
        self.seed = None if seed is None else int(seed)
        self._log = log or (lambda message: None)

        self.model: _AutoEncoder | None = None
        self.generation = 0
        self.history: list[dict] = []
        self.curves: list[dict] = []

    def __repr__(self) -> str:
        return (f"Compressor(n_bins={self.n_bins}, latent_dim={self.latent_dim}, "
                f"hidden_dims={self.hidden_dims}, generation={self.generation})")

    def fit(self, counts: Any, *, cstat: Any = None, max_cstat: float | None = None,
            epochs: int | None = None) -> dict:
        """Fit on ``(n, n_bins)`` spectra and return the fit record.

        With *max_cstat*, only the spectra at or below it are used, provided
        at least 200 remain.
        """
        start = perf_counter()
        x = np.require(counts, dtype=np.float32, requirements=["C", "W"])
        if not np.all(np.isfinite(x)):
            raise ValueError("counts must be finite.")
        if cstat is not None and max_cstat is not None:
            keep = np.asarray(cstat, dtype=np.float64).reshape(-1) <= float(max_cstat)
            if int(keep.sum()) >= _MIN_SPECTRA_AFTER_CUT:
                self._log(f"[compressor] training on the {int(keep.sum()):,} of {x.shape[0]:,} spectra "
                          f"with cstat <= {float(max_cstat):,.1f}.")
                x = x[keep]

        n = x.shape[0]
        n_val = max(1, int(round(self.validation_fraction * n)))
        n_train = n - n_val
        if n_train < 2:
            raise ValueError(f"Compressor.fit needs at least 2 training spectra after holding out "
                             f"{n_val} for validation, got {n} spectra in total.")
        if self.seed is not None:
            # Each refit gets its own reproducible split and initialisation.
            torch.manual_seed(self.seed + self.generation)
        perm = torch.randperm(n)
        data = torch.as_tensor(x).to(self.device)
        x_val, x_train = data[perm[:n_val].to(self.device)], data[perm[n_val:].to(self.device)]

        cold_start = self.model is None or not self.warm_start
        if cold_start:
            model = _AutoEncoder(self.n_bins, self.latent_dim, self.hidden_dims).to(self.device)
            with torch.no_grad():
                model.fit_scaler(x_train)
        else:
            model = copy.deepcopy(self.model)  # a failed fit leaves the current model intact
        n_epochs = self.epochs if epochs is None else int(epochs)
        best, epochs_run, curve = self._train(model, x_train, x_val, n_epochs)
        model.eval()

        self.model = model
        self.generation += 1
        record = {"generation": self.generation, "epochs": epochs_run, "val_cstat": best, "n_train": int(n_train),
                  "n_val": int(n_val), "warm": not cold_start, "seconds": perf_counter() - start}
        self.history.append(record)
        self.curves.append(curve)
        self._log(f"[compressor] generation {self.generation} ({'from scratch' if cold_start else 'warm'}, "
                  f"{self.device}): {epochs_run}/{n_epochs} epochs on {n_train:,} spectra, validation C-stat "
                  f"{best:,.1f} over {self.n_bins:,} channels, {record['seconds']:.1f} s.")
        return dict(record)

    def _train(self, model: _AutoEncoder, x_train: torch.Tensor, x_val: torch.Tensor,
               n_epochs: int) -> tuple[float, int, dict]:
        """Run the epochs of one fit and restore the best weights."""
        optimiser = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.3,
                                                               patience=max(3, self.patience // 3))
        curve: dict[str, list[float]] = {"epoch": [], "train": [], "val": [], "lr": []}
        best, best_state, epochs_since_best, epochs_run, last_val = math.inf, None, 0, 0, math.inf
        n_train = x_train.shape[0]
        for epoch in range(n_epochs):
            model.train()
            order = torch.randperm(n_train).to(x_train.device)
            total, n_batches = 0.0, 0
            for lo in range(0, n_train, self.batch_size):
                batch = x_train[order[lo : lo + self.batch_size]]
                if batch.shape[0] < 2:
                    continue  # batch normalisation has no variance for one spectrum
                optimiser.zero_grad()
                loss = cstat_loss(model(batch), batch)
                loss.backward()
                optimiser.step()
                total += float(loss.detach())
                n_batches += 1
            model.eval()
            last_val = _mean_loss(model, x_val)
            epochs_run = epoch + 1
            curve["epoch"].append(epochs_run)
            curve["train"].append(total / max(n_batches, 1))
            curve["val"].append(last_val)
            curve["lr"].append(float(optimiser.param_groups[0]["lr"]))
            scheduler.step(last_val)
            if last_val < best - _MIN_IMPROVEMENT:
                best, epochs_since_best = last_val, 0
                best_state = {name: t.detach().clone() for name, t in model.state_dict().items()}
            else:
                epochs_since_best += 1
            if self.patience and epochs_since_best >= self.patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        else:
            best = last_val
        return float(best), epochs_run, curve

    def encode(self, counts: Any, *, batch: int = 8192) -> np.ndarray:
        """Codes ``(n, latent_dim)`` of the spectra."""
        if self.model is None:
            raise RuntimeError("the compressor has not been fitted.")
        x = np.require(counts, dtype=np.float32, requirements=["C", "W"])
        if x.ndim == 1:
            x = x.reshape(1, -1)
        out = np.empty((x.shape[0], self.latent_dim), dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            for lo in range(0, x.shape[0], int(batch)):
                chunk = torch.as_tensor(x[lo : lo + int(batch)]).to(self.device)
                out[lo : lo + chunk.shape[0]] = self.model.encode(chunk).detach().cpu().numpy()
        return out


def _mean_loss(model: _AutoEncoder, x: torch.Tensor, chunk: int = 8192) -> float:
    """Mean per-spectrum C-stat of *model* on *x*."""
    total = 0.0
    with torch.no_grad():
        for lo in range(0, x.shape[0], chunk):
            part = x[lo : lo + chunk]
            total += float(cstat_loss(model(part), part)) * part.shape[0]
    return total / max(x.shape[0], 1)
