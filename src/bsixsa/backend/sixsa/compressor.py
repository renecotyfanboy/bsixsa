r"""Spectrum compressor: an autoencoder trained with the C-statistic.

The SIXSA density estimators condition on a short code of the spectrum rather
than on its thousands of channels. [`Compressor`][bsixsa.backend.sixsa.compressor.Compressor]
learns that code with a supervised signal of its own (one target per channel
per spectrum) instead of relying on the single log-density a flow provides.

Design, ported from the SIXSA v2 compressor:

- **Loss.** The data are Poisson counts and most channels of a real spectrum
  hold few of them, so the reconstruction is scored with the Cash statistic
  ([`cstat_loss`][bsixsa.backend.sixsa.compressor.cstat_loss]), not a squared
  error. Its floor is not zero but $E[C]$
  ([`cstat_moments`][bsixsa.backend.sixsa.compressor.cstat_moments]).
- **Network.** Counts are mapped by ``log1p`` and standardised per channel
  (constants fitted on the training split); the encoder is
  ``Linear -> norm -> GELU [-> Dropout]`` blocks ending in a linear latent,
  and the mirrored decoder ends in a softplus, so the reconstruction is a
  positive expected count.
- **Staleness contract.** Every [`fit`][bsixsa.backend.sixsa.compressor.Compressor.fit]
  moves the codes, and a flow trained on one generation of codes and
  evaluated on another returns numbers that are not about the observation.
  Each fit therefore increments ``generation``; anything that stores a code
  must store its generation too
  ([`encode_with_generation`][bsixsa.backend.sixsa.compressor.Compressor.encode_with_generation])
  and check it before use
  ([`check_generation`][bsixsa.backend.sixsa.compressor.Compressor.check_generation]).

Changes from v2: ``batch_size="auto"`` no longer crashes, seeding is local
(the global torch and numpy streams are never touched), encoding is batched,
the device defaults to the CPU, and the thread count goes through
[`torch_threads`][bsixsa._torch_utils.torch_threads]. Numpy arrays go in and
come out.
"""

from __future__ import annotations

import contextlib
import copy
import math
import operator
from collections.abc import Callable, Iterator
from time import perf_counter
from typing import Any

import numpy as np
import torch

from ..._cube import child_seed, make_seed_sequence
from ..._torch_utils import torch_threads

__all__ = [
    "Compressor",
    "cstat_loss",
    "cstat_moments",
    "default_hidden_dims",
    "default_latent_dim",
]

_NORMS = ("batch", "layer", None)
_MIN_IMPROVEMENT = 1e-4


def _no_log(message: str) -> None:
    """Default logger: discard the message."""


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def cstat_moments(mu: Any, k_sigma: float = 12.0) -> tuple[float, float]:
    r"""Expectation and variance of the Cash statistic at the true model.

    For expected counts $\mu_i$ and Poisson data $k_i$, each channel
    contributes $C_i = 2(\mu_i - k_i + k_i \ln(k_i / \mu_i))$; $E[C_i]$ and
    $\mathrm{Var}[C_i]$ are obtained by summing the Poisson pmf over
    $k \in [\mu - k_\sigma \sqrt{\mu} - 20, \mu + k_\sigma \sqrt{\mu} + 30]$
    (Kaastra 2017, A&A 605, A51). Channels are independent, so both moments
    add. $E[C_i] \to 1$ and $\mathrm{Var}[C_i] \to 2$ only for
    $\mu_i \gtrsim 10$; at $\mu_i = 0.1$ a channel contributes about 0.47, so
    "one per channel" is the wrong reference for a faint spectrum. Channels
    with $\mu_i \le 0$ contribute zero.

    Parameters:
        mu:
            Expected counts, any shape (flattened).
        k_sigma:
            Half-width of the summed count range, in Poisson sigmas.

    Returns:
        ``(E[C], Var[C])`` summed over channels.
    """
    from scipy.special import gammaln

    mu = np.clip(np.asarray(mu, dtype=np.float64).reshape(-1), 0.0, None)
    expectation = np.zeros_like(mu)
    variance = np.zeros_like(mu)
    for i in np.flatnonzero(mu > 0):
        m = float(mu[i])
        lo = max(0, int(np.floor(m - k_sigma * np.sqrt(m) - 20)))
        hi = int(np.ceil(m + k_sigma * np.sqrt(m) + 30))
        k = np.arange(lo, hi + 1, dtype=np.float64)
        log_p = k * np.log(m) - m - gammaln(k + 1.0)
        p = np.exp(log_p - log_p.max())
        p /= p.sum()
        term = np.where(k > 0, k * np.log(np.where(k > 0, k, 1.0) / m), 0.0)
        c = 2.0 * (m - k + term)
        e = float(np.sum(p * c))
        expectation[i] = e
        variance[i] = max(float(np.sum(p * c * c)) - e * e, 0.0)
    return float(expectation.sum()), float(variance.sum())


def cstat_loss(model_counts: torch.Tensor, data_counts: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r"""Cash statistic of a reconstruction, summed over channels, averaged over spectra.

    $$
    L = \frac{1}{n} \sum_{s=1}^{n} 2 \sum_i
        \left(\mu_{si} - k_{si} + k_{si} \ln \frac{k_{si}}{\mu_{si}}\right),
    $$

    with $\mu$ clamped below at *eps* and $k \ln k := 0$ at $k = 0$. The
    reconstruction plays the model and the input plays the data, which is
    why the decoder output must be positive.

    Parameters:
        model_counts:
            Reconstructed expected counts, tensor of shape ``(n, n_bins)``.
        data_counts:
            Observed counts, tensor of the same shape.
        eps:
            Lower clamp of the model counts (and of the counts inside the log).

    Returns:
        A scalar tensor.
    """
    mu = model_counts.clamp_min(eps)
    k = data_counts
    # The zero branch keeps k = 0 channels finite in value and in gradient.
    term = torch.where(k > 0, k * torch.log(k.clamp_min(eps) / mu), torch.zeros_like(k))
    return 2.0 * (mu - k + term).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Defaults that depend on the problem
# ---------------------------------------------------------------------------


def default_latent_dim(d: int, lo: int = 8, hi: int = 32, round_to: int = 4) -> int:
    r"""Latent size $\operatorname{clip}(2d, lo, hi)$ rounded to a multiple of *round_to*.

    Whitney: a smooth $d$-dimensional manifold embeds in $\mathbb{R}^{2d}$,
    so $2d$ components can carry everything the spectra vary by; beyond that,
    each extra component is noise the flow must learn to ignore (v2 measured
    the best inference at 32 for $d = 14$, and a worse one at 64).

    Parameters:
        d:
            Number of free parameters.
        lo:
            Lower clip.
        hi:
            Upper clip.
        round_to:
            Rounding step (Python's round-half-to-even applies).

    Returns:
        The latent size, at least *round_to*.
    """
    value = int(round(min(max(2 * int(d), int(lo)), int(hi)) / round_to) * round_to)
    return max(int(round_to), value)


def default_hidden_dims(n_bins: int, latent_dim: int, n_layers: int = 2, shrink: float = 4.0) -> list[int]:
    """Hidden widths tapering geometrically from the spectrum to the code.

    The first width is ``max(n_bins / shrink, 4 * latent_dim)`` and each next
    one is the previous divided by *shrink*, with a floor of
    ``2 * latent_dim``. The first layer is the expensive one (its input is
    *n_bins* wide), so depth is kept small.

    Parameters:
        n_bins:
            Number of channels.
        latent_dim:
            Latent size.
        n_layers:
            Number of hidden layers.
        shrink:
            Geometric factor between consecutive widths.

    Returns:
        A list of *n_layers* integer widths.
    """
    n_bins, latent_dim = int(n_bins), int(latent_dim)
    dims = []
    width = max(int(n_bins / shrink), latent_dim * 4)
    for _ in range(int(n_layers)):
        dims.append(int(max(width, latent_dim * 2)))
        width = int(width / shrink)
    return dims


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class _AutoEncoder(torch.nn.Module):
    """Counts -> code -> counts, with a positive output.

    The input is standardised in log space: counts span orders of magnitude
    across channels, and a linear standardisation would let the brightest
    channels dominate every gradient.

    Parameters:
        n_bins:
            Number of channels.
        latent_dim:
            Latent size.
        hidden_dims:
            Encoder hidden widths (the decoder mirrors them).
        norm:
            ``"batch"``, ``"layer"`` or ``None``.
        dropout_rate:
            Dropout probability after each hidden activation (0 disables it).
    """

    def __init__(
        self,
        n_bins: int,
        latent_dim: int,
        hidden_dims: list[int],
        norm: str | None = "batch",
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = norm
        self.dropout_rate = float(dropout_rate)
        self.n_bins = int(n_bins)
        self.latent_dim = int(latent_dim)
        self.register_buffer("mu", torch.zeros(1, self.n_bins))
        self.register_buffer("sd", torch.ones(1, self.n_bins))

        def block(n_in: int, n_out: int) -> list[torch.nn.Module]:
            layers: list[torch.nn.Module] = [torch.nn.Linear(n_in, n_out)]
            # Normalising each unit over the batch is what keeps a network of
            # millions of weights from specialising on individual spectra.
            if self.norm == "batch":
                layers.append(torch.nn.BatchNorm1d(n_out))
            elif self.norm == "layer":
                layers.append(torch.nn.LayerNorm(n_out))
            layers.append(torch.nn.GELU())
            if self.dropout_rate:
                layers.append(torch.nn.Dropout(self.dropout_rate))
            return layers

        encoder: list[torch.nn.Module] = []
        previous = self.n_bins
        for width in hidden_dims:
            encoder += block(previous, int(width))
            previous = int(width)
        encoder.append(torch.nn.Linear(previous, self.latent_dim))
        self.encoder = torch.nn.Sequential(*encoder)

        decoder: list[torch.nn.Module] = []
        previous = self.latent_dim
        for width in reversed(list(hidden_dims)):
            decoder += block(previous, int(width))
            previous = int(width)
        decoder.append(torch.nn.Linear(previous, self.n_bins))
        self.decoder = torch.nn.Sequential(*decoder)

    def fit_scaler(self, x: torch.Tensor) -> None:
        """Fit the per-channel log-space standardisation on *x* (sd floored at 1e-3)."""
        lx = torch.log1p(x.clamp_min(0.0))
        self.mu.copy_(lx.mean(0, keepdim=True))
        self.sd.copy_(lx.std(0, keepdim=True).clamp_min(1e-3))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map counts ``(n, n_bins)`` to codes ``(n, latent_dim)``."""
        lx = (torch.log1p(x.clamp_min(0.0)) - self.mu) / self.sd
        return self.encoder(lx)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map codes to positive expected counts (softplus output)."""
        return torch.nn.functional.softplus(self.decoder(z))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct expected counts from counts."""
        return self.decode(self.encode(x))


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------


class Compressor:
    r"""Autoencoder compressing spectra to short codes, with a generation counter.

    Parameters:
        n_bins:
            Number of channels of the spectra.
        d:
            Number of free parameters; used only to derive *latent_dim* when
            it is not given.
        latent_dim:
            Code size; defaults to [`default_latent_dim`][bsixsa.backend.sixsa.compressor.default_latent_dim] of *d*.
        hidden_dims:
            Encoder hidden widths; defaults to
            [`default_hidden_dims`][bsixsa.backend.sixsa.compressor.default_hidden_dims]
            with *n_layers* layers.
        n_layers:
            Number of hidden layers when *hidden_dims* is not given.
        norm:
            Normalisation after each hidden ``Linear``: ``"batch"``,
            ``"layer"`` or ``None``.
        dropout_rate:
            Dropout probability after each hidden activation, in ``[0, 1)``.
        epochs:
            Maximum epochs per fit.
        batch_size:
            Mini-batch size, or ``"auto"`` (``None`` is accepted as an alias)
            for ``max(16, min(512, n_train // 20))``, recomputed at every fit.
            A network is trained by gradient steps: a fixed 512 on a couple of
            thousand spectra gives three steps an epoch.
        lr:
            Adam learning rate, fresh at every fit.
        patience:
            Epochs without a validation improvement of at least 1e-4 before
            stopping; 0 disables early stopping.
        val_fraction:
            Fraction of the spectra held out for validation, in ``(0, 1)``.
        warm_start:
            Refit from the current weights and scaler when a model exists.
            Otherwise every fit starts from scratch.
        lr_schedule:
            Multiply the learning rate by *lr_factor* when the validation loss
            plateaus for ``max(3, patience // 3)`` epochs.
        lr_factor:
            Plateau factor of the schedule.
        device:
            Torch device of the network; ``None`` means the CPU.
        torch_threads:
            Intra-op torch threads during fits and encodings; ``None`` leaves
            the torch setting alone.
        seed:
            Default seed of the fits (see [`fit`][bsixsa.backend.sixsa.compressor.Compressor.fit]);
            ``None`` draws fresh entropy at every fit.
        log:
            Callable receiving progress messages; ``None`` discards them. It
            is shared, not copied, by ``copy.deepcopy`` and dropped by pickling.

    Attributes:
        generation:
            Number of completed fits; 0 before the first. Codes are valid for
            the generation that produced them only.
        history:
            One record per fit (see [`fit`][bsixsa.backend.sixsa.compressor.Compressor.fit]).
        curves:
            Per fit, the per-epoch ``{"epoch", "train", "val", "lr"}`` lists.
        model:
            The underlying torch module, ``None`` before the first fit.

    Raises:
        ValueError: If neither *latent_dim* nor *d* is given, or an argument
            is out of range.
    """

    def __init__(
        self,
        n_bins: int,
        d: int | None = None,
        *,
        latent_dim: int | None = None,
        hidden_dims: list[int] | None = None,
        n_layers: int = 3,
        norm: str | None = "batch",
        dropout_rate: float = 0.0,
        epochs: int = 1000,
        batch_size: int | str | None = 512,
        lr: float = 1e-3,
        patience: int = 20,
        val_fraction: float = 0.2,
        warm_start: bool = True,
        lr_schedule: bool = True,
        lr_factor: float = 0.3,
        device: str | torch.device | None = "cpu",
        torch_threads: int | None = None,
        seed: int | np.random.SeedSequence | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.n_bins = _check_int(n_bins, "n_bins")
        if latent_dim is None:
            if d is None:
                raise ValueError("Compressor needs latent_dim, or d to derive it as clip(2d, 8, 32).")
            latent_dim = default_latent_dim(_check_int(d, "d"))
        self._latent_dim = _check_int(latent_dim, "latent_dim")
        self.d = None if d is None else int(d)
        self.n_layers = _check_int(n_layers, "n_layers", minimum=0)
        if hidden_dims is None:
            hidden_dims = default_hidden_dims(self.n_bins, self._latent_dim, n_layers=self.n_layers)
        self.hidden_dims = [_check_int(width, "hidden_dims entry") for width in hidden_dims]
        if isinstance(norm, str) and norm.lower() == "none":
            norm = None
        if norm not in _NORMS:
            raise ValueError(f"norm must be 'batch', 'layer' or None, got {norm!r}.")
        self.norm = norm
        self.dropout_rate = float(dropout_rate)
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must lie in [0, 1), got {self.dropout_rate}.")
        self.epochs = _check_int(epochs, "epochs")
        if batch_size is None or batch_size == "auto":
            self.batch_size: int | str = "auto"
        elif isinstance(batch_size, str):
            raise ValueError(f"batch_size must be a positive integer or 'auto', got {batch_size!r}.")
        else:
            self.batch_size = _check_int(batch_size, "batch_size")
        self.lr = float(lr)
        if not (math.isfinite(self.lr) and self.lr > 0.0):
            raise ValueError(f"lr must be a positive finite number, got {lr}.")
        self.patience = _check_int(patience, "patience", minimum=0)
        self.val_fraction = float(val_fraction)
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must lie in (0, 1), got {val_fraction}.")
        self.warm_start = bool(warm_start)
        self.lr_schedule = bool(lr_schedule)
        self.lr_factor = float(lr_factor)
        if not 0.0 < self.lr_factor < 1.0:
            raise ValueError(f"lr_factor must lie in (0, 1), got {lr_factor}.")
        self.device = torch.device("cpu" if device is None else device)
        self.torch_threads = None if torch_threads is None else _check_int(torch_threads, "torch_threads")
        self.seed = seed
        self._log = log

        self.model: _AutoEncoder | None = None
        self.generation = 0
        self.history: list[dict] = []
        self.curves: list[dict] = []

    # -- identity ---------------------------------------------------------

    @property
    def latent_dim(self) -> int:
        """Size of the codes."""
        return self._latent_dim

    def __repr__(self) -> str:
        return (
            f"Compressor(n_bins={self.n_bins}, latent_dim={self._latent_dim}, "
            f"hidden_dims={self.hidden_dims}, generation={self.generation})"
        )

    def __deepcopy__(self, memo: dict) -> Compressor:
        """Copy the network, scaler and history; share the logger.

        A logger may hold an open file, which cannot be copied, and two
        copies writing to two duplicates of one log would be wrong anyway.
        A refit of the copy leaves the original untouched.
        """
        cls = type(self)
        new = cls.__new__(cls)
        memo[id(self)] = new
        for name, value in self.__dict__.items():
            setattr(new, name, value if name == "_log" else copy.deepcopy(value, memo))
        return new

    def __getstate__(self) -> dict:
        """Pickle everything but the logger, which may not be picklable."""
        state = dict(self.__dict__)
        state["_log"] = None
        return state

    # -- training ---------------------------------------------------------

    def fit(
        self,
        counts: Any,
        *,
        cstat: Any = None,
        max_cstat: float | None = None,
        min_kept: int = 200,
        epochs: int | None = None,
        seed: int | np.random.SeedSequence | None = None,
    ) -> dict:
        r"""Train the autoencoder, or refresh it on a new set.

        **Every call increments** ``generation``: codes taken before it are
        stale.

        With *cstat* and *max_cstat*, only the spectra with
        ``cstat <= max_cstat`` are used, provided at least *min_kept* survive
        (otherwise all are kept and a message is logged): the compressor
        should learn the spectra the inference will see. A random
        ``val_fraction`` of the spectra is held out; training minimises
        [`cstat_loss`][bsixsa.backend.sixsa.compressor.cstat_loss] with Adam,
        stops early on the validation loss and restores the best weights.
        With ``warm_start`` and an existing model, the weights and the scaler
        are kept (a new input scale would hand the trained weights data they
        have never seen); otherwise a new network is built and its scaler is
        fitted on the training split.

        Seeding is local: the split, the batch order, the initialisation and
        dropout are driven by ``child_seed(seed, generation)`` inside
        ``torch.random.fork_rng``, with *seed* falling back to ``self.seed``
        and then to fresh entropy. The global torch and numpy streams are left
        untouched, and two fits of the same data from the same state and seed
        give the same weights.

        Parameters:
            counts:
                Spectra, shape ``(n, n_bins)``.
            cstat:
                Their exact C-stat, shape ``(n,)``; used for the selection
                only.
            max_cstat:
                Selection threshold, typically the restriction's.
            min_kept:
                Minimum number of selected spectra for the selection to apply.
            epochs:
                Maximum epochs of this fit; defaults to ``self.epochs``.
            seed:
                Seed of this fit; defaults to ``self.seed``.

        Returns:
            A copy of the fit record appended to ``history``, with keys
            ``generation``, ``epochs`` (epochs run), ``val_cstat`` (best mean
            validation C-stat), ``n_train``, ``n_val``, ``seconds``, ``warm``,
            ``batch_size`` and ``final_lr``.

        Raises:
            ValueError: If *counts* does not have shape ``(n, n_bins)`` or
                holds non-finite values, *cstat* has the wrong length, or the
                split leaves no trainable set (fewer than 2 training spectra
                with batch normalisation, fewer than 1 otherwise).
        """
        log = self._log or _no_log
        start = perf_counter()
        # Writeable and contiguous, so torch can share the buffer without a warning.
        x = np.require(counts, dtype=np.float32, requirements=["C", "W"])
        if x.ndim != 2 or x.shape[1] != self.n_bins:
            raise ValueError(f"counts must have shape (n, {self.n_bins}), got {x.shape}.")
        if not np.all(np.isfinite(x)):
            raise ValueError("counts must be finite.")
        if cstat is not None:
            cs = np.asarray(cstat, dtype=np.float64).reshape(-1)
            if cs.shape[0] != x.shape[0]:
                raise ValueError(f"cstat has {cs.shape[0]} values for {x.shape[0]} spectra.")
            if max_cstat is not None:
                keep = cs <= float(max_cstat)
                n_keep = int(keep.sum())
                if n_keep >= int(min_kept):
                    if n_keep < x.shape[0]:
                        log(
                            f"[compressor] training on the {n_keep:,} of {x.shape[0]:,} spectra "
                            f"with cstat <= {float(max_cstat):,.1f}."
                        )
                    x = x[keep]
                else:
                    log(
                        f"[compressor] only {n_keep:,} spectra have cstat <= {float(max_cstat):,.1f} "
                        f"(fewer than {int(min_kept):,}); training on all {x.shape[0]:,}."
                    )

        n = x.shape[0]
        n_val = max(1, int(round(self.val_fraction * n)))
        n_train = n - n_val
        min_train = 2 if self.norm == "batch" else 1
        if n_train < min_train:
            raise ValueError(
                f"Compressor.fit needs at least {min_train} training spectra after holding out "
                f"{n_val} for validation, got {n} spectra in total."
            )
        batch_size = self._resolve_batch_size(n_train)
        n_epochs = self.epochs if epochs is None else _check_int(epochs, "epochs")
        seed_value = child_seed(make_seed_sequence(self.seed if seed is None else seed), self.generation)
        fresh = self.model is None or not self.warm_start

        with torch_threads(self.torch_threads), self._forked_rng(seed_value):
            generator = torch.Generator().manual_seed(seed_value)
            perm = torch.randperm(n, generator=generator)
            val_idx, train_idx = perm[:n_val], perm[n_val:]
            data = torch.as_tensor(x).to(self.device)
            x_train, x_val = data[train_idx.to(self.device)], data[val_idx.to(self.device)]

            if fresh:
                model = _AutoEncoder(
                    self.n_bins,
                    self._latent_dim,
                    self.hidden_dims,
                    norm=self.norm,
                    dropout_rate=self.dropout_rate,
                ).to(self.device)
                with torch.no_grad():
                    model.fit_scaler(x_train)
            else:
                # Train a copy: a fit that raises must not leave the current
                # generation's codes pointing at half-updated weights.
                model = copy.deepcopy(self.model)

            steps_per_epoch = max(1, n_train // batch_size)
            if steps_per_epoch * n_epochs < 3000:
                log(
                    f"[compressor] at most {steps_per_epoch * n_epochs:,} gradient steps "
                    f"({steps_per_epoch} per epoch x {n_epochs} epochs); below a few thousand a "
                    "reconstruction is often still improving when the fit stops."
                )
            best, epochs_run, final_lr, curve = self._train(
                model, x_train, x_val, batch_size=batch_size, n_epochs=n_epochs, generator=generator
            )
            model.eval()

        self.model = model
        self.generation += 1
        record = {
            "generation": self.generation,
            "epochs": epochs_run,
            "val_cstat": best,
            "n_train": int(n_train),
            "n_val": int(n_val),
            "seconds": perf_counter() - start,
            "warm": not fresh,
            "batch_size": int(batch_size),
            "final_lr": final_lr,
        }
        self.history.append(record)
        self.curves.append(curve)
        log(
            f"[compressor] generation {self.generation} ({'warm' if not fresh else 'from scratch'}): "
            f"{epochs_run}/{n_epochs} epochs on {n_train:,} spectra, batch {batch_size}, "
            f"validation C-stat {best:,.1f} over {self.n_bins:,} channels, {record['seconds']:.1f} s."
        )
        return dict(record)

    def _train(
        self,
        model: _AutoEncoder,
        x_train: torch.Tensor,
        x_val: torch.Tensor,
        *,
        batch_size: int,
        n_epochs: int,
        generator: torch.Generator,
    ) -> tuple[float, int, float, dict]:
        """Run the epochs of one fit and restore the best weights.

        Parameters:
            model:
                Network to train in place.
            x_train:
                Training spectra on the model device.
            x_val:
                Validation spectra on the model device.
            batch_size:
                Mini-batch size.
            n_epochs:
                Maximum epochs.
            generator:
                CPU generator of the batch order.

        Returns:
            ``(best validation loss, epochs run, final learning rate, curve)``.
        """
        optimiser = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = None
        if self.lr_schedule:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimiser, factor=self.lr_factor, patience=max(3, self.patience // 3)
            )
        curve: dict[str, list[float]] = {"epoch": [], "train": [], "val": [], "lr": []}
        best, best_state, since, epochs_run, last_val = math.inf, None, 0, 0, math.inf
        n_train = x_train.shape[0]
        for epoch in range(n_epochs):
            model.train()
            order = torch.randperm(n_train, generator=generator).to(x_train.device)
            total, n_batches = 0.0, 0
            for lo in range(0, n_train, batch_size):
                batch = x_train[order[lo : lo + batch_size]]
                # Batch normalisation has no variance for a single spectrum.
                if batch.shape[0] < 2 and self.norm == "batch":
                    continue
                optimiser.zero_grad()
                loss = cstat_loss(model(batch), batch)
                loss.backward()
                optimiser.step()
                total += float(loss.detach())
                n_batches += 1
            model.eval()
            last_val = _mean_loss(model, x_val)
            epochs_run = epoch + 1
            lr_now = float(optimiser.param_groups[0]["lr"])
            curve["epoch"].append(epochs_run)
            curve["train"].append(total / max(n_batches, 1))
            curve["val"].append(last_val)
            curve["lr"].append(lr_now)
            if scheduler is not None:
                scheduler.step(last_val)
            if last_val < best - _MIN_IMPROVEMENT:
                best, since = last_val, 0
                best_state = {name: t.detach().clone() for name, t in model.state_dict().items()}
            else:
                since += 1
            if self.patience and since >= self.patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        else:
            best = last_val
        final_lr = float(optimiser.param_groups[0]["lr"])
        return float(best), epochs_run, final_lr, curve

    def _resolve_batch_size(self, n_train: int) -> int:
        """Mini-batch size for *n_train* training spectra (``"auto"`` resolved)."""
        if self.batch_size == "auto":
            return int(max(16, min(512, n_train // 20)))
        return int(self.batch_size)

    @contextlib.contextmanager
    def _forked_rng(self, seed: int) -> Iterator[None]:
        """Fork the torch streams used by a fit and seed them with *seed*.

        ``torch.manual_seed`` would also reseed every accelerator stream
        outside the fork, so only the CPU default generator (and the model's
        own accelerator, forked separately) is seeded.
        """
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.random.fork_rng(devices=[]))
            torch.default_generator.manual_seed(int(seed))
            device = self.device
            if device.type == "cuda":
                stack.enter_context(
                    torch.random.fork_rng(devices=[device.index or 0], device_type="cuda")
                )
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(int(seed))
            elif device.type == "mps":
                stack.enter_context(torch.random.fork_rng(devices=[0], device_type="mps"))
                torch.mps.manual_seed(int(seed))
            yield

    # -- use --------------------------------------------------------------

    def encode(self, counts: Any, *, batch: int = 8192) -> np.ndarray:
        """Encode spectra into codes of the current generation.

        Deterministic: the network is put in evaluation mode and run without
        gradients, in batches of *batch* spectra.

        Parameters:
            counts:
                Spectra, shape ``(n, n_bins)`` or ``(n_bins,)``.
            batch:
                Spectra per forward pass.

        Returns:
            float32 array of shape ``(n, latent_dim)`` (``(1, latent_dim)``
            for a single spectrum).

        Raises:
            RuntimeError: If the compressor has not been fitted.
            ValueError: If *counts* does not have ``n_bins`` columns.
        """
        return self._forward(counts, batch=batch, what="encode")

    def encode_with_generation(self, counts: Any) -> tuple[np.ndarray, int]:
        """Encode spectra and return the generation the codes belong to.

        Use it wherever a code is stored, so the pair cannot be separated.

        Parameters:
            counts:
                Spectra, shape ``(n, n_bins)`` or ``(n_bins,)``.

        Returns:
            ``(codes, generation)``.

        Raises:
            RuntimeError: If the compressor has not been fitted.
            ValueError: If *counts* does not have ``n_bins`` columns.
        """
        return self.encode(counts), self.generation

    def check_generation(self, generation: int, what: str = "") -> None:
        """Refuse a code encoded by another generation of the compressor.

        Parameters:
            generation:
                Generation recorded with the code.
            what:
                Description of the code, for the message.

        Raises:
            RuntimeError: If *generation* differs from ``self.generation``.
        """
        if int(generation) != self.generation:
            label = what or "a code"
            raise RuntimeError(
                f"{label} was encoded at compressor generation {int(generation)} but the compressor is "
                f"now at generation {self.generation}; re-encode it before use."
            )

    def reconstruct(self, counts: Any, *, batch: int = 8192) -> np.ndarray:
        """Reconstruct expected counts through the code.

        Parameters:
            counts:
                Spectra, shape ``(n, n_bins)`` or ``(n_bins,)``.
            batch:
                Spectra per forward pass.

        Returns:
            float32 array of shape ``(n, n_bins)`` of positive expected counts.

        Raises:
            RuntimeError: If the compressor has not been fitted.
            ValueError: If *counts* does not have ``n_bins`` columns.
        """
        return self._forward(counts, batch=batch, what="reconstruct")

    def _forward(self, counts: Any, *, batch: int, what: str) -> np.ndarray:
        """Shared batched, gradient-free pass of `encode` and `reconstruct`."""
        if self.model is None:
            raise RuntimeError(f"cannot {what}: the compressor has not been fitted.")
        x = np.require(counts, dtype=np.float32, requirements=["C", "W"])
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != self.n_bins:
            raise ValueError(f"counts must have shape (n, {self.n_bins}) or ({self.n_bins},), got {x.shape}.")
        batch = _check_int(batch, "batch")
        width = self._latent_dim if what == "encode" else self.n_bins
        out = np.empty((x.shape[0], width), dtype=np.float32)
        model = self.model
        # Evaluation mode here rather than inherited: with dropout, a model
        # left in training mode would return random codes.
        model.eval()
        with torch_threads(self.torch_threads), torch.no_grad():
            for lo in range(0, x.shape[0], batch):
                chunk = torch.as_tensor(x[lo : lo + batch]).to(self.device)
                result = model.encode(chunk) if what == "encode" else model(chunk)
                out[lo : lo + chunk.shape[0]] = result.detach().cpu().numpy()
        return out


def _mean_loss(model: _AutoEncoder, x: torch.Tensor, chunk: int = 8192) -> float:
    """Mean per-spectrum C-stat of *model* on *x*, in chunks, without gradients."""
    total = 0.0
    with torch.no_grad():
        for lo in range(0, x.shape[0], chunk):
            part = x[lo : lo + chunk]
            total += float(cstat_loss(model(part), part)) * part.shape[0]
    return total / max(x.shape[0], 1)


def _check_int(value: Any, name: str, *, minimum: int = 1) -> int:
    """Return *value* as an ``int`` not smaller than *minimum*.

    Parameters:
        value:
            Integer-like value.
        name:
            Argument name used in error messages.
        minimum:
            Smallest accepted value.

    Returns:
        The value as a Python ``int``.

    Raises:
        ValueError: If *value* is not an integer or is smaller than *minimum*.
    """
    try:
        out = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from None
    if out < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {out}.")
    return out
