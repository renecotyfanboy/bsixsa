from .abc import Embedding
from abc import abstractmethod, ABC
from typing import Callable
from pathlib import Path
from torch import nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import torch
import torch.optim as optim
import xspec
from .nn_architectures import Autoencoder, VariationalAutoencoder


LossFn = Callable[
    [nn.Module, torch.Tensor], tuple[torch.Tensor, dict[str, torch.Tensor]]
]


def cstat_loss(y_true, y_pred, eps=1e-6):
    y_true = torch.clamp(y_true, min=eps)
    y_pred = torch.clamp(y_pred, min=eps)

    term = torch.where(
        (y_true > 0.0) & (y_pred > 0.0), torch.log(y_true) - torch.log(y_pred), 0.0
    )
    cstat = 2 * torch.sum(y_pred - y_true + y_true * term, dim=1)

    return cstat.mean()


def autoencoder_loss_fn(
    model: nn.Module,
    spectrum: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    outputs = model(spectrum)
    loss = cstat_loss(outputs, spectrum)
    return loss, {
        "loss": loss,
        "reconstruction": loss,
    }


def vae_loss_fn(
    model: nn.Module,
    spectrum: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    outputs, mu, logvar, _ = model(spectrum)
    reconstruction = cstat_loss(outputs, spectrum)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = reconstruction + kl_divergence
    metrics = {
        "loss": total_loss,
        "reconstruction": reconstruction,
        "kl_divergence": kl_divergence,
        "total_loss": total_loss,
    }
    return total_loss, metrics


def _plot_training_history(
    history: dict[str, list[dict[str, float]]],
    metrics_path: Path,
    title: str | None = None,
) -> None:
    if not history["train"]:
        return

    path = Path(metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train"]) + 1)
    train_loss = [
        epoch_metrics.get("loss", float("nan")) for epoch_metrics in history["train"]
    ]
    val_loss = [
        epoch_metrics.get("loss", float("nan")) for epoch_metrics in history["val"]
    ]

    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_loss, label="Train loss", marker="o")
    plt.plot(epochs, val_loss, label="Validation loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title or "Embedding training history")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _epoch_pass(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    loss_fn: LossFn,
    optimizer: optim.Optimizer | None = None,
    clip_grad_norm: float | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Run one dataloader sweep and aggregate metric statistics.

    Parameters:
        model (nn.Module): Model to evaluate or update.
        loader (DataLoader): Iterable that yields batches of spectra.
        device (torch.device): Target device for tensor computations.
        loss_fn (LossFn): Callable returning a scalar loss and metric dict.
        optimizer (optim.Optimizer | None): Optimizer used for gradient
            updates. If ``None``, the pass runs in evaluation mode.
        clip_grad_norm (float | None): Maximum gradient norm for clipping.

    Returns:
        tuple[dict[str, float], dict[str, float]]: Accumulated metric totals
            and corresponding averages across all batches.
    """

    is_training = optimizer is not None
    model.train() if is_training else model.eval()
    metrics_sum: dict[str, float] = {}
    batch_count = 0

    with torch.enable_grad() if is_training else torch.no_grad():
        for (spectrum,) in loader:
            spectrum = spectrum.to(device, non_blocking=True)
            loss, metrics = loss_fn(model, spectrum)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                if clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()

            for name, value in metrics.items():
                metrics_sum[name] = metrics_sum.get(name, 0.0) + float(value.detach())
            batch_count += 1

    batch_count = max(1, batch_count)
    metrics_avg = {name: total / batch_count for name, total in metrics_sum.items()}
    return metrics_sum, metrics_avg


def training_loop(
    model: nn.Module,
    data,
    *,
    loss_fn: LossFn,
    device=torch.get_default_device(),
    patience=20,
    max_epochs=100,
    min_delta=1e-2,
    learning_rate=5e-4,
    weight_decay=1e-5,
    clip_grad_norm=5.0,
    optimizer_cls: type[optim.Optimizer] = optim.Adam,
    optimizer_kwargs: dict | None = None,
    metrics_path: str | Path | None = None,
    **kwargs,
):
    train_dataset = TensorDataset(torch.from_numpy(data.astype(np.float32)))
    train_ds, val_ds = random_split(train_dataset, [0.9, 0.1])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    optimizer_params = dict(optimizer_kwargs or {})
    optimizer_params.setdefault("lr", learning_rate)
    optimizer_params.setdefault("weight_decay", weight_decay)

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=10
    )

    num_epochs = max_epochs
    early_stop_patience = (
        patience  # epochs with no sufficient improvement before stopping
    )
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0
    history: dict[str, list[dict[str, float]]] = {"train": [], "val": []}

    with tqdm() as pbar:
        for epoch in range(num_epochs):
            _, train_metrics_avg = _epoch_pass(
                model,
                train_loader,
                device,
                loss_fn=loss_fn,
                optimizer=optimizer,
                clip_grad_norm=clip_grad_norm,
            )

            val_metrics_sum, val_metrics_avg = _epoch_pass(
                model,
                val_loader,
                device,
                loss_fn=loss_fn,
            )

            history["train"].append(dict(train_metrics_avg))
            history["val"].append(dict(val_metrics_avg))

            if (
                "loss" not in val_metrics_sum
                or "loss" not in train_metrics_avg
                or "loss" not in val_metrics_avg
            ):
                raise KeyError(
                    "Loss function must provide a 'loss' metric for scheduling and logging."
                )

            scheduler.step(val_metrics_sum["loss"])
            lr_current = scheduler.optimizer.param_groups[0]["lr"]

            avg_train_loss = train_metrics_avg["loss"]
            avg_val_loss = val_metrics_avg["loss"]

            improvement = best_val_loss - avg_val_loss

            if improvement > min_delta:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                epochs_no_improve = 0

            else:
                epochs_no_improve += 1

            prefix = kwargs.get("prefix", "")

            pbar.set_description(
                f"{prefix}Epoch {epoch:03d}/{num_epochs} | "
                f"train {avg_train_loss:.2f} | val {avg_val_loss:.2f} | "
                f"best {best_val_loss:.2f} (epoch {best_epoch}) | "
                f"no_improve {epochs_no_improve}/{early_stop_patience} | {lr_current:.2e}"
            )

            # Trigger early stop
            if epochs_no_improve >= early_stop_patience:
                pbar.set_description(
                    f"{prefix}Early stopping at epoch {epoch}. Best was epoch {best_epoch} "
                    f"(val_loss={best_val_loss:.2f})."
                )
                break

            pbar.update(1)

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    model.eval()
    model.training_history = history

    if metrics_path:
        _plot_training_history(
            history, Path(metrics_path), title=kwargs.get("plot_title")
        )

    return model


class TrainableEmbedding(Embedding, ABC):
    model: nn.Module
    trainable = True

    @abstractmethod
    def train(self, data, *, loss_fn: Callable, metrics_path, **kwargs):
        pass


class TorchModuleEmbedding(TrainableEmbedding, ABC):
    def __init__(self, model, **kwargs):
        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = model.to(self.device)

    @property
    def input_dim(self):
        data = np.asarray(xspec.AllData(1).values)
        return len(data)

    @property
    def embedding_dim(self):
        data = np.asarray(xspec.AllData(1).values)
        return int(
            self.model(
                torch.from_numpy(data.astype(np.float32)).to(self.device).unsqueeze(0)
            ).shape[1]
        )

    def names(self) -> list[str]:
        return [f"latent {i}" for i in range(1, self.embedding_dim + 1)]

    def train(self, data, *, loss_fn: LossFn, metrics_path, **kwargs):
        self.model = training_loop(
            self.model,
            data,
            loss_fn=loss_fn,
            device=self.device,
            metrics_path=metrics_path,
            **kwargs,
        )


class AutoencoderEmbedding(TorchModuleEmbedding):
    def __init__(self, latent_dim=32, retrain_from_scratch: bool = False, **kwargs):
        self.latent_dim = latent_dim
        self.retrain_from_scratch = retrain_from_scratch
        model = self.build_model()
        super().__init__(model, **kwargs)

    def build_model(self):
        return Autoencoder(self.input_dim, self.latent_dim, [self.input_dim // 2])

    def __call__(self, spectra):
        if not isinstance(spectra, torch.Tensor):
            spectra = torch.from_numpy(spectra.astype(np.float32)).to(self.device)

        return self.model.encoder(spectra)

    def train(self, data, **kwargs):
        if self.retrain_from_scratch:
            self.model = self.build_model().to(self.device)

        self.model.scaler.fit(torch.from_numpy(data.astype(np.float32)).to(self.device))
        metrics_path = kwargs.pop("metrics_path", None)

        super().train(
            data,
            loss_fn=autoencoder_loss_fn,
            max_epochs=1_000,
            prefix="Autoencoder | ",
            metrics_path=metrics_path,
            **kwargs,
        )


class VAEEmbedding(TorchModuleEmbedding):
    def __init__(self, latent_dim=32, **kwargs):
        self.latent_dim = latent_dim
        model = self.build_model()

        super().__init__(model, **kwargs)

    def build_model(self):
        return VariationalAutoencoder(
            self.input_dim,
            self.latent_dim,
        )

    def __call__(self, spectra):
        if not isinstance(spectra, torch.Tensor):
            spectra = torch.from_numpy(spectra.astype(np.float32)).to(self.device)

        return torch.hstack(self.model(spectra)[1:3])

    def train(self, data, **kwargs):
        log_data = np.log1p(data)
        mean = torch.from_numpy(np.mean(log_data.astype(np.float32), axis=0)).to(
            self.device
        )
        std = torch.from_numpy(np.std(log_data.astype(np.float32), axis=0) + 1e-6).to(
            self.device
        )

        self.model = self.build_model().to(self.device)
        self.model.set_scaler(mean, std)
        metrics_path = kwargs.pop("metrics_path", None)

        super().train(
            data,
            loss_fn=vae_loss_fn,
            max_epochs=1_000,
            prefix="Autoencoder | ",
            metrics_path=metrics_path,
            **kwargs,
        )
