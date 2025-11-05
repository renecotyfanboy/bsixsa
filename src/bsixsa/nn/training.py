from pathlib import Path

import torch
from torch import nn, optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm


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
        for batch in loader:
            batch = tuple(value.to(device, non_blocking=True) for value in batch)
            loss, metrics = model.loss(batch)

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
    train_dataset,
    *,
    device=torch.get_default_device(),
    patience=20,
    max_epochs=100,
    min_delta=1e-2,
    learning_rate=3e-4,
    weight_decay=1e-5,
    clip_grad_norm=5.0,
    batch_size=256,
    optimizer_cls: type[optim.Optimizer] = optim.AdamW,
    optimizer_kwargs: dict | None = None,
    metrics_path: str | Path | None = None,
    **kwargs,
):
    train_ds, val_ds = random_split(train_dataset, [0.9, 0.1])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    optimizer_params = dict(optimizer_kwargs or {})
    optimizer_params.setdefault("lr", learning_rate)
    optimizer_params.setdefault("weight_decay", weight_decay)

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=1/3, patience=10, min_lr=1e-5
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
                optimizer=optimizer,
                clip_grad_norm=clip_grad_norm,
            )

            val_metrics_sum, val_metrics_avg = _epoch_pass(
                model,
                val_loader,
                device,
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
