from .abc import Embedding
from abc import abstractmethod, ABC
from torch import nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import torch
import torch.optim as optim
import xspec
from .nn_architectures import Autoencoder, VariationaAutoencoder


def cstat_loss(y_true, y_pred, eps=1e-6):
    y_true = torch.clamp(y_true, min=eps)
    y_pred = torch.clamp(y_pred, min=eps)

    term = torch.where((y_true>0.)&(y_pred>0.), torch.log(y_true) - torch.log(y_pred), 0.)
    cstat = 2 * torch.sum(y_pred - y_true + y_true * term, dim=1)

    return cstat.mean()


def vae_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def training_loop(
        model: nn.Module,
        data,
        device=torch.get_default_device(),
        patience=20, max_epochs=100, min_delta=1e-2,
        learning_rate=5e-4,
        weight_decay=1e-5,
        clip_grad_norm=5.,
        **kwargs
):

    train_dataset = TensorDataset(torch.from_numpy(data.astype(np.float32)))
    train_ds, val_ds = random_split(train_dataset, [0.9, 0.1])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.2,
        patience=10
    )

    num_epochs = max_epochs
    early_stop_patience = patience  # epochs with no sufficient improvement before stopping
    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    with tqdm() as pbar:

        for epoch in range(num_epochs):

            model.train()
            train_loss_sum = 0.

            # Training Loop
            for (spectrum,) in train_loader:
                spectrum = spectrum.to(device, non_blocking=True)
                outputs = model(spectrum)
                loss = cstat_loss(outputs, spectrum)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()

                train_loss_sum += loss.detach().item()

            avg_train_loss = train_loss_sum / len(train_loader)

            # Validation loop
            model.eval()
            val_loss_sum = 0.0

            with torch.no_grad():
                for (spectrum,) in val_loader:
                    spectrum = spectrum.to(device, non_blocking=True)
                    outputs = model(spectrum)
                    vloss = cstat_loss(outputs, spectrum)
                    val_loss_sum += vloss.detach().item()

            scheduler.step(val_loss_sum)
            lr_current = scheduler.optimizer.param_groups[0]['lr']

            avg_val_loss = val_loss_sum / max(1, len(val_loader))

            improvement = best_val_loss - avg_val_loss

            if improvement > min_delta:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
                pbar.set_description(f"{prefix}Early stopping at epoch {epoch}. Best was epoch {best_epoch} "
                      f"(val_loss={best_val_loss:.2f}).")
                break

            pbar.update(1)

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    model.eval()

    return model


def training_loop_vae(
        model: nn.Module,
        data,
        device=torch.get_default_device(),
        patience=20, max_epochs=100, min_delta=1e-2,
        learning_rate=5e-4,
        weight_decay=1e-5,
        clip_grad_norm=5.,
        **kwargs
):

    train_dataset = TensorDataset(torch.from_numpy(data.astype(np.float32)))
    train_ds, val_ds = random_split(train_dataset, [0.9, 0.1])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.2,
        patience=10
    )

    num_epochs = max_epochs
    early_stop_patience = patience  # epochs with no sufficient improvement before stopping
    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    with tqdm() as pbar:

        for epoch in range(num_epochs):

            model.train()
            train_loss_sum = 0.

            # Training Loop
            for (spectrum,) in train_loader:
                spectrum = spectrum.to(device, non_blocking=True)
                outputs, mu, logvar, _ = model(spectrum)
                reconstruction_loss = cstat_loss(outputs, spectrum)
                loss = reconstruction_loss + vae_loss(mu, logvar)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()

                train_loss_sum += reconstruction_loss.detach().item()

            avg_train_loss = train_loss_sum / len(train_loader)

            # Validation loop
            model.eval()
            val_loss_sum = 0.0

            with torch.no_grad():
                for (spectrum,) in val_loader:
                    spectrum = spectrum.to(device, non_blocking=True)
                    outputs, _, _, _ = model(spectrum)
                    reconstruction_loss = cstat_loss(outputs, spectrum)
                    val_loss_sum += reconstruction_loss.detach().item()

            scheduler.step(val_loss_sum)
            lr_current = scheduler.optimizer.param_groups[0]['lr']

            avg_val_loss = val_loss_sum / max(1, len(val_loader))

            improvement = best_val_loss - avg_val_loss

            if improvement > min_delta:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
                pbar.set_description(f"{prefix}Early stopping at epoch {epoch}. Best was epoch {best_epoch} "
                      f"(val_loss={best_val_loss:.2f}).")
                break

            pbar.update(1)

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    model.eval()

    return model



class TrainableEmbedding(Embedding, ABC):
    model: nn.Module
    trainable = True

    @abstractmethod
    def train(self, data, **kwargs):
        pass


class TorchEmbedding(TrainableEmbedding, ABC):

    def __init__(self, model, **kwargs):

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        self.model = model.to(self.device)

    @property
    def input_dim(self):
        data = np.asarray(xspec.AllData(1).values)
        return len(data)

    @property
    def output_dim(self):
        data = np.asarray(xspec.AllData(1).values)
        return int(self.model(torch.from_numpy(data.astype(np.float32)).to(self.device).unsqueeze(0)).shape[1])

    def names(self) -> list[str]:
        return [f"latent {i}" for i in range(1, self.output_dim+1)]

    def train(self, data, **kwargs):
        self.model = training_loop(self.model, data, device=self.device, **kwargs)

class AutoencoderEmbedding(TorchEmbedding):
    def __init__(self, latent_dim=32, **kwargs):

        self.latent_dim = latent_dim
        model = self.build_model()

        super().__init__(model, **kwargs)

    def build_model(self):
        return Autoencoder(
            self.input_dim,
            self.latent_dim,
        )

    def __call__(self, spectra):

        if not isinstance(spectra, torch.Tensor):
            spectra = torch.from_numpy(spectra.astype(np.float32)).to(self.device)

        return self.model.encoder(spectra)

    def train(self, data, **kwargs):

        log_data = np.log1p(data)
        mean = torch.from_numpy(np.mean(log_data.astype(np.float32), axis=0)).to(self.device)
        std = torch.from_numpy(np.std(log_data.astype(np.float32), axis=0) + 1e-6).to(self.device)

        self.model = self.build_model().to(self.device)
        self.model.set_scaler(mean, std)
        self.model = training_loop(
            self.model,
            data,
            max_epochs=1_000,
            device=self.device,
            prefix="Autoencoder | ",
            **kwargs
        )


class VAEEmbedding(TorchEmbedding):
    def __init__(self, latent_dim=32, **kwargs):

        self.latent_dim = latent_dim
        model = self.build_model()

        super().__init__(model, **kwargs)

    def build_model(self):
        return VariationaAutoencoder(
            self.input_dim,
            self.latent_dim,
        )

    def __call__(self, spectra):

        if not isinstance(spectra, torch.Tensor):
            spectra = torch.from_numpy(spectra.astype(np.float32)).to(self.device)

        return torch.hstack(self.model(spectra)[1:3])

    def train(self, data, **kwargs):

        log_data = np.log1p(data)
        mean = torch.from_numpy(np.mean(log_data.astype(np.float32), axis=0)).to(self.device)
        std = torch.from_numpy(np.std(log_data.astype(np.float32), axis=0) + 1e-6).to(self.device)

        self.model = self.build_model().to(self.device)
        self.model.set_scaler(mean, std)
        self.model = training_loop_vae(
            self.model,
            data,
            max_epochs=1_000,
            device=self.device,
            prefix="Autoencoder | ",
            **kwargs
        )