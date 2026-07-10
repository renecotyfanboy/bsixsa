"""Embedding networks trained jointly with the NPE via sbi's ``embedding_net``.

Unlike :class:`~bsixsa.backend.sixsa.embedding.abc.Embedding` (a numpy ``fit``/
``transform`` pre-processor that runs *before* training), the classes here describe a
:class:`torch.nn.Module` that is handed to ``posterior_nn(embedding_net=...)`` and
optimised *together* with the density estimator.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import nn


# Map string names to activation classes (``"none"`` -> a purely linear network).
ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leakyrelu": nn.LeakyReLU,
    "identity": nn.Identity,
    "none": None,
}


def resolve_activation(activation):
    """Resolve an activation from a string, an ``nn.Module`` subclass, or ``None``.

    Args:
        activation: ``"gelu"``/``"relu"``/... , an ``nn.Module`` subclass, or
            ``None`` (and ``"none"``) for a purely linear network.

    Returns:
        The activation class (not an instance) or ``None``.
    """
    if activation is None:
        return None
    if isinstance(activation, str):
        key = activation.lower().replace(" ", "")
        if key not in ACTIVATION_MAP:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Valid options: {list(ACTIVATION_MAP.keys())}"
            )
        return ACTIVATION_MAP[key]
    return activation


def compute_hidden_units(
    input_dim: int,
    output_dim: int,
    n_layers: int,
    max_compression: float | None = None,
) -> list[int]:
    """Geometrically spaced hidden-layer sizes between ``input_dim`` and ``output_dim``.

    Each layer compresses by the same factor, giving a smooth pyramid. With
    ``n_layers=0`` returns ``[]`` (a direct linear projection).

    Args:
        input_dim: Number of input features (e.g. spectral bins).
        output_dim: Embedding dimension.
        n_layers: Number of intermediate hidden layers.
        max_compression: Optional maximum compression factor per layer; raises
            if the required per-layer compression exceeds it.

    Returns:
        List of hidden-layer sizes (length ``n_layers``).
    """
    if output_dim >= input_dim:
        raise ValueError(
            f"output_dim ({output_dim}) must be smaller than input_dim ({input_dim})."
        )
    if n_layers == 0:
        return []

    # input_dim * ratio^(n_layers + 1) = output_dim
    ratio = (output_dim / input_dim) ** (1.0 / (n_layers + 1))
    compression_per_layer = 1.0 / ratio

    if max_compression is not None and compression_per_layer > max_compression:
        raise ValueError(
            f"Required compression per layer ({compression_per_layer:.2f}x) "
            f"exceeds max_compression ({max_compression:.2f}x). "
            f"Increase n_layers or relax max_compression."
        )

    hidden_units = []
    current = input_dim
    for _ in range(n_layers):
        current = math.ceil(current * ratio)
        hidden_units.append(current)
    return hidden_units


class FCEmbeddingAPD(nn.Module):
    """Fully-connected embedding net with automatic pyramid compression.

    Compresses ``input_dim`` to ``output_dim`` across ``n_layers`` hidden layers
    with geometrically spaced widths. Each hidden layer applies
    ``Linear -> [LayerNorm] -> [Activation] -> [Dropout]``; the final layer is a
    plain ``Linear`` with an optional (non-affine) ``LayerNorm``. With
    ``activation="none"`` and the norms disabled the network is purely linear —
    this is how ``_sixsa_dev`` embeds its spectra by default.

    Args:
        input_dim: Number of input features (spectral bins).
        output_dim: Embedding dimension fed to the flow.
        n_layers: Number of intermediate hidden layers (0 = direct projection).
        num_hiddens: ``None`` -> geometric pyramid; ``int`` -> equal sizes;
            ``list[int]`` -> explicit sizes (length must equal ``n_layers``).
        activation: Activation as a string, an ``nn.Module`` subclass, or ``None``
            (``"none"`` -> linear).
        enable_layer_norm: Apply ``LayerNorm`` after each hidden layer.
        normalize_output: Apply a non-affine ``LayerNorm`` to the output.
        dropout_rate: Dropout probability after each activation (0 disables).
        max_compression: Optional per-layer compression cap (only when
            ``num_hiddens=None``).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 32,
        n_layers: int = 2,
        num_hiddens: int | list[int] | None = None,
        activation=None,
        enable_layer_norm: bool = False,
        normalize_output: bool = False,
        dropout_rate: float = 0.0,
        max_compression: float | None = None,
    ):
        super().__init__()

        activation = resolve_activation(activation)

        if num_hiddens is None:
            hidden_sizes = compute_hidden_units(
                input_dim=input_dim,
                output_dim=output_dim,
                n_layers=n_layers,
                max_compression=max_compression,
            )
        elif isinstance(num_hiddens, int):
            hidden_sizes = [num_hiddens] * n_layers
        else:
            hidden_sizes = list(num_hiddens)
            if len(hidden_sizes) != n_layers:
                raise ValueError(
                    f"num_hiddens has length {len(hidden_sizes)} but n_layers={n_layers}."
                )

        layers = []
        prev = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev, hidden_dim))
            if enable_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            if activation is not None:
                layers.append(activation())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            prev = hidden_dim

        layers.append(nn.Linear(prev, output_dim))
        if normalize_output:
            layers.append(nn.LayerNorm(output_dim, elementwise_affine=False))

        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_sizes = hidden_sizes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Network forward pass mapping ``(batch, input_dim)`` to ``(batch, output_dim)``."""
        return self.net(x)

    def __repr__(self) -> str:
        arch = [self.input_dim] + self.hidden_sizes + [self.output_dim]
        return f"FCEmbeddingAPD(architecture={arch}, n_layers={self.n_layers})"


class FCEmbedding(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 32,
        num_hiddens=[256],
    ):
        """Fully-connected neural network to be used as an embedding network.

        Args:
            input_dim: Dimensionality of the input passed to the embedding net.
            output_dim: Dimensionality of the output.
            num_hiddens: Width of each hidden layer. An empty sequence yields a
                single linear projection ``Linear(input_dim, output_dim)``.
        """
        super().__init__()
        layers = []
        num_layers = len(num_hiddens)
        prev = input_dim
        for i in range(num_layers):
            layers.append(torch.nn.Linear(prev, num_hiddens[i]))
            layers.append(torch.nn.LayerNorm(num_hiddens[i]))
            layers.append(torch.nn.GELU())
            prev = num_hiddens[i]

        layers.append(torch.nn.Linear(prev, output_dim))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Network forward pass.

        Args:
            x: Input tensor ``(batch_size, input_dim)``.

        Returns:
            Network output ``(batch_size, output_dim)``.
        """
        return self.net(x)


class EmbeddingNet(ABC):
    """Factory for an embedding network trained jointly with the NPE.

    The input dimensionality is only known at training time, so subclasses implement
    :meth:`build` to produce a fresh :class:`torch.nn.Module` rather than holding one
    directly. Instances must stay picklable so they survive joblib's ``delayed``
    dispatch to the per-network worker processes.
    """

    @abstractmethod
    def build(self, input_dim: int) -> torch.nn.Module:
        """Return a fresh embedding module mapping ``input_dim`` features to the latent space."""
        ...


class DenseEmbeddingNet(EmbeddingNet):
    """Dense (fully-connected) embedding net, trained jointly with the NPE.

    With the default ``num_hiddens=()`` the network is a single
    ``nn.Linear(input_dim, output_dim)`` — a linear projection to ``output_dim`` latent
    features. Passing hidden widths (e.g. ``num_hiddens=(256,)``) makes it a deeper
    dense network.

    Args:
        output_dim: Dimensionality of the latent embedding. Defaults to 32.
        num_hiddens: Hidden layer widths. Empty (default) gives a linear projection.
    """

    def __init__(self, output_dim: int = 32, num_hiddens: Sequence[int] = ()):
        self.output_dim = output_dim
        self.num_hiddens = list(num_hiddens)

    def build(self, input_dim: int) -> torch.nn.Module:
        return FCEmbedding(input_dim, self.output_dim, self.num_hiddens)


class FCEmbeddingNet(EmbeddingNet):
    """Pyramid embedding net (``FCEmbeddingAPD``), trained jointly with the NPE.

    This is the default ``bsixsa`` embedding and mirrors how ``_sixsa_dev`` embeds
    spectra: a geometrically compressing fully-connected stack from the input
    spectrum down to ``output_dim`` latent features. With ``activation="none"``
    (the default) it is a purely linear compression — well suited to very
    high-resolution spectra.

    Args:
        output_dim: Dimensionality of the latent embedding. Defaults to 32.
        n_layers: Number of intermediate hidden layers. Defaults to 2.
        activation: Activation passed to :class:`FCEmbeddingAPD` (``"none"`` for a
            linear network). Defaults to ``"none"``.
        enable_layer_norm: Apply ``LayerNorm`` after each hidden layer.
        normalize_output: Apply a non-affine ``LayerNorm`` to the output.
        dropout_rate: Dropout probability after each activation (0 disables).
        num_hiddens: Optional explicit hidden sizes (overrides the pyramid).
    """

    def __init__(
        self,
        output_dim: int = 32,
        n_layers: int = 2,
        activation: str | None = "none",
        enable_layer_norm: bool = False,
        normalize_output: bool = False,
        dropout_rate: float = 0.0,
        num_hiddens: int | Sequence[int] | None = None,
    ):
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.activation = activation
        self.enable_layer_norm = enable_layer_norm
        self.normalize_output = normalize_output
        self.dropout_rate = dropout_rate
        self.num_hiddens = (
            list(num_hiddens) if isinstance(num_hiddens, Sequence) else num_hiddens
        )

    def build(self, input_dim: int) -> torch.nn.Module:
        return FCEmbeddingAPD(
            input_dim,
            output_dim=self.output_dim,
            n_layers=self.n_layers,
            num_hiddens=self.num_hiddens,
            activation=self.activation,
            enable_layer_norm=self.enable_layer_norm,
            normalize_output=self.normalize_output,
            dropout_rate=self.dropout_rate,
        )
