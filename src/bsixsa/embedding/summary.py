import numpy as np
import xspec
from abc import ABC
from .abc import Embedding
from dataclasses import dataclass, field
from typing import Callable


# Theses funcs assume a reduction over the last axis : (n_spectra, n_bins) -> (n_spectra,)
basic_global_stats = {
    "mean": lambda x: np.mean(x, axis=1),
    "std": lambda x: np.std(x, axis=1),
    "sum": lambda x: np.sum(x, axis=1),
}


@dataclass
class GlobalSummaryConfig:
    """Configuration object for a global summary statistics embedding."""

    energy_band: np.ndarray = field(default_factory=lambda: np.array([0.0, np.inf]))
    stats: dict[str, Callable] = field(default_factory=lambda: basic_global_stats)

    def __post_init__(self):
        self.energy_band = np.asarray(self.energy_band)  # normalize


class GlobalSummaryEmbedding(Embedding):
    """
    Compute global summary statistics over the spectra e.g. mean or variance.
    """

    def __init__(self, config: GlobalSummaryConfig = GlobalSummaryConfig()):
        self.stats = config.stats

    @property
    def names(self) -> list[str]:
        return list(self.stats.keys())

    def __call__(self, data):
        data_transformed_list = []

        if data.ndim == 1:
            data = data[np.newaxis, :]  # (1, M)

        for func in self.stats.values():
            data_transformed_list.append(func(data))

        return np.stack(data_transformed_list, axis=-1)


def local_sum(bin_data, data, bin_summary):
    energy_low_observation, energy_high_observation = bin_data
    energy_low_summary, energy_high_summary = bin_summary
    counts = np.zeros((len(data), len(energy_low_summary)))

    for i, (e_low_summary, e_high_summary) in enumerate(
        zip(energy_low_summary, energy_high_summary)
    ):
        bins_to_keep = (energy_low_observation >= e_low_summary) & (
            energy_high_observation <= e_high_summary
        )
        counts_in_bin = np.sum(data[:, bins_to_keep], axis=1)
        counts[:, i] += counts_in_bin

    return counts


class LocalSummaryEmbedding(Embedding, ABC):
    def __init__(self, energy_grid: np.ndarray = np.linspace(0.5, 8.0, 21)):
        if not (xspec.AllData.nSpectra > 0):
            raise RuntimeError(
                "There is no xspec data loaded. Please load your data before instanciating."
            )

        self.energy_grid = energy_grid
        self.energy_bins_data = np.asarray(xspec.AllData(1).energies).T
        self.energy_bins_summary = np.stack([energy_grid[:-1], energy_grid[1:]])


class LocalSumEmbedding(LocalSummaryEmbedding):
    @property
    def names(self) -> list[str]:
        names = []

        for e_low_summary, e_high_summary in zip(*self.energy_bins_summary):
            names.append(f"sum [{e_low_summary:.4f}-{e_high_summary:.4f}]")

        return names

    def __call__(self, data):
        return local_sum(self.energy_bins_data, data, self.energy_bins_summary)


class LocalSumWithExtra(LocalSummaryEmbedding):
    r"""
    Compute the sum and harness / differential ratios over a given number of energy bands. The harness / differential ratios (resp. $\text{HR}$ and $\text{DR}$)  are defined as

    $$
    \text{HR} = \frac{C_{i+1}}{C_{i} + \epsilon} \text{ and } \text{DR} = \frac{C_{i+1} - C_{i}}{C_{i+1} + C_{i} + \epsilon},
    $$

    where $i$ are the indexes of the instrumental bins, and $\epsilon = 1$.
    """

    def __init__(self, *args, with_ratio=False, with_diff=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_ratio = with_ratio
        self.with_diff = with_diff

    @property
    def names(self) -> list[str]:
        names = []
        bin_summary = self.energy_grid

        for e_low_summary, e_high_summary in zip(bin_summary[:-1], bin_summary[1:]):
            names.append(f"sum [{e_low_summary:.4f}-{e_high_summary:.4f}]")

        if self.with_ratio:
            for i, (e_low_1, e_high_1, e_low_2, e_high_2) in enumerate(
                zip(
                    bin_summary[:-2],
                    bin_summary[1:-1],
                    bin_summary[1:-1],
                    bin_summary[2:],
                )
            ):
                names.append(
                    f"ratio [{e_low_2:.2f}-{e_high_2:.2f}]/[{e_low_1:.2f}-{e_high_1:.2f}]"
                )

        if self.with_diff:
            for i, (e_low_1, e_high_1, e_low_2, e_high_2) in enumerate(
                zip(
                    bin_summary[:-2],
                    bin_summary[1:-1],
                    bin_summary[1:-1],
                    bin_summary[2:],
                )
            ):
                names.append(
                    f"diff [{e_low_2:.2f}-{e_high_2:.2f}]/[{e_low_1:.2f}-{e_high_1:.2f}]"
                )

        return names

    def __call__(self, data):
        embedded_data = []

        epsilon = 1
        counts = local_sum(self.energy_bins_data, data, self.energy_bins_summary)
        embedded_data.append(counts)

        if self.with_ratio:
            hardness_ratios = counts[:, 1:] / (counts[:, :-1] + epsilon)
            embedded_data.append(hardness_ratios)

        if self.with_diff:
            differential_ratios = (counts[:, :-1] - counts[:, 1:]) / (
                counts[:, :-1] + counts[:, 1:] + epsilon
            )
            embedded_data.append(differential_ratios)

        return np.hstack(embedded_data)


class WeightedEnergyEmbedding(LocalSummaryEmbedding):
    r"""
    Compute the energy-weighted mean over a given number of energy bands. The energy-weighted mean is defined as

    $$
    \bar{E}_i = \frac{\sum_j E_j C_j}{\sum_j C_j},
    $$

    where $i$ is the index of the coarser bins and $j$ are the indexes of the instrumental bins.
    """

    @property
    def names(self) -> list[str]:
        names = []

        for e_low_summary, e_high_summary in zip(*self.energy_bins_summary):
            names.append(f"weighted energy [{e_low_summary:.4f}-{e_high_summary:.4f}]")

        return names

    def __call__(self, data):
        energy_low_observation, energy_high_observation = self.energy_bins_data
        data_transformed_list = []

        for i, (e_low_summary, e_high_summary) in enumerate(
            zip(*self.energy_bins_summary)
        ):
            idx = (energy_low_observation >= e_low_summary) & (
                energy_high_observation <= e_high_summary
            )
            average_counts = data[:, idx]

            if average_counts.sum() < len(average_counts):
                average_counts = np.ones_like(average_counts)

            average_energy = (
                energy_low_observation[idx] + energy_high_observation[idx]
            ) / 2
            result = np.apply_along_axis(
                lambda x: np.average(average_energy, weights=x / x.sum()),
                1,
                average_counts,
            )
            data_transformed_list.append(result)

        return np.vstack(data_transformed_list).T
