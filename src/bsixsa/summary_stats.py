import numpy as np
import xspec
from scipy.stats import skew, entropy

def summary_statistics_func(
    data: np.ndarray,
    energy_grid=None,
    with_basic_stats=True,
    with_sum=True,
    with_ratio=True,
    with_diff=True,
    with_energy_weighted=False,
):

    if data.ndim == 1:
        data = data[np.newaxis, :]  # (1, M)
    num_spectrum, num_bins = data.shape

    data_transformed_list = []
    labels = []

    if with_basic_stats:

        mean_x = np.mean(data, axis=1)
        std_x = np.std(data, axis=1, ddof=1)
        sum_x = np.sum(data, axis=1)

        data_transformed_list.append(mean_x)
        labels.append("Mean")
        data_transformed_list.append(std_x)
        labels.append("Std")
        data_transformed_list.append(sum_x)
        labels.append("Sum")

    if len(energy_grid) == 2:

        energies = np.asarray(xspec.AllData(1).energies).T
        energy_bins_summary = np.append(energies[0], energies[1, -1])
        idx_low = np.searchsorted(energy_bins_summary, energy_grid.min())
        idx_high = np.searchsorted(energy_bins_summary, energy_grid.max())
        energy_bins_summary = energy_bins_summary[idx_low:idx_high + 1]

    else:
        energy_bins_summary = energy_grid

    counts = np.zeros((num_spectrum, len(energy_bins_summary),))
    energy_low_observation, energy_high_observation = np.asarray(xspec.AllData(1).energies).T

    for i, (e_low_summary, e_high_summary) in enumerate(zip(energy_bins_summary[:-1], energy_bins_summary[1:])):
        counts_in_bin = np.sum(data[:, (energy_low_observation >= e_low_summary) & (energy_high_observation <= e_high_summary)], axis=1)
        counts[:, i] += counts_in_bin

        if with_sum:

            data_transformed_list.append(counts_in_bin)
            labels.append(f"Sums in band {e_low_summary:.4f}-{e_high_summary:.4f}")

    epsilon = 1
    # Hardness ratios
    if with_ratio:
        hardness_ratios = counts[:, 1:] / (counts[:, :-1] + epsilon)

        for i, (e_low_1, e_high_1, e_low_2, e_high_2) in enumerate(
                zip(
                    energy_bins_summary[:-2],
                    energy_bins_summary[1:-1],
                    energy_bins_summary[1:-1],
                    energy_bins_summary[2:]
                )):

            data_transformed_list.append(hardness_ratios[:, i])
            labels.append(f"Hardness ratio [{e_low_2:.2f}-{e_high_2:.2f}]/[{e_low_1:.2f}-{e_high_1:.2f}]")

    # Differential ratios
    if with_diff:
        differential_ratios = (counts[:, :-1] - counts[:, 1:]) / (counts[:, :-1] + counts[:, 1:] + epsilon)

        for i, (e_low_1, e_high_1, e_low_2, e_high_2) in enumerate(
                zip(
                    energy_bins_summary[:-2],
                    energy_bins_summary[1:-1],
                    energy_bins_summary[1:-1],
                    energy_bins_summary[2:]
                )):

            data_transformed_list.append(differential_ratios[:, i])
            labels.append(f"Differential ratio [{e_low_2:.2f}-{e_high_2:.2f}]/[{e_low_1:.2f}-{e_high_1:.2f}]")

    if with_energy_weighted:
        for i, (e_low_summary, e_high_summary) in enumerate(zip(energy_bins_summary[:-1], energy_bins_summary[1:])):
            idx = (energy_low_observation >= e_low_summary) & (energy_high_observation <= e_high_summary)
            average_counts = data[:, idx]

            if average_counts.sum() < len(average_counts):
                average_counts = np.ones_like(average_counts)

            average_energy = (energy_low_observation[idx] + energy_high_observation[idx])/2
            result = np.apply_along_axis(lambda x : np.average(average_energy, weights=x/x.sum()), 1, average_counts)
            data_transformed_list.append(result)
            labels.append(f"Weighted energy in {e_low_summary:.4f}-{e_high_summary:.4f}")

    data_transformed = np.column_stack(data_transformed_list)

    return data_transformed, labels


def merge_summary(*args):

    reducers = args

    def summary_func(spectra):

        reduced_spectra_list = []
        summary_names_list = []

        if spectra.ndim == 1:
            spectra = spectra[None, :]

        for reducer in reducers:
            reduced_spectra, summary_names = reducer(spectra)
            reduced_spectra_list.append(reduced_spectra)
            summary_names_list += summary_names

        return np.hstack(reduced_spectra_list), summary_names_list

    return summary_func