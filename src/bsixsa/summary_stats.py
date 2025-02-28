import numpy as np
import xspec
from scipy.stats import skew, entropy

def summary_statistics_func(
    data: np.ndarray,
    energy_grid=None,
    with_ratio=True,
    with_diff=True,
):
    """
    Transforme un ou plusieurs vecteurs 1D en un ensemble de statistiques résumées,
    et ajoute un découpage en bandes d'énergie, le calcul de hardness ratios et de
    differential ratios, en utilisant NumPy et scipy.

    Paramètres
    ----------
    data : np.ndarray
        Tableau Numpy 1D ou 2D.
        - Si 1D, on considère un seul vecteur de taille (M,).
        - Si 2D, on considère N vecteurs de taille (N, M).
    compute_skew : bool
        Indique s'il faut calculer le coefficient d'asymétrie (skewness).
    compute_entropy_stat : bool
        Indique s'il faut calculer l'entropie de chaque vecteur.
    percentile_values : list
        Liste des percentiles à calculer. Par défaut [90, 95].
    energy_min, energy_max : float
        Limites min/max en énergie (keV) pour la création d'une grille log-spacée.
    number_of_intervals_for_summary_statistics : int
        Nombre d'intervalles (bandes) sur lesquels on veut sommer les comptes.
    e_min_folded, e_max_folded : np.ndarray
        Limites d'énergie réelles de chaque bin dans le spectre.
        Doivent être de taille M si data est de taille (N, M).
    epsilon : float
        Constante permettant d'éviter les divisions par zéro dans les ratios.

    Retour
    ------
    data_transformed : np.ndarray
        Tableau Numpy 2D de forme (N, K) où K est le nombre total de features calculées.
    labels : list
        Liste des noms de chaque feature, dans le même ordre que les colonnes de `data_transformed`.

    Exemple
    -------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> stats, names = summary_statistics_func(x)
    >>> print(stats.shape)
    (1, 8)  # (N=1, nb_features=8)
    >>> print(names)
    ['Mean', 'Std', 'Variance', 'Sum', 'Skewness', 'Entropy',
     'Percentile 90', 'Percentile 95', 'IQR (p95 - p90)',
     ... "Sums in each band", ... "Hardness ratios", ... "Differential ratios" ...]
    """
    # ------------------------------------------------------
    # 1) S'assurer que 'data' est 2D : (N, M)
    # ------------------------------------------------------
    if data.ndim == 1:
        data = data[np.newaxis, :]  # (1, M)
    num_spectrum, num_bins = data.shape

    data_transformed_list = []
    labels = []

    # ------------------------------------------------------
    # 2) Calcul des statistiques de base en NumPy
    # ------------------------------------------------------
    """

    mean_x = np.mean(data, axis=1)
    std_x = np.std(data, axis=1, ddof=1)
    var_x = np.var(data, axis=1, ddof=1)
    sum_x = np.sum(data, axis=1)

    # ------------------------------------------------------
    # 3) Skewness et Entropy (optionnels)
    # ------------------------------------------------------
    skew_x = None
    if compute_skew:
        skew_x = skew(data, axis=1)

    entropy_x = None
    if compute_entropy_stat:
        # scipy.stats.entropy agit normalement en dernier paramètre sur l'axe
        # qui représente les "catégories" (i.e. bins ici).
        # On utilise un petit offset pour éviter log(0).
        # N.B. si data contient des valeurs négatives, cela peut être un problème.
        # A adapter si besoin.
        # Ici, on suppose data >= 0. Sinon, il faudrait normaliser différemment.
        # (axis=1 => on calcule l'entropie par ligne)
        data_pos = data.copy()
        data_pos[data_pos < 0] = 0.0
        # Pour être strict, on pourrait normaliser chaque ligne pour faire
        # de la distribution de probas:
        row_sums = data_pos.sum(axis=1, keepdims=True) + 1e-12
        data_prob = data_pos / row_sums
        entropy_x = np.apply_along_axis(entropy, 1, data_prob)

    # ------------------------------------------------------
    # 6) Construction de la liste de features + noms
    # ------------------------------------------------------


    # -- Mean
    data_transformed_list.append(mean_x)
    labels.append("Mean")
    # -- Std
    data_transformed_list.append(std_x)
    labels.append("Std")
    # -- Variance
    data_transformed_list.append(var_x)
    labels.append("Variance")
    # -- Sum
    data_transformed_list.append(sum_x)
    labels.append("Sum")
    # -- Skewness
    if skew_x is not None:
        data_transformed_list.append(skew_x)
        labels.append("Skewness")
    # -- Entropy
    if entropy_x is not None:
        data_transformed_list.append(entropy_x)
        labels.append("Entropy")
    """
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

    # ------------------------------------------------------
    # 8) Concaténation finale en un seul tableau
    #    shape => (N, K)
    # ------------------------------------------------------
    # data_transformed_list est une liste de vecteurs (chacun de taille N)
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