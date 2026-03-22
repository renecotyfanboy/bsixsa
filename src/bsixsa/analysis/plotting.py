import itertools
import matplotlib.pyplot as plt
import numpy as np
import catppuccin
from catppuccin.extras.matplotlib import load_color
from scipy.stats import nbinom, norm
from xspec import Plot
from ..xspec import SpectrumState
from ..backend.abc import Backend

PALETTE = catppuccin.PALETTE.latte

COLOR_CYCLE = [
    load_color(PALETTE.identifier, color)
    for color in [
        "sky",
        "teal",
        "green",
        "yellow",
        "peach",
        "maroon",
        "red",
        "pink",
        "mauve",
        "blue",
    ][::-1]
]

SPECTRUM_COLOR = load_color(PALETTE.identifier, "blue")
SPECTRUM_DATA_COLOR = load_color(PALETTE.identifier, "overlay2")
BACKGROUND_DATA_COLOR = load_color(PALETTE.identifier, "text")
alpha_median = 0.7
alpha_envelope = (0.15, 0.25)

def adaptive_bin_1d(counts, min_counts):
    """Assign adjacent bins to groups so that each group has at least
    *min_counts* total counts.

    Parameters:
        counts (numpy.ndarray): Observed counts per original bin (1-D).
        min_counts (int): Minimum summed counts required per grouped bin.

    Returns:
        numpy.ndarray: Integer group id for every original bin.
    """
    bin_ids = np.zeros(len(counts), dtype=int)
    current_bin = 0
    running_sum = 0

    for i, c in enumerate(counts):
        running_sum += c
        bin_ids[i] = current_bin
        if running_sum >= min_counts:
            current_bin += 1
            running_sum = 0

    # Merge the last under-threshold group into its predecessor
    if running_sum < min_counts and current_bin > 0:
        bin_ids[bin_ids == current_bin] = current_bin - 1

    return bin_ids


def rebin_counts(data, bin_ids):
    """Sum counts inside each bin group.

    Parameters:
        data (numpy.ndarray): Count array — either 1-D ``(n_bins,)`` or 2-D
            ``(n_samples, n_bins)``.
        bin_ids (numpy.ndarray): Group id for every original bin (length
            ``n_bins``).

    Returns:
        numpy.ndarray: Rebinned array with the last axis reduced to the number
            of groups.
    """
    n_groups = bin_ids.max() + 1
    if data.ndim == 1:
        return np.bincount(bin_ids, weights=data, minlength=n_groups)
    agg = bin_ids[None, :] == np.arange(n_groups)[:, None]  # (n_groups, n_bins)
    return data @ agg.T  # (n_samples, n_groups)


def rebin_edges(bin_edges_1d, bin_ids):
    """Compute new 1-D bin edges by keeping the outer boundaries of each group.

    Parameters:
        bin_edges_1d (numpy.ndarray): Original edges of length ``n_bins + 1``.
        bin_ids (numpy.ndarray): Group id for every original bin (length
            ``n_bins``).

    Returns:
        numpy.ndarray: New edges of length ``n_groups + 1``.
    """
    starts = np.r_[0, np.flatnonzero(np.diff(bin_ids)) + 1]
    return bin_edges_1d[np.append(starts, len(bin_ids))]


def sigma_to_percentile_intervals(sigmas:list[float]) -> list[tuple[float, float]]:
    """
    Calculate percentile intervals corresponding to given sigma values.

    This function computes the lower and upper percentile bounds for each input
    sigma (standard deviation) value, assuming a normal distribution. The
    calculated intervals represent the percentage of data within ±sigma
    standard deviations of the mean.

    Parameters:
        sigmas : list[float]
            A list of standard deviation (sigma) values for which to calculate
            percentile intervals.

    Returns:
        list[tuple[float, float]]
            A list of tuples, where each tuple contains the lower and upper
            percentile bounds corresponding to a given sigma value.
    """
    intervals = []
    for sigma in sigmas:
        lower_bound = 100 * norm.cdf(-sigma)
        upper_bound = 100 * norm.cdf(sigma)
        intervals.append((lower_bound, upper_bound))
    return intervals


def poisson_error_bars(observed_counts:np.ndarray, sigma=1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Calculate Gamma-prior credible intervals for observed counts.

    Parameters:
        observed_counts (numpy.ndarray): Observed counts per bin.
        sigma (float, optional): Desired dispersion expressed in Gaussian
            sigmas for the resulting quantile interval. Defaults to 1.

    Returns:
        (tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]): Observed counts,
            lower bound, and upper bound corresponding to the requested
            credible interval.
    """

    percentile = sigma_to_percentile_intervals([sigma])[0]

    y_observed = observed_counts
    y_observed_low = nbinom.ppf(percentile[0] / 100, observed_counts, 0.5)
    y_observed_high = nbinom.ppf(percentile[1] / 100, observed_counts, 0.5)

    return y_observed, y_observed_low, y_observed_high


def plot_poisson_error_bars(center, edges, data, denominator, ax, color):
    r"""Plot observed counts with Gamma-prior credible error bars.

    Parameters:
        center (numpy.ndarray): Bin centers.
        edges (numpy.ndarray): Lower/upper bin edges with shape ``(2, n_bins)``.
        data (numpy.ndarray): Observed counts per bin.
        denominator (numpy.ndarray | float): Divisor applied to convert counts
            to plotting units.
        ax (matplotlib.axes.Axes): Target axis.
        color: The data color.

    Returns:
        matplotlib.container.ErrorbarContainer: Error-bar artist handle.
    """
    y_observed, y_observed_low, y_observed_high = poisson_error_bars(
        data, sigma=1
    )

    y_observed, y_observed_low, y_observed_high = (
        y_observed / denominator,
        y_observed_low / denominator,
        y_observed_high / denominator,
    )

    error_bar = ax.errorbar(
        center,
        y_observed,
        xerr=np.abs(edges - center[None, :]),
        yerr=[
            np.maximum(y_observed - y_observed_low, 0),
            np.maximum(y_observed_high - y_observed, 0),
        ],
        linestyle="none",
        color=SPECTRUM_DATA_COLOR,
        alpha=0.8,
        capsize=2,
        zorder=10,
    )

    return error_bar


def plot_median_and_bands(
    edges,
    data,
    denominator,
    ax,
    color,
    alpha_envelope=(0.15, 0.25),
    **kwargs,
):
    """Plot median, 68%, and 95% predictive bands for sampled spectra.

    Parameters:
        edges (numpy.ndarray): Bin edges passed to ``matplotlib.axes.Axes.stairs``.
        data (numpy.ndarray): Sampled spectra with shape
            ``(n_samples, n_bins)``.
        denominator (numpy.ndarray | float): Divisor applied to ``data`` before
            plotting.
        ax (matplotlib.axes.Axes): Target axis.
        color: Base color for the median line and envelopes.
        alpha_envelope (tuple[float, float], optional): Alpha values for the
            95% and 68% filled bands, respectively.
        **kwargs: Extra keyword arguments forwarded to the filled stairs and
            legend proxy patch.

    Returns:
        tuple[matplotlib.patches.StepPatch, matplotlib.patches.Polygon]:
            Handles for the median line and envelope proxy used in the legend.
    """
    data = data / denominator

    median = ax.stairs(
        np.median(data, axis=0),
        edges=edges,
        color=color,
        alpha=alpha_median,
        zorder=100,
        linestyle="solid",
    )

    low_band, high_band = np.percentile(data, [16, 84], axis=0)

    ax.stairs(
        high_band,
        edges=edges,
        baseline=low_band,
        fill=True,
        alpha=alpha_envelope[1],
        color=color,
        zorder=80,
        **kwargs
    )

    low_band, high_band = np.percentile(data, [2.5, 97.5], axis=0)

    ax.stairs(
        high_band,
        edges=edges,
        baseline=low_band,
        fill=True,
        alpha=alpha_envelope[0],
        color=color,
        zorder=60,
        **kwargs
    )

    # The legend cannot handle fill_between, so we pass a fill to get a fancy icon
    (envelope,) = ax.fill(
        np.nan, np.nan, alpha=alpha_envelope[-1], facecolor=color, **kwargs
    )

    return median, envelope


def plot_predictive_coverage(
    solver,
    distribution="posterior",
    x_lim=None,
    y_lim=None,
    figsize=(12, 6),
    plot_background=False,
    plot_models=True,
    plot_components=True,
    legend=True,
    n_samples=100,
    min_counts=None,
    grouping=None,
    residual_kind="pearson",
):
    r"""Plot posterior predictive spectra with residuals.

    The lower panel displays residuals between the observed data $D$ and the
    sampled model predictions $M_i$.  Three formulae are available, selected
    via the *residual_kind* parameter:

    * **Pearson** (``"pearson"``, default) — per-sample Poisson normalisation,
      well defined for both prior and posterior predictive checks:

    $$
    r_i = \frac{D - M_i}{\sqrt{M_i}}
    $$

    * **Sigma** (``"sigma"``) — normalised by the 68 % ensemble spread; can be
      misleading when the prior is broad:

    $$
    r = \frac{M - D}{P_{84} - P_{16}}
    $$

    * **Raw** (``"raw"``) — un-normalised residuals in data units
      (counts / keV / s):

    $$
    r_i = \frac{D - M_i}{\Delta E}
    $$

    Parameters:
        solver: Solver instance exposing ``distributions`` and ``simulate``.
        distribution (str, optional): Name of the distribution to sample from,
            looked up in ``solver.distributions``. Common values are
            ``"prior"`` and ``"posterior"``. Defaults to ``"posterior"``.
        x_lim (tuple[float, float], optional): Energy bounds in keV for the
            upper panel x-axis.
        y_lim (tuple[float, float], optional): Flux limits for the upper panel
            y-axis.
        figsize (tuple[float, float], optional): Matplotlib figure size in
            inches. Defaults to ``(12, 6)``.
        plot_background (bool, optional): Reserved for background overlays.
            This currently has no effect unless the simulated payload contains a
            ``background`` entry.
        plot_models (bool, optional): If ``True``, draw one predictive band per
            source model (in addition to the total model).
        plot_components (bool, optional): If ``True``, draw additive-component
            predictive bands for each source model.
        legend (bool, optional): Whether to display the legend on the spectrum
            panel. Defaults to ``True``.
        n_samples (int, optional): Number of parameter draws used to build the
            predictive bands. Defaults to ``100``.
        min_counts (int, optional): If set, adjacent bins are merged until every
            grouped bin contains at least *min_counts* observed counts.  The
            same grouping is applied to all plotted quantities (observed
            spectrum, models, components, and residuals).  ``None`` disables
            adaptive rebinning.  Defaults to ``None``.  Mutually exclusive with
            *grouping*.
        grouping (int, optional): If set, every *grouping* adjacent bins are
            merged into one (the last group may contain fewer bins).  Mutually
            exclusive with *min_counts*.  Defaults to ``None``.
        residual_kind (str, optional): Formula used for the residual panel.
            One of ``"pearson"``, ``"sigma"``, or ``"raw"`` (see above).
            Defaults to ``"pearson"``.

    Returns:
        matplotlib.figure.Figure: Figure containing spectrum and residual
            panels.
    """

    # TODO : check for background
    # TODO : let the user chose the components to plot
    # TODO : add ARF division when relevant
    # TODO : issue when sometime the components are too high compared to the reference model. Happens when
    # we charge a new xspec model. What could it be ?

    if min_counts is not None and grouping is not None:
        raise ValueError(
            "min_counts and grouping are mutually exclusive; use one or the other."
        )

    Plot.xAxis = "keV"
    state = SpectrumState(1)

    dist = solver.distributions[distribution]
    if isinstance(dist, Backend):
        parameters = dist.sample(n_samples)
    else:
        parameters = dist.sample((n_samples,))
    if hasattr(parameters, "numpy"):
        parameters = parameters.numpy()

    data = solver.simulate(parameters, return_kind="models_and_components")

    bin_edges_1d = state.bin_edges_1d
    observed_counts = state.observed_counts

    # Compute bin group ids (if any rebinning is requested)
    bin_ids = None
    if min_counts is not None:
        bin_ids = adaptive_bin_1d(observed_counts, min_counts)
    elif grouping is not None:
        n_bins = len(observed_counts)
        bin_ids = np.arange(n_bins) // grouping

    if bin_ids is not None:
        bin_edges_1d = rebin_edges(bin_edges_1d, bin_ids)
        observed_counts = rebin_counts(observed_counts, bin_ids)
        data["total_model_counts"] = rebin_counts(
            data["total_model_counts"], bin_ids
        )
        data["model_counts"] = {
            k: rebin_counts(v, bin_ids)
            for k, v in data["model_counts"].items()
        }
        data["component_counts"] = {
            k: rebin_counts(v, bin_ids)
            for k, v in data["component_counts"].items()
        }

    bin_edges = np.array([bin_edges_1d[:-1], bin_edges_1d[1:]])
    bin_width = np.diff(bin_edges, axis=0).squeeze()
    bin_center = np.sqrt(bin_edges[0] * bin_edges[1])
    denominator = bin_width

    legend_names = []
    legend_list = []

    fig, axs = plt.subplots(
        nrows=2, ncols=1, figsize=figsize, sharex=True, height_ratios=[4, 1]
    )

    ### OBSERVED SPECTRUM
    error_bar = plot_poisson_error_bars(bin_center, bin_edges, observed_counts, denominator, axs[0], SPECTRUM_DATA_COLOR)
    legend_list.append(error_bar)
    legend_names.append("Observed Spectrum")

    ### TOTAL MODEL
    total_model = np.random.poisson(data["total_model_counts"])
    median, envelope = plot_median_and_bands(bin_edges_1d, total_model, denominator, axs[0], color=SPECTRUM_COLOR)
    legend_list.append((median, envelope))
    legend_names.append("Total model")

    ### INDIVIDUAL MODELS
    colors = iter(COLOR_CYCLE[1:])

    for i, (model_name, model_counts) in enumerate(
        data["model_counts"].items()
    ):

        ### TOTAL MODEL COUNTS
        if plot_models and (len(data["model_counts"]) > 1):

            local_model = np.random.poisson(model_counts)
            median, envelope = plot_median_and_bands(bin_edges_1d, local_model, denominator, axs[0], color=next(colors))
            legend_list.append((median, envelope))
            legend_names.append(model_name)

        ### ADDITIVE COMPONENTS
        if plot_components:

            hatches = itertools.cycle(['//', r'\\', '||', '--', '++', 'xx', 'oo', 'OO', '..', '**'])

            for component_name, component_counts in data["component_counts"].items():
                if component_name.startswith(model_name + "_"):

                    local_model = np.random.poisson(component_counts)
                    median, envelope = plot_median_and_bands(
                        bin_edges_1d,
                        local_model, denominator, axs[0], next(colors),
                        alpha_envelope=(0.05, 0.1),
                        hatch=next(hatches),
                        hatch_linewidth=1,
                        edgecolor=(1/3, 1/3, 1/3, 1), #
                    )

                    legend_list.append((median, envelope))
                    legend_names.append(component_name.lstrip("_")) # We remove the extra "_" if the model name is ""

    """
    if plot_background and data.get("background") is not None:
        background = (
            np.random.negative_binomial(
                np.repeat(solver._background[None, :], len(models), axis=0) + 1, 1 / 2
            )
            * solver._backratio
        )
    
        total += background
    """

    total_model_flux = total_model / denominator
    y_observed_flux = observed_counts / denominator

    _VALID_RESIDUALS = {"pearson", "sigma", "raw"}
    if residual_kind not in _VALID_RESIDUALS:
        raise ValueError(
            f"residual_kind must be one of {_VALID_RESIDUALS!r}, "
            f"got {residual_kind!r}"
        )

    if residual_kind == "pearson":
        safe_model = np.where(total_model > 0, total_model, 1.0)
        residuals = (observed_counts - total_model) / np.sqrt(safe_model)
        residual_label = r"$\frac{D - M}{\sqrt{M}}$"
    elif residual_kind == "sigma":
        divider = (
            np.percentile(total_model_flux, 84, axis=0)
            - np.percentile(total_model_flux, 16, axis=0)
        )
        residuals = (total_model_flux - y_observed_flux) / np.where(divider > 0, divider, 1.0)
        residual_label = r"$\left[ \sigma \right]$"
    else:  # raw
        residuals = (observed_counts - total_model) / denominator
        residual_label = r"$\frac{\text{Counts}}{\text{keV} \text{s}}$"

    plot_median_and_bands(bin_edges_1d, residuals, np.ones_like(observed_counts), axs[1], color=SPECTRUM_COLOR)

    if x_lim is None:
        x_lim = (np.min(bin_edges_1d), np.max(bin_edges_1d))

    axs[0].set_xlim(*x_lim)

    if y_lim is not None:
        axs[0].set_ylim(*y_lim)

    low_env, high_env = np.percentile(residuals, [2.5, 97.5], axis=0)
    residual_extent = max(np.max(np.abs(low_env)), np.max(np.abs(high_env))) * 1.1
    if residual_kind in ("pearson", "sigma"):
        residual_lim = max(residual_extent, 3.2)
    else:
        residual_lim = residual_extent

    axs[1].set_ylim(-residual_lim, residual_lim)
    axs[1].axhline(0, color="black", linestyle="--", alpha=0.5)
    if residual_kind in ("pearson", "sigma"):
        axs[1].axhline(-3, color="black", linestyle="--", alpha=0.5)
        axs[1].axhline(3, color="black", linestyle="--", alpha=0.5)
        axs[1].set_yticks([-3, 0, 3])
    axs[1].set_ylabel("Residuals \n" + residual_label)
    axs[1].set_xlabel("Energy [keV]")
    axs[0].set_ylabel(
        "Observed Spectrum \n"
        + r"[$\frac{\text{Counts}}{\text{keV} \text{s}}$]"
    )

    if legend:
        axs[0].legend(legend_list, legend_names, loc="upper right")
    axs[0].loglog()
    fig.align_ylabels()

    return fig
