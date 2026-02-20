import itertools
import matplotlib.pyplot as plt
import numpy as np
import catppuccin
from catppuccin.extras.matplotlib import load_color
from scipy.stats import nbinom, norm
from xspec import Plot, AllData, AllModels
from ..xspec import SpectrumState

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

def sigma_to_percentile_intervals(sigmas):
    intervals = []
    for sigma in sigmas:
        lower_bound = 100 * norm.cdf(-sigma)
        upper_bound = 100 * norm.cdf(sigma)
        intervals.append((lower_bound, upper_bound))
    return intervals


def poisson_error_bars(observed_counts, sigma=1):
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


def plot_ppc(
    solver,
    sampler,
    x_lim=None,
    y_lim=None,
    figsize=(12, 6),
    plot_background=False,
    plot_models=True,
    plot_components=True,
    legend=True,
    n_samples=100,
):
    r"""Plot posterior predictive spectra with residuals.

    Parameters:
        solver: Solver instance exposing ``samplers`` and ``simulate``.
        sampler (str): Name of the sampler stored in ``solver.samplers``.
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

    Returns:
        matplotlib.figure.Figure: Figure containing spectrum and residual
            panels.
    """

    # TODO : check for background
    # TODO : let the user chose the components to plot
    # TODO : add ARF division when relevant
    # TODO : rebinning / grouping

    Plot.xAxis = "keV"
    state = SpectrumState(1)
    parameters = solver.samplers[sampler].sample((n_samples,))
    data = solver.simulate(parameters, return_kind="models_and_components")

    bin_edges = state.bin_edges
    bin_center = state.bin_center
    bin_edges_1d = state.bin_edges_1d
    bin_width = state.bin_width
    denominator = bin_width

    legend_names = []
    legend_list = []

    fig, axs = plt.subplots(
        nrows=2, ncols=1, figsize=figsize, sharex=True, height_ratios=[4, 1]
    )

    ### OBSERVED SPECTRUM
    error_bar = plot_poisson_error_bars(bin_center, bin_edges, state.observed_counts, denominator, axs[0], SPECTRUM_DATA_COLOR)
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

    total_model = total_model / denominator
    y_observed = state.observed_counts / denominator

    divider = np.percentile(total_model, 84, axis=0) - np.percentile(total_model, 16, axis=0)
    residuals = (total_model - y_observed) / np.where(divider > 0, divider, 1.0)

    plot_median_and_bands(bin_edges_1d, residuals, np.ones_like(state.observed_counts), axs[1], color=SPECTRUM_COLOR)

    if x_lim is None:
        x_lim = (np.min(state.bin_edges_1d), np.max(state.bin_edges_1d))

    axs[0].set_xlim(*x_lim)

    if y_lim is not None:
        axs[0].set_ylim(*y_lim)

    residual_lim = 3.2  # max(np.max(np.abs(residuals))*1.05, 3.2)

    axs[1].set_ylim(-residual_lim, residual_lim)
    axs[1].axhline(0, color="black", linestyle="--", alpha=0.5)
    axs[1].axhline(-3, color="black", linestyle="--", alpha=0.5)
    axs[1].axhline(3, color="black", linestyle="--", alpha=0.5)
    axs[1].set_ylabel("Residuals \n" + r"$\left[ \sigma \right]$")
    axs[1].set_xlabel("Energy [keV]")
    axs[0].set_ylabel(
        "Observed Spectrum \n"
        + r"[$\frac{\text{Counts}}{\text{keV} \text{s}}$]"
    )
    axs[1].set_yticks([-3, 0, 3])

    if legend:
        axs[0].legend(legend_list, legend_names, loc="upper right")
    axs[0].loglog()
    fig.align_ylabels()

    return fig
