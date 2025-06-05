import matplotlib.pyplot as plt
import numpy as np
import catppuccin
from catppuccin.extras.matplotlib import load_color
from scipy.stats import nbinom, norm
from xspec import Plot, AllData

PALETTE = catppuccin.PALETTE.latte

COLOR_CYCLE = [
    load_color(PALETTE.identifier, color)
    for color in ["sky", "teal", "green", "yellow", "peach", "maroon", "red", "pink", "mauve", "blue"][::-1]
]


SPECTRUM_COLOR = load_color(PALETTE.identifier, "blue")
SPECTRUM_DATA_COLOR = load_color(PALETTE.identifier, "overlay2")
BACKGROUND_DATA_COLOR = load_color(PALETTE.identifier, "text")

Plot.xAxis = 'keV'


def sigma_to_percentile_intervals(sigmas):
    intervals = []
    for sigma in sigmas:
        lower_bound = 100 * norm.cdf(-sigma)
        upper_bound = 100 * norm.cdf(sigma)
        intervals.append((lower_bound, upper_bound))
    return intervals


def error_bars_for_observed_data(observed_counts, sigma=1):
    r"""
    Compute the error bars for the observed data assuming a prior Gamma distribution

    Parameters:
        observed_counts: array of integer counts
        denominator: normalization factor (e.g. effective area)
        units: unit to convert to
        sigma: dispersion to use for quantiles computation

    Returns:
        y_observed: observed counts in the desired units
        y_observed_low: lower bound of the error bars
        y_observed_high: upper bound of the error bars
    """

    percentile = sigma_to_percentile_intervals([sigma])[0]

    y_observed = observed_counts
    y_observed_low = nbinom.ppf(percentile[0] / 100, observed_counts, 0.5)
    y_observed_high = nbinom.ppf(percentile[1] / 100, observed_counts, 0.5)

    return y_observed, y_observed_low, y_observed_high

def plot_ppc(solver, component_names=None, x_lim=None, y_lim=None, figsize=(12, 6), plot_background=False, legend=True):

    if component_names is None:
        raise ValueError("component_names must be specified")

    data = solver.posterior_predictions_convolved(component_names=component_names, nsamples=100,
                                                  plottype="counts")
    plt.close("all")

    if data.get("background") is not None:
        count_data = (data["data"] + data["background"]) * data["width"] * 2
    else:
        count_data = (data["data"]) * data["width"] * 2

    alpha_median = 0.7
    alpha_envelope = (0.15, 0.25)

    models = data['models']
    n_components = models.shape[1]

    low_energy, high_energy = data['bins'] - data['width'], data['bins'] + data['width']
    bin_edges = np.append(low_energy, high_energy[-1])

    area = get_effective_area()
    denominator = area * data['width'] * 2 * AllData(1).exposure

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=figsize, sharex=True, height_ratios=[4, 1])

    y_observed, y_observed_low, y_observed_high = error_bars_for_observed_data(count_data, sigma=1)
    y_observed, y_observed_low, y_observed_high = y_observed/denominator, y_observed_low/denominator, y_observed_high/denominator

    error_bar = axs[0].errorbar(
            np.sqrt(bin_edges[:-1] * bin_edges[1:]),
            y_observed,
            xerr=np.abs(np.stack([bin_edges[:-1], bin_edges[1:]]) - np.sqrt(bin_edges[:-1] * bin_edges[1:])),
            yerr=[
                np.maximum(y_observed - y_observed_low, 0),
                np.maximum(y_observed_high - y_observed, 0),
            ],
        linestyle="none",
        color=SPECTRUM_DATA_COLOR,
        alpha=0.8,
        capsize=2,
        zorder=10
    )

    legend_list = [error_bar]

    linestyles = ["solid"] + ["dashdot"] * (n_components - 1)

    for i, (color, component_name, linestyle) in enumerate(zip(COLOR_CYCLE, component_names, linestyles)):

        local_component = np.random.poisson(models[:,i]*data['width']*2).astype(float)

        if plot_background and data.get("background") is not None and (i == 0):

            background = np.random.negative_binomial(
                np.repeat(solver._background[None, :], len(models), axis=0) + 1, 1 / 2
            ) * solver._backratio

            local_component += background

        local_component = local_component/denominator

        median = axs[0].stairs(np.median(local_component, axis=0), edges=bin_edges, color=color, alpha=alpha_median,zorder=100, linestyle=linestyle)

        low_band, high_band = np.percentile(local_component, [16, 84], axis=0)
        axs[0].stairs(high_band, edges=bin_edges, baseline=low_band, fill=True, alpha=alpha_envelope[1], color=color, zorder=80)

        low_band, high_band = np.percentile(local_component, [2.5, 97.5], axis=0)
        axs[0].stairs(high_band, edges=bin_edges, baseline=low_band, fill=True, alpha=alpha_envelope[0], color=color, zorder=60)

        # The legend cannot handle fill_between, so we pass a fill to get a fancy icon
        (envelope,) = axs[0].fill(np.nan, np.nan, alpha=alpha_envelope[-1], facecolor=color)

        legend_list.append((median, envelope))

    if plot_background and data.get("background") is not None:

        background = data["background"] * data["width"] * 2 / solver._backratio

        y_observed_bkg, y_observed_low_bkg, y_observed_high_bkg = error_bars_for_observed_data(background, sigma=1)
        y_observed_bkg, y_observed_low_bkg, y_observed_high_bkg = (
            y_observed_bkg*solver._backratio, y_observed_low_bkg*solver._backratio, y_observed_high_bkg*solver._backratio)
        y_observed_bkg, y_observed_low_bkg, y_observed_high_bkg = y_observed_bkg/denominator, y_observed_low_bkg/denominator, y_observed_high_bkg/denominator

        """
        error_bar_bkg = axs[0].errorbar(
            np.sqrt(bin_edges[:-1] * bin_edges[1:]),
            y_observed_bkg,
            xerr=np.abs(np.stack([bin_edges[:-1], bin_edges[1:]]) - np.sqrt(bin_edges[:-1] * bin_edges[1:])),
            yerr=[
                np.maximum(y_observed_bkg - y_observed_low_bkg, 0),
                np.maximum(y_observed_high_bkg - y_observed_bkg, 0),
            ],
            linestyle="none",
            color=BACKGROUND_DATA_COLOR,
            alpha=0.2,
            capsize=2,
            zorder=10,
            label="Background"
        )
        """

        #legend_list.append(error_bar_bkg)

        background_envelope = np.random.poisson(
				np.repeat(solver._background[None, :], 1000, axis=0)
			) * solver._backratio


        background_envelope = background_envelope/denominator

        median = axs[0].stairs(np.median(background_envelope, axis=0), edges=bin_edges, color=BACKGROUND_DATA_COLOR, alpha=alpha_median, zorder=100, linestyle="dotted")
        low_band, high_band = np.percentile(background_envelope, [16, 84], axis=0)
        envelope = axs[0].stairs(high_band, edges=bin_edges, baseline=low_band, fill=True, alpha=alpha_envelope[1], color=BACKGROUND_DATA_COLOR, zorder=80)
        legend_list.append((median, envelope))

    total = np.random.poisson(models[:, 0]*data['width']*2).astype(float)

    if plot_background and data.get("background") is not None:
        background = np.random.negative_binomial(
            np.repeat(solver._background[None, :], len(models), axis=0) + 1, 1 / 2
        ) * solver._backratio

        total += background

    total = total/denominator

    residuals = (total-y_observed)/(np.percentile(total, 84, axis=0) - np.percentile(total, 16, axis=0))

    axs[1].stairs(np.median(residuals, axis=0), edges=bin_edges, color=SPECTRUM_COLOR, label="Total", alpha=alpha_median, zorder=100)
    axs[1].stairs(np.percentile(residuals, 84, axis=0), edges=bin_edges, baseline=np.percentile(residuals, 16, axis=0), fill=True, alpha=alpha_envelope[1], color=SPECTRUM_COLOR, zorder=80)
    axs[1].stairs(np.percentile(residuals, 97.5, axis=0), edges=bin_edges, baseline=np.percentile(residuals, 2.5, axis=0), fill=True, alpha=alpha_envelope[0], color=SPECTRUM_COLOR, zorder=60)


    if x_lim is not None:
        axs[0].set_xlim(*x_lim)

    if y_lim is not None:
        axs[0].set_ylim(*y_lim)

    residual_lim = 3.2#max(np.max(np.abs(residuals))*1.05, 3.2)

    legend_names = ["Data"] + component_names + (["Background"] if data.get('background') is not None else [])

    axs[1].set_ylim(-residual_lim, residual_lim)
    axs[1].axhline(0, color="black", linestyle="--", alpha=0.5)
    axs[1].axhline(-3, color="black", linestyle="--", alpha=0.5)
    axs[1].axhline(3, color="black", linestyle="--", alpha=0.5)
    axs[1].set_ylabel("Residuals \n"+r"$\left[ \sigma \right]$")
    axs[1].set_xlabel("Energy [keV]")
    axs[0].set_ylabel("Observed Spectrum \n" + r"[$\frac{\text{Counts}}{\text{cm}^2  \text{keV} \text{s}}$]")
    axs[1].set_yticks([-3, 0, 3])

    if legend:
        axs[0].legend(legend_list, legend_names, loc="upper right")
    axs[0].loglog()
    fig.align_ylabels()

    return fig
    #plt.savefig(outputfiles_basename + 'convolved_posterior.pdf', bbox_inches='tight')

def get_effective_area():

    Plot.area = True
    Plot("ldata")
    res_divided_by_area = Plot.model()

    Plot.area = False
    Plot("ldata")
    res_not_divided_by_area = Plot.model()

    return np.asarray(res_not_divided_by_area)/np.asarray(res_divided_by_area)