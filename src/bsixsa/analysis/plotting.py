"""Predictive checks of the spectra loaded in XSPEC.

[`plot_predictive_coverage`][bsixsa.analysis.plotting.plot_predictive_coverage]
compares the data with spectra simulated from the prior or the posterior of a
[`Solver`][bsixsa.solver.Solver]. It draws one column per spectrum: the
spectrum above its residuals. `min_counts` and `grouping` merge channels for
the figure only, never for the fit.
"""

import itertools
import os
from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
from scipy.stats import nbinom, norm

from ..xspec import SpectrumState, loaded_spectra

if TYPE_CHECKING:
    from ..solver import Solver

# Catppuccin Latte colours (https://github.com/catppuccin/catppuccin).
_TOTAL_MODEL_COLOR = "#1e66f5"  # blue: total model and residuals
_DATA_COLOR = "#7c7f93"  # overlay2: observed spectrum
# One colour per model or component curve, cycled.
_CURVE_COLORS = (
    "#8839ef",  # mauve
    "#ea76cb",  # pink
    "#d20f39",  # red
    "#e64553",  # maroon
    "#fe640b",  # peach
    "#df8e1d",  # yellow
    "#40a02b",  # green
    "#179299",  # teal
    "#04a5e5",  # sky
)
_COMPONENT_HATCHES = ("//", r"\\", "||", "--", "++", "xx", "oo", "OO", "..", "**")
_COMPONENT_EDGE_COLOR = (1 / 3, 1 / 3, 1 / 3, 1)  # colour of the component hatches
_MEDIAN_ALPHA = 0.7
_BAND_68 = (16, 84)  # percentiles bounding the central 68%
_BAND_95 = (2.5, 97.5)  # percentiles bounding the central 95%

_SPECTRUM_YLABEL = "Observed Spectrum\n[counts keV$^{-1}$]"
_UNNAMED_MODEL_LABEL = "unnamed model"
_RESIDUAL_KINDS = ("pearson", "sigma", "raw")
_RESIDUAL_MARGIN = 1.1  # 10% margin around the residuals
_MIN_SIGMA_EXTENT = 3.2  # just beyond the +/-3 sigma guide lines
_SINGLE_FIGSIZE = (12, 6)
_PANEL_SIZE = (6, 6)  # per spectrum, when several are plotted
_EDGE_RTOL = 1e-6  # channels closer than this share an edge (no ignored range between them)


def poisson_error_bars(
    observed_counts: np.ndarray, sigma: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Range of counts expected if the observation were repeated.

    For an observed count $k$, the bounds are the central interval of the
    negative binomial $\mathrm{NB}(k, 1/2)$: the distribution of a repeated
    Poisson count when the source rate has a $\mathrm{Gamma}(k, 1)$ posterior.
    Bins with zero counts get NaN bounds.

    Parameters:
        observed_counts: Observed counts per bin.
        sigma: Coverage in Gaussian standard deviations (`1` gives the central 68%).

    Returns:
        The observed counts (unchanged), the lower bounds and the upper bounds.
    """
    lower_quantile, upper_quantile = norm.cdf([-sigma, sigma])
    low = nbinom.ppf(lower_quantile, observed_counts, 0.5)
    high = nbinom.ppf(upper_quantile, observed_counts, 0.5)
    return observed_counts, low, high


def _check_regrouping(min_counts: int | None, grouping: int | None) -> None:
    """Refuse `min_counts` together with `grouping`, and a `grouping` below 1."""
    if min_counts is not None and grouping is not None:
        raise ValueError("min_counts and grouping are mutually exclusive; use one or the other.")
    if grouping is not None and grouping < 1:
        raise ValueError(f"grouping must be a positive number of channels, got {grouping}.")


def _contiguous_runs(low: np.ndarray, high: np.ndarray) -> list[tuple[int, int]]:
    """`(start, stop)` of each run of channels with no ignored energy range between neighbours."""
    adjacent = np.isclose(high[:-1], low[1:], rtol=_EDGE_RTOL, atol=0.0) | np.isclose(
        low[:-1], high[1:], rtol=_EDGE_RTOL, atol=0.0
    )
    bounds = [0, *(np.flatnonzero(~adjacent) + 1).tolist(), len(low)]
    return list(zip(bounds[:-1], bounds[1:]))


def _run_starts(observed_counts: np.ndarray, min_counts: int | None, grouping: int | None) -> list[int]:
    """First channel of each display bin of one run of channels."""
    if grouping is not None:
        return list(range(0, len(observed_counts), grouping))
    starts, running = [0], 0
    for channel, counts in enumerate(observed_counts):
        running += counts
        if running >= min_counts:
            starts.append(channel + 1)
            running = 0
    # Drop the last start: it opens nothing or an under-filled tail.
    if len(starts) > 1:
        starts.pop()
    return starts


def _display_bin_starts(
    observed_counts: np.ndarray,
    min_counts: int | None,
    grouping: int | None,
    runs: Sequence[tuple[int, int]] | None = None,
) -> np.ndarray | None:
    """First channel of each display bin, or None when no regrouping is asked.

    Channels are merged within each `(start, stop)` run of `runs` (the whole
    spectrum by default), never across an ignored energy range. With
    `min_counts`, an under-filled last bin of a run is merged into the previous one.
    """
    _check_regrouping(min_counts, grouping)
    if min_counts is None and grouping is None:
        return None
    if runs is None:
        runs = [(0, len(observed_counts))]
    starts = [
        start + first
        for start, stop in runs
        for first in _run_starts(observed_counts[start:stop], min_counts, grouping)
    ]
    return np.asarray(starts)


class _DisplayBins:
    """The channels of one spectrum merged into display bins and sorted by energy.

    Attributes:
        low: Lower energy of each display bin in keV, increasing.
        high: Upper energy of each display bin in keV.
        edges: Edges for `Axes.stairs`, with an extra stair over each ignored
            energy range (drawn empty: see `stairs`).
    """

    def __init__(
        self, bin_energies: np.ndarray, observed_counts: np.ndarray, min_counts: int | None, grouping: int | None
    ):
        low, high = bin_energies[:, 0], bin_energies[:, 1]
        starts = _display_bin_starts(observed_counts, min_counts, grouping, _contiguous_runs(low, high))
        self._starts = np.arange(len(low)) if starts is None else starts
        low, high = np.minimum.reduceat(low, self._starts), np.maximum.reduceat(high, self._starts)
        # Grating spectra list their channels from high to low energy.
        self._order = np.argsort(low, kind="stable")
        self.low, self.high = low[self._order], high[self._order]
        gap_before = np.r_[False, ~np.isclose(self.low[1:], self.high[:-1], rtol=_EDGE_RTOL, atol=0.0)]
        self._stair = np.arange(len(self.low)) + np.cumsum(gap_before)
        self.edges = np.empty(self._stair[-1] + 2)
        self.edges[self._stair] = self.low
        self.edges[self._stair + 1] = self.high

    @property
    def width(self) -> np.ndarray:
        """Width of each display bin in keV."""
        return self.high - self.low

    def sum(self, counts: np.ndarray) -> np.ndarray:
        """Sum per-channel counts (last axis) into the display bins, in energy order."""
        return np.add.reduceat(counts, self._starts, axis=-1)[..., self._order]

    def stairs(self, values: np.ndarray) -> np.ndarray:
        """Per-display-bin values (last axis) laid on `edges`, NaN over the ignored ranges."""
        laid = np.full((*values.shape[:-1], len(self.edges) - 1), np.nan)
        laid[..., self._stair] = values
        return laid


def _plotted_spectra(spectra: int | Sequence[int] | None) -> tuple[list[SpectrumState], list[SpectrumState]]:
    """All loaded spectra and the ones to plot (all of them by default).

    Raises:
        ValueError: If `spectra` names no spectrum or one that is not loaded.
    """
    loaded = loaded_spectra()
    if spectra is None:
        return loaded, loaded
    indices = [spectra] if isinstance(spectra, (int, np.integer)) else list(spectra)
    if not indices or any(not 1 <= index <= len(loaded) for index in indices):
        raise ValueError(
            f"spectra must be XSPEC spectrum indices between 1 and {len(loaded)} "
            f"(as in xspec.AllData(index)), got {spectra!r}."
        )
    return loaded, [loaded[index - 1] for index in indices]


def _component_selection(
    plot_components: bool | str | Sequence[str], spectra: Sequence[SpectrumState]
) -> bool | frozenset[str]:
    """`plot_components` as `True`, `False` or the set of models whose components are drawn.

    Raises:
        ValueError: If it names a model that is not folded into the plotted spectra.
    """
    if isinstance(plot_components, (bool, np.bool_)):
        return bool(plot_components)
    selected = frozenset([plot_components] if isinstance(plot_components, str) else plot_components)
    available = {name for state in spectra for name in state.sources.values()}
    unknown = selected - available
    if unknown:
        raise ValueError(
            f"plot_components names models that are not in the plotted spectra: {sorted(unknown)}; "
            f"their models are {sorted(available)} ('' is the unnamed model)."
        )
    return selected


def _component_model(key: str, model_names: Iterable[str]) -> str | None:
    """Model of a component key `"<model>_<component>"`: the longest model name that prefixes it."""
    owners = [name for name in model_names if key.startswith(name + "_")]
    return max(owners, key=len) if owners else None


def _model_curves(
    model_counts: dict[str, np.ndarray],
    component_counts: dict[str, np.ndarray],
    models: Sequence[str],
    plot_models: bool,
    plot_components: bool | frozenset[str],
) -> Iterator[tuple[tuple[str, str], str, np.ndarray]]:
    """Yield `(curve id, label, counts)` per curve of one spectrum, in legend order.

    Each model of `models`, with its own curve when there are several and
    `plot_models` is set, followed by its components when `plot_components`
    is `True` or names it.
    """
    for model in models:
        if plot_models and len(models) > 1:
            yield ("model", model), model or _UNNAMED_MODEL_LABEL, model_counts[model]
        if plot_components is True or (plot_components and model in plot_components):
            for key, counts in component_counts.items():
                if _component_model(key, model_counts) == model:
                    # The unnamed model gives keys like "_bapec".
                    yield ("component", key), key.lstrip("_"), counts


def _curve_styles(curve_ids: Iterable[tuple[str, str]]) -> dict[tuple[str, str], tuple[str, str | None]]:
    """Colour and hatch (`None` for a model curve) of each curve, the same in every panel."""
    colors = itertools.cycle(_CURVE_COLORS)
    hatches = itertools.cycle(_COMPONENT_HATCHES)
    styles = {}
    for curve_id in curve_ids:
        if curve_id not in styles:
            styles[curve_id] = (next(colors), next(hatches) if curve_id[0] == "component" else None)
    return styles


def _make_figure(n_columns: int, figsize: tuple[float, float] | None) -> tuple[Figure, list[Axes], list[Axes]]:
    """Return the figure, the spectrum axes and the residual axes, one per column."""
    if figsize is None:
        figsize = _SINGLE_FIGSIZE if n_columns == 1 else (_PANEL_SIZE[0] * n_columns, _PANEL_SIZE[1])
    fig, axes = plt.subplots(
        nrows=2, ncols=n_columns, figsize=figsize, sharex="col", height_ratios=[4, 1], squeeze=False
    )
    return fig, list(axes[0]), list(axes[1])


def _panel_title(state: SpectrumState) -> str:
    """Title of the column of a spectrum: its XSPEC index and file name."""
    return f"Spectrum {state.index}: {os.path.basename(state.spectrum.fileName)}"


def _draw_observed_spectrum(ax: Axes, bins: _DisplayBins, counts: np.ndarray) -> ErrorbarContainer:
    """Draw the observed counts per keV with the error bars of `poisson_error_bars`, at geometric bin centres."""
    center = np.sqrt(bins.low * bins.high)
    per_kev, low, high = (values / bins.width for values in poisson_error_bars(counts))
    return ax.errorbar(
        center,
        per_kev,
        xerr=[center - bins.low, bins.high - center],
        yerr=[np.maximum(per_kev - low, 0), np.maximum(high - per_kev, 0)],
        linestyle="none",
        color=_DATA_COLOR,
        alpha=0.8,
        capsize=2,
        zorder=10,
    )


def _draw_median_and_bands(
    ax: Axes,
    bins: _DisplayBins,
    samples: np.ndarray,
    color: str,
    alpha_68: float = 0.25,
    alpha_95: float = 0.15,
    **patch_kwargs,
) -> tuple:
    """Draw the median of `samples` (shape `(n_samples, n_display_bins)`) and its central 68% and 95% bands.

    Returns:
        A legend handle: the median line paired with the band swatch.
    """
    median = ax.stairs(
        bins.stairs(np.median(samples, axis=0)),
        edges=bins.edges,
        color=color,
        alpha=_MEDIAN_ALPHA,
        zorder=100,
        linestyle="solid",
    )
    for percentiles, alpha, zorder in ((_BAND_68, alpha_68, 80), (_BAND_95, alpha_95, 60)):
        low, high = np.percentile(samples, percentiles, axis=0)
        ax.stairs(
            bins.stairs(high),
            edges=bins.edges,
            baseline=bins.stairs(low),
            fill=True,
            alpha=alpha,
            color=color,
            zorder=zorder,
            **patch_kwargs,
        )
    # Invisible patch used only as the legend swatch of the bands.
    (swatch,) = ax.fill(np.nan, np.nan, alpha=alpha_68, facecolor=color, **patch_kwargs)
    return median, swatch


def _finish_residual_panel(ax: Axes, extent: float, label: str | None, in_sigma_units: bool) -> None:
    """Set the range (+/-`extent` plus a margin), guide lines and labels of the residual panel."""
    limit = extent * _RESIDUAL_MARGIN
    if in_sigma_units:
        limit = max(limit, _MIN_SIGMA_EXTENT)
    ax.set_ylim(-limit, limit)
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    if in_sigma_units:
        ax.axhline(-3, color="black", linestyle="--", alpha=0.5)
        ax.axhline(3, color="black", linestyle="--", alpha=0.5)
        ax.set_yticks([-3, 0, 3])
    if label is not None:
        ax.set_ylabel("Residuals\n" + label)
    ax.set_xlabel("Energy [keV]")


class _EnergyFormatter(ticker.Formatter):
    """Plain-number labels ("0.5", "2") on a log energy axis, at 1, 2 and 5 times a power of ten.

    An axis spanning less than a factor 3 labels every tick.
    """

    def __call__(self, x: float, pos: int | None = None) -> str:
        low, high = sorted(self.axis.get_view_interval())
        mantissa = x / 10 ** np.floor(np.log10(x))
        if high > 3 * low and not np.isclose(mantissa, [1, 2, 5, 10], rtol=1e-6).any():
            return ""
        return f"{x:g}"

    def format_data_short(self, value: float) -> str:
        """The cursor position shown by interactive backends."""
        return f"{value:.4g}"


def _finish_spectrum_panel(
    ax: Axes,
    bins: _DisplayBins,
    x_lim: tuple[float, float] | None,
    y_lim: tuple[float, float] | None,
    with_ylabel: bool,
) -> None:
    """Set the limits, the y label and the log-log scales of the spectrum panel (and of its residual panel)."""
    ax.set_xlim(*(x_lim if x_lim is not None else (bins.edges[0], bins.edges[-1])))
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    if with_ylabel:
        ax.set_ylabel(_SPECTRUM_YLABEL)
    ax.loglog()
    # Setting the scale resets the tick labels of every axis sharing x.
    for shared in ax.get_shared_x_axes().get_siblings(ax):
        shared.xaxis.set_major_formatter(_EnergyFormatter())
        shared.xaxis.set_minor_formatter(_EnergyFormatter())


def plot_predictive_coverage(
    solver: "Solver",
    distribution: str = "posterior",
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    *,
    plot_models: bool = True,
    plot_components: bool | str | Sequence[str] = True,
    legend: bool = True,
    n_samples: int = 100,
    min_counts: int | None = None,
    grouping: int | None = None,
    residual_kind: Literal["pearson", "sigma", "raw"] = "pearson",
    spectra: int | Sequence[int] | None = None,
) -> Figure:
    r"""Compare the observed spectra with spectra simulated from the prior or the posterior.

    Each of `n_samples` parameter vectors drawn from `distribution` is folded
    and Poisson-sampled into a simulated spectrum $M_i$; the line and bands
    show their median and central 68% and 95%. Before fitting, use `"prior"`
    to check that the prior can produce spectra like the data; after fitting,
    the data should lie within the `"posterior"` bands. Each spectrum gets a
    column, with the models folded into it.

    The residual panels show the same bands for a residual $r_i$ between the
    observed counts $D$ and each $M_i$, selected by `residual_kind`:

    * `"pearson"` (default): Poisson-normalised residuals.

    $$
    r_i = \frac{D - M_i}{\sqrt{M_i}}
    $$

    * `"sigma"`: model minus data, normalised by the width of the central
      68% of the $M_i$ ($P_{16}$ to $P_{84}$); a broad prior gives small
      residuals.

    $$
    r_i = \frac{M_i - D}{P_{84} - P_{16}}
    $$

    * `"raw"`: un-normalised residuals in counts per keV, with $\Delta E$
      the bin width.

    $$
    r_i = \frac{D - M_i}{\Delta E}
    $$

    Ignored energy ranges are left empty; backgrounds and the effective area
    are not accounted for.

    Parameters:
        solver: The solver whose prior or posterior is checked.
        distribution: `"prior"` or `"posterior"`; see
            [`Solver.sample_parameters`][bsixsa.solver.Solver.sample_parameters].
            Defaults to `"posterior"`.
        x_lim: Energy range in keV. Defaults to that of the noticed channels.
        y_lim: y-axis limits of the spectrum panels, in counts per keV.
        figsize: Figure size in inches. Defaults to `(12, 6)` for one
            spectrum and 6 by 6 inches per spectrum otherwise.
        plot_models: Draw one band per model when a spectrum has several.
            Defaults to `True`.
        plot_components: Draw one band per additive component of every model
            (`True`), of none (`False`), or of the models named in a list
            (`''` for the unnamed model). Defaults to `True`.
        legend: Show the legends. Defaults to `True`.
        n_samples: Number of simulated spectra. Defaults to `100`.
        min_counts: Merge adjacent channels until each displayed bin holds at
            least this many counts. Exclusive with `grouping`.
        grouping: Merge every `grouping` adjacent channels. Exclusive with `min_counts`.
        residual_kind: `"pearson"`, `"sigma"` or `"raw"`. Defaults to `"pearson"`.
        spectra: XSPEC index (or list of indices) of the spectra to plot.
            Defaults to every loaded spectrum.

    Returns:
        The figure.

    Raises:
        ValueError: If `residual_kind`, `min_counts`, `grouping`, `spectra` or
            `plot_components` is invalid (checked before any fold).
        KeyError: If `distribution` is not available, e.g. `"posterior"`
            before `solver.run()`.
    """
    if residual_kind not in _RESIDUAL_KINDS:
        raise ValueError(
            f"residual_kind must be 'pearson', 'sigma' or 'raw' (see the docstring), got {residual_kind!r}."
        )
    _check_regrouping(min_counts, grouping)
    loaded, states = _plotted_spectra(spectra)
    components = _component_selection(plot_components, states)

    parameters = solver.sample_parameters(n_samples, distribution)
    simulated = solver.simulate(parameters, return_kind="models_and_components")
    # The folds cover every loaded spectrum, concatenated in spectrum order.
    offsets = np.cumsum([0] + [len(state.observed_counts) for state in loaded])
    if simulated["total_model_counts"].shape[-1] != offsets[-1]:
        raise ValueError(
            f"The solver simulates {simulated['total_model_counts'].shape[-1]} bins, but the loaded spectra "
            f"have {offsets[-1]} noticed channels: plot in the XSPEC session the solver was built in."
        )

    panels = []
    for state in states:
        segment = slice(offsets[state.index - 1], offsets[state.index])
        model_counts = {name: counts[:, segment] for name, counts in simulated["model_counts"].items()}
        component_counts = {name: counts[:, segment] for name, counts in simulated["component_counts"].items()}
        folded_into = set(state.sources.values())
        models = [name for name in model_counts if name in folded_into]
        curves = list(_model_curves(model_counts, component_counts, models, plot_models, components))
        panels.append((state, segment, curves))
    styles = _curve_styles(curve_id for *_, curves in panels for curve_id, _, _ in curves)

    fig, spectrum_axes, residual_axes = _make_figure(len(panels), figsize)
    component_style = {"alpha_68": 0.1, "alpha_95": 0.05, "hatch_linewidth": 1, "edgecolor": _COMPONENT_EDGE_COLOR}
    for column, ((state, segment, curves), ax_spec, ax_res) in enumerate(zip(panels, spectrum_axes, residual_axes)):
        bins = _DisplayBins(state.bin_energies, state.observed_counts, min_counts, grouping)
        observed_counts = bins.sum(state.observed_counts)
        handles = [_draw_observed_spectrum(ax_spec, bins, observed_counts)]
        labels = ["Observed Spectrum"]

        # Poisson noise from NumPy's global generator (np.random.seed fixes it, not the parameter draws).
        # Model counts can dip below zero (e.g. a negative-normalisation line): clip them first.
        total_model = np.random.poisson(np.clip(bins.sum(simulated["total_model_counts"][:, segment]), 0, None))
        handles.append(_draw_median_and_bands(ax_spec, bins, total_model / bins.width, _TOTAL_MODEL_COLOR))
        labels.append("Total model")
        for curve_id, label, counts in curves:
            color, hatch = styles[curve_id]
            style = {} if hatch is None else {"hatch": hatch, **component_style}
            simulated_counts = np.random.poisson(np.clip(bins.sum(counts), 0, None))
            handles.append(_draw_median_and_bands(ax_spec, bins, simulated_counts / bins.width, color, **style))
            labels.append(label)

        if residual_kind == "pearson":
            safe_model = np.where(total_model > 0, total_model, 1.0)
            residuals = (observed_counts - total_model) / np.sqrt(safe_model)
            residual_label = r"$\frac{D - M}{\sqrt{M}}$"
        elif residual_kind == "sigma":
            total_model_per_kev = total_model / bins.width
            width_68 = np.percentile(total_model_per_kev, 84, axis=0) - np.percentile(total_model_per_kev, 16, axis=0)
            residuals = (total_model_per_kev - observed_counts / bins.width) / np.where(width_68 > 0, width_68, 1.0)
            residual_label = r"$\left[ \sigma \right]$"
        else:
            residuals = (observed_counts - total_model) / bins.width
            residual_label = "[counts keV$^{-1}$]"
        _draw_median_and_bands(ax_res, bins, residuals, _TOTAL_MODEL_COLOR)
        # The panel range covers the 95% band of the residuals.
        extent = np.max(np.abs(np.percentile(residuals, _BAND_95, axis=0)))
        _finish_residual_panel(
            ax_res, extent, residual_label if column == 0 else None, in_sigma_units=residual_kind != "raw"
        )

        _finish_spectrum_panel(ax_spec, bins, x_lim, y_lim, with_ylabel=column == 0)
        if len(panels) > 1:
            ax_spec.set_title(_panel_title(state))
        if legend:
            ax_spec.legend(handles, labels, loc="upper right", fontsize="small" if len(panels) > 1 else None)
    fig.align_ylabels()
    return fig
