"""Fold parameter vectors through XSPEC and read the fit statistic and model counts."""

import os
import re
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import xspec
from tqdm.auto import tqdm

from .convenience import XSilence, first_model_copy, iter_components

RETURN_KINDS = ("cstat", "full_model_counts", "models_and_components")


class SpectrumState:
    """Read-only view of one loaded spectrum and of the current model, in counts per noticed channel.

    The models of a spectrum are those folded into it: the models whose
    source has a response for this spectrum (a background model attached to
    another spectrum is left out).

    Attributes:
        index: XSPEC index of the spectrum, as in `xspec.AllData(index)`.
        spectrum: The PyXspec spectrum.
    """

    def __init__(self, spectrum_index: int = 1):
        self.index = spectrum_index
        self.spectrum: xspec.Spectrum = xspec.AllData(spectrum_index)

    @property
    def exposure(self) -> float:
        """Exposure time of the spectrum, in seconds."""
        return float(self.spectrum.exposure)

    @property
    def observed_counts(self) -> np.ndarray:
        """Observed counts per noticed channel (rate times exposure, not rounded)."""
        return np.asarray(self.spectrum.values) * self.exposure

    @property
    def n_bins(self) -> int:
        """Number of noticed channels."""
        return len(self.spectrum.values)

    @property
    def bin_energies(self) -> np.ndarray:
        """Lower and upper energy of each noticed channel in keV, shape `(n_bins, 2)`.

        An ignored energy range leaves a gap between two consecutive channels.
        """
        return np.asarray(self.spectrum.energies, dtype=np.float64).reshape(-1, 2)

    @property
    def sources(self) -> dict[int, str]:
        """Source number and model name (`''` for the unnamed model) of each model folded into this spectrum."""
        return {source: name for source, name in xspec.AllModels.sources.items() if self._has_response(source)}

    def _has_response(self, source: int) -> bool:
        try:
            return self.spectrum.multiresponse[source - 1] is not None
        except Exception:  # PyXspec: "No response is assigned to source ..."
            return False

    @property
    def component_counts(self) -> dict[str, np.ndarray]:
        """Counts per channel of each additive component, keyed `"<model name>_<component name>"`.

        Keys look like `"_powerlaw_3"` or `"nxb_bbodyrad"`. Models with a single
        additive component are left out (XSPEC reports no per-component plot
        for them); their counts are in
        [`model_counts`][bsixsa.xspec.SpectrumState.model_counts].

        Read from XSPEC's plot (sets `Plot.device = "/null"` and
        `Plot.add = True`), which must keep XSPEC's default settings: one plot
        group per spectrum, no `Plot.setRebin`, `Plot.area` off.

        Raises:
            RuntimeError: If the plot does not show the expected components.
        """
        with _counts_plot():
            return self._plotted_component_counts()

    def _plotted_component_counts(self) -> dict[str, np.ndarray]:
        """Read `component_counts` from the plot drawn by `_counts_plot`."""
        data_group = self.spectrum.dataGroup
        component_counts = {}
        # XSPEC numbers the components of a plot group consecutively across its models.
        plot_index = 1
        for model_name in self.sources.values():
            additive = list(iter_components(xspec.AllModels(data_group, model_name), additive_only=True))
            if len(additive) < 2:
                continue
            for component in additive:
                component_counts[f"{model_name}_{component.name}"] = np.asarray(
                    xspec.Plot.addComp(plot_index, self.index)
                )
                plot_index += 1
        n_plotted = xspec.Plot.nAddComps(self.index)
        if n_plotted != len(component_counts) or any(len(c) != self.n_bins for c in component_counts.values()):
            raise RuntimeError(
                f"XSPEC's plot of spectrum {self.index} shows {n_plotted} additive components, where bsixsa "
                f"expects {len(component_counts)} with {self.n_bins} channels each. Restore XSPEC's default "
                "plot settings (one plot group per spectrum, no Plot.setRebin, Plot.area off)."
            )
        return component_counts

    @property
    def model_counts(self) -> dict[str, np.ndarray]:
        """Folded counts per channel of each model folded into this spectrum, keyed by model name."""
        data_group = self.spectrum.dataGroup
        return {
            model_name: np.asarray(xspec.AllModels(data_group, model_name).folded(self.index)) * self.exposure
            for model_name in self.sources.values()
        }

    @property
    def total_model_counts(self) -> np.ndarray:
        """Folded counts per channel summed over the models of this spectrum."""
        return sum(self.model_counts.values(), np.zeros(self.n_bins))


def loaded_spectra() -> list[SpectrumState]:
    """One [`SpectrumState`][bsixsa.xspec.SpectrumState] per loaded spectrum, in XSPEC index order."""
    return [SpectrumState(index) for index in range(1, xspec.AllData.nSpectra + 1)]


@contextmanager
def _counts_plot() -> Iterator[None]:
    """Draw XSPEC's counts plot of every spectrum, with the additive components, on the null device."""
    previous_x_axis = xspec.Plot.xAxis
    xspec.Plot.device = "/null"
    xspec.Plot.add = True
    # Plot("counts") gives counts per unit of the x axis: only the channel
    # axis gives counts per channel, the unit of model_counts.
    xspec.Plot.xAxis = "channel"
    try:
        xspec.Plot("counts")
        yield
    finally:
        xspec.Plot.xAxis = previous_x_axis


def _concatenate_spectra(per_spectrum: list[dict[str, np.ndarray]], n_bins: list[int]) -> dict[str, np.ndarray]:
    """Join per-spectrum counts into arrays over the bins of all spectra, zero where a spectrum lacks a key."""
    keys = dict.fromkeys(key for counts in per_spectrum for key in counts)
    return {
        key: np.concatenate([counts.get(key, np.zeros(n)) for counts, n in zip(per_spectrum, n_bins)])
        for key in keys
    }


def _model_counts_of_all_spectra(with_components: bool) -> dict[str, Any]:
    """Model counts of every loaded spectrum, concatenated in spectrum order (see `parallel_folding`)."""
    spectra = loaded_spectra()
    n_bins = [state.n_bins for state in spectra]
    model_counts = [state.model_counts for state in spectra]
    outputs = {
        "total_model_counts": np.concatenate(
            [sum(counts.values(), np.zeros(n)) for counts, n in zip(model_counts, n_bins)]
        )
    }
    if with_components:
        with _counts_plot():
            component_counts = [state._plotted_component_counts() for state in spectra]
        outputs["model_counts"] = _concatenate_spectra(model_counts, n_bins)
        outputs["component_counts"] = _concatenate_spectra(component_counts, n_bins)
    return outputs


def set_parameters(
    values: Sequence[float | str],
    parameter_indices: Sequence[int],
    model_names: Sequence[str],
) -> None:
    """Load one parameter vector into the live XSPEC session.

    Use it to put a fitted point back into XSPEC for its own plot, flux and
    eqwidth commands, e.g.
    `set_parameters(fit_result.best_fit, solver.parameter_indices, solver.model_names)`.

    Parameters:
        values: Physical parameter values in prior order; strings are passed to XSPEC unchanged.
        parameter_indices: XSPEC index of each parameter within its model.
        model_names: Model name of each parameter (`''` for the unnamed model).
    """
    by_model = {}
    for value, index, model_name in zip(values, parameter_indices, model_names):
        by_model.setdefault(model_name, {})[index] = value if isinstance(value, str) else float(value)
    xspec.AllModels.setPars(
        *chain.from_iterable((first_model_copy(name), pars) for name, pars in by_model.items())
    )


# `model  [<source>:<name>] <expression>`; group 1 is the name.
_MODEL_LINE = re.compile(r"^model\s+(?:\d+:(\S+))?")


def get_model_block(lines: Sequence[str]) -> dict[tuple[str, int], tuple[str, int]]:
    """Find the parameter lines of every model in an `.xcm` file.

    Each `model` line of an `.xcm` written by `Xset.save(path, info="m")` is
    followed by one line per parameter; the model section ends at `bayes`.

    Returns:
        `{(model_name, parameter_index): (line, line_number)}`, with
        `line_number` 0-based.
    """
    parameter_lines = {}
    model_name = None
    for line_number, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        head = stripped.split()[0]
        if head == "model":
            model_name = _MODEL_LINE.match(stripped).group(1) or ""
            parameter_index = 1
        elif head == "bayes":
            model_name = None
        elif model_name is not None:
            parameter_lines[(model_name, parameter_index)] = (line, line_number)
            parameter_index += 1
    return parameter_lines


@contextmanager
def local_xcm_path(
    values: Sequence[float],
    parameter_indices: Sequence[int],
    model_names: Sequence[str],
    template_path: str,
    *,
    tmp_dir: str | None = None,
) -> Iterator[str]:
    """Write a temporary copy of an `.xcm` template with new parameter values.

    Values are clamped into the template's hard limits and written at full
    float64 precision. The file is deleted when the `with` block exits.

    Yields:
        Path of the temporary `.xcm` file.
    """
    with open(template_path) as template:
        lines = template.readlines()
    parameter_lines = get_model_block(lines)

    for model_name, parameter_index, value in zip(model_names, parameter_indices, values):
        line, line_number = parameter_lines[(model_name, parameter_index)]
        fields = line.split()  # value delta hardmin softmin softmax hardmax
        # XSPEC saves hard limits with 6 significant digits, a hair inside the
        # prior support, and Xset.restore rejects values outside them.
        hard_min, hard_max = float(fields[2]), float(fields[5])
        # repr() round-trips float64: rounding would break e.g. finite-difference steps.
        fields[0] = repr(float(np.clip(value, hard_min, hard_max)))
        lines[line_number] = " ".join(fields) + "\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xcm", dir=tmp_dir, delete=False) as xcm_file:
        xcm_file.writelines(lines)
    try:
        yield xcm_file.name
    finally:
        # The enclosing temporary directory may already be gone.
        Path(xcm_file.name).unlink(missing_ok=True)


def _stack_folding_outputs(outputs: list[dict]) -> dict:
    """Stack the per-vector result dicts of a fold (nested dicts included) along a new first axis."""
    if not outputs:
        return {}
    return {
        key: _stack_folding_outputs([output[key] for output in outputs])
        if isinstance(value, dict)
        else np.stack([output[key] for output in outputs])
        for key, value in outputs[0].items()
    }


def parallel_folding(
    parameter_batch: np.ndarray,
    parameter_indices: Sequence[int],
    model_names: Sequence[str],
    desc: str = "",
    progress_bar: bool = True,
    pool: Any = None,
    return_kind: str = "cstat",
) -> dict:
    """Fold a batch of parameter vectors through XSPEC.

    Without a pool the folds run in this session, which is left at the last
    folded values.

    Parameters:
        parameter_batch: Parameter vectors, shape `(n, d)`.
        parameter_indices: XSPEC index of each parameter within its model.
        model_names: Model name of each parameter (`''` for the unnamed model).
        desc: Prefix for the progress-bar label.
        progress_bar: Whether to display a progress bar.
        pool: Optional process pool.
        return_kind: `"cstat"` (fit statistic only), `"full_model_counts"`
            (adds `"total_model_counts"`) or `"models_and_components"` (adds
            `"model_counts"` and `"component_counts"`).

    Returns:
        Outputs stacked over the rows of `parameter_batch`. The counts cover
        the noticed channels of every loaded spectrum, concatenated in
        spectrum order; a model or component is zero in the spectra it is not
        folded into (see [`SpectrumState`][bsixsa.xspec.SpectrumState]).
    """
    if return_kind not in RETURN_KINDS:
        raise ValueError(f"return_kind must be one of {RETURN_KINDS}, got {return_kind!r}.")

    with tempfile.TemporaryDirectory(prefix="parallel_folding_") as tmp_dir:
        template_path = os.path.join(tmp_dir, "model_template.xcm")
        xspec.Xset.save(template_path, info="m")

        with tqdm(total=len(parameter_batch), desc=desc + "Folding model", disable=not progress_bar) as progress:
            if pool is None:
                outputs = []
                for values in parameter_batch:
                    outputs.append(
                        folded_model_from_parameters(
                            values, parameter_indices, model_names,
                            template_path=template_path, tmp_dir=tmp_dir,
                            in_pool=False, return_kind=return_kind,
                        )
                    )
                    progress.update()
            else:
                results = [
                    pool.apply_async(
                        folded_model_from_parameters,
                        (values, parameter_indices, model_names),
                        kwds=dict(
                            template_path=template_path, tmp_dir=tmp_dir,
                            in_pool=True, return_kind=return_kind,
                        ),
                        callback=lambda _: progress.update(),
                    )
                    for values in parameter_batch
                ]
                outputs = [result.get() for result in results]

    return _stack_folding_outputs(outputs)


def folded_model_from_parameters(
    values: Sequence[float],
    parameter_indices: Sequence[int],
    model_names: Sequence[str],
    *,
    template_path: str,
    tmp_dir: str,
    in_pool: bool,
    return_kind: str,
) -> dict:
    """Fold one parameter vector; see [`parallel_folding`][bsixsa.xspec.parallel_folding]."""
    with XSilence():
        with local_xcm_path(values, parameter_indices, model_names, template_path, tmp_dir=tmp_dir) as xcm_path:
            xspec.Xset.restore(xcm_path)

        result = {"cstat": float(xspec.Fit.statistic)}
        if return_kind != "cstat":
            result.update(_model_counts_of_all_spectra(with_components=return_kind == "models_and_components"))

        if in_pool:
            # Makes pooled folding ~4x faster (measured, cause unknown). Never
            # in the main process: it would delete the user's model.
            xspec.AllModels.clear()

        return result
