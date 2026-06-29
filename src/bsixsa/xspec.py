import os
import tempfile
import xspec
import numpy as np
import re
from tqdm.auto import tqdm
from contextlib import contextmanager, nullcontext
from .convenience import XSilence, iter_components


class SpectrumState:
    """
    A class which helps in extracting the current XSPEC state with less verbose for a given spectrum.
    """

    def __init__(self, spectrum_index: int = 1):
        self.idx = spectrum_index
        self.spectrum: xspec.Spectrum = xspec.AllData(self.idx)

    @property
    def exposure(self) -> float:
        """
        Return the exposure of the observed spectrum in seconds.
        """
        return float(self.spectrum.exposure)

    @property
    def observed_counts(self) -> np.ndarray:
        """
        Return the observed spectrum in counts per bin.
        """
        return np.asarray(self.spectrum.values) * self.exposure

    @property
    def bin_edges(self) -> np.ndarray:
        """
        Return the bin edges in KeV. `bin_edges[0]` yields the lower values and `bin_edges[1]` the upper ones.
        """
        return np.asarray(self.spectrum.energies).T

    @property
    def bin_edges_1d(self) -> np.ndarray:

        return np.append(self.bin_edges[0], self.bin_edges[1][-1])

    @property
    def bin_width(self) -> np.ndarray:
        return np.diff(self.bin_edges, axis=0).squeeze()

    @property
    def bin_center(self) -> np.ndarray:
        return np.sqrt(np.prod(self.bin_edges, axis=0).squeeze())

    @property
    def component_counts(self) -> dict[str, np.ndarray]:

        xspec.Plot.device = "/null"
        xspec.Plot.add = True
        xspec.Plot("counts")

        total_add_idx = 1
        component_counts = {}

        for model_name in xspec.AllModels.sources.values():
            model = xspec.AllModels(1, model_name)
            n_add_components = sum(
                1 for _ in iter_components(model, additive_only=True)
            )

            # There should be strictly more than one component for XSPEC to register their values
            if n_add_components > 1:
                for i, component in enumerate(
                        iter_components(model, additive_only=True)
                ):
                    component_counts[f"{model_name}_{component.name}"] = np.asarray(
                        xspec.Plot.addComp(total_add_idx)
                    )
                    total_add_idx += 1

        return component_counts


    @property
    def model_counts(self) -> dict[str, np.ndarray]:

        model_counts = {}

        for model_name in xspec.AllModels.sources.values():
            model = xspec.AllModels(1, model_name)
            model_counts[model_name] = np.asarray(model.folded(1)) * self.exposure

        return model_counts

    @property
    def total_model_counts(self) -> np.ndarray:

        return sum(self.model_counts.values())


def get_model_block(lines):
    """
    Find the lines related to the model parameters in an .xcm file
    """

    in_model_block = False
    out_lines = []
    out_lines_indexes = []
    model_indexes = []
    parameter_indexes = []

    model_pattern = re.compile(r'^model(?:\s+(?:(\d+):(\S+)))?\b')

    for i, line in enumerate(lines):
        stripped = line.lstrip()

        if not stripped or stripped.startswith("#"):
            continue

        tokens = stripped.split()
        head = tokens[0]

        if head == "model":
            in_model_block = True
            m = model_pattern.match(stripped)

            current_parameter_index = 1

            if m.group(2) is not None:
                current_model_name = str(m.group(2))
            else:
                current_model_name = ""

            # Than we skip to the next iteration
            continue

        if head == "bayes":
            in_model_block = False
            continue

        if in_model_block:
            out_lines.append(line)
            out_lines_indexes.append(i)
            parameter_indexes.append(current_parameter_index)
            model_indexes.append(current_model_name)

            current_parameter_index += 1
            continue


    return {
        (model_name, int(par_index)): (out_line, line_number)
        for par_index, model_name, out_line, line_number in zip(
            parameter_indexes, model_indexes, out_lines, out_lines_indexes
        )
    }


@contextmanager
def local_xcm_path(params, indexes, model_indexes, base_xcm_path, *, tmp_dir=None):
    """
    Build a local .xcm file path containing the values specified in params, using a base xcm path.
    """

    with open(base_xcm_path, "r") as f:
        lines = f.readlines()
        mapping = get_model_block(lines)

    for model_name, parameter_index, value in zip(model_indexes, indexes, params):
        line, line_index = mapping[(model_name, parameter_index)]
        line = f"{value:.8g}".rjust(15) + line[15:]
        lines[line_index] = line

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".xcm",
        dir=tmp_dir,
        delete=False,
    ) as tmp_file:
        tmp_file.writelines(lines)
        local_xcm_path = tmp_file.name

    try:
        yield local_xcm_path

    finally:
        try:
            os.remove(local_xcm_path)
        except FileNotFoundError:
            pass


def _stack_folding_outputs(outputs):
    """
    Helper function to stack the individual outputs in parallel_folding.
    """

    if not outputs:
        return {}

    stacked = {}
    for key, value in outputs[0].items():
        if isinstance(value, dict):
            nested = {}
            for nested_key in value:
                nested[nested_key] = np.stack(
                    [out[key][nested_key] for out in outputs]
                )
            stacked[key] = nested
        else:
            stacked[key] = np.stack([out[key] for out in outputs])

    return stacked


def parallel_folding(
    params,
    indexes,
    model_indexes,
    desc="",
    progress_bar=True,
    pool=None,
    return_kind="cstat",
):
    """Evaluate folded `xspec` outputs for a batch of parameter vectors.

    Parameters:
        params: Parameter vectors to evaluate.
        indexes: `xspec` parameter indices associated with `params`.
        model_indexes: `xspec` model names associated with `indexes`.
        desc: Prefix for the progress-bar label.
        progress_bar: Whether to display a tqdm progress bar.
        pool: Optional process pool for parallel evaluation.
        return_kind: One of `"cstat"`, `"full_model_counts"`, or
            `"models_and_components"`.

    Returns:
        Stacked outputs across all rows in `params`. The `"cstat"` key is always present. `"total_model_counts"` is included for `"full_model_counts"` and `"models_and_components"`. `"model_counts"` and `"component_counts"` are included only for `"models_and_components"`.
    """

    with tempfile.TemporaryDirectory(prefix="parallel_folding_") as tmp_dir:
        model_file = os.path.join(tmp_dir, "model_template.xcm")
        xspec.Xset.save(model_file, info="m")

        progress_cm = (
            tqdm(total=len(params), desc=desc + "Folding model")
            if progress_bar
            else nullcontext()
        )

        with progress_cm as pbar:

            def update_progress(_):
                if pbar is not None:
                    pbar.update()

            in_pool = pool is not None

            kwargs = dict(
                model_file=model_file,
                tmp_dir=tmp_dir,
                in_pool=in_pool,
                return_kind=return_kind,
            )

            if not in_pool:

                outputs = []
                for param in params:
                    outputs.append(folded_model_from_parameters(param, indexes, model_indexes, **kwargs))
                    update_progress(1)

            else:

                results = [
                    pool.apply_async(
                        folded_model_from_parameters,
                        (param, indexes, model_indexes),
                        kwds=kwargs,
                        callback=update_progress,
                    )
                    for param in params
                ]

                outputs = [result.get() for result in results]

        return _stack_folding_outputs(outputs)


def folded_model_from_parameters(
        params, indexes, model_indexes,
        *, model_file, tmp_dir, in_pool, return_kind
):
    """
    Evaluate a folded `xspec` model for a single parameter vector. This is run in parallel in the [`parallel_folding`][bsixsa.xspec.parallel_folding] function.
    """

    if return_kind not in {"cstat", "full_model_counts", "models_and_components"}:
        raise ValueError(
            "return_kind must be one of 'cstat', 'full_model_counts', or "
            "'models_and_components'."
        )

    with XSilence():
        with local_xcm_path(params, indexes, model_indexes, model_file, tmp_dir=tmp_dir) as local_xcm:
            xspec.Xset.restore(local_xcm)

        cstat = float(xspec.Fit.statistic)

        result = {"cstat": cstat}

        if return_kind == "full_model_counts":
            s = SpectrumState(1)
            result["total_model_counts"] = s.total_model_counts

        elif return_kind == "models_and_components":
            s = SpectrumState(1)
            result["model_counts"] = s.model_counts
            result["component_counts"] = s.component_counts
            result["total_model_counts"] = s.total_model_counts

        if in_pool:
            xspec.AllModels.clear()  # VERY IMPORTANT : speedup of ~4 for unknown reasons

        return result
