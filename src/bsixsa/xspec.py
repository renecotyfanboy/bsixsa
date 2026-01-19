import os
import tempfile
import xspec
import numpy as np
import torch
import pathos.multiprocessing as multiprocessing  # pathos
from tqdm.auto import tqdm
from contextlib import contextmanager, nullcontext


def transform_parameters_for_xspec(transformations, theta) -> dict[int, float]:
    """Transform the current parameters using BXA transformation for XSPEC (i.e. real space) and return a dictionary"""
    return {
        int(t["index"]): float(t["aftertransform"](theta[i]))
        for i, t in enumerate(transformations)
    }


def get_model_block(lines):
    """
    Find the lines related to the model parameters in an .xcm file
    """

    in_model_block = False
    out_lines = []
    out_lines_indexes = []

    for i, line in enumerate(lines):
        stripped = line.lstrip()

        if not stripped or stripped.startswith("#"):
            continue

        tokens = stripped.split()
        head = tokens[0]

        if head == "model":
            in_model_block = True
            continue

        if head == "bayes":
            in_model_block = False
            continue

        if in_model_block:
            out_lines.append(line)
            out_lines_indexes.append(i)
            continue

    parameter_index = np.argsort(out_lines_indexes) + 1

    return {
        int(par_index): (out_line, line_number)
        for par_index, out_line, line_number in zip(
            parameter_index, out_lines, out_lines_indexes
        )
    }


@contextmanager
def local_xcm_path(params, base_xcm_path, *, tmp_dir=None):
    """
    Build a local .xcm file path containing the values specified in params, using a base xcm path.
    """

    with open(base_xcm_path, "r") as f:
        lines = f.readlines()
        mapping = get_model_block(lines)

    for par_index, value in params.items():
        line, line_index = mapping[par_index]
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


def parallel_folding(
    params, n_jobs=None, return_stat=False, apply_stat=True, desc="", progress_bar=True
):
    """Perform simulation in parallel with XSPEC.

    Returns:
        dict[str, torch.Tensor]: Dictionary containing the stacked spectra under
            the ``spectra`` key and the corresponding Cash statistics under
            ``cstat``.
    """
    # Set up the number of workers
    if n_jobs is None:
        n_jobs = os.cpu_count()  # Use all available CPUs if n_jobs is not set

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

            with multiprocessing.Pool(processes=n_jobs) as pool:
                results = [
                    pool.apply_async(
                        folded_model_from_parameters,
                        (param, model_file, apply_stat, tmp_dir),
                        callback=update_progress,
                    )
                    for param in params
                ]

                outputs = [result.get() for result in results]

        spectra = torch.from_numpy(
            np.vstack([spectra for spectra, _ in outputs]).astype(np.float32)
        )
        cstat = torch.from_numpy(
            np.vstack([stat for _, stat in outputs]).squeeze().astype(np.float32)
        )

        return {
            "spectra": spectra,
            "cstat": cstat,
        }


def folded_model_from_parameters(params, model_file, apply_stat, tmp_dir):
    from bsixsa import XSilence

    with XSilence():
        with local_xcm_path(params, model_file, tmp_dir=tmp_dir) as local_xcm:
            xspec.Xset.restore(local_xcm)

        xspec.Fit.statMethod = "cstat"
        xspec.Fit.bayes = "off" # <- we handle the prior logprob on our own
        model = xspec.AllModels(1)
        # model.setPars(params)
        count_list = []
        stat_list = []

        for n in range(1, xspec.AllData.nSpectra + 1):
            expected_rate = (
                np.asarray(model.folded(n), dtype=np.float32)
                * xspec.AllData(n).exposure
            )
            count_list.append(expected_rate)

        if apply_stat:
            spectra = np.random.poisson(np.hstack(count_list))
        else:
            spectra = np.hstack(count_list)

        stat_list.append(float(xspec.Fit.statistic))

        xspec.AllModels.clear()  # VERY IMPORTANT : speedup of ~4 for unknown reasons

    return spectra, np.asarray(stat_list, dtype=np.float32).ravel()
