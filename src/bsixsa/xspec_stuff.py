import os
import xspec
import numpy as np
import uuid
import pathos.multiprocessing as multiprocessing #pathos
from tqdm.auto import tqdm
from contextlib import contextmanager, nullcontext


def transform_parameters_for_xspec(transformations, theta) -> dict[int, float]:
    """Transform the current parameters using BXA transformation for XSPEC (i.e. real space) and return a dictionary"""
    return {int(t['index']) : float(t['aftertransform'](theta[i])) for i, t in enumerate(transformations)}



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
        int(par_index) : (out_line, line_number) for par_index, out_line, line_number in zip(parameter_index, out_lines, out_lines_indexes)
    }


@contextmanager
def local_xcm_path(params, base_xcm_path):
    """
    Build a local .xcm file path containing the values specified in params, using a base xcm path.
    """

    local_xcm_path = f"local_values_{uuid.uuid4()}.xcm"

    with open(base_xcm_path, "r") as f:
        lines = f.readlines()
        mapping = get_model_block(lines)

    for par_index, value in params.items():
        line, line_index = mapping[par_index]
        line = f"{value:.8g}".rjust(15) + line[15:]
        lines[line_index] = line

    with open(local_xcm_path, "w") as f:
        f.writelines(lines)

    try:
        yield local_xcm_path

    finally:
        os.remove(local_xcm_path)


def parallel_folding(params, n_jobs=None, return_stat=False, apply_stat=True, desc="", progress_bar=True):
    """Perform simulation in parallel with XSPEC"""
    # Set up the number of workers
    if n_jobs is None:
        n_jobs = os.cpu_count()  # Use all available CPUs if n_jobs is not set

    model_file = f"parallel_folding_{uuid.uuid4()}.xcm"

    if os.path.exists(model_file):
        os.remove(model_file)

    xspec.Xset.save(model_file, info="m")

    try:

        # Create a progress bar
        with tqdm(total=len(params), desc=desc + "Folding model") if progress_bar else nullcontext() as pbar:

            def update_progress(_):
                if pbar is not None:
                    pbar.update()

            with multiprocessing.Pool(processes=n_jobs) as pool:

                results = [pool.apply_async(folded_model_from_parameters, (param, model_file, apply_stat), callback=update_progress) for param in params]

                if return_stat:
                    result_to_return =  np.vstack([result.get()[1] for result in results])

                else:
                    result_to_return = np.vstack([result.get()[0] for result in results])

    except Exception as e:
        print(f'Simulations interrupted by {e}')

    finally:
        files = os.listdir("./")

        for file in files:
            if file.startswith("parallel_folding_") or file.startswith("local_values_"):
                os.remove(file)

    return result_to_return


def folded_model_from_parameters(params, model_file, apply_stat):
    from bsixsa import XSilence

    with XSilence():
        with local_xcm_path(params, model_file) as local_xcm:
            xspec.Xset.restore(local_xcm)

        xspec.Fit.statMethod = "cstat"
        xspec.Fit.bayes = "on"
        model = xspec.AllModels(1)
        #model.setPars(params)
        count_list = []
        stat_list = []

        for n in range(1, xspec.AllData.nSpectra + 1):

            expected_rate = np.asarray(model.folded(n), dtype=np.float32) * xspec.AllData(n).exposure
            count_list.append(expected_rate)

        if apply_stat:
            spectra = np.random.poisson(np.hstack(count_list))
        else:
            spectra = np.hstack(count_list)

        stat_list.append(float(xspec.Fit.statistic))

        xspec.AllModels.clear() # VERY IMPORTANT : speedup of ~4 for unknown reasons

    return spectra, np.asarray(stat_list, dtype=np.float32).ravel()
