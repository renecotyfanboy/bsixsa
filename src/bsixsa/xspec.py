import os
import tempfile
import xspec
import numpy as np
from tqdm.auto import tqdm
from contextlib import contextmanager, nullcontext
from .convenience import XSilence


def get_model_block(lines):
    """
    Find the lines related to the model parameters in an .xcm file
    """

    in_model_block = False
    out_lines = []
    out_lines_indexes = []
    output = []
    nb_of_models = 0
    for i, line in enumerate(lines):
        stripped = line.lstrip()

        if not stripped or stripped.startswith("#"):
            continue

        tokens = stripped.split()
        head = tokens[0]

        if head == "model":

            nb_of_models += 1
            in_model_block = True
            if nb_of_models >=2 :

                parameter_index = np.argsort(out_lines_indexes) + 1
                output.append(
                                {
                                int(par_index): (out_line, line_number)
                                for par_index, out_line, line_number in zip(
                                    parameter_index, out_lines, out_lines_indexes
                                )
                                }
                             )
                out_lines = []
                out_lines_indexes = []


            continue

        if head == "bayes":
            in_model_block = False

            if i == len(lines)-1:
                output.append(
                                {
                                int(par_index): (out_line, line_number)
                                for par_index, out_line, line_number in zip(
                                    parameter_index, out_lines, out_lines_indexes
                                )
                                }
                             )
            continue

        if in_model_block:
            out_lines.append(line)
            out_lines_indexes.append(i)
            continue

        parameter_index = np.argsort(out_lines_indexes) + 1



    return output


@contextmanager
def local_xcm_path(params, indexes, nb_models, base_xcm_path, *, tmp_dir=None):
    """
    Build a local .xcm file path containing the values specified in params, using a base xcm path.
    """

    with open(base_xcm_path, "r") as f:
        lines = f.readlines()
        mapping = get_model_block(lines)

    #for index, value in zip(indexes, params):
    #    line, line_index = mapping[index]
    #    line = f"{value:.8g}".rjust(15) + line[15:]
    #    lines[line_index] = line

    idx = 0
    for model_nr in range(nb_models):
        nb_params_model = len(indexes[model_nr])
        for index, value in zip(indexes[model_nr], params[idx:idx+nb_params_model]):
            line, line_index = mapping[model_nr][index]
            line = f"{value:.8g}".rjust(15) + line[15:]
            lines[line_index] = line
        idx += nb_params_model


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
    params, indexes, nb_models, n_jobs=None, return_stat=False, apply_stat=True, desc="", progress_bar=True, pool=None
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

        #if isinstance(params, torch.Tensor):
        #    params = np.asarray(params.to("cpu").numpy())

        with progress_cm as pbar:

            def update_progress(_):
                if pbar is not None:
                    pbar.update()

            in_pool = pool is not None

            if not in_pool:

                outputs = [
                    folded_model_from_parameters(param, indexes, nb_models, model_file, apply_stat, tmp_dir, in_pool)
                    for param in params
                ]

            else:

                results = [
                    pool.apply_async(
                        folded_model_from_parameters,
                        (param, indexes, nb_models, model_file, apply_stat, tmp_dir, in_pool),
                        callback=update_progress,
                    )
                    for param in params
                ]

                outputs = [result.get() for result in results]

        spectra = np.vstack([spectra for spectra, _ in outputs])
        cstat = np.vstack([stat for _, stat in outputs]).squeeze()

        return {
            "spectra": spectra,
            "cstat": cstat,
        }


def folded_model_from_parameters(params, indexes,nb_models, model_file, apply_stat, tmp_dir, in_pool):

    with XSilence():
        with local_xcm_path(params, indexes, nb_models, model_file, tmp_dir=tmp_dir) as local_xcm:
            xspec.Xset.restore(local_xcm)

        xspec.Fit.statMethod = "cstat"
        xspec.Fit.bayes = "off" # <- we handle the prior logprob on our own
        sources_models = xspec.AllModels.sources


        count_list = []
        stat_list = []

        for n in range(1, xspec.AllData.nSpectra + 1):
            expected_rate = 0
            for source, model_name in zip(sources_models.keys(), sources_models.values()):
                model = xspec.AllModels(1,model_name)
                expected_rate += np.asarray(model.folded(n)) * xspec.AllData(n).exposure

            count_list.append((expected_rate))

        if apply_stat:
            spectra = np.random.poisson(np.hstack(count_list))
        else:
            spectra = np.hstack(count_list)

        stat_list.append(float(xspec.Fit.statistic))

        if in_pool:
            xspec.AllModels.clear()  # VERY IMPORTANT : speedup of ~4 for unknown reasons

    return spectra, np.asarray(stat_list).ravel()
