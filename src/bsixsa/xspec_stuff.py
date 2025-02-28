import os
import xspec
import numpy as np
import pathos.multiprocessing as multiprocessing #pathos
from tqdm.auto import tqdm


def parallel_folding(params, n_jobs=None, return_stat=False, desc=""):
    """Perform simulation in parallel with XSPEC"""
    # Set up the number of workers
    if n_jobs is None:
        n_jobs = os.cpu_count()  # Use all available CPUs if n_jobs is not set

    if os.path.exists("parallel_folding.xcm"):
        os.remove("parallel_folding.xcm")

    xspec.Xset.save("parallel_folding.xcm", info="m")

    # Create a progress bar
    with tqdm(total=len(params), desc=desc + "Folding model") as pbar:

        def update_progress(_):
            pbar.update()

        with multiprocessing.Pool(processes=n_jobs) as pool:

            results = [pool.apply_async(folded_model_from_parameters, (param,), callback=update_progress) for param in params]

            if return_stat:
                result_to_return =  np.vstack([result.get()[1] for result in results])

            else:
                result_to_return = np.vstack([result.get()[0] for result in results])

    os.remove("parallel_folding.xcm")

    return result_to_return


def transform_parameters_for_xspec(transformations, theta) -> dict[int, float]:
    """Transform the current parameters using BXA transformation for XSPEC (i.e. real space) and return a dictionary"""
    return {int(t['index']) : float(t['aftertransform'](theta[i])) for i, t in enumerate(transformations)}


def folded_model_from_parameters(params):
    from bsixsa import XSilence
    with XSilence():

        xspec.Xset.restore("parallel_folding.xcm")
        xspec.Fit.statMethod = "cstat"
        model = xspec.AllModels(1)
        model.setPars(params)
        count_list = []
        stat_list = []

        for n in range(1, xspec.AllData.nSpectra + 1):

            expected_rate = np.multiply(model.folded(n), xspec.AllData(n).exposure)
            count_list.append(expected_rate)

        poisson_realisation = np.random.poisson(np.hstack(count_list))
        stat_list.append(xspec.Fit.statistic)

    return poisson_realisation, np.asarray(stat_list).ravel()

    #return (
    #    f"Expected {params}, All pars {[model(i).values[0] for i in range(1, 6)]}",
    #    poisson_realisation
    #)