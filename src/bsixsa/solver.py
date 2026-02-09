from __future__ import print_function

from typing import Union, Tuple

import numpy as np
from ultranest.plot import PredictionBand
import os
import shutil

import numpy
import warnings
import pandas as pd
from xspec import Xset, AllModels, Fit, Plot, AllData
import xspec
import torch

from .convenience import XSilence
from .priors import build_prior
import matplotlib.pyplot as plt
from .xspec import parallel_folding
import pathos.multiprocessing as multiprocessing  # pathos
from bsixsa.analysis.plotting import plot_ppc
from tqdm.auto import tqdm


def store_chain(chainfilename, posterior, indexes, parameter_names, fit_statistic):
    """Write samples to an XSPEC-compatible chain FITS file.

    Parameters:
        chainfilename (str): Destination path for the FITS chain file.
        posterior (numpy.ndarray): Posterior samples with shape
            ``(n_samples, n_parameters)`` in unit-cube space.
        fit_statistic (numpy.ndarray): Fit statistic associated with each
            sample.
    """

    import astropy.io.fits as pyfits

    group_index = 1

    names = []

    for index in indexes:

        names.append(
            "%s__%d"
            % (parameter_names[index], index + (group_index - 1) * len(indexes))
        )

    columns = [
        pyfits.Column(name=name, format="D", array=np.asarray(posterior[:, i].numpy()))
        for i, name in enumerate(names)
    ]

    columns.append(pyfits.Column(name="FIT_STATISTIC", format="D", array=fit_statistic))
    table = pyfits.ColDefs(columns)
    header = pyfits.Header()
    header.add_comment("""Created with B-SISXA""")
    header.add_comment("""Based on BXA (Bayesian X-ray spectal Analysis) for Xspec""")
    header.add_comment("""refer to https://github.com/JohannesBuchner/""")
    header["TEMPR001"] = 1.0
    header["STROW001"] = 1
    header["EXTNAME"] = "CHAIN"
    tbhdu = pyfits.BinTableHDU.from_columns(table, header=header)
    tbhdu.writeto(chainfilename, overwrite=True)


def set_parameters(values, indexes, model_indexes):
    """Update the active XSPEC parameters using transformed values.

    Parameters:
        values (Sequence[float]): Parameter values in the physical space
    """
    parameter_to_set = {}  # Will contain {model : {par_index:prior}}


    for value, index, model_name in zip(values, indexes, model_indexes):
        parameter_to_set[model_name] = parameter_to_set.get(model_name, {})
        parameter_to_set[model_name][index] = float(value)

    parameter_to_set = sum(((AllModels(1, modName=k), v) for k, v in parameter_to_set.items()), ())
    AllModels.setPars(*parameter_to_set)

class SIXSASolver(object):
    allowed_stats = ["cstat", "pstat"]

    def __init__(
        self,
        prior,
        outputfiles_basename="./sixsa",
        overwrite=False,
        n_jobs=os.cpu_count(),
        sampler="nessai"
    ):

        #model = AllModels(1)
        prior, indexes, model_indexes, bounds = build_prior(prior, return_bounds=True)

        self.indexes = indexes
        self.model_indexes = model_indexes
        sources_models = AllModels.sources
        self.nb_models = len(sources_models)
        self.prior = prior
        self.samplers = {"prior": prior}
        self.bounds = bounds
        self.pool = multiprocessing.Pool(processes=n_jobs)
        self.posterior_samples = None
        self.sampler_kind:str = sampler
        self.sampler = None

        normalized_output_dir = os.path.normpath(outputfiles_basename)

        if not os.path.exists(normalized_output_dir):
            os.makedirs(normalized_output_dir)

        else:
            if not os.path.isdir(normalized_output_dir):
                raise ValueError(
                    f"Output path '{outputfiles_basename}' exists and is not a directory."
                )

            existing_entries = os.listdir(normalized_output_dir)
            if existing_entries:
                if not overwrite:
                    raise FileExistsError(
                        f"Output directory '{normalized_output_dir}' is not empty. "
                        "Pass overwrite=True to clear it."
                    )

                for entry in existing_entries:
                    entry_path = os.path.join(normalized_output_dir, entry)
                    if os.path.isdir(entry_path) and not os.path.islink(entry_path):
                        shutil.rmtree(entry_path)
                    else:
                        os.remove(entry_path)

        if not normalized_output_dir.endswith(os.sep):
            normalized_output_dir = normalized_output_dir + os.sep

        outputfiles_basename = normalized_output_dir
        self.outputfiles_basename = outputfiles_basename

    @property
    def num_parameters(self):
        return len(self.transformations)

    def sample_parameters(
        self,
        n_samples: int,
        sampler_name: str = "prior",
    ):
        r"""Sample parameters $\theta$ in the requested space.

        Parameters:
            n_samples (int): Number of draws to generate.
            kind (Literal["to_unit_cube", "to_bxa", "to_xspec"], optional):
                Target space for the returned samples. Defaults to
                ``"to_xspec"``.
            sampler (Callable | None): Optional custom sampler that returns
                unit-cube samples. If ``None``, a ``BoxUniform`` prior is used.

        Returns:
            torch.Tensor | numpy.ndarray | list[dict]: Samples expressed in the
                requested space.
        """

        sampler = self.samplers[sampler_name]
        theta = sampler.sample((n_samples,))  # In the unit cube space
        return theta

    def log_prob_fn(self, theta, x_o):
        r"""
        Return the log-posterior probability, defined as $\mathcal{LL} = -\frac{1}{2} \texttt{Cstat}$. Include the log-likelihood and any prior term defined in `xspec`.

        !!! note "On `x_o` parameter"
            `x_o` is a dummy parameter used for compatibility with `sbi` as the true spectrum is directly extracted from
            `xspec`. `sbi` require this parameter as in normal workflow, this function can be conditioned on any `x_o`.

        Parameters:
            theta (torch.Tensor): Array of samples on the unit cube
            x_o (torch.Tensor): Observed value, pass `None` if needed

        """

        return self.log_likelihood_fn(theta, x_o) + self.log_prior_fn(theta, x_o)

    def log_likelihood_fn(self, theta, x_o, progress_bar=True, no_pool=False):

        if no_pool:
            pool = None

        else:
            pool = self.pool

        simulation_outputs = parallel_folding(
            theta, self.indexes, self.model_indexes, desc="Evaluating C_stat - ", progress_bar=progress_bar, pool=pool
        )

        return -0.5 * simulation_outputs["cstat"]

    def log_prior_fn(self, theta, x_o):

        return self.samplers["prior"].log_prob(theta)

    def load_chain_in_xspec(self, n_samples, sampler_name: str = "exact_sampler"):

        posterior, _, c_stat = self.simulate(
            n_samples,
            sampler=sampler_name, desc="Computing posterior statistic - "
        )

        chainfilename = "%schain.fits" % self.outputfiles_basename

        store_chain(
            chainfilename, posterior, self.indexes, self.parameter_names, c_stat
        )

        xspec.AllChains.clear()
        xspec.AllChains += chainfilename

        warnings.warn("Cstat in chain does not account for prior logprob")


    def create_flux_chain(self, spectrum, erange="2.0 10.0", nsamples=None):
        """Evaluate fluxes for posterior samples within a given energy band.

        Parameters:
            spectrum: XSPEC spectrum object providing the ``flux`` attribute.
            erange (str, optional): Energy band passed to
                ``AllModels.calcFlux``. Defaults to ``"2.0 10.0"``.
            nsamples (int | None): Number of posterior samples to consider.
                Defaults to all available samples.

        Returns:
            (numpy.ndarray): Two-column array containing energy flux and photon
                flux for each posterior sample.
        """
        # prefix = analyzer.outputfiles_basename
        # modelnames = set([t['model'].name for t in transformations])

        with XSilence():
            # plot models
            flux = []
            for k, row in enumerate(tqdm(self.posterior[:nsamples], disable=None)):
                set_parameters(row, self.indexes, self.model_indexes)
                AllModels.calcFlux(erange)
                f = spectrum.flux
                # compute flux in current energies
                flux.append([f[0], f[3]])

            return numpy.array(flux)

    def posterior_predictions_convolved(
        self,
        component_names=None,
        plot_args=None,
        n_samples=400,
        sampler=None,
        plottype="counts",
    ):
        """Generate convolved posterior predictive bands for plotting.

        Parameters:
            component_names (list[str] | None): Labels associated with each
                additive model component. Use ``"ignore"`` to skip a component
                in the plot.
            plot_args (list[dict] | None): Matplotlib keyword arguments per
                component.
            n_samples (int, optional): Number of posterior samples to draw.
                Defaults to 400.
            plottype (str, optional): XSPEC plot type passed to ``Plot``.
                Defaults to ``"counts"``.

        Returns:
            dict: Observational data, model bands, and metadata needed for
                plotting posterior predictive checks.
        """
        # get data, binned to 10 counts
        # overplot models
        # can we do this component-wise?
        data = [None]  # bin, bin width, data and data error
        models = []  #
        if component_names is None:
            component_names = ["convolved model"] + [
                "component%d" for i in range(100 - 1)
            ]
        if plot_args is None:
            plot_args = [{}] * 100
            for i, c in enumerate(plt.rcParams["axes.prop_cycle"].by_key()["color"]):
                plot_args[i] = dict(color=c)
                del i, c
        bands = []
        Plot.background = True
        Plot.add = True

        for content in self.posterior_predictions_plot(
            sampler=sampler, plottype=plottype, n_samples=n_samples
        ):
            xmid = content[:, 0]
            ndata_columns = 6 if Plot.background else 4
            ncomponents = content.shape[1] - ndata_columns
            if data[0] is None:
                data[0] = content[:, 0:ndata_columns]
            model_contributions = []
            for component in range(ncomponents):
                y = content[:, ndata_columns + component]
                if component >= len(bands):
                    bands.append(PredictionBand(xmid))
                bands[component].add(y)

                model_contributions.append(y)
            models.append(model_contributions)

        for band, label, component_plot_args in zip(bands, component_names, plot_args):
            if label == "ignore":
                continue
            lineargs = dict(drawstyle="steps", color="k")
            lineargs.update(component_plot_args)
            shadeargs = dict(color=lineargs["color"])
            band.shade(alpha=0.5, **shadeargs)
            band.shade(q=0.495, alpha=0.1, **shadeargs)
            band.line(label=label, **lineargs)

        if Plot.background:
            results = dict(
                list(
                    zip(
                        "bins,width,data,error,background,backgrounderr".split(","),
                        data[0].transpose(),
                    )
                )
            )
        else:
            results = dict(
                list(zip("bins,width,data,error".split(","), data[0].transpose()))
            )
        results["models"] = numpy.array(models)
        return results

    def posterior_predictions_unconvolved(
        self,
        component_names=None,
        plot_args=None,
        nsamples=400,
        plottype="model",
    ):
        """Generate unconvolved posterior predictive bands for each component.

        Parameters:
            component_names (list[str] | None): Labels for model components;
                use ``"ignore"`` to skip drawing a component.
            plot_args (list[dict] | None): Matplotlib keyword arguments per
                component.
            nsamples (int, optional): Number of posterior samples to draw.
                Defaults to 400.
            plottype (str, optional): Argument passed to ``xspec.Plot``.
                Defaults to ``"model"``.
        """
        if component_names is None:
            component_names = ["model"] + ["component%d" for i in range(100 - 1)]
        if plot_args is None:
            plot_args = [{}] * 100
            for i, c in enumerate(plt.rcParams["axes.prop_cycle"].by_key()["color"]):
                plot_args[i] = dict(color=c)
                del i, c
        Plot.add = True
        bands = []

        for content in self.posterior_predictions_plot(
            plottype=plottype, n_samples=nsamples
        ):
            xmid = content[:, 0]
            ncomponents = content.shape[1] - 2
            for component in range(ncomponents):
                y = content[:, 2 + component]

                if component >= len(bands):
                    bands.append(PredictionBand(xmid))
                bands[component].add(y)

        for band, label, component_plot_args in zip(bands, component_names, plot_args):
            if label == "ignore":
                continue
            lineargs = dict(drawstyle="steps", color="k")
            lineargs.update(component_plot_args)
            shadeargs = dict(color=lineargs["color"])
            band.shade(alpha=0.5, **shadeargs)
            band.shade(q=0.495, alpha=0.1, **shadeargs)
            band.line(label=label, **lineargs)

    def posterior_predictions_plot(self, plottype, sampler=None, n_samples=None):
        """Yield XSPEC plot arrays for posterior predictive visualisations.

        Parameters:
            plottype (str): Plot type forwarded to ``xspec.Plot``.
            n_samples (int | None): Number of posterior samples to evaluate.

        Returns:
            (numpy.ndarray): Arrays containing plot-ready data for each sampled
                posterior draw.
        """
        # for plotting, we don't need so many points, and especially the
        # points that barely made it into the analysis are not that interesting.
        # so pick a random subset of at least nsamples points
        parameters = self.sample_parameters(
            sampler_name=sampler, n_samples=n_samples
        )

        with XSilence():
            olddevice = Plot.device
            Plot.device = "/null"

            # plot models
            Plot(plottype)

            for k, row in enumerate(tqdm(parameters, disable=None)):
                set_parameters(row, self.indexes, self.model_indexes)

                sources_models = xspec.AllModels.sources

                maxncomp = 0
                # get plot data
                if plottype == "model":
                    base_content = numpy.transpose(
                        [Plot.x(), Plot.xErr(), Plot.model()]
                    )
                elif Plot.background:
                    """
                    base_content = numpy.transpose(
                        [
                            Plot.x(),
                            Plot.xErr(),
                            Plot.y(),
                            Plot.yErr(),
                            Plot.backgroundVals(),
                            numpy.zeros_like(Plot.backgroundVals()),
                            Plot.model(),
                        ]
                    )
                    """
                    e = numpy.mean(numpy.asarray(AllData(1).energies), axis = 1)
                    e_widths = numpy.diff(numpy.asarray(AllData(1).energies), axis = 1).flatten()


                    expected_rates = []

                    add_comps_counter = 1
                    for source, model_name in zip(sources_models.keys(), sources_models.values()):
                        xspec.Plot('counts', 'model ' + model_name)
                        model = xspec.AllModels(1,model_name)
                    
                        # If the model has additive components
                        nb_of_add_components = model.expression.count('+')
                        if nb_of_add_components >= 1 :
                            for i in range(add_comps_counter, add_comps_counter + nb_of_add_components):
                                expected_rates.append(xspec.Plot.addComp(i))
                            add_comps_counter += nb_of_add_components
                    
                        # If not, plot the raw model
                        else :
                            expected_rates.append(np.asarray(model.folded(1)) * xspec.AllData(1).exposure / e_widths)

                    total_rate = np.sum(np.asarray(expected_rates), axis = 0)

                    data = numpy.asarray(xspec.AllData(1).values) * xspec.AllData(1).exposure / e_widths
                    data_err = numpy.sqrt(data / e_widths) #don't know why /e_widths but that matches pyxpec

                    base_content = [
                            e,
                            e_widths/2,
                            data,
                            data_err,
                            Plot.backgroundVals(),
                            numpy.zeros_like(Plot.backgroundVals()),
                            total_rate,
                        ]

                    # Add the model, where each model has its own additive components separated
                    for rate in expected_rates :
                        base_content.append(rate)


                    base_content = numpy.transpose(base_content)


                else:
                    base_content = numpy.transpose(
                        [Plot.x(), Plot.xErr(), Plot.y(), Plot.yErr(), Plot.model()]
                    )
                """
                # get additive components, if there are any
                comp = []
                for i in range(1, maxncomp):
                    try:
                        comp.append(Plot.addComp(i))
                    except Exception:
                        print(
                            'The error "***XSPEC Error: Requested array does not exist for this plot." can be ignored.'
                        )
                        maxncomp = i
                        break
                content = numpy.hstack(
                    (
                        base_content,
                        numpy.transpose(comp).reshape((len(base_content), -1)),
                    )
                )
                """
                yield base_content
            Plot.device = olddevice


    @property
    def parameter_names(self) -> list[str]:
        """Return unique parameter names aligned with XSPEC ordering.

        Returns:
            (list[str]): Parameter names augmented with component identifiers to
                avoid duplicates.
        """

        from .convenience import iter_thawn_parameters
        return [parameter.name for parameter in iter_thawn_parameters()]


    def build_dataframe(
        self,
        sampler: str = "exact_sampler",
        num_samples=10_000,
    ) -> pd.DataFrame:
        """Build a posterior sample table from the fitted neural network.

        Parameters:
            num_samples (int, optional): Number of samples to draw from the
                posterior sampler. Defaults to 10_000.

        Returns:
            pandas.DataFrame: Table with parameter samples and corresponding
                importance weights.
        """


        sampler_kwargs = {}

        samples = self.samplers[sampler].sample(
            (num_samples,), **sampler_kwargs
        )

        dict_of_params = {
            name: parameters for name, parameters in zip(self.parameter_names, samples.T)
        }

        return pd.DataFrame.from_dict(dict_of_params)


    def get_xspec_best_fit(self):
        """Run an XSPEC fit and return best-fit parameters with covariance.

        Returns:
            (tuple[numpy.ndarray, numpy.ndarray]): Flattened parameter vector and
                covariance matrix estimated by XSPEC.
        """

        with XSilence():
            xspec.Fit.perform()

        def build_covariance_matrix_np(covar_elements):
            covar_elements = np.asarray(covar_elements, dtype=float)

            M = len(covar_elements)
            N = int((np.sqrt(1 + 8 * M) - 1) // 2)

            cov_matrix = np.zeros((N, N), dtype=float)
            i, j = np.tril_indices(N)
            cov_matrix[i, j] = covar_elements
            cov_matrix[j, i] = covar_elements

            return cov_matrix

        sources_models = xspec.AllModels.sources
        best_fit_parameters = []
        for source, model_name in zip(sources_models.keys(), sources_models.values()):

            xspec_model = xspec.AllModels(1,model_name)


            best_fit_parameters.extend(
                [xspec_model(i + 1).values[0] for i in range(xspec_model.nParameters)]
            )
        covariance = build_covariance_matrix_np(xspec.Fit.covariance)
        best_fit_parameters = np.asarray(best_fit_parameters)
        return best_fit_parameters.ravel(), covariance


    def plot_ppc(self, sampler, **kwargs):
        return plot_ppc(self, sampler, **kwargs)

    def sample_parameters(
            self,
            n_samples: int,
            sampler_name: str = "prior",
    ):
        r"""Sample parameters $\theta$ in the requested space.

        Parameters:
            n_samples (int): Number of draws to generate.
            kind (Literal["to_unit_cube", "to_bxa", "to_xspec"], optional):
                Target space for the returned samples. Defaults to
                ``"to_xspec"``.
            sampler (Callable | None): Optional custom sampler that returns
                unit-cube samples. If ``None``, a ``BoxUniform`` prior is used.

        Returns:
            torch.Tensor | numpy.ndarray | list[dict]: Samples expressed in the
                requested space.
        """

        sampler = self.samplers[sampler_name]
        theta = sampler.sample((n_samples,))  # In the unit cube space
        return theta

    def run(self, **kwargs):

        # TODO : check for a way to distinguish sampler & samplers
        if self.sampler_kind == "nessai":
            from .sampler.nessai import NessaiSampler
            self.sampler = NessaiSampler(solver=self, **kwargs)

        elif self.sampler_kind == "nautilus":
            from .sampler.nautilus import NautilusSampler
            self.sampler = NautilusSampler(solver=self, **kwargs)

        elif self.sampler_kind == "ultranest":
            from .sampler.ultranest import UltranestSampler
            self.sampler = UltranestSampler(solver=self, **kwargs)

        else:
            raise NotImplementedError()

        self.posterior_samples = self.sampler.run()
        self.samplers["posterior"] = self.sampler
        return self.sampler.sampler


