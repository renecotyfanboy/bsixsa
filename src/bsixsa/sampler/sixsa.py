"""
WIP BACKUP CODE
"""

class ImportanceSamplingPosteriorUnweighted(ImportanceSamplingPosterior):
    # noinspection PyMethodOverriding
    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        weighted: bool = False,
        oversampling_factor: int = 10,
        show_progress_bars: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if len(sample_shape) != 1:
            raise ValueError("Sampling shape must have 1 dimension")

        if weighted:
            oversampling_factor = 1

        samples, log_weight = super().sample(
            (sample_shape[0] * oversampling_factor,),
            method="importance",
            show_progress_bars=show_progress_bars,
        )

        finite_weight = torch.isfinite(log_weight)
        samples = samples[finite_weight]
        log_weight = log_weight[finite_weight]

        max_log_weight = torch.max(log_weight)
        stable_log_weight = log_weight - max_log_weight
        weight = torch.exp(stable_log_weight)

        if weighted:
            return samples, weight

        cdf = torch.cumsum(weight, dim=0)
        cdf = cdf / cdf[-1]
        n_samples = int(sample_shape[0])
        u0 = torch.rand(1) / n_samples
        positions = u0 + torch.arange(n_samples, dtype=torch.float32) / n_samples
        idx = torch.searchsorted(cdf, positions, right=True)
        return samples.index_select(0, idx).to(torch.float32)


class SIXSASolver(object):
    allowed_stats = ["cstat", "pstat"]

    def __init__(
            self,
            prior,
            model,
            outputfiles_basename="chains/",
            use_background=False,
            device="cpu",
            overwrite=False,
            n_jobs=os.cpu_count(),
    ):

        self.epoch_trained = []
        self.training_loss = []
        self.validation_loss = []
        self.embedding = None
        self.embedding_net = nn.Identity()
        self.device = device
        self.samplers = dict()
        self.inference = None
        self.embedding_list = []
        self.current_round = 0
        self.pool = multiprocessing.Pool(processes=n_jobs)

        prior, indexes, bounds = build_prior(model, prior, return_bounds=True)

        self.indexes = indexes
        self.prior = prior
        self.bounds = bounds

        build_prior(prior)

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

        self.density_estimator_build_fun = posterior_nn(
            model="maf",
            # hidden_features=100,
            # num_transforms=10,
        )

        if self.background_to_compute:
            self._background = (
                    np.asarray(xspec.AllData(1).background.values)
                    * xspec.AllData(1).background.exposure
            ).astype(int)
            self._backratio = np.asarray(
                xspec.AllData(1).exposure / xspec.AllData(1).background.exposure
            )

        self.samplers["prior"] = prior
        self.outputfiles_basename = outputfiles_basename

    def set_best_fit(self):
        """Set current XSPEC parameters to the maximum a posteriori sample."""
        # TODO : Update this function

        raise NotImplementedError()

        x0 = self.posterior_unit_cube[
            :, self.posterior_sampler.log_prob(self.posterior_unit_cube.T).argmax()
        ]
        result = minimize(
            lambda p: -self.posterior_sampler.log_prob(p, track_gradients=True),
            x0,
            method="newton-exact",
        )
        best_fit = result.x
        params = self.prior_function(best_fit.numpy())
        set_parameters(transformations=self.transformations, values=params)

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
            theta, self.indexes, desc="Evaluating C_stat - ", progress_bar=progress_bar, pool=pool
        )

        return -0.5 * simulation_outputs["cstat"]

    def log_prior_fn(self, theta, x_o):

        return self.samplers["prior"].log_prob(theta)

    @property
    def available_samplers(self) -> list[str]:
        return list(self.samplers.keys())

    @property
    def observed_spectrum(self):
        """
        Return the observed spectrum read from `xspec`
        """
        return (
                np.asarray(xspec.AllData(1).values, dtype=np.float32)
                * xspec.AllData(1).exposure
        )

    def build_exact_sampler(self, sampler, x_o):
        """
        Build an exact sampler using a given proposal sampler and the associated observation.
        """

        return ImportanceSamplingPosteriorUnweighted(
            potential_fn=self.log_prob_fn, proposal=sampler, method="sir"
        ).set_default_x(x_o)

    def post_process(self, device="cpu"):
        embedding = self.embedding_list[-1]

        if isinstance(embedding, TorchModuleEmbedding):
            x_o = (
                embedding(
                    torch.from_numpy(
                        self.observed_spectrum.astype(np.float32)[None, :]
                    ).to(embedding.device)
                )
                .to(device)
                .squeeze()
            )

        else:
            x_o = torch.from_numpy(np.squeeze(embedding(self.observed_spectrum))).to(
                device
            )

        sampler = self.build_exact_sampler(list(self.samplers.values())[-1], x_o)

        self.samplers["exact_sampler"] = sampler

        return sampler

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

    def run(
            self,
            num_simulations_per_round: list[int],
            *,
            embedding: Optional[Embedding | list[Embedding]] = None,
            training_kwargs=None,
            plot_embedding_coverage=True,
            n_samples_xspec: int = 1000,
            device="cpu",
    ):
        """Run sequential simulation-based inference for the configured model.

        Parameters:
            num_simulations_per_round (list[int]): Number of simulations per
                inference round.
            embedding (Embedding | list[Embedding] | None): Embedding module or
                per-round list of embeddings applied to simulated spectra.
            npe_kwargs (dict | None): Keyword arguments forwarded to the NPE
                constructor.
            training_kwargs (dict | list[dict] | None): Per-round training
                arguments for the density estimator.
            plot_embedding_coverage (bool, optional): Enable diagnostic plots of
                embedding coverage. Defaults to ``True``.
            device (str, optional): Torch device identifier. Defaults to
                ``"cpu"``.

        Returns:
            ImportanceSamplingPosterior: Final importance-sampling posterior
                enriched with SIR corrections.
        """

        if Fit.statMethod.lower() not in SIXSASolver.allowed_stats:
            raise RuntimeError(
                "ERROR: not using cstat or pstat! set Fit.statMethod to cash before analysing (currently: %s)!"
                % Fit.statMethod
            )

        if training_kwargs is None:
            # TODO: Better handling for the default training parameters. The user should be able to provide a subset
            training_kwargs = dict(
                force_first_round_loss=True,
                retrain_from_scratch=True,
                discard_prior_samples=True,
                use_combined_loss=True,
                training_batch_size=256,
                validation_fraction=0.2,
                learning_rate=5e-4,
            )

        num_rounds = len(num_simulations_per_round)

        if not isinstance(training_kwargs, list):
            training_kwargs = [training_kwargs] * num_rounds

        if isinstance(embedding, Embedding):
            embedding_list = [embedding] * num_rounds

        elif isinstance(embedding, list):
            embedding_list = embedding

        else:
            raise NotImplementedError("Not implemented yet")

        for rounds in range(num_rounds):
            self.perform_inference_round(
                num_simulations_per_round[rounds],
                embedding=embedding_list[rounds],
                training_kwargs=training_kwargs[rounds],
                device=device,
                plot_embedding_coverage=plot_embedding_coverage,
            )

            """
            round_samples = self.sample_parameters(
                10_000, sampler=f"round_{rounds + 1}", kind="to_xspec"
            )

            chain = self.chain_from_sample(
                round_samples, name=f"Round {rounds + 1}", color=colors[rounds]
            )

            cc.add_chain(chain)
            """

        posterior_sampler = self.post_process(device=device)
        self.load_chain_in_xspec(n_samples_xspec)

        self.plot_training_summary(
            filename=f"{self.outputfiles_basename}training_summary.pdf"
        )

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
                set_parameters(row, self.indexes)
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
            maxncomp = 100 if Plot.add else 0
            for k, row in enumerate(tqdm(parameters, disable=None)):
                set_parameters(row, self.indexes)
                Plot(plottype)
                # get plot data
                if plottype == "model":
                    base_content = numpy.transpose(
                        [Plot.x(), Plot.xErr(), Plot.model()]
                    )
                elif Plot.background:
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
                else:
                    base_content = numpy.transpose(
                        [Plot.x(), Plot.xErr(), Plot.y(), Plot.yErr(), Plot.model()]
                    )
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
                yield content
            Plot.device = olddevice

    @property
    def parameter_names(self):
        """Return unique parameter names aligned with XSPEC ordering.

        Returns:
            (list[str]): Parameter names augmented with component identifiers to
                avoid duplicates.


        xspec_model = xspec.AllModels(1)

        def rename_parameters(params, comps):
            from collections import Counter

            # Count how many times each parameter name appears
            counts = Counter(params)

            new_names = []
            back_num = 0
            for param, comp in zip(params, comps):
                if counts[param] > 1:
                    # If this parameter appears more than once,
                    # append the first two letters of its component.
                    back_num += 1
                    new_names.append(f"{param} ({comp[:2]}_{back_num})")
                else:
                    # If it appears only once, keep it as-is
                    new_names.append(param)
            return new_names

        parameter_names = []
        component_names = []
        parameter_index = []

        for component in xspec_model.componentNames:
            for parameter in getattr(xspec_model, component).parameterNames:
                parameter_names.append(parameter)
                component_names.append(component)
                parameter_index.append(
                    getattr(getattr(xspec_model, component), parameter).index - 1
                )

        parameter_names_vanilla = list(np.asarray(parameter_names)[parameter_index])
        component_names = list(np.asarray(component_names)[parameter_index])
        return rename_parameters(parameter_names_vanilla, component_names)
        """

        from .convenience import build_parameter_name
        return build_parameter_name(xspec.AllModels(1))

    def build_dataframe(
            self,
            sampler: str = "exact_sampler",
            num_samples=10_000,
            oversampling_factor: int | None = 10,
            weighted: bool | None = True,
            likelihood=True,
    ):
        """Build a posterior sample table from the fitted neural network.

        Parameters:
            num_samples (int, optional): Number of samples to draw from the
                posterior sampler. Defaults to 10_000.

        Returns:
            pandas.DataFrame: Table with parameter samples and corresponding
                importance weights.
        """

        dict_of_params = {}
        sampler_kwargs = {}

        if (oversampling_factor is not None) and sampler == "exact_sampler":
            sampler_kwargs["oversampling_factor"] = oversampling_factor

        if (weighted is not None) and sampler == "exact_sampler":
            sampler_kwargs["weighted"] = weighted

        if weighted:
            samples, weights = self.samplers[sampler].sample(
                (num_samples,), weighted=True
            )
            dict_of_params = {
                "weight": weights.numpy().astype(np.float32),
            }

        else:
            samples = self.samplers[sampler].sample(
                (num_samples,), **sampler_kwargs
            )

        for index, parameters in zip(self.indexes, samples.T):
            name = self.parameter_names[index]
            dict_of_params[name] = np.asarray(
                parameters.numpy()
            )

        return pd.DataFrame.from_dict(dict_of_params)

    def simulate(
            self,
            n_samples,
            *,
            sampler: str,
            **kwargs
    ):
        r"""Simulate spectra using XSPEC, optionally adding background counts.

        Parameters:
            n_samples (int): Number of samples to draw
            sampler (str): Sampler to use for simulation.
            **kwargs: Additional keyword arguments forwarded to
                ``parallel_folding``.

        Returns:
            (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Sampled parameters $\theta$, simulated spectra in counts per bin and Cash statistic as computed by `xspec`.
        """

        parameters_xspec = self.sample_parameters(n_samples, sampler_name=sampler)

        if (not self.background_to_compute) or (kwargs.get("return_stat", False)):
            folding_outputs = parallel_folding(parameters_xspec, self.indexes, pool=self.pool, **kwargs)
            spectra = folding_outputs["spectra"]
            c_stat = folding_outputs["cstat"]
            return parameters_xspec, spectra, c_stat

        if self.background_to_compute:
            background = (
                    np.random.poisson(
                        np.repeat(self._background[None, :], len(parameters_xspec), axis=0)
                    )
                    * self._backratio
            )

            folding_outputs = parallel_folding(parameters_xspec, self.indexes, pool=self.pool, **kwargs)
            spectra = folding_outputs["spectra"]
            c_stat = folding_outputs["cstat"]

            return parameters_xspec, spectra + background, c_stat

        else:
            raise NotImplementedError

    def perform_inference_round(
            self,
            num_simulations,
            *,
            embedding: Embedding,
            restricted_prior: bool = True,
            reset_flow: bool = True,
            training_kwargs: Optional[dict] = None,
            plot_embedding_coverage=True,
            device="cpu",
    ):
        self.current_round += 1
        self.embedding_list.append(embedding)
        round_path = os.path.join(
            self.outputfiles_basename, f"round_{self.current_round}"
        )
        os.mkdir(round_path)
        training_kwargs = training_kwargs if training_kwargs is not None else dict()

        # Unpacking prior and proposal
        prior = self.samplers["prior"]
        proposal_name, proposal = list(self.samplers.items())[-1]

        if proposal_name == "exact_sampler":
            proposal_name, proposal = list(self.samplers.items())[-2]

        if restricted_prior and (prior != proposal):
            accept_reject_fn = get_density_thresholder(
                proposal, num_samples_to_estimate_support=100_000, quantile=1e-3
            )

            proposal = RestrictedPrior(
                self.samplers["prior"],
                accept_reject_fn,
                posterior=proposal,
                sample_with="sir",
                device=device,
            )

        if reset_flow:
            self.inference = NPE(
                prior=self.samplers["prior"],
                density_estimator=self.density_estimator_build_fun,
                device=device,
            )

            inference = self.inference

        else:
            inference = self.inference

        observation = torch.from_numpy(self.observed_spectrum)

        theta, all_simulations, _ = self.simulate(
            num_simulations,
            sampler=proposal_name,
            desc=f"Round {self.current_round} - "
            if self.current_round is not None
            else "",
        )

        if isinstance(embedding, TrainableEmbedding):
            (
                embedding.train(
                    all_simulations, metrics_path=f"{round_path}/embedding_training.pdf"
                ),
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if isinstance(embedding, TorchModuleEmbedding):
                all_simulations = all_simulations.to(embedding.device)

            features = embedding(all_simulations)

        if not isinstance(theta, torch.Tensor):
            theta_train = torch.from_numpy(theta.T.astype(np.float32))

        else:
            theta_train = theta

        if not isinstance(features, torch.Tensor):
            x_train = torch.from_numpy(features.astype(np.float32))

        else:
            x_train = features

        if isinstance(embedding, TorchModuleEmbedding):
            x_o = embedding(observation[None, :].to(embedding.device)).to(device).squeeze()

        else:
            x_o = torch.from_numpy(np.squeeze(embedding(observation))).to(device)

        theta_train = theta_train.to(device).detach()
        x_train = x_train.to(device).detach()

        density_estimator = inference.append_simulations(
            theta_train, x_train, proposal=proposal
        ).train(**training_kwargs)

        posterior = inference.build_posterior(density_estimator)

        if x_o is not None:
            posterior = posterior.set_default_x(x_o)

        original_sample = posterior.sample

        # This is just to avoid annoying repeated progress bar
        def sample(*args, **kwargs):
            kwargs["show_progress_bars"] = kwargs.get("show_progress_bars", False)
            return original_sample(*args, **kwargs)

        posterior.sample = sample
        num_epochs = inference.summary["epochs_trained"][-1]

        self.epoch_trained.append(num_epochs)
        self.training_loss.extend(inference.summary["training_loss"][-num_epochs:])
        self.validation_loss.extend(inference.summary["validation_loss"][-num_epochs:])

        self.samplers[f"round_{self.current_round}"] = posterior

        # Plot summary stat stuff
        if plot_embedding_coverage:
            embedding.plot_coverage(
                x_train,
                x_o,
                round_number=self.current_round,
                save_to_path=f"{round_path}/features_round_{self.current_round}.pdf",
            )

    def parameter_mapper(
            self,
            embedding: "Embedding",
            *,
            sampler_name: str = "prior",
            n_samples: int = 50_000,
    ):
        return fit_parameter_mapper(
            self,
            embedding=embedding,
            sampler_name=sampler_name,
            n_samples=n_samples
        )

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

        xspec_model = xspec.AllModels(1)
        best_fit_parameters = np.asarray(
            [xspec_model(i + 1).values[0] for i in range(xspec_model.nParameters)]
        )
        covariance = build_covariance_matrix_np(xspec.Fit.covariance)

        return best_fit_parameters.ravel(), covariance

    def chain_from_sampler(self, sampler_name: str = "prior", n_samples=1000, **kwargs):
        parameter_names = self.parameter_names

        samples = self.sample_parameters(
            n_samples, kind="to_xspec", sampler_name=sampler_name
        )
        param_dict = {}

        for i, t in enumerate(self.transformations):
            j = t["index"]
            name = parameter_names[j - 1]
            param_dict[name] = [sample[j] for sample in samples]

        return Chain(samples=pd.DataFrame.from_dict(param_dict), **kwargs)

    def round_colors(self, num_rounds):
        return cmr.take_cmap_colors(cmr.cosmic_r, num_rounds, cmap_range=(0.1, 0.8))

    def plot_posterior_from_rounds(self, n_samples=10_000):
        cc = ChainConsumer()
        colors = self.round_colors(self.current_round)

        for i, color in zip(range(1, self.current_round + 1), colors):
            cc.add_chain(
                self.chain_from_sampler(
                    f"round_{i}", name=f"round_{i}", color=color, n_samples=n_samples
                )
            )

        figure = cc.plotter.plot(
            filename=f"{self.outputfiles_basename}posterior_per_round.pdf"
        )
        plt.close("all")

        return figure

    def plot_training_summary(self, figsize=(10, 7), filename=None):
        figure = plt.figure(figsize=figsize)

        prev_num = 0
        colors = self.round_colors(len(self.epoch_trained))

        for round, num_epoch in enumerate(self.epoch_trained):
            steps = np.arange(prev_num + 1, prev_num + 1 + num_epoch)

            plt.plot(
                steps,
                self.training_loss[steps.min() - 1: steps.max()],
                color=colors[round],
                linestyle="dotted",
            )

            plt.plot(
                steps,
                self.validation_loss[steps.min() - 1: steps.max()],
                color=colors[round],
            )

            plt.axvline(
                prev_num + num_epoch, color="black", linestyle="dotted", alpha=0.3
            )
            prev_num += num_epoch

        plt.xlabel("Epoch")
        plt.ylabel("Validation loss")

        custom_lines = [
            Line2D([0], [0], color="black", linestyle="dotted"),
            Line2D([0], [0], color="black"),
        ]

        plt.legend(custom_lines, ["Training loss", "Validation loss"])

        if filename is not None:
            plt.savefig(filename)
            plt.close()

        return figure

    def plot_ppc(self, sampler, **kwargs):
        return plot_ppc(self, sampler, **kwargs)
