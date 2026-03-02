import os, contextlib
from torch import nn
import torch
import sbi
import numpy as np
import xspec
from ..abc import Sampler
from .embedding.abc import Embedding
from typing import Optional
import dill
import warnings
from sbi.neural_nets import posterior_nn
from sbi.utils import get_density_thresholder, RestrictedPrior, BoxUniform
from sbi.inference import NPE, EnsemblePosterior, ImportanceSamplingPosterior
from joblib import Parallel, delayed
import functools


def patch_sample_no_pbar(posterior):
    """
    Patch the sample method of a posterior to remove the progress bar.
    """
    original = posterior.sample
    posterior._original_sample = original

    @functools.wraps(original)
    def sample(*args, _original=original, **kwargs):
        kwargs.setdefault("show_progress_bars", False)
        return _original(*args, **kwargs)

    posterior.sample = sample

    return posterior


def work_quiet(func):
    """
    Decorator to run a function without printing anything to stdout or stderr.
    """
    def quiet_func(*args, **kwargs):
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f): #, contextlib.redirect_stderr(f):
            return func(*args, **kwargs)

    return quiet_func


@work_quiet
def training_job(number, theta, x, first_round=True, proposal=None, training_kwargs=None):

    if first_round:

        prior_sbi = BoxUniform(
            torch.zeros(theta.shape[-1]),
            torch.ones(theta.shape[-1])
        )

        build_fun = posterior_nn(model="maf")
        inference = NPE(prior=prior_sbi, density_estimator=build_fun, device="cpu")

    else:
        with open(f"results_sixsa/inference_{number}.pkl", "rb") as file:
            inference = dill.load(file)

    training_kwargs = {} if training_kwargs is None else training_kwargs

    with warnings.catch_warnings():
        # Catches warning about pickling NPE / Restricted prior not pointing to wrong address
        warnings.simplefilter("ignore", category=UserWarning)

        density_estimator = inference.append_simulations(
            torch.from_numpy(theta.copy()), torch.from_numpy(x.copy()), proposal=proposal
        ).train(**training_kwargs)

        posterior = inference.build_posterior(density_estimator)

        with open(f"results_sixsa/inference_{number}.pkl", "wb") as file:
            dill.dump(inference, file, recurse=True)

        with open(f"results_sixsa/posterior_{number}.pkl", "wb") as file:
            dill.dump(posterior, file, recurse=True)

    return posterior


class SIXSASampler(Sampler):

    def __init__(self, *, solver: 'SIXSASolver', **kwargs):
        super().__init__(solver=solver)
        self.proposals = []
        self.current_round = 0
        self.n_nets = os.cpu_count()
        self.prior = BoxUniform(
            low=torch.zeros(solver.prior.ndim),
            high=torch.ones(solver.prior.ndim)
        )

    def sample(self, shape):
        pass

    def inference_round(
            self,
            n_simulations,
            proposal=None,
            round_number=0,
            embedding=lambda x: x,
            training_kwargs=None,
    ):
        ## Sample & Simulate
        if proposal is None:
            parameters_unit_cube = self.prior.sample((n_simulations,))
        else:
            parameters_unit_cube = proposal.sample((n_simulations,))

        parameters = self.solver.prior.from_unit_cube(parameters_unit_cube.numpy())
        simulations_results = self.solver.simulate(parameters, return_kind="full_model_counts")
        spectra = simulations_results["total_model_counts"]

        ## Train the multiple NPEs
        parameters_unit_cube = parameters_unit_cube.numpy().copy()
        posteriors = Parallel(n_jobs=-1)(
            delayed(training_job)(
                i, parameters_unit_cube,
                embedding(np.random.poisson(spectra).astype(np.float32)),
                first_round=round_number == 1, proposal=proposal, training_kwargs=training_kwargs
            ) for i, simulations in enumerate(range(self.n_nets))
        )

        ## Gather a posterior ensemble
        posteriors = [patch_sample_no_pbar(posterior) for posterior in posteriors]
        ensemble = EnsemblePosterior(posteriors)
        ensemble.set_default_x(torch.from_numpy(embedding(self.solver.observed_spectrum)))

        return ensemble

    def run(
            self,
            num_simulations_per_round: list[int],
            *,
            embedding: Optional[Embedding | list[Embedding]] = None,
            training_kwargs=None,
            plot_embedding_coverage=True,
            device="cpu",
    ):

        num_rounds = len(num_simulations_per_round)

        if training_kwargs is None:
            # TODO: Better handling for the default training parameters. The user should be able to provide a subset
            training_kwargs = dict(
                retrain_from_scratch=True,
                discard_prior_samples=True,
                use_combined_loss=True,
                training_batch_size=256,
                validation_fraction=0.2,
                learning_rate=5e-4,
            )


        if not isinstance(training_kwargs, list):
            training_kwargs_list = [training_kwargs] * num_rounds

        else:
            raise NotImplementedError("Not implemented yet")

        if isinstance(embedding, Embedding):
            embedding_list = [embedding] * num_rounds

        elif isinstance(embedding, list):
            embedding_list = embedding

        else:
            raise NotImplementedError("Not implemented yet")

        proposal = None

        # Multiple round inference loop
        for round_number, (num_simulations, embedding, training_kwargs) in enumerate(zip(num_simulations_per_round, embedding_list, training_kwargs_list), start=1):

            proposal = self.inference_round(
                num_simulations,
                embedding=embedding,
                proposal=proposal,
                round_number=round_number,
                training_kwargs=training_kwargs
            )

            # We train a restricted proposal if not in the last round
            if round_number < num_rounds:

                accept_reject_fn = get_density_thresholder(
                    proposal, num_samples_to_estimate_support=100_000, quantile=1e-4
                )

                proposal = RestrictedPrior(
                    self.prior,
                    accept_reject_fn,
                    posterior=proposal,
                    sample_with="sir",
                    device=device,
                )

        return proposal
        #self.plot_training_summary(
        #    filename=f"{self.outputfiles_basename}training_summary.pdf"
        #)

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

    def round_colors(self, num_rounds):
        return cmr.take_cmap_colors(cmr.cosmic_r, num_rounds, cmap_range=(0.1, 0.8))


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
