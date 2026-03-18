import contextlib
import functools
import os
import warnings
from typing import TYPE_CHECKING, Optional

import dill
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sbi.inference import EnsemblePosterior, NPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform, RestrictedPrior, get_density_thresholder

from ..abc import Backend
from .. import register_backend
from .embedding import IdentityEmbedding
from .embedding.abc import Embedding

if TYPE_CHECKING:
    from ...solver import FitResults, SIXSASolver


class FCEmbedding(torch.nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int = 32,
            num_hiddens=[256]
            #dropout_rate: float = 0.01,
    ):
        """Fully-connected multi-layer neural network to be used as embedding network.

        Args:
            input_dim: Dimensionality of input that will be passed to the embedding net.
            output_dim: Dimensionality of the output.
            num_layers: Number of layers of the embedding network. (Minimum of 2).
            num_hiddens: Number of hidden units in each layer of the embedding network.
        """
        super().__init__()
        layers = []
        num_layers = len(num_hiddens)
        # first and last layer is defined by the input and output dimension.
        # therefor the "number of hidden layers" is num_layers-2
        prev = input_dim
        for i in range(num_layers):
            layers.append(torch.nn.Linear(prev, num_hiddens[i]))
            layers.append(torch.nn.LayerNorm(num_hiddens[i]))
            layers.append(torch.nn.GELU())
            #layers.append(nn.Dropout(dropout_rate))
            prev = num_hiddens[i]

        layers.append(torch.nn.Linear(prev, output_dim))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Network forward pass.

        Args:
            x: Input tensor (batch_size, num_features)

        Returns:
            Network output (batch_size, num_features).
        """
        return self.net(x)


def patch_sample_no_pbar(posterior):
    """Patch the sample method of a posterior to remove the progress bar."""
    original = posterior.sample
    posterior._original_sample = original

    @functools.wraps(original)
    def sample(*args, _original=original, **kwargs):
        kwargs.setdefault("show_progress_bars", False)
        return _original(*args, **kwargs)

    posterior.sample = sample
    return posterior


def work_quiet(func):
    """Decorator to run a function without printing anything to stdout."""

    def quiet_func(*args, **kwargs):
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            return func(*args, **kwargs)

    return quiet_func


@work_quiet
def training_job(
    number,
    theta,
    x,
    *,
    embedding,
    round_number,
    output_dir,
    proposal=None,
    training_kwargs=None,
):
    current_round_dir = os.path.join(output_dir, f"round_{round_number}")
    current_artifacts_dir = os.path.join(current_round_dir, "artifacts")
    os.makedirs(current_artifacts_dir, exist_ok=True)

    # Treat the spectra
    x = torch.as_tensor(np.random.poisson(x).astype(np.float32))

    if (round_number == 1) or training_kwargs.get("retrain_from_scratch", False):
        embedding_net = FCEmbedding(x.shape[1], 16)
        prior_sbi = BoxUniform(torch.zeros(theta.shape[-1]), torch.ones(theta.shape[-1]))
        build_fun = posterior_nn(model="maf", embedding_net=embedding_net, prior=prior_sbi)
        inference = NPE(prior=prior_sbi, density_estimator=build_fun, device="cpu")
    else:
        previous_round_dir = os.path.join(output_dir, f"round_{round_number - 1}")
        previous_artifacts_dir = os.path.join(previous_round_dir, "artifacts")
        previous_inference_path = os.path.join(
            previous_artifacts_dir,
            f"inference_{number}.pkl",
        )
        if not os.path.exists(previous_inference_path):
            raise FileNotFoundError(
                f"Missing previous round state for net {number}: {previous_inference_path}"
            )
        with open(previous_inference_path, "rb") as file:
            inference = dill.load(file)

    training_kwargs = {} if training_kwargs is None else training_kwargs

    with warnings.catch_warnings():
        # Catches warning about pickling NPE / Restricted prior not pointing to wrong address
        warnings.simplefilter("ignore", category=UserWarning)

        density_estimator = inference.append_simulations(
            torch.from_numpy(theta.copy()),
            x,
            proposal=proposal,
        ).train(**training_kwargs)

        posterior = inference.build_posterior(density_estimator)

        with open(
            os.path.join(current_artifacts_dir, f"inference_{number}.pkl"),
            "wb",
        ) as file:
            dill.dump(inference, file, recurse=True)

        with open(
            os.path.join(current_artifacts_dir, f"density_estimator_{number}.pkl"),
            "wb",
        ) as file:
            dill.dump(density_estimator, file, recurse=True)

        with open(
            os.path.join(current_artifacts_dir, f"posterior_{number}.pkl"),
            "wb",
        ) as file:
            dill.dump(posterior, file, recurse=True)

    return posterior


@register_backend
class SIXSABackend(Backend):

    name = "sixsa"

    def __init__(self, *, solver: "SIXSASolver", **kwargs):
        super().__init__(solver=solver)
        self.proposals = []
        self.n_nets = 8 #max(1, os.cpu_count() or 1)
        self.prior = BoxUniform(
            low=torch.zeros(solver.prior.ndim),
            high=torch.ones(solver.prior.ndim),
        )

    def sample(self, n: int) -> np.ndarray:
        current_sampler = self.proposals[-1] if self.proposals else self.prior
        return current_sampler.sample((n,))

    def inference_round(
        self,
        n_simulations,
        *,
        embedding: Embedding,
        proposal=None,
        round_number=0,
        training_kwargs=None,
    ):
        # Sample parameters and simulate spectra
        if proposal is None:
            parameters_unit_cube = self.prior.sample((n_simulations,))
        else:
            parameters_unit_cube = proposal.sample((n_simulations,))

        parameters = self.solver.prior.from_unit_cube(parameters_unit_cube.numpy())
        simulations_results = self.solver.simulate(parameters, return_kind="full_model_counts")
        spectra = simulations_results["total_model_counts"]

        # Fit embedding once, then apply it
        noisy_spectra = np.random.poisson(spectra).astype(np.float32)
        round_path = os.path.join(self.solver.outputfiles_basename, f"round_{round_number}")
        os.makedirs(round_path, exist_ok=True)
        embedding.fit(
            noisy_spectra,
            metrics_path=os.path.join(round_path, "embedding_training.pdf"),
        )

        embedded_observation = embedding.transform(self.solver.observed_spectrum)
        if isinstance(embedded_observation, torch.Tensor):
            embedded_observation = embedded_observation.detach().cpu().numpy()
        embedded_observation = np.asarray(embedded_observation, dtype=np.float32).squeeze()

        # Train the multiple NPEs
        parameters_unit_cube_np = parameters_unit_cube.numpy().copy()
        posteriors = Parallel(n_jobs=-1)(
            delayed(training_job)(
                i,
                parameters_unit_cube_np,
                spectra,
                embedding=None,
                round_number=round_number,
                output_dir=self.solver.outputfiles_basename,
                proposal=proposal,
                training_kwargs=training_kwargs,
            )
            for i in range(self.n_nets)
        )

        # Gather a posterior ensemble
        posteriors = [patch_sample_no_pbar(posterior) for posterior in posteriors]
        ensemble = EnsemblePosterior(posteriors)

        embedded_observation = np.atleast_1d(embedded_observation.astype(np.float32))

        ensemble.set_default_x(torch.from_numpy(self.solver.observed_spectrum))
        return ensemble

    def run(
        self,
        num_simulations_per_round: list[int] = None,
        *,
        embedding: Optional[Embedding | list[Embedding]] = None,
        training_kwargs=None,
        plot_embedding_coverage=True,
        device="cpu",
        **kwargs,
    ) -> 'FitResults':
        from ...solver import FitResults
        from ...convenience import catchtime

        if num_simulations_per_round is None:
            raise TypeError(
                "SIXSABackend.run() requires `num_simulations_per_round` as "
                "the first positional argument."
            )

        # kept for API compatibility for now
        _ = plot_embedding_coverage

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

        if isinstance(training_kwargs, list):
            if len(training_kwargs) != num_rounds:
                raise ValueError(
                    "`training_kwargs` length must match `num_simulations_per_round`."
                )
            training_kwargs_list = training_kwargs
        else:
            training_kwargs_list = [training_kwargs] * num_rounds

        if embedding is None:
            identity_embedding = IdentityEmbedding()
            embedding_list = [identity_embedding] * num_rounds
        elif isinstance(embedding, Embedding):
            embedding_list = [embedding] * num_rounds
        elif isinstance(embedding, list):
            if len(embedding) != num_rounds:
                raise ValueError("`embedding` list length must match number of rounds.")
            embedding_list = embedding
        else:
            raise TypeError("`embedding` must be an Embedding, list[Embedding], or None.")

        with catchtime("Running SBI inference", print_time=False) as run_time:
            proposal = None
            self.proposals = []

            # Multiple round inference loop
            for round_number, (num_simulations, current_embedding, current_training_kwargs) in enumerate(
                zip(num_simulations_per_round, embedding_list, training_kwargs_list),
                start=1,
            ):
                proposal = self.inference_round(
                    num_simulations,
                    embedding=current_embedding,
                    proposal=proposal,
                    round_number=round_number,
                    training_kwargs=current_training_kwargs,
                )
                self.proposals.append(proposal)

                # We train a restricted proposal if not in the last round
                if round_number < num_rounds:
                    accept_reject_fn = get_density_thresholder(
                        proposal,
                        num_samples_to_estimate_support=100_000,
                        quantile=1e-4,
                    )
                    proposal = RestrictedPrior(
                        self.prior,
                        accept_reject_fn,
                        posterior=proposal,
                        sample_with="sir",
                        device=device,
                    )

        # Build posterior samples from the final proposal
        n_posterior = 10_000
        posterior_unit = self.sample(n_posterior)
        if hasattr(posterior_unit, "numpy"):
            posterior_unit = posterior_unit.numpy()
        posterior_physical = self.solver.prior.from_unit_cube(np.asarray(posterior_unit))

        posterior_dict = {
            name: posterior_physical[:, i]
            for i, name in enumerate(self.solver.parameter_names)
        }

        return FitResults(
            time=float(run_time()),
            posterior_samples=pd.DataFrame.from_dict(posterior_dict),
            n_likelihood_evaluations=0,
            log_Z=float("nan"),
            log_Z_err=float("nan"),
        )
