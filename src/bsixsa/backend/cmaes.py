"""CMA-ES backend using *pycma*."""

from __future__ import annotations

import typing

import cma
import numpy as np
import pandas as pd

from . import register_backend
from .abc import Backend
from .config import BackendConfig
from ..solver import FitResults

if typing.TYPE_CHECKING:
    from ..solver import SIXSASolver

_EPS = 1e-10  # Small margin to keep unit-cube values away from the edges


@register_backend
class CMAESBackend(Backend):
    """Point-estimate fitter using CMA-ES (Covariance Matrix Adaptation).

    The optimisation is carried out directly in the unit cube with
    explicit ``[0, 1]`` bounds enforced by *pycma*.  Each unit-cube
    coordinate is mapped to physical space via the prior's quantile
    function (PPF).
    """

    name = "cmaes"
    config_cls = BackendConfig

    def __init__(
        self,
        *,
        solver: SIXSASolver,
        config: BackendConfig,
        sigma0: float = 0.25,
        **kwargs,
    ):
        super().__init__(solver=solver, config=config)
        self.sigma0 = sigma0
        self.best_fit_params: np.ndarray | None = None
        self.best_fit_cube: np.ndarray | None = None
        self.covariance_cube: np.ndarray | None = None
        self.es: cma.CMAEvolutionStrategy | None = None

    def _evaluate_batch(self, population: list[np.ndarray]) -> list[float]:
        """Evaluate a batch of candidates in unit-cube space.

        Stacks the population into a single ``(popsize, ndim)`` array,
        transforms to physical space, and calls
        :meth:`~SIXSASolver.simulate` once for the whole batch.
        """
        cube = np.asarray(population)
        physical = self._cube_to_physical(cube)
        sim = self.solver.simulate(
            physical,
            return_kind="cstat",
            progress_bar=False,
        )
        cstats = sim["cstat"].ravel()
        self.tracer.record(physical, cstats)
        return cstats.tolist()

    def run(
        self,
        *,
        x0_physical: np.ndarray | None = None,
        init_from_prior: bool = False,
        **cma_options,
    ) -> FitResults:
        """Run CMA-ES optimisation using the ask-and-tell interface.

        Each generation, the full population is evaluated in a single
        vectorised call to the simulator.

        Parameters:
            x0_physical: Initial guess in physical parameter space.
                Defaults to the prior median (0.5 in the unit cube).
            init_from_prior: If ``True``, draw a random sample from the
                prior and use it as the starting point (overrides
                ``x0_physical``).
            **cma_options: Extra options forwarded to
                :class:`cma.CMAEvolutionStrategy` (e.g. ``maxiter``,
                ``tolx``, ``verbose``).
        """
        from ..convenience import catchtime

        n_params = self.solver.num_parameters

        # --- starting point (in unit cube) ---
        if init_from_prior:
            x0_physical = self.solver.prior.sample(size=1).ravel()

        if x0_physical is not None:
            x0 = self._physical_to_cube(
                np.asarray(x0_physical, dtype=np.float64)
            )
        else:
            x0 = np.full(n_params, 0.5)

        x0 = np.asarray(x0, dtype=np.float64).ravel()

        # --- CMA-ES options with unit-cube bounds ---
        opts = cma.CMAOptions()
        opts["verbose"] = cma_options.pop("verbose", 3)
        opts["bounds"] = [0.0, 1.0]
        opts["CMA_stds"] = np.minimum(x0, 1.0 - x0)
        opts.update(cma_options)

        # --- run (ask-and-tell loop) ---
        with catchtime("Running CMA-ES", print_time=False) as run_time:
            self.es = cma.CMAEvolutionStrategy(x0, self.sigma0, opts)
            while not self.es.stop():
                population = self.es.ask()
                fitnesses = self._evaluate_batch(population)
                self.es.tell(population, fitnesses)
                self.es.disp()
            self.es.result_pretty()

        # --- extract results ---
        self.best_fit_cube = np.asarray(self.es.result.xbest).ravel()
        self.best_fit_params = self._cube_to_physical(self.es.result.xbest).ravel()
        D_scaled = self.es.sigma * self.es.sigma_vec.scaling * self.es.sm.D
        self.covariance_cube = self.es.sm.B @ np.diag(D_scaled ** 2) @ self.es.sm.B.T

        # --- posterior samples ---
        n_posterior = 10_000
        posterior_samples = self.sample(n_posterior)
        posterior_dict = {
            name: posterior_samples[:, i]
            for i, name in enumerate(self.solver.parameter_names)
        }

        return FitResults(
            time=float(run_time()),
            posterior_samples=pd.DataFrame.from_dict(posterior_dict),
            n_likelihood_evaluations=int(self.es.result.evaluations),
            log_Z=float("nan"),
            log_Z_err=float("nan"),
        )

    def sample(self, n: int) -> np.ndarray:
        """Draw samples from a Gaussian approximation in unit-cube space,
        clipped to ``[0, 1]``."""
        rng = np.random.default_rng()

        if self.covariance_cube is not None:
            samples_cube = rng.multivariate_normal(
                self.best_fit_cube, self.covariance_cube, size=n
            )
        else:
            samples_cube = np.tile(self.best_fit_cube, (n, 1))

        samples_cube = np.clip(samples_cube, _EPS, 1.0 - _EPS)
        return self._cube_to_physical(samples_cube)
