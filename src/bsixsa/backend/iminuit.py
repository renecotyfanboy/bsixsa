"""Iminuit backend using MINUIT's *MIGRAD* algorithm."""

from __future__ import annotations

import typing

import numpy as np
import pandas as pd
from iminuit import Minuit

from . import register_backend
from .abc import Backend
from .config import Iminuit
from ..solver import FitResults

if typing.TYPE_CHECKING:
    from ..solver import Solver


@register_backend
class IminuitBackend(Backend):
    """Point-estimate fitter using MINUIT's MIGRAD algorithm.

    Like :class:`LevenbergMarquardtBackend`, the optimisation is carried
    out in *unconstrained* space obtained via the logit transform of the
    unit cube, so the search is unbounded.
    """

    name = "iminuit"
    config_cls = Iminuit

    def __init__(
        self,
        *,
        solver: Solver,
        config: Iminuit,
    ):
        super().__init__(solver=solver, config=config)
        self.best_fit_params: np.ndarray | None = None
        self.best_fit_unconstrained: np.ndarray | None = None
        self.covariance_unconstrained: np.ndarray | None = None
        self.minuit: Minuit | None = None

    def _objective(self, unconstrained_params: np.ndarray) -> float:
        """C-statistic in unconstrained space (to minimise)."""
        physical = self._unconstrained_to_physical(unconstrained_params)
        sim = self.solver.simulate(
            physical,
            return_kind="cstat",
            progress_bar=False,
        )
        cstat = float(sim["cstat"].item())
        self.tracer.record(physical, cstat)
        return cstat

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def run(self) -> FitResults:
        """Run MIGRAD optimisation.
        """
        from ..convenience import catchtime

        n_params = self.solver.num_parameters
        x0_physical = self.config.x0_physical
        init_from_prior = self.config.init_from_prior
        run_hesse = self.config.run_hesse
        migrad_kwargs = self.config.engine_kwargs.copy()

        # --- starting point ---
        if init_from_prior:
            x0_physical = self.solver.prior.sample(size=1).ravel()

        if x0_physical is not None:
            x0 = self._physical_to_unconstrained(
                np.asarray(x0_physical, dtype=np.float64)
            )
        else:
            x0 = np.full(n_params, 0.0)

        x0 = np.asarray(x0, dtype=np.float64).ravel()

        # --- set up Minuit ---
        self.minuit = Minuit(self._objective, x0, name=self.solver.parameter_names)
        self.minuit.errordef = Minuit.LEAST_SQUARES
        self.minuit.print_level = 2

        # --- run ---
        with catchtime("Running MIGRAD", print_time=False) as run_time:
            self.minuit.migrad(**migrad_kwargs)
            if run_hesse:
                self.minuit.hesse()

        # --- extract results ---
        self.best_fit_unconstrained = np.asarray(self.minuit.values, dtype=np.float64)
        self.best_fit_params = self._unconstrained_to_physical(
            self.best_fit_unconstrained
        ).ravel()

        if self.minuit.covariance is not None:
            self.covariance_unconstrained = np.asarray(self.minuit.covariance)
        else:
            self.covariance_unconstrained = None

        # --- posterior samples ---
        n_posterior = 10_000
        posterior_samples = self.sample(n_posterior)
        posterior_dict = {
            name: posterior_samples[:, i]
            for i, name in enumerate(self.solver.parameter_names)
        }

        best_fit = pd.Series(
            self.best_fit_params, index=self.solver.parameter_names
        )
        best_fit_stat = float(self.minuit.fval)

        return FitResults(
            time=float(run_time()),
            posterior_samples=pd.DataFrame.from_dict(posterior_dict),
            n_likelihood_evaluations=int(self.minuit.nfcn),
            log_Z=float("nan"),
            log_Z_err=float("nan"),
            best_fit=best_fit,
            best_fit_stat=best_fit_stat,
        )

    def sample(self, n: int) -> np.ndarray:
        """Draw samples from a Gaussian approximation in unconstrained space."""
        rng = np.random.default_rng()

        if self.covariance_unconstrained is not None:
            samples_unconstrained = rng.multivariate_normal(
                self.best_fit_unconstrained, self.covariance_unconstrained, size=n
            )
        else:
            samples_unconstrained = np.tile(self.best_fit_unconstrained, (n, 1))

        return self._unconstrained_to_physical(samples_unconstrained)
