"""Best fit with MINUIT's MIGRAD algorithm through iminuit."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from iminuit import Minuit

from ..solver import FitResults
from .abc import OptimiserBackend
from .config import Iminuit

if TYPE_CHECKING:
    from ..solver import Solver


class IminuitBackend(OptimiserBackend):
    """Maximum a posteriori fit with MIGRAD (and HESSE) in logit-of-CDF space, plus a Gaussian approximation.

    Attributes:
        minuit: The `Minuit` object after `run`; `minuit.fval` is not
            `FitResults.best_fit_stat` (see `_objective`).
    """

    config_cls = Iminuit

    def __init__(self, *, solver: Solver, config: Iminuit) -> None:
        super().__init__(solver=solver, config=config)
        self.minuit: Minuit | None = None

    def _objective(self, unconstrained_params: np.ndarray) -> float:
        """`-2 ln(posterior)`: the C-stat minus twice the log prior density.

        The prior term uses SciPy's normalised densities, so its values differ
        from `FitResults.best_fit_stat` (XSPEC's `bayes` term) by a constant:
        same minimum, values not comparable.
        """
        physical = self._unconstrained_to_physical(unconstrained_params)
        cstat = float(self.solver.simulate(physical, return_kind="cstat", progress_bar=False)["cstat"].item())
        log_prior = float(self.solver.prior.log_prob(physical.ravel()))
        self.tracer.record(physical, cstat)
        return cstat - 2.0 * log_prior

    def run(self) -> FitResults:
        """Run MIGRAD (and HESSE) and return the best fit and its Gaussian approximation."""
        start_physical = self._starting_point()
        if start_physical is None:
            # u = 0 is the prior median. Exactly 0 keeps MIGRAD's initial step of
            # 0.1; logit(F(median)) is often ~1e-16, which would shrink it to ~1e-18.
            x0 = np.zeros(self.solver.num_parameters)
        else:
            x0 = self._physical_to_unconstrained(start_physical)

        self.minuit = Minuit(self._objective, x0, name=self.solver.parameter_names)
        # The objective is -2 ln(posterior), so a change of 1 is one standard deviation.
        self.minuit.errordef = Minuit.LEAST_SQUARES
        self.minuit.print_level = 2
        self.minuit.strategy = self.config.strategy
        if self.config.tol is not None:
            self.minuit.tol = self.config.tol
        if self.config.precision is not None:
            self.minuit.precision = self.config.precision

        start = time.perf_counter()
        self.minuit.migrad(**self.config.engine_kwargs)
        if self.config.run_hesse:
            self.minuit.hesse()
        seconds = time.perf_counter() - start

        self.best_fit_unconstrained = np.asarray(self.minuit.values, dtype=np.float64)
        self.best_fit_params = self._unconstrained_to_physical(self.best_fit_unconstrained).ravel()
        covariance = self.minuit.covariance
        self.covariance_unconstrained = None if covariance is None else np.asarray(covariance)
        return self._fit_results(seconds, self.minuit.nfcn, self._best_fit_stat())
