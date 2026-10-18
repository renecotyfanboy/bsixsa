"""Levenberg-Marquardt best fit: least squares on Poisson deviance residuals plus prior residuals.

The squared data residuals sum to XSPEC's C-statistic, and one extra residual
per parameter encodes `-2 log prior`, so the minimum is the maximum a
posteriori point.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import least_squares

from ..solver import FitResults
from .abc import OptimiserBackend, _prior_cdf_clipped
from .config import LevenbergMarquardt

if TYPE_CHECKING:
    from ..solver import Solver


def _log_prior_residuals(
    prior, physical_params: np.ndarray, shift: np.ndarray
) -> np.ndarray:
    """Prior residuals `sqrt(2 * (shift_i - log_pdf_i(theta_i)))`, with `shift_i` an upper bound of each log-pdf."""
    physical_params = np.asarray(physical_params, dtype=np.float64).ravel()
    out = np.empty_like(physical_params)
    for j, dist in enumerate(prior.dists):
        excess = 2.0 * (shift[j] - float(dist.logpdf(physical_params[j])))
        out[j] = np.sqrt(max(0.0, excess))
    return out


def _delta_method_to_unconstrained(
    prior, params_phys: np.ndarray, cov_phys: np.ndarray
) -> np.ndarray:
    """Map a physical covariance to logit-of-CDF space with the diagonal Jacobian `pdf / (CDF (1 - CDF))`."""
    params_phys = np.asarray(params_phys, dtype=np.float64).ravel()
    cdf_vals = _prior_cdf_clipped(prior, params_phys)
    pdf_vals = np.array(
        [d.pdf(p) for d, p in zip(prior.dists, params_phys)],
        dtype=np.float64,
    )
    j_diag = pdf_vals / (cdf_vals * (1.0 - cdf_vals))
    return (j_diag[:, None] * cov_phys) * j_diag[None, :]


def covariance_from_jacobian(jac: np.ndarray) -> np.ndarray:
    """Laplace covariance `(J^T J)^-1` from the Jacobian of the residuals.

    The pseudo-inverse gives zero variance to flat directions instead of failing.
    """
    jac_pinv = np.linalg.pinv(jac)
    cov = jac_pinv @ jac_pinv.T
    return 0.5 * (cov + cov.T)


def _poisson_deviance_residuals(observed: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Signed square root of the per-bin Poisson deviance; the squares sum to XSPEC's C-statistic."""
    observed = np.asarray(observed, dtype=np.float64)
    model = np.asarray(model, dtype=np.float64)

    safe_model = np.clip(model, np.finfo(np.float64).tiny, None)
    deviance = np.empty_like(safe_model)

    zero_observed = observed == 0.0
    deviance[zero_observed] = 2.0 * safe_model[zero_observed]

    positive_observed = ~zero_observed
    obs = observed[positive_observed]
    pred = safe_model[positive_observed]
    deviance[positive_observed] = 2.0 * (obs * (np.log(obs) - np.log(pred)) + pred - obs)

    deviance = np.maximum(deviance, 0.0)
    return np.sign(observed - model) * np.sqrt(deviance)


class LevenbergMarquardtBackend(OptimiserBackend):
    """Best fit like XSPEC's `fit`, with the prior included, plus a Gaussian approximation around it.

    Attributes:
        result: SciPy's `OptimizeResult` after `run`.
    """

    config_cls = LevenbergMarquardt

    def __init__(self, *, solver: Solver, config: LevenbergMarquardt) -> None:
        super().__init__(solver=solver, config=config)
        self.result = None

    def run(self) -> FitResults:
        """Run the least-squares fit and return the best fit and its Gaussian approximation."""
        folds_before = sum(self.solver.fold_counts.values())
        observed = self.solver.observed_counts
        prior = self.solver.prior
        prior_shift = prior.max_log_pdf_per_axis()

        # Called once: with init_from_prior it draws from the prior.
        start_physical = self._starting_point()
        x0_physical = start_physical
        if x0_physical is None:
            x0_physical = prior.from_unit_cube(np.full(self.solver.num_parameters, 0.5))

        def residuals_physical(physical_params):
            physical_params = np.atleast_2d(physical_params)
            sim = self.solver.simulate(physical_params, return_kind="full_model_counts", progress_bar=False)
            r_data = _poisson_deviance_residuals(observed, sim["total_model_counts"].ravel())
            r_prior = _log_prior_residuals(prior, physical_params.ravel(), prior_shift)
            self.tracer.record(physical_params.ravel(), sim["cstat"])
            return np.concatenate([r_data, r_prior])

        if self.config.reparametrise:
            if start_physical is None:
                # u = 0 is the prior median. Exactly 0: MINPACK's first trust region is
                # 100 * |x0|, so logit(F(median)) ~ 1e-16 would stall the fit there.
                x0_search = np.zeros(self.solver.num_parameters)
            else:
                x0_search = self._physical_to_unconstrained(start_physical)

            def residuals_in_search_space(unconstrained_params):
                return residuals_physical(self._unconstrained_to_physical(unconstrained_params).ravel())

            options = dict(method="lm", verbose=1, jac="3-point")
        else:
            x0_search = x0_physical
            residuals_in_search_space = residuals_physical
            low, high = np.asarray(prior.bounds, dtype=np.float64).T
            options = dict(
                method="trf",
                verbose=1,
                jac="3-point",
                bounds=(low, high),
                x_scale="jac",
                diff_step=1e-2,
            )
        options.update(self.config.engine_kwargs)

        start = time.perf_counter()
        self.result = least_squares(residuals_in_search_space, x0_search, **options)
        seconds = time.perf_counter() - start

        if self.config.reparametrise:
            self.best_fit_unconstrained = self.result.x.copy()
            self.best_fit_params = self._unconstrained_to_physical(self.result.x).ravel()
            self.covariance_unconstrained = covariance_from_jacobian(self.result.jac)
        else:
            self.best_fit_params = self.result.x.copy()
            self.best_fit_unconstrained = self._physical_to_unconstrained(self.best_fit_params)
            self.covariance_unconstrained = _delta_method_to_unconstrained(
                prior, self.best_fit_params, covariance_from_jacobian(self.result.jac)
            )

        best_fit_stat = self._best_fit_stat()
        n_evaluations = sum(self.solver.fold_counts.values()) - folds_before
        return self._fit_results(seconds, n_evaluations, best_fit_stat)
