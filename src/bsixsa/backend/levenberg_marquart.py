from __future__ import annotations

import typing

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .abc import Backend
from . import register_backend
from .config import LevenbergMarquardt
from ..solver import FitResults

if typing.TYPE_CHECKING:
    from ..solver import Solver


def _delta_method_to_unconstrained(
    prior, params_phys: np.ndarray, cov_phys: np.ndarray
) -> np.ndarray:
    """Pushforward a physical-space covariance into unconstrained logit-of-CDF space.

    For independent priors, the unconstrained coordinate is
    ``u_i = logit(CDF_i(theta_i))``, so ``du_i/dtheta_i = pdf_i(theta_i) / (CDF_i(theta_i) * (1 - CDF_i(theta_i)))``,
    the change-of-variables Jacobian ``J`` is diagonal, and the
    delta-method covariance is ``Sigma_u = J Sigma_phys J^T``.

    Sampling in this unconstrained space and mapping back through
    ``Backend._unconstrained_to_physical`` yields posterior samples that are
    bounded by the prior support.
    """
    params_phys = np.asarray(params_phys, dtype=np.float64).ravel()
    cdf_vals = prior.to_unit_cube(np.atleast_2d(params_phys)).ravel()
    cdf_vals = np.clip(cdf_vals, 1e-25, 1.0 - 1e-25)
    pdf_vals = np.array(
        [d.pdf(p) for d, p in zip(prior.dists, params_phys)],
        dtype=np.float64,
    )
    j_diag = pdf_vals / (cdf_vals * (1.0 - cdf_vals))
    return (j_diag[:, None] * cov_phys) * j_diag[None, :]


def covariance_from_jacobian(jac: np.ndarray) -> np.ndarray:
    """Fisher-information covariance from a Poisson-deviance Jacobian.

    For deviance residuals of a Poisson likelihood the Hessian of ``-2 ln L``
    at the MLE is ``2 J^T J`` to leading order, so the Fisher-information
    covariance is ``(J^T J)^{-1}``.

    Rather than forming ``J^T J`` and inverting it (which squares the
    condition number and fails outright on rank-deficient Jacobians), we
    take the SVD-based Moore-Penrose pseudoinverse of *J* itself and use
    the identity ``J^+ (J^+)^T = (J^T J)^{-1}`` (in the full-rank case).
    On rank-deficient input this returns the pseudoinverse of ``J^T J``
    instead of raising — directions with no curvature get exactly zero
    variance, which propagates cleanly through the delta-method push to
    unconstrained space and through ``rng.multivariate_normal``.

    Parameters:
        jac: Jacobian at the solution, shape (m, n), as returned by
            ``least_squares``.

    Returns:
        Symmetric, finite covariance matrix of shape (n, n) — always.
    """
    jac_pinv = np.linalg.pinv(jac)
    cov = jac_pinv @ jac_pinv.T
    # Mathematically symmetric; symmetrise to scrub tiny float roundoff
    # before downstream `multivariate_normal` checks PSD-ness.
    return 0.5 * (cov + cov.T)


def _poisson_deviance_residuals(observed: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Signed square-root of the per-bin Poisson deviance.

    For binned Poisson data, ctools defines the negative log-likelihood as
    ``-ln L = sum_i(e_i - n_i ln e_i)``. The corresponding deviance residual is

    ``r_i = sign(n_i - e_i) * sqrt(2 * (n_i * ln(n_i / e_i) + e_i - n_i))``.

    We handle the ``n_i = 0`` branch analytically and clip ``e_i`` to a tiny
    positive value so the residual vector remains finite for SciPy's LM solver.
    """
    observed = np.asarray(observed, dtype=np.float64)
    model = np.asarray(model, dtype=np.float64)

    safe_model = np.clip(model, np.finfo(np.float64).tiny, None)
    deviance = np.empty_like(safe_model)

    zero_observed = observed == 0.0
    deviance[zero_observed] = 2.0 * safe_model[zero_observed]

    positive_observed = ~zero_observed
    if np.any(positive_observed):
        obs = observed[positive_observed]
        pred = safe_model[positive_observed]
        deviance[positive_observed] = 2.0 * (
            obs * (np.log(obs) - np.log(pred)) + pred - obs
        )

    deviance = np.maximum(deviance, 0.0)
    return np.sign(observed - model) * np.sqrt(deviance)


@register_backend
class LevenbergMarquardtBackend(Backend):
    """Point-estimate fitter using Levenberg-Marquardt on Poisson deviance.

    By default the search runs in *physical* parameter space with explicit
    prior bounds (matches XSPEC's ``leven``). The legacy logit-of-CDF
    reparametrisation is still available behind ``config.reparametrise=True``,
    which falls back to scipy's bounds-free ``method="lm"``.
    """

    name = "levenberg_marquardt"
    config_cls = LevenbergMarquardt

    def __init__(
        self,
        *,
        solver: Solver,
        config: LevenbergMarquardt,
    ):
        super().__init__(solver=solver, config=config)
        self.result = None
        self.best_fit_params = None
        self.best_fit_unconstrained = None
        self.covariance_unconstrained = None
        self.covariance_physical = None

    def sample(self, n: int) -> np.ndarray:
        """Sample the Laplace approximation of the posterior in unconstrained
        space, then map back to physical parameters.

        Sampling in unconstrained logit-of-CDF space ensures that all returned
        samples lie inside the prior support: a Gaussian on the line, no matter
        how wide, collapses through ``sigmoid`` and the prior inverse-CDF onto
        the bounded physical interval. This holds in both fit modes; the
        physical-space fit's covariance is pushed forward via the delta method
        in :func:`_delta_method_to_unconstrained`.

        ``covariance_from_jacobian`` is now pseudoinverse-based and always
        returns a finite matrix, so a rank-deficient fit yields zero variance
        in the un-identified directions rather than ``None`` — the resulting
        samples just collapse onto the best fit along those axes.
        """
        rng = np.random.default_rng()
        samples_unconstrained = rng.multivariate_normal(
            self.best_fit_unconstrained,
            self.covariance_unconstrained,
            size=n,
        )
        return self._unconstrained_to_physical(samples_unconstrained)

    def run(self) -> 'FitResults':
        """Run the Levenberg-Marquardt fit."""
        from ..convenience import catchtime

        n_params = self.solver.num_parameters
        observed = self.solver.observed_spectrum.astype(np.float64)
        x0_physical = self.config.x0_physical
        init_from_prior = self.config.init_from_prior
        reparametrise = self.config.reparametrise
        least_squares_kwargs = self.config.engine_kwargs.copy()

        # --- starting point in physical space ---
        if init_from_prior:
            x0_physical = self.solver.prior.sample(size=1).ravel()
        elif x0_physical is None:
            # Prior median (cube midpoint) as the neutral default.
            x0_physical = self.solver.prior.from_unit_cube(
                np.full(n_params, 0.5)
            ).ravel()
        x0_physical = np.asarray(x0_physical, dtype=np.float64).ravel()

        # --- residuals + scipy options depending on parametrisation ---
        if reparametrise:
            x0_search = self._physical_to_unconstrained(x0_physical)

            def residuals(unconstrained_params):
                physical_params = self._unconstrained_to_physical(
                    unconstrained_params
                )
                sim = self.solver.simulate(
                    physical_params,
                    return_kind="full_model_counts",
                    progress_bar=False,
                )
                model_counts = sim["total_model_counts"].ravel().astype(np.float64)
                r = _poisson_deviance_residuals(observed, model_counts)
                self.tracer.record(physical_params, sim["cstat"])
                return r

            defaults = dict(method="lm", verbose=1, jac="3-point")
        else:
            x0_search = x0_physical
            low = np.array(
                [b[0] for b in self.solver.prior.bounds], dtype=np.float64
            )
            high = np.array(
                [b[1] for b in self.solver.prior.bounds], dtype=np.float64
            )

            def residuals(physical_params):
                physical_params_2d = np.atleast_2d(physical_params)
                sim = self.solver.simulate(
                    physical_params_2d,
                    return_kind="full_model_counts",
                    progress_bar=False,
                )
                model_counts = sim["total_model_counts"].ravel().astype(np.float64)
                r = _poisson_deviance_residuals(observed, model_counts)
                self.tracer.record(physical_params_2d.ravel(), sim["cstat"])
                return r

            defaults = dict(
                method="trf",
                verbose=1,
                jac="3-point",
                bounds=(low, high),
                x_scale="jac",
            )

        defaults.update(least_squares_kwargs)
        x0_search = np.asarray(x0_search, dtype=np.float64).ravel()

        with catchtime("Running Levenberg-Marquardt", print_time=False) as run_time:
            self.result = least_squares(
                residuals,
                x0_search,
                **defaults,
            )

        # --- store best fit and covariance in both spaces ---
        if reparametrise:
            self.best_fit_unconstrained = self.result.x.copy()
            self.best_fit_params = self._unconstrained_to_physical(
                self.result.x
            ).ravel()
            self.covariance_unconstrained = covariance_from_jacobian(self.result.jac)
            self.covariance_physical = None
        else:
            self.best_fit_params = self.result.x.copy()
            self.best_fit_unconstrained = self._physical_to_unconstrained(
                self.best_fit_params
            )
            self.covariance_physical = covariance_from_jacobian(self.result.jac)
            self.covariance_unconstrained = _delta_method_to_unconstrained(
                self.solver.prior,
                self.best_fit_params,
                self.covariance_physical,
            )

        # Build posterior samples for the FitResults
        n_posterior = 10_000
        posterior_samples = self.sample(n_posterior)
        posterior_dict = {
            name: posterior_samples[:, i]
            for i, name in enumerate(self.solver.parameter_names)
        }

        best_fit = pd.Series(
            self.best_fit_params, index=self.solver.parameter_names
        )
        # sum of squared deviance residuals == C-statistic (see XSPEC manual).
        best_fit_stat = float(2.0 * self.result.cost)

        return FitResults(
            time=float(run_time()),
            posterior_samples=pd.DataFrame.from_dict(posterior_dict),
            n_likelihood_evaluations=self.tracer.n_evals,
            log_Z=float("nan"),
            log_Z_err=float("nan"),
            best_fit=best_fit,
            best_fit_stat=best_fit_stat,
        )
