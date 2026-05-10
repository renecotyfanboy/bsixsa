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


def _log_prior_residuals(
    prior, physical_params: np.ndarray, shift: np.ndarray
) -> np.ndarray:
    """Per-parameter residuals encoding the log-prior for least-squares MAP.

    For independent prior axis ``i``,
    ``r_prior_i = sqrt(2 * (shift_i - log_pdf_i(theta_i)))``,
    so concatenating these onto cstat deviance residuals turns LM into a
    MAP fit. ``shift`` is an upper bound on each axis's log-pdf
    (see :meth:`MultipleIndependent.max_log_pdf_per_axis`).
    """
    physical_params = np.asarray(physical_params, dtype=np.float64).ravel()
    out = np.empty_like(physical_params)
    for j, dist in enumerate(prior.dists):
        excess = 2.0 * (shift[j] - float(dist.logpdf(physical_params[j])))
        out[j] = np.sqrt(max(0.0, excess))
    return out


def _delta_method_to_unconstrained(
    prior, params_phys: np.ndarray, cov_phys: np.ndarray
) -> np.ndarray:
    """Pushforward a physical-space covariance to logit-of-CDF space.

    For independent priors the change-of-variables Jacobian is diagonal
    with ``du_i/dtheta_i = pdf_i(theta_i) / (CDF_i(theta_i) * (1 - CDF_i(theta_i)))``,
    so ``Sigma_u = J Sigma_phys J^T``.
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

    Uses the SVD-based Moore-Penrose pseudoinverse of ``J`` directly so the
    calculation stays well defined on rank-deficient input — directions
    with no curvature get exactly zero variance instead of raising. In the
    full-rank case ``J^+ (J^+)^T == (J^T J)^{-1}``.
    """
    jac_pinv = np.linalg.pinv(jac)
    cov = jac_pinv @ jac_pinv.T
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

    def sample(self, n: int) -> np.ndarray:
        """Sample the Laplace approximation in logit-of-CDF space, then map
        back to physical parameters. The bounded inverse-CDF mapping
        guarantees every sample lies inside the prior support."""
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

        prior = self.solver.prior
        prior_shift = prior.max_log_pdf_per_axis()

        # --- starting point in physical space ---
        if init_from_prior:
            x0_physical = prior.sample(size=1).ravel()
        elif x0_physical is None:
            x0_physical = prior.from_unit_cube(np.full(n_params, 0.5)).ravel()
        x0_physical = np.asarray(x0_physical, dtype=np.float64).ravel()

        # --- single residual closure on physical params (data + prior) ---
        def _residuals(physical_params):
            physical_params = np.atleast_2d(physical_params)
            sim = self.solver.simulate(
                physical_params,
                return_kind="full_model_counts",
                progress_bar=False,
            )
            model_counts = sim["total_model_counts"].ravel().astype(np.float64)
            r_data = _poisson_deviance_residuals(observed, model_counts)
            r_prior = _log_prior_residuals(
                prior, physical_params.ravel(), prior_shift
            )
            self.tracer.record(physical_params.ravel(), sim["cstat"])
            return np.concatenate([r_data, r_prior])

        # --- search-space wiring ---
        if reparametrise:
            x0_search = self._physical_to_unconstrained(x0_physical)

            def residuals(unconstrained_params):
                return _residuals(
                    self._unconstrained_to_physical(unconstrained_params).ravel()
                )

            defaults = dict(method="lm", verbose=1, jac="3-point")
        else:
            x0_search = x0_physical
            residuals = _residuals
            low = np.array([b[0] for b in prior.bounds], dtype=np.float64)
            high = np.array([b[1] for b in prior.bounds], dtype=np.float64)
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
            self.result = least_squares(residuals, x0_search, **defaults)

        # --- store best fit and unconstrained-space covariance ---
        if reparametrise:
            self.best_fit_unconstrained = self.result.x.copy()
            self.best_fit_params = self._unconstrained_to_physical(
                self.result.x
            ).ravel()
            self.covariance_unconstrained = covariance_from_jacobian(self.result.jac)
        else:
            self.best_fit_params = self.result.x.copy()
            self.best_fit_unconstrained = self._physical_to_unconstrained(
                self.best_fit_params
            )
            cov_phys = covariance_from_jacobian(self.result.jac)
            self.covariance_unconstrained = _delta_method_to_unconstrained(
                prior, self.best_fit_params, cov_phys
            )

        # --- bare cstat at the best fit (matches XSPEC's Fit.statistic) ---
        sim_at_best = self.solver.simulate(
            np.atleast_2d(self.best_fit_params),
            return_kind="cstat",
            progress_bar=False,
        )
        best_fit_stat = float(np.asarray(sim_at_best["cstat"]).ravel()[0])

        # --- posterior samples + FitResults ---
        n_posterior = 10_000
        posterior_samples = self.sample(n_posterior)
        posterior_dict = {
            name: posterior_samples[:, i]
            for i, name in enumerate(self.solver.parameter_names)
        }
        best_fit = pd.Series(
            self.best_fit_params, index=self.solver.parameter_names
        )

        return FitResults(
            time=float(run_time()),
            posterior_samples=pd.DataFrame.from_dict(posterior_dict),
            n_likelihood_evaluations=self.tracer.n_evals,
            log_Z=float("nan"),
            log_Z_err=float("nan"),
            best_fit=best_fit,
            best_fit_stat=best_fit_stat,
        )
