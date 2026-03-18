from __future__ import annotations

import typing
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from scipy.optimize import least_squares

from .abc import Backend
from . import register_backend
from ..solver import FitResults

if typing.TYPE_CHECKING:
    from ..solver import SIXSASolver


@dataclass
class LMTrace:
    """Accumulates per-iteration diagnostics during a Levenberg-Marquardt run.

    With finite-difference Jacobians, each iteration triggers many residual
    evaluations at tiny perturbations of the iterate.  Only evaluations at
    genuinely new iterates are recorded; perturbation calls (distance from
    the last recorded point below ``jac_eps``) are counted but not stored.
    """

    params: list[np.ndarray] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    n_evals: int = 0
    jac_eps: float = 1e-6
    _last_recorded: np.ndarray | None = field(default=None, repr=False)

    def __call__(self, params: np.ndarray, residuals: np.ndarray) -> None:
        """Callback invoked after each residual evaluation."""
        self.n_evals += 1

        # Skip Jacobian perturbation calls — tiny shifts from last iterate
        if self._last_recorded is not None:
            diff = np.max(np.abs(params.ravel() - self._last_recorded))
            if diff < self.jac_eps:
                return

        self._last_recorded = params.ravel().copy()
        self.params.append(params.copy())
        self.costs.append(float(np.sum(residuals ** 2)))


# Tiny floor to avoid log(0) and division by zero in Poisson deviance
_EPS = 1e-25


def covariance_from_jacobian(
    jac: np.ndarray, cost: float, n_params: int
) -> np.ndarray | None:
    """Estimate the parameter covariance from a least-squares Jacobian.

    Uses the Gauss-Newton approximation: ``H ≈ J^T J``, then scales the
    inverse by the reduced chi-squared (``2 * cost / dof``).

    Args:
        jac: Jacobian at the solution, shape (m, n), as returned by
            ``least_squares``.
        cost: Value of the cost function at the solution (``0.5 * sum(r**2)``).
        n_params: Number of fitted parameters (used to compute degrees of
            freedom).

    Returns:
        Covariance matrix of shape (n, n), or None if the Hessian is singular.
    """
    jtj = jac.T @ jac
    try:
        jtj_inv = np.linalg.inv(jtj)
    except np.linalg.LinAlgError:
        return None
    n_bins = jac.shape[0]
    s2 = cost * 2.0 / max(n_bins - n_params, 1)
    return jtj_inv * s2


def _poisson_deviance_residuals(observed: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Signed square-root of the per-bin Poisson deviance (Cash C-statistic).

    Minimising ``sum(r**2)`` is equivalent to minimising the C-statistic,
    i.e. maximising the Poisson log-likelihood.

    For a bin with observed counts *y* and predicted counts *µ*:

        d_i = 2 (µ − y + y ln(y / µ))

    The signed residual is ``sign(y − µ) sqrt(d_i)``.
    """
    model = np.maximum(model, _EPS)
    observed = np.asarray(observed, dtype=np.float64)

    deviance = np.empty_like(observed)
    nonzero = observed > 0
    deviance[nonzero] = 2.0 * (
        model[nonzero]
        - observed[nonzero]
        + observed[nonzero] * np.log(observed[nonzero] / model[nonzero])
    )
    deviance[~nonzero] = 2.0 * model[~nonzero]

    deviance = np.maximum(deviance, 0.0)
    return np.sign(observed - model) * np.sqrt(deviance)


@register_backend
class LevenbergMarquardtBackend(Backend):
    """Point-estimate fitter using Levenberg-Marquardt on Poisson deviance.

    The optimisation is carried out in the *unit-cube* parameter space so
    that all parameters live on [0, 1] and box bounds are trivially enforced.
    """

    name = "levenberg_marquardt"

    def __init__(self, *, solver: SIXSASolver, **kwargs):
        super().__init__(solver=solver)
        self.result = None
        self.best_fit_params = None
        self.best_fit_unconstrained = None
        self.covariance_unconstrained = None
        self.trace: LMTrace | None = None

    def sample(self, n: int) -> np.ndarray:

        if self.covariance_unconstrained is not None:
            rng = np.random.default_rng()
            samples_unconstrained = rng.multivariate_normal(
                self.best_fit_unconstrained, self.covariance_unconstrained, size=n
            )

        else:
            samples_unconstrained = np.tile(self.best_fit_unconstrained, (n, 1))

        return self._unconstrained_to_physical(samples_unconstrained)

    def _unconstrained_to_physical(self, unconstrained: np.ndarray) -> np.ndarray:
        """Map unconstrained-space parameters back to physical space (sigmoid then inverse cdf)."""
        unit_cube = sigmoid(np.atleast_2d(unconstrained))
        return self.solver.prior.from_unit_cube(unit_cube)

    def _physical_to_unconstrained(self, physical: np.ndarray) -> np.ndarray:
        """Map physical-space parameters to unconstrained space (logit than cdf)."""
        unit_cube = self.solver.prior.to_unit_cube(
            np.atleast_2d(physical)
        ).ravel()
        unit_cube = np.clip(unit_cube, _EPS, 1.0 - _EPS)
        return np.log(unit_cube / (1.0 - unit_cube))

    def run(self, *, x0_physical: np.ndarray | None = None, init_from_prior: bool = False, **least_squares_kwargs) -> 'FitResults':
        """Run the Levenberg-Marquardt fit.

        Args:
            x0_physical: Initial guess in physical parameter space. Defaults to
                the prior median (0.5 in the unit cube).
            init_from_prior: If ``True``, draw a random sample from the prior
                and use it as the starting point (overrides ``x0_physical``).
            **least_squares_kwargs: Extra keyword arguments forwarded to
                ``scipy.optimize.least_squares`` (e.g. ``ftol``, ``xtol``,
                ``max_nfev``).
        """
        from ..convenience import catchtime

        n_params = self.solver.num_parameters
        observed = self.solver.observed_spectrum.astype(np.float64)

        if init_from_prior:
            x0_physical = self.solver.prior.sample(size=1).ravel()

        if x0_physical is not None:
            x0_unconstrained = self._physical_to_unconstrained(
                np.asarray(x0_physical, dtype=np.float64)
            )
        else:
            x0_unconstrained = np.full(n_params, 0.0)

        x0_unconstrained = np.asarray(x0_unconstrained, dtype=np.float64).ravel()

        callback = LMTrace()

        def residuals(unconstrained_params):
            physical_params = self._unconstrained_to_physical(unconstrained_params)
            sim = self.solver.simulate(
                physical_params,
                return_kind="full_model_counts",
                progress_bar=False,
            )
            model_counts = sim["total_model_counts"].ravel().astype(np.float64)
            r = _poisson_deviance_residuals(observed, model_counts)
            callback(physical_params, sim["cstat"])
            return r

        defaults = dict(method="lm", verbose=1, jac="3-point")
        defaults.update(least_squares_kwargs)

        with catchtime("Running Levenberg-Marquardt", print_time=False) as run_time:
            self.result = least_squares(
                residuals,
                x0_unconstrained,
                **defaults
            )

        self.trace = callback

        self.best_fit_unconstrained = self.result.x.copy()
        self.best_fit_params = self._unconstrained_to_physical(
            self.result.x
        ).ravel()

        self.covariance_unconstrained = covariance_from_jacobian(
            self.result.jac, self.result.cost, n_params
        )

        # Build posterior samples for the FitResults
        n_posterior = 10_000
        posterior_samples = self.sample(n_posterior)
        posterior_dict = {
            name: posterior_samples[:, i]
            for i, name in enumerate(self.solver.parameter_names)
        }

        return FitResults(
            time=float(run_time()),
            posterior_samples=pd.DataFrame.from_dict(posterior_dict),
            n_likelihood_evaluations=self.trace.n_evals,
            log_Z=float("nan"),
            log_Z_err=float("nan"),
        )
