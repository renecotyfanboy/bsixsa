from __future__ import annotations

import typing

import numpy as np
from scipy.special import expit as sigmoid
from scipy.optimize import least_squares

from .abc import Sampler

if typing.TYPE_CHECKING:
    from ..solver import SIXSASolver


# Tiny floor to avoid log(0) and division by zero in Poisson deviance
_EPS = 1e-25


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


class LevenbergMarquardtSampler(Sampler):
    """Point-estimate fitter using Levenberg-Marquardt on Poisson deviance.

    The optimisation is carried out in the *unit-cube* parameter space so
    that all parameters live on [0, 1] and box bounds are trivially enforced.
    """

    def __init__(self, *, solver: SIXSASolver, **kwargs):
        super().__init__(solver=solver)
        self.result = None
        self.best_fit_params = None
        self.best_fit_unconstrained = None
        self.covariance_unconstrained = None

    # -- Sampler ABC ---------------------------------------------------------
    def sample(self, shape):
        if self.best_fit_unconstrained is None:
            raise RuntimeError("Must call run() before sample().")

        n = int(shape[0]) if hasattr(shape, "__len__") else int(shape)

        if self.covariance_unconstrained is not None:
            rng = np.random.default_rng()
            samples_unconstrained = rng.multivariate_normal(
                self.best_fit_unconstrained, self.covariance_unconstrained, size=n
            )
        else:
            samples_unconstrained = np.tile(self.best_fit_unconstrained, (n, 1))

        samples_unit_cube = sigmoid(samples_unconstrained)
        return self.solver.prior.from_unit_cube(samples_unit_cube)

    def _physical_to_unconstrained(self, physical: np.ndarray) -> np.ndarray:
        """Map physical-space parameters to unconstrained space (logit ∘ cdf)."""
        unit_cube = self.solver.prior.to_unit_cube(
            np.atleast_2d(physical)
        ).ravel()
        unit_cube = np.clip(unit_cube, _EPS, 1.0 - _EPS)
        return np.log(unit_cube / (1.0 - unit_cube))

    def run(self, *, x0_physical: np.ndarray | None = None, init_from_prior: bool = False, **least_squares_kwargs):
        """Run the Levenberg-Marquardt fit.

        Parameters
        ----------
        x0_physical : array-like, optional
            Initial guess in physical parameter space.  Defaults to the
            prior median (0.5 in the unit cube).
        init_from_prior : bool, optional
            If ``True``, draw a random sample from the prior and use it as
            the starting point (overrides ``x0_physical``).
        **least_squares_kwargs
            Extra keyword arguments forwarded to
            :func:`scipy.optimize.least_squares` (e.g. ``ftol``, ``xtol``,
            ``max_nfev``).
        """
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

        def residuals(unconstrained_params):
            unit_cube_params = sigmoid(unconstrained_params)
            physical_params = self.solver.prior.from_unit_cube(
                unit_cube_params[np.newaxis]
            )
            sim = self.solver.simulate(
                physical_params,
                return_kind="full_model_counts",
                progress_bar=False,
            )
            model_counts = sim["total_model_counts"].ravel().astype(np.float64)
            return _poisson_deviance_residuals(observed, model_counts)

        defaults = dict(method="lm", verbose=1, jac="3-point")
        defaults.update(least_squares_kwargs)

        self.result = least_squares(
            residuals,
            x0_unconstrained,
            **defaults
        )

        self.best_fit_unconstrained = self.result.x.copy()
        self.best_fit_params = self.solver.prior.from_unit_cube(
            sigmoid(self.result.x)[np.newaxis]
        ).ravel()

        # Approximate covariance in unconstrained space from the Jacobian
        jac = self.result.jac
        jtj = jac.T @ jac
        try:
            jtj_inv = np.linalg.inv(jtj)
            n_bins = len(observed)
            s2 = self.result.cost * 2.0 / max(n_bins - n_params, 1)
            self.covariance_unconstrained = jtj_inv * s2
        except np.linalg.LinAlgError:
            self.covariance_unconstrained = None

        return self.result
