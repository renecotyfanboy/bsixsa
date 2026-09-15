import inspect
import typing

import numpy as np
from nautilus import Prior, Sampler as NestedSampler

from ..convenience import available_cpu_count
from .abc import Backend
from . import register_backend
from .config import Nautilus
from ..solver import FitResults


if typing.TYPE_CHECKING:
    from ..solver import Solver


def _accepts_keyword(function, name: str) -> bool:
    """Whether *function* can be called with the keyword argument *name*.

    Parameters:
        function: Callable to inspect (e.g. a class, for its constructor).
        name: Keyword argument name.

    Returns:
        `True` if the signature names *name* or takes `**kwargs`, or if the
        signature cannot be inspected (e.g. a mock); `False` otherwise.
    """
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return True
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


@register_backend
class NautilusBackend(Backend):

    name = "nautilus"
    config_cls = Nautilus

    def __init__(
        self,
        *,
        solver: 'Solver',
        config: Nautilus,
    ):
        super().__init__(solver=solver, config=config)

        prior = Prior()

        # Names come from the solver (not the xspec session) so that they stay
        # aligned with its working prior, which a restriction may have boxed.
        for name, dist in zip(self.solver.parameter_names, self.solver.prior.dists):
            prior.add_parameter(name, dist)

        self.n_live_points = config.num_live_points

        def likelihood(x):
            ll = solver.log_likelihood_fn(x, None, progress_bar=False, no_pool=False)
            self.tracer.record(np.atleast_2d(x), -2.0 * np.atleast_1d(ll))
            return ll

        n_pool = min(config.n_pool, available_cpu_count())

        sampler_kwargs = dict(
            n_live=config.num_live_points,
            vectorized=True,
            pass_dict=False,
            n_batch=config.n_batch,
            n_networks=config.n_networks,
            pool=(None, n_pool),
        )
        # Older nautilus releases take no `seed`.
        if _accepts_keyword(NestedSampler, "seed"):
            sampler_kwargs["seed"] = config.seed

        self._nested_sampler = NestedSampler(prior, likelihood, **sampler_kwargs)

    def run(self, **kwargs) -> 'FitResults':
        from ..solver import FitResults
        from ..convenience import catchtime

        with catchtime("Running nested sampler", print_time=False) as run_time:
            self._nested_sampler.run(verbose=True, discard_exploration=True)

        points = np.asarray(self._nested_sampler.posterior(equal_weight=True)[0], dtype=np.float64)

        return FitResults(
            time=float(run_time()),
            posterior_samples=self._posterior_dataframe(points),
            n_likelihood_evaluations=self._nested_sampler.n_like,
            log_Z=float(self._nested_sampler.log_z),
            log_Z_err=float(1 / np.sqrt(self._nested_sampler.n_eff)),
        )

    def sample(self, n: int) -> np.ndarray:
        """Draw up to *n* rows of the equal-weight posterior, in random order.

        The rows are a seeded random subset rather than the head: nautilus
        returns its equal-weight points grouped by bound, so a head is not a
        representative sample.

        Parameters:
            n: Number of samples requested.

        Returns:
            Array of shape `(min(n, n_available), d)`, seeded with
            `config.seed` (fresh entropy when it is `None`).
        """
        samples = np.asarray(self._nested_sampler.posterior(equal_weight=True)[0])
        size = min(int(n), samples.shape[0])
        order = np.random.default_rng(self.config.seed).permutation(samples.shape[0])
        return samples[order[:size]]
