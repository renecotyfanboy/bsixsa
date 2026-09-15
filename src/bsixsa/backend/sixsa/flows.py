r"""Neural proposals of SIXSA: trained flows, their mixture, and parallel training.

Every proposal here follows the raw protocol of the certificates. `sample_raw`
returns the estimator's draws as they are, *not* rejected to the unit cube, and
`log_prob_raw` returns the estimator's un-truncated log density, so an
importance weight

$$\log w = -C/2 - \log q_{raw}(u)$$

with zero weight outside the cube estimates the evidence without any leakage
correction: the draws that leak out of the cube are counted in $N$ and weigh
nothing.

`train_members` trains the ensemble in loky worker processes. The worker code
lives in [`_nde_worker`][bsixsa.backend.sixsa._nde_worker] and is pickled by
value, so the workers never import `bsixsa` (nor XSPEC).
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import joblib
import numpy as np
import torch
from scipy.special import logsumexp

from ..._cube import child_seed
from ..._torch_utils import torch_threads
from . import _nde_worker

__all__ = [
    "FlowProposal",
    "MixtureProposal",
    "member_seeds",
    "shutdown_workers",
    "train_members",
]

# Purpose key of the member seeds under the run's SeedSequence.
_MEMBER_SEED_PURPOSE = 10
# Seed of the one throwaway draw that reveals a proposal's dimension.
_PROBE_SEED = 0

_worker_registered = False


def _probe_dimension(proposal: Any) -> int:
    """Dimension of a proposal without a usable ``d`` attribute.

    Parameters:
        proposal:
            Object with ``sample_raw(n, rng)``.

    Returns:
        The number of columns of one raw draw, made with a private generator.
    """
    draw = np.asarray(proposal.sample_raw(1, np.random.default_rng(_PROBE_SEED)))
    return int(draw.reshape(1, -1).shape[1])


class FlowProposal:
    r"""A trained conditional estimator $q(u \mid z)$ frozen at the observed code.

    Parameters:
        estimator:
            sbi density estimator (e.g. a ``ZukoFlow``) with ``sample(shape,
            condition=...)`` returning ``(m, 1, d)`` and ``log_prob(inputs,
            condition=...)`` returning ``(m, 1)`` for inputs ``(m, 1, d)``.
        z_obs:
            Compressed observation the flow is conditioned on, any shape with
            ``n_features`` elements.
        name:
            Label of the proposal.
        generation:
            Compressor generation that produced *z_obs*, kept so the flow and
            the code it is conditioned on cannot be separated.
        batch:
            Maximum number of rows per estimator call.

    Raises:
        ValueError: If *z_obs* is empty or *batch* is smaller than 1.
    """

    def __init__(
        self,
        estimator: Any,
        z_obs: np.ndarray,
        *,
        name: str,
        generation: int | None = None,
        batch: int = 50_000,
    ) -> None:
        z = np.array(z_obs, dtype=np.float64, copy=True).reshape(-1)
        if z.size == 0:
            raise ValueError("z_obs must not be empty.")
        if int(batch) < 1:
            raise ValueError(f"batch must be >= 1, got {batch}.")
        self.estimator = estimator
        self.z_obs = z
        self.name = str(name)
        self.generation = generation
        self.batch = int(batch)
        self._condition = torch.as_tensor(z, dtype=torch.float32).reshape(1, -1)
        shape = getattr(estimator, "input_shape", None)
        self._d: int | None = int(np.prod(tuple(shape))) if shape is not None else None

    @property
    def d(self) -> int:
        """Dimension of the parameter space."""
        if self._d is None:
            self._d = _probe_dimension(self)
        return self._d

    def __repr__(self) -> str:
        return f"FlowProposal(name={self.name!r}, d={self._d}, generation={self.generation})"

    def sample_raw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw from the flow, without rejecting the draws outside the cube.

        The torch global generator is forked and seeded from *rng*, so the
        draws are a pure function of *rng*'s state and the caller's torch
        random state is left untouched.

        Parameters:
            n:
                Number of draws.
            rng:
                Generator consumed for one torch seed.

        Returns:
            float64 array of shape ``(n, d)``.

        Raises:
            ValueError: If *n* is negative.
        """
        n = int(n)
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}.")
        # Consumed even for n == 0, so a caller's stream does not depend on
        # which requests happened to be empty.
        seed = int(rng.integers(2**63 - 1))
        if n == 0:
            return np.empty((0, self.d), dtype=np.float64)
        parts = []
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.default_generator.manual_seed(seed)
            for start in range(0, n, self.batch):
                m = min(self.batch, n - start)
                draws = self.estimator.sample((m,), condition=self._condition)
                parts.append(draws.reshape(m, -1).to(torch.float64).numpy())
        if self._d is None:
            self._d = int(parts[0].shape[1])
        return np.concatenate(parts, axis=0)

    def log_prob_raw(self, u: np.ndarray) -> np.ndarray:
        """Un-truncated log density of the flow at *u*.

        Parameters:
            u:
                Points of shape ``(n, d)``, inside the cube or not. A 1-D array
                is one point, or ``n`` points when ``d == 1``.

        Returns:
            float64 array of shape ``(n,)``; ``-inf`` on rows with a
            non-finite coordinate (or one that overflows float32).

        Raises:
            ValueError: If *u* does not have ``d`` columns.
        """
        d = self.d
        arr = np.asarray(u, dtype=np.float64)
        valid = (arr.ndim == 2 and arr.shape[1] == d) or (
            arr.ndim == 1 and (d == 1 or arr.size == d)
        )
        if not valid:
            raise ValueError(f"u must have shape (n, {d}), got {arr.shape}.")
        with np.errstate(over="ignore", invalid="ignore"):
            points = arr.reshape(-1, d).astype(np.float32)
        out = np.full(points.shape[0], -np.inf, dtype=np.float64)
        rows = np.flatnonzero(np.isfinite(points).all(axis=1))
        with torch.no_grad():
            for start in range(0, rows.size, self.batch):
                idx = rows[start : start + self.batch]
                inputs = torch.from_numpy(points[idx]).reshape(-1, 1, d)
                values = self.estimator.log_prob(inputs, condition=self._condition)
                out[idx] = values.reshape(-1).to(torch.float64).numpy()
        return out


class MixtureProposal:
    r"""Mixture of raw proposals, sampled in strata.

    With $K$ members, the equal-weight density is

    $$\log q(u) = \operatorname{logsumexp}_k \log q_k(u) - \log K,$$

    the log of the mean density, not the mean of the log densities (the
    geometric mean, which would bias every weight). Draws are stratified:
    ``sample_raw(n)`` takes exactly ``n // K`` draws from each member, plus one
    from each of the first ``n % K``, concatenated in member order. The
    realised allocation is kept in ``last_counts``; the density that matches
    it exactly is [`log_prob_stratified`][bsixsa.backend.sixsa.flows.MixtureProposal.log_prob_stratified].

    Parameters:
        members:
            Raw proposals sharing the same dimension.
        name:
            Label of the mixture.

    Raises:
        ValueError: If *members* is empty or the members' dimensions differ.
    """

    def __init__(self, members: Sequence[Any], name: str = "mixture") -> None:
        self.members = list(members)
        if not self.members:
            raise ValueError("A mixture needs at least one member.")
        self.name = str(name)
        self.last_counts: list[int] | None = None
        dims = {int(m.d) for m in self.members if getattr(m, "d", None) is not None}
        if len(dims) > 1:
            raise ValueError(f"Mixture members have different dimensions: {sorted(dims)}.")
        self._d: int | None = dims.pop() if dims else None

    @property
    def d(self) -> int:
        """Dimension of the parameter space."""
        if self._d is None:
            self._d = _probe_dimension(self.members[0])
        return self._d

    def __len__(self) -> int:
        return len(self.members)

    def __repr__(self) -> str:
        return f"MixtureProposal(name={self.name!r}, members={len(self.members)})"

    def allocation(self, n: int) -> list[int]:
        """Deterministic stratified allocation of *n* draws over the members.

        Parameters:
            n:
                Total number of draws.

        Returns:
            ``[n // K + (k < n % K) for k in range(K)]``, which sums to *n*.

        Raises:
            ValueError: If *n* is negative.
        """
        n = int(n)
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}.")
        k_members = len(self.members)
        return [n // k_members + int(k < n % k_members) for k in range(k_members)]

    def _check_counts(self, counts: Sequence[int]) -> list[int]:
        values = [int(c) for c in counts]
        if len(values) != len(self.members):
            raise ValueError(
                f"counts must have one entry per member ({len(self.members)}), got {len(values)}."
            )
        if any(c < 0 for c in values):
            raise ValueError(f"counts must be non-negative, got {values}.")
        return values

    def sample_raw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw *n* stratified raw points (see ``allocation``).

        Parameters:
            n:
                Number of draws.
            rng:
                Generator passed to every member in turn.

        Returns:
            float64 array of shape ``(n, d)``, member 0's draws first.

        Raises:
            ValueError: If *n* is negative.
        """
        return self.sample_stratified(self.allocation(n), rng)

    def sample_stratified(self, counts: Sequence[int], rng: np.random.Generator) -> np.ndarray:
        """Concatenate ``counts[k]`` raw draws of member ``k``, in member order.

        Parameters:
            counts:
                Number of draws per member.
            rng:
                Generator passed to every member in turn.

        Returns:
            float64 array of shape ``(sum(counts), d)``.

        Raises:
            ValueError: If *counts* does not have one non-negative entry per
                member.
        """
        values = self._check_counts(counts)
        d = self.d
        parts = [
            np.asarray(member.sample_raw(c, rng), dtype=np.float64).reshape(c, d)
            for member, c in zip(self.members, values, strict=True)
            if c > 0
        ]
        self.last_counts = values
        if not parts:
            return np.empty((0, d), dtype=np.float64)
        return np.concatenate(parts, axis=0)

    def log_prob_members(self, u: np.ndarray) -> np.ndarray:
        """Raw log density of every member at *u*.

        Parameters:
            u:
                Points of shape ``(n, d)``.

        Returns:
            float64 array of shape ``(K, n)``.
        """
        points = np.asarray(u, dtype=np.float64).reshape(-1, self.d)
        return np.stack(
            [np.asarray(m.log_prob_raw(points), dtype=np.float64).reshape(-1) for m in self.members]
        )

    def log_prob_raw(self, u: np.ndarray) -> np.ndarray:
        r"""Equal-weight mixture density, $\operatorname{logsumexp}_k \log q_k - \log K$.

        Parameters:
            u:
                Points of shape ``(n, d)``.

        Returns:
            float64 array of shape ``(n,)``.
        """
        log_q = self.log_prob_members(u)
        return logsumexp(log_q, axis=0) - np.log(len(self.members))

    def log_prob_stratified(self, u: np.ndarray, counts: Sequence[int]) -> np.ndarray:
        r"""Density of the stratified design that drew ``counts[k]`` points from member ``k``.

        $$\log q(u) = \operatorname{logsumexp}_k \left[\log q_k(u) + \log(n_k / N)\right]$$

        With these weights $\sum_i w_i$ over the pooled stratified draws is an
        unbiased estimate of $N Z$, whatever the allocation. Members with
        ``counts[k] == 0`` are not evaluated.

        Parameters:
            u:
                Points of shape ``(n, d)``.
            counts:
                Draws per member.

        Returns:
            float64 array of shape ``(n,)``.

        Raises:
            ValueError: If *counts* does not have one non-negative entry per
                member, or sums to zero.
        """
        values = self._check_counts(counts)
        total = sum(values)
        if total == 0:
            raise ValueError("counts must not all be zero.")
        points = np.asarray(u, dtype=np.float64).reshape(-1, self.d)
        terms = [
            np.asarray(m.log_prob_raw(points), dtype=np.float64).reshape(-1) + np.log(c / total)
            for m, c in zip(self.members, values, strict=True)
            if c > 0
        ]
        return logsumexp(np.stack(terms), axis=0)


def member_seeds(seed_seq: np.random.SeedSequence, round_index: int, n: int) -> list[int]:
    """Torch seeds of the ensemble members of one round.

    Parameters:
        seed_seq:
            Root seed sequence of the run.
        round_index:
            Round number, in $[0, 2^{32})$.
        n:
            Number of members.

    Returns:
        ``[child_seed(seed_seq, 10, round_index, k) for k in range(n)]``.

    Raises:
        ValueError: If *n* is negative or *round_index* is out of range.
    """
    if int(n) < 0:
        raise ValueError(f"n must be >= 0, got {n}.")
    return [child_seed(seed_seq, _MEMBER_SEED_PURPOSE, int(round_index), k) for k in range(int(n))]


def _register_worker() -> None:
    """Make loky ship `_nde_worker` by value (idempotent)."""
    global _worker_registered
    if not _worker_registered:
        from joblib.externals.cloudpickle import register_pickle_by_value

        register_pickle_by_value(_nde_worker)
        _worker_registered = True


def _effective_jobs(n_jobs: int | None, n_members: int) -> int:
    """Number of worker processes for *n_members* trainings.

    Parameters:
        n_jobs:
            ``None`` for one per CPU, a positive count, or a negative joblib
            count (``-1`` = all CPUs).
        n_members:
            Number of members to train.

    Returns:
        A count in ``[1, n_members]``.

    Raises:
        ValueError: If *n_jobs* is 0.
    """
    if n_jobs is None:
        jobs = os.cpu_count() or 1
    else:
        jobs = int(n_jobs)
        if jobs == 0:
            raise ValueError("n_jobs must not be 0.")
        if jobs < 0:
            jobs = joblib.effective_n_jobs(jobs)
    return max(1, min(n_members, jobs))


def train_members(
    theta_u: np.ndarray,
    z: np.ndarray,
    *,
    nde: dict,
    training: dict,
    seeds: Sequence[int],
    n_jobs: int | None = None,
    threads_per_member: int = 1,
    log: Callable[[str], None] | None = None,
) -> tuple[list, list[dict]]:
    """Train one NPE member per seed, in parallel when possible.

    With more than one job the members are trained in loky worker processes,
    one torch thread each by default (more threads per member are slower on
    small flows). When the pool fails for any reason other than an interrupt,
    a ``RuntimeWarning`` is emitted and every member is retrained in-process
    with the same seeds, so the result does not depend on which path ran. The
    caller's torch thread count and random state are unchanged.

    Parameters:
        theta_u:
            Training parameters in the unit cube, shape ``(n, d)``.
        z:
            Matching compressed data, shape ``(n, n_features)``.
        nde:
            Keyword arguments of ``sbi.neural_nets.posterior_nn``.
        training:
            Keyword arguments of ``NPE_C.train``.
        seeds:
            One torch seed per member (see ``member_seeds``).
        n_jobs:
            Worker processes; ``None`` means ``min(len(seeds), os.cpu_count())``.
            1 trains in-process.
        threads_per_member:
            Intra-op torch threads of each training.
        log:
            Optional logger of the fallback message.

    Returns:
        ``(estimators, infos)`` in seed order; see
        [`train_member`][bsixsa.backend.sixsa._nde_worker.train_member]. Each
        info also holds ``seed`` and ``executor`` (``"loky"`` or
        ``"in_process"``).

    Raises:
        ValueError: If *seeds* is empty, *n_jobs* is 0, *threads_per_member*
            is smaller than 1, or *theta_u* and *z* do not have one row per
            training pair.
    """
    seed_list = [int(s) for s in seeds]
    if not seed_list:
        raise ValueError("seeds must not be empty.")
    threads = int(threads_per_member)
    if threads < 1:
        raise ValueError(f"threads_per_member must be >= 1, got {threads}.")
    jobs = _effective_jobs(n_jobs, len(seed_list))
    theta = np.ascontiguousarray(theta_u, dtype=np.float32)
    data = np.ascontiguousarray(z, dtype=np.float32)
    # Checked here so that a malformed training set raises at once instead of
    # breaking the pool and being retrained in-process first.
    if theta.ndim != 2 or data.ndim not in (1, 2) or data.shape[0] != theta.shape[0]:
        raise ValueError(
            f"theta_u must have shape (n, d) and z shape (n, n_features), "
            f"got {theta.shape} and {data.shape}."
        )

    results = None
    executor = "in_process"
    if jobs > 1:
        _register_worker()
        try:
            with joblib.parallel_config(backend="loky", inner_max_num_threads=threads):
                results = joblib.Parallel(n_jobs=jobs)(
                    joblib.delayed(_nde_worker.train_member)(theta, data, nde, training, s, threads)
                    for s in seed_list
                )
            executor = "loky"
        except Exception as exc:
            message = (
                f"[sixsa] parallel NDE training failed ({type(exc).__name__}: {exc}); "
                f"retraining the {len(seed_list)} members in-process with the same seeds."
            )
            if log is not None:
                log(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            results = None

    if results is None:
        with torch_threads(threads):
            results = [
                _nde_worker.train_member(theta, data, nde, training, s, threads) for s in seed_list
            ]

    estimators = []
    infos = []
    for seed, (estimator, info) in zip(seed_list, results, strict=True):
        estimators.append(estimator)
        infos.append({**info, "seed": seed, "executor": executor})
    return estimators, infos


def shutdown_workers() -> None:
    """Shut down the reusable loky pool that holds the training workers.

    Idle workers otherwise stay alive (and keep torch in memory) until loky's
    idle timeout. Errors are ignored: there may be no pool at all.
    """
    try:
        from joblib.externals.loky import get_reusable_executor

        # reuse=True returns the live pool as is; the default would compare
        # joblib's settings with ours and build a second pool first.
        get_reusable_executor(reuse=True).shutdown(wait=True)
    except Exception:
        pass
