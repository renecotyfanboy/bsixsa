"""Regression guard for the benchmark's best-fit statistic convention."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from bsixsa.priors import MultipleIndependent, loguniform


def test_reference_refinement_includes_the_xspec_prior_term():
    path = Path(__file__).resolve().parents[1] / "benchmark/sixsa_vs_nautilus_synthetic.py"
    spec = importlib.util.spec_from_file_location("sixsa_benchmark", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)

    class Solver:
        base_prior = MultipleIndependent([loguniform(1e-4, 1e-2)])
        num_parameters = 1

        def simulate(self, theta, **kwargs):
            log_ratio = np.log(np.asarray(theta)[:, 0] / 1e-3)
            return {"cstat": 12.0 + (log_ratio / 0.1) ** 2}

    # C = 12 + (log(theta/theta0)/s)^2 and prior term = 2 log(theta).
    # The minimum is at log(theta/theta0) = -s^2, not at theta0.
    expected = 12 + 2 * np.log(1e-3) - 0.1**2
    actual = benchmark.reference_best_fit_stat(Solver(), np.array([1e-3]))
    assert actual == pytest.approx(expected, abs=1e-7)
