from __future__ import annotations

import pytest

from bsixsa import Nautilus, Nessai, Solver

pytestmark = pytest.mark.xspec


def test_low_resolution_nautilus_notebook_setup_builds_solver(
    low_res_xspec_setup,
    low_res_prior,
    tmp_path,
):
    solver = Solver(
        low_res_prior,
        outputfiles_basename=str(tmp_path / "nautilus-low-res"),
        overwrite=True,
        n_jobs=1,
        backend=Nautilus(trace=False, num_live_points=64, n_batch=8),
    )

    assert solver.backend_name == "nautilus"
    assert solver.num_parameters == len(low_res_prior)
    assert solver.observed_counts.ndim == 1
    assert solver.observed_counts.size > 0


def test_high_resolution_nessai_notebook_setup_builds_solver(
    high_res_xspec_setup,
    high_res_prior,
    tmp_path,
):
    solver = Solver(
        high_res_prior,
        outputfiles_basename=str(tmp_path / "nessai-high-res"),
        overwrite=True,
        n_jobs=1,
        backend=Nessai(trace=False, num_live_points=64),
    )

    assert solver.backend_name == "nessai"
    assert solver.num_parameters == len(high_res_prior)
    assert solver.observed_counts.ndim == 1
    assert solver.observed_counts.size > 0
    assert solver.model_names == ["cluster"] * 5 + ["nxb"]
    assert solver.parameter_names == [
        "cluster_bapec_2_kT",
        "cluster_bapec_2_Abundanc",
        "cluster_bapec_2_Redshift",
        "cluster_bapec_2_Velocity",
        "cluster_bapec_2_norm",
        "nxb_powerlaw_1_norm",
    ]
