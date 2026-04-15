from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import xspec

from bsixsa import Nessai, SIXSASolver


@pytest.fixture
def dummy_prior_definition():
    xspec.AllData.nSpectra = 1
    xspec.AllModels.sources = {1: "mock-model"}
    return [("powerlaw", "PhoIndex", object())]


def test_solver_raises_when_output_dir_not_empty(dummy_prior_definition, tmp_path):
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "placeholder.txt").write_text("content")

    with pytest.raises(FileExistsError, match="not empty"):
        SIXSASolver(
            dummy_prior_definition,
            outputfiles_basename=str(output_dir),
            backend=Nessai(),
        )


def test_solver_overwrite_clears_directory(dummy_prior_definition, tmp_path):
    output_dir = tmp_path / "needs_overwrite"
    nested_dir = output_dir / "nested"
    nested_dir.mkdir(parents=True)
    (nested_dir / "placeholder.txt").write_text("content")

    with patch(
        "bsixsa.solver.build_prior",
        return_value=(SimpleNamespace(ndim=1, bounds=[(0.0, 1.0)]), [1], ["mock-model"]),
    ), patch(
        "bsixsa.solver.multiprocessing.Pool",
        return_value=SimpleNamespace(),
    ):
        solver = SIXSASolver(
            dummy_prior_definition,
            outputfiles_basename=str(output_dir),
            overwrite=True,
            backend=Nessai(),
        )

    assert list(output_dir.iterdir()) == []
    assert solver.outputfiles_basename.rstrip(os.sep) == str(output_dir)


def test_solver_initialises_backend_from_backend_object(dummy_prior_definition, tmp_path):
    output_dir = tmp_path / "configured"
    backend = Nessai(num_live_points=64, trace=False)

    with patch(
        "bsixsa.solver.build_prior",
        return_value=(SimpleNamespace(ndim=1, bounds=[(0.0, 1.0)]), [1], ["mock-model"]),
    ), patch(
        "bsixsa.solver.multiprocessing.Pool",
        return_value=SimpleNamespace(),
    ):
        solver = SIXSASolver(
            dummy_prior_definition,
            outputfiles_basename=str(output_dir),
            overwrite=True,
            backend=backend,
        )

    assert solver.backend_config is backend
    assert solver.backend_name == "nessai"


def test_solver_rejects_non_backend_object(dummy_prior_definition, tmp_path):
    output_dir = tmp_path / "invalid-config"

    with pytest.raises(TypeError, match="`config` must define a backend_name"):
        SIXSASolver(
            dummy_prior_definition,
            outputfiles_basename=str(output_dir),
            overwrite=True,
            backend=object(),
        )
