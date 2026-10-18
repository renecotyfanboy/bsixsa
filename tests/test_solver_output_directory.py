from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from bsixsa import Nessai, Solver


@pytest.fixture(autouse=True)
def _any_sampler_is_installed(monkeypatch):
    """`Nessai()` stands for any backend here: these tests also run without the nessai package."""
    monkeypatch.setattr("bsixsa.solver._check_sampler_installed", lambda config: None)


@pytest.fixture
def dummy_prior_definition():
    fake = MagicMock()
    fake.AllData.nSpectra = 1
    fake.AllModels.sources = {1: "mock-model"}
    with patch("bsixsa.solver.xspec", fake):
        yield [("powerlaw", "PhoIndex", object())]


def _patched_build_prior():
    """Stand in for `build_prior` and for the free-parameter names, which would
    otherwise be read from whatever the live XSPEC session holds."""
    return patch.multiple(
        "bsixsa.solver",
        build_prior=MagicMock(return_value=(SimpleNamespace(ndim=1, bounds=[(0.0, 1.0)]), [1], ["mock-model"])),
        iter_free_parameters=MagicMock(side_effect=lambda: iter([SimpleNamespace(name="powerlaw_1_PhoIndex")])),
    )


def test_solver_raises_when_output_dir_not_empty(dummy_prior_definition, tmp_path):
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "placeholder.txt").write_text("content")

    with pytest.raises(FileExistsError, match="not empty"):
        Solver(
            dummy_prior_definition,
            outputfiles_basename=str(output_dir),
            backend=Nessai(),
        )


def test_solver_overwrite_clears_directory(dummy_prior_definition, tmp_path):
    output_dir = tmp_path / "needs_overwrite"
    nested_dir = output_dir / "nested"
    nested_dir.mkdir(parents=True)
    (nested_dir / "placeholder.txt").write_text("content")

    with _patched_build_prior():
        solver = Solver(
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

    with _patched_build_prior():
        solver = Solver(
            dummy_prior_definition,
            outputfiles_basename=str(output_dir),
            overwrite=True,
            backend=backend,
        )

    assert solver.backend_config is backend
    assert solver.backend_name == "nessai"


def test_solver_rejects_non_backend_object(dummy_prior_definition, tmp_path):
    output_dir = tmp_path / "invalid-config"

    with pytest.raises(TypeError, match=r"backend must be a backend configuration such as bsixsa\.Nautilus\(\)"):
        Solver(
            dummy_prior_definition,
            outputfiles_basename=str(output_dir),
            overwrite=True,
            backend=object(),
        )


@pytest.fixture
def user_directories(tmp_path, monkeypatch):
    """A home directory, a working directory two levels below it, and data in each."""
    home = tmp_path / "home"
    work = home / "project" / "fits"
    work.mkdir(parents=True)
    (home / "notes.txt").write_text("keep")
    (work / "source.pha").write_text("keep")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)
    return SimpleNamespace(home=home, work=work)


@pytest.mark.parametrize("target", [".", "./", "..", "../.."])
def test_overwrite_refuses_to_clear_the_working_directory_or_a_parent(dummy_prior_definition, user_directories, target):
    """Regression: overwrite=True with outputfiles_basename='.' deleted the user's data files."""
    with pytest.raises(ValueError, match="Refusing to clear"):
        Solver(dummy_prior_definition, outputfiles_basename=target, overwrite=True, backend=Nessai())

    assert (user_directories.work / "source.pha").exists()
    assert (user_directories.home / "notes.txt").exists()


def test_overwrite_refuses_to_clear_the_home_directory(dummy_prior_definition, user_directories, tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    with pytest.raises(ValueError, match="home directory"):
        Solver(dummy_prior_definition, outputfiles_basename=str(user_directories.home), overwrite=True, backend=Nessai())

    assert (user_directories.home / "notes.txt").exists()


@pytest.mark.parametrize("target", ["../FITS", "../../PROJECT"])
def test_overwrite_refuses_a_case_variant_of_the_working_directory_or_a_parent(
    dummy_prior_definition, user_directories, target
):
    """Regression: on a case-insensitive disk (the macOS default), a spelling of the working
    directory or of one of its parents in another letter case passed the guard."""
    if not os.path.exists(target):
        pytest.skip("the file system holding tmp_path is case-sensitive")

    with pytest.raises(ValueError, match="Refusing to clear"):
        Solver(dummy_prior_definition, outputfiles_basename=target, overwrite=True, backend=Nessai())

    assert (user_directories.work / "source.pha").exists()
    assert (user_directories.home / "notes.txt").exists()


def test_overwrite_refuses_to_clear_a_parent_of_the_home_directory(dummy_prior_definition, tmp_path, monkeypatch):
    """Regression: a parent of $HOME that is not a parent of the working directory (e.g. /Users,
    when running from /tmp) passed the guard."""
    users = tmp_path / "users"
    home = users / "me"
    home.mkdir(parents=True)
    (home / "notes.txt").write_text("keep")
    elsewhere = tmp_path / "scratch"
    elsewhere.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(elsewhere)

    with pytest.raises(ValueError, match="home directory"):
        Solver(dummy_prior_definition, outputfiles_basename=str(users), overwrite=True, backend=Nessai())

    assert (home / "notes.txt").exists()


def test_overwrite_still_clears_a_dedicated_directory_below_the_working_directory(
    dummy_prior_definition, user_directories
):
    output = user_directories.work / "fit-results"
    output.mkdir()
    (output / "old_chain.txt").write_text("old")

    with _patched_build_prior():
        solver = Solver(dummy_prior_definition, outputfiles_basename="fit-results", overwrite=True, backend=Nessai())

    assert list(output.iterdir()) == []
    assert solver.outputfiles_basename == os.path.join("fit-results", "")
    assert (user_directories.work / "source.pha").exists()
