from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

TESTS_DIR = Path(__file__).resolve().parent
LOW_RES_DATA_DIR = TESTS_DIR / "low_res_data"
HIGH_RES_DATA_DIR = TESTS_DIR / "high_res_data"

try:
    import xspec
except ModuleNotFoundError:  # pragma: no cover - depends on local HEASoft install
    xspec = MagicMock()
    sys.modules["xspec"] = xspec
    XSPEC_AVAILABLE = False
else:
    XSPEC_AVAILABLE = True


@pytest.fixture(scope="session")
def low_res_data_dir() -> Path:
    return LOW_RES_DATA_DIR


@pytest.fixture(scope="session")
def high_res_data_dir() -> Path:
    return HIGH_RES_DATA_DIR


@pytest.fixture(scope="session")
def low_res_prior():
    from bsixsa.priors import loguniform, uniform

    return [
        ("TBabs", "nH", uniform(0.0, 0.5)),
        ("bbodyrad", "kT", uniform(0.0, 5.0)),
        ("bbodyrad", "norm", loguniform(1e-1, 1e1)),
        ("powerlaw", "PhoIndex", uniform(1.0, 4.0)),
        ("powerlaw", "norm", loguniform(1e-5, 1e-2)),
    ]


@pytest.fixture(scope="session")
def high_res_prior():
    from bsixsa.priors import loguniform, uniform

    return [
        ("cluster", "bapec", "kT", uniform(0.0, 7.0)),
        ("cluster", "bapec", "Abundanc", uniform(0.0, 4.0)),
        ("cluster", "bapec", "Redshift", uniform(0.05, 0.3)),
        ("cluster", "bapec", "Velocity", uniform(0.5, 1000.0)),
        ("cluster", "bapec", "norm", loguniform(1e-5, 1e-2)),
        ("nxb", "powerlaw", "norm", loguniform(1e-4, 1e0)),
    ]


@pytest.fixture
def xspec_session():
    if not XSPEC_AVAILABLE:
        pytest.skip("xspec is not available in this environment")

    xspec.AllData.clear()
    xspec.AllModels.clear()
    xspec.Xset.chatter = 0
    xspec.Xset.logChatter = 0
    xspec.Fit.statMethod = "cstat"
    xspec.Fit.bayes = "off"
    xspec.Xset.xsect = "vern"
    xspec.Xset.abund = "wilm"

    yield xspec

    xspec.AllData.clear()
    xspec.AllModels.clear()


@pytest.fixture
def low_res_xspec_setup(xspec_session, low_res_data_dir: Path, monkeypatch):
    monkeypatch.chdir(low_res_data_dir)

    observation = xspec_session.Spectrum(
        "PN_spectrum_grp20_opt.pha",
        respFile="PN.rmf",
        arfFile="PN.arf",
    )
    xspec_session.AllData.ignore("bad")
    observation.ignore("0.0-0.3 10.0-**")
    xspec_session.Model("tbabs*(bbodyrad + powerlaw)")

    return {
        "observation": observation,
        "data_dir": low_res_data_dir,
    }


@pytest.fixture
def high_res_xspec_setup(xspec_session, high_res_data_dir: Path, monkeypatch):
    monkeypatch.chdir(high_res_data_dir)

    observation = xspec_session.Spectrum("spec_170.pha")
    xspec_session.AllData.ignore("bad")
    observation.ignore("0.0-1.5 8.0-**")

    xspec_session.Model("tbabs*bapec", sourceNum=1, modName="cluster")
    observation.multiresponse[1] = observation.multiresponse[0].rmf
    xspec_session.Model(
        "pow",
        sourceNum=2,
        modName="nxb",
        setPars={1: 0, 2: 1e-5},
    )
    xspec_session.AllModels(1, modName="cluster").TBabs.nH = "0.01, -1, 0, 0, 1, 1"
    xspec_session.AllModels(1, modName="nxb").powerlaw.PhoIndex = "0., -1, 0, 0, 1, 1"

    return {
        "observation": observation,
        "data_dir": high_res_data_dir,
    }
