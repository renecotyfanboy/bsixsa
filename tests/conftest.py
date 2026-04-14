import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest import fixture

try:
    import xspec
except ModuleNotFoundError:  # pragma: no cover - depends on local HEASoft install
    xspec = MagicMock()
    sys.modules["xspec"] = xspec
    XSPEC_AVAILABLE = False
else:
    XSPEC_AVAILABLE = True


@fixture
def xspec_config():
    if not XSPEC_AVAILABLE:
        pytest.skip("xspec is not available in this environment")
    xspec.AllData.clear()
    xspec.AllModels.clear()
    xspec.Xset.chatter = 0
    xspec.Xset.logChatter = 0
    xspec.Fit.statMethod = "cstat"
    xspec.Fit.bayes = "on"
    xspec.Fit.query = "no"
    xspec.Fit.criticalDelta = 1e-3
    xspec.Fit.nIterations = 50
    xspec.Fit.bayes = "cons"


@fixture
def data(xspec_config, request, monkeypatch):
    # Set paths and move to data directory
    file_pha = "tests/mock_data/spectrum_opt.pha"
    dir_path = os.path.dirname(file_pha)
    monkeypatch.chdir(os.path.join(os.path.dirname(request.fspath.dirname), dir_path))

    # Load XSPEC obs and set energy band
    xspec_observation = xspec.Spectrum("spectrum_opt.pha")
    xspec_observation.background = None
    low_energy, high_energy = 0.3, 12.0
    xspec_observation.ignore(f"0.0-{low_energy:.1f} {high_energy:.1f}-**")

    # Load XSPEC model
    xspec.Xset.restore("model.xcm")
    xspec_model = xspec.AllModels(1)

    return xspec_model, xspec_observation


@fixture
def mock_data(data):
    xspec_model, xspec_observation = data
    return np.random.poisson(12.7, size=(100, len(xspec_observation.values)))
