from __future__ import annotations

import importlib.util

import pytest


@pytest.fixture(params=("pure-python", "numba"))
def transport_numba_mode(request, monkeypatch):
    """Run selected transport tests through pure NumPy and Numba paths."""

    monkeypatch.delenv("WOMBAT_TPCORE_NUMBA", raising=False)
    monkeypatch.delenv("WOMBAT_VDIFF_NUMBA", raising=False)
    monkeypatch.delenv("WOMBAT_CONVECTION_NUMBA", raising=False)
    if request.param == "numba":
        if importlib.util.find_spec("numba") is None:
            pytest.skip("numba is not available")
        monkeypatch.setenv("WOMBAT_NUMBA", "1")
    else:
        monkeypatch.setenv("WOMBAT_NUMBA", "0")
    return request.param
