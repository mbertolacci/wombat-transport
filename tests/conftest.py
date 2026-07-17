from __future__ import annotations

import importlib.util

import pytest

from wombat_transport.transport import numba_control


@pytest.fixture(autouse=True)
def reset_numba_runtime_state():
    """Keep process-global Numba configuration isolated between tests."""

    numba_control._numba_threads_configured = False
    numba_control._configured_numba_thread_count = None
    numba_control._numba_warning_emitted = False
    yield
    numba_control._numba_threads_configured = False
    numba_control._configured_numba_thread_count = None
    numba_control._numba_warning_emitted = False


@pytest.fixture(params=("pure-python", "numba"))
def transport_numba_mode(request, monkeypatch):
    """Run selected transport tests through pure NumPy and Numba paths."""

    if request.param == "numba":
        if importlib.util.find_spec("numba") is None:
            pytest.skip("numba is not available")
        monkeypatch.setenv("WOMBAT_NUMBA", "1")
    else:
        monkeypatch.setenv("WOMBAT_NUMBA", "0")
    return request.param
