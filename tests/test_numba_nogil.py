from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from wombat_transport.transport.convection import _native as convection_native
from wombat_transport.transport.convection import _numba as convection_numba
from wombat_transport.transport.pbl import _numba as pbl_numba
from wombat_transport.transport.tpcore import _numba as tpcore_numba


NUMBA_MODULES = (tpcore_numba, pbl_numba, convection_numba)


@pytest.mark.skipif(not tpcore_numba._NUMBA_AVAILABLE, reason="numba is unavailable")
def test_transport_numba_kernels_release_gil():
    dispatchers = [
        value
        for module in NUMBA_MODULES
        for value in vars(module).values()
        if getattr(value, "targetoptions", None) is not None
    ]

    assert dispatchers
    assert all(dispatcher.targetoptions.get("nogil") is True for dispatcher in dispatchers)


@pytest.mark.parametrize(
    ("getter", "args"),
    [
        (tpcore_numba._get_tpcore_numba_workspace, (2, 3, 4, 1, 1)),
        (pbl_numba._get_vdiff_fullgrid_workspace, (1, 2, 3, 4, 1)),
        (convection_numba._get_convection_kernel_workspace, (1, 1)),
        (convection_native._get_convection_light_workspace, (2, 3, 4, 1)),
    ],
)
def test_reusable_transport_workspaces_are_thread_local(getter, args):
    main_workspace = getter(*args)

    with ThreadPoolExecutor(max_workers=1) as executor:
        worker_workspace, reused_worker_workspace = executor.submit(
            lambda: (getter(*args), getter(*args))
        ).result()

    assert worker_workspace is reused_worker_workspace
    assert worker_workspace is not main_workspace
