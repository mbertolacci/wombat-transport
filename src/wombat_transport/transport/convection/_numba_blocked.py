"""Numba convection kernel for persistent blocked tracer storage."""

from __future__ import annotations

from wombat_transport.transport.convection import _numba as nb

if nb._NUMBA_AVAILABLE:
    from numba import njit
else:  # pragma: no cover
    njit = None


if njit is not None:
    _convect_block_serial = njit(nogil=True, fastmath={"contract"})(
        nb._convect_fullgrid_top_numba_kernel.py_func
    )
else:
    _convect_block_serial = None
