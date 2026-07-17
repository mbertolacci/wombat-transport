"""Numba convection kernel for persistent block-native tracer storage."""

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
    _convect_block_spatial = nb._convect_fullgrid_top_numba_kernel
else:
    _convect_block_serial = None
    _convect_block_spatial = None
