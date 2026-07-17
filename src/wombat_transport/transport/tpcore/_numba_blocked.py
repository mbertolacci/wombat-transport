"""Numba TPCORE executor for persistent blocked tracer storage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wombat_transport.transport.tpcore import _numba as nb
from wombat_transport.transport.tpcore.types import TpcoreSetup

if nb._NUMBA_AVAILABLE:
    from numba import njit
else:  # pragma: no cover - exercised in environments without numba.
    njit = None


@dataclass(frozen=True)
class TpcoreBlockPlan:
    """Tracer-independent values shared by every block in one TPCORE step."""

    setup: TpcoreSetup
    area_1d_m2: np.ndarray
    ua: np.ndarray
    va: np.ndarray
    jn: np.ndarray
    js: np.ndarray


@dataclass
class TpcoreBlockWorkspace:
    """Reusable contiguous storage owned independently by each tracer block."""

    tracer_shape: tuple[int, int, int, int]
    lane_width: int
    state_a: np.ndarray
    state_b: np.ndarray
    blocks: list[nb._TpcoreNumbaWorkspace]


def make_tpcore_block_workspace(
    tracer_shape: tuple[int, int, int, int], lane_width: int
) -> TpcoreBlockWorkspace:
    """Allocate reusable per-block input, output, and scratch arrays."""

    if len(tracer_shape) != 4 or tracer_shape[-1] < 1:
        raise ValueError("tracer_shape must describe at least one canonical tracer")
    if lane_width < 1:
        raise ValueError("lane_width must be positive")
    nlev, nlat, nlon, ntracer = tracer_shape
    nblock = (ntracer + lane_width - 1) // lane_width
    state_a = np.empty((nblock, nlev, nlat, nlon, lane_width), dtype=np.float64)
    state_b = np.empty_like(state_a)
    blocks = [nb._TpcoreNumbaWorkspace(nlev, nlat, nlon, lane_width, 1) for _ in range(nblock)]
    for block_index, block in enumerate(blocks):
        block.q = state_a[block_index]
        block.dq1 = state_b[block_index]
    return TpcoreBlockWorkspace(
        tracer_shape=tracer_shape,
        lane_width=lane_width,
        state_a=state_a,
        state_b=state_b,
        blocks=blocks,
    )


def prepare_tpcore_block_plan(*, setup: TpcoreSetup, area_m2: np.ndarray) -> TpcoreBlockPlan:
    """Prepare the scalar TPCORE state once for all tracer blocks."""

    if not nb._NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available")
    if area_m2.shape != setup.delp1_hpa.shape[1:]:
        raise ValueError("area_m2 shape does not match the TPCORE setup")
    ua = np.empty_like(setup.cx)
    va = np.empty_like(setup.cy)
    jn = np.empty(setup.cx.shape[0], dtype=np.int64)
    js = np.empty(setup.cx.shape[0], dtype=np.int64)
    nb._set_cross_terms_numba_kernel(setup.cx, setup.cy, ua, va)
    nb._set_jn_js_numba_kernel(setup.cx, jn, js)
    return TpcoreBlockPlan(
        setup=setup,
        area_1d_m2=np.ascontiguousarray(area_m2[:, 0], dtype=np.float64),
        ua=ua,
        va=va,
        jn=jn,
        js=js,
    )


def pack_tracer_blocks(tracer_conc: np.ndarray, lane_width: int) -> tuple[np.ndarray, int]:
    """Pack canonical tracers into padded contiguous block-major storage."""

    if tracer_conc.ndim != 4:
        raise ValueError("tracer_conc must have shape (lev, lat, lon, tracer)")
    if lane_width < 1:
        raise ValueError("lane_width must be positive")
    nlev, nlat, nlon, ntracer = tracer_conc.shape
    nblock = (ntracer + lane_width - 1) // lane_width
    blocks = np.zeros((nblock, nlev, nlat, nlon, lane_width), dtype=np.float64)
    for block in range(nblock):
        start = block * lane_width
        stop = min(start + lane_width, ntracer)
        blocks[block, :, :, :, : stop - start] = tracer_conc[:, :, :, start:stop]
    return blocks, ntracer


def unpack_tracer_blocks(blocks: np.ndarray, ntracer: int) -> np.ndarray:
    """Unpack active block lanes into the canonical tracer layout."""

    if blocks.ndim != 5:
        raise ValueError("blocks must have shape (block, lev, lat, lon, lane)")
    lane_width = blocks.shape[-1]
    if ntracer < 1 or ntracer > blocks.shape[0] * lane_width:
        raise ValueError("ntracer is outside the capacity of blocks")
    output = np.empty((*blocks.shape[1:4], ntracer), dtype=np.float64)
    for block in range(blocks.shape[0]):
        start = block * lane_width
        stop = min(start + lane_width, ntracer)
        if start >= stop:
            break
        output[:, :, :, start:stop] = blocks[block, :, :, :, : stop - start]
    return output


def load_tracer_block_workspace(tracer_conc: np.ndarray, workspace: TpcoreBlockWorkspace) -> None:
    """Copy canonical active tracers into reusable padded block inputs."""

    tracer_conc = np.asarray(tracer_conc, dtype=np.float64)
    if tracer_conc.shape != workspace.tracer_shape:
        raise ValueError("block workspace does not match tracer_conc")
    lane_width = workspace.lane_width
    ntracer = tracer_conc.shape[-1]
    for block, block_workspace in enumerate(workspace.blocks):
        start = block * lane_width
        stop = min(start + lane_width, ntracer)
        block_workspace.q.fill(0.0)
        block_workspace.q[:, :, :, : stop - start] = tracer_conc[:, :, :, start:stop]


def unpack_tpcore_block_workspace(workspace: TpcoreBlockWorkspace) -> np.ndarray:
    """Copy active block outputs back to canonical tracer storage."""

    nlev, nlat, nlon, ntracer = workspace.tracer_shape
    lane_width = workspace.lane_width
    output = np.empty((nlev, nlat, nlon, ntracer), dtype=np.float64)
    for block, block_workspace in enumerate(workspace.blocks):
        start = block * lane_width
        stop = min(start + lane_width, ntracer)
        output[:, :, :, start:stop] = block_workspace.dq1[:, :, :, : stop - start]
    return output


if njit is not None:
    # Recompile the existing leaf implementations without their internal
    # parallel regions.  This keeps arithmetic identical while assigning
    # concurrency exclusively to independent tracer blocks.
    _average_const_poles_serial = njit(nogil=True)(nb._average_const_poles_batch_numba_kernel.py_func)
    _init_dq_mass_serial = njit(nogil=True)(nb._init_dq_mass_numba_kernel.py_func)
    _calc_cross_terms_serial = njit(nogil=True)(nb._calc_advec_cross_terms_batch_numba_kernel.py_func)
    _xadv_dao2_apply_serial = njit(nogil=True)(nb._xadv_dao2_apply_batch_numba_kernel.py_func)
    _yadv_dao2_apply_serial = njit(nogil=True)(nb._yadv_dao2_apply_batch_numba_kernel.py_func)
    _xtp_serial = njit(nogil=True)(nb._xtp_batch_numba_kernel.py_func)
    _ytp_serial = njit(nogil=True)(nb._ytp_batch_numba_kernel.py_func)
    _fzppm_serial = njit(nogil=True, fastmath={"contract"})(nb._fzppm_batch_numba_kernel.py_func)
    _qck_needs_fill_serial = njit(nogil=True)(nb._qckxyz_needs_fill_numba_kernel.py_func)
    _qck_serial = njit(nogil=True)(nb._qckxyz_batch_numba_kernel.py_func)
    _finalize_serial = njit(nogil=True)(nb._finalize_tpcore_output_numba_kernel.py_func)
    _advect_one_block_spatial = nb._advect_tracers_fused_numba_kernel

    @njit(nogil=True)
    def _advect_one_block_serial(
        q: np.ndarray,
        dq1: np.ndarray,
        delp1: np.ndarray,
        delp2: np.ndarray,
        pu: np.ndarray,
        xmass: np.ndarray,
        ymass: np.ndarray,
        wz: np.ndarray,
        cx: np.ndarray,
        cy: np.ndarray,
        geofac: np.ndarray,
        geofac_pc: float,
        ua: np.ndarray,
        va: np.ndarray,
        jn: np.ndarray,
        js: np.ndarray,
        area_1d: np.ndarray,
        fill: bool,
        qqu: np.ndarray,
        qqv: np.ndarray,
        dcx: np.ndarray,
        fx: np.ndarray,
        al_x: np.ndarray,
        ar_x: np.ndarray,
        a6_x: np.ndarray,
        dc_x: np.ndarray,
        qa_x: np.ndarray,
        dcy: np.ndarray,
        al_y: np.ndarray,
        ar_y: np.ndarray,
        a6_y: np.ndarray,
        south_flux_y: np.ndarray,
        north_flux_y: np.ndarray,
        south_dao2_y: np.ndarray,
        north_dao2_y: np.ndarray,
        dpi_z: np.ndarray,
        dc_z: np.ndarray,
        al_z: np.ndarray,
        ar_z: np.ndarray,
        a6_z: np.ndarray,
        dca_z: np.ndarray,
        prev_flux_z: np.ndarray,
    ) -> None:
        for level in range(q.shape[0]):
            _average_const_poles_serial(q[level], delp1[level], area_1d)
            _init_dq_mass_serial(q[level], dq1[level], delp1[level])
            _calc_cross_terms_serial(q[level], ua[level], va[level], int(jn[level]), int(js[level]), qqu, qqv)
            _xadv_dao2_apply_serial(q[level], qqv, ua[level], int(jn[level]), int(js[level]))
            _yadv_dao2_apply_serial(q[level], qqu, va[level], south_dao2_y, north_dao2_y)
            _xtp_serial(
                dq1[level], qqv, pu[level], cx[level], xmass[level], int(jn[level]), int(js[level]),
                dcx, fx, al_x, ar_x, a6_x, dc_x, qa_x,
            )
            _ytp_serial(
                dq1[level], qqu, qqv, cy[level], ymass[level], geofac, geofac_pc,
                dcy, al_y, ar_y, a6_y, south_flux_y, north_flux_y,
            )
        _fzppm_serial(delp1, wz, dq1, q, dpi_z, dc_z, al_z, ar_z, a6_z, dca_z, prev_flux_z)
        if fill and _qck_needs_fill_serial(dq1):
            _qck_serial(dq1)
        _finalize_serial(dq1, delp2)

else:
    _advect_one_block_serial = None
    _advect_one_block_spatial = None
