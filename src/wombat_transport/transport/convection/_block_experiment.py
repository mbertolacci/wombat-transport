"""Experimental persistent tracer-block executor for convection."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from wombat_transport.transport.convection import _numba as nb
from wombat_transport.transport.tpcore._block_experiment import TpcoreBlockWorkspace

if nb._NUMBA_AVAILABLE:
    from numba import njit
else:  # pragma: no cover
    njit = None


def apply_convection_to_vdiff_blocks(
    *,
    workspace: TpcoreBlockWorkspace,
    cmfmc: np.ndarray,
    dtrain: np.ndarray,
    delp_hpa: np.ndarray,
    delp_dry: np.ndarray,
    bmass: np.ndarray,
    dqrcu: np.ndarray,
    reevapcn: np.ndarray,
    reconstruct_conv_precip_flux: bool,
    internal_steps: int,
    internal_dt_s: float,
    workers: int,
) -> None:
    """Apply convection in place to the VDIFF result held in each block's ``q``."""

    nlev, nlat, nlon, _ntracer = workspace.tracer_shape
    ncol = nlat * nlon
    scalar_shape = (nlev, ncol)
    scalar_inputs = [np.ascontiguousarray(value.reshape(scalar_shape)) for value in (
        cmfmc, dtrain, delp_hpa, delp_dry, bmass, dqrcu, reevapcn
    )]
    stride = max(workspace.lane_width, nb._CONVECTION_SCRATCH_PAD_TRACERS)
    scratch = [
        nb._ConvectionKernelWorkspace(
            shape=(1, stride),
            qc=np.empty((1, stride), dtype=np.float64),
            qb_num=np.empty((1, stride), dtype=np.float64),
            delq_work=np.empty((1, stride), dtype=np.float64),
            current_work=np.empty((1, stride), dtype=np.float64),
        )
        for _ in workspace.blocks
    ]

    def run_block(index: int) -> None:
        block = workspace.blocks[index]
        local = scratch[index]
        _convect_block_serial(
            block.q.reshape(nlev, ncol, workspace.lane_width),
            *scalar_inputs,
            reconstruct_conv_precip_flux,
            internal_steps,
            internal_dt_s,
            local.qc,
            local.qb_num,
            local.delq_work,
            local.current_work,
        )

    with ThreadPoolExecutor(max_workers=min(workers, len(workspace.blocks))) as executor:
        tuple(executor.map(run_block, range(len(workspace.blocks))))


if njit is not None:
    _convect_block_serial = njit(nogil=True, fastmath={"contract"})(
        nb._convect_fullgrid_top_numba_kernel.py_func
    )
else:
    _convect_block_serial = None
