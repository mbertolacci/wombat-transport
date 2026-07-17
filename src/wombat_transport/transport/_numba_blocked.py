"""Top-level Numba executor for persistent blocked tracer storage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wombat_transport.transport.convection import _numba_blocked as convection_block
from wombat_transport.transport.pbl import _numba_blocked as vdiff_block
from wombat_transport.transport.pbl._numba_blocked import VdiffBlockPlan
from wombat_transport.transport.tpcore import _numba_blocked as tpcore_block
from wombat_transport.transport.tpcore import _numba as tpcore_nb
from wombat_transport.transport.tpcore._numba_blocked import TpcoreBlockPlan
from wombat_transport.transport.tpcore._numba_blocked import TpcoreBlockWorkspace

if tpcore_nb._NUMBA_AVAILABLE:
    from numba import get_thread_id, njit, prange, set_num_threads
else:  # pragma: no cover - exercised in environments without numba.
    get_thread_id = None
    njit = None
    prange = range
    set_num_threads = None

_advect_one_block_serial = tpcore_block._advect_one_block_serial
_apply_vdiff_block = vdiff_block._apply_vdiff_zero_flux_block
_convect_block_serial = convection_block._convect_block_serial


@dataclass
class NumbaBlockPipelineScratch:
    """Scratch shared by outer Numba workers rather than persistent blocks."""

    workers: int
    qqu: np.ndarray
    qqv: np.ndarray
    x: tuple[np.ndarray, ...]
    y: tuple[np.ndarray, ...]
    z: tuple[np.ndarray, ...]
    tracer_diffused: np.ndarray
    before_mass: np.ndarray
    after_mass: np.ndarray
    qmx: np.ndarray
    adjust: np.ndarray
    convection_qc: np.ndarray
    convection_qb_num: np.ndarray
    convection_delq_work: np.ndarray
    convection_current_work: np.ndarray
    negative_counts: np.ndarray


def make_numba_block_pipeline_scratch(
    workspace: TpcoreBlockWorkspace, workers: int
) -> NumbaBlockPipelineScratch:
    """Allocate scratch for a single outer ``prange(block)`` region."""

    if workers < 1:
        raise ValueError("workers must be positive")
    nlev, nlat, nlon, _ntracer = workspace.tracer_shape
    nblock = len(workspace.blocks)
    lane = workspace.lane_width
    stride = max(lane, convection_block.nb._CONVECTION_SCRATCH_PAD_TRACERS)
    y = tpcore_nb._make_ytp_numba_workspace(workers, nlat, nlon, lane)
    y = (
        *y[:4],
        *(np.empty((nblock, nlon, lane), dtype=np.float64) for _ in range(4)),
    )
    return NumbaBlockPipelineScratch(
        workers=workers,
        qqu=np.empty((nblock, nlat, nlon, lane), dtype=np.float64),
        qqv=np.empty((nblock, nlat, nlon, lane), dtype=np.float64),
        x=tpcore_nb._make_xtp_numba_workspace(workers, nlat, nlon, lane),
        y=y,
        z=tpcore_nb._make_fzppm_numba_workspace(workers, nlev, lane),
        tracer_diffused=np.empty((workers, nlon, nlev, lane), dtype=np.float64),
        before_mass=np.empty((workers, nlon, lane), dtype=np.float64),
        after_mass=np.empty((workers, nlon, lane), dtype=np.float64),
        qmx=np.empty((workers, nlon, nlev, lane), dtype=np.float64),
        adjust=np.empty((workers, nlon, lane), dtype=np.bool_),
        convection_qc=np.empty((workers, stride), dtype=np.float64),
        convection_qb_num=np.empty((workers, stride), dtype=np.float64),
        convection_delq_work=np.empty((workers, stride), dtype=np.float64),
        convection_current_work=np.empty((workers, stride), dtype=np.float64),
        negative_counts=np.empty(nblock, dtype=np.int64),
    )


def apply_numba_block_pipeline(
    *,
    tpcore_plan: TpcoreBlockPlan,
    vdiff_plan: VdiffBlockPlan,
    workspace: TpcoreBlockWorkspace,
    scratch: NumbaBlockPipelineScratch,
    surface_flux_kg_m2_s: np.ndarray | None,
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
    fill: bool = True,
) -> int:
    """Run each block through TPCORE, VDIFF, and convection in one region."""

    if njit is None or _apply_numba_block_pipeline_kernel is None:
        raise RuntimeError("numba is not available")
    if scratch.workers < 1:
        raise ValueError("pipeline scratch has no workers")

    nlev, nlat, nlon, ntracer = workspace.tracer_shape
    lane = workspace.lane_width
    flux = np.zeros((len(workspace.blocks), nlat, nlon, lane), dtype=np.float64)
    if surface_flux_kg_m2_s is not None:
        if surface_flux_kg_m2_s.shape != (nlat, nlon, ntracer):
            raise ValueError("surface flux shape does not match tracer state")
        for block in range(len(workspace.blocks)):
            start = block * lane
            stop = min(start + lane, ntracer)
            flux[block, :, :, : stop - start] = surface_flux_kg_m2_s[:, :, start:stop]
    has_flux = bool(np.any(flux != 0.0))
    scalar_shape = (nlev, nlat * nlon)
    scalar_inputs = tuple(
        np.ascontiguousarray(value.reshape(scalar_shape))
        for value in (cmfmc, dtrain, delp_hpa, delp_dry, bmass, dqrcu, reevapcn)
    )

    set_num_threads(scratch.workers)
    _apply_numba_block_pipeline_kernel(
        workspace.state_a,
        workspace.state_b,
        tpcore_plan.setup.delp1_hpa,
        tpcore_plan.setup.delp2_hpa,
        tpcore_plan.setup.pu_hpa,
        tpcore_plan.setup.xmass_hpa,
        tpcore_plan.setup.ymass_hpa,
        tpcore_plan.setup.vertical_mass_flux_hpa,
        tpcore_plan.setup.cx,
        tpcore_plan.setup.cy,
        tpcore_plan.setup.geofac,
        tpcore_plan.setup.geofac_pc,
        tpcore_plan.ua,
        tpcore_plan.va,
        tpcore_plan.jn,
        tpcore_plan.js,
        tpcore_plan.area_1d_m2,
        bool(fill),
        scratch.qqu,
        scratch.qqv,
        *scratch.x,
        *scratch.y,
        *scratch.z,
        vdiff_plan.cch,
        vdiff_plan.zeh,
        vdiff_plan.termh,
        vdiff_plan.dry_mass,
        vdiff_plan.area_m2,
        vdiff_plan.cgs,
        vdiff_plan.kvh,
        vdiff_plan.potbar,
        vdiff_plan.rpdel,
        vdiff_plan.rrho,
        vdiff_plan.tmp1,
        vdiff_plan.dt_s,
        vdiff_plan.start_level,
        flux,
        has_flux,
        scratch.tracer_diffused,
        scratch.before_mass,
        scratch.after_mass,
        scratch.qmx,
        scratch.adjust,
        *scalar_inputs,
        reconstruct_conv_precip_flux,
        internal_steps,
        internal_dt_s,
        scratch.convection_qc,
        scratch.convection_qb_num,
        scratch.convection_delq_work,
        scratch.convection_current_work,
        scratch.negative_counts,
    )
    return int(np.sum(scratch.negative_counts))


if njit is not None:

    @njit(parallel=True, nogil=True)
    def _apply_numba_block_pipeline_kernel(
        state_a,
        state_b,
        delp1,
        delp2,
        pu,
        xmass,
        ymass,
        wz,
        cx,
        cy,
        geofac,
        geofac_pc,
        ua,
        va,
        jn,
        js,
        area_1d,
        fill,
        qqu,
        qqv,
        dcx,
        fx,
        al_x,
        ar_x,
        a6_x,
        dc_x,
        qa_x,
        dcy,
        al_y,
        ar_y,
        a6_y,
        south_flux_y,
        north_flux_y,
        south_dao2_y,
        north_dao2_y,
        dpi_z,
        dc_z,
        al_z,
        ar_z,
        a6_z,
        dca_z,
        prev_flux_z,
        cch,
        zeh,
        termh,
        dry_mass,
        area_m2,
        cgs,
        kvh,
        potbar,
        rpdel,
        rrho,
        tmp1,
        dt_s,
        start_level,
        surface_flux,
        has_flux,
        tracer_diffused,
        before_mass,
        after_mass,
        qmx,
        adjust,
        cmfmc,
        dtrain,
        delp_hpa,
        delp_dry,
        bmass,
        dqrcu,
        reevapcn,
        reconstruct_conv_precip_flux,
        internal_steps,
        internal_dt_s,
        convection_qc,
        convection_qb_num,
        convection_delq_work,
        convection_current_work,
        negative_counts,
    ):
        nlev = state_a.shape[1]
        ncol = state_a.shape[2] * state_a.shape[3]
        lane = state_a.shape[4]
        for block in prange(state_a.shape[0]):
            thread = get_thread_id()
            _advect_one_block_serial(
                state_a[block], state_b[block], delp1, delp2, pu, xmass, ymass, wz,
                cx, cy, geofac, geofac_pc, ua, va, jn, js, area_1d, fill,
                qqu[block], qqv[block], dcx, fx, al_x, ar_x, a6_x, dc_x, qa_x,
                dcy, al_y, ar_y, a6_y, south_flux_y[block], north_flux_y[block],
                south_dao2_y[block], north_dao2_y[block], dpi_z, dc_z, al_z, ar_z, a6_z,
                dca_z, prev_flux_z,
            )
            negative_counts[block] = _apply_vdiff_block(
                state_b[block], state_a[block], cch, zeh, termh, dry_mass, area_m2,
                cgs, kvh, potbar, rpdel, rrho, tmp1, dt_s, start_level,
                surface_flux[block], has_flux, tracer_diffused[thread],
                before_mass[thread], after_mass[thread], qmx[thread], adjust[thread],
            )
            _convect_block_serial(
                state_a[block].reshape(nlev, ncol, lane), cmfmc, dtrain, delp_hpa,
                delp_dry, bmass, dqrcu, reevapcn, reconstruct_conv_precip_flux,
                internal_steps, internal_dt_s, convection_qc, convection_qb_num,
                convection_delq_work, convection_current_work,
            )

else:
    _apply_numba_block_pipeline_kernel = None
