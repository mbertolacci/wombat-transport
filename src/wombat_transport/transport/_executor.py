"""Compiled executor for the block-native TPCORE, VDIFF, convection chain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wombat_transport.fields import TracerField
from wombat_transport.transport.numba_control import configure_numba_threads
from wombat_transport.transport.convection import _operator as convection_operator
from wombat_transport.transport.pbl import _operator as vdiff_operator
from wombat_transport.transport.pbl._plan import VdiffPlan
from wombat_transport.transport.pbl._plan import VdiffPlanWorkspace
from wombat_transport.transport.pbl._plan import make_vdiff_plan_workspace
from wombat_transport.transport.tpcore import _operator as tpcore_operator
from wombat_transport.transport.tpcore import _kernels as tpcore_kernels
from wombat_transport.transport.tpcore._operator import TpcoreWorkspace
from wombat_transport.transport.tpcore._operator import make_tpcore_workspace
from wombat_transport.transport.tpcore._plan import TpcorePlan

if tpcore_kernels._NUMBA_AVAILABLE:
    from numba import njit, prange
else:  # pragma: no cover - exercised in environments without numba.
    njit = None
    prange = range

_advect_one_block_serial = tpcore_operator._advect_one_block_serial
_advect_one_block_spatial = tpcore_operator._advect_one_block_spatial
_apply_vdiff_block_serial = vdiff_operator._apply_vdiff_block_serial
_apply_vdiff_block_spatial = vdiff_operator._apply_vdiff_block_spatial
_convect_block_serial = convection_operator._convect_block_serial
_convect_block_spatial = convection_operator._convect_block_spatial


@dataclass
class TransportWorkspace:
    """Persistent block state plus block-shared and worker-local scratch."""

    tpcore: TpcoreWorkspace
    vdiff_plan: VdiffPlanWorkspace
    workers: int
    qqu: np.ndarray
    qqv: np.ndarray
    x: tuple[np.ndarray, ...]
    y: tuple[np.ndarray, ...]
    y_spatial: tuple[np.ndarray, ...]
    z: tuple[np.ndarray, ...]
    tracer_diffused: np.ndarray
    before_mass: np.ndarray
    after_mass: np.ndarray
    qmx: np.ndarray
    adjust: np.ndarray
    convection_qc: np.ndarray
    convection_diag_empty: np.ndarray
    convection_qb_num: np.ndarray
    convection_delq_work: np.ndarray
    convection_current_work: np.ndarray
    negative_counts: np.ndarray


@dataclass
class TransportExecutor:
    """Persistent state and scratch for one compiled transport strategy."""

    workspace: TransportWorkspace

    @classmethod
    def create(cls, field: TracerField) -> TransportExecutor:
        if field.block_data.shape[0] != 1:
            raise ValueError("transport requires exactly one time slice")
        workers = configure_numba_threads(available=True)
        workspace = make_transport_workspace(field.shape[1:], field.block_width, workers)
        workspace.tpcore.bind_state_storage(field.block_data[0])
        return cls(workspace=workspace)


def make_transport_workspace(
    tracer_shape: tuple[int, int, int, int], lane_width: int, workers: int
) -> TransportWorkspace:
    """Allocate persistent state and scratch for every execution policy."""

    if workers < 1:
        raise ValueError("workers must be positive")
    tpcore = make_tpcore_workspace(tracer_shape, lane_width)
    nlev, nlat, nlon, _ntracer = tpcore.tracer_shape
    nblock = len(tpcore.blocks)
    lane = tpcore.lane_width
    stride = max(lane, convection_operator._CONVECTION_SCRATCH_PAD_TRACERS)
    y_spatial = tpcore_kernels._make_ytp_numba_workspace(workers, nlat, nlon, lane)
    y = (
        *y_spatial[:4],
        *(np.empty((nblock, nlon, lane), dtype=np.float64) for _ in range(4)),
    )
    return TransportWorkspace(
        tpcore=tpcore,
        vdiff_plan=make_vdiff_plan_workspace(nlev, nlat, nlon),
        workers=workers,
        qqu=np.empty((nblock, nlat, nlon, lane), dtype=np.float64),
        qqv=np.empty((nblock, nlat, nlon, lane), dtype=np.float64),
        x=tpcore_kernels._make_xtp_numba_workspace(workers, nlat, nlon, lane),
        y=y,
        y_spatial=y_spatial,
        z=tpcore_kernels._make_fzppm_numba_workspace(workers, nlev, lane),
        tracer_diffused=np.empty((workers, nlon, nlev, lane), dtype=np.float64),
        before_mass=np.empty((workers, nlon, lane), dtype=np.float64),
        after_mass=np.empty((workers, nlon, lane), dtype=np.float64),
        qmx=np.empty((workers, nlon, nlev, lane), dtype=np.float64),
        adjust=np.empty((workers, nlon, lane), dtype=np.bool_),
        convection_qc=np.empty((workers, stride), dtype=np.float64),
        convection_diag_empty=np.empty((0, 0, 0), dtype=np.float64),
        convection_qb_num=np.empty((workers, stride), dtype=np.float64),
        convection_delq_work=np.empty((workers, stride), dtype=np.float64),
        convection_current_work=np.empty((workers, stride), dtype=np.float64),
        negative_counts=np.empty(nblock, dtype=np.int64),
    )


def apply_transport(
    *,
    tpcore_plan: TpcorePlan,
    vdiff_plan: VdiffPlan,
    workspace: TransportWorkspace,
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
    execution: str = "blocks",
) -> int:
    """Apply one prepared transport step using the selected parallel policy."""

    if njit is None or _multi_block_transport_step_parallel is None:
        raise RuntimeError("numba is not available")
    if workspace.workers < 1:
        raise ValueError("transport workspace has no workers")
    if execution not in {"blocks", "serial", "spatial"}:
        raise ValueError("execution must be 'blocks', 'serial', or 'spatial'")

    tpcore_workspace = workspace.tpcore
    nlev, nlat, nlon, ntracer = tpcore_workspace.tracer_shape
    lane = tpcore_workspace.lane_width
    flux = np.zeros((len(tpcore_workspace.blocks), nlat, nlon, lane), dtype=np.float64)
    if surface_flux_kg_m2_s is not None:
        if surface_flux_kg_m2_s.shape != (nlat, nlon, ntracer):
            raise ValueError("surface flux shape does not match tracer state")
        for block in range(len(tpcore_workspace.blocks)):
            start = block * lane
            stop = min(start + lane, ntracer)
            flux[block, :, :, : stop - start] = surface_flux_kg_m2_s[:, :, start:stop]
    has_flux = bool(np.any(flux != 0.0))
    scalar_shape = (nlev, nlat * nlon)
    scalar_inputs = tuple(
        np.ascontiguousarray(value.reshape(scalar_shape))
        for value in (cmfmc, dtrain, delp_hpa, delp_dry, bmass, dqrcu, reevapcn)
    )
    configured_workers = configure_numba_threads(available=True)
    if workspace.workers != configured_workers:
        raise ValueError(
            f"transport workspace has {workspace.workers} workers but "
            f"WOMBAT_NUMBA_THREADS configured {configured_workers}"
        )
    if execution == "spatial":
        return _multi_block_transport_step_spatial(
            tpcore_plan,
            vdiff_plan,
            workspace,
            flux,
            has_flux,
            scalar_inputs,
            reconstruct_conv_precip_flux,
            internal_steps,
            internal_dt_s,
            fill,
        )
    kernel = (
        _multi_block_transport_step_parallel
        if execution == "blocks"
        else _multi_block_transport_step_serial
    )
    kernel(
        tpcore_workspace.state_a,
        tpcore_workspace.state_b,
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
        workspace.qqu,
        workspace.qqv,
        *workspace.x,
        *workspace.y,
        *workspace.z,
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
        workspace.tracer_diffused,
        workspace.before_mass,
        workspace.after_mass,
        workspace.qmx,
        workspace.adjust,
        *scalar_inputs,
        reconstruct_conv_precip_flux,
        internal_steps,
        internal_dt_s,
        workspace.convection_qc,
        workspace.convection_diag_empty,
        workspace.convection_qb_num,
        workspace.convection_delq_work,
        workspace.convection_current_work,
        workspace.negative_counts,
    )
    return int(np.sum(workspace.negative_counts))


def _multi_block_transport_step_spatial(
    tpcore_plan: TpcorePlan,
    vdiff_plan: VdiffPlan,
    workspace: TransportWorkspace,
    surface_flux: np.ndarray,
    has_flux: bool,
    scalar_inputs: tuple[np.ndarray, ...],
    reconstruct_conv_precip_flux: bool,
    internal_steps: int,
    internal_dt_s: float,
    fill: bool,
) -> int:
    """Visit blocks serially while each one-block operator uses spatial threads."""

    negative_count = 0
    for block in range(workspace.tpcore.state_a.shape[0]):
        negative_count += _one_block_transport_step_spatial(
            block,
            tpcore_plan,
            vdiff_plan,
            workspace,
            surface_flux,
            has_flux,
            scalar_inputs,
            reconstruct_conv_precip_flux,
            internal_steps,
            internal_dt_s,
            fill,
        )
    return negative_count


def _one_block_transport_step_spatial(
    block: int,
    tpcore_plan: TpcorePlan,
    vdiff_plan: VdiffPlan,
    workspace: TransportWorkspace,
    surface_flux: np.ndarray,
    has_flux: bool,
    scalar_inputs: tuple[np.ndarray, ...],
    reconstruct_conv_precip_flux: bool,
    internal_steps: int,
    internal_dt_s: float,
    fill: bool,
) -> int:
    """Run one block with spatial parallelism inside each operator."""

    tpcore_workspace = workspace.tpcore
    q = tpcore_workspace.state_a[block]
    dq1 = tpcore_workspace.state_b[block]
    setup = tpcore_plan.setup
    _advect_one_block_spatial(
        q,
        dq1,
        setup.delp1_hpa,
        setup.delp2_hpa,
        setup.pu_hpa,
        setup.xmass_hpa,
        setup.ymass_hpa,
        setup.vertical_mass_flux_hpa,
        setup.cx,
        setup.cy,
        setup.geofac,
        setup.geofac_pc,
        tpcore_plan.ua,
        tpcore_plan.va,
        tpcore_plan.jn,
        tpcore_plan.js,
        tpcore_plan.area_1d_m2,
        bool(fill),
        workspace.qqu[block],
        workspace.qqv[block],
        *workspace.x,
        *workspace.y_spatial,
        *workspace.z,
        True,
    )
    negative_count = _apply_vdiff_block_spatial(
        dq1,
        q,
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
        surface_flux[block],
        has_flux,
        workspace.tracer_diffused,
        workspace.before_mass,
        workspace.after_mass,
        workspace.qmx,
        workspace.adjust,
    )
    nlev, nlat, nlon, lane = q.shape
    _convect_block_spatial(
        q.reshape(nlev, nlat * nlon, lane),
        workspace.convection_diag_empty,
        *scalar_inputs,
        vdiff_plan.area_m2.reshape(nlat * nlon),
        False,
        reconstruct_conv_precip_flux,
        internal_steps,
        internal_dt_s,
        workspace.convection_qc,
        workspace.convection_qb_num,
        workspace.convection_delq_work,
        workspace.convection_current_work,
    )
    return int(negative_count)


if njit is not None:

    @njit(nogil=True, cache=True)
    def _one_block_transport_step_serial(
        q,
        dq1,
        tpcore_plan,
        tpcore_block_work,
        tpcore_worker_work,
        vdiff_plan,
        surface_flux,
        has_flux,
        vdiff_worker_work,
        convection_inputs,
        convection_work,
    ):
        (
            delp1, delp2, pu, xmass, ymass, wz, cx, cy, geofac, geofac_pc,
            ua, va, jn, js, area_1d, fill,
        ) = tpcore_plan
        qqu, qqv, south_flux, north_flux, south_dao2, north_dao2 = tpcore_block_work
        (
            dcx, fx, al_x, ar_x, a6_x, dc_x, qa_x,
            dcy, al_y, ar_y, a6_y,
            dpi_z, dc_z, al_z, ar_z, a6_z, dca_z, prev_flux_z,
        ) = tpcore_worker_work
        _advect_one_block_serial(
            q, dq1, delp1, delp2, pu, xmass, ymass, wz, cx, cy,
            geofac, geofac_pc, ua, va, jn, js, area_1d, fill, qqu, qqv,
            dcx, fx, al_x, ar_x, a6_x, dc_x, qa_x,
            dcy, al_y, ar_y, a6_y, south_flux, north_flux, south_dao2, north_dao2,
            dpi_z, dc_z, al_z, ar_z, a6_z, dca_z, prev_flux_z,
        )
        (
            cch, zeh, termh, dry_mass, area_m2, cgs, kvh, potbar,
            rpdel, rrho, tmp1, dt_s, start_level,
        ) = vdiff_plan
        tracer_diffused, before_mass, after_mass, qmx, adjust = vdiff_worker_work
        negative_count = _apply_vdiff_block_serial(
            dq1, q, cch, zeh, termh, dry_mass, area_m2, cgs, kvh, potbar,
            rpdel, rrho, tmp1, dt_s, start_level, surface_flux, has_flux,
            tracer_diffused, before_mass, after_mass, qmx, adjust,
        )
        (
            diag, cmfmc, dtrain, delp_hpa, delp_dry, bmass, dqrcu, reevapcn, area_m2,
            diagnostics,
            reconstruct_conv_precip_flux, internal_steps, internal_dt_s,
        ) = convection_inputs
        qc, qb_num, delq_work, current_work = convection_work
        nlev, nlat, nlon, lane = q.shape
        _convect_block_serial(
            q.reshape(nlev, nlat * nlon, lane), diag, cmfmc, dtrain, delp_hpa,
            delp_dry, bmass, dqrcu, reevapcn, area_m2, diagnostics,
            reconstruct_conv_precip_flux,
            internal_steps, internal_dt_s, qc, qb_num, delq_work, current_work,
        )
        return negative_count

    def _multi_block_transport_step_impl(
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
        convection_diag_empty,
        convection_qb_num,
        convection_delq_work,
        convection_current_work,
        negative_counts,
    ):
        tpcore_plan = (
            delp1, delp2, pu, xmass, ymass, wz, cx, cy, geofac, geofac_pc,
            ua, va, jn, js, area_1d, fill,
        )
        tpcore_worker_work = (
            dcx, fx, al_x, ar_x, a6_x, dc_x, qa_x,
            dcy, al_y, ar_y, a6_y,
            dpi_z, dc_z, al_z, ar_z, a6_z, dca_z, prev_flux_z,
        )
        vdiff_plan = (
            cch, zeh, termh, dry_mass, area_m2, cgs, kvh, potbar,
            rpdel, rrho, tmp1, dt_s, start_level,
        )
        vdiff_worker_work = (tracer_diffused, before_mass, after_mass, qmx, adjust)
        convection_inputs = (
            convection_diag_empty, cmfmc, dtrain, delp_hpa, delp_dry, bmass,
            dqrcu, reevapcn, area_m2.reshape(area_m2.size), False,
            reconstruct_conv_precip_flux, internal_steps, internal_dt_s,
        )
        convection_work = (
            convection_qc, convection_qb_num, convection_delq_work,
            convection_current_work,
        )
        for block in prange(state_a.shape[0]):
            tpcore_block_work = (
                qqu[block], qqv[block], south_flux_y[block], north_flux_y[block],
                south_dao2_y[block], north_dao2_y[block],
            )
            negative_counts[block] = _one_block_transport_step_serial(
                state_a[block], state_b[block], tpcore_plan, tpcore_block_work,
                tpcore_worker_work, vdiff_plan, surface_flux[block], has_flux,
                vdiff_worker_work, convection_inputs, convection_work,
            )

    # These dispatchers share one Python function, so Numba gives them the same
    # disk-cache key even though only one requests the parallel pipeline. Keep
    # the production parallel specialization cached and compile the diagnostic
    # serial policy in memory to prevent first-compiler-wins cache poisoning.
    _multi_block_transport_step_serial = njit(nogil=True)(
        _multi_block_transport_step_impl
    )
    _multi_block_transport_step_parallel = njit(parallel=True, nogil=True, cache=True)(
        _multi_block_transport_step_impl
    )

else:
    _multi_block_transport_step_serial = None
    _multi_block_transport_step_parallel = None
