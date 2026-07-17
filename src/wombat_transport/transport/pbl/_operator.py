"""Runtime VDIFF operator, workspaces, and plan application."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import wraps

import numpy as np

from wombat_transport.transport.numba_control import synchronized_transport_numba
from wombat_transport.transport.pbl import _kernels as nb
from wombat_transport.transport.pbl import _reference
from wombat_transport.transport.pbl._plan import VdiffPlan
from wombat_transport.transport.pbl._plan import VdiffPlanWorkspace
from wombat_transport.transport.pbl._plan import make_vdiff_plan_workspace
from wombat_transport.transport.pbl._plan import prepare_vdiff_plan

if nb._NUMBA_AVAILABLE:
    from numba import get_thread_id, njit, prange
else:  # pragma: no cover - exercised in environments without numba.
    get_thread_id = None
    njit = None
    prange = range

_G0_M_PER_S2 = nb.G0_M_PER_S2
_RD_J_PER_KG_K = nb.RD_J_PER_KG_K
_numba_vdiff_mode = nb._numba_vdiff_mode
_numba_vdiff_enabled = nb._numba_vdiff_enabled
_numba_vdiff_thread_count = nb._numba_vdiff_thread_count

for _name in dir(_reference):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_reference, _name))


if njit is not None:
    def _apply_vdiff_block_impl(
        tracer_in: np.ndarray,
        tracer_out: np.ndarray,
        cch: np.ndarray,
        zeh: np.ndarray,
        termh: np.ndarray,
        dry_mass: np.ndarray,
        area_m2: np.ndarray,
        cgs: np.ndarray,
        kvh: np.ndarray,
        potbar: np.ndarray,
        rpdel: np.ndarray,
        rrho: np.ndarray,
        tmp1: np.ndarray,
        dt_s: float,
        start_level: int,
        surface_flux: np.ndarray,
        has_flux: bool,
        tracer_diffused_workspace: np.ndarray,
        before_mass_workspace: np.ndarray,
        after_mass_workspace: np.ndarray,
        qmx_workspace: np.ndarray,
        adjust_workspace: np.ndarray,
    ) -> int:
        nlev, nlat, nlon, nlane = tracer_in.shape
        negative_count = 0
        ztodtgor = dt_s * _G0_M_PER_S2 / _RD_J_PER_KG_K
        for lat in prange(nlat):
            thread = get_thread_id()
            tracer_diffused = tracer_diffused_workspace[thread]
            before_mass = before_mass_workspace[thread]
            after_mass = after_mass_workspace[thread]
            qmx = qmx_workspace[thread]
            adjust = adjust_workspace[thread]
            if has_flux:
                for lon in range(nlon):
                    for lane in range(nlane):
                        adjust[lon, lane] = False
                    for lev in range(nlev):
                        for lane in range(nlane):
                            qmx[lon, lev, lane] = tracer_in[lev, lat, lon, lane]
                for lev in range(start_level, nlev):
                    for lon in range(nlon):
                        scale = ztodtgor * rpdel[lev, lat, lon]
                        term_next = potbar[lev + 1, lat, lon] * kvh[lev + 1, lat, lon]
                        term_now = potbar[lev, lat, lon] * kvh[lev, lat, lon]
                        for lane in range(nlane):
                            flux_rrho = surface_flux[lat, lon, lane] * rrho[lat, lon]
                            cgq_next = flux_rrho * cgs[lev + 1, lat, lon]
                            cgq_now = flux_rrho * cgs[lev, lat, lon]
                            value = tracer_in[lev, lat, lon, lane] + scale * (
                                term_next * cgq_next - term_now * cgq_now
                            )
                            qmx[lon, lev, lane] = value
                            if value < 0.0:
                                adjust[lon, lane] = True
                for lon in range(nlon):
                    for lane in range(nlane):
                        if adjust[lon, lane]:
                            for lev in range(start_level, nlev):
                                qmx[lon, lev, lane] = tracer_in[lev, lat, lon, lane]
            for lon in range(nlon):
                for lane in range(nlane):
                    value = tracer_in[0, lat, lon, lane]
                    before_mass[lon, lane] = value * dry_mass[0, lat, lon]
                    source = qmx[lon, 0, lane] if has_flux else value
                    tracer_diffused[lon, 0, lane] = source * termh[0, lat, lon]
            for lev in range(1, nlev - 1):
                for lon in range(nlon):
                    cch_value = cch[lev, lat, lon]
                    termh_value = termh[lev, lat, lon]
                    mass = dry_mass[lev, lat, lon]
                    for lane in range(nlane):
                        value = tracer_in[lev, lat, lon, lane]
                        before_mass[lon, lane] += value * mass
                        source = qmx[lon, lev, lane] if has_flux else value
                        tracer_diffused[lon, lev, lane] = (
                            source + cch_value * tracer_diffused[lon, lev - 1, lane]
                        ) * termh_value
            for lon in range(nlon):
                tmp1d = 1.0 / (1.0 + cch[nlev - 1, lat, lon] * (1.0 - zeh[nlev - 2, lat, lon]))
                mass = dry_mass[nlev - 1, lat, lon]
                for lane in range(nlane):
                    value = tracer_in[nlev - 1, lat, lon, lane]
                    before_mass[lon, lane] += value * mass
                    source = qmx[lon, nlev - 1, lane] if has_flux else value
                    tracer_diffused[lon, nlev - 1, lane] = (
                        source
                        + (surface_flux[lat, lon, lane] * tmp1[lat, lon] if has_flux else 0.0)
                        + cch[nlev - 1, lat, lon] * tracer_diffused[lon, nlev - 2, lane]
                    ) * tmp1d
            for lev in range(nlev - 2, -1, -1):
                for lon in range(nlon):
                    zeh_value = zeh[lev, lat, lon]
                    for lane in range(nlane):
                        tracer_diffused[lon, lev, lane] += zeh_value * tracer_diffused[lon, lev + 1, lane]

            for lon in range(nlon):
                for lane in range(nlane):
                    after_mass[lon, lane] = 0.0
                for lev in range(nlev):
                    mass = dry_mass[lev, lat, lon]
                    for lane in range(nlane):
                        value = tracer_diffused[lon, lev, lane]
                        if value < 0.0:
                            negative_count += 1
                            value = 0.0
                            tracer_diffused[lon, lev, lane] = 0.0
                        after_mass[lon, lane] += value * mass
                for lane in range(nlane):
                    ratio = 1.0
                    if has_flux:
                        before_mass[lon, lane] += surface_flux[lat, lon, lane] * area_m2[lat, lon] * dt_s
                    if abs(before_mass[lon, lane]) > 0.0 and abs(after_mass[lon, lane]) > 0.0:
                        ratio = before_mass[lon, lane] / after_mass[lon, lane]
                    before_mass[lon, lane] = ratio
                for lev in range(nlev):
                    for lane in range(nlane):
                        tracer_out[lev, lat, lon, lane] = tracer_diffused[lon, lev, lane] * before_mass[lon, lane]
        return negative_count

    _apply_vdiff_block_serial = njit(nogil=True)(_apply_vdiff_block_impl)
    _apply_vdiff_block_spatial = njit(parallel=True, nogil=True, cache=True)(
        _apply_vdiff_block_impl
    )

else:
    _apply_vdiff_block_serial = None
    _apply_vdiff_block_spatial = None


@dataclass
class _VdiffOneBlockWorkspace:
    shape: tuple[int, int, int, int, int]
    plan: VdiffPlanWorkspace
    tracer_out: np.ndarray
    tracer_diffused: np.ndarray
    before_mass: np.ndarray
    after_mass: np.ndarray
    qmx: np.ndarray
    adjust: np.ndarray
    input_converted: np.ndarray


_VDIFF_ONE_BLOCK_WORKSPACES = threading.local()


def _get_vdiff_one_block_workspace(
    workers: int, nlev: int, nlat: int, nlon: int, ntracer: int
) -> _VdiffOneBlockWorkspace:
    shape = (workers, nlev, nlat, nlon, ntracer)
    existing = getattr(_VDIFF_ONE_BLOCK_WORKSPACES, "workspace", None)
    if existing is not None and existing.shape == shape:
        return existing
    workspace = _VdiffOneBlockWorkspace(
        shape=shape,
        plan=make_vdiff_plan_workspace(nlev, nlat, nlon, diagnostics=True),
        tracer_out=np.empty((nlev, nlat, nlon, ntracer), dtype=np.float64),
        tracer_diffused=np.empty((workers, nlon, nlev, ntracer), dtype=np.float64),
        before_mass=np.empty((workers, nlon, ntracer), dtype=np.float64),
        after_mass=np.empty((workers, nlon, ntracer), dtype=np.float64),
        qmx=np.empty((workers, nlon, nlev, ntracer), dtype=np.float64),
        adjust=np.empty((workers, nlon, ntracer), dtype=np.bool_),
        input_converted=np.empty((nlev, nlat, nlon, ntracer), dtype=np.float64),
    )
    _VDIFF_ONE_BLOCK_WORKSPACES.workspace = workspace
    return workspace


@synchronized_transport_numba
def run_vdiff_one_block_compiled(
    *,
    tracer_top: np.ndarray,
    u_top: np.ndarray,
    v_top: np.ndarray,
    temperature_top: np.ndarray,
    sphu_top: np.ndarray,
    pmid_hpa: np.ndarray,
    pint_hpa: np.ndarray,
    virtual_temperature_top: np.ndarray,
    bxheight_top: np.ndarray,
    dry_mass_top: np.ndarray,
    pblh_m: np.ndarray,
    hflux_w_m2: np.ndarray,
    water_flux_kg_m2_s: np.ndarray,
    surface_flux_kg_m2_s: np.ndarray,
    ustar_m_s: np.ndarray,
    area_m2: np.ndarray,
    dt_s: float,
    workers: int,
    diagnostics: bool,
    reuse_output: bool,
    output_buffer: np.ndarray | None,
    input_mass_pressure_hpa: np.ndarray | None,
):
    """Run the shared prepare/apply VDIFF path for one canonical tracer block."""

    if _apply_vdiff_block_spatial is None:
        raise RuntimeError("numba is not available")
    nlev, nlat, nlon, ntracer = tracer_top.shape
    workspace = _get_vdiff_one_block_workspace(workers, nlev, nlat, nlon, ntracer)
    tracer_input = tracer_top
    if input_mass_pressure_hpa is not None:
        nb._finalize_deferred_tpcore_poles_numba_kernel(tracer_top, input_mass_pressure_hpa)
        np.copyto(workspace.input_converted, tracer_top)
        for lat in range(nlat):
            if lat == 1 or lat == nlat - 2:
                continue
            workspace.input_converted[:, lat, :, :] *= (
                1.0 / input_mass_pressure_hpa[:, lat, :]
            )[..., np.newaxis]
            converted = workspace.input_converted[:, lat, :, :]
            converted[converted < 0.0] = 1.0e-26
        tracer_input = workspace.input_converted
    if output_buffer is not None:
        tracer_out = np.asarray(output_buffer)
        if tracer_out.shape != tracer_top.shape or tracer_out.dtype != np.float64:
            raise ValueError("output_buffer must match tracer_conc shape and float64 dtype")
        if not tracer_out.flags.c_contiguous or not tracer_out.flags.writeable:
            raise ValueError("output_buffer must be writable and C-contiguous")
        if np.shares_memory(tracer_out, tracer_top):
            raise ValueError("output_buffer must not overlap tracer_conc")
    elif reuse_output and not np.shares_memory(tracer_top, workspace.tracer_out):
        tracer_out = workspace.tracer_out
    else:
        tracer_out = np.empty_like(tracer_top)

    initial_mass = (
        nb._tracer_working_mass_numba_kernel(tracer_input, dry_mass_top)
        if diagnostics
        else np.empty((0,), dtype=np.float64)
    )
    plan = prepare_vdiff_plan(
        u_top=u_top,
        v_top=v_top,
        temperature_top=temperature_top,
        sphu_top=sphu_top,
        pmid_hpa=pmid_hpa,
        pint_hpa=pint_hpa,
        virtual_temperature_top=virtual_temperature_top,
        bxheight_top=bxheight_top,
        dry_mass_top=dry_mass_top,
        pblh_m=pblh_m,
        hflux_w_m2=hflux_w_m2,
        water_flux_kg_m2_s=water_flux_kg_m2_s,
        ustar_m_s=ustar_m_s,
        area_m2=area_m2,
        dt_s=dt_s,
        workers=workers,
        workspace=workspace.plan,
    )
    negative_count = _apply_vdiff_block_spatial(
        tracer_input,
        tracer_out,
        plan.cch,
        plan.zeh,
        plan.termh,
        plan.dry_mass,
        plan.area_m2,
        plan.cgs,
        plan.kvh,
        plan.potbar,
        plan.rpdel,
        plan.rrho,
        plan.tmp1,
        plan.dt_s,
        plan.start_level,
        surface_flux_kg_m2_s,
        bool(np.any(surface_flux_kg_m2_s != 0.0)),
        workspace.tracer_diffused,
        workspace.before_mass,
        workspace.after_mass,
        workspace.qmx,
        workspace.adjust,
    )
    final_mass = (
        nb._tracer_working_mass_numba_kernel(tracer_out, dry_mass_top)
        if diagnostics
        else np.empty((0,), dtype=np.float64)
    )
    empty = np.empty((0,), dtype=np.float64)
    return nb.VdiffDrResult(
        tracer_conc=tracer_out,
        specific_humidity_kg_kg=(
            plan.specific_humidity_after
            if reuse_output and not diagnostics
            else plan.specific_humidity_after.copy()
        ),
        kvh_m2_s=plan.kvh if diagnostics else empty,
        kvm_m2_s=workspace.plan.diagnostic_kvm if diagnostics else empty,
        pbl_top_m=np.asarray(pblh_m).copy(),
        tpert_k=workspace.plan.diagnostic_tpert if diagnostics else empty,
        qpert_kg_kg=workspace.plan.diagnostic_qpert if diagnostics else empty,
        negative_count_before_clip=int(negative_count),
        negative_count_after_clip=0,
        initial_tracer_mass=initial_mass,
        final_tracer_mass=final_mass,
    )


@wraps(_reference.run_vdiffdr_one_step)
def run_vdiffdr_one_step(*args, **kwargs):
    """Run VDIFF through the reference or compiled operator implementation."""

    if nb._numba_vdiff_enabled():
        kwargs["_compiled_impl"] = run_vdiff_one_block_compiled
        kwargs["_compiled_workers"] = nb._numba_vdiff_thread_count()
        kwargs["_mass_impl"] = nb._tracer_working_mass_numba
    return _reference.run_vdiffdr_one_step(*args, **kwargs)
