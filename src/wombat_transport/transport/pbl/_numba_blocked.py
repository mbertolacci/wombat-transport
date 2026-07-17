"""Numba VDIFF path for persistent blocked tracer storage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wombat_transport.transport.pbl import _numba as nb

if nb._NUMBA_AVAILABLE:
    from numba import get_thread_id, njit, prange, set_num_threads
else:  # pragma: no cover - exercised in environments without numba.
    get_thread_id = None
    njit = None
    prange = range
    set_num_threads = None

_G0_M_PER_S2 = nb.G0_M_PER_S2
_RD_J_PER_KG_K = nb.RD_J_PER_KG_K


@dataclass(frozen=True)
class VdiffBlockPlan:
    """Tracer-independent diffusion coefficients and humidity result."""

    cch: np.ndarray
    zeh: np.ndarray
    termh: np.ndarray
    cgs: np.ndarray
    kvh: np.ndarray
    potbar: np.ndarray
    rpdel: np.ndarray
    rrho: np.ndarray
    tmp1: np.ndarray
    dry_mass: np.ndarray
    area_m2: np.ndarray
    dt_s: float
    start_level: int
    specific_humidity_after: np.ndarray


def prepare_vdiff_zero_flux_block_plan(
    *,
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
    ustar_m_s: np.ndarray,
    area_m2: np.ndarray,
    dt_s: float,
    workers: int,
) -> VdiffBlockPlan:
    """Prepare exact zero-surface-flux coefficients once for all tracer blocks."""

    if not nb._NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available")
    if workers < 1:
        raise ValueError("workers must be positive")
    nlev, nlat, nlon = temperature_top.shape
    if pint_hpa.shape != (nlev + 1, nlat, nlon):
        raise ValueError("pint_hpa shape does not match the VDIFF grid")
    npbl = nb._max_pbl_levels_from_pressure(np.asarray(pmid_hpa, dtype=np.float64))
    cch = np.empty((nlev, nlat, nlon), dtype=np.float64)
    zeh = np.empty_like(cch)
    termh = np.empty_like(cch)
    cgs = np.empty((nlev + 1, nlat, nlon), dtype=np.float64)
    kvh = np.empty_like(cgs)
    potbar = np.empty_like(cgs)
    rpdel = np.empty_like(cch)
    rrho = np.empty((nlat, nlon), dtype=np.float64)
    tmp1 = np.empty_like(rrho)
    dummy_tracer = np.zeros((nlev, nlat, nlon, 1), dtype=np.float64)
    dummy_flux = np.zeros((nlat, nlon, 1), dtype=np.float64)
    set_num_threads(workers)
    result = nb._run_vdiffdr_one_step_fullgrid_numba(
        tracer_top=dummy_tracer,
        u_top=np.asarray(u_top, dtype=np.float64),
        v_top=np.asarray(v_top, dtype=np.float64),
        temperature_top=np.asarray(temperature_top, dtype=np.float64),
        sphu_top=np.asarray(sphu_top, dtype=np.float64),
        pmid_hpa=np.asarray(pmid_hpa, dtype=np.float64),
        pint_hpa=np.asarray(pint_hpa, dtype=np.float64),
        virtual_temperature_top=np.asarray(virtual_temperature_top, dtype=np.float64),
        bxheight_top=np.asarray(bxheight_top, dtype=np.float64),
        dry_mass_top=np.asarray(dry_mass_top, dtype=np.float64),
        pblh_m=np.asarray(pblh_m, dtype=np.float64),
        hflux_w_m2=np.asarray(hflux_w_m2, dtype=np.float64),
        water_flux_kg_m2_s=np.asarray(water_flux_kg_m2_s, dtype=np.float64),
        surface_flux_kg_m2_s=dummy_flux,
        ustar_m_s=np.asarray(ustar_m_s, dtype=np.float64),
        area_m2=np.asarray(area_m2, dtype=np.float64),
        dt_s=float(dt_s),
        npbl=int(npbl),
        surface_flux_is_zero=True,
        nthreads=workers,
        reuse_output=False,
        output_buffer=None,
        input_mass_pressure_hpa=None,
        plan_output=(cch, zeh, termh, cgs, kvh, potbar, rpdel, rrho, tmp1),
    )
    return VdiffBlockPlan(
        cch=cch,
        zeh=zeh,
        termh=termh,
        cgs=cgs,
        kvh=kvh,
        potbar=potbar,
        rpdel=rpdel,
        rrho=rrho,
        tmp1=tmp1,
        dry_mass=np.asarray(dry_mass_top, dtype=np.float64),
        area_m2=np.asarray(area_m2, dtype=np.float64),
        dt_s=float(dt_s),
        start_level=max(0, nlev - int(npbl)),
        specific_humidity_after=result.specific_humidity_kg_kg,
    )


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
    _apply_vdiff_block_spatial = njit(parallel=True, nogil=True)(_apply_vdiff_block_impl)

else:
    _apply_vdiff_block_serial = None
    _apply_vdiff_block_spatial = None
