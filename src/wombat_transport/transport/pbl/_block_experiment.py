"""Experimental persistent tracer-block path for VDIFF."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from wombat_transport.transport.pbl import _numba as nb
from wombat_transport.transport.tpcore._block_experiment import TpcoreBlockWorkspace

if nb._NUMBA_AVAILABLE:
    from numba import njit
else:  # pragma: no cover - exercised in environments without numba.
    njit = None


@dataclass(frozen=True)
class VdiffBlockPlan:
    """Tracer-independent diffusion coefficients and humidity result."""

    cch: np.ndarray
    zeh: np.ndarray
    termh: np.ndarray
    dry_mass: np.ndarray
    specific_humidity_after: np.ndarray


@dataclass
class VdiffBlockScratch:
    tracer_diffused: np.ndarray
    before_mass: np.ndarray
    after_mass: np.ndarray


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
    dummy_tracer = np.zeros((nlev, nlat, nlon, 1), dtype=np.float64)
    dummy_flux = np.zeros((nlat, nlon, 1), dtype=np.float64)
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
        plan_output=(cch, zeh, termh),
    )
    return VdiffBlockPlan(
        cch=cch,
        zeh=zeh,
        termh=termh,
        dry_mass=np.asarray(dry_mass_top, dtype=np.float64),
        specific_humidity_after=result.specific_humidity_kg_kg,
    )


def make_vdiff_block_scratch(workspace: TpcoreBlockWorkspace) -> list[VdiffBlockScratch]:
    """Allocate one serial tracer solve scratch set per persistent block."""

    nlev, _nlat, nlon, _ntracer = workspace.tracer_shape
    lane_width = workspace.lane_width
    return [
        VdiffBlockScratch(
            tracer_diffused=np.empty((nlon, nlev, lane_width), dtype=np.float64),
            before_mass=np.empty((nlon, lane_width), dtype=np.float64),
            after_mass=np.empty((nlon, lane_width), dtype=np.float64),
        )
        for _ in workspace.blocks
    ]


def apply_vdiff_zero_flux_to_tpcore_blocks(
    *,
    plan: VdiffBlockPlan,
    workspace: TpcoreBlockWorkspace,
    scratch: list[VdiffBlockScratch],
    workers: int,
) -> int:
    """Diffuse TPCORE block outputs into their reusable alternate buffers."""

    if len(scratch) != len(workspace.blocks):
        raise ValueError("VDIFF scratch does not match the TPCORE block workspace")

    def run_block(block: int) -> int:
        block_workspace = workspace.blocks[block]
        block_scratch = scratch[block]
        return int(
            _apply_vdiff_zero_flux_block(
                block_workspace.dq1,
                block_workspace.q,
                plan.cch,
                plan.zeh,
                plan.termh,
                plan.dry_mass,
                block_scratch.tracer_diffused,
                block_scratch.before_mass,
                block_scratch.after_mass,
            )
        )

    with ThreadPoolExecutor(max_workers=min(workers, len(workspace.blocks))) as executor:
        return sum(executor.map(run_block, range(len(workspace.blocks))))


if njit is not None:
    @njit(nogil=True)
    def _apply_vdiff_zero_flux_block(
        tracer_in: np.ndarray,
        tracer_out: np.ndarray,
        cch: np.ndarray,
        zeh: np.ndarray,
        termh: np.ndarray,
        dry_mass: np.ndarray,
        tracer_diffused: np.ndarray,
        before_mass: np.ndarray,
        after_mass: np.ndarray,
    ) -> int:
        nlev, nlat, nlon, nlane = tracer_in.shape
        negative_count = 0
        for lat in range(nlat):
            for lon in range(nlon):
                for lane in range(nlane):
                    value = tracer_in[0, lat, lon, lane]
                    before_mass[lon, lane] = value * dry_mass[0, lat, lon]
                    tracer_diffused[lon, 0, lane] = value * termh[0, lat, lon]
            for lev in range(1, nlev - 1):
                for lon in range(nlon):
                    cch_value = cch[lev, lat, lon]
                    termh_value = termh[lev, lat, lon]
                    mass = dry_mass[lev, lat, lon]
                    for lane in range(nlane):
                        value = tracer_in[lev, lat, lon, lane]
                        before_mass[lon, lane] += value * mass
                        tracer_diffused[lon, lev, lane] = (
                            value + cch_value * tracer_diffused[lon, lev - 1, lane]
                        ) * termh_value
            for lon in range(nlon):
                tmp1d = 1.0 / (1.0 + cch[nlev - 1, lat, lon] * (1.0 - zeh[nlev - 2, lat, lon]))
                mass = dry_mass[nlev - 1, lat, lon]
                for lane in range(nlane):
                    value = tracer_in[nlev - 1, lat, lon, lane]
                    before_mass[lon, lane] += value * mass
                    tracer_diffused[lon, nlev - 1, lane] = (
                        value
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
                    if abs(before_mass[lon, lane]) > 0.0 and abs(after_mass[lon, lane]) > 0.0:
                        ratio = before_mass[lon, lane] / after_mass[lon, lane]
                    before_mass[lon, lane] = ratio
                for lev in range(nlev):
                    for lane in range(nlane):
                        tracer_out[lev, lat, lon, lane] = tracer_diffused[lon, lev, lane] * before_mass[lon, lane]
        return negative_count

else:
    _apply_vdiff_zero_flux_block = None
