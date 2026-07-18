"""Tracer-independent preparation for the compiled VDIFF operator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wombat_transport.transport.numba_control import configure_numba_threads
from wombat_transport.transport.pbl import _kernels
from wombat_transport.transport.pbl._reference import _max_pbl_levels_from_pressure


@dataclass(frozen=True)
class VdiffPlan:
    """Diffusion coefficients and humidity output shared by tracer blocks."""

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


@dataclass
class VdiffPlanWorkspace:
    """Reusable outputs and dummy inputs for VDIFF preparation."""

    cch: np.ndarray
    zeh: np.ndarray
    termh: np.ndarray
    cgs: np.ndarray
    kvh: np.ndarray
    potbar: np.ndarray
    rpdel: np.ndarray
    rrho: np.ndarray
    tmp1: np.ndarray
    specific_humidity_after: np.ndarray
    dummy_tracer: np.ndarray
    dummy_flux: np.ndarray
    diagnostic_kvm: np.ndarray
    diagnostic_tpert: np.ndarray
    diagnostic_qpert: np.ndarray


def make_vdiff_plan_workspace(
    nlev: int, nlat: int, nlon: int, *, diagnostics: bool = False
) -> VdiffPlanWorkspace:
    """Allocate coefficient storage reused by every transport step."""

    center = np.empty((nlev, nlat, nlon), dtype=np.float64)
    edge = np.empty((nlev + 1, nlat, nlon), dtype=np.float64)
    horizontal = np.empty((nlat, nlon), dtype=np.float64)
    return VdiffPlanWorkspace(
        cch=center,
        zeh=np.empty_like(center),
        termh=np.empty_like(center),
        cgs=edge,
        kvh=np.empty_like(edge),
        potbar=np.empty_like(edge),
        rpdel=np.empty_like(center),
        rrho=horizontal,
        tmp1=np.empty_like(horizontal),
        specific_humidity_after=np.empty_like(center),
        dummy_tracer=np.zeros((nlev, nlat, nlon, 1), dtype=np.float64),
        dummy_flux=np.zeros((nlat, nlon, 1), dtype=np.float64),
        diagnostic_kvm=(
            np.empty((nlev + 1, nlat, nlon), dtype=np.float64)
            if diagnostics
            else np.empty((0,), dtype=np.float64)
        ),
        diagnostic_tpert=(
            np.empty((nlat, nlon), dtype=np.float64)
            if diagnostics
            else np.empty((0,), dtype=np.float64)
        ),
        diagnostic_qpert=(
            np.empty((nlat, nlon), dtype=np.float64)
            if diagnostics
            else np.empty((0,), dtype=np.float64)
        ),
    )


def prepare_vdiff_plan(
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
    workspace: VdiffPlanWorkspace | None = None,
) -> VdiffPlan:
    """Prepare exact zero-surface-flux coefficients for all tracer blocks."""

    if not _kernels._NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available")
    if workers < 1:
        raise ValueError("workers must be positive")
    nlev, nlat, nlon = temperature_top.shape
    if pint_hpa.shape != (nlev + 1, nlat, nlon):
        raise ValueError("pint_hpa shape does not match the VDIFF grid")
    if workspace is None:
        workspace = make_vdiff_plan_workspace(nlev, nlat, nlon)
    if workspace.cch.shape != (nlev, nlat, nlon):
        raise ValueError("VDIFF plan workspace does not match the grid")
    npbl = _max_pbl_levels_from_pressure(np.asarray(pmid_hpa, dtype=np.float64))
    configured_workers = configure_numba_threads(available=True)
    if workers != configured_workers:
        raise ValueError(
            f"VDIFF plan requested {workers} workers but "
            f"WOMBAT_NUMBA_THREADS configured {configured_workers}"
        )
    result = _kernels._prepare_vdiff_plan_numba(
        tracer_top=workspace.dummy_tracer,
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
        surface_flux_kg_m2_s=workspace.dummy_flux,
        ustar_m_s=np.asarray(ustar_m_s, dtype=np.float64),
        area_m2=np.asarray(area_m2, dtype=np.float64),
        dt_s=float(dt_s),
        npbl=int(npbl),
        surface_flux_is_zero=True,
        nthreads=workers,
        reuse_output=True,
        output_buffer=None,
        input_mass_pressure_hpa=None,
        plan_output=(
            workspace.cch,
            workspace.zeh,
            workspace.termh,
            workspace.cgs,
            workspace.kvh,
            workspace.potbar,
            workspace.rpdel,
            workspace.rrho,
            workspace.tmp1,
        ),
        plan_only=True,
        sphu_output_buffer=workspace.specific_humidity_after,
        diagnostic_plan_output=(
            workspace.diagnostic_kvm,
            workspace.diagnostic_tpert,
            workspace.diagnostic_qpert,
        )
        if workspace.diagnostic_kvm.size
        else None,
    )
    return VdiffPlan(
        cch=workspace.cch,
        zeh=workspace.zeh,
        termh=workspace.termh,
        cgs=workspace.cgs,
        kvh=workspace.kvh,
        potbar=workspace.potbar,
        rpdel=workspace.rpdel,
        rrho=workspace.rrho,
        tmp1=workspace.tmp1,
        dry_mass=np.asarray(dry_mass_top, dtype=np.float64),
        area_m2=np.asarray(area_m2, dtype=np.float64),
        dt_s=float(dt_s),
        start_level=max(0, nlev - int(npbl)),
        specific_humidity_after=result.specific_humidity_kg_kg,
    )
