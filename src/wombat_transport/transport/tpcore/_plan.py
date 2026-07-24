"""Tracer-independent preparation for the compiled TPCORE operator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wombat_transport.transport.pjc import _assert_pjc_mass_flux_shapes
from wombat_transport.transport.tpcore import _kernels
from wombat_transport.transport.tpcore import _preparation
from wombat_transport.transport.tpcore._reference import (
    _validate_tpcore_grid_identity,
    build_tpcore_static_terms,
)
from wombat_transport.transport.tpcore.types import TpcoreSetup
from wombat_transport.transport.tpcore.types import TpcoreStaticTerms


@dataclass(frozen=True)
class TpcorePlan:
    """TPCORE setup and derived values shared by every tracer block."""

    setup: TpcoreSetup
    normalized_vertical_courant: np.ndarray
    area_1d_m2: np.ndarray
    ua: np.ndarray
    va: np.ndarray
    jn: np.ndarray
    js: np.ndarray


@dataclass
class TpcorePlanWorkspace:
    """Reusable outputs and scratch for compiled TPCORE preparation."""

    delp1_hpa: np.ndarray
    delpm_hpa: np.ndarray
    delp2_hpa: np.ndarray
    pu_hpa: np.ndarray
    surface_pressure_hpa: np.ndarray
    xmass_hpa: np.ndarray
    ymass_hpa: np.ndarray
    vertical_mass_flux_hpa: np.ndarray
    normalized_vertical_courant: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    ua: np.ndarray
    va: np.ndarray
    jn: np.ndarray
    js: np.ndarray
    area_1d_m2: np.ndarray
    p1_hpa: np.ndarray
    p2_hpa: np.ndarray
    work_3d: np.ndarray
    work_2d: np.ndarray
    xfix: np.ndarray
    mmfd: np.ndarray
    mmf: np.ndarray
    fxintegral: np.ndarray


def make_tpcore_plan_workspace(
    nlev: int, nlat: int, nlon: int
) -> TpcorePlanWorkspace:
    """Allocate one plan and its preparation scratch for repeated steps."""

    center = np.empty((nlev, nlat, nlon), dtype=np.float64)
    horizontal = np.empty((nlat, nlon), dtype=np.float64)
    return TpcorePlanWorkspace(
        delp1_hpa=center,
        delpm_hpa=np.empty_like(center),
        delp2_hpa=np.empty_like(center),
        pu_hpa=np.empty_like(center),
        surface_pressure_hpa=horizontal,
        xmass_hpa=np.empty_like(center),
        ymass_hpa=np.empty_like(center),
        vertical_mass_flux_hpa=np.empty_like(center),
        normalized_vertical_courant=np.empty_like(center),
        cx=np.empty_like(center),
        cy=np.empty_like(center),
        ua=np.empty_like(center),
        va=np.empty_like(center),
        jn=np.empty(nlev, dtype=np.int64),
        js=np.empty(nlev, dtype=np.int64),
        area_1d_m2=np.empty(nlat, dtype=np.float64),
        p1_hpa=np.empty_like(horizontal),
        p2_hpa=np.empty_like(horizontal),
        work_3d=np.empty_like(center),
        work_2d=np.empty_like(horizontal),
        xfix=np.empty_like(horizontal),
        mmfd=np.empty(nlat, dtype=np.float64),
        mmf=np.empty(nlat, dtype=np.float64),
        fxintegral=np.empty(nlon + 1, dtype=np.float64),
    )


def prepare_tpcore_plan(*, setup: TpcoreSetup, area_m2: np.ndarray) -> TpcorePlan:
    """Prepare tracer-independent TPCORE values for one transport step."""

    if not _kernels._NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available")
    if area_m2.shape != setup.delp1_hpa.shape[1:]:
        raise ValueError("area_m2 shape does not match the TPCORE setup")
    ua = np.empty_like(setup.cx)
    va = np.empty_like(setup.cy)
    jn = np.empty(setup.cx.shape[0], dtype=np.int64)
    js = np.empty(setup.cx.shape[0], dtype=np.int64)
    _kernels._set_cross_terms_numba_kernel(setup.cx, setup.cy, ua, va)
    _kernels._set_jn_js_numba_kernel(setup.cx, jn, js)
    return TpcorePlan(
        setup=setup,
        normalized_vertical_courant=_kernels._normalized_vertical_courant_numba(
            setup.delp1_hpa, setup.vertical_mass_flux_hpa
        ),
        area_1d_m2=np.ascontiguousarray(area_m2[:, 0], dtype=np.float64),
        ua=ua,
        va=va,
        jn=jn,
        js=js,
    )


def prepare_tpcore_met_plan(
    *,
    p1_hpa: np.ndarray,
    p2_hpa: np.ndarray,
    u_m_s: np.ndarray,
    v_m_s: np.ndarray,
    area_m2: np.ndarray,
    hyai_hpa: np.ndarray,
    hybi: np.ndarray,
    lat_deg: np.ndarray,
    dt_s: float,
    static_terms: TpcoreStaticTerms | None = None,
    workspace: TpcorePlanWorkspace | None = None,
) -> TpcorePlan:
    """Prepare a plan whose arrays remain valid until the workspace is reused."""

    if not _kernels._NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available")
    p1 = np.asarray(p1_hpa, dtype=np.float64)
    p2 = np.asarray(p2_hpa, dtype=np.float64)
    u = np.asarray(u_m_s, dtype=np.float64)
    v = np.asarray(v_m_s, dtype=np.float64)
    area = np.asarray(area_m2, dtype=np.float64)
    hyai = np.asarray(hyai_hpa, dtype=np.float64)
    hybi_arr = np.asarray(hybi, dtype=np.float64)
    lat = np.asarray(lat_deg, dtype=np.float64)
    _assert_pjc_mass_flux_shapes(p1, p2, u, v, area, hyai, hybi_arr, lat)

    if static_terms is None:
        static = build_tpcore_static_terms(
            area_m2=area,
            hyai_hpa=hyai,
            hybi=hybi_arr,
            lat_deg=lat,
        )
    else:
        static = static_terms
        _validate_tpcore_grid_identity(
            static.grid_identity, area, hyai, hybi_arr, lat
        )

    nlev, nlat, nlon = u.shape
    if workspace is None:
        workspace = make_tpcore_plan_workspace(nlev, nlat, nlon)
    _validate_plan_workspace(workspace, nlev, nlat, nlon)
    workspace.area_1d_m2[:] = area[:, 0]
    geometry = static.pjc_geometry
    _preparation.prepare_tpcore_arrays(
        p1,
        p2,
        u,
        v,
        float(dt_s),
        geometry.rel_area,
        geometry.geofac,
        geometry.geofac_pc,
        geometry.cose,
        geometry.cosp,
        float(static.ak_top_hpa[0]),
        static.dap_geos_hpa,
        static.dbk_geos,
        static.dap_top_hpa,
        static.dbk_top,
        (
            workspace.delp1_hpa,
            workspace.delpm_hpa,
            workspace.delp2_hpa,
            workspace.pu_hpa,
            workspace.surface_pressure_hpa,
            workspace.xmass_hpa,
            workspace.ymass_hpa,
            workspace.vertical_mass_flux_hpa,
            workspace.normalized_vertical_courant,
            workspace.cx,
            workspace.cy,
            workspace.ua,
            workspace.va,
            workspace.jn,
            workspace.js,
        ),
        (
            workspace.p1_hpa,
            workspace.p2_hpa,
            workspace.work_3d,
            workspace.work_2d,
            workspace.xfix,
            workspace.mmfd,
            workspace.mmf,
            workspace.fxintegral,
        ),
    )
    setup = TpcoreSetup(
        xmass_hpa=workspace.xmass_hpa,
        ymass_hpa=workspace.ymass_hpa,
        surface_pressure_hpa=workspace.surface_pressure_hpa,
        delp1_hpa=workspace.delp1_hpa,
        delpm_hpa=workspace.delpm_hpa,
        delp2_hpa=workspace.delp2_hpa,
        pu_hpa=workspace.pu_hpa,
        vertical_mass_flux_hpa=workspace.vertical_mass_flux_hpa,
        cx=workspace.cx,
        cy=workspace.cy,
        geofac=geometry.geofac,
        geofac_pc=geometry.geofac_pc,
    )
    return TpcorePlan(
        setup=setup,
        normalized_vertical_courant=workspace.normalized_vertical_courant,
        area_1d_m2=workspace.area_1d_m2,
        ua=workspace.ua,
        va=workspace.va,
        jn=workspace.jn,
        js=workspace.js,
    )


def _validate_plan_workspace(
    workspace: TpcorePlanWorkspace, nlev: int, nlat: int, nlon: int
) -> None:
    center_shape = (nlev, nlat, nlon)
    center_arrays = (
        workspace.delp1_hpa,
        workspace.delpm_hpa,
        workspace.delp2_hpa,
        workspace.pu_hpa,
        workspace.xmass_hpa,
        workspace.ymass_hpa,
        workspace.vertical_mass_flux_hpa,
        workspace.normalized_vertical_courant,
        workspace.cx,
        workspace.cy,
        workspace.ua,
        workspace.va,
        workspace.work_3d,
    )
    if any(array.shape != center_shape for array in center_arrays):
        raise ValueError("TPCORE plan workspace does not match the grid")
    horizontal_shape = (nlat, nlon)
    horizontal_arrays = (
        workspace.surface_pressure_hpa,
        workspace.p1_hpa,
        workspace.p2_hpa,
        workspace.work_2d,
        workspace.xfix,
    )
    if any(array.shape != horizontal_shape for array in horizontal_arrays):
        raise ValueError("TPCORE plan workspace does not match the grid")
    if (
        workspace.jn.shape != (nlev,)
        or workspace.js.shape != (nlev,)
        or workspace.area_1d_m2.shape != (nlat,)
        or workspace.mmfd.shape != (nlat,)
        or workspace.mmf.shape != (nlat,)
        or workspace.fxintegral.shape != (nlon + 1,)
    ):
        raise ValueError("TPCORE plan workspace does not match the grid")
