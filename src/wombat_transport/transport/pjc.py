from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wombat_transport.constants import EARTH_RADIUS_M


@dataclass(frozen=True)
class PjcHorizontalGeometry:
    rel_area: np.ndarray
    geofac: np.ndarray
    geofac_pc: float
    cose: np.ndarray
    cosp: np.ndarray

def pjc_mass_flux_hpa(
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
    horizontal_geometry: PjcHorizontalGeometry | None = None,
    dap_hpa: np.ndarray | None = None,
    dbk: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Port GEOS-Chem ``DO_PJC_PFIX`` horizontal mass fluxes to NumPy.

    Inputs and outputs use the Wombat/NetCDF orientation ``(lev, lat, lon)``.
    The returned fluxes match GEOS-Chem's sign convention: positive x flux
    enters a grid box from its western edge, and positive y flux enters from
    its southern edge.
    """

    p1 = np.asarray(p1_hpa, dtype=np.float64).copy()
    p2 = np.asarray(p2_hpa, dtype=np.float64).copy()
    u = np.asarray(u_m_s, dtype=np.float64)
    v = np.asarray(v_m_s, dtype=np.float64)
    area = np.asarray(area_m2, dtype=np.float64)
    hyai = np.asarray(hyai_hpa, dtype=np.float64)
    hybi_arr = np.asarray(hybi, dtype=np.float64)
    lat = np.asarray(lat_deg, dtype=np.float64)

    _assert_pjc_mass_flux_shapes(p1, p2, u, v, area, hyai, hybi_arr, lat)

    geometry = horizontal_geometry if horizontal_geometry is not None else build_pjc_horizontal_geometry(area, lat)
    rel_area = geometry.rel_area
    geofac = geometry.geofac
    geofac_pc = geometry.geofac_pc
    cose = geometry.cose
    cosp = geometry.cosp
    dap = hyai[:-1] - hyai[1:] if dap_hpa is None else np.asarray(dap_hpa, dtype=np.float64)
    dbk_arr = hybi_arr[:-1] - hybi_arr[1:] if dbk is None else np.asarray(dbk, dtype=np.float64)

    dgpress = np.sum((p2 - p1) * rel_area)
    p2 -= dgpress

    _average_pjc_poles(p1, rel_area)
    _average_pjc_poles(p2, rel_area)

    delpm = dap[:, np.newaxis, np.newaxis] + (
        dbk_arr[:, np.newaxis, np.newaxis] * 0.5 * (p1 + p2)[np.newaxis, :, :]
    )
    xmass, ymass = _pjc_raw_mass_flux(delpm, u, v, cose, cosp, dt_s=float(dt_s))
    dpi = _pjc_divergence(xmass, ymass, geofac, geofac_pc)
    dps_ctm = np.sum(dpi, axis=0)
    dps = p2 - p1
    return _pjc_pressure_fixed_fluxes(xmass, ymass, dps, dps_ctm, rel_area, geofac, geofac_pc, dbk_arr)

def _assert_pjc_mass_flux_shapes(
    p1: np.ndarray,
    p2: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    area: np.ndarray,
    hyai: np.ndarray,
    hybi_arr: np.ndarray,
    lat: np.ndarray,
) -> None:
    if p1.ndim != 2:
        raise ValueError(f"p1_hpa must have shape (lat, lon), found {p1.shape}")
    if p2.shape != p1.shape:
        raise ValueError(f"p2_hpa must have shape {p1.shape}, found {p2.shape}")
    expected_3d = (hyai.size - 1, p1.shape[0], p1.shape[1])
    if u.shape != expected_3d or v.shape != expected_3d:
        raise ValueError(f"u_m_s and v_m_s must have shape {expected_3d}, found {u.shape} and {v.shape}")
    if area.shape != p1.shape:
        raise ValueError(f"area_m2 must have shape {p1.shape}, found {area.shape}")
    if hybi_arr.shape != hyai.shape:
        raise ValueError(f"hybi must have shape {hyai.shape}, found {hybi_arr.shape}")
    if lat.shape != (p1.shape[0],):
        raise ValueError(f"lat_deg must have shape ({p1.shape[0]},), found {lat.shape}")
    if p1.shape[0] < 5:
        raise ValueError("PJC polar-cap path requires at least five latitudes")

def build_pjc_horizontal_geometry(area_m2: np.ndarray, lat_deg: np.ndarray) -> PjcHorizontalGeometry:
    nlon = area_m2.shape[1]
    nlat = area_m2.shape[0]
    rel_area = area_m2 / np.sum(area_m2)
    dp = np.pi / float(nlat - 1)
    geofac = dp / (2.0 * rel_area[:, 0] * float(nlon))
    geofac_pc = dp / (2.0 * np.sum(rel_area[:2, 0]) * float(nlon))

    clat = np.deg2rad(lat_deg)
    elat = np.empty(nlat + 1, dtype=np.float64)
    sine = np.empty(nlat + 1, dtype=np.float64)
    cose = np.empty(nlat + 1, dtype=np.float64)
    elat[0] = -0.5 * np.pi
    sine[0] = -1.0
    cose[0] = 0.0
    elat[1:nlat] = 0.5 * (clat[:-1] + clat[1:])
    sine[1:nlat] = np.sin(elat[1:nlat])
    cose[1:nlat] = np.cos(elat[1:nlat])
    elat[nlat] = 0.5 * np.pi
    sine[nlat] = 1.0
    cose[nlat] = 0.0

    dlat = np.empty(nlat, dtype=np.float64)
    dlat[0] = 2.0 * (elat[1] - elat[0])
    dlat[1:-1] = elat[2:-1] - elat[1:-2]
    dlat[-1] = 2.0 * (elat[-1] - elat[-2])
    gw = sine[1:] - sine[:-1]
    cosp = gw / dlat
    return PjcHorizontalGeometry(
        rel_area=rel_area,
        geofac=geofac,
        geofac_pc=float(geofac_pc),
        cose=cose,
        cosp=cosp,
    )


def _pjc_horizontal_geometry(area_m2: np.ndarray, lat_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    geometry = build_pjc_horizontal_geometry(area_m2, lat_deg)
    return geometry.rel_area, geometry.geofac, geometry.geofac_pc, geometry.cose, geometry.cosp

def _average_pjc_poles(pressure_hpa: np.ndarray, rel_area: np.ndarray) -> None:
    south_weight = rel_area[:2, :]
    north_weight = rel_area[-2:, :]
    pressure_hpa[:2, :] = np.sum(pressure_hpa[:2, :] * south_weight) / np.sum(south_weight)
    pressure_hpa[-2:, :] = np.sum(pressure_hpa[-2:, :] * north_weight) / np.sum(north_weight)

def _pjc_raw_mass_flux(
    delpm_hpa: np.ndarray,
    u_m_s: np.ndarray,
    v_m_s: np.ndarray,
    cose: np.ndarray,
    cosp: np.ndarray,
    *,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    nlon = delpm_hpa.shape[2]
    nlat = delpm_hpa.shape[1]
    dlambda = 2.0 * np.pi / float(nlon)
    dphi = np.pi / float(nlat - 1)

    factx = 0.5 * dt_s / (dlambda * EARTH_RADIUS_M * cosp)
    xmass = factx[np.newaxis, :, np.newaxis] * (
        u_m_s * delpm_hpa + np.roll(u_m_s, 1, axis=2) * np.roll(delpm_hpa, 1, axis=2)
    )

    facty = 0.5 * dt_s / (EARTH_RADIUS_M * dphi)
    ymass = np.zeros_like(delpm_hpa)
    ymass[:, 1:, :] = facty * cose[np.newaxis, 1:nlat, np.newaxis] * (
        v_m_s[:, 1:, :] * delpm_hpa[:, 1:, :] + v_m_s[:, :-1, :] * delpm_hpa[:, :-1, :]
    )
    ymass[:, 0, :] = facty * cose[0] * v_m_s[:, 0, :] * delpm_hpa[:, 0, :]
    return xmass, ymass

def _pjc_divergence(xmass_hpa: np.ndarray, ymass_hpa: np.ndarray, geofac: np.ndarray, geofac_pc: float) -> np.ndarray:
    dpi = np.zeros_like(xmass_hpa)
    j1p = 2
    j2p = xmass_hpa.shape[1] - 3

    dpi[:, j1p : j2p + 1, :] = (
        ymass_hpa[:, j1p : j2p + 1, :] - ymass_hpa[:, j1p + 1 : j2p + 2, :]
    ) * geofac[np.newaxis, j1p : j2p + 1, np.newaxis]
    dpi[:, j1p : j2p + 1, :] += xmass_hpa[:, j1p : j2p + 1, :] - np.roll(
        xmass_hpa[:, j1p : j2p + 1, :],
        -1,
        axis=2,
    )

    dpi[:, 0, :] = -np.mean(ymass_hpa[:, j1p, :], axis=1)[:, np.newaxis] * geofac_pc
    dpi[:, -1, :] = np.mean(ymass_hpa[:, j2p + 1, :], axis=1)[:, np.newaxis] * geofac_pc
    dpi[:, 1, :] = dpi[:, 0, :]
    dpi[:, -2, :] = dpi[:, -1, :]
    return dpi

def _pjc_pressure_fixed_fluxes(
    xmass_hpa: np.ndarray,
    ymass_hpa: np.ndarray,
    dps_hpa: np.ndarray,
    dps_ctm_hpa: np.ndarray,
    rel_area: np.ndarray,
    geofac: np.ndarray,
    geofac_pc: float,
    dbk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    nlon = xmass_hpa.shape[2]
    nlat = xmass_hpa.shape[1]
    j1p = 2
    j2p = nlat - 3

    ddps = dps_hpa - dps_ctm_hpa
    dgpress = np.sum(ddps * rel_area)
    mmfd = np.zeros(nlat, dtype=np.float64)
    mmfd[j1p : j2p + 1] = -(np.mean(ddps[j1p : j2p + 1, :], axis=1) - dgpress)
    mmfd[0] = -(ddps[0, 0] - dgpress)
    mmfd[1] = -(ddps[1, 0] - dgpress)
    mmfd[-2] = -(ddps[-2, 0] - dgpress)
    mmfd[-1] = -(ddps[-1, 0] - dgpress)

    mmf = np.zeros(nlat, dtype=np.float64)
    mmf[j1p] = mmfd[0] / geofac_pc
    for j in range(j1p, j2p + 1):
        mmf[j + 1] = mmf[j] + mmfd[j] / geofac[j]

    xcolmass_fix = np.zeros((nlat, nlon), dtype=np.float64)
    for j in range(j1p, j2p + 1):
        fxintegral = np.zeros(nlon + 1, dtype=np.float64)
        for i in range(nlon):
            fxintegral[i + 1] = fxintegral[i] - (ddps[j, i] - dgpress) - mmfd[j]
        fxmean = np.mean(fxintegral[1:])
        xcolmass_fix[j, :] = fxintegral[:-1] - fxmean

    xmass_fixed = xmass_hpa + dbk[:, np.newaxis, np.newaxis] * xcolmass_fix[np.newaxis, :, :]
    ymass_fixed = ymass_hpa.copy()
    ymass_fixed[:, j1p : j2p + 2, :] += dbk[:, np.newaxis, np.newaxis] * mmf[
        np.newaxis,
        j1p : j2p + 2,
        np.newaxis,
    ]
    return xmass_fixed, ymass_fixed
