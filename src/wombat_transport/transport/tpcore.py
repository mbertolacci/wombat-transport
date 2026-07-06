"""GEOS-Chem-oriented NumPy TPCORE pieces.

The first tracked oracle fixture is intentionally compact and low-Courant:
``tests/fixtures/tpcore_snapshot_v1`` has max ``|cx|`` around 0.0023 and max
``|cy|`` around 0.0008. Matching that fixture is useful one-step coverage for
the ordinary low-Courant branches, but it does not exercise TPCORE's
large-Courant polar/semi-Lagrangian branches. Full-grid validation must keep
those branch limits visible instead of treating this fixture as comprehensive
TPCORE parity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wombat_transport.transport.pjc import _pjc_horizontal_geometry, pjc_mass_flux_hpa


@dataclass(frozen=True)
class TpcoreState:
    """One-step TPCORE state arrays in NetCDF order.

    Arrays exposed by this dataclass use the project orientation
    ``(lev, lat, lon)`` for 3-D fields and ``(tracer, lev, lat, lon)`` for
    tracers. GEOS-Chem TPCORE internally runs in top-to-bottom vertical order;
    this module reverses only inside the implementation.
    """

    tracer_conc_after: np.ndarray
    xmass_hpa: np.ndarray
    ymass_hpa: np.ndarray
    surface_pressure_hpa: np.ndarray
    delp1_hpa: np.ndarray
    delpm_hpa: np.ndarray
    delp2_hpa: np.ndarray
    vertical_mass_flux_hpa: np.ndarray


@dataclass(frozen=True)
class TpcoreSetup:
    xmass_hpa: np.ndarray
    ymass_hpa: np.ndarray
    surface_pressure_hpa: np.ndarray
    delp1_hpa: np.ndarray
    delpm_hpa: np.ndarray
    delp2_hpa: np.ndarray
    vertical_mass_flux_hpa: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    geofac: np.ndarray
    geofac_pc: float


def run_tpcore_one_step(
    *,
    tracer_conc: np.ndarray,
    p1_hpa: np.ndarray,
    p2_hpa: np.ndarray,
    u_m_s: np.ndarray,
    v_m_s: np.ndarray,
    area_m2: np.ndarray,
    hyai_hpa: np.ndarray,
    hybi: np.ndarray,
    lat_deg: np.ndarray,
    dt_s: float,
    fill: bool = True,
) -> TpcoreState:
    """Run the first NumPy TPCORE one-step path.

    This path currently ports GEOS-Chem pressure and mass-flux bookkeeping
    exactly for the compact oracle fixture and then applies the no-op tracer
    placeholder. The placeholder is deliberately isolated here so direct ports
    of ``Xtp``, ``Ytp``, ``Fzppm``, and ``Qckxyz`` can replace it without
    changing the harness-facing API.
    """

    setup = setup_tpcore_terms(
        p1_hpa=p1_hpa,
        p2_hpa=p2_hpa,
        u_m_s=u_m_s,
        v_m_s=v_m_s,
        area_m2=area_m2,
        hyai_hpa=hyai_hpa,
        hybi=hybi,
        lat_deg=lat_deg,
        dt_s=dt_s,
    )
    tracer = np.asarray(tracer_conc, dtype=np.float64).copy()
    if fill:
        tracer[tracer < 0.0] = 1.0e-26
    return TpcoreState(
        tracer_conc_after=tracer,
        xmass_hpa=setup.xmass_hpa,
        ymass_hpa=setup.ymass_hpa,
        surface_pressure_hpa=setup.surface_pressure_hpa,
        delp1_hpa=setup.delp1_hpa,
        delpm_hpa=setup.delpm_hpa,
        delp2_hpa=setup.delp2_hpa,
        vertical_mass_flux_hpa=setup.vertical_mass_flux_hpa,
    )


def setup_tpcore_terms(
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
) -> TpcoreSetup:
    """Port the non-tracer setup performed before TPCORE tracer advection."""

    p1 = np.asarray(p1_hpa, dtype=np.float64).copy()
    p2 = np.asarray(p2_hpa, dtype=np.float64).copy()
    area = np.asarray(area_m2, dtype=np.float64)
    hyai = np.asarray(hyai_hpa, dtype=np.float64)
    hybi_arr = np.asarray(hybi, dtype=np.float64)
    lat = np.asarray(lat_deg, dtype=np.float64)

    xmass, ymass = pjc_mass_flux_hpa(
        p1_hpa=p1,
        p2_hpa=p2,
        u_m_s=np.asarray(u_m_s, dtype=np.float64),
        v_m_s=np.asarray(v_m_s, dtype=np.float64),
        area_m2=area,
        hyai_hpa=hyai,
        hybi=hybi_arr,
        lat_deg=lat,
        dt_s=dt_s,
    )

    rel_area, geofac, geofac_pc, cose, _cosp = _pjc_horizontal_geometry(area, lat)
    _average_poles_in_place(p1, rel_area)
    _average_poles_in_place(p2, rel_area)

    ak = hyai[::-1]
    bk = hybi_arr[::-1]
    dap = ak[1:] - ak[:-1]
    dbk = bk[1:] - bk[:-1]

    x_tpcore = xmass[::-1]
    y_tpcore = ymass[::-1]
    delp1_t, delpm_t, pu_t = _set_press_terms(dap, dbk, p1, p2)
    cx_t, cy_t = _calc_courant(cose, delpm_t, pu_t, x_tpcore, y_tpcore)
    dpi_t = _calc_divergence(x_tpcore, y_tpcore, geofac, geofac_pc)
    dps_ctm = np.sum(dpi_t, axis=0)
    wz_t = _calc_vertical_mass_flux(dbk, dps_ctm, dpi_t)
    delp2_t = dap[:, np.newaxis, np.newaxis] + dbk[:, np.newaxis, np.newaxis] * (
        p1[np.newaxis, :, :] + dps_ctm[np.newaxis, :, :]
    )
    ps = ak[0] + np.sum(delp2_t, axis=0)

    return TpcoreSetup(
        xmass_hpa=xmass,
        ymass_hpa=ymass,
        surface_pressure_hpa=ps,
        delp1_hpa=delp1_t[::-1],
        delpm_hpa=delpm_t[::-1],
        delp2_hpa=delp2_t[::-1],
        vertical_mass_flux_hpa=wz_t[::-1],
        cx=cx_t[::-1],
        cy=cy_t[::-1],
        geofac=geofac,
        geofac_pc=geofac_pc,
    )


def _average_poles_in_place(pressure_hpa: np.ndarray, rel_area: np.ndarray) -> None:
    south_weight = rel_area[:2, :]
    north_weight = rel_area[-2:, :]
    pressure_hpa[:2, :] = np.sum(pressure_hpa[:2, :] * south_weight) / np.sum(south_weight)
    pressure_hpa[-2:, :] = np.sum(pressure_hpa[-2:, :] * north_weight) / np.sum(north_weight)


def _set_press_terms(
    dap: np.ndarray,
    dbk: np.ndarray,
    pres1: np.ndarray,
    pres2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delp1 = dap[:, np.newaxis, np.newaxis] + dbk[:, np.newaxis, np.newaxis] * pres1[np.newaxis, :, :]
    delpm = dap[:, np.newaxis, np.newaxis] + dbk[:, np.newaxis, np.newaxis] * 0.5 * (
        pres1[np.newaxis, :, :] + pres2[np.newaxis, :, :]
    )
    pu = np.zeros_like(delpm)
    j1p, j2p = _polar_cap_bounds(pres1.shape[0])
    pu[:, j1p : j2p + 1, 0] = 0.5 * (delpm[:, j1p : j2p + 1, 0] + delpm[:, j1p : j2p + 1, -1])
    pu[:, j1p : j2p + 1, 1:] = 0.5 * (
        delpm[:, j1p : j2p + 1, 1:] + delpm[:, j1p : j2p + 1, :-1]
    )
    return delp1, delpm, pu


def _calc_courant(
    cose: np.ndarray,
    delpm: np.ndarray,
    pu: np.ndarray,
    xmass: np.ndarray,
    ymass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cx = np.zeros_like(xmass)
    cy = np.zeros_like(ymass)
    j1p, j2p = _polar_cap_bounds(xmass.shape[1])
    cx[:, j1p : j2p + 1, :] = xmass[:, j1p : j2p + 1, :] / pu[:, j1p : j2p + 1, :]
    cy[:, j1p : j2p + 1, :] = ymass[:, j1p : j2p + 1, :] / (
        0.5
        * cose[np.newaxis, j1p : j2p + 1, np.newaxis]
        * (delpm[:, j1p : j2p + 1, :] + delpm[:, j1p - 1 : j2p, :])
    )
    cy[:, j2p + 1, :] = ymass[:, j2p + 1, :] / (
        0.5 * cose[j2p + 1] * (delpm[:, j2p + 1, :] + delpm[:, j2p, :])
    )
    return cx, cy


def _calc_divergence(
    xmass: np.ndarray,
    ymass: np.ndarray,
    geofac: np.ndarray,
    geofac_pc: float,
) -> np.ndarray:
    dpi = np.zeros_like(xmass)
    j1p, j2p = _polar_cap_bounds(xmass.shape[1])
    dpi[:, j1p : j2p + 1, :] = (
        ymass[:, j1p : j2p + 1, :] - ymass[:, j1p + 1 : j2p + 2, :]
    ) * geofac[np.newaxis, j1p : j2p + 1, np.newaxis]
    dpi[:, j1p : j2p + 1, :] += xmass[:, j1p : j2p + 1, :] - np.roll(
        xmass[:, j1p : j2p + 1, :],
        -1,
        axis=2,
    )
    dpi[:, 0, :] = -np.mean(ymass[:, j1p, :], axis=1)[:, np.newaxis] * geofac_pc
    dpi[:, -1, :] = np.mean(ymass[:, j2p + 1, :], axis=1)[:, np.newaxis] * geofac_pc
    dpi[:, 1, :] = dpi[:, 0, :]
    dpi[:, -2, :] = dpi[:, -1, :]
    return dpi


def _calc_vertical_mass_flux(dbk: np.ndarray, dps_ctm: np.ndarray, dpi: np.ndarray) -> np.ndarray:
    wz = np.zeros_like(dpi)
    wz[0] = dpi[0] - dbk[0] * dps_ctm
    for level in range(1, dpi.shape[0] - 1):
        wz[level] = wz[level - 1] + dpi[level] - dbk[level] * dps_ctm
    wz[-1] = 0.0
    return wz


def _polar_cap_bounds(nlat: int) -> tuple[int, int]:
    return 2, nlat - 3
