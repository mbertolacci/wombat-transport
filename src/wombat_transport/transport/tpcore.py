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
    pu_hpa: np.ndarray
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

    This path ports the GEOS-Chem pressure/mass bookkeeping and the active
    tracer branches for the compact low-Courant oracle fixture.
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
    tracer = _advect_tracers(
        tracer_conc=np.asarray(tracer_conc, dtype=np.float64),
        setup=setup,
        area_m2=np.asarray(area_m2, dtype=np.float64),
        fill=fill,
    )
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
        pu_hpa=pu_t[::-1],
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


def _advect_tracers(
    *,
    tracer_conc: np.ndarray,
    setup: TpcoreSetup,
    area_m2: np.ndarray,
    fill: bool,
) -> np.ndarray:
    if tracer_conc.ndim != 4:
        raise ValueError(f"tracer_conc must have shape (tracer, lev, lat, lon), found {tracer_conc.shape}")
    ntracer, nlev, nlat, nlon = tracer_conc.shape
    if setup.delp1_hpa.shape != (nlev, nlat, nlon):
        raise ValueError("TPCORE setup shape does not match tracer_conc")

    delp1 = setup.delp1_hpa[::-1]
    delp2 = setup.delp2_hpa[::-1]
    pu = setup.pu_hpa[::-1]
    xmass = setup.xmass_hpa[::-1]
    ymass = setup.ymass_hpa[::-1]
    wz = setup.vertical_mass_flux_hpa[::-1]
    cx = setup.cx[::-1]
    cy = setup.cy[::-1]
    geofac = setup.geofac
    geofac_pc = setup.geofac_pc
    ua, va = _set_cross_terms(cx, cy)
    jn, js = _set_jn_js(cx)
    area_1d = area_m2[:, 0]

    out = np.empty_like(tracer_conc, dtype=np.float64)
    for tracer_idx in range(ntracer):
        q = tracer_conc[tracer_idx, ::-1, :, :].copy()
        dq1 = np.zeros_like(q)
        for level in range(nlev):
            _average_const_poles(q[level], delp1[level], area_1d)
            dq1[level] = q[level] * delp1[level]
            qqu, qqv = _calc_advec_cross_terms(q[level], ua[level], va[level], int(jn[level]), int(js[level]))
            adx = _xadv_dao2(qqv, ua[level], int(jn[level]), int(js[level]))
            ady = _yadv_dao2(qqu, va[level])
            q[level] = q[level] + ady + adx
            _xtp(
                dq1[level],
                qqv,
                pu[level],
                cx[level],
                xmass[level],
                int(jn[level]),
                int(js[level]),
            )
            _ytp(
                dq1[level],
                qqu,
                qqv,
                cy[level],
                ymass[level],
                geofac,
                geofac_pc,
            )
        _fzppm(delp1, wz, dq1, q)
        if fill:
            _qckxyz(dq1)
        q_after = dq1 / delp2
        q_after[:, 1, :] = q_after[:, 0, :]
        q_after[:, -2, :] = q_after[:, -1, :]
        q_after[q_after < 0.0] = 1.0e-26
        out[tracer_idx] = q_after[::-1]
    return out


def _average_const_poles(q: np.ndarray, delp1: np.ndarray, area_1d: np.ndarray) -> None:
    south_weight = delp1[:2] * area_1d[:2, np.newaxis]
    north_weight = delp1[-2:] * area_1d[-2:, np.newaxis]
    q[:2, :] = np.sum(q[:2, :] * south_weight) / np.sum(south_weight)
    q[-2:, :] = np.sum(q[-2:, :] * north_weight) / np.sum(north_weight)


def _set_cross_terms(cx: np.ndarray, cy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ua = np.zeros_like(cx)
    va = np.zeros_like(cy)
    _nlev, nlat, _nlon = cx.shape
    j1p, j2p = _polar_cap_bounds(nlat)
    ua[:, j1p : j2p + 1, :] = 0.5 * (
        cx[:, j1p : j2p + 1, :] + np.roll(cx[:, j1p : j2p + 1, :], -1, axis=2)
    )
    va[:, 1:nlat - 1, :] = 0.5 * (cy[:, 1:nlat - 1, :] + cy[:, 2:nlat, :])
    if j1p == 1:
        half = cx.shape[2] // 2
        va[:, 0, :half] = 0.5 * (cy[:, 1, :half] - cy[:, 1, half:])
        va[:, 0, half:] = -va[:, 0, :half]
        va[:, -1, :half] = 0.5 * (cy[:, -1, :half] - cy[:, -2, half:])
        va[:, -1, half:] = -va[:, -1, :half]
    return ua, va


def _set_jn_js(cx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nlev, nlat, _nlon = cx.shape
    j1p, j2p = _polar_cap_bounds(nlat)
    js0 = (nlat + 1) // 2 - 1
    jn0 = nlat - (js0 + 1)
    js = np.full(nlev, j1p, dtype=np.int64)
    jn = np.full(nlev, j2p, dtype=np.int64)
    for level in range(nlev):
        for j in range(min(nlat - 1, js0), max(0, j1p) - 1, -1):
            if np.any(np.abs(cx[level, j, :]) > 1.0):
                js[level] = j
                break
        for j in range(max(0, jn0), min(nlat - 1, j2p) + 1):
            if np.any(np.abs(cx[level, j, :]) > 1.0):
                jn[level] = j
                break
    return jn, js


def _calc_advec_cross_terms(
    q: np.ndarray,
    ua: np.ndarray,
    va: np.ndarray,
    jn: int,
    js: int,
) -> tuple[np.ndarray, np.ndarray]:
    nlat, nlon = q.shape
    j1p, j2p = _polar_cap_bounds(nlat)
    qqu = np.zeros_like(q)
    qqv = np.zeros_like(q)
    for j in range(j1p, j2p + 1):
        for i in range(nlon):
            if j <= js or j >= jn:
                iu0 = _trunc_toward_zero(ua[j, i])
                ru = ua[j, i] - float(iu0)
                iu = i - iu0
                if ua[j, i] >= 0.0:
                    qqu[j, i] = _q_lon(q, j, iu) + ru * (_q_lon(q, j, iu - 1) - _q_lon(q, j, iu))
                else:
                    qqu[j, i] = _q_lon(q, j, iu) + ru * (_q_lon(q, j, iu) - _q_lon(q, j, iu + 1))
                qqu[j, i] -= q[j, i]
            else:
                iu = i - _trunc_toward_zero(ua[j, i])
                qqu[j, i] = ua[j, i] * (_q_lon(q, j, iu) - _q_lon(q, j, iu + 1))
            jv = j - _trunc_toward_zero(va[j, i])
            qqv[j, i] = va[j, i] * (_q_lat(q, i, jv) - _q_lat(q, i, jv + 1))
    qqu = q + 0.5 * qqu
    qqv = q + 0.5 * qqv
    return qqu, qqv


def _xadv_dao2(qqv: np.ndarray, ua: np.ndarray, jn: int, js: int) -> np.ndarray:
    nlat, nlon = qqv.shape
    j1p, j2p = _polar_cap_bounds(nlat)
    adx = np.zeros_like(qqv)
    for j in range(j1p, j2p + 1):
        for i in range(nlon):
            iu0 = _nint(ua[j, i])
            ru = float(iu0) - ua[j, i]
            iu = i - iu0
            a1 = 0.5 * (_q_lon(qqv, j, iu + 1) + _q_lon(qqv, j, iu - 1)) - _q_lon(qqv, j, iu)
            b1 = 0.5 * (_q_lon(qqv, j, iu + 1) - _q_lon(qqv, j, iu - 1))
            c1 = _q_lon(qqv, j, iu) - qqv[j, i]
            adx[j, i] = ru * (a1 * ru + b1) + c1
    adx[0, :] = 0.0
    adx[1, :] = 0.0
    adx[-2, :] = 0.0
    adx[-1, :] = 0.0
    return adx


def _yadv_dao2(qqu: np.ndarray, va: np.ndarray) -> np.ndarray:
    nlat, nlon = qqu.shape
    j1p, j2p = _polar_cap_bounds(nlat)
    ady = np.zeros_like(qqu)
    for j in range(j1p - 1, j2p + 2):
        for i in range(nlon):
            jv0 = _nint(va[j, i])
            rv = float(jv0) - va[j, i]
            jv = j - jv0
            a1 = 0.5 * (_q_lat(qqu, i, jv + 1) + _q_lat(qqu, i, jv - 1)) - _q_lat(qqu, i, jv)
            b1 = 0.5 * (_q_lat(qqu, i, jv + 1) - _q_lat(qqu, i, jv - 1))
            c1 = _q_lat(qqu, i, jv) - qqu[j, i]
            ady[j, i] = rv * (a1 * rv + b1) + c1
    _do_y_pole_sum(ady)
    return ady


def _xtp(
    dq1: np.ndarray,
    qqv: np.ndarray,
    pu: np.ndarray,
    cx: np.ndarray,
    xmass: np.ndarray,
    jn: int,
    js: int,
) -> None:
    nlat, nlon = dq1.shape
    j1p, j2p = _polar_cap_bounds(nlat)
    dcx = _xmist(qqv)
    fx = np.zeros_like(dq1)
    jvan = max(1, nlat // 18)
    for j in range(j1p, j2p + 1):
        if j > js and j < jn:
            if j == j1p or j == j2p:
                for i in range(nlon):
                    iu = i - _trunc_toward_zero(cx[j, i])
                    fx[j, i] = _q_lon(qqv, j, iu)
            elif j <= j1p + jvan or j >= j2p - jvan:
                for i in range(nlon):
                    iu = i - _trunc_toward_zero(cx[j, i])
                    fx[j, i] = _q_lon(qqv, j, iu) + _dcx_lon(dcx, j, iu) * (_sign(1.0, cx[j, i]) - cx[j, i])
            else:
                raise NotImplementedError("Fxppm branch is not active for the compact low-Courant fixture")
            fx[j, :] *= xmass[j, :]
        else:
            for i in range(nlon):
                ic = _trunc_toward_zero(cx[j, i])
                isav = i - ic
                iu = i - _trunc_toward_zero(cx[j, i])
                rc = cx[j, i] - float(ic)
                if j == j1p or j == j2p:
                    val = rc * _q_lon(qqv, j, iu)
                else:
                    val = rc * (_q_lon(qqv, j, iu) + _dcx_lon(dcx, j, iu) * (_sign(1.0, rc) - rc))
                if cx[j, i] > 1.0:
                    for ix in range(isav, i):
                        val += _q_lon(qqv, j, ix)
                elif cx[j, i] < -1.0:
                    for ix in range(i, isav):
                        val -= _q_lon(qqv, j, ix)
                fx[j, i] = pu[j, i] * val
    for j in range(j1p, j2p + 1):
        dq1[j, :-1] += fx[j, :-1] - fx[j, 1:]
        dq1[j, -1] += fx[j, -1] - fx[j, 0]


def _ytp(
    dq1: np.ndarray,
    qqu: np.ndarray,
    qqv: np.ndarray,
    cy: np.ndarray,
    ymass: np.ndarray,
    geofac: np.ndarray,
    geofac_pc: float,
) -> None:
    nlat, nlon = dq1.shape
    j1p, j2p = _polar_cap_bounds(nlat)
    dcy = _ymist(qqu)
    _fyppm(cy, dcy, qqu, qqv)
    for j in range(j1p, j2p + 2):
        qqv[j, :] *= ymass[j, :]
    for j in range(j1p, j2p + 1):
        dq1[j, :] += (qqv[j, :] - qqv[j + 1, :]) * geofac[j]
    sumsp = np.sum(qqv[j1p, :])
    sumnp = np.sum(qqv[j2p + 1, :])
    dq_sp = dq1[0, 0] - sumsp / float(nlon) * geofac_pc
    dq_np = dq1[-1, 0] + sumnp / float(nlon) * geofac_pc
    dq1[0, :] = dq_sp
    dq1[-1, :] = dq_np
    dq1[1, :] = dq_sp
    dq1[-2, :] = dq_np


def _xmist(qqv: np.ndarray) -> np.ndarray:
    nlat, nlon = qqv.shape
    j1p, j2p = _polar_cap_bounds(nlat)
    dcx = np.zeros_like(qqv)
    r24 = 1.0 / 24.0
    for j in range(j1p + 1, j2p):
        for i in range(nlon):
            tmp = (8.0 * (_q_lon(qqv, j, i + 1) - _q_lon(qqv, j, i - 1)) + _q_lon(qqv, j, i - 2) - _q_lon(qqv, j, i + 2)) * r24
            pmax = max(_q_lon(qqv, j, i - 1), qqv[j, i], _q_lon(qqv, j, i + 1)) - qqv[j, i]
            pmin = qqv[j, i] - min(_q_lon(qqv, j, i - 1), qqv[j, i], _q_lon(qqv, j, i + 1))
            dcx[j, i] = _sign(min(abs(tmp), pmin, pmax), tmp)
    return dcx


def _ymist(qqu: np.ndarray) -> np.ndarray:
    nlat, nlon = qqu.shape
    dcy = np.zeros_like(qqu)
    r24 = 1.0 / 24.0
    for j in range(-2, nlat - 2):
        out_j = j + 2
        for i in range(nlon):
            tmp = (8.0 * (_q_lat(qqu, i, j + 3) - _q_lat(qqu, i, j + 1)) + _q_lat(qqu, i, j) - _q_lat(qqu, i, j + 4)) * r24
            pmax = max(_q_lat(qqu, i, j + 1), _q_lat(qqu, i, j + 2), _q_lat(qqu, i, j + 3)) - _q_lat(qqu, i, j + 2)
            pmin = _q_lat(qqu, i, j + 2) - min(_q_lat(qqu, i, j + 1), _q_lat(qqu, i, j + 2), _q_lat(qqu, i, j + 3))
            dcy[out_j, i] = _sign(min(abs(tmp), pmin, pmax), tmp)
    dcy[0, :] = 0.0
    dcy[-1, :] = 0.0
    return dcy


def _fyppm(cy: np.ndarray, dcy: np.ndarray, qqu: np.ndarray, qqv: np.ndarray) -> None:
    nlat, nlon = qqu.shape
    j1p, j2p = _polar_cap_bounds(nlat)
    r13 = 1.0 / 3.0
    r23 = 2.0 / 3.0
    a6 = np.zeros_like(qqu)
    al = np.zeros_like(qqu)
    ar = np.zeros_like(qqu)
    for j in range(1, nlat):
        al[j, :] = 0.5 * (qqu[j - 1, :] + qqu[j, :]) + (dcy[j - 1, :] - dcy[j, :]) * r13
        ar[j - 1, :] = al[j, :]
    half = nlon // 2
    al[0, :half] = al[1, half:]
    al[0, half:] = al[1, :half]
    ar[-1, :half] = ar[-2, half:]
    ar[-1, half:] = ar[-2, :half]
    for j in range(1, nlat - 1):
        a6[j, :] = 3.0 * (qqu[j, :] + qqu[j, :] - (al[j, :] + ar[j, :]))
    _lmtppm_2d(a6, al, ar, dcy, qqu, range(1, nlat - 1))
    for j in range(j1p, j2p + 2):
        jm1 = j - 1
        for i in range(nlon):
            if cy[j, i] > 0.0:
                qqv[j, i] = ar[jm1, i] + 0.5 * cy[j, i] * (
                    al[jm1, i] - ar[jm1, i] + a6[jm1, i] * (1.0 - r23 * cy[j, i])
                )
            else:
                qqv[j, i] = al[j, i] - 0.5 * cy[j, i] * (
                    ar[j, i] - al[j, i] + a6[j, i] * (1.0 + r23 * cy[j, i])
                )


def _fzppm(delp1: np.ndarray, wz: np.ndarray, dq1: np.ndarray, q: np.ndarray) -> None:
    nlev, nlat, nlon = q.shape
    r13 = 1.0 / 3.0
    r23 = 2.0 / 3.0
    dpi = np.zeros_like(q)
    dc = np.zeros_like(q)
    for k in range(nlev - 1):
        dpi[k] = q[k + 1] - q[k]
    for k in range(1, nlev - 1):
        for j in range(nlat):
            for i in range(nlon):
                c0 = delp1[k, j, i] / (delp1[k - 1, j, i] + delp1[k, j, i] + delp1[k + 1, j, i])
                c1 = (delp1[k - 1, j, i] + 0.5 * delp1[k, j, i]) / (delp1[k + 1, j, i] + delp1[k, j, i])
                c2 = (delp1[k + 1, j, i] + 0.5 * delp1[k, j, i]) / (delp1[k - 1, j, i] + delp1[k, j, i])
                tmp = c0 * (c1 * dpi[k, j, i] + c2 * dpi[k - 1, j, i])
                qmax = max(q[k - 1, j, i], q[k, j, i], q[k + 1, j, i]) - q[k, j, i]
                qmin = q[k, j, i] - min(q[k - 1, j, i], q[k, j, i], q[k + 1, j, i])
                dc[k, j, i] = _sign(min(abs(tmp), qmax, qmin), tmp)
    for j in range(nlat):
        if j in (1, nlat - 2):
            continue
        al = np.zeros((nlev, nlon), dtype=np.float64)
        ar = np.zeros((nlev, nlon), dtype=np.float64)
        a6 = np.zeros((nlev, nlon), dtype=np.float64)
        dca = dc[:, j, :].copy()
        dlp = delp1[:, j, :]
        qq = q[:, j, :]
        wza = wz[:, j, :]
        fac1 = dpi[1, j, :] - dpi[0, j, :] * (dlp[1, :] + dlp[2, :]) / (dlp[0, :] + dlp[1, :])
        fac2 = (dlp[1, :] + dlp[2, :]) * (dlp[0, :] + dlp[1, :] + dlp[2, :])
        aa = 3.0 * fac1 / fac2
        bb = 2.0 * dpi[0, j, :] / (dlp[0, :] + dlp[1, :]) - r23 * aa * (2.0 * dlp[0, :] + dlp[1, :])
        al[0, :] = qq[0, :] - dlp[0, :] * (r13 * aa * dlp[0, :] + 0.5 * bb)
        al[1, :] = dlp[0, :] * (aa * dlp[0, :] + bb) + al[0, :]
        for i in range(nlon):
            if qq[0, i] * al[0, i] <= 0.0:
                al[0, i] = 0.0
                dca[0, i] = 0.0
            else:
                dca[0, i] = qq[0, i] - al[0, i]
        fac1b = dpi[-2, j, :] * (dlp[-1, :] * dlp[-1, :]) / ((dlp[-1, :] + dlp[-2, :]) * (2.0 * dlp[-1, :] + dlp[-2, :]))
        ar[-1, :] = qq[-1, :] + fac1b
        al[-1, :] = qq[-1, :] - (fac1b + fac1b)
        ar[-1, :] = np.where(qq[-1, :] * ar[-1, :] <= 0.0, 0.0, ar[-1, :])
        dca[-1, :] = ar[-1, :] - qq[-1, :]
        for k in range(2, nlev - 1):
            c1 = dpi[k - 1, j, :] * dlp[k - 1, :] / (dlp[k - 1, :] + dlp[k, :])
            c2 = 2.0 / (dlp[k - 2, :] + dlp[k - 1, :] + dlp[k, :] + dlp[k + 1, :])
            a1 = (dlp[k - 2, :] + dlp[k - 1, :]) / (2.0 * dlp[k - 1, :] + dlp[k, :])
            a2 = (dlp[k, :] + dlp[k + 1, :]) / (2.0 * dlp[k, :] + dlp[k - 1, :])
            al[k, :] = qq[k - 1, :] + c1 + c2 * (
                dlp[k, :] * (c1 * (a1 - a2) + a2 * dca[k - 1, :])
                - dlp[k - 1, :] * a1 * dca[k, :]
            )
        ar[:-1, :] = al[1:, :]
        for k in (0, 1, nlev - 2, nlev - 1):
            a6[k, :] = 3.0 * (qq[k, :] + qq[k, :] - (al[k, :] + ar[k, :]))
            _lmtppm_1d(a6[k, :], al[k, :], ar[k, :], dca[k, :], qq[k, :], 0)
        for k in range(1, nlev - 1):
            dca[k, :] = dpi[k, j, :] - dpi[k - 1, j, :]
        for k in range(2, nlev - 2):
            qmp = qq[k, :] + 2.0 * dpi[k - 1, j, :]
            lac = qq[k, :] + 1.5 * dca[k - 1, :] + 0.5 * dpi[k - 1, j, :]
            qmin = np.minimum.reduce([qq[k, :], qmp, lac])
            qmax = np.maximum.reduce([qq[k, :], qmp, lac])
            ar[k, :] = np.minimum(np.maximum(ar[k, :], qmin), qmax)
            qmp = qq[k, :] - 2.0 * dpi[k, j, :]
            lac = qq[k, :] + 1.5 * dca[k + 1, :] - 0.5 * dpi[k, j, :]
            qmin = np.minimum.reduce([qq[k, :], qmp, lac])
            qmax = np.maximum.reduce([qq[k, :], qmp, lac])
            al[k, :] = np.minimum(np.maximum(al[k, :], qmin), qmax)
            a6[k, :] = 3.0 * (qq[k, :] + qq[k, :] - (ar[k, :] + al[k, :]))
        flux = np.zeros((nlev, nlon), dtype=np.float64)
        for k in range(nlev - 1):
            for i in range(nlon):
                if wza[k, i] > 0.0:
                    cm = wza[k, i] / dlp[k, i]
                    val = ar[k, i] + 0.5 * cm * (al[k, i] - ar[k, i] + a6[k, i] * (1.0 - r23 * cm))
                else:
                    cp = wza[k, i] / dlp[k + 1, i]
                    val = al[k + 1, i] + 0.5 * cp * (al[k + 1, i] - ar[k + 1, i] - a6[k + 1, i] * (1.0 + r23 * cp))
                flux[k + 1, i] = wza[k, i] * val
        dq1[0, j, :] -= flux[1, :]
        dq1[-1, j, :] += flux[-1, :]
        for k in range(1, nlev - 1):
            dq1[k, j, :] += flux[k, :] - flux[k + 1, :]


def _qckxyz(dq1: np.ndarray) -> None:
    _nlev, nlat, _nlon = dq1.shape
    j1p, j2p = _polar_cap_bounds(nlat)
    for j in range(j1p, j2p + 1):
        for i in range(dq1.shape[2]):
            if dq1[0, j, i] < 0.0:
                dq1[1, j, i] += dq1[0, j, i]
                dq1[0, j, i] = 0.0
            for k in range(1, dq1.shape[0] - 1):
                if dq1[k, j, i] < 0.0:
                    qup = dq1[k - 1, j, i]
                    qly = -dq1[k, j, i]
                    dup = min(qly, qup)
                    dq1[k - 1, j, i] = qup - dup
                    dq1[k, j, i] = dup - qly
                    dq1[k + 1, j, i] += dq1[k, j, i]
                    dq1[k, j, i] = 0.0
            if dq1[-1, j, i] < 0.0:
                qup = dq1[-2, j, i]
                qly = -dq1[-1, j, i]
                dup = min(qly, qup)
                dq1[-2, j, i] = qup - dup
                dq1[-1, j, i] = 0.0


def _do_y_pole_sum(ady: np.ndarray) -> None:
    south = float(np.mean(ady[1, :]))
    north = float(np.mean(ady[-2, :]))
    ady[0, :] = south
    ady[1, :] = south
    ady[-2, :] = north
    ady[-1, :] = north


def _lmtppm_2d(
    a6: np.ndarray,
    al: np.ndarray,
    ar: np.ndarray,
    dc: np.ndarray,
    qa: np.ndarray,
    rows: range,
) -> None:
    for j in rows:
        _lmtppm_1d(a6[j, :], al[j, :], ar[j, :], dc[j, :], qa[j, :], 0)


def _lmtppm_1d(a6: np.ndarray, al: np.ndarray, ar: np.ndarray, dc: np.ndarray, qa: np.ndarray, lmt: int) -> None:
    if lmt != 0:
        raise NotImplementedError("Only the full monotonic PPM limiter is needed for the current TPCORE path")
    for idx in range(qa.size):
        if dc[idx] == 0.0:
            a6[idx] = 0.0
            al[idx] = qa[idx]
            ar[idx] = qa[idx]
        else:
            da1 = ar[idx] - al[idx]
            da2 = da1 * da1
            a6da = a6[idx] * da1
            if a6da < -da2:
                a6[idx] = 3.0 * (al[idx] - qa[idx])
                ar[idx] = al[idx] - a6[idx]
            elif a6da > da2:
                a6[idx] = 3.0 * (ar[idx] - qa[idx])
                al[idx] = ar[idx] - a6[idx]


def _q_lon(q: np.ndarray, j: int, i: int) -> float:
    if j < 0 or j >= q.shape[0]:
        return 0.0
    return float(q[j, i % q.shape[1]])


def _dcx_lon(dcx: np.ndarray, j: int, i: int) -> float:
    return float(dcx[j, i % dcx.shape[1]])


def _q_lat(q: np.ndarray, i: int, j: int) -> float:
    if j < 0 or j >= q.shape[0]:
        return 0.0
    return float(q[j, i])


def _trunc_toward_zero(value: float) -> int:
    return int(value)


def _nint(value: float) -> int:
    return int(np.rint(value))


def _sign(magnitude: float, sign_source: float) -> float:
    return abs(magnitude) if sign_source >= 0.0 else -abs(magnitude)
