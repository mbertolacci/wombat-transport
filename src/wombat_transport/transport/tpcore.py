"""GEOS-Chem-oriented NumPy TPCORE pieces.

The first tracked oracle fixture is intentionally compact and low-Courant:
``tests/fixtures/tpcore_snapshot_v1`` has max ``|cx|`` around 0.0023 and max
``|cy|`` around 0.0008. Matching that fixture is useful one-step coverage for
the ordinary low-Courant branches. Additional branch fixtures cover X full-PPM
and compact large-Courant E-W behavior. Full-grid validation must
keep those branch limits visible instead of treating any compact fixture as
comprehensive TPCORE parity.
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
class TpcoreTrace:
    """Optional one-step tracer checkpoints in project orientation."""

    q_after_pole_average: np.ndarray
    dq_after_init: np.ndarray
    q_after_cross_terms: np.ndarray
    dq_after_xtp: np.ndarray
    dq_after_ytp: np.ndarray
    dq_after_fzppm: np.ndarray
    dq_after_fill: np.ndarray
    tracer_conc_after: np.ndarray


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


@dataclass(frozen=True)
class TpcoreBranchReport:
    shape: tuple[int, int, int]
    max_abs_cx: float
    max_abs_cy: float
    has_large_cx: bool
    has_large_cy: bool
    needs_fxppm: bool
    x_ffsl_active: bool
    x_ffsl_endpoint_active: bool
    x_near_pole_vanleer_active: bool
    jn: tuple[int, ...]
    js: tuple[int, ...]
    unsupported_reasons: tuple[str, ...]

    @property
    def is_supported(self) -> bool:
        return not self.unsupported_reasons


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
    validate_tpcore_branch_support(setup)
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


def trace_tpcore_one_step(
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
) -> tuple[TpcoreState, TpcoreTrace]:
    """Run TPCORE and return diagnostic checkpoints for discrepancy searches."""

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
    validate_tpcore_branch_support(setup)
    tracer, trace = _advect_tracers(
        tracer_conc=np.asarray(tracer_conc, dtype=np.float64),
        setup=setup,
        area_m2=np.asarray(area_m2, dtype=np.float64),
        fill=fill,
        trace=True,
    )
    if fill:
        tracer[tracer < 0.0] = 1.0e-26
    state = TpcoreState(
        tracer_conc_after=tracer,
        xmass_hpa=setup.xmass_hpa,
        ymass_hpa=setup.ymass_hpa,
        surface_pressure_hpa=setup.surface_pressure_hpa,
        delp1_hpa=setup.delp1_hpa,
        delpm_hpa=setup.delpm_hpa,
        delp2_hpa=setup.delp2_hpa,
        vertical_mass_flux_hpa=setup.vertical_mass_flux_hpa,
    )
    if trace is None:
        raise AssertionError("trace=True did not produce a TPCORE trace")
    return state, trace


def analyze_tpcore_branches(setup: TpcoreSetup) -> TpcoreBranchReport:
    """Classify which currently ported TPCORE branches a setup would use."""

    cx = setup.cx[::-1]
    cy = setup.cy[::-1]
    nlev, nlat, nlon = cx.shape
    j1p, j2p = _polar_cap_bounds(nlat)
    jn, js = _set_jn_js(cx)
    jvan = max(1, nlat // 18)
    max_abs_cx = float(np.max(np.abs(cx)))
    max_abs_cy = float(np.max(np.abs(cy)))
    has_large_cx = bool(max_abs_cx > 1.0)
    has_large_cy = bool(max_abs_cy > 1.0)
    needs_fxppm = False
    x_ffsl_active = False
    x_ffsl_endpoint_active = False
    x_near_pole_vanleer_active = False
    for level in range(nlev):
        for lat_index in range(j1p, j2p + 1):
            if lat_index <= js[level] or lat_index >= jn[level]:
                x_ffsl_active = True
                if lat_index == j1p or lat_index == j2p:
                    x_ffsl_endpoint_active = True
                continue
            if lat_index == j1p or lat_index == j2p:
                continue
            if lat_index <= j1p + jvan or lat_index >= j2p - jvan:
                x_near_pole_vanleer_active = True
                continue
            needs_fxppm = True
            break
        if needs_fxppm:
            break

    reasons: list[str] = []
    if has_large_cy:
        reasons.append("large-Courant N-S branch has not been validated")

    return TpcoreBranchReport(
        shape=(nlev, nlat, nlon),
        max_abs_cx=max_abs_cx,
        max_abs_cy=max_abs_cy,
        has_large_cx=has_large_cx,
        has_large_cy=has_large_cy,
        needs_fxppm=needs_fxppm,
        x_ffsl_active=x_ffsl_active,
        x_ffsl_endpoint_active=x_ffsl_endpoint_active,
        x_near_pole_vanleer_active=x_near_pole_vanleer_active,
        jn=tuple(int(value) for value in jn),
        js=tuple(int(value) for value in js),
        unsupported_reasons=tuple(reasons),
    )


def validate_tpcore_branch_support(setup: TpcoreSetup) -> TpcoreBranchReport:
    """Raise before tracer advection if the setup leaves the validated path."""

    report = analyze_tpcore_branches(setup)
    if not report.is_supported:
        reasons = "; ".join(report.unsupported_reasons)
        raise NotImplementedError(
            "TPCORE branch set is outside the currently validated compact low-Courant path: "
            f"{reasons}. shape={report.shape}, max_abs_cx={report.max_abs_cx:.8e}, "
            f"max_abs_cy={report.max_abs_cy:.8e}"
        )
    return report


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
    trace: bool = False,
) -> np.ndarray | tuple[np.ndarray, TpcoreTrace | None]:
    if tracer_conc.ndim != 4:
        raise ValueError(f"tracer_conc must have shape (tracer, lev, lat, lon), found {tracer_conc.shape}")
    _ntracer, nlev, nlat, nlon = tracer_conc.shape
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

    q = tracer_conc[:, ::-1, :, :].copy()
    dq1 = np.zeros_like(q)
    q_after_pole_average = np.empty_like(tracer_conc, dtype=np.float64) if trace else None
    dq_after_init = np.empty_like(tracer_conc, dtype=np.float64) if trace else None
    q_after_cross_terms = np.empty_like(tracer_conc, dtype=np.float64) if trace else None
    dq_after_xtp = np.empty_like(tracer_conc, dtype=np.float64) if trace else None
    dq_after_ytp = np.empty_like(tracer_conc, dtype=np.float64) if trace else None
    dq_after_fzppm = np.empty_like(tracer_conc, dtype=np.float64) if trace else None
    dq_after_fill = np.empty_like(tracer_conc, dtype=np.float64) if trace else None
    for level in range(nlev):
        _average_const_poles_batch(q[:, level], delp1[level], area_1d)
        if trace:
            q_after_pole_average[:, nlev - level - 1] = q[:, level]
        dq1[:, level] = q[:, level] * delp1[level][np.newaxis, :, :]
        if trace:
            dq_after_init[:, nlev - level - 1] = dq1[:, level]
        qqu, qqv = _calc_advec_cross_terms_batch(q[:, level], ua[level], va[level], int(jn[level]), int(js[level]))
        adx = _xadv_dao2_batch(qqv, ua[level], int(jn[level]), int(js[level]))
        ady = _yadv_dao2_batch(qqu, va[level])
        q[:, level] = q[:, level] + ady + adx
        if trace:
            q_after_cross_terms[:, nlev - level - 1] = q[:, level]
        _xtp_batch(
            dq1[:, level],
            qqv,
            pu[level],
            cx[level],
            xmass[level],
            int(jn[level]),
            int(js[level]),
        )
        if trace:
            dq_after_xtp[:, nlev - level - 1] = dq1[:, level]
        _ytp_batch(
            dq1[:, level],
            qqu,
            qqv,
            cy[level],
            ymass[level],
            geofac,
            geofac_pc,
        )
        if trace:
            dq_after_ytp[:, nlev - level - 1] = dq1[:, level]
    _fzppm_batch(delp1, wz, dq1, q)
    if trace:
        dq_after_fzppm[:] = dq1[:, ::-1]
    if fill:
        _qckxyz_batch(dq1)
    if trace:
        dq_after_fill[:] = dq1[:, ::-1]
    q_after = dq1 / delp2[np.newaxis, :, :, :]
    q_after[:, :, 1, :] = q_after[:, :, 0, :]
    q_after[:, :, -2, :] = q_after[:, :, -1, :]
    q_after[q_after < 0.0] = 1.0e-26
    out = q_after[:, ::-1].copy()
    if not trace:
        return out
    return out, TpcoreTrace(
        q_after_pole_average=_require_trace_array(q_after_pole_average),
        dq_after_init=_require_trace_array(dq_after_init),
        q_after_cross_terms=_require_trace_array(q_after_cross_terms),
        dq_after_xtp=_require_trace_array(dq_after_xtp),
        dq_after_ytp=_require_trace_array(dq_after_ytp),
        dq_after_fzppm=_require_trace_array(dq_after_fzppm),
        dq_after_fill=_require_trace_array(dq_after_fill),
        tracer_conc_after=out.copy(),
    )


def _require_trace_array(values: np.ndarray | None) -> np.ndarray:
    if values is None:
        raise AssertionError("missing TPCORE trace array")
    return values


def _average_const_poles(q: np.ndarray, delp1: np.ndarray, area_1d: np.ndarray) -> None:
    south_weight = delp1[:2] * area_1d[:2, np.newaxis]
    north_weight = delp1[-2:] * area_1d[-2:, np.newaxis]
    q[:2, :] = np.sum(q[:2, :] * south_weight) / np.sum(south_weight)
    q[-2:, :] = np.sum(q[-2:, :] * north_weight) / np.sum(north_weight)


def _average_const_poles_batch(q: np.ndarray, delp1: np.ndarray, area_1d: np.ndarray) -> None:
    south_weight = delp1[:2] * area_1d[:2, np.newaxis]
    north_weight = delp1[-2:] * area_1d[-2:, np.newaxis]
    south = np.sum(q[:, :2, :] * south_weight[np.newaxis, :, :], axis=(1, 2)) / np.sum(south_weight)
    north = np.sum(q[:, -2:, :] * north_weight[np.newaxis, :, :], axis=(1, 2)) / np.sum(north_weight)
    q[:, :2, :] = south[:, np.newaxis, np.newaxis]
    q[:, -2:, :] = north[:, np.newaxis, np.newaxis]


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
                iu = _real_index_offset(i, ua[j, i])
                qqu[j, i] = ua[j, i] * (_q_lon(q, j, iu) - _q_lon(q, j, iu + 1))
            jv = _real_index_offset(j, va[j, i])
            qqv[j, i] = va[j, i] * (_q_lat(q, i, jv) - _q_lat(q, i, jv + 1))
    qqu = q + 0.5 * qqu
    qqv = q + 0.5 * qqv
    return qqu, qqv


def _calc_advec_cross_terms_batch(
    q: np.ndarray,
    ua: np.ndarray,
    va: np.ndarray,
    jn: int,
    js: int,
) -> tuple[np.ndarray, np.ndarray]:
    _ntracer, nlat, nlon = q.shape
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
                    qqu[:, j, i] = _q_lon_batch(q, j, iu) + ru * (
                        _q_lon_batch(q, j, iu - 1) - _q_lon_batch(q, j, iu)
                    )
                else:
                    qqu[:, j, i] = _q_lon_batch(q, j, iu) + ru * (
                        _q_lon_batch(q, j, iu) - _q_lon_batch(q, j, iu + 1)
                    )
                qqu[:, j, i] -= q[:, j, i]
            else:
                iu = _real_index_offset(i, ua[j, i])
                qqu[:, j, i] = ua[j, i] * (_q_lon_batch(q, j, iu) - _q_lon_batch(q, j, iu + 1))
            jv = _real_index_offset(j, va[j, i])
            qqv[:, j, i] = va[j, i] * (_q_lat_batch(q, i, jv) - _q_lat_batch(q, i, jv + 1))
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


def _xadv_dao2_batch(qqv: np.ndarray, ua: np.ndarray, jn: int, js: int) -> np.ndarray:
    _ntracer, nlat, nlon = qqv.shape
    j1p, j2p = _polar_cap_bounds(nlat)
    adx = np.zeros_like(qqv)
    for j in range(j1p, j2p + 1):
        for i in range(nlon):
            iu0 = _nint(ua[j, i])
            ru = float(iu0) - ua[j, i]
            iu = i - iu0
            a1 = 0.5 * (_q_lon_batch(qqv, j, iu + 1) + _q_lon_batch(qqv, j, iu - 1)) - _q_lon_batch(
                qqv, j, iu
            )
            b1 = 0.5 * (_q_lon_batch(qqv, j, iu + 1) - _q_lon_batch(qqv, j, iu - 1))
            c1 = _q_lon_batch(qqv, j, iu) - qqv[:, j, i]
            adx[:, j, i] = ru * (a1 * ru + b1) + c1
    adx[:, 0, :] = 0.0
    adx[:, 1, :] = 0.0
    adx[:, -2, :] = 0.0
    adx[:, -1, :] = 0.0
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


def _yadv_dao2_batch(qqu: np.ndarray, va: np.ndarray) -> np.ndarray:
    _ntracer, nlat, nlon = qqu.shape
    j1p, j2p = _polar_cap_bounds(nlat)
    ady = np.zeros_like(qqu)
    for j in range(j1p - 1, j2p + 2):
        for i in range(nlon):
            jv0 = _nint(va[j, i])
            rv = float(jv0) - va[j, i]
            jv = j - jv0
            a1 = 0.5 * (_q_lat_batch(qqu, i, jv + 1) + _q_lat_batch(qqu, i, jv - 1)) - _q_lat_batch(
                qqu, i, jv
            )
            b1 = 0.5 * (_q_lat_batch(qqu, i, jv + 1) - _q_lat_batch(qqu, i, jv - 1))
            c1 = _q_lat_batch(qqu, i, jv) - qqu[:, j, i]
            ady[:, j, i] = rv * (a1 * rv + b1) + c1
    _do_y_pole_sum_batch(ady)
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
                    iu = _real_index_offset(i, cx[j, i])
                    fx[j, i] = _q_lon(qqv, j, iu)
            elif j <= j1p + jvan or j >= j2p - jvan:
                for i in range(nlon):
                    iu = _real_index_offset(i, cx[j, i])
                    fx[j, i] = _q_lon(qqv, j, iu) + _dcx_lon(dcx, j, iu) * (_sign(1.0, cx[j, i]) - cx[j, i])
            else:
                _fxppm_row(j, cx, dcx, fx, qqv)
            fx[j, :] *= xmass[j, :]
        else:
            for i in range(nlon):
                ic = _trunc_toward_zero(cx[j, i])
                isav = i - ic
                iu = _real_index_offset(i, cx[j, i])
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


def _xtp_batch(
    dq1: np.ndarray,
    qqv: np.ndarray,
    pu: np.ndarray,
    cx: np.ndarray,
    xmass: np.ndarray,
    jn: int,
    js: int,
) -> None:
    _ntracer, nlat, nlon = dq1.shape
    j1p, j2p = _polar_cap_bounds(nlat)
    dcx = _xmist_batch(qqv)
    fx = np.zeros_like(dq1)
    jvan = max(1, nlat // 18)
    for j in range(j1p, j2p + 1):
        if j > js and j < jn:
            if j == j1p or j == j2p:
                for i in range(nlon):
                    iu = _real_index_offset(i, cx[j, i])
                    fx[:, j, i] = _q_lon_batch(qqv, j, iu)
            elif j <= j1p + jvan or j >= j2p - jvan:
                for i in range(nlon):
                    iu = _real_index_offset(i, cx[j, i])
                    fx[:, j, i] = _q_lon_batch(qqv, j, iu) + _dcx_lon_batch(dcx, j, iu) * (
                        _sign(1.0, cx[j, i]) - cx[j, i]
                    )
            else:
                _fxppm_row_batch(j, cx, dcx, fx, qqv)
            fx[:, j, :] *= xmass[j, :][np.newaxis, :]
        else:
            for i in range(nlon):
                ic = _trunc_toward_zero(cx[j, i])
                isav = i - ic
                iu = _real_index_offset(i, cx[j, i])
                rc = cx[j, i] - float(ic)
                if j == j1p or j == j2p:
                    val = rc * _q_lon_batch(qqv, j, iu)
                else:
                    val = rc * (_q_lon_batch(qqv, j, iu) + _dcx_lon_batch(dcx, j, iu) * (_sign(1.0, rc) - rc))
                if cx[j, i] > 1.0:
                    for ix in range(isav, i):
                        val += _q_lon_batch(qqv, j, ix)
                elif cx[j, i] < -1.0:
                    for ix in range(i, isav):
                        val -= _q_lon_batch(qqv, j, ix)
                fx[:, j, i] = pu[j, i] * val
    for j in range(j1p, j2p + 1):
        dq1[:, j, :-1] += fx[:, j, :-1] - fx[:, j, 1:]
        dq1[:, j, -1] += fx[:, j, -1] - fx[:, j, 0]


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


def _ytp_batch(
    dq1: np.ndarray,
    qqu: np.ndarray,
    qqv: np.ndarray,
    cy: np.ndarray,
    ymass: np.ndarray,
    geofac: np.ndarray,
    geofac_pc: float,
) -> None:
    _ntracer, nlat, nlon = dq1.shape
    j1p, j2p = _polar_cap_bounds(nlat)
    dcy = _ymist_batch(qqu)
    _fyppm_batch(cy, dcy, qqu, qqv)
    for j in range(j1p, j2p + 2):
        qqv[:, j, :] *= ymass[j, :][np.newaxis, :]
    for j in range(j1p, j2p + 1):
        dq1[:, j, :] += (qqv[:, j, :] - qqv[:, j + 1, :]) * geofac[j]
    sumsp = np.sum(qqv[:, j1p, :], axis=1)
    sumnp = np.sum(qqv[:, j2p + 1, :], axis=1)
    dq_sp = dq1[:, 0, 0] - sumsp / float(nlon) * geofac_pc
    dq_np = dq1[:, -1, 0] + sumnp / float(nlon) * geofac_pc
    dq1[:, 0, :] = dq_sp[:, np.newaxis]
    dq1[:, -1, :] = dq_np[:, np.newaxis]
    dq1[:, 1, :] = dq_sp[:, np.newaxis]
    dq1[:, -2, :] = dq_np[:, np.newaxis]


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


def _xmist_batch(qqv: np.ndarray) -> np.ndarray:
    _ntracer, nlat, _nlon = qqv.shape
    j1p, j2p = _polar_cap_bounds(nlat)
    dcx = np.zeros_like(qqv)
    r24 = 1.0 / 24.0
    row = qqv[:, j1p + 1 : j2p, :]
    left1 = np.roll(row, 1, axis=2)
    right1 = np.roll(row, -1, axis=2)
    tmp = (8.0 * (right1 - left1) + np.roll(row, 2, axis=2) - np.roll(row, -2, axis=2)) * r24
    pmax = np.maximum.reduce([left1, row, right1]) - row
    pmin = row - np.minimum.reduce([left1, row, right1])
    dcx[:, j1p + 1 : j2p, :] = _sign_array(np.minimum.reduce([np.abs(tmp), pmin, pmax]), tmp)
    return dcx


def _fxppm_row(j: int, cx: np.ndarray, dcx: np.ndarray, fx: np.ndarray, qqv: np.ndarray) -> None:
    nlon = qqv.shape[1]
    r13 = 1.0 / 3.0
    r23 = 2.0 / 3.0
    a6 = np.zeros(nlon, dtype=np.float64)
    al = np.zeros(nlon, dtype=np.float64)
    ar = np.zeros(nlon, dtype=np.float64)
    dc = np.zeros(nlon, dtype=np.float64)
    qa = np.zeros(nlon, dtype=np.float64)
    for i in range(nlon):
        rval = 0.5 * (_q_lon(qqv, j, i - 1) + _q_lon(qqv, j, i)) + (
            _dcx_lon(dcx, j, i - 1) - _dcx_lon(dcx, j, i)
        ) * r13
        al[i] = rval
        ar[(i - 1) % nlon] = rval
        dc[i] = _dcx_lon(dcx, j, i)
        qa[i] = _q_lon(qqv, j, i)
    a6[:] = 3.0 * (qa + qa - (al + ar))
    _lmtppm_1d(a6, al, ar, dc, qa, 0)
    for i in range(nlon):
        if cx[j, i] > 0.0:
            im1 = (i - 1) % nlon
            fx[j, i] = ar[im1] + 0.5 * cx[j, i] * (
                al[im1] - ar[im1] + a6[im1] * (1.0 - r23 * cx[j, i])
            )
        else:
            fx[j, i] = al[i] - 0.5 * cx[j, i] * (ar[i] - al[i] + a6[i] * (1.0 + r23 * cx[j, i]))


def _fxppm_row_batch(j: int, cx: np.ndarray, dcx: np.ndarray, fx: np.ndarray, qqv: np.ndarray) -> None:
    _ntracer, _nlat, nlon = qqv.shape
    r13 = 1.0 / 3.0
    r23 = 2.0 / 3.0
    qa = qqv[:, j, :]
    dc = dcx[:, j, :]
    al = 0.5 * (np.roll(qa, 1, axis=1) + qa) + (np.roll(dc, 1, axis=1) - dc) * r13
    ar = np.roll(al, -1, axis=1)
    a6 = 3.0 * (qa + qa - (al + ar))
    _lmtppm_last_axis(a6, al, ar, dc, qa, 0)
    for i in range(nlon):
        if cx[j, i] > 0.0:
            im1 = (i - 1) % nlon
            fx[:, j, i] = ar[:, im1] + 0.5 * cx[j, i] * (
                al[:, im1] - ar[:, im1] + a6[:, im1] * (1.0 - r23 * cx[j, i])
            )
        else:
            fx[:, j, i] = al[:, i] - 0.5 * cx[j, i] * (
                ar[:, i] - al[:, i] + a6[:, i] * (1.0 + r23 * cx[j, i])
            )


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


def _ymist_batch(qqu: np.ndarray) -> np.ndarray:
    _ntracer, nlat, nlon = qqu.shape
    dcy = np.zeros_like(qqu)
    r24 = 1.0 / 24.0
    padded = np.zeros((_ntracer, nlat + 4, nlon), dtype=qqu.dtype)
    padded[:, 2 : nlat + 2, :] = qqu
    qjm2 = padded[:, 0:nlat, :]
    qjm1 = padded[:, 1 : nlat + 1, :]
    qj = padded[:, 2 : nlat + 2, :]
    qjp1 = padded[:, 3 : nlat + 3, :]
    qjp2 = padded[:, 4 : nlat + 4, :]
    tmp = (8.0 * (qjp1 - qjm1) + qjm2 - qjp2) * r24
    pmax = np.maximum.reduce([qjm1, qj, qjp1]) - qj
    pmin = qj - np.minimum.reduce([qjm1, qj, qjp1])
    dcy[:] = _sign_array(np.minimum.reduce([np.abs(tmp), pmin, pmax]), tmp)
    dcy[:, 0, :] = 0.0
    dcy[:, -1, :] = 0.0
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


def _fyppm_batch(cy: np.ndarray, dcy: np.ndarray, qqu: np.ndarray, qqv: np.ndarray) -> None:
    _ntracer, nlat, nlon = qqu.shape
    j1p, j2p = _polar_cap_bounds(nlat)
    r13 = 1.0 / 3.0
    r23 = 2.0 / 3.0
    a6 = np.zeros_like(qqu)
    al = np.zeros_like(qqu)
    ar = np.zeros_like(qqu)
    al[:, 1:, :] = 0.5 * (qqu[:, :-1, :] + qqu[:, 1:, :]) + (dcy[:, :-1, :] - dcy[:, 1:, :]) * r13
    ar[:, :-1, :] = al[:, 1:, :]
    half = nlon // 2
    al[:, 0, :half] = al[:, 1, half:]
    al[:, 0, half:] = al[:, 1, :half]
    ar[:, -1, :half] = ar[:, -2, half:]
    ar[:, -1, half:] = ar[:, -2, :half]
    a6[:, 1:-1, :] = 3.0 * (qqu[:, 1:-1, :] + qqu[:, 1:-1, :] - (al[:, 1:-1, :] + ar[:, 1:-1, :]))
    for j in range(1, nlat - 1):
        _lmtppm_last_axis(a6[:, j, :], al[:, j, :], ar[:, j, :], dcy[:, j, :], qqu[:, j, :], 0)
    for j in range(j1p, j2p + 2):
        jm1 = j - 1
        c = cy[j, :]
        pos = c > 0.0
        if np.any(pos):
            cp = c[pos][np.newaxis, :]
            qqv[:, j, pos] = ar[:, jm1, pos] + 0.5 * cp * (
                al[:, jm1, pos] - ar[:, jm1, pos] + a6[:, jm1, pos] * (1.0 - r23 * cp)
            )
        neg = ~pos
        if np.any(neg):
            cn = c[neg][np.newaxis, :]
            qqv[:, j, neg] = al[:, j, neg] - 0.5 * cn * (
                ar[:, j, neg] - al[:, j, neg] + a6[:, j, neg] * (1.0 + r23 * cn)
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


def _fzppm_batch(delp1: np.ndarray, wz: np.ndarray, dq1: np.ndarray, q: np.ndarray) -> None:
    _ntracer, nlev, nlat, nlon = q.shape
    r13 = 1.0 / 3.0
    r23 = 2.0 / 3.0
    dpi = np.zeros_like(q)
    dc = np.zeros_like(q)
    for k in range(nlev - 1):
        dpi[:, k] = q[:, k + 1] - q[:, k]
    for k in range(1, nlev - 1):
        for j in range(nlat):
            for i in range(nlon):
                c0 = delp1[k, j, i] / (delp1[k - 1, j, i] + delp1[k, j, i] + delp1[k + 1, j, i])
                c1 = (delp1[k - 1, j, i] + 0.5 * delp1[k, j, i]) / (delp1[k + 1, j, i] + delp1[k, j, i])
                c2 = (delp1[k + 1, j, i] + 0.5 * delp1[k, j, i]) / (delp1[k - 1, j, i] + delp1[k, j, i])
                tmp = c0 * (c1 * dpi[:, k, j, i] + c2 * dpi[:, k - 1, j, i])
                qmax = np.maximum.reduce([q[:, k - 1, j, i], q[:, k, j, i], q[:, k + 1, j, i]]) - q[:, k, j, i]
                qmin = q[:, k, j, i] - np.minimum.reduce(
                    [q[:, k - 1, j, i], q[:, k, j, i], q[:, k + 1, j, i]]
                )
                dc[:, k, j, i] = _sign_array(np.minimum.reduce([np.abs(tmp), qmax, qmin]), tmp)
    for j in range(nlat):
        if j in (1, nlat - 2):
            continue
        al = np.zeros((_ntracer, nlev, nlon), dtype=np.float64)
        ar = np.zeros((_ntracer, nlev, nlon), dtype=np.float64)
        a6 = np.zeros((_ntracer, nlev, nlon), dtype=np.float64)
        dca = dc[:, :, j, :].copy()
        dlp = delp1[:, j, :]
        qq = q[:, :, j, :]
        wza = wz[:, j, :]
        fac1 = dpi[:, 1, j, :] - dpi[:, 0, j, :] * (dlp[1, :] + dlp[2, :]) / (dlp[0, :] + dlp[1, :])
        fac2 = (dlp[1, :] + dlp[2, :]) * (dlp[0, :] + dlp[1, :] + dlp[2, :])
        aa = 3.0 * fac1 / fac2
        bb = 2.0 * dpi[:, 0, j, :] / (dlp[0, :] + dlp[1, :]) - r23 * aa * (2.0 * dlp[0, :] + dlp[1, :])
        al[:, 0, :] = qq[:, 0, :] - dlp[0, :] * (r13 * aa * dlp[0, :] + 0.5 * bb)
        al[:, 1, :] = dlp[0, :] * (aa * dlp[0, :] + bb) + al[:, 0, :]
        mask = qq[:, 0, :] * al[:, 0, :] <= 0.0
        al[:, 0, :] = np.where(mask, 0.0, al[:, 0, :])
        dca[:, 0, :] = np.where(mask, 0.0, qq[:, 0, :] - al[:, 0, :])
        fac1b = dpi[:, -2, j, :] * (dlp[-1, :] * dlp[-1, :]) / (
            (dlp[-1, :] + dlp[-2, :]) * (2.0 * dlp[-1, :] + dlp[-2, :])
        )
        ar[:, -1, :] = qq[:, -1, :] + fac1b
        al[:, -1, :] = qq[:, -1, :] - (fac1b + fac1b)
        ar[:, -1, :] = np.where(qq[:, -1, :] * ar[:, -1, :] <= 0.0, 0.0, ar[:, -1, :])
        dca[:, -1, :] = ar[:, -1, :] - qq[:, -1, :]
        for k in range(2, nlev - 1):
            c1 = dpi[:, k - 1, j, :] * dlp[k - 1, :] / (dlp[k - 1, :] + dlp[k, :])
            c2 = 2.0 / (dlp[k - 2, :] + dlp[k - 1, :] + dlp[k, :] + dlp[k + 1, :])
            a1 = (dlp[k - 2, :] + dlp[k - 1, :]) / (2.0 * dlp[k - 1, :] + dlp[k, :])
            a2 = (dlp[k, :] + dlp[k + 1, :]) / (2.0 * dlp[k, :] + dlp[k - 1, :])
            al[:, k, :] = qq[:, k - 1, :] + c1 + c2 * (
                dlp[k, :] * (c1 * (a1 - a2) + a2 * dca[:, k - 1, :])
                - dlp[k - 1, :] * a1 * dca[:, k, :]
            )
        ar[:, :-1, :] = al[:, 1:, :]
        for k in (0, 1, nlev - 2, nlev - 1):
            a6[:, k, :] = 3.0 * (qq[:, k, :] + qq[:, k, :] - (al[:, k, :] + ar[:, k, :]))
            _lmtppm_last_axis(a6[:, k, :], al[:, k, :], ar[:, k, :], dca[:, k, :], qq[:, k, :], 0)
        for k in range(1, nlev - 1):
            dca[:, k, :] = dpi[:, k, j, :] - dpi[:, k - 1, j, :]
        for k in range(2, nlev - 2):
            qmp = qq[:, k, :] + 2.0 * dpi[:, k - 1, j, :]
            lac = qq[:, k, :] + 1.5 * dca[:, k - 1, :] + 0.5 * dpi[:, k - 1, j, :]
            qmin = np.minimum.reduce([qq[:, k, :], qmp, lac])
            qmax = np.maximum.reduce([qq[:, k, :], qmp, lac])
            ar[:, k, :] = np.minimum(np.maximum(ar[:, k, :], qmin), qmax)
            qmp = qq[:, k, :] - 2.0 * dpi[:, k, j, :]
            lac = qq[:, k, :] + 1.5 * dca[:, k + 1, :] - 0.5 * dpi[:, k, j, :]
            qmin = np.minimum.reduce([qq[:, k, :], qmp, lac])
            qmax = np.maximum.reduce([qq[:, k, :], qmp, lac])
            al[:, k, :] = np.minimum(np.maximum(al[:, k, :], qmin), qmax)
            a6[:, k, :] = 3.0 * (qq[:, k, :] + qq[:, k, :] - (ar[:, k, :] + al[:, k, :]))
        flux = np.zeros((_ntracer, nlev, nlon), dtype=np.float64)
        for k in range(nlev - 1):
            for i in range(nlon):
                if wza[k, i] > 0.0:
                    cm = wza[k, i] / dlp[k, i]
                    val = ar[:, k, i] + 0.5 * cm * (al[:, k, i] - ar[:, k, i] + a6[:, k, i] * (1.0 - r23 * cm))
                else:
                    cp = wza[k, i] / dlp[k + 1, i]
                    val = al[:, k + 1, i] + 0.5 * cp * (
                        al[:, k + 1, i] - ar[:, k + 1, i] - a6[:, k + 1, i] * (1.0 + r23 * cp)
                    )
                flux[:, k + 1, i] = wza[k, i] * val
        dq1[:, 0, j, :] -= flux[:, 1, :]
        dq1[:, -1, j, :] += flux[:, -1, :]
        for k in range(1, nlev - 1):
            dq1[:, k, j, :] += flux[:, k, :] - flux[:, k + 1, :]


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


def _qckxyz_batch(dq1: np.ndarray) -> None:
    _ntracer, nlev, nlat, nlon = dq1.shape
    j1p, j2p = _polar_cap_bounds(nlat)
    for j in range(j1p, j2p + 1):
        for i in range(nlon):
            mask = dq1[:, 0, j, i] < 0.0
            dq1[mask, 1, j, i] += dq1[mask, 0, j, i]
            dq1[mask, 0, j, i] = 0.0
            for k in range(1, nlev - 1):
                mask = dq1[:, k, j, i] < 0.0
                qup = dq1[mask, k - 1, j, i]
                qly = -dq1[mask, k, j, i]
                dup = np.minimum(qly, qup)
                dq1[mask, k - 1, j, i] = qup - dup
                dq1[mask, k, j, i] = dup - qly
                dq1[mask, k + 1, j, i] += dq1[mask, k, j, i]
                dq1[mask, k, j, i] = 0.0
            mask = dq1[:, -1, j, i] < 0.0
            qup = dq1[mask, -2, j, i]
            qly = -dq1[mask, -1, j, i]
            dup = np.minimum(qly, qup)
            dq1[mask, -2, j, i] = qup - dup
            dq1[mask, -1, j, i] = 0.0


def _do_y_pole_sum(ady: np.ndarray) -> None:
    south = float(np.mean(ady[1, :]))
    north = float(np.mean(ady[-2, :]))
    ady[0, :] = south
    ady[1, :] = south
    ady[-2, :] = north
    ady[-1, :] = north


def _do_y_pole_sum_batch(ady: np.ndarray) -> None:
    south = np.mean(ady[:, 1, :], axis=1)
    north = np.mean(ady[:, -2, :], axis=1)
    ady[:, 0, :] = south[:, np.newaxis]
    ady[:, 1, :] = south[:, np.newaxis]
    ady[:, -2, :] = north[:, np.newaxis]
    ady[:, -1, :] = north[:, np.newaxis]


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


def _lmtppm_last_axis(
    a6: np.ndarray,
    al: np.ndarray,
    ar: np.ndarray,
    dc: np.ndarray,
    qa: np.ndarray,
    lmt: int,
) -> None:
    if lmt != 0:
        raise NotImplementedError("Only the full monotonic PPM limiter is needed for the current TPCORE path")
    for idx in range(qa.shape[-1]):
        a6_col = a6[..., idx]
        al_col = al[..., idx]
        ar_col = ar[..., idx]
        qa_col = qa[..., idx]
        zero_mask = dc[..., idx] == 0.0
        a6_col[zero_mask] = 0.0
        al_col[zero_mask] = qa_col[zero_mask]
        ar_col[zero_mask] = qa_col[zero_mask]

        da1 = ar_col - al_col
        da2 = da1 * da1
        a6da = a6_col * da1
        low_mask = a6da < -da2
        a6_col[low_mask] = 3.0 * (al_col[low_mask] - qa_col[low_mask])
        ar_col[low_mask] = al_col[low_mask] - a6_col[low_mask]
        high_mask = a6da > da2
        a6_col[high_mask] = 3.0 * (ar_col[high_mask] - qa_col[high_mask])
        al_col[high_mask] = ar_col[high_mask] - a6_col[high_mask]


def _q_lon(q: np.ndarray, j: int, i: int) -> float:
    if j < 0 or j >= q.shape[0]:
        return 0.0
    return float(q[j, i % q.shape[1]])


def _dcx_lon(dcx: np.ndarray, j: int, i: int) -> float:
    return float(dcx[j, i % dcx.shape[1]])


def _q_lon_batch(q: np.ndarray, j: int, i: int) -> np.ndarray:
    if j < 0 or j >= q.shape[1]:
        return np.zeros(q.shape[0], dtype=q.dtype)
    return q[:, j, i % q.shape[2]]


def _dcx_lon_batch(dcx: np.ndarray, j: int, i: int) -> np.ndarray:
    return dcx[:, j, i % dcx.shape[2]]


def _q_lat(q: np.ndarray, i: int, j: int) -> float:
    if j < 0 or j >= q.shape[0]:
        return 0.0
    return float(q[j, i])


def _q_lat_batch(q: np.ndarray, i: int, j: int) -> np.ndarray:
    if j < 0 or j >= q.shape[1]:
        return np.zeros(q.shape[0], dtype=q.dtype)
    return q[:, j, i]


def _trunc_toward_zero(value: float) -> int:
    return int(value)


def _real_index_offset(index: int, offset: float) -> int:
    """Return Python index for GEOS-Chem ``INTEGER(real_index - offset)``."""

    return _trunc_toward_zero((index + 1.0) - offset) - 1


def _nint(value: float) -> int:
    return int(np.rint(value))


def _sign(magnitude: float, sign_source: float) -> float:
    return abs(magnitude) if sign_source >= 0.0 else -abs(magnitude)


def _sign_array(magnitude: np.ndarray, sign_source: np.ndarray) -> np.ndarray:
    return np.where(sign_source >= 0.0, np.abs(magnitude), -np.abs(magnitude))
