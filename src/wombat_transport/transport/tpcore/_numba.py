"""Numba-accelerated TPCORE kernels.

This module owns the optional compiled path. The reference implementation and
public API stay in ``_core.py`` so WOMBAT_TPCORE_NUMBA=0 remains a plain NumPy
path.
"""

from __future__ import annotations

import os

import numpy as np

from wombat_transport.transport.tpcore.types import TpcoreSetup

try:  # Optional acceleration dependency.
    from numba import njit
except ImportError:  # pragma: no cover - exercised in environments without numba.
    njit = None


_NUMBA_AVAILABLE = njit is not None
_TPCORE_NUMBA_WORKSPACE = None


class _TpcoreNumbaWorkspace:
    __slots__ = ("shape", "qqu", "qqv", "x_workspace", "y_workspace", "ua", "va", "jn", "js")

    def __init__(self, nlev: int, nlat: int, nlon: int, ntracer: int) -> None:
        self.shape = (nlev, nlat, nlon, ntracer)
        self.qqu = np.empty((nlat, nlon, ntracer), dtype=np.float64)
        self.qqv = np.empty((nlat, nlon, ntracer), dtype=np.float64)
        self.x_workspace = _make_xtp_numba_workspace(nlat, nlon, ntracer)
        self.y_workspace = _make_ytp_numba_workspace(nlat, nlon, ntracer)
        self.ua = np.empty((nlev, nlat, nlon), dtype=np.float64)
        self.va = np.empty((nlev, nlat, nlon), dtype=np.float64)
        self.jn = np.empty(nlev, dtype=np.int64)
        self.js = np.empty(nlev, dtype=np.int64)


def _get_tpcore_numba_workspace(nlev: int, nlat: int, nlon: int, ntracer: int) -> _TpcoreNumbaWorkspace:
    global _TPCORE_NUMBA_WORKSPACE
    shape = (nlev, nlat, nlon, ntracer)
    workspace = _TPCORE_NUMBA_WORKSPACE
    if workspace is None or workspace.shape != shape:
        workspace = _TpcoreNumbaWorkspace(nlev, nlat, nlon, ntracer)
        _TPCORE_NUMBA_WORKSPACE = workspace
    return workspace

def _xtp_batch_numba(
    dq1: np.ndarray,
    qqv: np.ndarray,
    pu: np.ndarray,
    cx: np.ndarray,
    xmass: np.ndarray,
    jn: int,
    js: int,
    workspace: tuple[np.ndarray, ...],
) -> None:
    if not _NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available")
    _xtp_batch_numba_kernel(dq1, qqv, pu, cx, xmass, jn, js, *workspace)


def _ytp_batch_numba(
    dq1: np.ndarray,
    qqu: np.ndarray,
    qqv: np.ndarray,
    cy: np.ndarray,
    ymass: np.ndarray,
    geofac: np.ndarray,
    geofac_pc: float,
    workspace: tuple[np.ndarray, ...],
) -> None:
    if not _NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available")
    _ytp_batch_numba_kernel(dq1, qqu, qqv, cy, ymass, geofac, geofac_pc, *workspace)


def _calc_advec_cross_terms_batch_numba(
    q: np.ndarray,
    ua: np.ndarray,
    va: np.ndarray,
    jn: int,
    js: int,
    qqu: np.ndarray,
    qqv: np.ndarray,
) -> None:
    if not _NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available")
    _calc_advec_cross_terms_batch_numba_kernel(q, ua, va, jn, js, qqu, qqv)


def _xadv_dao2_batch_numba(qqv: np.ndarray, ua: np.ndarray, jn: int, js: int, adx: np.ndarray) -> None:
    if not _NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available")
    _xadv_dao2_batch_numba_kernel(qqv, ua, jn, js, adx)


def _yadv_dao2_batch_numba(qqu: np.ndarray, va: np.ndarray, ady: np.ndarray) -> None:
    if not _NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available")
    _yadv_dao2_batch_numba_kernel(qqu, va, ady)


def _make_tpcore_prepass_numba_workspace(nlat: int, nlon: int, ntracer: int) -> tuple[np.ndarray, ...]:
    return (
        np.empty((nlat, nlon, ntracer), dtype=np.float64),
        np.empty((nlat, nlon, ntracer), dtype=np.float64),
        np.empty((nlat, nlon, ntracer), dtype=np.float64),
        np.empty((nlat, nlon, ntracer), dtype=np.float64),
    )


def _make_xtp_numba_workspace(nlat: int, nlon: int, ntracer: int) -> tuple[np.ndarray, ...]:
    return (
        np.empty((nlon, ntracer), dtype=np.float64),
        np.empty((nlon, ntracer), dtype=np.float64),
        np.empty((nlon, ntracer), dtype=np.float64),
        np.empty((nlon, ntracer), dtype=np.float64),
        np.empty((nlon, ntracer), dtype=np.float64),
        np.empty((nlon, ntracer), dtype=np.float64),
        np.empty((nlon, ntracer), dtype=np.float64),
    )


def _make_ytp_numba_workspace(nlat: int, nlon: int, ntracer: int) -> tuple[np.ndarray, ...]:
    return (
        np.empty((nlat, ntracer), dtype=np.float64),
        np.empty((nlat, ntracer), dtype=np.float64),
        np.empty((nlat, ntracer), dtype=np.float64),
        np.empty((nlat, ntracer), dtype=np.float64),
    )


def _advect_tracers_fused_numba(
    *,
    tracer_conc: np.ndarray,
    setup: TpcoreSetup,
    area_m2: np.ndarray,
    fill: bool,
) -> np.ndarray:
    if not _NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available")
    nlev, nlat, nlon, ntracer = tracer_conc.shape
    q = np.ascontiguousarray(tracer_conc).copy()
    dq1 = np.empty_like(q)
    workspace = _get_tpcore_numba_workspace(nlev, nlat, nlon, ntracer)
    _set_cross_terms_numba_kernel(setup.cx, setup.cy, workspace.ua, workspace.va)
    _set_jn_js_numba_kernel(setup.cx, workspace.jn, workspace.js)
    _advect_tracers_fused_numba_kernel(
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
        workspace.ua,
        workspace.va,
        workspace.jn,
        workspace.js,
        area_m2[:, 0],
        bool(fill),
        workspace.qqu,
        workspace.qqv,
        *workspace.x_workspace,
        *workspace.y_workspace,
    )
    return dq1


def _numba_tpcore_mode() -> str:
    return os.environ.get("WOMBAT_TPCORE_NUMBA", "1").lower()


def _numba_tpcore_enabled() -> bool:
    if not _NUMBA_AVAILABLE:
        return False
    return _numba_tpcore_mode() not in {"0", "false", "no", "off", "none"}


def _numba_tpcore_z_enabled() -> bool:
    return _numba_tpcore_enabled()


def _numba_tpcore_x_enabled() -> bool:
    return _numba_tpcore_enabled()


def _numba_tpcore_y_enabled() -> bool:
    return _numba_tpcore_enabled()


def _numba_tpcore_prepass_enabled() -> bool:
    return _numba_tpcore_enabled()


def _finalize_tpcore_output_numba(dq1: np.ndarray, delp2: np.ndarray) -> None:
    if not _NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available")
    _finalize_tpcore_output_numba_kernel(dq1, delp2)


def _fzppm_batch_numba(delp1: np.ndarray, wz: np.ndarray, dq1: np.ndarray, q: np.ndarray) -> None:
    if not _NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available")
    _fzppm_batch_numba_kernel(delp1, wz, dq1, q)


if njit is not None:

    @njit(cache=True)
    def _set_cross_terms_numba_kernel(cx: np.ndarray, cy: np.ndarray, ua: np.ndarray, va: np.ndarray) -> None:
        nlev = cx.shape[0]
        nlat = cx.shape[1]
        nlon = cx.shape[2]
        j1p = 2
        j2p = nlat - 3
        for level in range(nlev):
            for j in range(nlat):
                for i in range(nlon):
                    ua[level, j, i] = 0.0
                    va[level, j, i] = 0.0
            for j in range(j1p, j2p + 1):
                for i in range(nlon - 1):
                    ua[level, j, i] = 0.5 * (cx[level, j, i] + cx[level, j, i + 1])
                ua[level, j, nlon - 1] = 0.5 * (cx[level, j, nlon - 1] + cx[level, j, 0])
            for j in range(1, nlat - 1):
                for i in range(nlon):
                    va[level, j, i] = 0.5 * (cy[level, j, i] + cy[level, j + 1, i])


    @njit(cache=True)
    def _set_jn_js_numba_kernel(cx: np.ndarray, jn: np.ndarray, js: np.ndarray) -> None:
        nlev = cx.shape[0]
        nlat = cx.shape[1]
        nlon = cx.shape[2]
        j1p = 2
        j2p = nlat - 3
        js0 = (nlat + 1) // 2 - 1
        jn0 = nlat - (js0 + 1)
        for level in range(nlev):
            js_value = j1p
            for j in range(min(nlat - 1, js0), max(0, j1p) - 1, -1):
                found = False
                for i in range(nlon):
                    if abs(cx[level, j, i]) > 1.0:
                        found = True
                        break
                if found:
                    js_value = j
                    break
            jn_value = j2p
            for j in range(max(0, jn0), min(nlat - 1, j2p) + 1):
                found = False
                for i in range(nlon):
                    if abs(cx[level, j, i]) > 1.0:
                        found = True
                        break
                if found:
                    jn_value = j
                    break
            js[level] = js_value
            jn[level] = jn_value

    @njit(cache=True)
    def _average_const_poles_batch_numba_kernel(q: np.ndarray, delp1: np.ndarray, area_1d: np.ndarray) -> None:
        nlat = q.shape[0]
        nlon = q.shape[1]
        ntracer = q.shape[2]

        south_denom = 0.0
        north_denom = 0.0
        for j in range(2):
            area = area_1d[j]
            for i in range(nlon):
                south_denom += delp1[j, i] * area
        for j in range(nlat - 2, nlat):
            area = area_1d[j]
            for i in range(nlon):
                north_denom += delp1[j, i] * area

        for tracer in range(ntracer):
            south = 0.0
            north = 0.0
            for j in range(2):
                area = area_1d[j]
                for i in range(nlon):
                    south += q[j, i, tracer] * delp1[j, i] * area
            for j in range(nlat - 2, nlat):
                area = area_1d[j]
                for i in range(nlon):
                    north += q[j, i, tracer] * delp1[j, i] * area
            south /= south_denom
            north /= north_denom
            for j in range(2):
                for i in range(nlon):
                    q[j, i, tracer] = south
            for j in range(nlat - 2, nlat):
                for i in range(nlon):
                    q[j, i, tracer] = north


    @njit(cache=True)
    def _qckxyz_batch_numba_kernel(dq1: np.ndarray) -> None:
        nlev = dq1.shape[0]
        nlat = dq1.shape[1]
        nlon = dq1.shape[2]
        ntracer = dq1.shape[3]
        j1p = 2
        j2p = nlat - 3
        for j in range(j1p, j2p + 1):
            for i in range(nlon):
                for tracer in range(ntracer):
                    if dq1[0, j, i, tracer] < 0.0:
                        dq1[1, j, i, tracer] += dq1[0, j, i, tracer]
                        dq1[0, j, i, tracer] = 0.0
                    for k in range(1, nlev - 1):
                        if dq1[k, j, i, tracer] < 0.0:
                            qup = dq1[k - 1, j, i, tracer]
                            qly = -dq1[k, j, i, tracer]
                            dup = min(qly, qup)
                            dq1[k - 1, j, i, tracer] = qup - dup
                            dq1[k, j, i, tracer] = dup - qly
                            dq1[k + 1, j, i, tracer] += dq1[k, j, i, tracer]
                            dq1[k, j, i, tracer] = 0.0
                    if dq1[nlev - 1, j, i, tracer] < 0.0:
                        qup = dq1[nlev - 2, j, i, tracer]
                        qly = -dq1[nlev - 1, j, i, tracer]
                        dup = min(qly, qup)
                        dq1[nlev - 2, j, i, tracer] = qup - dup
                        dq1[nlev - 1, j, i, tracer] = 0.0


    @njit(cache=True)
    def _qckxyz_needs_fill_numba_kernel(dq1: np.ndarray) -> bool:
        nlev = dq1.shape[0]
        nlat = dq1.shape[1]
        nlon = dq1.shape[2]
        ntracer = dq1.shape[3]
        j1p = 2
        j2p = nlat - 3
        for k in range(nlev):
            for j in range(j1p, j2p + 1):
                for i in range(nlon):
                    for tracer in range(ntracer):
                        if dq1[k, j, i, tracer] < 0.0:
                            return True
        return False


    @njit(cache=True)
    def _finalize_tpcore_output_numba_kernel(dq1: np.ndarray, delp2: np.ndarray) -> None:
        nlev = dq1.shape[0]
        nlat = dq1.shape[1]
        nlon = dq1.shape[2]
        ntracer = dq1.shape[3]

        for lev in range(nlev):
            for lat in range(nlat):
                if lat == 1 or lat == nlat - 2:
                    continue
                for lon in range(nlon):
                    inv_delp = 1.0 / delp2[lev, lat, lon]
                    for tracer in range(ntracer):
                        value = dq1[lev, lat, lon, tracer] * inv_delp
                        if value < 0.0:
                            value = 1.0e-26
                        dq1[lev, lat, lon, tracer] = value

            for lon in range(nlon):
                for tracer in range(ntracer):
                    dq1[lev, 1, lon, tracer] = dq1[lev, 0, lon, tracer]
                    dq1[lev, nlat - 2, lon, tracer] = dq1[lev, nlat - 1, lon, tracer]

    @njit(cache=True)
    def _advect_tracers_fused_numba_kernel(
        q: np.ndarray,
        dq1: np.ndarray,
        delp1: np.ndarray,
        delp2: np.ndarray,
        pu: np.ndarray,
        xmass: np.ndarray,
        ymass: np.ndarray,
        wz: np.ndarray,
        cx: np.ndarray,
        cy: np.ndarray,
        geofac: np.ndarray,
        geofac_pc: float,
        ua: np.ndarray,
        va: np.ndarray,
        jn: np.ndarray,
        js: np.ndarray,
        area_1d: np.ndarray,
        fill: bool,
        qqu: np.ndarray,
        qqv: np.ndarray,
        dcx: np.ndarray,
        fx: np.ndarray,
        al_x: np.ndarray,
        ar_x: np.ndarray,
        a6_x: np.ndarray,
        dc_x: np.ndarray,
        qa_x: np.ndarray,
        dcy: np.ndarray,
        al_y: np.ndarray,
        ar_y: np.ndarray,
        a6_y: np.ndarray,
    ) -> None:
        nlev = q.shape[0]
        nlat = q.shape[1]
        nlon = q.shape[2]
        ntracer = q.shape[3]

        for level in range(nlev):
            _average_const_poles_batch_numba_kernel(q[level], delp1[level], area_1d)
            for j in range(nlat):
                for i in range(nlon):
                    mass = delp1[level, j, i]
                    for tracer in range(ntracer):
                        dq1[level, j, i, tracer] = q[level, j, i, tracer] * mass

            _calc_advec_cross_terms_batch_numba_kernel(
                q[level],
                ua[level],
                va[level],
                int(jn[level]),
                int(js[level]),
                qqu,
                qqv,
            )
            _xadv_dao2_apply_batch_numba_kernel(q[level], qqv, ua[level], int(jn[level]), int(js[level]))
            _yadv_dao2_apply_batch_numba_kernel(q[level], qqu, va[level])

            _xtp_batch_numba_kernel(
                dq1[level],
                qqv,
                pu[level],
                cx[level],
                xmass[level],
                int(jn[level]),
                int(js[level]),
                dcx,
                fx,
                al_x,
                ar_x,
                a6_x,
                dc_x,
                qa_x,
            )
            _ytp_batch_numba_kernel(
                dq1[level],
                qqu,
                qqv,
                cy[level],
                ymass[level],
                geofac,
                geofac_pc,
                dcy,
                al_y,
                ar_y,
                a6_y,
            )

        _fzppm_batch_numba_kernel(delp1, wz, dq1, q)
        if fill:
            if _qckxyz_needs_fill_numba_kernel(dq1):
                _qckxyz_batch_numba_kernel(dq1)
        _finalize_tpcore_output_numba_kernel(dq1, delp2)


    @njit(cache=True)
    def _calc_advec_cross_terms_batch_numba_kernel(
        q: np.ndarray,
        ua: np.ndarray,
        va: np.ndarray,
        jn: int,
        js: int,
        qqu: np.ndarray,
        qqv: np.ndarray,
    ) -> None:
        nlat = q.shape[0]
        nlon = q.shape[1]
        ntracer = q.shape[2]
        j1p = 2
        j2p = nlat - 3

        for j in range(j1p):
            for i in range(nlon):
                for tracer in range(ntracer):
                    qqu[j, i, tracer] = q[j, i, tracer]
                    qqv[j, i, tracer] = q[j, i, tracer]
        for j in range(j2p + 1, nlat):
            for i in range(nlon):
                for tracer in range(ntracer):
                    qqu[j, i, tracer] = q[j, i, tracer]
                    qqv[j, i, tracer] = q[j, i, tracer]

        for j in range(j1p, j2p + 1):
            if j <= js or j >= jn:
                for i in range(nlon):
                    ua_value = ua[j, i]
                    va_value = va[j, i]
                    iu0 = int(ua_value)
                    ru = ua_value - float(iu0)
                    iu = i - iu0
                    iu_mod = iu % nlon
                    im1 = (iu - 1) % nlon
                    ip1 = (iu + 1) % nlon
                    jv = int((j + 1.0) - va_value) - 1
                    jvp1 = jv + 1
                    coeff_v = va_value
                    left_valid = jv >= 0 and jv < nlat
                    right_valid = jvp1 >= 0 and jvp1 < nlat
                    if ua_value >= 0.0:
                        if left_valid and right_valid:
                            for tracer in range(ntracer):
                                q_center = q[j, i, tracer]
                                q_i = q[j, iu_mod, tracer]
                                delta = q_i + ru * (q[j, im1, tracer] - q_i) - q_center
                                qqu[j, i, tracer] = q_center + 0.5 * delta
                                qqv[j, i, tracer] = q_center + 0.5 * (coeff_v * (q[jv, i, tracer] - q[jvp1, i, tracer]))
                        elif left_valid:
                            for tracer in range(ntracer):
                                q_center = q[j, i, tracer]
                                q_i = q[j, iu_mod, tracer]
                                delta = q_i + ru * (q[j, im1, tracer] - q_i) - q_center
                                qqu[j, i, tracer] = q_center + 0.5 * delta
                                qqv[j, i, tracer] = q_center + 0.5 * (coeff_v * q[jv, i, tracer])
                        elif right_valid:
                            for tracer in range(ntracer):
                                q_center = q[j, i, tracer]
                                q_i = q[j, iu_mod, tracer]
                                delta = q_i + ru * (q[j, im1, tracer] - q_i) - q_center
                                qqu[j, i, tracer] = q_center + 0.5 * delta
                                qqv[j, i, tracer] = q_center + 0.5 * (coeff_v * (0.0 - q[jvp1, i, tracer]))
                        else:
                            for tracer in range(ntracer):
                                q_center = q[j, i, tracer]
                                q_i = q[j, iu_mod, tracer]
                                delta = q_i + ru * (q[j, im1, tracer] - q_i) - q_center
                                qqu[j, i, tracer] = q_center + 0.5 * delta
                                qqv[j, i, tracer] = q_center
                    else:
                        if left_valid and right_valid:
                            for tracer in range(ntracer):
                                q_center = q[j, i, tracer]
                                q_i = q[j, iu_mod, tracer]
                                delta = q_i + ru * (q_i - q[j, ip1, tracer]) - q_center
                                qqu[j, i, tracer] = q_center + 0.5 * delta
                                qqv[j, i, tracer] = q_center + 0.5 * (coeff_v * (q[jv, i, tracer] - q[jvp1, i, tracer]))
                        elif left_valid:
                            for tracer in range(ntracer):
                                q_center = q[j, i, tracer]
                                q_i = q[j, iu_mod, tracer]
                                delta = q_i + ru * (q_i - q[j, ip1, tracer]) - q_center
                                qqu[j, i, tracer] = q_center + 0.5 * delta
                                qqv[j, i, tracer] = q_center + 0.5 * (coeff_v * q[jv, i, tracer])
                        elif right_valid:
                            for tracer in range(ntracer):
                                q_center = q[j, i, tracer]
                                q_i = q[j, iu_mod, tracer]
                                delta = q_i + ru * (q_i - q[j, ip1, tracer]) - q_center
                                qqu[j, i, tracer] = q_center + 0.5 * delta
                                qqv[j, i, tracer] = q_center + 0.5 * (coeff_v * (0.0 - q[jvp1, i, tracer]))
                        else:
                            for tracer in range(ntracer):
                                q_center = q[j, i, tracer]
                                q_i = q[j, iu_mod, tracer]
                                delta = q_i + ru * (q_i - q[j, ip1, tracer]) - q_center
                                qqu[j, i, tracer] = q_center + 0.5 * delta
                                qqv[j, i, tracer] = q_center
            else:
                for i in range(nlon):
                    ua_value = ua[j, i]
                    va_value = va[j, i]
                    iu = int((i + 1.0) - ua_value) - 1
                    iu_mod = iu % nlon
                    ip1 = (iu + 1) % nlon
                    coeff = ua_value
                    jv = int((j + 1.0) - va_value) - 1
                    jvp1 = jv + 1
                    coeff_v = va_value
                    left_valid = jv >= 0 and jv < nlat
                    right_valid = jvp1 >= 0 and jvp1 < nlat
                    if left_valid and right_valid:
                        for tracer in range(ntracer):
                            q_center = q[j, i, tracer]
                            delta = coeff * (q[j, iu_mod, tracer] - q[j, ip1, tracer])
                            qqu[j, i, tracer] = q_center + 0.5 * delta
                            qqv[j, i, tracer] = q_center + 0.5 * (coeff_v * (q[jv, i, tracer] - q[jvp1, i, tracer]))
                    elif left_valid:
                        for tracer in range(ntracer):
                            q_center = q[j, i, tracer]
                            delta = coeff * (q[j, iu_mod, tracer] - q[j, ip1, tracer])
                            qqu[j, i, tracer] = q_center + 0.5 * delta
                            qqv[j, i, tracer] = q_center + 0.5 * (coeff_v * q[jv, i, tracer])
                    elif right_valid:
                        for tracer in range(ntracer):
                            q_center = q[j, i, tracer]
                            delta = coeff * (q[j, iu_mod, tracer] - q[j, ip1, tracer])
                            qqu[j, i, tracer] = q_center + 0.5 * delta
                            qqv[j, i, tracer] = q_center + 0.5 * (coeff_v * (0.0 - q[jvp1, i, tracer]))
                    else:
                        for tracer in range(ntracer):
                            q_center = q[j, i, tracer]
                            delta = coeff * (q[j, iu_mod, tracer] - q[j, ip1, tracer])
                            qqu[j, i, tracer] = q_center + 0.5 * delta
                            qqv[j, i, tracer] = q_center


    @njit(cache=True)
    def _xadv_dao2_batch_numba_kernel(
        qqv: np.ndarray,
        ua: np.ndarray,
        jn: int,
        js: int,
        adx: np.ndarray,
    ) -> None:
        nlat = qqv.shape[0]
        nlon = qqv.shape[1]
        ntracer = qqv.shape[2]
        j1p = 2
        j2p = nlat - 3

        for j in range(nlat):
            for i in range(nlon):
                for tracer in range(ntracer):
                    adx[j, i, tracer] = 0.0

        for j in range(j1p, j2p + 1):
            for i in range(nlon):
                iu0 = int(np.rint(ua[j, i]))
                ru = float(iu0) - ua[j, i]
                iu = i - iu0
                im1 = (iu - 1) % nlon
                iu_mod = iu % nlon
                ip1 = (iu + 1) % nlon
                for tracer in range(ntracer):
                    q_i = qqv[j, iu_mod, tracer]
                    q_ip1 = qqv[j, ip1, tracer]
                    q_im1 = qqv[j, im1, tracer]
                    a1 = 0.5 * (q_ip1 + q_im1) - q_i
                    b1 = 0.5 * (q_ip1 - q_im1)
                    c1 = q_i - qqv[j, i, tracer]
                    adx[j, i, tracer] = ru * (a1 * ru + b1) + c1


    @njit(cache=True)
    def _yadv_dao2_batch_numba_kernel(qqu: np.ndarray, va: np.ndarray, ady: np.ndarray) -> None:
        nlat = qqu.shape[0]
        nlon = qqu.shape[1]
        ntracer = qqu.shape[2]
        j1p = 2
        j2p = nlat - 3

        for j in range(nlat):
            for i in range(nlon):
                for tracer in range(ntracer):
                    ady[j, i, tracer] = 0.0

        for j in range(j1p - 1, j2p + 2):
            for i in range(nlon):
                jv0 = int(np.rint(va[j, i]))
                rv = float(jv0) - va[j, i]
                jv = j - jv0
                jm1 = jv - 1
                jp1 = jv + 1
                q_j_valid = jv >= 0 and jv < nlat
                q_jm1_valid = jm1 >= 0 and jm1 < nlat
                q_jp1_valid = jp1 >= 0 and jp1 < nlat
                for tracer in range(ntracer):
                    q_j = qqu[jv, i, tracer] if q_j_valid else 0.0
                    q_jp1 = qqu[jp1, i, tracer] if q_jp1_valid else 0.0
                    q_jm1 = qqu[jm1, i, tracer] if q_jm1_valid else 0.0
                    a1 = 0.5 * (q_jp1 + q_jm1) - q_j
                    b1 = 0.5 * (q_jp1 - q_jm1)
                    c1 = q_j - qqu[j, i, tracer]
                    ady[j, i, tracer] = rv * (a1 * rv + b1) + c1

        for tracer in range(ntracer):
            south = 0.0
            north = 0.0
            for i in range(nlon):
                south += ady[1, i, tracer]
                north += ady[nlat - 2, i, tracer]
            south /= float(nlon)
            north /= float(nlon)
            for i in range(nlon):
                ady[0, i, tracer] = south
                ady[1, i, tracer] = south
                ady[nlat - 2, i, tracer] = north
                ady[nlat - 1, i, tracer] = north


    @njit(cache=True)
    def _xadv_dao2_apply_batch_numba_kernel(
        q: np.ndarray,
        qqv: np.ndarray,
        ua: np.ndarray,
        jn: int,
        js: int,
    ) -> None:
        nlat = qqv.shape[0]
        nlon = qqv.shape[1]
        ntracer = qqv.shape[2]
        j1p = 2
        j2p = nlat - 3

        for j in range(j1p, j2p + 1):
            for i in range(nlon):
                iu0 = int(np.rint(ua[j, i]))
                ru = float(iu0) - ua[j, i]
                iu = i - iu0
                im1 = (iu - 1) % nlon
                iu_mod = iu % nlon
                ip1 = (iu + 1) % nlon
                for tracer in range(ntracer):
                    q_i = qqv[j, iu_mod, tracer]
                    q_ip1 = qqv[j, ip1, tracer]
                    q_im1 = qqv[j, im1, tracer]
                    a1 = 0.5 * (q_ip1 + q_im1) - q_i
                    b1 = 0.5 * (q_ip1 - q_im1)
                    c1 = q_i - qqv[j, i, tracer]
                    q[j, i, tracer] += ru * (a1 * ru + b1) + c1


    @njit(cache=True)
    def _yadv_dao2_apply_batch_numba_kernel(q: np.ndarray, qqu: np.ndarray, va: np.ndarray) -> None:
        nlat = qqu.shape[0]
        nlon = qqu.shape[1]
        ntracer = qqu.shape[2]
        j1p = 2
        j2p = nlat - 3

        for j in range(j1p, j2p + 1):
            for i in range(nlon):
                jv0 = int(np.rint(va[j, i]))
                rv = float(jv0) - va[j, i]
                jv = j - jv0
                jm1 = jv - 1
                jp1 = jv + 1
                q_j_valid = jv >= 0 and jv < nlat
                q_jm1_valid = jm1 >= 0 and jm1 < nlat
                q_jp1_valid = jp1 >= 0 and jp1 < nlat
                for tracer in range(ntracer):
                    q_j = qqu[jv, i, tracer] if q_j_valid else 0.0
                    q_jp1 = qqu[jp1, i, tracer] if q_jp1_valid else 0.0
                    q_jm1 = qqu[jm1, i, tracer] if q_jm1_valid else 0.0
                    a1 = 0.5 * (q_jp1 + q_jm1) - q_j
                    b1 = 0.5 * (q_jp1 - q_jm1)
                    c1 = q_j - qqu[j, i, tracer]
                    q[j, i, tracer] += rv * (a1 * rv + b1) + c1

        for tracer in range(ntracer):
            south = 0.0
            north = 0.0
            for i in range(nlon):
                j = 1
                jv0 = int(np.rint(va[j, i]))
                rv = float(jv0) - va[j, i]
                jv = j - jv0
                jm1 = jv - 1
                jp1 = jv + 1
                q_j = qqu[jv, i, tracer] if jv >= 0 and jv < nlat else 0.0
                q_jp1 = qqu[jp1, i, tracer] if jp1 >= 0 and jp1 < nlat else 0.0
                q_jm1 = qqu[jm1, i, tracer] if jm1 >= 0 and jm1 < nlat else 0.0
                a1 = 0.5 * (q_jp1 + q_jm1) - q_j
                b1 = 0.5 * (q_jp1 - q_jm1)
                c1 = q_j - qqu[j, i, tracer]
                south += rv * (a1 * rv + b1) + c1

                j = nlat - 2
                jv0 = int(np.rint(va[j, i]))
                rv = float(jv0) - va[j, i]
                jv = j - jv0
                jm1 = jv - 1
                jp1 = jv + 1
                q_j = qqu[jv, i, tracer] if jv >= 0 and jv < nlat else 0.0
                q_jp1 = qqu[jp1, i, tracer] if jp1 >= 0 and jp1 < nlat else 0.0
                q_jm1 = qqu[jm1, i, tracer] if jm1 >= 0 and jm1 < nlat else 0.0
                a1 = 0.5 * (q_jp1 + q_jm1) - q_j
                b1 = 0.5 * (q_jp1 - q_jm1)
                c1 = q_j - qqu[j, i, tracer]
                north += rv * (a1 * rv + b1) + c1
            south /= float(nlon)
            north /= float(nlon)
            for i in range(nlon):
                q[0, i, tracer] += south
                q[1, i, tracer] += south
                q[nlat - 2, i, tracer] += north
                q[nlat - 1, i, tracer] += north


    @njit(cache=True)
    def _xtp_batch_numba_kernel(
        dq1: np.ndarray,
        qqv: np.ndarray,
        pu: np.ndarray,
        cx: np.ndarray,
        xmass: np.ndarray,
        jn: int,
        js: int,
        dcx: np.ndarray,
        fx: np.ndarray,
        al: np.ndarray,
        ar: np.ndarray,
        a6: np.ndarray,
        dc: np.ndarray,
        qa: np.ndarray,
    ) -> None:
        nlat = dq1.shape[0]
        nlon = dq1.shape[1]
        ntracer = dq1.shape[2]
        j1p = 2
        j2p = nlat - 3
        jvan = max(1, nlat // 18)
        r13 = 1.0 / 3.0
        r23 = 2.0 / 3.0
        r24 = 1.0 / 24.0

        for j in range(j1p, j2p + 1):
            if j > j1p and j < j2p:
                for i in range(nlon):
                    im1 = (i - 1) % nlon
                    ip1 = (i + 1) % nlon
                    im2 = (i - 2) % nlon
                    ip2 = (i + 2) % nlon
                    for tracer in range(ntracer):
                        q_im1 = qqv[j, im1, tracer]
                        q_i = qqv[j, i, tracer]
                        q_ip1 = qqv[j, ip1, tracer]
                        tmp = (8.0 * (q_ip1 - q_im1) + qqv[j, im2, tracer] - qqv[j, ip2, tracer]) * r24
                        pmax = max(q_im1, q_i, q_ip1) - q_i
                        pmin = q_i - min(q_im1, q_i, q_ip1)
                        bounded = min(abs(tmp), pmin, pmax)
                        dcx[i, tracer] = bounded if tmp >= 0.0 else -bounded

            if j > js and j < jn:
                if j == j1p or j == j2p:
                    for i in range(nlon):
                        iu = (int((i + 1.0) - cx[j, i]) - 1) % nlon
                        for tracer in range(ntracer):
                            fx[i, tracer] = qqv[j, iu, tracer]
                elif j <= j1p + jvan or j >= j2p - jvan:
                    for i in range(nlon):
                        iu = (int((i + 1.0) - cx[j, i]) - 1) % nlon
                        sign_value = 1.0 if cx[j, i] >= 0.0 else -1.0
                        for tracer in range(ntracer):
                            fx[i, tracer] = qqv[j, iu, tracer] + dcx[iu, tracer] * (sign_value - cx[j, i])
                else:
                    for i in range(nlon):
                        im1 = (i - 1) % nlon
                        for tracer in range(ntracer):
                            rval = 0.5 * (qqv[j, im1, tracer] + qqv[j, i, tracer])
                            rval += (dcx[im1, tracer] - dcx[i, tracer]) * r13
                            al[i, tracer] = rval
                            ar[im1, tracer] = rval
                            dc[i, tracer] = dcx[i, tracer]
                            qa[i, tracer] = qqv[j, i, tracer]
                    for i in range(nlon):
                        for tracer in range(ntracer):
                            a6[i, tracer] = 3.0 * (qa[i, tracer] + qa[i, tracer] - (al[i, tracer] + ar[i, tracer]))
                            if dc[i, tracer] == 0.0:
                                a6[i, tracer] = 0.0
                                al[i, tracer] = qa[i, tracer]
                                ar[i, tracer] = qa[i, tracer]
                            else:
                                da1 = ar[i, tracer] - al[i, tracer]
                                da2 = da1 * da1
                                a6da = a6[i, tracer] * da1
                                if a6da < -da2:
                                    a6[i, tracer] = 3.0 * (al[i, tracer] - qa[i, tracer])
                                    ar[i, tracer] = al[i, tracer] - a6[i, tracer]
                                elif a6da > da2:
                                    a6[i, tracer] = 3.0 * (ar[i, tracer] - qa[i, tracer])
                                    al[i, tracer] = ar[i, tracer] - a6[i, tracer]
                    for i in range(nlon):
                        c = cx[j, i]
                        if c > 0.0:
                            im1 = (i - 1) % nlon
                            for tracer in range(ntracer):
                                fx[i, tracer] = ar[im1, tracer] + 0.5 * c * (
                                    al[im1, tracer]
                                    - ar[im1, tracer]
                                    + a6[im1, tracer] * (1.0 - r23 * c)
                                )
                        else:
                            for tracer in range(ntracer):
                                fx[i, tracer] = al[i, tracer] - 0.5 * c * (
                                    ar[i, tracer] - al[i, tracer] + a6[i, tracer] * (1.0 + r23 * c)
                                )
                for i in range(nlon):
                    for tracer in range(ntracer):
                        fx[i, tracer] *= xmass[j, i]
            else:
                for i in range(nlon):
                    ic = int(cx[j, i])
                    isav = i - ic
                    iu_mod = (int((i + 1.0) - cx[j, i]) - 1) % nlon
                    rc = cx[j, i] - float(ic)
                    sign_value = 1.0 if rc >= 0.0 else -1.0
                    if j == j1p or j == j2p:
                        if cx[j, i] > 1.0:
                            for tracer in range(ntracer):
                                val = rc * qqv[j, iu_mod, tracer]
                                for ix in range(isav, i):
                                    val += qqv[j, ix % nlon, tracer]
                                fx[i, tracer] = pu[j, i] * val
                        elif cx[j, i] < -1.0:
                            for tracer in range(ntracer):
                                val = rc * qqv[j, iu_mod, tracer]
                                for ix in range(i, isav):
                                    val -= qqv[j, ix % nlon, tracer]
                                fx[i, tracer] = pu[j, i] * val
                        else:
                            for tracer in range(ntracer):
                                fx[i, tracer] = pu[j, i] * (rc * qqv[j, iu_mod, tracer])
                    else:
                        if cx[j, i] > 1.0:
                            for tracer in range(ntracer):
                                val = rc * (qqv[j, iu_mod, tracer] + dcx[iu_mod, tracer] * (sign_value - rc))
                                for ix in range(isav, i):
                                    val += qqv[j, ix % nlon, tracer]
                                fx[i, tracer] = pu[j, i] * val
                        elif cx[j, i] < -1.0:
                            for tracer in range(ntracer):
                                val = rc * (qqv[j, iu_mod, tracer] + dcx[iu_mod, tracer] * (sign_value - rc))
                                for ix in range(i, isav):
                                    val -= qqv[j, ix % nlon, tracer]
                                fx[i, tracer] = pu[j, i] * val
                        else:
                            for tracer in range(ntracer):
                                fx[i, tracer] = pu[j, i] * (
                                    rc * (qqv[j, iu_mod, tracer] + dcx[iu_mod, tracer] * (sign_value - rc))
                                )

            for i in range(nlon - 1):
                for tracer in range(ntracer):
                    dq1[j, i, tracer] += fx[i, tracer] - fx[i + 1, tracer]
            for tracer in range(ntracer):
                dq1[j, nlon - 1, tracer] += fx[nlon - 1, tracer] - fx[0, tracer]


    @njit(cache=True)
    def _ytp_batch_numba_kernel(
        dq1: np.ndarray,
        qqu: np.ndarray,
        qqv: np.ndarray,
        cy: np.ndarray,
        ymass: np.ndarray,
        geofac: np.ndarray,
        geofac_pc: float,
        dcy: np.ndarray,
        al: np.ndarray,
        ar: np.ndarray,
        a6: np.ndarray,
    ) -> None:
        nlat = dq1.shape[0]
        nlon = dq1.shape[1]
        ntracer = dq1.shape[2]
        j1p = 2
        j2p = nlat - 3
        r13 = 1.0 / 3.0
        r23 = 2.0 / 3.0
        r24 = 1.0 / 24.0

        for tracer in range(ntracer):
            al[0, tracer] = 0.0
            ar[nlat - 1, tracer] = 0.0

        for i in range(nlon):
            for tracer in range(ntracer):
                dcy[0, tracer] = 0.0
                dcy[nlat - 1, tracer] = 0.0

            for j in range(1, nlat - 1):
                for tracer in range(ntracer):
                    qjm2 = 0.0 if j < 2 else qqu[j - 2, i, tracer]
                    qjm1 = qqu[j - 1, i, tracer]
                    qj = qqu[j, i, tracer]
                    qjp1 = qqu[j + 1, i, tracer]
                    qjp2 = 0.0 if j + 2 >= nlat else qqu[j + 2, i, tracer]
                    tmp = (8.0 * (qjp1 - qjm1) + qjm2 - qjp2) * r24
                    pmax = max(qjm1, qj, qjp1) - qj
                    pmin = qj - min(qjm1, qj, qjp1)
                    bounded = min(abs(tmp), pmin, pmax)
                    dcy[j, tracer] = bounded if tmp >= 0.0 else -bounded

            for j in range(1, nlat):
                for tracer in range(ntracer):
                    al[j, tracer] = 0.5 * (qqu[j - 1, i, tracer] + qqu[j, i, tracer])
                    al[j, tracer] += (dcy[j - 1, tracer] - dcy[j, tracer]) * r13
                    ar[j - 1, tracer] = al[j, tracer]

            for j in range(1, nlat - 1):
                for tracer in range(ntracer):
                    a6[j, tracer] = 3.0 * (
                        qqu[j, i, tracer] + qqu[j, i, tracer] - (al[j, tracer] + ar[j, tracer])
                    )
                    if dcy[j, tracer] == 0.0:
                        a6[j, tracer] = 0.0
                        al[j, tracer] = qqu[j, i, tracer]
                        ar[j, tracer] = qqu[j, i, tracer]
                    else:
                        da1 = ar[j, tracer] - al[j, tracer]
                        da2 = da1 * da1
                        a6da = a6[j, tracer] * da1
                        if a6da < -da2:
                            a6[j, tracer] = 3.0 * (al[j, tracer] - qqu[j, i, tracer])
                            ar[j, tracer] = al[j, tracer] - a6[j, tracer]
                        elif a6da > da2:
                            a6[j, tracer] = 3.0 * (ar[j, tracer] - qqu[j, i, tracer])
                            al[j, tracer] = ar[j, tracer] - a6[j, tracer]

            for j in range(j1p, j2p + 2):
                jm1 = j - 1
                c = cy[j, i]
                if c > 0.0:
                    for tracer in range(ntracer):
                        qqv[j, i, tracer] = ar[jm1, tracer] + 0.5 * c * (
                            al[jm1, tracer]
                            - ar[jm1, tracer]
                            + a6[jm1, tracer] * (1.0 - r23 * c)
                        )
                else:
                    for tracer in range(ntracer):
                        qqv[j, i, tracer] = al[j, tracer] - 0.5 * c * (
                            ar[j, tracer] - al[j, tracer] + a6[j, tracer] * (1.0 + r23 * c)
                        )

            for tracer in range(ntracer):
                qqv[j1p, i, tracer] *= ymass[j1p, i]
            for j in range(j1p, j2p + 1):
                for tracer in range(ntracer):
                    qqv[j + 1, i, tracer] *= ymass[j + 1, i]
                    dq1[j, i, tracer] += (qqv[j, i, tracer] - qqv[j + 1, i, tracer]) * geofac[j]
            for tracer in range(ntracer):
                al[0, tracer] += qqv[j1p, i, tracer]
                ar[nlat - 1, tracer] += qqv[j2p + 1, i, tracer]

        for tracer in range(ntracer):
            sumsp = al[0, tracer]
            sumnp = ar[nlat - 1, tracer]
            dq_sp = dq1[0, 0, tracer] - sumsp / float(nlon) * geofac_pc
            dq_np = dq1[nlat - 1, 0, tracer] + sumnp / float(nlon) * geofac_pc
            for i in range(nlon):
                dq1[0, i, tracer] = dq_sp
                dq1[nlat - 1, i, tracer] = dq_np
                dq1[1, i, tracer] = dq_sp
                dq1[nlat - 2, i, tracer] = dq_np


    @njit(cache=True)
    def _fzppm_batch_numba_kernel(delp1: np.ndarray, wz: np.ndarray, dq1: np.ndarray, q: np.ndarray) -> None:
        nlev = q.shape[0]
        nlat = q.shape[1]
        nlon = q.shape[2]
        ntracer = q.shape[3]
        r13 = 1.0 / 3.0
        r23 = 2.0 / 3.0

        dpi = np.empty((nlev, ntracer), dtype=np.float64)
        dc = np.empty((nlev, ntracer), dtype=np.float64)
        al = np.empty((nlev, ntracer), dtype=np.float64)
        ar = np.empty((nlev, ntracer), dtype=np.float64)
        a6 = np.empty((nlev, ntracer), dtype=np.float64)
        dca = np.empty((nlev, ntracer), dtype=np.float64)
        prev_flux = np.empty(ntracer, dtype=np.float64)

        for j in range(nlat):
            if j == 1 or j == nlat - 2:
                continue

            for i in range(nlon):
                for k in range(nlev - 1):
                    for tracer in range(ntracer):
                        dpi[k, tracer] = q[k + 1, j, i, tracer] - q[k, j, i, tracer]
                for tracer in range(ntracer):
                    dpi[nlev - 1, tracer] = 0.0

                for k in range(nlev):
                    for tracer in range(ntracer):
                        dc[k, tracer] = 0.0

                for k in range(1, nlev - 1):
                    dlp_km1 = delp1[k - 1, j, i]
                    dlp_k = delp1[k, j, i]
                    dlp_kp1 = delp1[k + 1, j, i]
                    c0 = dlp_k / (dlp_km1 + dlp_k + dlp_kp1)
                    c1 = (dlp_km1 + 0.5 * dlp_k) / (dlp_kp1 + dlp_k)
                    c2 = (dlp_kp1 + 0.5 * dlp_k) / (dlp_km1 + dlp_k)
                    for tracer in range(ntracer):
                        tmp = c0 * (c1 * dpi[k, tracer] + c2 * dpi[k - 1, tracer])
                        q_center = q[k, j, i, tracer]
                        q_prev = q[k - 1, j, i, tracer]
                        q_next = q[k + 1, j, i, tracer]
                        qmax = max(q_prev, q_center, q_next) - q_center
                        qmin = q_center - min(q_prev, q_center, q_next)
                        bounded = min(abs(tmp), qmax, qmin)
                        dc[k, tracer] = np.copysign(bounded, tmp)

                dlp0 = delp1[0, j, i]
                dlp1_i = delp1[1, j, i]
                dlp2_i = delp1[2, j, i]
                fac2 = (dlp1_i + dlp2_i) * (dlp0 + dlp1_i + dlp2_i)
                top_ratio = (dlp1_i + dlp2_i) / (dlp0 + dlp1_i)
                for tracer in range(ntracer):
                    fac1 = dpi[1, tracer] - dpi[0, tracer] * top_ratio
                    aa = 3.0 * fac1 / fac2
                    bb = 2.0 * dpi[0, tracer] / (dlp0 + dlp1_i)
                    bb -= r23 * aa * (2.0 * dlp0 + dlp1_i)
                    al[0, tracer] = q[0, j, i, tracer] - dlp0 * (r13 * aa * dlp0 + 0.5 * bb)
                    al[1, tracer] = dlp0 * (aa * dlp0 + bb) + al[0, tracer]
                    if q[0, j, i, tracer] * al[0, tracer] <= 0.0:
                        al[0, tracer] = 0.0
                        dca[0, tracer] = 0.0
                    else:
                        dca[0, tracer] = q[0, j, i, tracer] - al[0, tracer]

                dlp_last = delp1[nlev - 1, j, i]
                dlp_prev = delp1[nlev - 2, j, i]
                bottom_ratio = (dlp_last * dlp_last) / ((dlp_last + dlp_prev) * (2.0 * dlp_last + dlp_prev))
                for tracer in range(ntracer):
                    fac1b = dpi[nlev - 2, tracer] * bottom_ratio
                    ar[nlev - 1, tracer] = q[nlev - 1, j, i, tracer] + fac1b
                    al[nlev - 1, tracer] = q[nlev - 1, j, i, tracer] - (fac1b + fac1b)
                    if q[nlev - 1, j, i, tracer] * ar[nlev - 1, tracer] <= 0.0:
                        ar[nlev - 1, tracer] = 0.0
                    dca[nlev - 1, tracer] = ar[nlev - 1, tracer] - q[nlev - 1, j, i, tracer]

                for k in range(2, nlev - 1):
                    dlp_km2 = delp1[k - 2, j, i]
                    dlp_km1 = delp1[k - 1, j, i]
                    dlp_k = delp1[k, j, i]
                    dlp_kp1 = delp1[k + 1, j, i]
                    c2 = 2.0 / (dlp_km2 + dlp_km1 + dlp_k + dlp_kp1)
                    a1 = (dlp_km2 + dlp_km1) / (2.0 * dlp_km1 + dlp_k)
                    a2 = (dlp_k + dlp_kp1) / (2.0 * dlp_k + dlp_km1)
                    c1_scale = dlp_km1 / (dlp_km1 + dlp_k)
                    for tracer in range(ntracer):
                        c1 = dpi[k - 1, tracer] * c1_scale
                        al[k, tracer] = q[k - 1, j, i, tracer] + c1 + c2 * (
                            dlp_k * (c1 * (a1 - a2) + a2 * dc[k - 1, tracer])
                            - dlp_km1 * a1 * dc[k, tracer]
                        )

                for k in range(nlev - 1):
                    for tracer in range(ntracer):
                        ar[k, tracer] = al[k + 1, tracer]

                for kk in range(2):
                    if kk == 0:
                        k = 0
                    else:
                        k = nlev - 1
                    for tracer in range(ntracer):
                        qa = q[k, j, i, tracer]
                        a6[k, tracer] = 3.0 * (qa + qa - (al[k, tracer] + ar[k, tracer]))
                        if dca[k, tracer] == 0.0:
                            a6[k, tracer] = 0.0
                            al[k, tracer] = qa
                            ar[k, tracer] = qa
                        else:
                            da1 = ar[k, tracer] - al[k, tracer]
                            da2 = da1 * da1
                            a6da = a6[k, tracer] * da1
                            if a6da < -da2:
                                a6[k, tracer] = 3.0 * (al[k, tracer] - qa)
                                ar[k, tracer] = al[k, tracer] - a6[k, tracer]
                            elif a6da > da2:
                                a6[k, tracer] = 3.0 * (ar[k, tracer] - qa)
                                al[k, tracer] = ar[k, tracer] - a6[k, tracer]

                for kk in range(2):
                    if kk == 0:
                        k = 1
                    else:
                        k = nlev - 2
                    for tracer in range(ntracer):
                        qa = q[k, j, i, tracer]
                        a6[k, tracer] = 3.0 * (qa + qa - (al[k, tracer] + ar[k, tracer]))
                        if dc[k, tracer] == 0.0:
                            a6[k, tracer] = 0.0
                            al[k, tracer] = qa
                            ar[k, tracer] = qa
                        else:
                            da1 = ar[k, tracer] - al[k, tracer]
                            da2 = da1 * da1
                            a6da = a6[k, tracer] * da1
                            if a6da < -da2:
                                a6[k, tracer] = 3.0 * (al[k, tracer] - qa)
                                ar[k, tracer] = al[k, tracer] - a6[k, tracer]
                            elif a6da > da2:
                                a6[k, tracer] = 3.0 * (ar[k, tracer] - qa)
                                al[k, tracer] = ar[k, tracer] - a6[k, tracer]

                for k in range(1, nlev - 1):
                    for tracer in range(ntracer):
                        dca[k, tracer] = dpi[k, tracer] - dpi[k - 1, tracer]

                for k in range(2, nlev - 2):
                    for tracer in range(ntracer):
                        qq = q[k, j, i, tracer]
                        qmp = qq + 2.0 * dpi[k - 1, tracer]
                        lac = qq + 1.5 * dca[k - 1, tracer] + 0.5 * dpi[k - 1, tracer]
                        qmin = min(qq, qmp, lac)
                        qmax = max(qq, qmp, lac)
                        if ar[k, tracer] < qmin:
                            ar[k, tracer] = qmin
                        elif ar[k, tracer] > qmax:
                            ar[k, tracer] = qmax

                        qmp = qq - 2.0 * dpi[k, tracer]
                        lac = qq + 1.5 * dca[k + 1, tracer] - 0.5 * dpi[k, tracer]
                        qmin = min(qq, qmp, lac)
                        qmax = max(qq, qmp, lac)
                        if al[k, tracer] < qmin:
                            al[k, tracer] = qmin
                        elif al[k, tracer] > qmax:
                            al[k, tracer] = qmax
                        a6[k, tracer] = 3.0 * (qq + qq - (ar[k, tracer] + al[k, tracer]))

                if wz[0, j, i] > 0.0:
                    cm = wz[0, j, i] / delp1[0, j, i]
                    for tracer in range(ntracer):
                        val = ar[0, tracer] + 0.5 * cm * (
                            al[0, tracer] - ar[0, tracer] + a6[0, tracer] * (1.0 - r23 * cm)
                        )
                        flux = wz[0, j, i] * val
                        dq1[0, j, i, tracer] -= flux
                        prev_flux[tracer] = flux
                else:
                    cp = wz[0, j, i] / delp1[1, j, i]
                    for tracer in range(ntracer):
                        val = al[1, tracer] + 0.5 * cp * (
                            al[1, tracer] - ar[1, tracer] - a6[1, tracer] * (1.0 + r23 * cp)
                        )
                        flux = wz[0, j, i] * val
                        dq1[0, j, i, tracer] -= flux
                        prev_flux[tracer] = flux
                for k in range(1, nlev - 1):
                    if wz[k, j, i] > 0.0:
                        cm = wz[k, j, i] / delp1[k, j, i]
                        for tracer in range(ntracer):
                            val = ar[k, tracer] + 0.5 * cm * (
                                al[k, tracer] - ar[k, tracer] + a6[k, tracer] * (1.0 - r23 * cm)
                            )
                            flux = wz[k, j, i] * val
                            dq1[k, j, i, tracer] += prev_flux[tracer] - flux
                            prev_flux[tracer] = flux
                    else:
                        cp = wz[k, j, i] / delp1[k + 1, j, i]
                        for tracer in range(ntracer):
                            val = al[k + 1, tracer] + 0.5 * cp * (
                                al[k + 1, tracer]
                                - ar[k + 1, tracer]
                                - a6[k + 1, tracer] * (1.0 + r23 * cp)
                            )
                            flux = wz[k, j, i] * val
                            dq1[k, j, i, tracer] += prev_flux[tracer] - flux
                            prev_flux[tracer] = flux
                for tracer in range(ntracer):
                    dq1[nlev - 1, j, i, tracer] += prev_flux[tracer]


else:

    def _average_const_poles_batch_numba_kernel(q: np.ndarray, delp1: np.ndarray, area_1d: np.ndarray) -> None:
        raise RuntimeError("numba is not available")


    def _qckxyz_batch_numba_kernel(dq1: np.ndarray) -> None:
        raise RuntimeError("numba is not available")


    def _qckxyz_needs_fill_numba_kernel(dq1: np.ndarray) -> bool:
        raise RuntimeError("numba is not available")


    def _finalize_tpcore_output_numba_kernel(dq1: np.ndarray, delp2: np.ndarray) -> None:
        raise RuntimeError("numba is not available")


    def _advect_tracers_fused_numba_kernel(
        q: np.ndarray,
        dq1: np.ndarray,
        delp1: np.ndarray,
        delp2: np.ndarray,
        pu: np.ndarray,
        xmass: np.ndarray,
        ymass: np.ndarray,
        wz: np.ndarray,
        cx: np.ndarray,
        cy: np.ndarray,
        geofac: np.ndarray,
        geofac_pc: float,
        ua: np.ndarray,
        va: np.ndarray,
        jn: np.ndarray,
        js: np.ndarray,
        area_1d: np.ndarray,
        fill: bool,
        qqu: np.ndarray,
        qqv: np.ndarray,
        dcx: np.ndarray,
        fx: np.ndarray,
        al_x: np.ndarray,
        ar_x: np.ndarray,
        a6_x: np.ndarray,
        dc_x: np.ndarray,
        qa_x: np.ndarray,
        dcy: np.ndarray,
        al_y: np.ndarray,
        ar_y: np.ndarray,
        a6_y: np.ndarray,
    ) -> None:
        raise RuntimeError("numba is not available")


    def _fzppm_batch_numba_kernel(delp1: np.ndarray, wz: np.ndarray, dq1: np.ndarray, q: np.ndarray) -> None:
        raise RuntimeError("numba is not available")
