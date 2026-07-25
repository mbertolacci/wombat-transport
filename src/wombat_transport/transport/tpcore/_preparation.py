"""Allocation-free compiled preparation for the TPCORE plan."""

from __future__ import annotations

import numpy as np

from wombat_transport.constants import EARTH_RADIUS_M

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


def _prepare_tpcore_arrays_python(
    p1_input,
    p2_input,
    u,
    v,
    dt_s,
    rel_area,
    geofac,
    geofac_pc,
    cose,
    cosp,
    ak_top_first,
    dap_geos,
    dbk_geos,
    dap_top,
    dbk_top,
    plan,
    scratch,
):
    (
        delp1,
        delpm,
        delp2,
        pu,
        surface_pressure,
        xmass,
        ymass,
        wz,
        normalized_vertical_courant,
        cx,
        cy,
        ua,
        va,
        jn,
        js,
    ) = plan
    p1, p2, work3, work2, xfix, mmfd, mmf, fxintegral = scratch
    nlev, nlat, nlon = delp1.shape
    j1p = 2
    j2p = nlat - 3

    dgpress = 0.0
    for j in range(nlat):
        for i in range(nlon):
            p1[j, i] = p1_input[j, i]
            p2[j, i] = p2_input[j, i]
            dgpress += (p2[j, i] - p1[j, i]) * rel_area[j, i]
    for j in range(nlat):
        for i in range(nlon):
            p2[j, i] -= dgpress
    _average_poles(p1, rel_area)
    _average_poles(p2, rel_area)

    dlambda = 2.0 * np.pi / nlon
    dphi = np.pi / (nlat - 1)
    for lev in range(nlev):
        for j in range(nlat):
            for i in range(nlon):
                work3[lev, j, i] = dap_geos[lev] + dbk_geos[lev] * 0.5 * (
                    p1[j, i] + p2[j, i]
                )
        for j in range(nlat):
            factx = 0.5 * dt_s / (dlambda * EARTH_RADIUS_M * cosp[j])
            for i in range(nlon):
                im1 = nlon - 1 if i == 0 else i - 1
                xmass[nlev - 1 - lev, j, i] = factx * (
                    u[lev, j, i] * work3[lev, j, i]
                    + u[lev, j, im1] * work3[lev, j, im1]
                )
        facty = 0.5 * dt_s / (EARTH_RADIUS_M * dphi)
        for i in range(nlon):
            ymass[nlev - 1 - lev, 0, i] = (
                facty * cose[0] * v[lev, 0, i] * work3[lev, 0, i]
            )
        for j in range(1, nlat):
            for i in range(nlon):
                ymass[nlev - 1 - lev, j, i] = facty * cose[j] * (
                    v[lev, j, i] * work3[lev, j, i]
                    + v[lev, j - 1, i] * work3[lev, j - 1, i]
                )

    _divergence(xmass, ymass, geofac, geofac_pc, work3, True)
    for j in range(nlat):
        for i in range(nlon):
            value = 0.0
            for lev in range(nlev):
                value += work3[nlev - 1 - lev, j, i]
            work2[j, i] = value
    dgpress = 0.0
    for j in range(nlat):
        for i in range(nlon):
            dgpress += (p2[j, i] - p1[j, i] - work2[j, i]) * rel_area[j, i]
    for j in range(nlat):
        total = 0.0
        for i in range(nlon):
            total += p2[j, i] - p1[j, i] - work2[j, i]
        mean = total / nlon
        mmfd[j] = -(mean - dgpress)
    mmfd[0] = -(p2[0, 0] - p1[0, 0] - work2[0, 0] - dgpress)
    mmfd[1] = -(p2[1, 0] - p1[1, 0] - work2[1, 0] - dgpress)
    mmfd[nlat - 2] = -(
        p2[nlat - 2, 0] - p1[nlat - 2, 0] - work2[nlat - 2, 0] - dgpress
    )
    mmfd[nlat - 1] = -(
        p2[nlat - 1, 0] - p1[nlat - 1, 0] - work2[nlat - 1, 0] - dgpress
    )
    for j in range(nlat):
        mmf[j] = 0.0
        for i in range(nlon):
            xfix[j, i] = 0.0
    mmf[j1p] = mmfd[0] / geofac_pc
    for j in range(j1p, j2p + 1):
        mmf[j + 1] = mmf[j] + mmfd[j] / geofac[j]
        fxintegral[0] = 0.0
        total = 0.0
        for i in range(nlon):
            ddps = p2[j, i] - p1[j, i] - work2[j, i]
            fxintegral[i + 1] = fxintegral[i] - (ddps - dgpress) - mmfd[j]
            total += fxintegral[i + 1]
        mean = total / nlon
        for i in range(nlon):
            xfix[j, i] = fxintegral[i] - mean
    for lev_top in range(nlev):
        lev_bottom = nlev - 1 - lev_top
        for j in range(nlat):
            for i in range(nlon):
                xmass[lev_top, j, i] += dbk_geos[lev_bottom] * xfix[j, i]
        for j in range(j1p, j2p + 2):
            for i in range(nlon):
                ymass[lev_top, j, i] += dbk_geos[lev_bottom] * mmf[j]

    # PJC pressure fixing operates on private pressure copies. TPCORE pressure
    # terms use the original boundaries with only polar averaging.
    for j in range(nlat):
        for i in range(nlon):
            p1[j, i] = p1_input[j, i]
            p2[j, i] = p2_input[j, i]
    _average_poles(p1, rel_area)
    _average_poles(p2, rel_area)

    for lev in range(nlev):
        for j in range(nlat):
            for i in range(nlon):
                delp1[lev, j, i] = dap_top[lev] + dbk_top[lev] * p1[j, i]
                delpm[lev, j, i] = dap_top[lev] + dbk_top[lev] * 0.5 * (
                    p1[j, i] + p2[j, i]
                )
                delp2[lev, j, i] = dap_top[lev] + dbk_top[lev] * p2[j, i]
                pu[lev, j, i] = 0.0
                cx[lev, j, i] = 0.0
                cy[lev, j, i] = 0.0
        for j in range(j1p, j2p + 1):
            pu[lev, j, 0] = 0.5 * (delpm[lev, j, 0] + delpm[lev, j, nlon - 1])
            for i in range(1, nlon):
                pu[lev, j, i] = 0.5 * (delpm[lev, j, i] + delpm[lev, j, i - 1])
            for i in range(nlon):
                cx[lev, j, i] = xmass[lev, j, i] / pu[lev, j, i]
                cy[lev, j, i] = ymass[lev, j, i] / (
                    0.5
                    * cose[j]
                    * (delpm[lev, j, i] + delpm[lev, j - 1, i])
                )
        for i in range(nlon):
            cy[lev, j2p + 1, i] = ymass[lev, j2p + 1, i] / (
                0.5
                * cose[j2p + 1]
                * (delpm[lev, j2p + 1, i] + delpm[lev, j2p, i])
            )

    for j in range(nlat):
        for i in range(nlon):
            value = ak_top_first
            for lev in range(nlev):
                value += delp2[lev, j, i]
            surface_pressure[j, i] = value

    _divergence(xmass, ymass, geofac, geofac_pc, work3, False)
    for j in range(nlat):
        for i in range(nlon):
            total = 0.0
            for lev in range(nlev):
                total += work3[lev, j, i]
            work2[j, i] = total
    for j in range(nlat):
        for i in range(nlon):
            wz[0, j, i] = work3[0, j, i] - dbk_top[0] * work2[j, i]
            if wz[0, j, i] > 0.0:
                normalized_vertical_courant[0, j, i] = wz[0, j, i] / delp1[0, j, i]
            else:
                normalized_vertical_courant[0, j, i] = wz[0, j, i] / delp1[1, j, i]
            for lev in range(1, nlev - 1):
                wz[lev, j, i] = (
                    wz[lev - 1, j, i]
                    + work3[lev, j, i]
                    - dbk_top[lev] * work2[j, i]
                )
                if wz[lev, j, i] > 0.0:
                    normalized_vertical_courant[lev, j, i] = (
                        wz[lev, j, i] / delp1[lev, j, i]
                    )
                else:
                    normalized_vertical_courant[lev, j, i] = (
                        wz[lev, j, i] / delp1[lev + 1, j, i]
                    )
            wz[nlev - 1, j, i] = 0.0
            normalized_vertical_courant[nlev - 1, j, i] = 0.0

    for lev in range(nlev):
        for j in range(nlat):
            for i in range(nlon):
                ua[lev, j, i] = 0.0
                va[lev, j, i] = 0.0
        for j in range(j1p, j2p + 1):
            for i in range(nlon - 1):
                ua[lev, j, i] = 0.5 * (cx[lev, j, i] + cx[lev, j, i + 1])
            ua[lev, j, nlon - 1] = 0.5 * (
                cx[lev, j, nlon - 1] + cx[lev, j, 0]
            )
        for j in range(1, nlat - 1):
            for i in range(nlon):
                va[lev, j, i] = 0.5 * (cy[lev, j, i] + cy[lev, j + 1, i])
    _set_jn_js(cx, jn, js)


def _average_poles(pressure, rel_area):
    nlat, nlon = pressure.shape
    south = 0.0
    south_weight = 0.0
    north = 0.0
    north_weight = 0.0
    for j in range(2):
        for i in range(nlon):
            south += pressure[j, i] * rel_area[j, i]
            south_weight += rel_area[j, i]
    for j in range(nlat - 2, nlat):
        for i in range(nlon):
            north += pressure[j, i] * rel_area[j, i]
            north_weight += rel_area[j, i]
    south /= south_weight
    north /= north_weight
    for i in range(nlon):
        pressure[0, i] = south
        pressure[1, i] = south
        pressure[nlat - 2, i] = north
        pressure[nlat - 1, i] = north


def _divergence(xmass, ymass, geofac, geofac_pc, output, bottom_reversed):
    nlev, nlat, nlon = xmass.shape
    j1p = 2
    j2p = nlat - 3
    for lev in range(nlev):
        target = nlev - 1 - lev if bottom_reversed else lev
        for j in range(nlat):
            for i in range(nlon):
                output[target, j, i] = 0.0
        for j in range(j1p, j2p + 1):
            for i in range(nlon):
                ip1 = 0 if i == nlon - 1 else i + 1
                output[target, j, i] = (
                    (ymass[lev, j, i] - ymass[lev, j + 1, i]) * geofac[j]
                    + xmass[lev, j, i]
                    - xmass[lev, j, ip1]
                )
        south = 0.0
        north = 0.0
        for i in range(nlon):
            south += ymass[lev, j1p, i]
            north += ymass[lev, j2p + 1, i]
        south = -(south / nlon) * geofac_pc
        north = (north / nlon) * geofac_pc
        for i in range(nlon):
            output[target, 0, i] = south
            output[target, 1, i] = south
            output[target, nlat - 2, i] = north
            output[target, nlat - 1, i] = north


def _set_jn_js(cx, jn, js):
    nlev, nlat, nlon = cx.shape
    j1p = 2
    j2p = nlat - 3
    js0 = (nlat + 1) // 2 - 1
    jn0 = nlat - (js0 + 1)
    for lev in range(nlev):
        js_value = j1p
        for j in range(min(nlat - 1, js0), max(0, j1p) - 1, -1):
            found = False
            for i in range(nlon):
                if abs(cx[lev, j, i]) > 1.0:
                    found = True
                    break
            if found:
                js_value = j
                break
        jn_value = j2p
        for j in range(max(0, jn0), min(nlat - 1, j2p) + 1):
            found = False
            for i in range(nlon):
                if abs(cx[lev, j, i]) > 1.0:
                    found = True
                    break
            if found:
                jn_value = j
                break
        js[lev] = js_value
        jn[lev] = jn_value


if njit is not None:
    _average_poles = njit(inline="always", nogil=True)(_average_poles)
    _divergence = njit(inline="always", nogil=True)(_divergence)
    _set_jn_js = njit(inline="always", nogil=True)(_set_jn_js)
    prepare_tpcore_arrays = njit(nogil=True, cache=True)(_prepare_tpcore_arrays_python)
else:  # pragma: no cover
    prepare_tpcore_arrays = _prepare_tpcore_arrays_python
