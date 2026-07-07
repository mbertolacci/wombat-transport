from __future__ import annotations

import gc
import time

import numpy as np

from wombat_transport.transport.tpcore import _fzppm_batch, _lmtppm_last_axis, _signed_minimum_abs


def _fzppm_batch_tracer_last(delp1: np.ndarray, wz: np.ndarray, dq1: np.ndarray, q: np.ndarray) -> None:
    nlev, nlat, nlon, ntracer = q.shape
    r13 = 1.0 / 3.0
    r23 = 2.0 / 3.0

    dpi = np.empty((nlev, nlon, ntracer), dtype=np.float64)
    dc = np.empty((nlev, nlon, ntracer), dtype=np.float64)
    al = np.empty((nlev, nlon, ntracer), dtype=np.float64)
    ar = np.empty((nlev, nlon, ntracer), dtype=np.float64)
    a6 = np.empty((nlev, nlon, ntracer), dtype=np.float64)
    dca = np.empty((nlev, nlon, ntracer), dtype=np.float64)
    tmp = np.empty((nlon, ntracer), dtype=np.float64)
    qmin = np.empty((nlon, ntracer), dtype=np.float64)
    qmax = np.empty((nlon, ntracer), dtype=np.float64)
    prev_flux = np.empty((nlon, ntracer), dtype=np.float64)
    flux = np.empty((nlon, ntracer), dtype=np.float64)

    for j in range(nlat):
        if j in (1, nlat - 2):
            continue
        dlp = delp1[:, j, :]
        qq = q[:, j, :, :]
        wza = wz[:, j, :]

        dpi[:-1, :, :] = qq[1:, :, :] - qq[:-1, :, :]
        dpi[-1, :, :] = 0.0
        dc.fill(0.0)

        for k in range(1, nlev - 1):
            c0 = dlp[k] / (dlp[k - 1] + dlp[k] + dlp[k + 1])
            c1 = (dlp[k - 1] + 0.5 * dlp[k]) / (dlp[k + 1] + dlp[k])
            c2 = (dlp[k + 1] + 0.5 * dlp[k]) / (dlp[k - 1] + dlp[k])
            tmp[:] = dpi[k, :, :] * c1[:, np.newaxis]
            tmp += dpi[k - 1, :, :] * c2[:, np.newaxis]
            tmp *= c0[:, np.newaxis]

            np.maximum(qq[k - 1, :, :], qq[k, :, :], out=qmax)
            np.maximum(qmax, qq[k + 1, :, :], out=qmax)
            qmax -= qq[k, :, :]
            np.minimum(qq[k - 1, :, :], qq[k, :, :], out=qmin)
            np.minimum(qmin, qq[k + 1, :, :], out=qmin)
            np.subtract(qq[k, :, :], qmin, out=qmin)
            _signed_minimum_abs(tmp, qmax, qmin, dc[k, :, :])

        dca[:] = dc
        fac1 = dpi[1, :, :] - dpi[0, :, :] * ((dlp[1, :] + dlp[2, :]) / (dlp[0, :] + dlp[1, :]))[:, np.newaxis]
        fac2 = (dlp[1, :] + dlp[2, :]) * (dlp[0, :] + dlp[1, :] + dlp[2, :])
        aa = 3.0 * fac1 / fac2[:, np.newaxis]
        bb = 2.0 * dpi[0, :, :] / (dlp[0, :] + dlp[1, :])[:, np.newaxis]
        bb -= r23 * aa * (2.0 * dlp[0, :] + dlp[1, :])[:, np.newaxis]
        al[0, :, :] = qq[0, :, :] - dlp[0, :, np.newaxis] * (r13 * aa * dlp[0, :, np.newaxis] + 0.5 * bb)
        al[1, :, :] = dlp[0, :, np.newaxis] * (aa * dlp[0, :, np.newaxis] + bb) + al[0, :, :]
        mask = qq[0, :, :] * al[0, :, :] <= 0.0
        dca[0, :, :] = qq[0, :, :] - al[0, :, :]
        al[0, :, :][mask] = 0.0
        dca[0, :, :][mask] = 0.0

        fac1b = dpi[-2, :, :] * (
            (dlp[-1, :] * dlp[-1, :]) / ((dlp[-1, :] + dlp[-2, :]) * (2.0 * dlp[-1, :] + dlp[-2, :]))
        )[:, np.newaxis]
        ar[-1, :, :] = qq[-1, :, :] + fac1b
        al[-1, :, :] = qq[-1, :, :] - (fac1b + fac1b)
        ar[-1, :, :][qq[-1, :, :] * ar[-1, :, :] <= 0.0] = 0.0
        dca[-1, :, :] = ar[-1, :, :] - qq[-1, :, :]

        for k in range(2, nlev - 1):
            c1 = dpi[k - 1, :, :] * (dlp[k - 1, :] / (dlp[k - 1, :] + dlp[k, :]))[:, np.newaxis]
            c2 = 2.0 / (dlp[k - 2, :] + dlp[k - 1, :] + dlp[k, :] + dlp[k + 1, :])
            a1 = (dlp[k - 2, :] + dlp[k - 1, :]) / (2.0 * dlp[k - 1, :] + dlp[k, :])
            a2 = (dlp[k, :] + dlp[k + 1, :]) / (2.0 * dlp[k, :] + dlp[k - 1, :])
            al[k, :, :] = qq[k - 1, :, :] + c1 + c2[:, np.newaxis] * (
                dlp[k, :, np.newaxis] * (c1 * (a1 - a2)[:, np.newaxis] + a2[:, np.newaxis] * dca[k - 1, :, :])
                - dlp[k - 1, :, np.newaxis] * a1[:, np.newaxis] * dca[k, :, :]
            )

        ar[:-1, :, :] = al[1:, :, :]
        for k in (0, 1, nlev - 2, nlev - 1):
            a6[k, :, :] = 3.0 * (qq[k, :, :] + qq[k, :, :] - (al[k, :, :] + ar[k, :, :]))
            _lmtppm_last_axis(a6[k, :, :], al[k, :, :], ar[k, :, :], dca[k, :, :], qq[k, :, :], 0)

        for k in range(1, nlev - 1):
            dca[k, :, :] = dpi[k, :, :] - dpi[k - 1, :, :]

        for k in range(2, nlev - 2):
            tmp[:] = qq[k, :, :] + 2.0 * dpi[k - 1, :, :]
            qmin[:] = qq[k, :, :]
            np.minimum(qmin, tmp, out=qmin)
            qmax[:] = qq[k, :, :]
            np.maximum(qmax, tmp, out=qmax)
            tmp[:] = qq[k, :, :] + 1.5 * dca[k - 1, :, :] + 0.5 * dpi[k - 1, :, :]
            np.minimum(qmin, tmp, out=qmin)
            np.maximum(qmax, tmp, out=qmax)
            np.maximum(ar[k, :, :], qmin, out=tmp)
            np.minimum(tmp, qmax, out=ar[k, :, :])

            tmp[:] = qq[k, :, :] - 2.0 * dpi[k, :, :]
            qmin[:] = qq[k, :, :]
            np.minimum(qmin, tmp, out=qmin)
            qmax[:] = qq[k, :, :]
            np.maximum(qmax, tmp, out=qmax)
            tmp[:] = qq[k, :, :] + 1.5 * dca[k + 1, :, :] - 0.5 * dpi[k, :, :]
            np.minimum(qmin, tmp, out=qmin)
            np.maximum(qmax, tmp, out=qmax)
            np.maximum(al[k, :, :], qmin, out=tmp)
            np.minimum(tmp, qmax, out=al[k, :, :])
            a6[k, :, :] = 3.0 * (qq[k, :, :] + qq[k, :, :] - (ar[k, :, :] + al[k, :, :]))

        prev_flux.fill(0.0)
        for k in range(nlev - 1):
            pos = wza[k, :] > 0.0
            if np.any(pos):
                cm = (wza[k, pos] / dlp[k, pos])[:, np.newaxis]
                val = ar[k, pos, :] + 0.5 * cm * (
                    al[k, pos, :] - ar[k, pos, :] + a6[k, pos, :] * (1.0 - r23 * cm)
                )
                flux[pos, :] = wza[k, pos, np.newaxis] * val
            neg = ~pos
            if np.any(neg):
                cp = (wza[k, neg] / dlp[k + 1, neg])[:, np.newaxis]
                val = al[k + 1, neg, :] + 0.5 * cp * (
                    al[k + 1, neg, :] - ar[k + 1, neg, :] - a6[k + 1, neg, :] * (1.0 + r23 * cp)
                )
                flux[neg, :] = wza[k, neg, np.newaxis] * val
            if k == 0:
                dq1[0, j, :, :] -= flux
            else:
                dq1[k, j, :, :] += prev_flux - flux
            prev_flux[:] = flux
        dq1[-1, j, :, :] += prev_flux


def _make_inputs(ntracer: int, nlev: int = 47, nlat: int = 91, nlon: int = 144):
    rng = np.random.default_rng(24680)
    lev = np.arange(nlev, dtype=np.float64)[:, np.newaxis, np.newaxis]
    lat = np.linspace(-89.0, 89.0, nlat, dtype=np.float64)[np.newaxis, :, np.newaxis]
    lon = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False, dtype=np.float64)[np.newaxis, np.newaxis, :]
    tracer = np.arange(ntracer, dtype=np.float64)[:, np.newaxis, np.newaxis, np.newaxis]

    delp1 = 18.0 + 0.1 * lev + 0.2 * np.cos(np.deg2rad(lat)) + 0.03 * np.cos(lon)
    wz = 0.02 * np.sin((lev + 1.0) / nlev * np.pi) * np.cos(lon) * np.cos(np.deg2rad(lat))
    q = 4.0e-4 + (tracer + 1.0) * 1.0e-7
    q = q + 2.5e-8 * lev[np.newaxis, :, :, :] / max(float(nlev - 1), 1.0)
    q = q + 1.5e-8 * np.sin(np.deg2rad(lat))[np.newaxis, :, :, :]
    q = q + 7.5e-9 * np.cos(lon)[np.newaxis, :, :, :]
    q = q + rng.normal(0.0, 1.0e-11, size=q.shape)
    dq1 = q * delp1[np.newaxis, :, :, :]
    return delp1, wz, dq1, q


def _time_call(fn, repeat: int = 5) -> tuple[float, float]:
    values: list[float] = []
    for _ in range(repeat):
        gc.collect()
        start = time.perf_counter()
        fn()
        values.append(time.perf_counter() - start)
    return min(values), sum(values) / len(values)


def main() -> None:
    print("operator,_fzppm_batch")
    print("layout,current=(tracer,lev,lat,lon),candidate=(lev,lat,lon,tracer) C-contiguous")
    print("tracers,max_abs,baseline_best_s,baseline_mean_s,tracer_last_best_s,tracer_last_mean_s,speedup,baseline_tracer_s,tracer_last_tracer_s")
    for ntracer in (24, 96, 256, 512):
        delp1, wz, dq1, q = _make_inputs(ntracer)
        q_t = np.ascontiguousarray(np.transpose(q, (1, 2, 3, 0)))
        dq1_t = np.ascontiguousarray(np.transpose(dq1, (1, 2, 3, 0)))

        expected = dq1.copy()
        actual_t = dq1_t.copy()
        _fzppm_batch(delp1, wz, expected, q)
        _fzppm_batch_tracer_last(delp1, wz, actual_t, q_t)
        actual = np.transpose(actual_t, (3, 0, 1, 2))
        max_abs = float(np.max(np.abs(actual - expected)))
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

        def baseline_once() -> None:
            out = dq1.copy()
            _fzppm_batch(delp1, wz, out, q)

        def tracer_last_once() -> None:
            out = dq1_t.copy()
            _fzppm_batch_tracer_last(delp1, wz, out, q_t)

        baseline_best, baseline_mean = _time_call(baseline_once)
        tracer_last_best, tracer_last_mean = _time_call(tracer_last_once)
        print(
            f"{ntracer},{max_abs:.3e},{baseline_best:.8f},{baseline_mean:.8f},"
            f"{tracer_last_best:.8f},{tracer_last_mean:.8f},{baseline_best / tracer_last_best:.4f},"
            f"{ntracer / baseline_best:.4f},{ntracer / tracer_last_best:.4f}"
        )
        print(
            f"# flags tracers={ntracer} baseline_c={q.flags.c_contiguous} baseline_f={q.flags.f_contiguous} "
            f"baseline_strides={q.strides} tracer_last_c={q_t.flags.c_contiguous} "
            f"tracer_last_f={q_t.flags.f_contiguous} tracer_last_strides={q_t.strides}"
        )


if __name__ == "__main__":
    main()
