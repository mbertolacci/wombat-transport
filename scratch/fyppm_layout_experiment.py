from __future__ import annotations

import gc
import time

import numpy as np

from wombat_transport.transport.tpcore import _fyppm_batch, _lmtppm_last_axis, _polar_cap_bounds


def _fyppm_batch_tracer_last(cy: np.ndarray, dcy: np.ndarray, qqu: np.ndarray, qqv: np.ndarray) -> None:
    nlat, nlon, _ntracer = qqu.shape
    j1p, j2p = _polar_cap_bounds(nlat)
    r13 = 1.0 / 3.0
    r23 = 2.0 / 3.0
    a6 = np.empty_like(qqu)
    al = np.empty_like(qqu)
    ar = np.empty_like(qqu)

    al[1:, :, :] = qqu[:-1, :, :]
    al[1:, :, :] += qqu[1:, :, :]
    al[1:, :, :] *= 0.5
    al[1:, :, :] += (dcy[:-1, :, :] - dcy[1:, :, :]) * r13
    ar[:-1, :, :] = al[1:, :, :]

    half = nlon // 2
    al[0, :half, :] = al[1, half:, :]
    al[0, half:, :] = al[1, :half, :]
    ar[-1, :half, :] = ar[-2, half:, :]
    ar[-1, half:, :] = ar[-2, :half, :]

    a6[1:-1, :, :] = 3.0 * (qqu[1:-1, :, :] + qqu[1:-1, :, :] - (al[1:-1, :, :] + ar[1:-1, :, :]))
    for j in range(1, nlat - 1):
        _lmtppm_last_axis(a6[j, :, :], al[j, :, :], ar[j, :, :], dcy[j, :, :], qqu[j, :, :], 0)

    for j in range(j1p, j2p + 2):
        jm1 = j - 1
        c = cy[j, :]
        pos = c > 0.0
        if np.any(pos):
            cp = c[pos][:, np.newaxis]
            qqv[j, pos, :] = ar[jm1, pos, :] + 0.5 * cp * (
                al[jm1, pos, :] - ar[jm1, pos, :] + a6[jm1, pos, :] * (1.0 - r23 * cp)
            )
        neg = ~pos
        if np.any(neg):
            cn = c[neg][:, np.newaxis]
            qqv[j, neg, :] = al[j, neg, :] - 0.5 * cn * (
                ar[j, neg, :] - al[j, neg, :] + a6[j, neg, :] * (1.0 + r23 * cn)
            )


def _make_inputs(ntracer: int, nlat: int = 91, nlon: int = 144) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12345)
    lat = np.linspace(-89.0, 89.0, nlat, dtype=np.float64)
    lon = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False, dtype=np.float64)
    tracer = np.arange(ntracer, dtype=np.float64)[:, np.newaxis, np.newaxis]
    lat_wave = np.sin(np.deg2rad(lat))[np.newaxis, :, np.newaxis]
    lon_wave = np.cos(lon)[np.newaxis, np.newaxis, :]
    qqu = 4.0e-4 + (tracer + 1.0) * 1.0e-7 + 2.0e-8 * lat_wave + 1.0e-8 * lon_wave
    qqu += rng.normal(0.0, 2.0e-10, size=qqu.shape)
    dcy = rng.normal(0.0, 1.0e-10, size=qqu.shape)
    cy = 0.45 * np.sin(lat[:, np.newaxis] / 90.0) * np.cos(lon[np.newaxis, :])
    qqv = np.zeros_like(qqu)
    return cy, dcy, qqu, qqv


def _time_call(fn, repeat: int = 7) -> tuple[float, float]:
    times: list[float] = []
    for _ in range(repeat):
        gc.collect()
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times), sum(times) / len(times)


def main() -> None:
    print("operator,_fyppm_batch")
    print("layout,current=(tracer,lat,lon),candidate=(lat,lon,tracer) C-contiguous")
    print("tracers,max_abs,baseline_best_s,baseline_mean_s,tracer_last_best_s,tracer_last_mean_s,speedup,baseline_tracer_s,tracer_last_tracer_s")
    for ntracer in (24, 96, 256, 512):
        cy, dcy, qqu, qqv0 = _make_inputs(ntracer)
        qqu_t = np.ascontiguousarray(np.transpose(qqu, (1, 2, 0)))
        dcy_t = np.ascontiguousarray(np.transpose(dcy, (1, 2, 0)))
        qqv0_t = np.ascontiguousarray(np.transpose(qqv0, (1, 2, 0)))

        expected = qqv0.copy()
        actual_t = qqv0_t.copy()
        _fyppm_batch(cy, dcy, qqu, expected)
        _fyppm_batch_tracer_last(cy, dcy_t, qqu_t, actual_t)
        actual = np.transpose(actual_t, (2, 0, 1))
        max_abs = float(np.max(np.abs(actual - expected)))
        np.testing.assert_array_equal(actual, expected)

        def baseline_once() -> None:
            out = qqv0.copy()
            _fyppm_batch(cy, dcy, qqu, out)

        def tracer_last_once() -> None:
            out = qqv0_t.copy()
            _fyppm_batch_tracer_last(cy, dcy_t, qqu_t, out)

        baseline_best, baseline_mean = _time_call(baseline_once)
        tracer_last_best, tracer_last_mean = _time_call(tracer_last_once)
        print(
            f"{ntracer},{max_abs:.3e},{baseline_best:.8f},{baseline_mean:.8f},"
            f"{tracer_last_best:.8f},{tracer_last_mean:.8f},{baseline_best / tracer_last_best:.4f},"
            f"{ntracer / baseline_best:.4f},{ntracer / tracer_last_best:.4f}"
        )
        print(
            f"# flags tracers={ntracer} baseline_c={qqu.flags.c_contiguous} baseline_f={qqu.flags.f_contiguous} "
            f"baseline_strides={qqu.strides} tracer_last_c={qqu_t.flags.c_contiguous} "
            f"tracer_last_f={qqu_t.flags.f_contiguous} tracer_last_strides={qqu_t.strides}"
        )


if __name__ == "__main__":
    main()
