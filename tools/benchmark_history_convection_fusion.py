from __future__ import annotations

import argparse
import gc
import hashlib
import os
import time
from pathlib import Path

import numpy as np

from _scaling_support import positive_int
from benchmark_convection_scaling import _build_synthetic_convection_inputs
from wombat_transport.history_accumulation import accumulate_history_sum
from wombat_transport.transport.convection import G0_100
from wombat_transport.transport.convection import _operator as convection_operator
from wombat_transport.transport.numba_control import configure_numba_threads

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover - benchmark requires Numba.
    njit = None
    prange = range


DEFAULT_CONFIG = Path(
    "validation_runs/cases/realistic_restart_noemis_2x25/wombat/main/run.yml"
)


if njit is not None:
    _convect_block_serial = convection_operator._convect_block_serial
    _convect_block_serial_history = convection_operator._convect_block_serial_history

    @njit(parallel=True, nogil=True)
    def _run_block_convection(
        state,
        diag,
        cmfmc,
        dtrain,
        delp_hpa,
        delp_dry,
        bmass,
        dqrcu,
        reevapcn,
        area,
        reconstruct_conv_precip_flux,
        internal_steps,
        internal_dt_s,
        qc,
        qb_num,
        delq_work,
        current_work,
    ):
        nlev = state.shape[1]
        ncol = cmfmc.shape[1]
        lane = state.shape[-1]
        for block in prange(state.shape[0]):
            _convect_block_serial(
                state[block].reshape(nlev, ncol, lane),
                diag,
                cmfmc,
                dtrain,
                delp_hpa,
                delp_dry,
                bmass,
                dqrcu,
                reevapcn,
                area,
                reconstruct_conv_precip_flux,
                internal_steps,
                internal_dt_s,
                qc,
                qb_num,
                delq_work,
                current_work,
            )

    @njit(parallel=True, nogil=True)
    def _run_block_convection_with_history(
        state,
        history_sum,
        diag,
        cmfmc,
        dtrain,
        delp_hpa,
        delp_dry,
        bmass,
        dqrcu,
        reevapcn,
        area,
        reconstruct_conv_precip_flux,
        internal_steps,
        internal_dt_s,
        qc,
        qb_num,
        delq_work,
        current_work,
    ):
        nlev = state.shape[1]
        ncol = cmfmc.shape[1]
        lane = state.shape[-1]
        for block in prange(state.shape[0]):
            _convect_block_serial_history(
                state[block].reshape(nlev, ncol, lane),
                history_sum[block].reshape(nlev, ncol, lane),
                diag,
                cmfmc,
                dtrain,
                delp_hpa,
                delp_dry,
                bmass,
                dqrcu,
                reevapcn,
                area,
                reconstruct_conv_precip_flux,
                internal_steps,
                internal_dt_s,
                qc,
                qb_num,
                delq_work,
                current_work,
            )

else:
    _run_block_convection = None
    _run_block_convection_with_history = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare separate and column-hot HISTORY accumulation."
    )
    parser.add_argument("--run-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--counts", type=positive_int, nargs="+", default=[128, 512])
    parser.add_argument("--lane-width", type=positive_int, default=16)
    parser.add_argument("--workers", type=positive_int, default=8)
    parser.add_argument("--warmup", type=positive_int, default=2)
    parser.add_argument("--repeat", type=positive_int, default=8)
    args = parser.parse_args(argv)
    if _run_block_convection is None or _run_block_convection_with_history is None:
        raise RuntimeError("Numba is required")

    os.environ["WOMBAT_NUMBA_THREADS"] = str(args.workers)
    workers = configure_numba_threads(available=True)
    if workers != args.workers:
        raise RuntimeError(f"configured {workers} Numba threads, expected {args.workers}")

    print(
        "tracers,separate_best_s,separate_mean_s,fused_best_s,fused_mean_s,"
        "best_speedup,mean_speedup,array_equal"
    )
    for tracer_count in args.counts:
        result = _benchmark_count(
            args.run_config,
            tracer_count=tracer_count,
            lane_width=args.lane_width,
            workers=workers,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        print(
            f"{tracer_count},{result[0]:.9f},{result[1]:.9f},"
            f"{result[2]:.9f},{result[3]:.9f},"
            f"{result[0] / result[2]:.6f},{result[1] / result[3]:.6f},"
            f"{str(result[4]).lower()}"
        )
        gc.collect()
    return 0


def _benchmark_count(
    run_config: Path,
    *,
    tracer_count: int,
    lane_width: int,
    workers: int,
    warmup: int,
    repeat: int,
) -> tuple[float, float, float, float, bool]:
    inputs = _build_synthetic_convection_inputs(run_config, tracer_count, dt_s=600.0)
    nlev, nlat, nlon, _ = inputs.tracer_conc.shape
    ncol = nlat * nlon
    nblock = (tracer_count + lane_width - 1) // lane_width
    initial = np.zeros((nblock, nlev, ncol, lane_width), dtype=np.float64)
    canonical = inputs.tracer_conc.reshape(nlev, ncol, tracer_count)
    for block in range(nblock):
        start = block * lane_width
        stop = min(start + lane_width, tracer_count)
        initial[block, :, :, : stop - start] = canonical[:, :, start:stop]

    cmfmc = np.ascontiguousarray(inputs.cmfmc_kg_m2_s.reshape(nlev, ncol))
    dtrain = np.ascontiguousarray(inputs.dtrain_kg_m2_s.reshape(nlev, ncol))
    delp_hpa = np.ascontiguousarray(inputs.delp_hpa.reshape(nlev, ncol))
    delp_dry = np.ascontiguousarray(inputs.delp_dry_hpa.reshape(nlev, ncol))
    bmass = np.ascontiguousarray(delp_dry * G0_100)
    dqrcu = np.ascontiguousarray(inputs.dqrcu_kg_kg_s.reshape(nlev, ncol))
    reevapcn = np.ascontiguousarray(inputs.reevapcn_kg_kg_s.reshape(nlev, ncol))
    area = np.ascontiguousarray(inputs.area_m2.reshape(ncol))
    reconstruct = inputs.reconstruct_conv_precip_flux
    internal_steps = 2
    internal_dt_s = 300.0
    del inputs, canonical
    gc.collect()

    state = np.empty_like(initial)
    history_sum = np.empty_like(initial)
    history_initial = 0.125
    diag = np.empty((0, 0, 0), dtype=np.float64)
    stride = max(lane_width, convection_operator._CONVECTION_SCRATCH_PAD_TRACERS)
    qc = np.empty((workers, stride), dtype=np.float64)
    qb_num = np.empty((workers, stride), dtype=np.float64)
    delq_work = np.empty((workers, stride), dtype=np.float64)
    current_work = np.empty((workers, stride), dtype=np.float64)
    kernel_args = (
        diag,
        cmfmc,
        dtrain,
        delp_hpa,
        delp_dry,
        bmass,
        dqrcu,
        reevapcn,
        area,
        reconstruct,
        internal_steps,
        internal_dt_s,
        qc,
        qb_num,
        delq_work,
        current_work,
    )

    def separate() -> None:
        _run_block_convection(state, *kernel_args)
        accumulate_history_sum(history_sum, state)

    def fused() -> None:
        _run_block_convection_with_history(state, history_sum, *kernel_args)

    for function in (separate, fused):
        for _ in range(warmup):
            np.copyto(state, initial)
            history_sum.fill(history_initial)
            function()

    separate_times: list[float] = []
    fused_times: list[float] = []
    for index in range(repeat):
        order = (
            ((separate, separate_times), (fused, fused_times))
            if index % 2 == 0
            else ((fused, fused_times), (separate, separate_times))
        )
        for function, elapsed_values in order:
            np.copyto(state, initial)
            history_sum.fill(history_initial)
            start = time.perf_counter()
            function()
            elapsed_values.append(time.perf_counter() - start)

    np.copyto(state, initial)
    history_sum.fill(history_initial)
    separate()
    separate_digest = _digest_arrays(state, history_sum)
    np.copyto(state, initial)
    history_sum.fill(history_initial)
    fused()
    fused_digest = _digest_arrays(state, history_sum)

    return (
        min(separate_times),
        sum(separate_times) / len(separate_times),
        min(fused_times),
        sum(fused_times) / len(fused_times),
        separate_digest == fused_digest,
    )


def _digest_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
