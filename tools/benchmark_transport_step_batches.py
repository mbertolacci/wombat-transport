"""Benchmark several transport steps inside one block-parallel launch.

This is deliberately an experimental, kernel-only benchmark.  It captures the
fully prepared arguments to the production block kernel, copies met-dependent
inputs into distinct time-indexed storage, then compares K ordinary launches
with one launch in which every block advances K times.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np
from numba import njit, prange

from wombat_transport.history_accumulation import accumulate_history_sum
from wombat_transport.transport._executor import _one_block_transport_step_serial


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_RUN_CONFIG = Path(
    "validation_runs/cases/realistic_restart_noemis_2x25/wombat/main/run.yml"
)
DEFAULT_CASES = ((96, 12), (128, 16), (192, 12), (96, 8), (120, 8))
DEFAULT_BATCH_SIZES = (1, 2, 3, 4, 6)
TIME_INDEXED_ARGUMENTS = (
    *range(2, 10),
    *range(12, 16),
    *range(42, 46),
    *range(47, 53),
    55,
    56,
    *range(62, 69),
)


@njit(parallel=True, nogil=True, cache=True)
def _multi_step_block_kernel(
    state_a,
    state_b,
    delp1,
    delp2,
    pu,
    xmass,
    ymass,
    wz,
    cx,
    cy,
    geofac,
    geofac_pc,
    ua,
    va,
    jn,
    js,
    area_1d,
    fill,
    qqu,
    qqv,
    dcx,
    fx,
    al_x,
    ar_x,
    a6_x,
    dc_x,
    qa_x,
    dcy,
    al_y,
    ar_y,
    a6_y,
    south_flux_y,
    north_flux_y,
    south_dao2_y,
    north_dao2_y,
    dpi_z,
    dc_z,
    al_z,
    ar_z,
    a6_z,
    dca_z,
    prev_flux_z,
    cch,
    zeh,
    termh,
    dry_mass,
    area_m2,
    cgs,
    kvh,
    potbar,
    rpdel,
    rrho,
    tmp1,
    dt_s,
    start_level,
    surface_flux,
    has_flux,
    tracer_diffused,
    before_mass,
    after_mass,
    qmx,
    adjust,
    cmfmc,
    dtrain,
    delp_hpa,
    delp_dry,
    bmass,
    dqrcu,
    reevapcn,
    reconstruct_conv_precip_flux,
    internal_steps,
    internal_dt_s,
    convection_qc,
    convection_diag_empty,
    convection_qb_num,
    convection_delq_work,
    convection_current_work,
    negative_counts,
    history_sum,
    history_collections,
    batch_size,
):
    tpcore_worker_work = (
        dcx, fx, al_x, ar_x, a6_x, dc_x, qa_x,
        dcy, al_y, ar_y, a6_y,
        dpi_z, dc_z, al_z, ar_z, a6_z, dca_z, prev_flux_z,
    )
    vdiff_worker_work = (tracer_diffused, before_mass, after_mass, qmx, adjust)
    convection_work = (
        convection_qc, convection_qb_num, convection_delq_work,
        convection_current_work,
    )
    for block in prange(state_a.shape[0]):
        tpcore_block_work = (
            qqu[block], qqv[block], south_flux_y[block], north_flux_y[block],
            south_dao2_y[block], north_dao2_y[block],
        )
        negative_count = 0
        for step in range(batch_size):
            tpcore_plan = (
                delp1[step], delp2[step], pu[step], xmass[step], ymass[step],
                wz[step], cx[step], cy[step], geofac, geofac_pc,
                ua[step], va[step], jn[step], js[step], area_1d, fill,
            )
            vdiff_plan = (
                cch[step], zeh[step], termh[step], dry_mass[step], area_m2,
                cgs[step], kvh[step], potbar[step], rpdel[step], rrho[step],
                tmp1[step], dt_s, start_level,
            )
            convection_inputs = (
                convection_diag_empty, cmfmc[step], dtrain[step],
                delp_hpa[step], delp_dry[step], bmass[step], dqrcu[step],
                reevapcn[step], area_m2.reshape(area_m2.size), False,
                reconstruct_conv_precip_flux, internal_steps, internal_dt_s,
            )
            negative_count += _one_block_transport_step_serial(
                state_a[block], state_b[block], tpcore_plan, tpcore_block_work,
                tpcore_worker_work, vdiff_plan, surface_flux[step, block],
                has_flux[step],
                vdiff_worker_work, convection_inputs, convection_work,
            )
            for collection in range(history_collections):
                values = state_a[block].reshape(state_a[block].size)
                accumulator = history_sum[collection, block].reshape(values.size)
                for index in range(values.size):
                    accumulator[index] += values[index]
        negative_counts[block] = negative_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, default=DEFAULT_RUN_CONFIG)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[f"{tracers}:{width}" for tracers, width in DEFAULT_CASES],
        help="tracer-count:block-width pairs",
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=list(DEFAULT_BATCH_SIZES)
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=9)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--history-collections", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeat < 1:
        parser.error("warmup must be nonnegative and repeat must be positive")
    if any(value < 1 for value in args.batch_sizes):
        parser.error("batch sizes must be positive")
    if args.history_collections < 0:
        parser.error("history collections must be nonnegative")

    cases = [_parse_case(value) for value in args.cases]
    rows = []
    for tracer_count, block_width in cases:
        rows.extend(
            _benchmark_case(
                args.run_config,
                tracer_count,
                block_width,
                args.batch_sizes,
                warmup=args.warmup,
                repeat=args.repeat,
                seed=args.seed,
                history_collections=args.history_collections,
            )
        )
    _write_rows(rows, args.output)
    return 0


def _benchmark_case(
    run_config: Path,
    tracer_count: int,
    block_width: int,
    batch_sizes: list[int],
    *,
    warmup: int,
    repeat: int,
    seed: int,
    history_collections: int,
) -> list[dict[str, object]]:
    tools_dir = str(SCRIPT_PATH.parent)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    from benchmark_transport_driver_scaling import _build_synthetic_driver_inputs
    from wombat_transport.transport import _executor as executor_mod
    from wombat_transport.transport._executor import TransportExecutor
    from wombat_transport.transport.driver import (
        build_tpcore_static_terms,
        run_transport_step_with_executor,
    )

    inputs = _build_synthetic_driver_inputs(run_config, tracer_count, dt_s=600.0)
    state = inputs.tracer_field.reblock(block_width)
    executor = TransportExecutor.create(state)
    static_terms = build_tpcore_static_terms(
        area_m2=inputs.grid.area_m2,
        hyai_hpa=inputs.grid.hyai_hpa,
        hybi=inputs.grid.hybi,
        lat_deg=inputs.grid.lat_deg,
    )
    captured: list[tuple[object, ...]] = []

    def capture(*kernel_args: object) -> None:
        negative_counts = kernel_args[-1]
        assert isinstance(negative_counts, np.ndarray)
        negative_counts.fill(0)
        captured.append(kernel_args)

    production_kernel = executor_mod._multi_block_transport_step_parallel
    executor_mod._multi_block_transport_step_parallel = capture
    try:
        run_transport_step_with_executor(
            state,
            inputs.forcing,
            inputs.grid,
            executor,
            dt_s=inputs.dt_s,
            tpcore_static_terms=static_terms,
            validate_tpcore_branches=False,
            execution="blocks",
        )
    finally:
        executor_mod._multi_block_transport_step_parallel = production_kernel
    if len(captured) != 1 or production_kernel is None:
        raise RuntimeError("failed to capture the prepared production kernel call")
    kernel_args = captured[0]
    initial_state = np.array(executor.workspace.tpcore.state_a, copy=True)
    history_sum = np.zeros(
        (history_collections, *executor.workspace.tpcore.state_a.shape),
        dtype=np.float64,
    )

    production_kernel(*kernel_args)
    executor.workspace.tpcore.state_a[...] = initial_state
    indexed_one = _make_time_indexed_args(kernel_args, 1)
    _multi_step_block_kernel(
        *indexed_one, history_sum, history_collections, 1
    )
    history_sum.fill(0.0)
    executor.workspace.tpcore.state_a[...] = initial_state
    production_kernel(*kernel_args)
    executor.workspace.tpcore.state_a[...] = initial_state
    initial_scratch_state = np.array(executor.workspace.tpcore.state_b, copy=True)
    scratch_snapshot = _snapshot_scratch(executor.workspace)
    _reset_state(executor, initial_state, initial_scratch_state, scratch_snapshot)

    rows = []
    for batch_size in batch_sizes:
        indexed_args = _make_time_indexed_args(kernel_args, batch_size)
        step_args = _make_step_args(indexed_args, batch_size)
        _reset_state(executor, initial_state, initial_scratch_state, scratch_snapshot)
        history_sum.fill(0.0)
        for _ in range(warmup):
            _run_sequential(production_kernel, step_args, history_sum)
            _reset_state(
                executor, initial_state, initial_scratch_state, scratch_snapshot
            )
            history_sum.fill(0.0)
            _multi_step_block_kernel(
                *indexed_args, history_sum, history_collections, batch_size
            )
            _reset_state(
                executor, initial_state, initial_scratch_state, scratch_snapshot
            )
            history_sum.fill(0.0)

        sequential_negative = _run_sequential(
            production_kernel, step_args, history_sum
        )
        sequential_result = np.array(executor.workspace.tpcore.state_a, copy=True)
        sequential_history = np.array(history_sum, copy=True)
        _reset_state(executor, initial_state, initial_scratch_state, scratch_snapshot)
        history_sum.fill(0.0)
        _multi_step_block_kernel(
            *indexed_args, history_sum, history_collections, batch_size
        )
        batched_negative = int(np.sum(executor.workspace.negative_counts))
        exact = bool(np.array_equal(executor.workspace.tpcore.state_a, sequential_result))
        history_exact = bool(np.array_equal(history_sum, sequential_history))
        max_abs_diff = float(
            np.max(np.abs(executor.workspace.tpcore.state_a - sequential_result))
        )
        if not exact or not history_exact or batched_negative != sequential_negative:
            batched_result = np.array(executor.workspace.tpcore.state_a, copy=True)
            _reset_state(
                executor, initial_state, initial_scratch_state, scratch_snapshot
            )
            production_kernel(*kernel_args)
            max_abs_diff_from_one_step = float(
                np.max(np.abs(batched_result - executor.workspace.tpcore.state_a))
            )
            raise AssertionError(
                f"batch {batch_size} differs: exact={exact}, "
                f"history_exact={history_exact}, "
                f"max_abs_diff={max_abs_diff}, negatives "
                f"{batched_negative} != {sequential_negative}, "
                f"difference from one step={max_abs_diff_from_one_step}"
            )

        sequential_times = []
        batched_times = []
        order = ["sequential", "batched"] * repeat
        random.Random(seed + tracer_count * 100 + block_width * 10 + batch_size).shuffle(order)
        for strategy in order:
            _reset_state(
                executor, initial_state, initial_scratch_state, scratch_snapshot
            )
            history_sum.fill(0.0)
            start = time.perf_counter()
            if strategy == "sequential":
                _run_sequential(production_kernel, step_args, history_sum)
            else:
                _multi_step_block_kernel(
                    *indexed_args, history_sum, history_collections, batch_size
                )
            elapsed = time.perf_counter() - start
            if strategy == "sequential":
                sequential_times.append(elapsed)
            else:
                batched_times.append(elapsed)

        sequential_median = statistics.median(sequential_times)
        batched_median = statistics.median(batched_times)
        row = {
            "tracer_count": tracer_count,
            "block_width": block_width,
            "block_count": executor.workspace.tpcore.state_a.shape[0],
            "threads": executor.workspace.workers,
            "batch_size": batch_size,
            "plan_storage": "time_indexed",
            "history_collections": history_collections,
            "repeat": repeat,
            "sequential_median_s": sequential_median,
            "batched_median_s": batched_median,
            "sequential_s_per_step": sequential_median / batch_size,
            "batched_s_per_step": batched_median / batch_size,
            "speedup": sequential_median / batched_median,
            "saving_percent": 100.0 * (1.0 - batched_median / sequential_median),
            "exact": exact,
            "history_exact": history_exact,
            "max_abs_diff": max_abs_diff,
            "negative_count": batched_negative,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        del sequential_history, sequential_result
    return rows


def _make_time_indexed_args(
    kernel_args: tuple[object, ...], batch_size: int
) -> tuple[object, ...]:
    indexed = list(kernel_args)
    for position in TIME_INDEXED_ARGUMENTS:
        value = kernel_args[position]
        if position == 56:
            indexed[position] = np.full(batch_size, bool(value), dtype=np.bool_)
        else:
            assert isinstance(value, np.ndarray)
            indexed[position] = np.repeat(value[np.newaxis, ...], batch_size, axis=0)
    return tuple(indexed)


def _make_step_args(
    indexed_args: tuple[object, ...], batch_size: int
) -> list[tuple[object, ...]]:
    result = []
    for step in range(batch_size):
        values = list(indexed_args)
        for position in TIME_INDEXED_ARGUMENTS:
            indexed = indexed_args[position]
            assert isinstance(indexed, np.ndarray)
            value = indexed[step]
            values[position] = bool(value) if position == 56 else value
        result.append(tuple(values))
    return result


def _run_sequential(
    kernel,
    step_args: list[tuple[object, ...]],
    history_sum: np.ndarray,
) -> int:
    negative_count = 0
    for kernel_args in step_args:
        kernel(*kernel_args)
        for accumulator in history_sum:
            state_a = kernel_args[0]
            assert isinstance(state_a, np.ndarray)
            accumulate_history_sum(accumulator, state_a)
        negative_counts = kernel_args[-1]
        assert isinstance(negative_counts, np.ndarray)
        negative_count += int(np.sum(negative_counts))
    return negative_count


def _snapshot_scratch(workspace) -> list[tuple[np.ndarray, np.ndarray]]:
    snapshot = []
    for value in vars(workspace).values():
        arrays = value if isinstance(value, tuple) else (value,)
        for array in arrays:
            if isinstance(array, np.ndarray):
                snapshot.append((array, np.array(array, copy=True)))
    return snapshot


def _reset_state(
    executor,
    state_a: np.ndarray,
    state_b: np.ndarray,
    scratch_snapshot: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    executor.workspace.tpcore.state_a[...] = state_a
    executor.workspace.tpcore.state_b[...] = state_b
    for target, source in scratch_snapshot:
        target[...] = source


def _parse_case(value: str) -> tuple[int, int]:
    try:
        tracer_text, width_text = value.split(":", maxsplit=1)
        tracers = int(tracer_text)
        width = int(width_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid case {value!r}; expected TRACERS:WIDTH") from exc
    if tracers < 1 or width < 1:
        raise argparse.ArgumentTypeError("case values must be positive")
    return tracers, width


def _write_rows(rows: list[dict[str, object]], output: Path | None) -> None:
    if not rows:
        return
    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
