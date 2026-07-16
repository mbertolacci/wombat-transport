from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _scaling_support import positive_int, write_rows
from benchmark_tpcore_scaling import _build_synthetic_tpcore_inputs
from wombat_transport.transport.tpcore import run_tpcore_one_step_with_setup
from wombat_transport.transport.tpcore import setup_tpcore_terms
from wombat_transport.transport.tpcore._block_experiment import advect_tracer_blocks
from wombat_transport.transport.tpcore._block_experiment import apply_tpcore_block_workspace
from wombat_transport.transport.tpcore._block_experiment import load_tracer_block_workspace
from wombat_transport.transport.tpcore._block_experiment import make_tpcore_block_workspace
from wombat_transport.transport.tpcore._block_experiment import prepare_tpcore_block_plan
from wombat_transport.transport.tpcore._block_experiment import unpack_tpcore_block_workspace


CSV_FIELDS = (
    "tracer_count",
    "mode",
    "lane_width",
    "workers",
    "best_apply_s",
    "mean_apply_s",
    "plan_s",
    "best_total_s",
    "tracers_per_second",
    "speedup_vs_fused",
    "array_equal",
    "max_abs_error",
    "checksum",
)


@dataclass(frozen=True)
class BenchmarkRow:
    values: dict[str, object]

    def as_csv_row(self) -> dict[str, object]:
        return self.values


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows: list[dict[str, object]] = []
    for tracer_count in args.counts:
        inputs = _build_synthetic_tpcore_inputs(args.run_config, tracer_count, dt_s=args.dt_s)
        setup = setup_tpcore_terms(
            p1_hpa=inputs.p1_hpa,
            p2_hpa=inputs.p2_hpa,
            u_m_s=inputs.u_m_s,
            v_m_s=inputs.v_m_s,
            area_m2=inputs.area_m2,
            hyai_hpa=inputs.hyai_hpa,
            hybi=inputs.hybi,
            lat_deg=inputs.lat_deg,
            dt_s=inputs.dt_s,
        )

        def run_fused() -> np.ndarray:
            return run_tpcore_one_step_with_setup(
                tracer_conc=inputs.tracer_conc,
                setup=setup,
                area_m2=inputs.area_m2,
                validate_branches=False,
            ).tracer_conc_after

        fused_times, reference = _time_call(run_fused, warmup=args.warmup, repeat=args.repeat)
        fused_best = min(fused_times)
        rows.append(
            _row(
                tracer_count=tracer_count,
                mode="fused",
                lane_width=0,
                workers=args.workers,
                times=fused_times,
                plan_s=0.0,
                fused_best=fused_best,
                actual=reference,
                reference=reference,
            )
        )

        for lane_width in args.lanes:
            plan_start = time.perf_counter()
            plan = prepare_tpcore_block_plan(setup=setup, area_m2=inputs.area_m2)
            plan_s = time.perf_counter() - plan_start
            workspace = make_tpcore_block_workspace(inputs.tracer_conc.shape, lane_width)

            def run_blocked() -> np.ndarray:
                return advect_tracer_blocks(
                    tracer_conc=inputs.tracer_conc,
                    plan=plan,
                    lane_width=lane_width,
                    workers=args.workers,
                    workspace=workspace,
                )

            blocked_times, actual = _time_call(run_blocked, warmup=args.warmup, repeat=args.repeat)
            rows.append(
                _row(
                    tracer_count=tracer_count,
                    mode="blocked",
                    lane_width=lane_width,
                    workers=args.workers,
                    times=blocked_times,
                    plan_s=plan_s,
                    fused_best=fused_best,
                    actual=actual,
                    reference=reference,
                )
            )

            def load_blocks() -> None:
                load_tracer_block_workspace(inputs.tracer_conc, workspace)

            def apply_blocks() -> np.ndarray:
                apply_tpcore_block_workspace(plan=plan, workspace=workspace, workers=args.workers)
                return workspace.blocks[0].dq1

            apply_times, _ = _time_preloaded_call(
                load_blocks, apply_blocks, warmup=args.warmup, repeat=args.repeat
            )
            actual = unpack_tpcore_block_workspace(workspace)
            rows.append(
                _row(
                    tracer_count=tracer_count,
                    mode="block_apply",
                    lane_width=lane_width,
                    workers=args.workers,
                    times=apply_times,
                    plan_s=plan_s,
                    fused_best=fused_best,
                    actual=actual,
                    reference=reference,
                )
            )
        del inputs, setup, reference
        gc.collect()

    write_rows(rows, CSV_FIELDS, args.output)
    return 0


def _time_call(call, *, warmup: int, repeat: int) -> tuple[list[float], np.ndarray]:
    result = None
    for _ in range(warmup):
        result = call()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = call()
        times.append(time.perf_counter() - start)
    if result is None:
        raise AssertionError("benchmark produced no result")
    return times, result


def _time_preloaded_call(load, call, *, warmup: int, repeat: int) -> tuple[list[float], np.ndarray]:
    result = None
    for _ in range(warmup):
        load()
        result = call()
    times = []
    for _ in range(repeat):
        load()
        start = time.perf_counter()
        result = call()
        times.append(time.perf_counter() - start)
    if result is None:
        raise AssertionError("benchmark produced no result")
    return times, result


def _row(
    *,
    tracer_count: int,
    mode: str,
    lane_width: int,
    workers: int,
    times: list[float],
    plan_s: float,
    fused_best: float,
    actual: np.ndarray,
    reference: np.ndarray,
) -> BenchmarkRow:
    best = min(times)
    total = best + plan_s
    max_abs_error = float(np.max(np.abs(actual - reference)))
    return BenchmarkRow({
        "tracer_count": tracer_count,
        "mode": mode,
        "lane_width": lane_width,
        "workers": workers,
        "best_apply_s": f"{best:.9f}",
        "mean_apply_s": f"{np.mean(times):.9f}",
        "plan_s": f"{plan_s:.9f}",
        "best_total_s": f"{total:.9f}",
        "tracers_per_second": f"{tracer_count / total:.6f}",
        "speedup_vs_fused": f"{fused_best / total:.6f}",
        "array_equal": str(bool(np.array_equal(actual, reference))).lower(),
        "max_abs_error": f"{max_abs_error:.16g}",
        "checksum": f"{np.mean(actual[0, 0, 0, :]):.16g}",
    })


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the experimental TPCORE tracer-block executor.")
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--counts", type=positive_int, nargs="+", default=[24, 96, 192])
    parser.add_argument("--lanes", type=positive_int, nargs="+", default=[8, 16])
    parser.add_argument("--workers", type=positive_int, default=8)
    parser.add_argument("--repeat", type=positive_int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dt-s", type=float, default=600.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.dt_s <= 0.0:
        parser.error("--dt-s must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
