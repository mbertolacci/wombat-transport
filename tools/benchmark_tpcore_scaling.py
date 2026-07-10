from __future__ import annotations

import argparse
import cProfile
import csv
import gc
import io
import os
import pstats
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np

from wombat_transport.grid import load_transport_grid
from wombat_transport.run_config import load_run_config
from wombat_transport.transport.tpcore import run_tpcore_one_step


DEFAULT_COUNTS = (1, 24, 96, 256, 512)
DEFAULT_DT_S = 600.0
DEFAULT_WORKING_SET_MULTIPLIER = 12.0
DEFAULT_FIXED_OVERHEAD_BYTES = 512 * 1024**2
AUTO_MEMORY_FRACTION = 0.55
CSV_FIELDS = (
    "tracer_count",
    "status",
    "repeat",
    "best_wall_s",
    "mean_wall_s",
    "seconds_per_tracer",
    "tracers_per_second",
    "gridcell_tracers_per_second",
    "tracer_state_mib",
    "estimated_peak_mib",
    "memory_limit_mib",
    "peak_rss_mib",
    "checksum",
    "reason",
)


@dataclass(frozen=True)
class SyntheticTpcoreInputs:
    tracer_conc: np.ndarray
    p1_hpa: np.ndarray
    p2_hpa: np.ndarray
    u_m_s: np.ndarray
    v_m_s: np.ndarray
    area_m2: np.ndarray
    hyai_hpa: np.ndarray
    hybi: np.ndarray
    lat_deg: np.ndarray
    dt_s: float


@dataclass(frozen=True)
class BenchmarkRow:
    tracer_count: int
    status: str
    repeat: int
    best_wall_s: float | None
    mean_wall_s: float | None
    seconds_per_tracer: float | None
    tracers_per_second: float | None
    gridcell_tracers_per_second: float | None
    tracer_state_mib: float
    estimated_peak_mib: float
    memory_limit_mib: float | None
    peak_rss_mib: float | None
    checksum: float | None
    reason: str

    def as_csv_row(self) -> dict[str, str]:
        return {
            "tracer_count": str(self.tracer_count),
            "status": self.status,
            "repeat": str(self.repeat),
            "best_wall_s": _format_optional(self.best_wall_s),
            "mean_wall_s": _format_optional(self.mean_wall_s),
            "seconds_per_tracer": _format_optional(self.seconds_per_tracer),
            "tracers_per_second": _format_optional(self.tracers_per_second),
            "gridcell_tracers_per_second": _format_optional(self.gridcell_tracers_per_second),
            "tracer_state_mib": f"{self.tracer_state_mib:.3f}",
            "estimated_peak_mib": f"{self.estimated_peak_mib:.3f}",
            "memory_limit_mib": _format_optional(self.memory_limit_mib, precision=3),
            "peak_rss_mib": _format_optional(self.peak_rss_mib, precision=3),
            "checksum": _format_optional(self.checksum, precision=16),
            "reason": self.reason,
        }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    grid_shape = _read_fullgrid_shape(args.run_config)
    memory_limit = _memory_limit_bytes(args.max_memory_gb)
    rows: list[BenchmarkRow] = []

    for tracer_count in args.counts:
        state_bytes = _tracer_state_bytes(tracer_count, grid_shape)
        peak_bytes = _estimate_peak_bytes(tracer_count, grid_shape)
        allowed, reason = _count_is_allowed(peak_bytes, memory_limit)
        if not allowed:
            rows.append(
                BenchmarkRow(
                    tracer_count=tracer_count,
                    status="skipped",
                    repeat=args.repeat,
                    best_wall_s=None,
                    mean_wall_s=None,
                    seconds_per_tracer=None,
                    tracers_per_second=None,
                    gridcell_tracers_per_second=None,
                    tracer_state_mib=_bytes_to_mib(state_bytes),
                    estimated_peak_mib=_bytes_to_mib(peak_bytes),
                    memory_limit_mib=_bytes_to_mib(memory_limit) if memory_limit is not None else None,
                    peak_rss_mib=_peak_rss_mib(),
                    checksum=None,
                    reason=reason,
                )
            )
            continue

        inputs = _build_synthetic_tpcore_inputs(args.run_config, tracer_count, dt_s=args.dt_s)
        rows.append(
            _benchmark_inputs(
                inputs,
                tracer_count=tracer_count,
                repeat=args.repeat,
                warmup=args.warmup,
                state_bytes=state_bytes,
                peak_bytes=peak_bytes,
                memory_limit=memory_limit,
            )
        )
        if args.profile:
            profile_text = _profile_inputs(inputs, profile_top=args.profile_top)
            print(f"\n# cProfile top {args.profile_top} for {tracer_count} tracers", file=sys.stderr)
            print(profile_text, file=sys.stderr, end="")
        del inputs
        gc.collect()

    _write_rows(rows, args.output)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark full-grid synthetic TPCORE scaling by tracer count.")
    parser.add_argument(
        "--run-config",
        type=Path,
        default=Path("base_wombat/run.yml"),
        help="Run config used only to locate the grid template. Defaults to base_wombat/run.yml.",
    )
    parser.add_argument("--counts", type=_positive_int, nargs="+", default=list(DEFAULT_COUNTS))
    parser.add_argument("--repeat", type=_positive_int, default=1)
    parser.add_argument(
        "--warmup",
        type=_nonnegative_int,
        default=1,
        help="Untimed runs per tracer count before measurement. Defaults to 1 to exclude Numba compilation.",
    )
    parser.add_argument("--dt-s", type=float, default=DEFAULT_DT_S)
    parser.add_argument(
        "--max-memory-gb",
        default="auto",
        help="Memory budget in GB, or 'auto' for a conservative fraction of physical memory.",
    )
    parser.add_argument("--profile", action="store_true", help="Run one additional cProfile pass per completed count.")
    parser.add_argument("--profile-top", type=_positive_int, default=20)
    parser.add_argument("--output", type=Path, help="Optional CSV output path. Defaults to stdout.")
    args = parser.parse_args(argv)
    if args.dt_s <= 0.0:
        parser.error("--dt-s must be positive")
    return args


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _read_fullgrid_shape(run_config_path: Path) -> tuple[int, int, int]:
    config = load_run_config(run_config_path)
    return load_transport_grid(config.grid_template).shape


def _build_synthetic_tpcore_inputs(run_config_path: Path, ntracer: int, *, dt_s: float) -> SyntheticTpcoreInputs:
    config = load_run_config(run_config_path)
    grid = load_transport_grid(config.grid_template)
    lat = grid.lat_deg
    lon = grid.lon_deg
    hyai = grid.hyai_hpa
    hybi = grid.hybi
    area = grid.area_m2

    nlev = hyai.size - 1
    level = np.arange(nlev, dtype=np.float64)[:, np.newaxis, np.newaxis]
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lat_2d = lat_rad[:, np.newaxis]
    lon_2d = lon_rad[np.newaxis, :]

    p1 = 965.0 + 22.0 * np.cos(lat_2d) ** 2
    p1 = p1 + 2.0 * np.sin(lon_2d) * np.cos(lat_2d)
    p2 = p1 + 0.25 * np.cos(2.0 * lon_2d) * np.cos(lat_2d)

    lat_3d = lat_rad[np.newaxis, :, np.newaxis]
    lon_3d = lon_rad[np.newaxis, np.newaxis, :]
    vertical_wave = np.sin((level + 1.0) / float(nlev) * np.pi)
    u = 5.0 * vertical_wave * np.cos(lat_3d)
    u = u * (1.0 + 0.15 * np.cos(lon_3d))
    v = 0.35 * np.cos((level + 1.0) / float(nlev) * np.pi) * np.sin(2.0 * lon_3d)
    v = v * np.cos(lat_3d)

    tracer = np.empty((nlev, lat.size, lon.size, ntracer), dtype=np.float64)
    tracer[:] = 4.0e-4
    tracer += (np.arange(ntracer, dtype=np.float64)[np.newaxis, np.newaxis, np.newaxis, :] + 1.0) * 1.0e-7
    tracer += 2.5e-8 * np.arange(nlev, dtype=np.float64)[:, np.newaxis, np.newaxis, np.newaxis] / max(
        float(nlev - 1), 1.0
    )
    tracer += 1.5e-8 * np.sin(lat_rad)[np.newaxis, :, np.newaxis, np.newaxis]
    tracer += 7.5e-9 * np.cos(lon_rad)[np.newaxis, np.newaxis, :, np.newaxis]

    return SyntheticTpcoreInputs(
        tracer_conc=tracer,
        p1_hpa=p1,
        p2_hpa=p2,
        u_m_s=u,
        v_m_s=v,
        area_m2=area,
        hyai_hpa=hyai,
        hybi=hybi,
        lat_deg=lat,
        dt_s=dt_s,
    )


def _benchmark_inputs(
    inputs: SyntheticTpcoreInputs,
    *,
    tracer_count: int,
    repeat: int,
    warmup: int,
    state_bytes: int,
    peak_bytes: int,
    memory_limit: int | None,
) -> BenchmarkRow:
    for _ in range(warmup):
        state = run_tpcore_one_step(
            tracer_conc=inputs.tracer_conc,
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
        del state
        gc.collect()

    elapsed_values: list[float] = []
    checksum = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        state = run_tpcore_one_step(
            tracer_conc=inputs.tracer_conc,
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
        elapsed_values.append(time.perf_counter() - start)
        checksum = float(np.mean(state.tracer_conc_after[0, 0, 0, :]))
        del state
        gc.collect()

    best = min(elapsed_values)
    mean = sum(elapsed_values) / len(elapsed_values)
    gridcell_tracers = int(np.prod(inputs.tracer_conc.shape))
    return BenchmarkRow(
        tracer_count=tracer_count,
        status="completed",
        repeat=repeat,
        best_wall_s=best,
        mean_wall_s=mean,
        seconds_per_tracer=best / float(tracer_count),
        tracers_per_second=float(tracer_count) / best,
        gridcell_tracers_per_second=float(gridcell_tracers) / best,
        tracer_state_mib=_bytes_to_mib(state_bytes),
        estimated_peak_mib=_bytes_to_mib(peak_bytes),
        memory_limit_mib=_bytes_to_mib(memory_limit) if memory_limit is not None else None,
        peak_rss_mib=_peak_rss_mib(),
        checksum=checksum,
        reason="",
    )


def _profile_inputs(inputs: SyntheticTpcoreInputs, *, profile_top: int) -> str:
    profiler = cProfile.Profile()
    profiler.enable()
    state = run_tpcore_one_step(
        tracer_conc=inputs.tracer_conc,
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
    profiler.disable()
    checksum = float(np.mean(state.tracer_conc_after[0, 0, 0, :]))
    del state
    stream = io.StringIO()
    stream.write(f"checksum,{checksum:.16g}\n")
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(profile_top)
    return stream.getvalue()


def _write_rows(rows: list[BenchmarkRow], output: Path | None) -> None:
    if output is None:
        _write_csv(rows, sys.stdout)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        _write_csv(rows, handle)


def _write_csv(rows: list[BenchmarkRow], handle: TextIO) -> None:
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow(row.as_csv_row())


def _tracer_state_bytes(ntracer: int, grid_shape: tuple[int, int, int]) -> int:
    return int(ntracer) * int(np.prod(grid_shape)) * np.dtype(np.float64).itemsize


def _estimate_peak_bytes(
    ntracer: int,
    grid_shape: tuple[int, int, int],
    *,
    multiplier: float = DEFAULT_WORKING_SET_MULTIPLIER,
    fixed_overhead_bytes: int = DEFAULT_FIXED_OVERHEAD_BYTES,
) -> int:
    return int(_tracer_state_bytes(ntracer, grid_shape) * multiplier + fixed_overhead_bytes)


def _memory_limit_bytes(value: str) -> int | None:
    if value == "auto":
        physical = _physical_memory_bytes()
        if physical is None:
            return None
        return int(physical * AUTO_MEMORY_FRACTION)
    gb = float(value)
    if gb <= 0.0:
        raise ValueError("--max-memory-gb must be positive or 'auto'")
    return int(gb * 1024**3)


def _physical_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    if page_size <= 0 or page_count <= 0:
        return None
    return int(page_size * page_count)


def _count_is_allowed(estimated_peak_bytes: int, memory_limit_bytes: int | None) -> tuple[bool, str]:
    if memory_limit_bytes is None:
        return True, "memory limit unavailable; running without auto skip"
    if estimated_peak_bytes > memory_limit_bytes:
        return (
            False,
            f"estimated peak {_bytes_to_mib(estimated_peak_bytes):.1f} MiB exceeds memory limit "
            f"{_bytes_to_mib(memory_limit_bytes):.1f} MiB",
        )
    return True, ""


def _peak_rss_mib() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss / 1024.0**2
    return rss / 1024.0


def _bytes_to_mib(value: int | None) -> float:
    if value is None:
        return 0.0
    return float(value) / 1024.0**2


def _format_optional(value: float | None, *, precision: int = 8) -> str:
    if value is None:
        return ""
    return f"{value:.{precision}g}"


if __name__ == "__main__":
    raise SystemExit(main())
