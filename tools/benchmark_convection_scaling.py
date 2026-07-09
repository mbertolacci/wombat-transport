from __future__ import annotations

import argparse
import csv
import gc
import os
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import netCDF4
import numpy as np

from wombat_transport.grid import load_transport_grid
from wombat_transport.run_config import load_run_config
from wombat_transport.transport.convection import run_cloud_convection_one_step


DEFAULT_COUNTS = (1, 24, 96, 256, 512)
DEFAULT_DT_S = 600.0
DEFAULT_WORKING_SET_MULTIPLIER = 8.0
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
    "active_columns",
    "total_columns",
    "reason",
)


@dataclass(frozen=True)
class ConvectionInputs:
    tracer_conc: np.ndarray
    cmfmc_kg_m2_s: np.ndarray
    dtrain_kg_m2_s: np.ndarray
    dqrcu_kg_kg_s: np.ndarray
    reevapcn_kg_kg_s: np.ndarray
    delp_dry_hpa: np.ndarray
    delp_hpa: np.ndarray
    area_m2: np.ndarray
    bxheight_m: np.ndarray
    pficu_kg_m2_s: np.ndarray
    pflcu_kg_m2_s: np.ndarray
    temperature_k: np.ndarray
    precccon_mm_day: np.ndarray
    dt_s: float
    reconstruct_conv_precip_flux: bool


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
    active_columns: int | None
    total_columns: int | None
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
            "active_columns": "" if self.active_columns is None else str(self.active_columns),
            "total_columns": "" if self.total_columns is None else str(self.total_columns),
            "reason": self.reason,
        }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    memory_limit = _memory_limit_bytes(args.max_memory_gb)
    if args.input is not None:
        inputs = _read_inputs(args.input)
        grid_shape = tuple(int(value) for value in inputs.tracer_conc.shape[:3])
        tracer_count = int(inputs.tracer_conc.shape[-1])
        rows = [
            _benchmark_inputs(
                inputs,
                tracer_count=tracer_count,
                repeat=args.repeat,
                warmup=args.warmup,
                state_bytes=_tracer_state_bytes(tracer_count, grid_shape),
                peak_bytes=_estimate_peak_bytes(tracer_count, grid_shape),
                memory_limit=memory_limit,
            )
        ]
    else:
        grid_shape = _read_fullgrid_shape(args.run_config)
        rows = []
        for tracer_count in args.counts:
            state_bytes = _tracer_state_bytes(tracer_count, grid_shape)
            peak_bytes = _estimate_peak_bytes(tracer_count, grid_shape)
            allowed, reason = _count_is_allowed(peak_bytes, memory_limit)
            if not allowed:
                rows.append(_skipped_row(tracer_count, args.repeat, state_bytes, peak_bytes, memory_limit, reason))
                continue
            inputs = _build_synthetic_convection_inputs(args.run_config, tracer_count, dt_s=args.dt_s)
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
            del inputs
            gc.collect()

    _write_rows(rows, args.output)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark full-grid synthetic Wombat cloud convection scaling.")
    parser.add_argument(
        "--run-config",
        type=Path,
        default=Path("base_wombat/run.yml"),
        help="Run config used only to locate the grid template. Defaults to base_wombat/run.yml.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional NetCDF convection input fixture. When set, benchmark only that fixture.",
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


def _build_synthetic_convection_inputs(run_config_path: Path, ntracer: int, *, dt_s: float) -> ConvectionInputs:
    config = load_run_config(run_config_path)
    grid = load_transport_grid(config.grid_template)
    lat = grid.lat_deg
    lon = grid.lon_deg
    area = grid.area_m2
    nlev = grid.shape[0]

    nlat = lat.size
    nlon = lon.size
    lev = np.arange(nlev, dtype=np.float64)[:, np.newaxis, np.newaxis]
    lat_rad = np.deg2rad(lat)[np.newaxis, :, np.newaxis]
    lon_rad = np.deg2rad(lon)[np.newaxis, np.newaxis, :]
    tracer_index = np.arange(ntracer, dtype=np.float64)[np.newaxis, np.newaxis, np.newaxis, :]

    delp_profile = np.linspace(8.0, 62.0, nlev, dtype=np.float64)
    delp_dry = np.broadcast_to(delp_profile[:, np.newaxis, np.newaxis], (nlev, nlat, nlon)).copy()
    delp_dry *= 1.0 + 0.01 * np.sin(lat_rad) + 0.005 * np.cos(lon_rad)
    delp = delp_dry * 1.01
    bxheight = 140.0 + 2.5 * lev + np.zeros((1, nlat, nlon), dtype=np.float64)
    temperature = 215.0 + 0.55 * lev + 1.2 * np.sin(lat_rad) + 0.2 * np.cos(lon_rad)

    tracer = 4.0e-4 + 1.0e-7 * tracer_index
    tracer = tracer + 3.0e-9 * lev[..., np.newaxis]
    tracer = tracer + 2.0e-9 * np.sin(lat_rad)[..., np.newaxis] + 1.0e-9 * np.cos(lon_rad)[..., np.newaxis]
    tracer = np.ascontiguousarray(tracer)

    cloud_shape = (nlev, nlat, nlon)
    vertical_wave = np.sin((lev + 1.0) / float(nlev) * np.pi)
    horizontal = (1.0 + 0.04 * np.sin(lat_rad)) * (1.0 + 0.02 * np.cos(lon_rad))
    cmfmc = 0.0035 * vertical_wave * horizontal
    cmfmc = np.broadcast_to(cmfmc, cloud_shape).copy()
    dtrain = np.broadcast_to(0.00025 * vertical_wave * horizontal, cloud_shape).copy()
    dqrcu = np.broadcast_to(1.0e-8 * vertical_wave * horizontal, cloud_shape).copy()
    reevapcn = np.broadcast_to(2.0e-9 * vertical_wave * horizontal, cloud_shape).copy()
    pficu = np.broadcast_to(0.00018 * vertical_wave * horizontal, cloud_shape).copy()
    pflcu = np.broadcast_to(0.00014 * vertical_wave * horizontal, cloud_shape).copy()
    precccon = np.full((nlat, nlon), 2.0, dtype=np.float64)

    return ConvectionInputs(
        tracer_conc=tracer,
        cmfmc_kg_m2_s=cmfmc,
        dtrain_kg_m2_s=dtrain,
        dqrcu_kg_kg_s=dqrcu,
        reevapcn_kg_kg_s=reevapcn,
        delp_dry_hpa=delp_dry,
        delp_hpa=delp,
        area_m2=area,
        bxheight_m=bxheight,
        pficu_kg_m2_s=pficu,
        pflcu_kg_m2_s=pflcu,
        temperature_k=temperature,
        precccon_mm_day=precccon,
        dt_s=dt_s,
        reconstruct_conv_precip_flux=False,
    )


def _read_inputs(path: Path) -> ConvectionInputs:
    with netCDF4.Dataset(path) as dataset:
        return ConvectionInputs(
            tracer_conc=np.asarray(dataset.variables["tracer_conc"][:], dtype=np.float64),
            cmfmc_kg_m2_s=np.asarray(dataset.variables["cmfmc_kg_m2_s"][:], dtype=np.float64),
            dtrain_kg_m2_s=np.asarray(dataset.variables["dtrain_kg_m2_s"][:], dtype=np.float64),
            dqrcu_kg_kg_s=np.asarray(dataset.variables["dqrcu_kg_kg_s"][:], dtype=np.float64),
            reevapcn_kg_kg_s=np.asarray(dataset.variables["reevapcn_kg_kg_s"][:], dtype=np.float64),
            delp_dry_hpa=np.asarray(dataset.variables["delp_dry_hpa"][:], dtype=np.float64),
            delp_hpa=np.asarray(dataset.variables["delp_hpa"][:], dtype=np.float64),
            area_m2=np.asarray(dataset.variables["area_m2"][:], dtype=np.float64),
            bxheight_m=np.asarray(dataset.variables["bxheight_m"][:], dtype=np.float64),
            pficu_kg_m2_s=np.asarray(dataset.variables["pficu_kg_m2_s"][:], dtype=np.float64),
            pflcu_kg_m2_s=np.asarray(dataset.variables["pflcu_kg_m2_s"][:], dtype=np.float64),
            temperature_k=np.asarray(dataset.variables["temperature_k"][:], dtype=np.float64),
            precccon_mm_day=np.asarray(dataset.variables["precccon_mm_day"][:], dtype=np.float64),
            dt_s=float(dataset.dt_s),
            reconstruct_conv_precip_flux=bool(getattr(dataset, "reconstruct_conv_precip_flux", 0)),
        )


def _benchmark_inputs(
    inputs: ConvectionInputs,
    *,
    tracer_count: int,
    repeat: int,
    warmup: int,
    state_bytes: int,
    peak_bytes: int,
    memory_limit: int | None,
) -> BenchmarkRow:
    active = (np.max(np.abs(inputs.cmfmc_kg_m2_s), axis=0) > 1.0e-14) | (
        np.max(np.abs(inputs.dtrain_kg_m2_s), axis=0) > 1.0e-14
    )
    for _ in range(warmup):
        result = run_cloud_convection_one_step(
            tracer_conc=inputs.tracer_conc,
            cmfmc_kg_m2_s=inputs.cmfmc_kg_m2_s,
            dtrain_kg_m2_s=inputs.dtrain_kg_m2_s,
            dqrcu_kg_kg_s=inputs.dqrcu_kg_kg_s,
            reevapcn_kg_kg_s=inputs.reevapcn_kg_kg_s,
            delp_dry_hpa=inputs.delp_dry_hpa,
            delp_hpa=inputs.delp_hpa,
            area_m2=inputs.area_m2,
            bxheight_m=inputs.bxheight_m,
            pficu_kg_m2_s=inputs.pficu_kg_m2_s,
            pflcu_kg_m2_s=inputs.pflcu_kg_m2_s,
            temperature_k=inputs.temperature_k,
            precccon_mm_day=inputs.precccon_mm_day,
            dt_s=inputs.dt_s,
            reconstruct_conv_precip_flux=inputs.reconstruct_conv_precip_flux,
            diagnostics=False,
        )
        del result
        gc.collect()

    elapsed_values: list[float] = []
    checksum = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        result = run_cloud_convection_one_step(
            tracer_conc=inputs.tracer_conc,
            cmfmc_kg_m2_s=inputs.cmfmc_kg_m2_s,
            dtrain_kg_m2_s=inputs.dtrain_kg_m2_s,
            dqrcu_kg_kg_s=inputs.dqrcu_kg_kg_s,
            reevapcn_kg_kg_s=inputs.reevapcn_kg_kg_s,
            delp_dry_hpa=inputs.delp_dry_hpa,
            delp_hpa=inputs.delp_hpa,
            area_m2=inputs.area_m2,
            bxheight_m=inputs.bxheight_m,
            pficu_kg_m2_s=inputs.pficu_kg_m2_s,
            pflcu_kg_m2_s=inputs.pflcu_kg_m2_s,
            temperature_k=inputs.temperature_k,
            precccon_mm_day=inputs.precccon_mm_day,
            dt_s=inputs.dt_s,
            reconstruct_conv_precip_flux=inputs.reconstruct_conv_precip_flux,
            diagnostics=False,
        )
        elapsed_values.append(time.perf_counter() - start)
        checksum = float(np.mean(result.tracer_conc))
        del result
        gc.collect()
    best = min(elapsed_values)
    mean = sum(elapsed_values) / float(len(elapsed_values))
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
        active_columns=int(active.sum()),
        total_columns=int(active.size),
        reason="",
    )


def _skipped_row(
    tracer_count: int,
    repeat: int,
    state_bytes: int,
    peak_bytes: int,
    memory_limit: int | None,
    reason: str,
) -> BenchmarkRow:
    return BenchmarkRow(
        tracer_count=tracer_count,
        status="skipped",
        repeat=repeat,
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
        active_columns=None,
        total_columns=None,
        reason=reason,
    )


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


def _count_is_allowed(estimated_peak_bytes: int, memory_limit_bytes: int | None) -> tuple[bool, str]:
    if memory_limit_bytes is None:
        return True, "memory limit unavailable"
    if estimated_peak_bytes > memory_limit_bytes:
        return False, "estimated peak exceeds memory limit"
    return True, ""


def _memory_limit_bytes(value: str) -> int | None:
    if value == "auto":
        physical = _physical_memory_bytes()
        if physical is None:
            return None
        return int(physical * AUTO_MEMORY_FRACTION)
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError("memory limit must be positive")
    return int(parsed * 1024**3)


def _physical_memory_bytes() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    if pages <= 0 or page_size <= 0:
        return None
    return int(pages) * int(page_size)


def _peak_rss_mib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _bytes_to_mib(value: int | None) -> float:
    if value is None:
        return float("nan")
    return float(value) / 1024.0**2


def _format_optional(value: float | None, *, precision: int = 8) -> str:
    if value is None:
        return ""
    return f"{value:.{precision}f}"


if __name__ == "__main__":
    raise SystemExit(main())
