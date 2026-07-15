from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from _scaling_support import (
    bytes_to_mib as _bytes_to_mib,
    count_is_allowed,
    estimate_peak_bytes,
    format_optional_fixed as _format_optional,
    memory_limit_bytes as _memory_limit_bytes,
    nonnegative_int as _nonnegative_int,
    peak_rss_mib as _peak_rss_mib,
    positive_int as _positive_int,
    tracer_state_bytes as _tracer_state_bytes,
    write_rows,
)
from wombat_transport.fields import TracerField
from wombat_transport.grid import TransportGrid, load_transport_grid
from wombat_transport.run_config import load_run_config
from wombat_transport.transport import driver as driver_mod
from wombat_transport.transport.forcing import MERRA2_72_TO_47_MAPPING, TransportForcing
from wombat_transport.transport.pressure import dry_surface_pressure_hpa, wet_surface_pressure_hpa


DEFAULT_COUNTS = (1, 24, 96, 256, 512)
DEFAULT_DT_S = 600.0
DEFAULT_WORKING_SET_MULTIPLIER = 24.0
DEFAULT_FIXED_OVERHEAD_BYTES = 768 * 1024**2
CSV_FIELDS = (
    "tracer_count",
    "status",
    "repeat",
    "best_wall_s",
    "mean_wall_s",
    "best_setup_s",
    "best_tpcore_s",
    "best_vdiff_s",
    "best_convection_s",
    "best_overhead_s",
    "mean_setup_s",
    "mean_tpcore_s",
    "mean_vdiff_s",
    "mean_convection_s",
    "mean_overhead_s",
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
STAGE_NAMES = ("setup", "tpcore", "vdiff", "convection")


@dataclass(frozen=True)
class SyntheticDriverInputs:
    tracer_field: TracerField
    forcing: TransportForcing
    grid: TransportGrid
    dt_s: float


@dataclass(frozen=True)
class TimedRun:
    total_s: float
    setup_s: float
    tpcore_s: float
    vdiff_s: float
    convection_s: float
    overhead_s: float
    checksum: float


@dataclass(frozen=True)
class BenchmarkRow:
    tracer_count: int
    status: str
    repeat: int
    best_wall_s: float | None
    mean_wall_s: float | None
    best_setup_s: float | None
    best_tpcore_s: float | None
    best_vdiff_s: float | None
    best_convection_s: float | None
    best_overhead_s: float | None
    mean_setup_s: float | None
    mean_tpcore_s: float | None
    mean_vdiff_s: float | None
    mean_convection_s: float | None
    mean_overhead_s: float | None
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
            "best_setup_s": _format_optional(self.best_setup_s),
            "best_tpcore_s": _format_optional(self.best_tpcore_s),
            "best_vdiff_s": _format_optional(self.best_vdiff_s),
            "best_convection_s": _format_optional(self.best_convection_s),
            "best_overhead_s": _format_optional(self.best_overhead_s),
            "mean_setup_s": _format_optional(self.mean_setup_s),
            "mean_tpcore_s": _format_optional(self.mean_tpcore_s),
            "mean_vdiff_s": _format_optional(self.mean_vdiff_s),
            "mean_convection_s": _format_optional(self.mean_convection_s),
            "mean_overhead_s": _format_optional(self.mean_overhead_s),
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
        peak_bytes = estimate_peak_bytes(
            tracer_count,
            grid_shape,
            multiplier=DEFAULT_WORKING_SET_MULTIPLIER,
            fixed_overhead_bytes=DEFAULT_FIXED_OVERHEAD_BYTES,
        )
        allowed, reason = count_is_allowed(peak_bytes, memory_limit)
        if not allowed:
            rows.append(_skipped_row(tracer_count, args.repeat, state_bytes, peak_bytes, memory_limit, reason))
            continue

        inputs = _build_synthetic_driver_inputs(args.run_config, tracer_count, dt_s=args.dt_s)
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

    write_rows(rows, CSV_FIELDS, args.output)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark full-grid synthetic TPCORE + VDIFF + convection scaling.")
    parser.add_argument(
        "--run-config",
        type=Path,
        default=Path("validation_runs/cases/realistic_restart_noemis_2x25/wombat/main/run.yml"),
        help="Run config used to locate the grid template. Defaults to validation_runs/cases/realistic_restart_noemis_2x25/wombat/main/run.yml.",
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


def _read_fullgrid_shape(run_config_path: Path) -> tuple[int, int, int]:
    config = load_run_config(run_config_path)
    return load_transport_grid(config.grid_template).shape


def _build_synthetic_driver_inputs(run_config_path: Path, ntracer: int, *, dt_s: float) -> SyntheticDriverInputs:
    config = load_run_config(run_config_path)
    grid = load_transport_grid(config.grid_template)
    lat = grid.lat_deg
    lon = grid.lon_deg
    nlev, nlat, nlon = grid.shape

    lev = np.arange(nlev, dtype=np.float64)[:, np.newaxis, np.newaxis]
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lat_2d = lat_rad[:, np.newaxis]
    lon_2d = lon_rad[np.newaxis, :]
    lat_3d = lat_rad[np.newaxis, :, np.newaxis]
    lon_3d = lon_rad[np.newaxis, np.newaxis, :]
    tracer_index = np.arange(ntracer, dtype=np.float64)[np.newaxis, np.newaxis, np.newaxis, :]

    surface_pressure_hpa = 965.0 + 22.0 * np.cos(lat_2d) ** 2
    surface_pressure_hpa = surface_pressure_hpa + 2.0 * np.sin(lon_2d) * np.cos(lat_2d)
    vertical_wave = np.sin((lev + 1.0) / float(nlev) * np.pi)
    u = 5.0 * vertical_wave * np.cos(lat_3d)
    u = u * (1.0 + 0.15 * np.cos(lon_3d))
    v = 0.35 * np.cos((lev + 1.0) / float(nlev) * np.pi) * np.sin(2.0 * lon_3d)
    v = v * np.cos(lat_3d)

    tracer = 4.0e-4 + 1.0e-7 * tracer_index
    tracer = tracer + 2.5e-8 * lev[..., np.newaxis] / max(float(nlev - 1), 1.0)
    tracer = tracer + 1.5e-8 * np.sin(lat_3d)[..., np.newaxis] + 7.5e-9 * np.cos(lon_3d)[..., np.newaxis]
    tracer = np.ascontiguousarray(tracer[np.newaxis, ...])

    qv = 0.010 * np.exp(-lev / 18.0) * (1.0 + 0.03 * np.sin(lat_3d)) * np.ones((1, 1, nlon))
    temperature = 289.0 - 0.45 * lev + 1.5 * np.sin(lat_3d) + 0.2 * np.cos(lon_3d)
    cloud_horizontal = (1.0 + 0.04 * np.sin(lat_3d)) * (1.0 + 0.02 * np.cos(lon_3d))
    cmfmc = 0.0035 * vertical_wave * cloud_horizontal
    dtrain = 0.00025 * vertical_wave * cloud_horizontal
    dqrcu = 1.0e-8 * vertical_wave * cloud_horizontal
    reevapcn = 2.0e-9 * vertical_wave * cloud_horizontal
    pficu = 0.00018 * vertical_wave * cloud_horizontal
    pflcu = 0.00014 * vertical_wave * cloud_horizontal

    field = TracerField(
        names=tuple(f"tracer_{index + 1:03d}" for index in range(ntracer)),
        data=tracer,
        units=tuple("mol mol-1 dry" for _ in range(ntracer)),
        coords={"lev": grid.lev[::-1], "lat": lat, "lon": lon},
    )
    surface_pressure_pa = surface_pressure_hpa[np.newaxis, ...] * 100.0
    specific_humidity = qv[np.newaxis, ...]
    temperature = temperature[np.newaxis, ...]
    wet_surface_pressure = wet_surface_pressure_hpa(surface_pressure_pa)
    dry_surface_pressure = dry_surface_pressure_hpa(
        surface_pressure_pa,
        specific_humidity,
        grid.hyai_hpa,
        grid.hybi,
    )
    forcing = TransportForcing(
        u_m_s=u[np.newaxis, ...],
        v_m_s=v[np.newaxis, ...],
        omega_pa_s=np.zeros((1, nlev, nlat, nlon), dtype=np.float64),
        surface_pressure_start_pa=surface_pressure_pa,
        surface_pressure_pa=surface_pressure_pa,
        restart_surface_pressure_pa=surface_pressure_pa,
        wet_surface_pressure_start_hpa=wet_surface_pressure,
        wet_surface_pressure_hpa=wet_surface_pressure,
        restart_wet_surface_pressure_hpa=wet_surface_pressure,
        dry_surface_pressure_start_hpa=dry_surface_pressure,
        dry_surface_pressure_hpa=dry_surface_pressure,
        restart_dry_surface_pressure_hpa=dry_surface_pressure,
        i3_start_wet_surface_pressure_hpa=wet_surface_pressure,
        i3_start_dry_surface_pressure_hpa=dry_surface_pressure,
        i3_start_specific_humidity_kg_kg=specific_humidity,
        specific_humidity_kg_kg=specific_humidity,
        restart_specific_humidity_kg_kg=specific_humidity,
        i3_start_temperature_k=temperature,
        temperature_k=temperature,
        restart_temperature_k=temperature,
        pbl_height_m=np.full((1, nlat, nlon), 950.0, dtype=np.float64),
        sensible_heat_flux_w_m2=np.full((1, nlat, nlon), 65.0, dtype=np.float64),
        latent_heat_flux_w_m2=np.full((1, nlat, nlon), 90.0, dtype=np.float64),
        friction_velocity_m_s=np.full((1, nlat, nlon), 0.35, dtype=np.float64),
        convective_mass_flux_kg_m2_s=cmfmc[np.newaxis, ...],
        convective_detrainment_kg_m2_s=dtrain[np.newaxis, ...],
        convective_precip_prod_kg_kg_s=dqrcu[np.newaxis, ...],
        convective_precip_reevap_kg_kg_s=reevapcn[np.newaxis, ...],
        convective_ice_flux_kg_m2_s=pficu[np.newaxis, ...],
        convective_liquid_flux_kg_m2_s=pflcu[np.newaxis, ...],
        convective_precip_mm_day=np.full((1, nlat, nlon), 2.0, dtype=np.float64),
        lat_deg=lat,
        lon_deg=lon,
        vertical_mapping=MERRA2_72_TO_47_MAPPING,
        a1_path=Path("<synthetic>"),
        a3dyn_path=Path("<synthetic>"),
        a3mstc_path=Path("<synthetic>"),
        a3mste_path=Path("<synthetic>"),
        i3_path=Path("<synthetic>"),
    )
    return SyntheticDriverInputs(tracer_field=field, forcing=forcing, grid=grid, dt_s=dt_s)


def _benchmark_inputs(
    inputs: SyntheticDriverInputs,
    *,
    tracer_count: int,
    repeat: int,
    warmup: int,
    state_bytes: int,
    peak_bytes: int,
    memory_limit: int | None,
) -> BenchmarkRow:
    for _ in range(warmup):
        _run_timed_step(inputs)
        gc.collect()

    runs: list[TimedRun] = []
    for _ in range(repeat):
        runs.append(_run_timed_step(inputs))
        gc.collect()
    best = min(runs, key=lambda run: run.total_s)
    mean = _mean_run(runs)
    gridcell_tracers = int(np.prod(inputs.tracer_field.data.shape[1:]))
    return BenchmarkRow(
        tracer_count=tracer_count,
        status="completed",
        repeat=repeat,
        best_wall_s=best.total_s,
        mean_wall_s=mean.total_s,
        best_setup_s=best.setup_s,
        best_tpcore_s=best.tpcore_s,
        best_vdiff_s=best.vdiff_s,
        best_convection_s=best.convection_s,
        best_overhead_s=best.overhead_s,
        mean_setup_s=mean.setup_s,
        mean_tpcore_s=mean.tpcore_s,
        mean_vdiff_s=mean.vdiff_s,
        mean_convection_s=mean.convection_s,
        mean_overhead_s=mean.overhead_s,
        seconds_per_tracer=best.total_s / float(tracer_count),
        tracers_per_second=float(tracer_count) / best.total_s,
        gridcell_tracers_per_second=float(gridcell_tracers) / best.total_s,
        tracer_state_mib=_bytes_to_mib(state_bytes),
        estimated_peak_mib=_bytes_to_mib(peak_bytes),
        memory_limit_mib=_bytes_to_mib(memory_limit) if memory_limit is not None else None,
        peak_rss_mib=_peak_rss_mib(),
        checksum=best.checksum,
        reason="",
    )


def _run_timed_step(inputs: SyntheticDriverInputs) -> TimedRun:
    stage_times = {name: 0.0 for name in STAGE_NAMES}
    timing_stack: list[dict[str, float]] = []
    originals: dict[str, Callable] = {
        "setup_tpcore_terms": driver_mod.setup_tpcore_terms,
        "run_tpcore_one_step_with_setup": driver_mod.run_tpcore_one_step_with_setup,
        "run_vdiffdr_one_step": driver_mod.run_vdiffdr_one_step,
        "run_cloud_convection_one_step": driver_mod.run_cloud_convection_one_step,
    }

    def timed(stage: str, function: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            timing_stack.append({"child_s": 0.0})
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                frame = timing_stack.pop()
                stage_times[stage] += elapsed - frame["child_s"]
                if timing_stack:
                    timing_stack[-1]["child_s"] += elapsed

        return wrapper

    try:
        driver_mod.setup_tpcore_terms = timed("setup", originals["setup_tpcore_terms"])
        driver_mod.run_tpcore_one_step_with_setup = timed("tpcore", originals["run_tpcore_one_step_with_setup"])
        driver_mod.run_vdiffdr_one_step = timed("vdiff", originals["run_vdiffdr_one_step"])
        driver_mod.run_cloud_convection_one_step = timed("convection", originals["run_cloud_convection_one_step"])
        start = time.perf_counter()
        result = driver_mod.run_transport_one_step(
            inputs.tracer_field,
            inputs.forcing,
            inputs.grid,
            dt_s=inputs.dt_s,
        )
        total = time.perf_counter() - start
    finally:
        driver_mod.setup_tpcore_terms = originals["setup_tpcore_terms"]
        driver_mod.run_tpcore_one_step_with_setup = originals["run_tpcore_one_step_with_setup"]
        driver_mod.run_vdiffdr_one_step = originals["run_vdiffdr_one_step"]
        driver_mod.run_cloud_convection_one_step = originals["run_cloud_convection_one_step"]

    stage_total = sum(stage_times.values())
    return TimedRun(
        total_s=total,
        setup_s=stage_times["setup"],
        tpcore_s=stage_times["tpcore"],
        vdiff_s=stage_times["vdiff"],
        convection_s=stage_times["convection"],
        overhead_s=total - stage_total,
        checksum=float(np.mean(result.state.data)),
    )


def _mean_run(runs: list[TimedRun]) -> TimedRun:
    count = float(len(runs))
    return TimedRun(
        total_s=sum(run.total_s for run in runs) / count,
        setup_s=sum(run.setup_s for run in runs) / count,
        tpcore_s=sum(run.tpcore_s for run in runs) / count,
        vdiff_s=sum(run.vdiff_s for run in runs) / count,
        convection_s=sum(run.convection_s for run in runs) / count,
        overhead_s=sum(run.overhead_s for run in runs) / count,
        checksum=sum(run.checksum for run in runs) / count,
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
        best_setup_s=None,
        best_tpcore_s=None,
        best_vdiff_s=None,
        best_convection_s=None,
        best_overhead_s=None,
        mean_setup_s=None,
        mean_tpcore_s=None,
        mean_vdiff_s=None,
        mean_convection_s=None,
        mean_overhead_s=None,
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


if __name__ == "__main__":
    raise SystemExit(main())
