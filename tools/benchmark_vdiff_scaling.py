from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _scaling_support import (
    bytes_to_mib as _bytes_to_mib,
    count_is_allowed,
    estimate_peak_bytes,
    format_optional_general as _format_optional,
    memory_limit_bytes as _memory_limit_bytes,
    nonnegative_int as _nonnegative_int,
    peak_rss_mib as _peak_rss_mib,
    positive_int as _positive_int,
    tracer_state_bytes as _tracer_state_bytes,
    write_rows,
)
from wombat_transport.grid import load_transport_grid
from wombat_transport.run_config import load_run_config
from wombat_transport.transport.pbl import G0_M_PER_S2, ZVIR, run_vdiffdr_one_step


DEFAULT_COUNTS = (1, 24, 96, 256, 512)
DEFAULT_DT_S = 600.0
DEFAULT_WORKING_SET_MULTIPLIER = 8.0
DEFAULT_FIXED_OVERHEAD_BYTES = 512 * 1024**2
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
class SyntheticVdiffInputs:
    tracer_conc: np.ndarray
    u_m_s: np.ndarray
    v_m_s: np.ndarray
    temperature_k: np.ndarray
    specific_humidity_kg_kg: np.ndarray
    pmid_hpa: np.ndarray
    pedge_hpa: np.ndarray
    virtual_temperature_k: np.ndarray
    bxheight_m: np.ndarray
    dry_air_mass_kg: np.ndarray
    pbl_top_m: np.ndarray
    hflux_w_m2: np.ndarray
    eflux_w_m2: np.ndarray
    ustar_m_s: np.ndarray
    area_m2: np.ndarray
    surface_flux_kg_m2_s: np.ndarray
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
        peak_bytes = estimate_peak_bytes(
            tracer_count,
            grid_shape,
            multiplier=DEFAULT_WORKING_SET_MULTIPLIER,
            fixed_overhead_bytes=DEFAULT_FIXED_OVERHEAD_BYTES,
        )
        allowed, reason = count_is_allowed(peak_bytes, memory_limit)
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

        inputs = _build_synthetic_vdiff_inputs(
            args.run_config,
            tracer_count,
            dt_s=args.dt_s,
            surface_flux_kg_m2_s=args.surface_flux_kg_m2_s,
        )
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
    parser = argparse.ArgumentParser(description="Benchmark full-grid synthetic VDIFFDR scaling by tracer count.")
    parser.add_argument(
        "--run-config",
        type=Path,
        default=Path("validation_runs/cases/realistic_restart_noemis_2x25/wombat/main/run.yml"),
        help="Run config used only to locate the grid template. Defaults to validation_runs/cases/realistic_restart_noemis_2x25/wombat/main/run.yml.",
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
        "--surface-flux-kg-m2-s",
        type=float,
        default=0.0,
        help="Uniform synthetic tracer surface flux. Defaults to the zero-flux fast path.",
    )
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


def _build_synthetic_vdiff_inputs(
    run_config_path: Path,
    ntracer: int,
    *,
    dt_s: float,
    surface_flux_kg_m2_s: float = 0.0,
) -> SyntheticVdiffInputs:
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

    pedge_profile = np.linspace(50.0, 1000.0, nlev + 1, dtype=np.float64)
    pedge = np.broadcast_to(pedge_profile[:, np.newaxis, np.newaxis], (nlev + 1, nlat, nlon)).copy()
    pmid = 0.5 * (pedge[:-1] + pedge[1:])
    temperature = 289.0 - 0.45 * lev + 1.5 * np.sin(lat_rad) + 0.2 * np.cos(lon_rad)
    sphu = 0.010 * np.exp(-lev / 18.0) * (1.0 + 0.03 * np.sin(lat_rad)) * np.ones((1, 1, nlon))
    virtual_temperature = temperature * (1.0 + ZVIR * sphu)
    bxheight = np.full((nlev, nlat, nlon), 125.0, dtype=np.float64)
    dry_air_mass = (pedge[1:] - pedge[:-1]) * 100.0 / G0_M_PER_S2
    dry_air_mass = dry_air_mass * area[np.newaxis, :, :]

    vertical_wave = np.sin((lev + 1.0) / float(nlev) * np.pi)
    u = 4.0 + 0.05 * lev + 0.2 * np.cos(lon_rad)
    u = u * (1.0 + 0.05 * vertical_wave) * np.ones((1, nlat, 1), dtype=np.float64)
    v = 0.3 * vertical_wave * np.sin(2.0 * lon_rad) + 0.02 * np.sin(lat_rad)

    tracer = 4.0e-4 + 1.0e-7 * tracer_index
    tracer = tracer + 4.0e-9 * lev[..., np.newaxis]
    tracer = tracer + 2.0e-9 * np.sin(lat_rad)[..., np.newaxis] + 1.0e-9 * np.cos(lon_rad)[..., np.newaxis]
    tracer = np.ascontiguousarray(tracer)

    pbl_top = np.full((nlat, nlon), 950.0, dtype=np.float64)
    hflux = np.full((nlat, nlon), 65.0, dtype=np.float64)
    eflux = np.full((nlat, nlon), 90.0, dtype=np.float64)
    ustar = np.full((nlat, nlon), 0.35, dtype=np.float64)
    surface_flux = np.full((nlat, nlon, ntracer), surface_flux_kg_m2_s, dtype=np.float64)

    return SyntheticVdiffInputs(
        tracer_conc=tracer,
        u_m_s=u,
        v_m_s=v,
        temperature_k=temperature,
        specific_humidity_kg_kg=sphu,
        pmid_hpa=pmid,
        pedge_hpa=pedge,
        virtual_temperature_k=virtual_temperature,
        bxheight_m=bxheight,
        dry_air_mass_kg=dry_air_mass,
        pbl_top_m=pbl_top,
        hflux_w_m2=hflux,
        eflux_w_m2=eflux,
        ustar_m_s=ustar,
        area_m2=area,
        surface_flux_kg_m2_s=surface_flux,
        dt_s=dt_s,
    )


def _benchmark_inputs(
    inputs: SyntheticVdiffInputs,
    *,
    tracer_count: int,
    repeat: int,
    warmup: int,
    state_bytes: int,
    peak_bytes: int,
    memory_limit: int | None,
) -> BenchmarkRow:
    for _ in range(warmup):
        state = run_vdiffdr_one_step(
            tracer_conc=inputs.tracer_conc,
            u_m_s=inputs.u_m_s,
            v_m_s=inputs.v_m_s,
            temperature_k=inputs.temperature_k,
            specific_humidity_kg_kg=inputs.specific_humidity_kg_kg,
            pmid_hpa=inputs.pmid_hpa,
            pedge_hpa=inputs.pedge_hpa,
            virtual_temperature_k=inputs.virtual_temperature_k,
            bxheight_m=inputs.bxheight_m,
            dry_air_mass_kg=inputs.dry_air_mass_kg,
            pbl_top_m=inputs.pbl_top_m,
            hflux_w_m2=inputs.hflux_w_m2,
            eflux_w_m2=inputs.eflux_w_m2,
            ustar_m_s=inputs.ustar_m_s,
            area_m2=inputs.area_m2,
            dt_s=inputs.dt_s,
            surface_flux_kg_m2_s=inputs.surface_flux_kg_m2_s,
            diagnostics=False,
            reuse_output=True,
        )
        del state
        gc.collect()

    elapsed_values: list[float] = []
    checksum = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        state = run_vdiffdr_one_step(
            tracer_conc=inputs.tracer_conc,
            u_m_s=inputs.u_m_s,
            v_m_s=inputs.v_m_s,
            temperature_k=inputs.temperature_k,
            specific_humidity_kg_kg=inputs.specific_humidity_kg_kg,
            pmid_hpa=inputs.pmid_hpa,
            pedge_hpa=inputs.pedge_hpa,
            virtual_temperature_k=inputs.virtual_temperature_k,
            bxheight_m=inputs.bxheight_m,
            dry_air_mass_kg=inputs.dry_air_mass_kg,
            pbl_top_m=inputs.pbl_top_m,
            hflux_w_m2=inputs.hflux_w_m2,
            eflux_w_m2=inputs.eflux_w_m2,
            ustar_m_s=inputs.ustar_m_s,
            area_m2=inputs.area_m2,
            dt_s=inputs.dt_s,
            surface_flux_kg_m2_s=inputs.surface_flux_kg_m2_s,
            diagnostics=False,
            reuse_output=True,
        )
        elapsed_values.append(time.perf_counter() - start)
        checksum = float(np.sum(state.tracer_conc))
    best = min(elapsed_values)
    mean = sum(elapsed_values) / float(len(elapsed_values))
    gridcell_tracers = np.prod(inputs.tracer_conc.shape)
    return BenchmarkRow(
        tracer_count=tracer_count,
        status="ok",
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


if __name__ == "__main__":
    raise SystemExit(main())
