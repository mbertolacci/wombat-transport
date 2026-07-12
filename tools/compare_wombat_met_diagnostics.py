#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4
import numpy as np

from wombat_transport.grid import load_transport_grid
from wombat_transport.io import initialize_tracers
from wombat_transport.met_diagnostics import (
    AirQuantityDiagnostics,
    airqnt_diagnostics_from_fields,
    airqnt_diagnostics_from_forcing,
)
from wombat_transport.output import OutputStorageConfig, _copy_common_coordinates, _create_common_dimensions, _create_output_variable, _write_time
from wombat_transport.run_config import (
    load_run_config,
    meteorology_chunk_multiple,
    meteorology_initial_time_index,
    meteorology_root,
    simulation_start,
    transport_timestep_s,
)
from wombat_transport.transport.forcing import TransportForcingProvider
from wombat_transport.transport.driver import trace_transport_one_step
from wombat_transport.runner import _initial_dry_air_mass


@dataclass(frozen=True)
class WindowDiagnostics:
    timestamp: datetime
    diagnostics: AirQuantityDiagnostics


@dataclass(frozen=True)
class FieldMetrics:
    name: str
    max_abs: float
    mean_abs: float
    mean_bias: float
    max_index: tuple[int, ...]
    candidate_at_max: float
    reference_at_max: float


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Wombat AIRQNT-style met diagnostics with GEOS-Chem HISTORY files.")
    parser.add_argument("run_config", type=Path)
    parser.add_argument("--level-edge", type=Path, required=True)
    parser.add_argument("--state-met", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--steps-per-window", type=int, default=18)
    parser.add_argument("--phase", choices=["after-vdiff", "forcing"], default="after-vdiff")
    parser.add_argument("--write-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_run_config(args.run_config)
    windows = compute_window_diagnostics(
        config,
        window_count=args.windows,
        steps_per_window=args.steps_per_window,
        phase=args.phase,
    )
    if args.write_dir is not None:
        write_met_diagnostic_files(args.write_dir, config.grid_template, windows)

    for metrics in compare_to_geos_chem(windows, args.level_edge, args.state_met):
        print(
            f"{metrics.name},max_abs={metrics.max_abs:.12e},mean_abs={metrics.mean_abs:.12e},"
            f"mean_bias={metrics.mean_bias:.12e},max_index={metrics.max_index},"
            f"candidate={metrics.candidate_at_max:.12e},reference={metrics.reference_at_max:.12e}"
        )
    return 0


def compute_window_diagnostics(config, *, window_count: int, steps_per_window: int, phase: str) -> list[WindowDiagnostics]:
    grid = load_transport_grid(config.grid_template)
    met_root = meteorology_root(config)
    start = simulation_start(config)
    dt_s = float(transport_timestep_s(config))
    forcing_provider = TransportForcingProvider(
        met_root,
        start,
        grid,
        initial_met_time_index=meteorology_initial_time_index(config),
        chunk_multiple=meteorology_chunk_multiple(config),
    )
    windows: list[WindowDiagnostics] = []
    state = None
    dry_air_mass = None
    if phase == "after-vdiff":
        state = initialize_tracers(
            config.initial_restart,
            config.species_database,
            template_path=config.grid_template,
        )

    total_steps = window_count * steps_per_window
    sums: dict[str, np.ndarray] = {}
    count = 0
    window_start = start
    for step in range(total_steps):
        current = start + timedelta(seconds=step * dt_s)
        forcing = forcing_provider.forcing_for_step(current, dt_s=dt_s)
        if phase == "forcing":
            diag = airqnt_diagnostics_from_forcing(forcing, grid)
        elif phase == "after-vdiff":
            assert state is not None
            if dry_air_mass is None:
                dry_air_mass = _initial_dry_air_mass(config, forcing, grid)
            trace = trace_transport_one_step(state, forcing, grid, dt_s=dt_s, dry_air_mass_kg=dry_air_mass)
            state = trace.result.state
            dry_air_mass = trace.result.dry_air_mass_kg
            diag = airqnt_diagnostics_from_fields(
                wet_surface_pressure_hpa=forcing.wet_surface_pressure_hpa[0],
                specific_humidity_kg_kg=trace.vdiff_output.specific_humidity_kg_kg[::-1],
                temperature_k=forcing.temperature_k[0],
                grid=grid,
            )
        else:
            raise ValueError(f"unsupported phase {phase!r}")
        fields = {
            "wet_pressure_edges_hpa": diag.wet_pressure_edges_hpa,
            "dry_partial_pressure_edges_hpa": diag.dry_partial_pressure_edges_hpa,
            "water_vapor_volume_mixing_ratio": diag.water_vapor_volume_mixing_ratio,
            "box_height_m": diag.box_height_m,
        }
        if count == 0:
            sums = {name: np.zeros_like(value, dtype=np.float64) for name, value in fields.items()}
        for name, value in fields.items():
            sums[name] += value
        count += 1

        if count == steps_per_window:
            windows.append(
                WindowDiagnostics(
                    timestamp=window_start,
                    diagnostics=AirQuantityDiagnostics(
                        wet_pressure_edges_hpa=sums["wet_pressure_edges_hpa"] / count,
                        dry_partial_pressure_edges_hpa=sums["dry_partial_pressure_edges_hpa"] / count,
                        water_vapor_volume_mixing_ratio=sums["water_vapor_volume_mixing_ratio"] / count,
                        box_height_m=sums["box_height_m"] / count,
                    ),
                )
            )
            count = 0
            window_start = window_start + timedelta(seconds=steps_per_window * dt_s)

    return windows


def compare_to_geos_chem(windows: list[WindowDiagnostics], level_edge_path: Path, state_met_path: Path) -> list[FieldMetrics]:
    candidate = {
        "Met_PEDGE": np.stack([item.diagnostics.wet_pressure_edges_hpa for item in windows], axis=0),
        "Met_PEDGEDRY": np.stack([item.diagnostics.dry_partial_pressure_edges_hpa for item in windows], axis=0),
        "Met_AVGW": np.stack([item.diagnostics.water_vapor_volume_mixing_ratio for item in windows], axis=0),
        "Met_BXHEIGHT": np.stack([item.diagnostics.box_height_m for item in windows], axis=0),
    }
    reference: dict[str, np.ndarray] = {}
    with netCDF4.Dataset(level_edge_path) as dataset:
        reference["Met_PEDGE"] = np.asarray(dataset.variables["Met_PEDGE"][: len(windows)], dtype=np.float64)
        reference["Met_PEDGEDRY"] = np.asarray(dataset.variables["Met_PEDGEDRY"][: len(windows)], dtype=np.float64)
    with netCDF4.Dataset(state_met_path) as dataset:
        reference["Met_AVGW"] = np.asarray(dataset.variables["Met_AVGW"][: len(windows)], dtype=np.float64)
        reference["Met_BXHEIGHT"] = np.asarray(dataset.variables["Met_BXHEIGHT"][: len(windows)], dtype=np.float64)
    return [_field_metrics(name, candidate[name], reference[name]) for name in candidate]


def write_met_diagnostic_files(output_dir: Path, template_path: Path, windows: list[WindowDiagnostics]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    level_edge = output_dir / f"GEOSChem.LevelEdgeDiagsThreeHourly.{windows[0].timestamp:%Y%m%d_%H%Mz}.nc4"
    state_met = output_dir / f"GEOSChem.StateMetThreeHourly.{windows[0].timestamp:%Y%m%d_%H%Mz}.nc4"
    storage = OutputStorageConfig()
    _write_collection(
        level_edge,
        template_path,
        windows,
        {
            "Met_PEDGE": ("ilev", "hPa", "Moist air pressure at level edges", "wet_pressure_edges_hpa"),
            "Met_PEDGEDRY": ("ilev", "hPa", "Dry air partial pressure at level edges", "dry_partial_pressure_edges_hpa"),
        },
        storage,
    )
    _write_collection(
        state_met,
        template_path,
        windows,
        {
            "Met_AVGW": ("lev", "mol mol-1 dry", "Water vapor volume mixing ratio", "water_vapor_volume_mixing_ratio"),
            "Met_BXHEIGHT": ("lev", "m", "Grid box height", "box_height_m"),
        },
        storage,
    )
    return level_edge, state_met


def _write_collection(
    path: Path,
    template_path: Path,
    windows: list[WindowDiagnostics],
    variables: dict[str, tuple[str, str, str, str]],
    storage: OutputStorageConfig,
) -> None:
    with netCDF4.Dataset(template_path) as template, netCDF4.Dataset(path, "w") as output:
        _create_common_dimensions(output, template, time_size=len(windows), include_bounds=True)
        _copy_common_coordinates(output, template, include_bounds=True, storage=storage)
        _write_time(output, [item.timestamp for item in windows], base=windows[0].timestamp, storage=storage)
        output.title = "Wombat AIRQNT diagnostic collection"
        output.format = "NetCDF-4"
        for name, (vertical_dimension, units, long_name, attr_name) in variables.items():
            variable = _create_output_variable(output, name, ("time", vertical_dimension, "lat", "lon"), storage)
            variable.units = units
            variable.long_name = long_name
            variable[:] = np.stack([getattr(item.diagnostics, attr_name) for item in windows], axis=0)


def _field_metrics(name: str, candidate: np.ndarray, reference: np.ndarray) -> FieldMetrics:
    if candidate.shape != reference.shape:
        raise ValueError(f"{name} shape mismatch: candidate {candidate.shape}, reference {reference.shape}")
    diff = candidate - reference
    max_index = tuple(int(i) for i in np.unravel_index(np.argmax(np.abs(diff)), diff.shape))
    return FieldMetrics(
        name=name,
        max_abs=float(np.max(np.abs(diff))),
        mean_abs=float(np.mean(np.abs(diff))),
        mean_bias=float(np.mean(diff)),
        max_index=max_index,
        candidate_at_max=float(candidate[max_index]),
        reference_at_max=float(reference[max_index]),
    )


if __name__ == "__main__":
    raise SystemExit(main())
