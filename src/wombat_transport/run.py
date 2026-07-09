from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import netCDF4
import numpy as np

from wombat_transport.compare import compare_to_time_slice, format_metrics
from wombat_transport.grid import load_transport_grid
from wombat_transport.io import initialize_tracers, load_species_conc, write_restart_like
from wombat_transport.run_config import load_run_config
from wombat_transport.runner import run_emissions_replay, run_tracer_simulation
from wombat_transport.species import load_species_database
from wombat_transport.transport import (
    dry_pressure_edges_from_thickness_hpa,
    load_transport_forcing,
    run_transport_one_step,
    run_transport_window,
)

CONFIG_TIME_FORMAT = "%Y-%m-%d %H:%M"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Wombat transport prototype mode. Emissions replay uses "
            "GEOS-Chem HEMCO diagnostic outputs as cached source terms, not raw HEMCO inputs."
        )
    )
    parser.add_argument("run_config", type=Path)
    parser.add_argument(
        "--mode",
        choices=["run", "init-only", "emissions-only", "transport-one-step", "transport-window"],
        default="run",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--write-output", type=Path, default=None)
    args = parser.parse_args(argv)

    config = load_run_config(args.run_config)
    if args.mode == "run":
        result = run_tracer_simulation(config, max_steps=args.max_steps)
        state = result.state
        transport_result = result
        comparison_state = state
        comparison_delp_dry_hpa = result.final_delp_dry_hpa
    elif args.mode == "init-only":
        state = initialize_tracers(
            config.initial_restart,
            config.species_database,
            template_path=config.grid_template,
        )
        result = None
        transport_result = None
        comparison_state = state
        comparison_delp_dry_hpa = None
    elif args.mode == "transport-one-step":
        grid = load_transport_grid(config.grid_template)
        state = initialize_tracers(
            config.initial_restart,
            config.species_database,
            template_path=config.grid_template,
        )
        forcing = load_transport_forcing(
            _resolve_config_value(config.root, config.transport["met_root"]),
            datetime.strptime(config.transport["start"], CONFIG_TIME_FORMAT),
            grid,
            time_index=int(config.transport.get("met_time_index", 0)),
        )
        transport_result = run_transport_one_step(
            state,
            forcing,
            grid,
            dt_s=float(config.transport.get("dt_s", 600.0)),
        )
        state = transport_result.state
        comparison_state = state
        comparison_delp_dry_hpa = transport_result.delp_dry_hpa
        result = None
    elif args.mode == "transport-window":
        grid = load_transport_grid(config.grid_template)
        state = initialize_tracers(
            config.initial_restart,
            config.species_database,
            template_path=config.grid_template,
        )
        steps = int(config.transport.get("window_steps", args.max_steps or 18))
        transport_result = run_transport_window(
            state,
            _resolve_config_value(config.root, config.transport["met_root"]),
            datetime.strptime(config.transport["start"], CONFIG_TIME_FORMAT),
            grid,
            steps=steps,
            dt_s=float(config.transport.get("dt_s", 600.0)),
            initial_met_time_index=int(config.transport.get("met_time_index", 0)),
        )
        state = transport_result.state
        comparison_state = transport_result.average_state
        comparison_delp_dry_hpa = transport_result.average_delp_dry_hpa
        result = None
    else:
        result = run_emissions_replay(config, max_steps=args.max_steps)
        state = result.state
        transport_result = None
        comparison_state = state
        comparison_delp_dry_hpa = None

    if args.write_output is not None:
        write_restart_like(args.write_output, state, config.grid_template)

    species = load_species_database(config.species_database)
    print(f"name: {config.name}")
    print(f"mode: {args.mode}")
    print(f"state_shape: {state.shape}")
    if args.write_output is not None:
        print(f"wrote_output: {args.write_output}")
    if result is not None:
        print(f"emissions_files_discovered: {len(result.discovered_files)}")
        print(f"emissions_files_processed: {len(result.processed_files)}")
        print(f"emissions_files_skipped: {len(result.skipped_files)}")
        for diagnostic in result.skipped_files:
            print(f"skipped_invalid_emissions: {diagnostic.path.name}")
        print(f"total_emitted_mass_kg: {result.total_emitted_mass:.8e}")
    if transport_result is not None:
        scalar_mass_error = None
        if hasattr(transport_result, "final_scalar_mass") and hasattr(transport_result, "initial_scalar_mass"):
            scalar_mass_error = transport_result.final_scalar_mass - transport_result.initial_scalar_mass
        transport_steps = getattr(transport_result, "steps", 1)
        if hasattr(transport_result, "transport_steps"):
            transport_steps = transport_result.transport_steps
        transport_dt_s = getattr(transport_result, "dt_s", float(config.transport.get("dt_s", 600.0)))
        if hasattr(transport_result, "transport_dt_s"):
            transport_dt_s = transport_result.transport_dt_s
        print(f"transport_operators: {','.join(transport_result.transport_operators)}")
        print(f"transport_steps: {transport_steps}")
        print(f"transport_dt_s: {transport_dt_s:.8e}")
        if hasattr(transport_result, "emissions_steps"):
            print(f"emissions_steps: {transport_result.emissions_steps}")
            print(f"emissions_dt_s: {transport_result.emissions_dt_s:.8e}")
        for stage in transport_result.stage_masses:
            stage_error = stage.final_scalar_mass - stage.initial_scalar_mass
            print(f"{stage.operator}_max_scalar_mass_error: {np.max(np.abs(stage_error)):.8e}")
        if scalar_mass_error is not None:
            print(f"max_transport_scalar_mass_error: {np.max(np.abs(scalar_mass_error)):.8e}")

    validation = config.validation or config.comparison
    comparison_path = validation.get("species_conc_sample")
    if comparison_path:
        reference = load_species_conc(config.root / comparison_path)
        time_index = int(validation.get("species_conc_time_index", -1))
        with netCDF4.Dataset(config.grid_template) as dataset:
            area_m2 = np.asarray(dataset.variables["AREA"][:])
            if comparison_delp_dry_hpa is None:
                delp_dry_hpa = np.asarray(dataset.variables["Met_DELPDRY"][:])
            else:
                delp_dry_hpa = comparison_delp_dry_hpa
        metrics = compare_to_time_slice(
            comparison_state,
            reference,
            reference_time_index=time_index,
            species=species,
            delp_dry_hpa=delp_dry_hpa,
            area_m2=area_m2,
        )
        print(format_metrics(metrics))

    level_edge_path = config.diagnostics.get("level_edge_sample")
    if transport_result is not None and level_edge_path and comparison_delp_dry_hpa is not None:
        time_index = int(validation.get("species_conc_time_index", 0))
        with netCDF4.Dataset(config.root / level_edge_path) as dataset:
            reference_edges = np.asarray(dataset.variables["Met_PEDGEDRY"][time_index : time_index + 1])
        reference_delp = np.abs(reference_edges[:, :-1, :, :] - reference_edges[:, 1:, :, :])
        pressure_error = np.abs(comparison_delp_dry_hpa - reference_delp)
        modeled_edges = dry_pressure_edges_from_thickness_hpa(
            comparison_delp_dry_hpa,
            top_edge_hpa=reference_edges[:, -1:, :, :],
        )
        edge_error = np.abs(modeled_edges - reference_edges)
        print(f"pressure_dry_max_abs_error_hpa: {np.max(pressure_error):.8e}")
        print(f"pressure_dry_mean_abs_error_hpa: {np.mean(pressure_error):.8e}")
        print(f"pressure_edge_dry_max_abs_error_hpa: {np.max(edge_error):.8e}")
        print(f"pressure_edge_dry_mean_abs_error_hpa: {np.mean(edge_error):.8e}")

    return 0


def _resolve_config_value(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


if __name__ == "__main__":
    raise SystemExit(main())
