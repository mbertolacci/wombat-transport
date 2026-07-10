from __future__ import annotations

import argparse
import logging
from pathlib import Path

import netCDF4
import numpy as np

from wombat_transport.compare import compare_to_time_slice, format_metrics
from wombat_transport.grid import load_transport_grid
from wombat_transport.io import initialize_tracers, load_species_conc, write_restart_like
from wombat_transport.run_config import (
    logging_level,
    load_run_config,
    meteorology_initial_time_index,
    meteorology_root,
    simulation_start,
    transport_timestep_s,
)
from wombat_transport.runner import run_tracer_simulation
from wombat_transport.runner import _initial_dry_air_mass
from wombat_transport.species import load_species_database
from wombat_transport.transport import (
    dry_pressure_edges_from_thickness_hpa,
    load_transport_forcing_for_step,
    run_transport_one_step,
    run_transport_window,
)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a Wombat transport prototype mode."
    )
    parser.add_argument("run_config", type=Path)
    parser.add_argument(
        "--mode",
        choices=["run", "init-only", "transport-one-step", "transport-window"],
        default="run",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--write-output", type=Path, default=None)
    args = parser.parse_args(argv)

    config = load_run_config(args.run_config)
    _configure_logging(config)
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
        forcing = load_transport_forcing_for_step(
            meteorology_root(config),
            simulation_start(config),
            simulation_start(config),
            grid,
            dt_s=transport_timestep_s(config),
            initial_met_time_index=meteorology_initial_time_index(config),
        )
        transport_result = run_transport_one_step(
            state,
            forcing,
            grid,
            dt_s=transport_timestep_s(config),
            dry_air_mass_kg=_initial_dry_air_mass(config, forcing, grid),
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
        steps = args.max_steps or 18
        transport_result = run_transport_window(
            state,
            meteorology_root(config),
            simulation_start(config),
            grid,
            steps=steps,
            dt_s=transport_timestep_s(config),
            initial_met_time_index=meteorology_initial_time_index(config),
        )
        state = transport_result.state
        comparison_state = transport_result.average_state
        comparison_delp_dry_hpa = transport_result.average_delp_dry_hpa
        result = None
    else:
        raise AssertionError(f"unhandled mode {args.mode}")

    if args.write_output is not None:
        write_restart_like(args.write_output, state, config.grid_template)

    species = load_species_database(config.species_database)
    print(f"name: {config.name}")
    print(f"mode: {args.mode}")
    print(f"state_shape: {state.shape}")
    if args.write_output is not None:
        print(f"wrote_output: {args.write_output}")
    if result is not None:
        print(f"total_emitted_mass_kg: {result.total_emitted_mass:.8e}")
    if transport_result is not None:
        transport_steps = getattr(transport_result, "steps", 1)
        if hasattr(transport_result, "transport_steps"):
            transport_steps = transport_result.transport_steps
        transport_dt_s = getattr(transport_result, "dt_s", transport_timestep_s(config))
        if hasattr(transport_result, "transport_dt_s"):
            transport_dt_s = transport_result.transport_dt_s
        print(f"transport_operators: {','.join(transport_result.transport_operators)}")
        print(f"transport_steps: {transport_steps}")
        print(f"transport_dt_s: {transport_dt_s:.8e}")
        if hasattr(transport_result, "emissions_steps"):
            print(f"emissions_steps: {transport_result.emissions_steps}")
            print(f"emissions_dt_s: {transport_result.emissions_dt_s:.8e}")

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


def _configure_logging(config) -> None:
    level_name = logging_level(config).upper()
    logging.basicConfig(
        level=getattr(logging, level_name),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


if __name__ == "__main__":
    raise SystemExit(main())
