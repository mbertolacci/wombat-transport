from __future__ import annotations

import argparse
from pathlib import Path

import netCDF4
import numpy as np

from wombat_transport.compare import compare_to_time_slice, format_metrics
from wombat_transport.io import initialize_tracers, load_species_conc, write_restart_like
from wombat_transport.run_config import load_run_config
from wombat_transport.runner import run_emissions_replay
from wombat_transport.species import load_species_database


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Wombat transport prototype mode. Emissions replay uses "
            "GEOS-Chem HEMCO diagnostic outputs as cached source terms, not raw HEMCO inputs."
        )
    )
    parser.add_argument("run_config", type=Path)
    parser.add_argument("--mode", choices=["init-only", "emissions-only"], default="emissions-only")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--write-output", type=Path, default=None)
    args = parser.parse_args(argv)

    config = load_run_config(args.run_config)
    if args.mode == "init-only":
        state = initialize_tracers(
            config.initial_restart,
            config.species_database,
            template_path=config.grid_template,
        )
        result = None
    else:
        result = run_emissions_replay(config, max_steps=args.max_steps)
        state = result.state

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

    comparison_path = config.comparison.get("species_conc_sample")
    if comparison_path:
        reference = load_species_conc(config.root / comparison_path)
        time_index = int(config.comparison.get("species_conc_time_index", -1))
        with netCDF4.Dataset(config.grid_template) as dataset:
            delp_dry_hpa = np.asarray(dataset.variables["Met_DELPDRY"][:])
            area_m2 = np.asarray(dataset.variables["AREA"][:])
        metrics = compare_to_time_slice(
            state,
            reference,
            reference_time_index=time_index,
            species=species,
            delp_dry_hpa=delp_dry_hpa,
            area_m2=area_m2,
        )
        print(format_metrics(metrics))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
