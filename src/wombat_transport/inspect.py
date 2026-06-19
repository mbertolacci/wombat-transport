from __future__ import annotations

import argparse
from pathlib import Path

from wombat_transport.io import (
    initialize_tracers,
    load_hemco_emissions,
    load_species_conc,
)
from wombat_transport.run_config import load_run_config
from wombat_transport.species import load_species_database


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a wombat transport run config")
    parser.add_argument("run_config", type=Path)
    args = parser.parse_args(argv)

    config = load_run_config(args.run_config)
    species = load_species_database(config.species_database)
    initialized = initialize_tracers(
        config.initial_restart,
        config.species_database,
        template_path=config.grid_template,
    )

    print(f"name: {config.name}")
    print(f"source_run_dir: {config.source_run_dir}")
    print(f"species_count: {len(species)}")
    print(f"initialized_shape: {initialized.shape}")
    print(f"tracers: {', '.join(initialized.names)}")

    species_conc = config.diagnostics.get("species_conc_sample")
    if species_conc:
        field = load_species_conc(config.root / species_conc)
        print(f"species_conc_sample_shape: {field.shape}")

    hemco = config.diagnostics.get("hemco_sample")
    if hemco:
        try:
            field = load_hemco_emissions(config.root / hemco)
        except KeyError as exc:
            print(f"hemco_sample: {exc}")
        else:
            print(f"hemco_sample_shape: {field.shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
