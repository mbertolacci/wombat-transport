from __future__ import annotations

import argparse
from pathlib import Path

from wombat_transport.compare import compare_to_time_slice, format_metrics
from wombat_transport.io import load_species_conc
from wombat_transport.run_config import load_run_config
from wombat_transport.runner import run_emissions_replay


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Wombat transport prototype mode")
    parser.add_argument("run_config", type=Path)
    parser.add_argument("--mode", choices=["emissions-only"], default="emissions-only")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_run_config(args.run_config)
    result = run_emissions_replay(config, max_steps=args.max_steps)

    print(f"name: {config.name}")
    print(f"mode: {args.mode}")
    print(f"state_shape: {result.state.shape}")
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
        metrics = compare_to_time_slice(result.state, reference, reference_time_index=time_index)
        print(format_metrics(metrics))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
