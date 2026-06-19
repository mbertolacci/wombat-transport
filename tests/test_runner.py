from __future__ import annotations

import subprocess
import sys

import numpy as np

from wombat_transport.compare import compare_to_time_slice
from wombat_transport.io import FIXED_GRID, load_hemco_emissions, load_species_conc
from wombat_transport.run_config import load_run_config
from wombat_transport.runner import (
    discover_hemco_diagnostics,
    emitted_mass_by_tracer_for_step,
    has_invalid_emissions,
    parse_hemco_timestamp,
    run_emissions_replay,
)

RESIDUAL_CONFIG = "residual_20140901_part001_split01_wombat/run.yml"


def test_parse_hemco_timestamp_from_filename():
    timestamp = parse_hemco_timestamp("HEMCO_diagnostics.201409010030.nc")

    assert timestamp.year == 2014
    assert timestamp.month == 9
    assert timestamp.day == 1
    assert timestamp.hour == 0
    assert timestamp.minute == 30


def test_discover_hemco_diagnostics_orders_configured_window():
    config = load_run_config(RESIDUAL_CONFIG)
    files = discover_hemco_diagnostics(config)

    assert len(files) == 119
    assert files[0].path.name == "HEMCO_diagnostics.201409010030.nc"
    assert files[-1].path.name == "HEMCO_diagnostics.201409052230.nc"
    assert [item.timestamp for item in files] == sorted(item.timestamp for item in files)


def test_emissions_replay_processes_prefix_and_accumulates_mass():
    config = load_run_config(RESIDUAL_CONFIG)
    result = run_emissions_replay(config, max_steps=2)

    assert len(result.discovered_files) == 119
    assert len(result.processed_files) == 2
    assert result.state.shape == (24, 1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"])

    expected = np.zeros(24)
    for diagnostic in result.processed_files:
        expected += emitted_mass_by_tracer_for_step(load_hemco_emissions(diagnostic.path), dt_s=3600.0)
    np.testing.assert_allclose(result.emitted_mass_by_tracer, expected)


def test_invalid_hemco_fill_values_are_detected():
    config = load_run_config(RESIDUAL_CONFIG)
    files = discover_hemco_diagnostics(config)

    assert has_invalid_emissions(load_hemco_emissions(files[-1].path))
    assert not has_invalid_emissions(load_hemco_emissions(files[-2].path))


def test_comparison_metrics_return_one_value_per_tracer():
    config = load_run_config(RESIDUAL_CONFIG)
    result = run_emissions_replay(config, max_steps=1)
    reference = load_species_conc(config.root / config.comparison["species_conc_sample"])

    metrics = compare_to_time_slice(
        result.state,
        reference,
        reference_time_index=int(config.comparison["species_conc_time_index"]),
    )

    assert metrics.names == result.state.names
    assert metrics.max_abs_error.shape == (24,)
    assert metrics.mean_abs_error.shape == (24,)
    assert np.all(np.isfinite(metrics.max_abs_error))
    assert np.all(np.isfinite(metrics.mean_abs_error))


def test_run_cli_smoke_with_prefix():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "wombat_transport.run",
            RESIDUAL_CONFIG,
            "--mode",
            "emissions-only",
            "--max-steps",
            "1",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "emissions_files_discovered: 119" in completed.stdout
    assert "emissions_files_processed: 1" in completed.stdout
    assert "emissions_files_skipped: 0" in completed.stdout
    assert "total_emitted_mass_kg:" in completed.stdout
    assert "tracer,max_abs_error,mean_abs_error" in completed.stdout
