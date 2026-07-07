from __future__ import annotations

import subprocess
import sys

import netCDF4
import numpy as np

from wombat_transport.compare import compare_to_time_slice, tracer_mass_kg
from wombat_transport.fields import TracerField
from wombat_transport.io import FIXED_GRID, initialize_tracers, load_hemco_emissions, load_species_conc, load_restart
from wombat_transport.run_config import load_run_config
from wombat_transport.runner import (
    discover_hemco_diagnostics,
    emitted_mass_by_tracer_for_step,
    has_invalid_emissions,
    parse_hemco_timestamp,
    run_emissions_replay,
)
from wombat_transport.species import load_species_database

BASE_CONFIG = "base_wombat/run.yml"
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
    assert result.state.shape == (1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"], 24)

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


def test_mass_metrics_are_zero_for_identical_base_field():
    config = load_run_config(BASE_CONFIG)
    species = load_species_database(config.species_database)
    field = initialize_tracers(config.initial_restart, config.species_database)
    with netCDF4.Dataset(config.grid_template) as dataset:
        delp = np.asarray(dataset.variables["Met_DELPDRY"][:])
        area = np.asarray(dataset.variables["AREA"][:])

    metrics = compare_to_time_slice(
        field,
        field,
        species=species,
        delp_dry_hpa=delp,
        area_m2=area,
    )

    assert metrics.mass_error_kg is not None
    assert metrics.max_abs_column_error_kg is not None
    np.testing.assert_array_equal(metrics.mass_error_kg, np.zeros(1))
    np.testing.assert_array_equal(metrics.max_abs_column_error_kg, np.zeros(1))


def test_mass_metrics_match_controlled_uniform_perturbation():
    config = load_run_config(BASE_CONFIG)
    species = load_species_database(config.species_database)
    reference = initialize_tracers(config.initial_restart, config.species_database)
    perturbation = 1.0e-9
    candidate = TracerField(
        names=reference.names,
        data=reference.data + perturbation,
        units=reference.units,
        coords=reference.coords,
    )
    with netCDF4.Dataset(config.grid_template) as dataset:
        delp = np.asarray(dataset.variables["Met_DELPDRY"][:])
        area = np.asarray(dataset.variables["AREA"][:])

    metrics = compare_to_time_slice(
        candidate,
        reference,
        species=species,
        delp_dry_hpa=delp,
        area_m2=area,
    )
    expected_mass_error = tracer_mass_kg(candidate.data - reference.data, species, delp, area)

    assert metrics.mass_error_kg is not None
    assert metrics.max_abs_column_error_kg is not None
    np.testing.assert_allclose(metrics.mass_error_kg, expected_mass_error)
    np.testing.assert_allclose(metrics.max_abs_error[0], perturbation)
    assert metrics.max_abs_column_error_kg[0] > 0.0


def test_residual_full_replay_has_finite_mass_metrics_after_skipping_invalid_tail():
    config = load_run_config(RESIDUAL_CONFIG)
    species = load_species_database(config.species_database)
    result = run_emissions_replay(config)
    reference = load_species_conc(config.root / config.comparison["species_conc_sample"])
    with netCDF4.Dataset(config.grid_template) as dataset:
        delp = np.asarray(dataset.variables["Met_DELPDRY"][:])
        area = np.asarray(dataset.variables["AREA"][:])

    metrics = compare_to_time_slice(
        result.state,
        reference,
        reference_time_index=int(config.comparison["species_conc_time_index"]),
        species=species,
        delp_dry_hpa=delp,
        area_m2=area,
    )

    assert len(result.skipped_files) == 1
    assert metrics.mass_error_kg is not None
    assert np.all(np.isfinite(metrics.mass_error_kg))
    assert np.all(np.isfinite(metrics.max_abs_column_error_kg))


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
    assert "tracer,max_abs_error,mean_abs_error,candidate_mass_kg" in completed.stdout


def test_run_cli_base_init_only_smoke():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "wombat_transport.run",
            BASE_CONFIG,
            "--mode",
            "init-only",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "mode: init-only" in completed.stdout
    assert "state_shape: (1, 47, 91, 144, 1)" in completed.stdout
    assert "tracer,max_abs_error,mean_abs_error,candidate_mass_kg" in completed.stdout


def test_run_cli_base_transport_one_step_smoke():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "wombat_transport.run",
            BASE_CONFIG,
            "--mode",
            "transport-one-step",
            "--max-steps",
            "1",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "mode: transport-one-step" in completed.stdout
    assert "transport_operators: tpcore,vdiff,convection" in completed.stdout
    assert "transport_steps: 1" in completed.stdout
    assert "tpcore_max_scalar_mass_error:" in completed.stdout
    assert "vdiff_max_scalar_mass_error:" in completed.stdout
    assert "convection_max_scalar_mass_error:" in completed.stdout
    assert "max_transport_scalar_mass_error:" in completed.stdout
    assert "tracer,max_abs_error,mean_abs_error,candidate_mass_kg" in completed.stdout


def test_run_cli_base_transport_window_smoke():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "wombat_transport.run",
            BASE_CONFIG,
            "--mode",
            "transport-window",
            "--max-steps",
            "2",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "mode: transport-window" in completed.stdout
    assert "transport_operators: tpcore,vdiff,convection" in completed.stdout
    assert "transport_steps: 2" in completed.stdout
    assert "tpcore_max_scalar_mass_error:" in completed.stdout
    assert "vdiff_max_scalar_mass_error:" in completed.stdout
    assert "convection_max_scalar_mass_error:" in completed.stdout
    assert "max_transport_scalar_mass_error:" in completed.stdout
    assert "tracer,max_abs_error,mean_abs_error,candidate_mass_kg" in completed.stdout
    assert "pressure_dry_max_abs_error_hpa:" in completed.stdout


def test_run_cli_writes_restart_like_output(tmp_path):
    output_path = tmp_path / "wombat_restart.nc4"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "wombat_transport.run",
            BASE_CONFIG,
            "--mode",
            "init-only",
            "--write-output",
            str(output_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert f"wrote_output: {output_path}" in completed.stdout
    loaded = load_restart(output_path, load_species_database("base/species_database.yml"))
    assert loaded.names == ("CO2",)
    assert loaded.shape == (1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"], 1)
