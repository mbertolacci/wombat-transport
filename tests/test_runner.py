from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import subprocess
import sys

import netCDF4
import numpy as np
import pytest

from wombat_transport.compare import compare_to_time_slice, tracer_mass_kg
from wombat_transport.emissions import EmissionsOperator
from wombat_transport.fields import TracerField
from wombat_transport.grid import load_transport_grid
from wombat_transport.io import FIXED_GRID, initialize_tracers, load_hemco_emissions, load_species_conc, load_restart
from wombat_transport.run_config import load_run_config
from wombat_transport.runner import (
    _is_time_for_emissions,
    _load_emissions_operator,
    _validate_timestep_schedule,
    has_invalid_emissions,
    run_tracer_simulation,
)
from wombat_transport.species import load_species_database

BASE_CONFIG = "base_wombat/run.yml"
RESIDUAL_CONFIG = "residual_20140901_part001_split01_wombat/run.yml"


def test_configured_residual_emissions_config_covers_expected_species():
    config = load_run_config(RESIDUAL_CONFIG)
    assert config.emissions == "emissions.yml"
    operator = _residual_emissions_operator(config)

    assert len(operator.config.scales) == 32
    assert len(operator.config.fields) == 23
    assert "r0002p001s001" not in operator.emitted_species
    assert operator.emitted_species[0] == "r0002p001s002"
    assert operator.emitted_species[-1] == "r0002p001s024"


def test_inline_emissions_mapping_is_accepted_in_run_config():
    config = load_run_config(RESIDUAL_CONFIG)
    inline = {
        "unit_conversion": "none",
        "missing_species": "zero",
        "scales": {},
        "fields": [
            {
                "name": "field_a",
                "species": "r0002p001s001",
                "path_template": "../fluxes/SOM_FFN_vBAMS2024v2_residual.nc",
                "variable": "residual",
                "frequency": "monthly",
                "dimensions": "xy",
            }
        ],
    }
    config = replace(config, emissions=inline)

    operator = _residual_emissions_operator(config)
    emissions = operator.evaluate(datetime(2014, 9, 1, 0, 30))

    assert emissions.names[0] == "r0002p001s001"
    assert emissions.shape[-1] == 24


def test_configured_residual_emissions_match_hemco_diagnostic_sample():
    config = load_run_config(RESIDUAL_CONFIG)
    operator = _residual_emissions_operator(config)
    expected = load_hemco_emissions(config.root / config.diagnostics["hemco_sample"])

    actual = operator.evaluate(datetime(2014, 9, 1, 0, 30))

    assert actual.shape == expected.shape
    assert actual.names == expected.names
    np.testing.assert_allclose(actual.data, expected.data, rtol=2.0e-5, atol=1.0e-10)


def test_geos_chem_emissions_schedule_uses_centered_emissions_timestep():
    assert _is_time_for_emissions(0, 600.0, 1200.0)
    assert not _is_time_for_emissions(600, 600.0, 1200.0)
    assert _is_time_for_emissions(1200, 600.0, 1200.0)

    assert not _is_time_for_emissions(0, 600.0, 3600.0)
    assert not _is_time_for_emissions(600, 600.0, 3600.0)
    assert _is_time_for_emissions(1200, 600.0, 3600.0)


def test_emissions_timestep_must_be_transport_multiple():
    with pytest.raises(ValueError, match="integer multiple"):
        _validate_timestep_schedule(600.0, 1000.0)


def test_tracer_simulation_uses_configured_residual_emissions_source():
    config = load_run_config(RESIDUAL_CONFIG)

    result = run_tracer_simulation(config, max_steps=1)

    assert result.transport_steps == 1
    assert result.emissions_steps == 1
    assert len(result.emissions_processed) == 1
    assert result.emissions_processed[0].timestamp == datetime(2014, 9, 1, 0, 10)
    assert np.isfinite(result.total_emitted_mass)
    assert result.emitted_mass_by_tracer[0] == 0.0
    assert result.emitted_mass_by_tracer[1] != 0.0
    assert result.emitted_mass_by_tracer[12] != 0.0


def test_invalid_hemco_fill_values_are_detected():
    config = load_run_config(RESIDUAL_CONFIG)
    invalid = config.root / "../residual_20140901_part001_split01/OutputDir/HEMCO_diagnostics.201409052230.nc"
    valid = config.root / "../residual_20140901_part001_split01/OutputDir/HEMCO_diagnostics.201409052130.nc"

    assert has_invalid_emissions(load_hemco_emissions(invalid))
    assert not has_invalid_emissions(load_hemco_emissions(valid))


def test_comparison_metrics_return_one_value_per_tracer():
    config = load_run_config(RESIDUAL_CONFIG)
    result = run_tracer_simulation(config, max_steps=1)
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


def test_run_cli_default_configured_coupled_run_smoke():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "wombat_transport.run",
            RESIDUAL_CONFIG,
            "--max-steps",
            "1",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "mode: run" in completed.stdout
    assert "transport_operators: tpcore,vdiff,convection" in completed.stdout
    assert "transport_steps: 1" in completed.stdout
    assert "emissions_steps: 1" in completed.stdout
    assert "emissions_dt_s: 1.20000000e+03" in completed.stdout
    assert "total_emitted_mass_kg:" in completed.stdout


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


def _residual_emissions_operator(config):
    species = load_species_database(config.species_database)
    grid = load_transport_grid(config.grid_template)
    return _load_emissions_operator(config, species, grid)
