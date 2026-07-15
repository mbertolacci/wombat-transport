from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
import gzip
import json
from types import SimpleNamespace
import subprocess
import sys

import netCDF4
import numpy as np
import pytest
from yaml12 import write_yaml

from wombat_transport.compare import compare_to_time_slice, tracer_mass_kg
from wombat_transport.emissions import EmissionsOperator
from wombat_transport.fields import TracerField
from wombat_transport.grid import load_transport_grid
from wombat_transport.io import FIXED_GRID, initialize_tracers, load_hemco_emissions, load_species_conc, load_restart
from wombat_transport.run_config import load_run_config, logging_level, meteorology_chunk_multiple, meteorology_root
from wombat_transport.runner import (
    RUN_METADATA_NAME,
    _is_time_for_emissions,
    _load_emissions_operator,
    _load_simulation_forcing,
    _validate_timestep_schedule,
    has_invalid_emissions,
    run_tracer_simulation,
)
from wombat_transport.species import load_species_database
from tests.data_paths import (
    BASE_CONFIG,
    RESIDUAL_CONFIG,
    requires_residual_data,
    requires_restart,
    requires_transport_data,
)


@requires_restart
def test_configured_residual_emissions_config_covers_expected_species():
    config = load_run_config(RESIDUAL_CONFIG)
    assert config.emissions == "emissions.yml"
    operator = _residual_emissions_operator(config)

    assert len(operator.config.scales) == 32
    assert len(operator.config.fields) == 23
    assert "r0002p001s001" not in operator.emitted_species
    assert operator.emitted_species[0] == "r0002p001s002"
    assert operator.emitted_species[-1] == "r0002p001s024"


@requires_residual_data
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
                "path_template": "../../../../../external_data/fluxes/SOM_FFN_vBAMS2024v2_residual.nc",
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


def test_run_config_logging_level_defaults_and_validates():
    config = load_run_config(RESIDUAL_CONFIG)

    assert logging_level(config) == "info"
    assert logging_level(replace(config, logging={})) == "warning"
    assert logging_level(replace(config, logging={"level": "DEBUG"})) == "debug"
    with pytest.raises(ValueError, match="logging.level"):
        logging_level(replace(config, logging={"level": "trace"}))


def test_run_config_meteorology_chunk_multiple_defaults_and_validates():
    config = load_run_config(RESIDUAL_CONFIG)

    assert meteorology_chunk_multiple(config) == 1
    assert meteorology_chunk_multiple(replace(config, meteorology={"root": "met", "chunk_multiple": 2})) == 2
    with pytest.raises(ValueError, match="meteorology.chunk_multiple"):
        meteorology_chunk_multiple(replace(config, meteorology={"root": "met", "chunk_multiple": 0}))
    with pytest.raises(ValueError, match="meteorology.chunk_multiple"):
        meteorology_chunk_multiple(replace(config, meteorology={"root": "met", "chunk_multiple": "many"}))


def test_simulation_forcing_uses_provider_timestamps():
    calls = []

    class FakeProvider:
        def forcing_for_step(self, current, *, dt_s):
            forcing = object()
            calls.append((current, dt_s, forcing))
            return forcing

    provider = FakeProvider()
    start = datetime(2014, 9, 1)

    first = _load_simulation_forcing(provider, start, transport_dt_s=600.0)
    same = _load_simulation_forcing(provider, start + timedelta(minutes=10), transport_dt_s=600.0)
    next_met = _load_simulation_forcing(provider, start + timedelta(hours=3), transport_dt_s=600.0)

    assert first is calls[0][2]
    assert same is calls[1][2]
    assert next_met is calls[2][2]
    assert [call[0] for call in calls] == [
        start,
        start + timedelta(minutes=10),
        start + timedelta(hours=3),
    ]


@requires_residual_data
def test_tracer_simulation_uses_configured_residual_emissions_source(tmp_path):
    config = _isolated_config(load_run_config(RESIDUAL_CONFIG), tmp_path, outputs={})

    result = run_tracer_simulation(config, max_steps=1)

    assert result.transport_steps == 1
    assert result.emissions_steps == 1
    assert len(result.emissions_processed) == 1
    assert result.emissions_processed[0].timestamp == datetime(2014, 9, 1, 0, 10)
    assert np.isfinite(result.total_emitted_mass)
    assert result.emitted_mass_by_tracer[0] == 0.0
    assert result.emitted_mass_by_tracer[1] != 0.0
    assert result.emitted_mass_by_tracer[12] != 0.0


@requires_restart
def test_tracer_simulation_holds_active_emissions_for_transport_substeps(monkeypatch, tmp_path):
    config = _isolated_config(load_run_config(RESIDUAL_CONFIG), tmp_path, outputs={})
    initial = initialize_tracers(config.initial_restart, config.species_database, template_path=config.grid_template)
    active_emissions_seen = []
    state_inputs = []
    validation_flags = []

    def fake_load_forcing(*args, **kwargs):
        return SimpleNamespace(
            dry_surface_pressure_start_hpa=np.full((1, FIXED_GRID["lat"], FIXED_GRID["lon"]), 1000.0),
            dry_surface_pressure_hpa=np.full((1, FIXED_GRID["lat"], FIXED_GRID["lon"]), 1000.0),
        )

    def fake_run_transport_one_step(
        tracer_field,
        forcing,
        grid,
        *,
        dt_s,
        active_emissions=None,
        surface_flux_to_vmr_factor=None,
        dry_air_mass_kg=None,
        tpcore_static_terms=None,
        validate_tpcore_branches=True,
        consume_input=False,
    ):
        state_inputs.append(tracer_field.data.copy())
        active_emissions_seen.append(active_emissions)
        validation_flags.append(validate_tpcore_branches)
        assert consume_input
        assert surface_flux_to_vmr_factor is not None
        assert tpcore_static_terms is not None
        return SimpleNamespace(
            state=tracer_field,
            dry_air_mass_kg=dry_air_mass_kg,
            delp_dry_hpa=np.zeros(tracer_field.data.shape[:-1]),
        )

    monkeypatch.setattr("wombat_transport.runner._load_simulation_forcing", fake_load_forcing)
    monkeypatch.setattr("wombat_transport.runner.run_transport_one_step", fake_run_transport_one_step)

    result = run_tracer_simulation(config, max_steps=2)

    assert result.emissions_steps == 1
    assert len(active_emissions_seen) == 2
    assert validation_flags == [True, False]
    assert active_emissions_seen[0] is not None
    assert active_emissions_seen[0] is active_emissions_seen[1]
    np.testing.assert_array_equal(state_inputs[0], initial.data)
    np.testing.assert_array_equal(state_inputs[1], initial.data)


@requires_residual_data
def test_tracer_simulation_writes_configured_history_outputs(tmp_path):
    config = load_run_config(RESIDUAL_CONFIG)
    outputs = {
        "expid": str(tmp_path / "OutputDir" / "GEOSChem"),
        "collections": {
            "Restart": {
                "filename": str(tmp_path / "Restarts" / "GEOSChem.Restart.%y4%m2%d2_%h2%n2z.nc4"),
                "frequency": "00000000 001000",
                "duration": "00000000 001000",
                "mode": "instantaneous",
                "fields": ["SpeciesRst_?ALL?", "Met_DELPDRY", "Met_PS1WET", "Met_PS1DRY", "Met_SPHU1", "Met_TMPU1"],
            },
            "SpeciesConcThreeHourly": {
                "template": "%y4%m2%d2_%h2%n2z.nc4",
                "frequency": "00000000 030000",
                "duration": "00000001 000000",
                "mode": "time-averaged",
                "fields": ["SpeciesConcVV_?ADV?"],
            },
        },
    }

    run_tracer_simulation(_isolated_config(config, tmp_path, outputs=outputs), max_steps=1)

    species_conc = tmp_path / "OutputDir" / "GEOSChem.SpeciesConcThreeHourly.20140901_0000z.nc4"
    restart = tmp_path / "Restarts" / "GEOSChem.Restart.20140901_0010z.nc4"
    assert species_conc.exists()
    assert restart.exists()
    assert load_species_conc(species_conc).shape[-1] == 24
    assert load_restart(restart, load_species_database(config.species_database)).shape[-1] == 24
    with netCDF4.Dataset(restart) as dataset:
        assert "Met_DELPDRY" in dataset.variables
        assert "Met_PS1WET" in dataset.variables


@requires_residual_data
def test_tracer_simulation_samples_obsoperator_after_first_transport_step(tmp_path):
    config = load_run_config(RESIDUAL_CONFIG)
    first_name = initialize_tracers(
        config.initial_restart,
        config.species_database,
        template_path=config.grid_template,
    ).names[0]
    input_path = tmp_path / "obsoperator-20140901.yml.gz"
    with gzip.open(input_path, "wt", encoding="utf-8") as handle:
        write_yaml(
            {
                "entries": [
                    {
                        "id": "first-step",
                        "fields": f"SpeciesConcVV_{first_name}",
                        "time_operator": {"type": "point", "unit": "time_index", "time": 0},
                        "horizontal_operator": {
                            "type": "point",
                            "unit": "grid_index",
                            "longitude": 1,
                            "latitude": 1,
                        },
                        "vertical_operator": {"type": "point", "unit": "pressure_level", "value": 1},
                    },
                    {
                        "id": "unfinished",
                        "fields": f"SpeciesConcVV_{first_name}",
                        "time_operator": {
                            "type": "range",
                            "unit": "time_index",
                            "start": 0,
                            "end": 1,
                        },
                        "horizontal_operator": {
                            "type": "point",
                            "unit": "grid_index",
                            "longitude": 1,
                            "latitude": 1,
                        },
                        "vertical_operator": {"type": "point", "unit": "pressure_level", "value": 1},
                    },
                ]
            },
            handle,
        )
    output_template = tmp_path / "GEOSChem.ObsOperator.YYYYMMDD_hhmmz.nc4"
    restart_template = tmp_path / "Wombat.ObsOperator.Restart.YYYYMMDD_hhmmss.nc4"
    outputs = {
        "obsoperator": {
            "activate": True,
            "input_file": str(tmp_path / "obsoperator-YYYYMMDD.yml.gz"),
            "output_file": str(output_template),
            "restart_file": str(restart_template),
            "restart_missing": "ignore",
        }
    }

    result = run_tracer_simulation(_isolated_config(config, tmp_path, outputs=outputs), max_steps=1)

    output_path = tmp_path / "GEOSChem.ObsOperator.20140901_0000z.nc4"
    restart_path = tmp_path / "Wombat.ObsOperator.Restart.20140901_001000.nc4"
    assert restart_path.is_file()
    with netCDF4.Dataset(output_path) as dataset:
        np.testing.assert_allclose(
            dataset.variables["sample"][:],
            np.array([result.state.data[0, -1, 0, 0, 0]], dtype=np.float32),
        )
    with netCDF4.Dataset(restart_path) as dataset:
        assert _decode_char_rows(dataset.variables["id"][:]) == ["unfinished"]
        assert dataset.restart_time_us == 1409530200000000


def test_invalid_hemco_fill_values_are_detected():
    invalid = "tests/fixtures/io_readers_v1/hemco_invalid.nc4"
    valid = "tests/fixtures/io_readers_v1/hemco.nc4"

    assert has_invalid_emissions(load_hemco_emissions(invalid))
    assert not has_invalid_emissions(load_hemco_emissions(valid))


@requires_restart
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


@requires_restart
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


@requires_residual_data
def test_run_cli_default_configured_coupled_run_smoke(tmp_path):
    config_path = _write_temp_residual_run_config(tmp_path, log_level="info")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "wombat_transport.run",
            config_path,
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


@requires_residual_data
def test_run_cli_logs_info_messages_to_stderr(tmp_path):
    config_path = _write_temp_residual_run_config(tmp_path, log_level="info")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "wombat_transport.run",
            config_path,
            "--max-steps",
            "1",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "mode: run" in completed.stdout
    assert "INFO wombat_transport.runner simulation_start" in completed.stderr
    assert "INFO wombat_transport.runner transport_timestep step=1" in completed.stderr
    assert "DEBUG wombat_transport.runner" not in completed.stderr


@requires_residual_data
def test_run_cli_debug_logging_includes_runner_substeps(tmp_path):
    config_path = _write_temp_residual_run_config(tmp_path, log_level="debug")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "wombat_transport.run",
            config_path,
            "--max-steps",
            "1",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "mode: run" in completed.stdout
    assert "DEBUG wombat_transport.runner loading_forcing step=1" in completed.stderr
    assert "DEBUG wombat_transport.runner evaluating_emissions step=1" in completed.stderr
    assert "DEBUG wombat_transport.runner running_transport step=1" in completed.stderr
    assert "DEBUG wombat_transport.runner output_manager enabled=False" in completed.stderr
    metadata = json.loads((tmp_path / RUN_METADATA_NAME).read_text(encoding="utf-8"))
    assert metadata["kind"] == "wombat-run"
    assert metadata["run_name"] == "logging_smoke"
    assert "written_at_utc" in metadata
    assert {"available", "commit", "dirty", "tracked_dirty", "untracked_present"} <= set(metadata["git"])


@requires_restart
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


@requires_transport_data
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


@requires_transport_data
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


@requires_restart
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
    loaded = load_restart(
        output_path,
        load_species_database("validation_runs/cases/realistic_restart_noemis/wombat/main/species_database.yml"),
    )
    assert loaded.names == ("CO2",)
    assert loaded.shape == (1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"], 1)


def _residual_emissions_operator(config):
    species = load_species_database(config.species_database)
    grid = load_transport_grid(config.grid_template)
    return _load_emissions_operator(config, species, grid)


def _decode_char_rows(values):
    return [row.tobytes().split(b"\x00", 1)[0].decode("utf-8") for row in values]


def _write_temp_residual_run_config(tmp_path, *, log_level: str):
    config = load_run_config(RESIDUAL_CONFIG)
    raw = {
        "name": "logging_smoke",
        "source_run_dir": str(config.source_run_dir),
        "species_database": str(config.species_database),
        "initial_restart": None,
        "grid_template": str(config.grid_template),
        "output_dir": str(tmp_path / "OutputDir"),
        "simulation": config.simulation,
        "meteorology": {"root": str(config.root / config.meteorology["root"])},
        "emissions": str(config.root / config.emissions),
        "logging": {"level": log_level},
        "outputs": {},
        "diagnostics": {},
        "comparison": {},
    }
    path = tmp_path / "run.yml"
    write_yaml(raw, path)
    return path


def _isolated_config(config, tmp_path, *, outputs):
    emissions = config.emissions
    if isinstance(emissions, str):
        emissions = str((config.root / emissions).resolve())
    meteorology = dict(config.meteorology)
    meteorology["root"] = str(meteorology_root(config))
    return replace(
        config,
        root=tmp_path,
        source_run_dir=tmp_path,
        output_dir=tmp_path / "OutputDir",
        meteorology=meteorology,
        emissions=emissions,
        outputs=outputs,
    )
