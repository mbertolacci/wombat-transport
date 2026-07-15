from __future__ import annotations

from pathlib import Path

import numpy as np

from wombat_transport.io import (
    FIXED_GRID,
    initialize_tracers,
    load_base_met,
    load_hemco_emissions,
    load_restart,
    load_species_conc,
    write_restart_like,
)
from wombat_transport.run_config import load_run_config
from wombat_transport.species import load_species_database

FIXTURE_ROOT = "tests/fixtures/io_readers_2x25_v1"
BASE_RESTART = f"{FIXTURE_ROOT}/restart.nc4"
BASE_SPECIES = "validation_runs/cases/realistic_restart_noemis_2x25/wombat/main/species_database.yml"
RESIDUAL_SPECIES = "validation_runs/cases/residual_24tracer_emissions_1day_2x25/wombat/main/species_database.yml"
RESIDUAL_MONTHLY_RESTART = BASE_RESTART
BASE_SPECIES_CONC = f"{FIXTURE_ROOT}/base_species_conc.nc4"
RESIDUAL_SPECIES_CONC = f"{FIXTURE_ROOT}/residual_species_conc.nc4"
RESIDUAL_HEMCO = f"{FIXTURE_ROOT}/hemco.nc4"
BASE_MET = f"{FIXTURE_ROOT}/met_diagnostics.nc4"


def test_species_database_parses_backgrounds():
    base = load_species_database(BASE_SPECIES)
    residual = load_species_database(RESIDUAL_SPECIES)

    assert [item.name for item in base] == ["CO2"]
    assert base[0].background_vv == 0.000355
    assert len(residual) == 24
    assert all(item.background_vv == 0.0004 for item in residual)


def test_base_restart_initializes_from_restart_variable():
    initialized = initialize_tracers(BASE_RESTART, BASE_SPECIES)
    direct = load_restart(BASE_RESTART, load_species_database(BASE_SPECIES))

    assert initialized.names == ("CO2",)
    assert initialized.shape == (1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"], 1)
    np.testing.assert_array_equal(initialized.data, direct.data.astype(np.float32).astype(np.float64))
    assert not np.all(initialized.data == 0.000355)


def test_residual_missing_restart_initializes_from_background():
    initialized = initialize_tracers(
        None,
        RESIDUAL_SPECIES,
        template_path=BASE_RESTART,
    )

    assert len(initialized.names) == 24
    assert initialized.shape == (1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"], 24)
    assert np.all(initialized.data == np.float32(0.0004).astype(np.float64))


def test_residual_monthly_restart_stacks_in_species_order():
    species = load_species_database(RESIDUAL_SPECIES)
    restart = load_restart(RESIDUAL_MONTHLY_RESTART, species)

    assert restart.names == tuple(item.name for item in species)
    assert restart.shape == (1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"], 24)


def test_write_restart_like_roundtrips_base_initialized_field(tmp_path):
    initialized = initialize_tracers(BASE_RESTART, BASE_SPECIES)
    output_path = tmp_path / "base_restart_like.nc4"

    write_restart_like(output_path, initialized, BASE_RESTART)
    loaded = load_restart(output_path, load_species_database(BASE_SPECIES))

    assert loaded.names == initialized.names
    assert loaded.shape == initialized.shape
    assert loaded.units == initialized.units
    np.testing.assert_array_equal(loaded.data, initialized.data)


def test_write_restart_like_roundtrips_residual_species_order(tmp_path):
    initialized = initialize_tracers(None, RESIDUAL_SPECIES, template_path=BASE_RESTART)
    output_path = tmp_path / "residual_restart_like.nc4"

    write_restart_like(output_path, initialized, BASE_RESTART)
    loaded = load_restart(output_path, load_species_database(RESIDUAL_SPECIES))

    assert loaded.names == initialized.names
    assert loaded.shape == initialized.shape
    assert loaded.units == initialized.units
    np.testing.assert_array_equal(loaded.data, initialized.data)


def test_species_conc_readers_stack_base_and_residual():
    base = load_species_conc(BASE_SPECIES_CONC)
    residual = load_species_conc(RESIDUAL_SPECIES_CONC)

    assert base.names == ("CO2",)
    assert base.shape == (1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"], 1)
    assert len(residual.names) == 24
    assert residual.shape == (1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"], 24)


def test_hemco_reader_stacks_residual_emissions():
    emissions = load_hemco_emissions(RESIDUAL_HEMCO)

    assert len(emissions.names) == 24
    assert emissions.shape == (1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"], 24)
    assert set(emissions.units) == {"kg/m2/s"}


def test_base_met_reader_loads_expected_variables():
    level_edge = load_base_met(BASE_MET)
    state_met = load_base_met(BASE_MET)

    expected = {"Met_PEDGE", "Met_PEDGEDRY", "Met_BXHEIGHT", "Met_AVGW"}
    assert set(level_edge) == expected
    assert set(state_met) == expected
    assert level_edge["Met_PEDGE"].shape == (1, 48, FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert state_met["Met_AVGW"].shape == (1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"])


def test_run_configs_resolve_fixture_paths():
    base = load_run_config("validation_runs/cases/realistic_restart_noemis_2x25/wombat/main/run.yml")
    residual = load_run_config("validation_runs/cases/residual_24tracer_emissions_1day_2x25/wombat/main/run.yml")

    assert base.initial_restart is not None
    external_root = (Path.cwd() / "external_data").resolve()
    assert base.initial_restart.is_relative_to(external_root)
    assert base.species_database.exists()
    assert residual.initial_restart is None
    assert residual.grid_template.is_relative_to(external_root)
    assert residual.species_database.exists()


def test_run_config_uses_yaml_12_core_scalar_semantics(tmp_path):
    path = tmp_path / "run.yml"
    path.write_text(
        """\
name: yes
source_run_dir: .
species_database: species.yml
grid_template: grid.nc
output_dir: output
comparison:
  legacy_flag: off
simulation:
  leading_zero: 0123
""",
        encoding="utf-8",
    )

    config = load_run_config(path)

    assert config.name == "yes"
    assert config.comparison["legacy_flag"] == "off"
    assert config.simulation["leading_zero"] == 123
