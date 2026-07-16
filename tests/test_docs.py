from pathlib import Path

from wombat_transport.output import parse_output_collections, parse_output_storage, parse_output_writer
from wombat_transport.run_config import (
    emissions_timestep_s,
    load_run_config,
    meteorology_chunk_multiple,
    meteorology_initial_time_index,
    meteorology_root,
    simulation_end,
    simulation_start,
    transport_timestep_s,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "basic_2x25"


def test_basic_documented_run_config_matches_the_public_example():
    config = load_run_config(EXAMPLE_DIR / "run.yml")

    assert config.name == "basic_2x25"
    assert config.root == EXAMPLE_DIR.resolve()
    assert config.source_run_dir == EXAMPLE_DIR.resolve()
    assert config.species_database == (EXAMPLE_DIR / "species_database.yml").resolve()
    assert config.initial_restart == (
        REPO_ROOT / "external_data/restarts/2x25/GEOSChem.Restart.20140901_0000z.nc4"
    ).resolve()
    assert config.grid_template == config.initial_restart
    assert config.output_dir == (EXAMPLE_DIR / "OutputDir").resolve()

    assert simulation_start(config).isoformat() == "2014-09-01T00:00:00"
    assert simulation_end(config).isoformat() == "2014-09-01T03:00:00"
    assert transport_timestep_s(config) == 600.0
    assert emissions_timestep_s(config) == 600.0
    assert meteorology_root(config) == (
        REPO_ROOT / "external_data/geoschem/GEOS_2x2.5/MERRA2"
    ).resolve()
    assert meteorology_initial_time_index(config) == 0
    assert meteorology_chunk_multiple(config) == 1

    writer = parse_output_writer(config.outputs)
    storage = parse_output_storage(config.outputs)
    collections = parse_output_collections(config.outputs)
    assert writer.mode == "sync"
    assert storage.dtype == "float32"
    assert storage.compression.enabled
    assert len(collections) == 1
    assert collections[0].name == "SpeciesConcThreeHourly"
    assert collections[0].duration == collections[0].frequency
    assert collections[0].fields == ("SpeciesConcVV_?ADV?",)
