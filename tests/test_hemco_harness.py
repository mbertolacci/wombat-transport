from __future__ import annotations

import os
from datetime import datetime

import netCDF4
import numpy as np
import pytest
import yaml

from wombat_transport.emissions import EmissionsOperator
from wombat_transport.hemco_harness import (
    SPECIES,
    TARGET_LAT,
    TARGET_LON,
    compare_scenario,
    find_hemco_standalone,
    run_hemco_standalone,
    scenario_names,
    write_scenario_run_directory,
)
from wombat_transport.grid import TransportGrid


def test_hemco_harness_generates_expected_run_directory(tmp_path):
    run_dir = write_scenario_run_directory("source_regrid_then_scale", tmp_path / "run")

    expected = {
        "HEMCO_sa_Config.rc",
        "HEMCO_Config.rc",
        "HEMCO_Diagn.rc",
        "HEMCO_sa_Grid.rc",
        "HEMCO_sa_Spec.rc",
        "HEMCO_sa_Time.rc",
        "wombat_emissions.yml",
        "scenario.yml",
    }
    assert expected <= {path.name for path in run_dir.iterdir()}
    assert (run_dir / "inputs" / "source_1x1.nc").exists()
    assert (run_dir / "inputs" / "scale.nc").exists()
    assert ">>>include HEMCO_Config.rc" in (run_dir / "HEMCO_sa_Config.rc").read_text(encoding="utf-8")
    assert "source_1x1 inputs/source_1x1.nc emis 2014/9/1/0 RF xy kg/m2/s A 1 1 1" in (
        run_dir / "HEMCO_Config.rc"
    ).read_text(encoding="utf-8")


def test_hemco_harness_wombat_config_mirrors_scenario(tmp_path):
    run_dir = write_scenario_run_directory("constant_and_file_scale", tmp_path / "run")
    config = yaml.safe_load((run_dir / "wombat_emissions.yml").read_text(encoding="utf-8"))

    assert config["unit_conversion"] == "none"
    assert config["missing_species"] == "zero"
    assert set(config["scales"]) == {"scale_a", "negative"}
    assert config["scales"]["negative"]["value"] == -2.0
    assert config["fields"][0]["scales"] == ["scale_a", "negative"]

    emissions = EmissionsOperator.from_yaml(
        run_dir / "wombat_emissions.yml",
        root=run_dir,
        species=list(SPECIES),
        grid=_grid_from_generated(run_dir),
    ).evaluate(datetime(2014, 9, 1))
    assert emissions.shape == (1, 47, 91, 144, 2)
    assert np.count_nonzero(emissions.data[..., 0]) > 0
    assert np.count_nonzero(emissions.data[..., 1]) == 0


def test_all_hemco_harness_scenarios_generate_wombat_output(tmp_path):
    for scenario in scenario_names():
        run_dir = write_scenario_run_directory(scenario, tmp_path / scenario)
        emissions = EmissionsOperator.from_yaml(
            run_dir / "wombat_emissions.yml",
            root=run_dir,
            species=list(SPECIES),
            grid=_grid_from_generated(run_dir),
        ).evaluate(datetime(2014, 9, 1))

        assert emissions.shape == (1, 47, 91, 144, 2)
        assert np.all(np.isfinite(emissions.data))


def test_hemco_harness_npft_scenario_writes_hemco_selection_syntax(tmp_path):
    run_dir = write_scenario_run_directory("npft_select_with_scale", tmp_path / "run")

    text = (run_dir / "HEMCO_Config.rc").read_text(encoding="utf-8")
    assert 'xy+"npft=4"' in text
    with netCDF4.Dataset(run_dir / "inputs" / "source_npft.nc") as dataset:
        assert dataset.variables["emis"].dimensions == ("time", "npft", "lat", "lon")
        np.testing.assert_array_equal(dataset.variables["npft"][:], np.array([1, 2, 3, 4, 5]))


@pytest.mark.skipif(find_hemco_standalone() is None, reason="HEMCO_STANDALONE is not available")
@pytest.mark.parametrize("scenario", ["same_grid_file_scale", "source_regrid_then_scale", "source_and_scale_regrid"])
def test_hemco_harness_optional_standalone_compare(tmp_path, scenario):
    run_dir = write_scenario_run_directory(scenario, tmp_path / scenario)
    diagnostic = run_hemco_standalone(run_dir, os.environ.get("HEMCO_STANDALONE"))

    comparisons = compare_scenario(run_dir, diagnostic)

    assert {item.species for item in comparisons} == {"A", "B"}
    for item in comparisons:
        assert item.bottom_level_only
        assert item.nonzero_mismatch_count == 0
        assert item.max_abs_error < 1.0e-10


def _grid_from_generated(run_dir) -> TransportGrid:
    return TransportGrid(
        lat_deg=TARGET_LAT.copy(),
        lon_deg=TARGET_LON.copy(),
        lev=np.arange(47.0, 0.0, -1.0),
        area_m2=np.ones((TARGET_LAT.size, TARGET_LON.size), dtype=np.float64),
        hyai_hpa=np.linspace(0.0, 1000.0, 48),
        hybi=np.linspace(0.0, 1.0, 48),
        template_path=run_dir / "HEMCO_sa_Grid.rc",
    )
