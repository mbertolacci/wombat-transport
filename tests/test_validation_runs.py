from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import netCDF4
import numpy as np
import pytest


def _load_compare_module():
    path = Path(__file__).parents[1] / "tools" / "compare_validation_run.py"
    spec = importlib.util.spec_from_file_location("compare_validation_run", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


compare_validation_run = _load_compare_module()


def test_compare_validation_run_reports_species_conc_and_restart_metrics(tmp_path):
    case_dir = tmp_path / "cases" / "synthetic_restart_chain"
    case_dir.mkdir(parents=True)
    (case_dir / "case.yml").write_text(
        """
schema_version: 1
name: synthetic_restart_chain
title: Synthetic restart chain
modes:
  quick:
    comparisons: [species_conc]
  restart:
    comparisons: [species_conc, restart]
stages:
  - id: window1
    engines:
      geoschem:
        work_dir: "{work}/window1/geoschem"
      wombat:
        work_dir: "{work}/window1/wombat"
    comparisons:
      - id: species_conc
        kind: species_conc
        geoschem_glob: "OutputDir/GEOSChem.SpeciesConcThreeHourly.*.nc4"
        wombat_glob: "OutputDir/GEOSChem.SpeciesConcThreeHourly.*.nc4"
        fields: ["SpeciesConcVV_?ALL?"]
        tolerance_mode: report
      - id: restart
        kind: restart
        geoschem_glob: "Restarts/GEOSChem.Restart.20140902_0000z.nc4"
        wombat_glob: "Restarts/GEOSChem.Restart.20140902_0000z.nc4"
        fields: ["SpeciesRst_?ALL?", "Met_DELPDRY"]
        tolerance_mode: report
  - id: window2
    depends_on: window1
    engines:
      geoschem:
        work_dir: "{work}/window2/geoschem"
      wombat:
        work_dir: "{work}/window2/wombat"
    comparisons:
      - id: species_conc
        kind: species_conc
        geoschem_glob: "OutputDir/GEOSChem.SpeciesConcThreeHourly.*.nc4"
        wombat_glob: "OutputDir/GEOSChem.SpeciesConcThreeHourly.*.nc4"
        fields: ["SpeciesConcVV_?ALL?"]
        tolerance_mode: report
""",
        encoding="utf-8",
    )
    work_case = tmp_path / "work" / "synthetic_restart_chain"
    for stage in ("window1", "window2"):
        geoschem_output = work_case / stage / "geoschem" / "OutputDir"
        wombat_output = work_case / stage / "wombat" / "OutputDir"
        geoschem_output.mkdir(parents=True)
        wombat_output.mkdir(parents=True)
        filename = "GEOSChem.SpeciesConcThreeHourly.20140901_0000z.nc4"
        _write_species_conc(geoschem_output / filename, value=1.0)
        _write_species_conc(wombat_output / filename, value=1.25)

    geoschem_restart = work_case / "window1" / "geoschem" / "Restarts"
    wombat_restart = work_case / "window1" / "wombat" / "Restarts"
    geoschem_restart.mkdir()
    wombat_restart.mkdir()
    restart_name = "GEOSChem.Restart.20140902_0000z.nc4"
    _write_restart(geoschem_restart / restart_name, value=2.0)
    _write_restart(wombat_restart / restart_name, value=2.5)

    rows, output_dir = compare_validation_run.compare_case(case_dir, mode="restart", work_dir=tmp_path / "work")

    by_key = {(row.stage, row.comparison, row.variable): row for row in rows}
    assert by_key[("window1", "species_conc", "SpeciesConcVV_CO2")].max_abs_error == 0.25
    assert by_key[("window2", "species_conc", "SpeciesConcVV_CO2")].mean_abs_error == 0.25
    assert by_key[("window1", "restart", "SpeciesRst_CO2")].max_abs_error == 0.5
    assert by_key[("window1", "restart", "Met_DELPDRY")].max_abs_error == 0.5
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "summary.json").exists()


def test_compare_validation_run_errors_when_outputs_are_missing(tmp_path):
    case_dir = tmp_path / "cases" / "missing"
    case_dir.mkdir(parents=True)
    (case_dir / "case.yml").write_text(
        """
schema_version: 1
name: missing
modes:
  quick:
    comparisons: [species_conc]
stages:
  - id: main
    engines:
      geoschem:
        work_dir: "{work}/main/geoschem"
      wombat:
        work_dir: "{work}/main/wombat"
    comparisons:
      - id: species_conc
        kind: species_conc
        geoschem_glob: "OutputDir/GEOSChem.SpeciesConcThreeHourly.*.nc4"
        wombat_glob: "OutputDir/GEOSChem.SpeciesConcThreeHourly.*.nc4"
        fields: ["SpeciesConcVV_?ALL?"]
""",
        encoding="utf-8",
    )

    with pytest.raises(compare_validation_run.ValidationRunError, match="no files matched"):
        compare_validation_run.compare_case(case_dir, mode="quick", work_dir=tmp_path / "work")


def _write_species_conc(path: Path, *, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("time", 2)
        dataset.createDimension("lev", 2)
        dataset.createDimension("lat", 1)
        dataset.createDimension("lon", 1)
        variable = dataset.createVariable("SpeciesConcVV_CO2", "f8", ("time", "lev", "lat", "lon"))
        variable[:] = value


def _write_restart(path: Path, *, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("time", 1)
        dataset.createDimension("lev", 2)
        dataset.createDimension("lat", 1)
        dataset.createDimension("lon", 1)
        species = dataset.createVariable("SpeciesRst_CO2", "f8", ("time", "lev", "lat", "lon"))
        species[:] = value
        delp = dataset.createVariable("Met_DELPDRY", "f8", ("time", "lev", "lat", "lon"))
        delp[:] = value + 10.0
