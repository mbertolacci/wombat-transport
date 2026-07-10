from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
import yaml

from wombat_transport.emissions import EmissionsOperator
from wombat_transport.fields import public_tracer5_to_canonical
from wombat_transport.grid import TransportGrid
from wombat_transport.species import Species


HARNESS_START = datetime(2014, 9, 1, 0)
TARGET_LAT = np.array(
    [-89.5] + [float(value) for value in range(-88, 90, 2)] + [89.5],
    dtype=np.float64,
)
TARGET_LON = np.arange(-180.0, 180.0, 2.5, dtype=np.float64)
SOURCE_LAT_1X1 = np.arange(-89.5, 90.0, 1.0, dtype=np.float64)
SOURCE_LON_1X1 = np.arange(-179.5, 180.0, 1.0, dtype=np.float64)
SPECIES = (
    Species("A", molecular_weight_g=44.0, background_vv=0.0, full_name="A"),
    Species("B", molecular_weight_g=44.0, background_vv=0.0, full_name="B"),
)


@dataclass(frozen=True)
class HemcoHarnessComparison:
    scenario: str
    species: str
    max_abs_error: float
    mean_abs_error: float
    global_mass_error: float
    max_gridcell_mass_error: float
    nonzero_mismatch_count: int
    bottom_level_only: bool


def scenario_names() -> tuple[str, ...]:
    return tuple(_scenario_builders())


def write_scenario_run_directory(scenario: str, output_dir: str | Path) -> Path:
    """Write a standalone HEMCO run directory and matching Wombat config."""

    if scenario not in _scenario_builders():
        raise ValueError(f"unknown HEMCO harness scenario {scenario}")
    root = Path(output_dir)
    if root.exists():
        shutil.rmtree(root)
    (root / "inputs").mkdir(parents=True)
    (root / "OutputDir").mkdir()
    (root / "Restarts").mkdir()

    grid = _transport_grid(root)
    scenario_config = _scenario_builders()[scenario](root, grid)
    _write_hemco_sa_config(root)
    _write_hemco_grid_file(root)
    _write_hemco_time_file(root)
    _write_hemco_species_file(root)
    _write_hemco_diagnostics_file(root)
    _write_hemco_config(root, scenario_config)
    _write_wombat_config(root, scenario_config)
    _write_scenario_metadata(root, scenario, scenario_config)
    return root


def run_hemco_standalone(run_dir: str | Path, executable: str | Path | None = None) -> Path:
    """Run HEMCO standalone in a generated harness directory."""

    binary = Path(executable) if executable is not None else find_hemco_standalone()
    if binary is None:
        raise FileNotFoundError("hemco_standalone was not found; set HEMCO_STANDALONE")
    root = Path(run_dir)
    completed = subprocess.run(
        [str(binary), "-c", "HEMCO_sa_Config.rc"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    (root / "HEMCO_sa.stdout").write_text(completed.stdout, encoding="utf-8")
    (root / "HEMCO_sa.stderr").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"hemco_standalone failed with exit code {completed.returncode}")
    if "HEMCO_STANDALONE EXITED WITH ERROR" in completed.stdout or "HEMCO ERROR" in completed.stdout:
        raise RuntimeError(f"hemco_standalone reported an error; see {root / 'HEMCO_sa.stdout'}")
    diagnostics = sorted((root / "OutputDir").glob("HEMCO_sa_diagnostics*.nc*"))
    if not diagnostics:
        raise FileNotFoundError(f"{root / 'OutputDir'} contains no HEMCO_sa_diagnostics files")
    return diagnostics[0]


def compare_scenario(run_dir: str | Path, hemco_diagnostic: str | Path | None = None) -> tuple[HemcoHarnessComparison, ...]:
    """Compare a HEMCO standalone diagnostic with Wombat emissions output."""

    root = Path(run_dir)
    metadata = yaml.safe_load((root / "scenario.yml").read_text(encoding="utf-8"))
    scenario = str(metadata["scenario"])
    diagnostic_path = Path(hemco_diagnostic) if hemco_diagnostic is not None else _find_hemco_diagnostic(root)
    hemco = _load_hemco_diagnostic(diagnostic_path)
    grid = _transport_grid(root)
    wombat = EmissionsOperator.from_yaml("wombat_emissions.yml", root=root, species=list(SPECIES), grid=grid).evaluate(HARNESS_START)
    area_5d = grid.area_m2[np.newaxis, np.newaxis, :, :, np.newaxis]
    comparisons: list[HemcoHarnessComparison] = []
    for index, species in enumerate(wombat.names):
        difference = wombat.data[..., index] - hemco.data[..., index]
        mass_difference = difference * area_5d[..., 0]
        comparisons.append(
            HemcoHarnessComparison(
                scenario=scenario,
                species=species,
                max_abs_error=float(np.max(np.abs(difference))),
                mean_abs_error=float(np.mean(np.abs(difference))),
                global_mass_error=float(np.sum(mass_difference)),
                max_gridcell_mass_error=float(np.max(np.abs(mass_difference))),
                nonzero_mismatch_count=int(np.count_nonzero((wombat.data[..., index] != 0.0) != (hemco.data[..., index] != 0.0))),
                bottom_level_only=bool(np.all(wombat.data[:, :-1, :, :, index] == 0.0) and np.all(hemco.data[:, :-1, :, :, index] == 0.0)),
            )
        )
    return tuple(comparisons)


def write_comparison_csv(comparisons: tuple[HemcoHarnessComparison, ...], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(HemcoHarnessComparison.__dataclass_fields__))
        writer.writeheader()
        for item in comparisons:
            writer.writerow(item.__dict__)


def find_hemco_standalone() -> Path | None:
    env = os.environ.get("HEMCO_STANDALONE")
    candidates = [Path(env)] if env else []
    candidates.extend(
        [
            Path("tools/hemco_harness/build/bin/hemco_standalone"),
            Path("tools/hemco_harness/build/src/HEMCO/src/Interfaces/Standalone/hemco_standalone"),
            Path("GCClassic/src/HEMCO/build/bin/hemco_standalone"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate and compare HEMCO standalone emissions harness scenarios.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-scenarios")
    list_parser.set_defaults(func=_cli_list_scenarios)

    generate = subparsers.add_parser("generate")
    generate.add_argument("scenario", choices=scenario_names())
    generate.add_argument("output_dir", type=Path)
    generate.set_defaults(func=_cli_generate)

    run = subparsers.add_parser("run")
    run.add_argument("run_dir", type=Path)
    run.add_argument("--executable", type=Path, default=None)
    run.set_defaults(func=_cli_run)

    compare = subparsers.add_parser("compare")
    compare.add_argument("run_dir", type=Path)
    compare.add_argument("--diagnostic", type=Path, default=None)
    compare.add_argument("--csv", type=Path, default=None)
    compare.set_defaults(func=_cli_compare)

    args = parser.parse_args(argv)
    return int(args.func(args))


def _cli_list_scenarios(args) -> int:
    for name in scenario_names():
        print(name)
    return 0


def _cli_generate(args) -> int:
    path = write_scenario_run_directory(args.scenario, args.output_dir)
    print(path)
    return 0


def _cli_run(args) -> int:
    diagnostic = run_hemco_standalone(args.run_dir, args.executable)
    print(diagnostic)
    return 0


def _cli_compare(args) -> int:
    comparisons = compare_scenario(args.run_dir, args.diagnostic)
    if args.csv is not None:
        write_comparison_csv(comparisons, args.csv)
    print("scenario,species,max_abs_error,mean_abs_error,global_mass_error,max_gridcell_mass_error,nonzero_mismatch_count,bottom_level_only")
    for item in comparisons:
        print(
            f"{item.scenario},{item.species},{item.max_abs_error:.8e},{item.mean_abs_error:.8e},"
            f"{item.global_mass_error:.8e},{item.max_gridcell_mass_error:.8e},"
            f"{item.nonzero_mismatch_count},{item.bottom_level_only}"
        )
    return 0


def _scenario_builders():
    return {
        "same_grid_no_scale": _same_grid_no_scale,
        "same_grid_file_scale": _same_grid_file_scale,
        "constant_and_file_scale": _constant_and_file_scale,
        "source_regrid_then_scale": _source_regrid_then_scale,
        "source_and_scale_regrid": _source_and_scale_regrid,
        "npft_select_with_scale": _npft_select_with_scale,
        "multiple_entries_same_species": _multiple_entries_same_species,
    }


def _same_grid_no_scale(root: Path, grid: TransportGrid) -> dict[str, Any]:
    _write_xy_file(root / "inputs" / "source_a.nc", _target_field())
    return {
        "scales": {},
        "fields": [_entry("source_a", "A", "inputs/source_a.nc")],
    }


def _same_grid_file_scale(root: Path, grid: TransportGrid) -> dict[str, Any]:
    _write_xy_file(root / "inputs" / "source_a.nc", _target_field())
    _write_xy_file(root / "inputs" / "scale.nc", _target_scale(), variable="scale")
    return {
        "scales": {"scale_a": _scale("inputs/scale.nc")},
        "fields": [_entry("source_a", "A", "inputs/source_a.nc", scales=["scale_a"])],
    }


def _constant_and_file_scale(root: Path, grid: TransportGrid) -> dict[str, Any]:
    _write_xy_file(root / "inputs" / "source_a.nc", _target_field())
    _write_xy_file(root / "inputs" / "scale.nc", _target_scale(), variable="scale")
    return {
        "scales": {"scale_a": _scale("inputs/scale.nc"), "negative": {"value": -2.0}},
        "fields": [_entry("source_a", "A", "inputs/source_a.nc", scales=["scale_a", "negative"])],
    }


def _source_regrid_then_scale(root: Path, grid: TransportGrid) -> dict[str, Any]:
    _write_xy_file(root / "inputs" / "source_1x1.nc", _source_field(), lat=SOURCE_LAT_1X1, lon=SOURCE_LON_1X1)
    _write_xy_file(root / "inputs" / "scale.nc", _target_scale(), variable="scale")
    return {
        "scales": {"scale_a": _scale("inputs/scale.nc")},
        "fields": [_entry("source_1x1", "A", "inputs/source_1x1.nc", scales=["scale_a"])],
    }


def _source_and_scale_regrid(root: Path, grid: TransportGrid) -> dict[str, Any]:
    _write_xy_file(root / "inputs" / "source_1x1.nc", _source_field(), lat=SOURCE_LAT_1X1, lon=SOURCE_LON_1X1)
    _write_xy_file(root / "inputs" / "scale_1x1.nc", _source_scale(), variable="scale", lat=SOURCE_LAT_1X1, lon=SOURCE_LON_1X1)
    return {
        "scales": {"scale_a": _scale("inputs/scale_1x1.nc")},
        "fields": [_entry("source_1x1", "A", "inputs/source_1x1.nc", scales=["scale_a"])],
    }


def _npft_select_with_scale(root: Path, grid: TransportGrid) -> dict[str, Any]:
    values = np.stack([_target_field() * factor for factor in (1.0, 2.0, 3.0, 4.0, 5.0)])
    _write_npft_file(root / "inputs" / "source_npft.nc", values, npft=[1, 2, 3, 4, 5])
    _write_xy_file(root / "inputs" / "scale.nc", _target_scale(), variable="scale")
    return {
        "scales": {"scale_a": _scale("inputs/scale.nc")},
        "fields": [
            _entry(
                "source_npft",
                "A",
                "inputs/source_npft.nc",
                src_dim='xy+"npft=4"',
                select={"dimension": "npft", "value": 4},
                scales=["scale_a"],
            )
        ],
    }


def _multiple_entries_same_species(root: Path, grid: TransportGrid) -> dict[str, Any]:
    _write_xy_file(root / "inputs" / "source_a1.nc", _target_field())
    _write_xy_file(root / "inputs" / "source_a2.nc", _target_field() * 2.0)
    _write_xy_file(root / "inputs" / "source_b.nc", _target_field() * 4.0)
    return {
        "scales": {},
        "fields": [
            _entry("source_a1", "A", "inputs/source_a1.nc"),
            _entry("source_b", "B", "inputs/source_b.nc"),
            _entry("source_a2", "A", "inputs/source_a2.nc"),
        ],
    }


def _entry(
    name: str,
    species: str,
    path_template: str,
    *,
    scales: list[str] | None = None,
    src_dim: str = "xy",
    select: dict[str, int | str] | None = None,
) -> dict[str, Any]:
    item = {
        "name": name,
        "species": species,
        "path_template": path_template,
        "variable": "emis",
        "frequency": "constant",
        "dimensions": "xy",
        "src_dim": src_dim,
        "units": "kg/m2/s",
        "scales": scales or [],
    }
    if select:
        item["select"] = select
    return item


def _scale(path_template: str) -> dict[str, Any]:
    return {
        "path_template": path_template,
        "variable": "scale",
        "frequency": "constant",
        "dimensions": "xy",
        "src_dim": "xy",
    }


def _write_hemco_sa_config(root: Path) -> None:
    (root / "HEMCO_sa_Config.rc").write_text(
        """# Synthetic Wombat HEMCO standalone harness
### BEGIN SECTION SETTINGS
ROOT:                .
GridFile:            HEMCO_sa_Grid.rc
SpecFile:            HEMCO_sa_Spec.rc
TimeFile:            HEMCO_sa_Time.rc
DiagnFile:           HEMCO_Diagn.rc
Logfile:             *
MET:                 GEOS
RES:                 2x25
DefaultDiagnOn:      false
DefaultDiagnSname:   TOTAL_
DefaultDiagnLname:   HEMCO_total_emissions_
DefaultDiagnDim:     3
DefaultDiagnUnit:    kg/m2/s
DiagnPrefix:         OutputDir/HEMCO_sa_diagnostics
DiagnFreq:           End
Unit tolerance:      1
Negative values:     2
Separator:           /
Verbose:             false
PBL dry deposition:  False
### END SECTION SETTINGS ###

### BEGIN SECTION EXTENSION SWITCHES
0       Base                   : on    *
### END SECTION EXTENSION SWITCHES

### BEGIN SECTION BASE EMISSIONS
# ExtNr Name sourceFile sourceVar sourceTime C/R/E SrcDim SrcUnit Species ScalIDs Cat Hier
>>>include HEMCO_Config.rc
### END SECTION BASE EMISSIONS

### BEGIN SECTION SCALE FACTORS
# ScalID Name sourceFile sourceVar sourceTime C/R/E SrcDim SrcUnit Oper
### END SECTION SCALE FACTORS

### BEGIN SECTION MASKS
# ScalID Name sourceFile sourceVar sourceTime C/R/E SrcDim SrcUnit Oper Lon1/Lat1/Lon2/Lat2
### END SECTION MASKS
""",
        encoding="utf-8",
    )


def _write_hemco_grid_file(root: Path) -> None:
    ymid = " ".join(f"{value:g}" for value in TARGET_LAT)
    yedge = " ".join(f"{value:g}" for value in _target_lat_edges(TARGET_LAT))
    (root / "HEMCO_sa_Grid.rc").write_text(
        f"""XMIN: -181.25
XMAX:  178.75
YMIN: -90.0
YMAX:  90.0
NX: {TARGET_LON.size}
NY: {TARGET_LAT.size}
NZ: 47
YEDGE: {yedge}
YMID: {ymid}
""",
        encoding="utf-8",
    )


def _write_hemco_time_file(root: Path) -> None:
    (root / "HEMCO_sa_Time.rc").write_text(
        """START:   2014-09-01 00:00:00
END:     2014-09-01 01:00:00
TS_EMIS: 3600
""",
        encoding="utf-8",
    )


def _write_hemco_species_file(root: Path) -> None:
    (root / "HEMCO_sa_Spec.rc").write_text(
        """#ID NAME MW K0 CR PKA
1 A 44.0 0.0 0.0 0.0
2 B 44.0 0.0 0.0 0.0
""",
        encoding="utf-8",
    )


def _write_hemco_diagnostics_file(root: Path) -> None:
    (root / "HEMCO_Diagn.rc").write_text(
        """# Name Spec ExtNr Cat Hier Dim OutUnit LongName
EmisA_Total A -1 -1 -1 3 kg/m2/s A_total_emissions
EmisB_Total B -1 -1 -1 3 kg/m2/s B_total_emissions
""",
        encoding="utf-8",
    )


def _write_hemco_config(root: Path, config: dict[str, Any]) -> None:
    scale_names = list(config["scales"])
    scale_ids = {name: index + 1 for index, name in enumerate(scale_names)}
    lines = [
        "### BEGIN SECTION SETTINGS",
        "ROOT: .",
        "Logfile: *",
        "DiagnFile: HEMCO_Diagn.rc",
        "DiagnPrefix: OutputDir/HEMCO_sa_diagnostics",
        "DiagnFreq: End",
        "Wildcard: *",
        "Separator: /",
        "Unit tolerance: 1",
        "Negative values: 2",
        "Only unitless scale factors: false",
        "Verbose: false",
        "### END SECTION SETTINGS",
        "",
        "### BEGIN SECTION EXTENSION SWITCHES",
        "0 Base : on *",
        "### END SECTION EXTENSION SWITCHES",
        "",
        "### BEGIN SECTION BASE EMISSIONS",
        "# ExtNr Name sourceFile sourceVar sourceTime C/R/E SrcDim SrcUnit Species ScalIDs Cat Hier",
    ]
    for field in config["fields"]:
        scale_text = "/".join(str(scale_ids[name]) for name in field.get("scales", ())) or "-"
        lines.append(
            "0 {name} {path} {var} 2014/9/1/0 RF {src_dim} kg/m2/s {species} {scales} 1 1".format(
                name=field["name"],
                path=field["path_template"],
                var=field["variable"],
                src_dim=field.get("src_dim", "xy"),
                species=field["species"],
                scales=scale_text,
            )
        )
    lines.append("")
    lines.append("### END SECTION BASE EMISSIONS")
    lines.append("")
    lines.append("### BEGIN SECTION SCALE FACTORS")
    lines.append("# ScalID Name sourceFile sourceVar sourceTime C/R/E SrcDim SrcUnit Oper")
    for name in scale_names:
        scale = config["scales"][name]
        if "value" in scale:
            lines.append(f"{scale_ids[name]} {name} {scale['value']} - 2000/1/1/0 C xy 1 1")
        else:
            lines.append(
                f"{scale_ids[name]} {name} {scale['path_template']} {scale['variable']} "
                f"2014/9/1/0 RF {scale.get('src_dim', 'xy')} 1 1"
            )
    lines.append("### END SECTION SCALE FACTORS")
    lines.append("")
    lines.append("### BEGIN SECTION MASKS")
    lines.append("# ScalID Name sourceFile sourceVar sourceTime C/R/E SrcDim SrcUnit Oper Lon1/Lat1/Lon2/Lat2")
    lines.append("### END SECTION MASKS")
    (root / "HEMCO_Config.rc").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_wombat_config(root: Path, config: dict[str, Any]) -> None:
    scales = {}
    for name, scale in config["scales"].items():
        if "value" in scale:
            scales[name] = {"value": scale["value"]}
        else:
            scales[name] = {
                "path_template": scale["path_template"],
                "variable": scale["variable"],
                "frequency": scale.get("frequency", "constant"),
                "dimensions": scale.get("dimensions", "xy"),
            }
    fields = []
    for field in config["fields"]:
        fields.append(
            {
                key: value
                for key, value in field.items()
                if key in {"name", "species", "path_template", "variable", "frequency", "dimensions", "units", "scales", "select"}
            }
        )
    with (root / "wombat_emissions.yml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "unit_conversion": "none",
                "missing_species": "zero",
                "scales": scales,
                "fields": fields,
            },
            handle,
            sort_keys=False,
        )


def _write_scenario_metadata(root: Path, scenario: str, config: dict[str, Any]) -> None:
    with (root / "scenario.yml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"scenario": scenario, "fields": config["fields"], "scales": config["scales"]}, handle, sort_keys=False)


def _write_xy_file(
    path: Path,
    values: np.ndarray,
    *,
    variable: str = "emis",
    lat: np.ndarray = TARGET_LAT,
    lon: np.ndarray = TARGET_LON,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("time", 1)
        dataset.createDimension("lat", lat.size)
        dataset.createDimension("lon", lon.size)
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "hours since 2000-01-01 00:00:00 UTC"
        time[:] = netCDF4.date2num([HARNESS_START], time.units)
        lat_var = dataset.createVariable("lat", "f8", ("lat",))
        lat_var.units = "degrees_north"
        lat_var[:] = lat
        lon_var = dataset.createVariable("lon", "f8", ("lon",))
        lon_var.units = "degrees_east"
        lon_var[:] = lon
        output = dataset.createVariable(variable, "f8", ("time", "lat", "lon"))
        output.units = "kg/m2/s" if variable == "emis" else "1"
        output[:] = np.asarray(values, dtype=np.float64)[np.newaxis, :, :]


def _write_npft_file(path: Path, values: np.ndarray, *, npft: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("time", 1)
        dataset.createDimension("npft", len(npft))
        dataset.createDimension("lat", TARGET_LAT.size)
        dataset.createDimension("lon", TARGET_LON.size)
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "hours since 2000-01-01 00:00:00 UTC"
        time[:] = netCDF4.date2num([HARNESS_START], time.units)
        dataset.createVariable("npft", "i4", ("npft",))[:] = np.asarray(npft, dtype=np.int32)
        lat_var = dataset.createVariable("lat", "f8", ("lat",))
        lat_var.units = "degrees_north"
        lat_var[:] = TARGET_LAT
        lon_var = dataset.createVariable("lon", "f8", ("lon",))
        lon_var.units = "degrees_east"
        lon_var[:] = TARGET_LON
        output = dataset.createVariable("emis", "f8", ("time", "npft", "lat", "lon"))
        output.units = "kg/m2/s"
        output[:] = np.asarray(values, dtype=np.float64)[np.newaxis, :, :, :]


def _transport_grid(root: Path) -> TransportGrid:
    return TransportGrid(
        lat_deg=TARGET_LAT.copy(),
        lon_deg=TARGET_LON.copy(),
        lev=np.arange(47.0, 0.0, -1.0),
        area_m2=_cell_areas(TARGET_LAT, TARGET_LON),
        hyai_hpa=np.linspace(0.0, 1000.0, 48),
        hybi=np.linspace(0.0, 1.0, 48),
        template_path=root / "HEMCO_sa_Grid.rc",
    )


def _target_field() -> np.ndarray:
    lat_term = 1.0 + 0.01 * np.arange(TARGET_LAT.size)[:, np.newaxis]
    lon_term = 1.0 + 0.001 * np.arange(TARGET_LON.size)[np.newaxis, :]
    return 1.0e-9 * lat_term * lon_term


def _target_scale() -> np.ndarray:
    lat_term = 0.5 + 0.002 * np.arange(TARGET_LAT.size)[:, np.newaxis]
    lon_term = 1.0 + 0.0005 * np.arange(TARGET_LON.size)[np.newaxis, :]
    return lat_term * lon_term


def _source_field() -> np.ndarray:
    lat_term = 1.0 + 0.003 * np.arange(SOURCE_LAT_1X1.size)[:, np.newaxis]
    lon_term = 1.0 + 0.0007 * np.arange(SOURCE_LON_1X1.size)[np.newaxis, :]
    return 1.0e-9 * lat_term * lon_term


def _source_scale() -> np.ndarray:
    lat_term = 0.6 + 0.001 * np.arange(SOURCE_LAT_1X1.size)[:, np.newaxis]
    lon_term = 1.0 + 0.0003 * np.arange(SOURCE_LON_1X1.size)[np.newaxis, :]
    return lat_term * lon_term


def _cell_areas(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    low_lat, high_lat = _lat_bounds(lat)
    low_lon, high_lon = _lon_bounds(lon)
    lat_weight = np.sin(np.deg2rad(high_lat)) - np.sin(np.deg2rad(low_lat))
    lon_weight = high_lon - low_lon
    return lat_weight[:, np.newaxis] * lon_weight[np.newaxis, :]


def _lat_bounds(lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bounds = _target_lat_edges(lat)
    return bounds[:-1], bounds[1:]


def _target_lat_edges(lat: np.ndarray) -> np.ndarray:
    midpoints = (lat[:-1] + lat[1:]) / 2.0
    bounds = np.empty(lat.size + 1, dtype=np.float64)
    bounds[1:-1] = midpoints
    bounds[0] = -90.0
    bounds[-1] = 90.0
    if np.isclose(lat[0], -89.5, rtol=0.0, atol=1.0e-12):
        bounds[1] = -89.0
    if np.isclose(lat[-1], 89.5, rtol=0.0, atol=1.0e-12):
        bounds[-2] = 89.0
    return bounds


def _lon_bounds(lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized = np.sort((lon + 360.0) % 360.0)
    step = float(np.median(np.diff(normalized)))
    low = normalized - step / 2.0
    high = normalized + step / 2.0
    low[0] = 0.0
    high[-1] = 360.0
    return low, high


def _find_hemco_diagnostic(root: Path) -> Path:
    diagnostics = sorted((root / "OutputDir").glob("HEMCO_sa_diagnostics*.nc*"))
    if not diagnostics:
        raise FileNotFoundError(f"{root / 'OutputDir'} contains no HEMCO_sa_diagnostics files")
    return diagnostics[0]


def _load_hemco_diagnostic(path: Path):
    with netCDF4.Dataset(path) as dataset:
        lat = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(dataset.variables["lon"][:], dtype=np.float64)
        lev_name = "lev" if "lev" in dataset.variables else "level"
        lev = np.asarray(dataset.variables[lev_name][:], dtype=np.float64) if lev_name in dataset.variables else np.arange(47.0, 0.0, -1.0)
        arrays = []
        units = []
        for item in SPECIES:
            variable_name = f"Emis{item.name}_Total"
            if variable_name not in dataset.variables:
                arrays.append(np.zeros((1, lev.size, lat.size, lon.size), dtype=np.float64))
                units.append("")
                continue
            variable = dataset.variables[variable_name]
            values = _as_time_lev_lat_lon(variable[:], variable.dimensions, lev.size, lat.size, lon.size)
            arrays.append(values)
            units.append(str(getattr(variable, "units", "")))
    return type(
        "HemcoDiagnostic",
        (),
        {
            "names": tuple(item.name for item in SPECIES),
            "data": public_tracer5_to_canonical(np.stack(arrays, axis=0)),
            "units": tuple(units),
            "coords": {"lev": lev[::-1], "lat": lat, "lon": lon, "AREA": _cell_areas(lat, lon)},
        },
    )()


def _as_time_lev_lat_lon(values, dimensions, nlev: int, nlat: int, nlon: int) -> np.ndarray:
    array = np.ma.filled(values, 0.0)
    dims = list(dimensions)
    if "time" not in dims:
        array = array[np.newaxis, ...]
        dims.insert(0, "time")
    if "lev" not in dims and "level" not in dims:
        array = array[:, np.newaxis, ...]
        dims.insert(1, "lev")
    lev_dim = "lev" if "lev" in dims else "level"
    axis_order = [dims.index("time"), dims.index(lev_dim), dims.index("lat"), dims.index("lon")]
    result = np.transpose(array, axis_order)
    if result.shape[1] == 1 and nlev > 1:
        expanded = np.zeros((result.shape[0], nlev, nlat, nlon), dtype=np.float64)
        expanded[:, 0, :, :] = result[:, 0, :, :]
        return expanded
    return np.asarray(result, dtype=np.float64)


if __name__ == "__main__":
    raise SystemExit(main())
