from __future__ import annotations

import argparse
from pathlib import Path

import netCDF4
import numpy as np

from wombat_transport.emissions import ConservativeRemappingWeights
from wombat_transport.grid import (
    MODEL_LEVELS,
    geos_chem_grid_cell_area_m2,
    geos_chem_horizontal_centers,
    geos_chem_horizontal_resolution,
)


COPY_VARIABLES = ("time", "lev", "ilev", "hyam", "hybm", "hyai", "hybi", "P0")


def regrid_restart(source_path: Path, output_path: Path, *, target_grid: str, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing restart: {output_path}")
    if target_grid != "4x5":
        raise ValueError(f"unsupported target grid {target_grid!r}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_lat, target_lon = geos_chem_horizontal_centers(target_grid)
    with netCDF4.Dataset(source_path) as source:
        source_lat = np.asarray(source.variables["lat"][:], dtype=np.float64)
        source_lon = np.asarray(source.variables["lon"][:], dtype=np.float64)
        geos_chem_horizontal_resolution(source_lat, source_lon)
        if len(source.dimensions["lev"]) != MODEL_LEVELS:
            raise ValueError(f"expected {MODEL_LEVELS} source levels")
        species_names = sorted(name for name in source.variables if name.startswith("SpeciesRst_"))
        if not species_names:
            raise ValueError(f"{source_path} contains no SpeciesRst_* variables")
        remapping = ConservativeRemappingWeights(
            source_lat=source_lat,
            source_lon=source_lon,
            target_lat=target_lat,
            target_lon=target_lon,
        )

        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        try:
            with netCDF4.Dataset(temporary, "w") as output:
                _copy_global_attributes(source, output)
                _create_dimensions(source, output, target_lat.size, target_lon.size)
                for name in COPY_VARIABLES:
                    if name in source.variables:
                        _copy_variable(source.variables[name], output)
                _write_horizontal_coordinates(source, output, target_lat, target_lon)
                _write_area(source, output, target_lat, target_lon)
                for name in species_names:
                    _regrid_species(
                        source.variables[name], output, remapping
                    )
            temporary.replace(output_path)
        except BaseException:
            if temporary.exists():
                temporary.unlink()
            raise


def _copy_global_attributes(source: netCDF4.Dataset, output: netCDF4.Dataset) -> None:
    output.setncatts({name: source.getncattr(name) for name in source.ncattrs()})
    output.history = f"Regridded to GEOS 4x5 for Wombat/GEOS-Chem parity testing; {getattr(source, 'history', '')}"


def _create_dimensions(
    source: netCDF4.Dataset, output: netCDF4.Dataset, target_nlat: int, target_nlon: int
) -> None:
    for name in ("time", "lev", "ilev"):
        dimension = source.dimensions[name]
        output.createDimension(name, None if dimension.isunlimited() else len(dimension))
    output.createDimension("lat", target_nlat)
    output.createDimension("lon", target_nlon)


def _variable_options(source: netCDF4.Variable) -> tuple[object | None, dict[str, object]]:
    fill_value = source.getncattr("_FillValue") if "_FillValue" in source.ncattrs() else None
    attrs = {name: source.getncattr(name) for name in source.ncattrs() if name != "_FillValue"}
    return fill_value, attrs


def _create_variable_like(
    source: netCDF4.Variable, output: netCDF4.Dataset, *, dimensions: tuple[str, ...] | None = None
) -> netCDF4.Variable:
    fill_value, attrs = _variable_options(source)
    kwargs = {} if fill_value is None else {"fill_value": fill_value}
    variable = output.createVariable(source.name, source.datatype, dimensions or source.dimensions, **kwargs)
    variable.setncatts(attrs)
    return variable


def _copy_variable(source: netCDF4.Variable, output: netCDF4.Dataset) -> None:
    variable = _create_variable_like(source, output)
    if source.dimensions:
        variable[:] = source[:]
    else:
        variable.assignValue(source.getValue())


def _write_horizontal_coordinates(
    source: netCDF4.Dataset, output: netCDF4.Dataset, lat: np.ndarray, lon: np.ndarray
) -> None:
    lat_variable = _create_variable_like(source.variables["lat"], output)
    lon_variable = _create_variable_like(source.variables["lon"], output)
    lat_variable[:] = lat
    lon_variable[:] = lon


def _write_area(
    source: netCDF4.Dataset, output: netCDF4.Dataset, lat: np.ndarray, lon: np.ndarray
) -> None:
    area = geos_chem_grid_cell_area_m2(lat, lon)
    if "AREA" in source.variables:
        _, attrs = _variable_options(source.variables["AREA"])
        variable = output.createVariable("AREA", "f8", ("lat", "lon"))
        variable.setncatts(attrs)
    else:
        variable = output.createVariable("AREA", "f8", ("lat", "lon"))
        variable.long_name = "Surface area"
        variable.units = "m2"
    variable[:] = area


def _regrid_species(
    source: netCDF4.Variable,
    output: netCDF4.Dataset,
    remapping: ConservativeRemappingWeights,
) -> None:
    if source.dimensions != ("time", "lev", "lat", "lon"):
        raise ValueError(f"{source.name} has unsupported dimensions {source.dimensions}")
    values = np.ma.filled(source[:], np.nan)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{source.name} contains missing or non-finite values")
    variable = _create_variable_like(source, output)
    variable[:] = remapping.apply(values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Conservatively regrid a GEOS-Chem restart horizontally.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-grid", choices=("4x5",), default="4x5")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    regrid_restart(args.source, args.output, target_grid=args.target_grid, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
