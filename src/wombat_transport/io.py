from __future__ import annotations

from pathlib import Path
from typing import Iterable

import netCDF4
import numpy as np

from wombat_transport.fields import TracerField
from wombat_transport.species import Species, load_species_database

FIXED_GRID = {"lev": 47, "lat": 91, "lon": 144}
GRID_COORDS = ("time", "lev", "ilev", "lat", "lon", "hyam", "hybm", "hyai", "hybi", "P0", "AREA")


def load_restart(path: str | Path, species: Iterable[Species] | None = None) -> TracerField:
    return _load_tracer_variables(Path(path), "SpeciesRst_", species)


def load_species_conc(path: str | Path) -> TracerField:
    return _load_tracer_variables(Path(path), "SpeciesConcVV_")


def load_hemco_emissions(path: str | Path) -> TracerField:
    return _load_tracer_variables(Path(path), "Emis_")


def load_base_met(path: str | Path) -> dict[str, np.ndarray]:
    wanted = {"Met_PEDGE", "Met_PEDGEDRY", "Met_BXHEIGHT", "Met_AVGW"}
    with netCDF4.Dataset(path) as dataset:
        _assert_fixed_grid(dataset)
        return {name: np.asarray(dataset.variables[name][:]) for name in wanted if name in dataset.variables}


def write_restart_like(path: str | Path, tracer_field: TracerField, template_path: str | Path) -> None:
    """Write a restart-like NetCDF file with one ``SpeciesRst_*`` variable per tracer."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with netCDF4.Dataset(template_path) as template, netCDF4.Dataset(output_path, "w") as output:
        _assert_fixed_grid(template)
        _assert_tracer_shape(tracer_field)

        for dim_name in ("time", "lev", "ilev", "lat", "lon"):
            size = tracer_field.data.shape[1] if dim_name == "time" else len(template.dimensions[dim_name])
            output.createDimension(dim_name, size)

        for coord_name in GRID_COORDS:
            if coord_name not in template.variables:
                continue
            source = template.variables[coord_name]
            variable = output.createVariable(coord_name, source.datatype, source.dimensions)
            variable.setncatts({name: source.getncattr(name) for name in source.ncattrs()})
            if source.dimensions:
                variable[:] = np.asarray(source[:])
            else:
                variable.assignValue(source.getValue())

        output.title = "Wombat Transport restart-like output"
        output.Conventions = "COARDS"

        for index, name in enumerate(tracer_field.names):
            variable = output.createVariable(f"SpeciesRst_{name}", "f8", ("time", "lev", "lat", "lon"))
            variable.units = tracer_field.units[index] if index < len(tracer_field.units) else ""
            variable.long_name = f"Wombat restart-like concentration of species {name}"
            variable[:] = tracer_field.data[index]


def initialize_tracers(
    restart_path: str | Path | None,
    species_db_path: str | Path,
    *,
    template_path: str | Path | None = None,
) -> TracerField:
    """Initialize all species from restart variables or species Background_VV.

    If ``restart_path`` is absent, ``template_path`` supplies grid coordinates.
    If a restart file exists but a specific species variable is missing, that
    species is filled from its ``Background_VV``.
    """

    species = load_species_database(species_db_path)
    restart = Path(restart_path) if restart_path is not None else None
    template = Path(template_path) if template_path is not None else restart
    if template is None:
        raise ValueError("initialize_tracers requires restart_path or template_path")

    with netCDF4.Dataset(template) as template_dataset:
        _assert_fixed_grid(template_dataset)
        coords = _read_coords(template_dataset)
        shape = (
            len(species),
            len(template_dataset.dimensions["time"]),
            FIXED_GRID["lev"],
            FIXED_GRID["lat"],
            FIXED_GRID["lon"],
        )

    data = np.empty(shape, dtype=np.float64)
    units: list[str] = []

    restart_dataset = None
    if restart is not None and restart.exists():
        restart_dataset = netCDF4.Dataset(restart)

    try:
        for index, item in enumerate(species):
            var_name = f"SpeciesRst_{item.name}"
            if restart_dataset is not None and var_name in restart_dataset.variables:
                variable = restart_dataset.variables[var_name]
                data[index] = np.asarray(variable[:], dtype=np.float64)
                units.append(str(getattr(variable, "units", "")))
            else:
                data[index].fill(item.background_vv)
                units.append("mol mol-1 dry")
    finally:
        if restart_dataset is not None:
            restart_dataset.close()

    return TracerField(
        names=tuple(item.name for item in species),
        data=data,
        units=tuple(units),
        coords=coords,
    )


def _load_tracer_variables(
    path: Path,
    prefix: str,
    species: Iterable[Species] | None = None,
) -> TracerField:
    with netCDF4.Dataset(path) as dataset:
        _assert_fixed_grid(dataset)
        coords = _read_coords(dataset)
        if species is None:
            names = _sorted_suffixes(dataset.variables, prefix)
        else:
            names = tuple(item.name for item in species)
        if not names:
            raise KeyError(f"{path} contains no variables with prefix {prefix}")

        arrays: list[np.ndarray] = []
        units: list[str] = []
        for name in names:
            variable_name = f"{prefix}{name}"
            if variable_name not in dataset.variables:
                raise KeyError(f"{path} is missing variable {variable_name}")
            variable = dataset.variables[variable_name]
            arrays.append(np.asarray(variable[:], dtype=np.float64))
            units.append(str(getattr(variable, "units", "")))

    data = np.stack(arrays, axis=0)
    return TracerField(names=names, data=data, units=tuple(units), coords=coords)


def _sorted_suffixes(variables: dict[str, object], prefix: str) -> tuple[str, ...]:
    return tuple(sorted(name.removeprefix(prefix) for name in variables if name.startswith(prefix)))


def _assert_fixed_grid(dataset: netCDF4.Dataset) -> None:
    for dim, expected in FIXED_GRID.items():
        actual = len(dataset.dimensions[dim])
        if actual != expected:
            raise ValueError(f"expected {dim}={expected}, found {actual}")


def _assert_tracer_shape(tracer_field: TracerField) -> None:
    if tracer_field.data.ndim != 5:
        raise ValueError(f"expected tracer data to be 5-D, found {tracer_field.data.ndim}-D")
    if tracer_field.data.shape[0] != len(tracer_field.names):
        raise ValueError("tracer name count does not match data first dimension")
    expected = (FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"])
    if tracer_field.data.shape[2:] != expected:
        raise ValueError(f"expected tracer grid {expected}, found {tracer_field.data.shape[2:]}")


def _read_coords(dataset: netCDF4.Dataset) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(dataset.variables[name][:])
        for name in GRID_COORDS
        if name in dataset.variables
    }
