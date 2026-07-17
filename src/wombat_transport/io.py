from __future__ import annotations

from pathlib import Path
from typing import Iterable

import netCDF4
import numpy as np

from wombat_transport.fields import TracerField, public_tracer5_to_canonical
from wombat_transport.grid import MODEL_LEVELS, geos_chem_horizontal_resolution
from wombat_transport.species import Species, load_species_database

# Kept as the canonical 2x2.5 fixture dimensions for compatibility. Runtime
# validation derives horizontal dimensions from the supplied template.
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
        _assert_supported_grid(dataset)
        return {name: np.asarray(dataset.variables[name][:]) for name in wanted if name in dataset.variables}


def write_restart_like(
    path: str | Path, tracer_field: TracerField, template_path: str | Path
) -> None:
    """Write a restart-like NetCDF file with one ``SpeciesRst_*`` variable per tracer."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with netCDF4.Dataset(template_path) as template, netCDF4.Dataset(output_path, "w") as output:
        template_shape = _assert_supported_grid(template)
        _assert_tracer_shape(tracer_field, template_shape)

        for dim_name in ("time", "lev", "ilev", "lat", "lon"):
            size = tracer_field.shape[0] if dim_name == "time" else len(template.dimensions[dim_name])
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
            variable[:] = tracer_field.tracer(index)[:, ::-1, :, :]


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
        template_shape = _assert_supported_grid(template_dataset)
        coords = _read_coords(template_dataset)
        shape = (
            len(template_dataset.dimensions["time"]),
            template_shape[0],
            template_shape[1],
            template_shape[2],
            len(species),
        )

    data = np.empty(shape, dtype=np.float64)
    units: list[str] = []

    restart_dataset = None
    if restart is not None and restart.exists():
        restart_dataset = netCDF4.Dataset(restart)
        restart_shape = _assert_supported_grid(restart_dataset)
        if restart_shape != template_shape:
            restart_dataset.close()
            raise ValueError(f"restart grid {restart_shape} does not match template grid {template_shape}")

    try:
        for index, item in enumerate(species):
            var_name = f"SpeciesRst_{item.name}"
            if restart_dataset is not None and var_name in restart_dataset.variables:
                variable = restart_dataset.variables[var_name]
                data[..., index] = _geos_chem_initial_vv(np.asarray(variable[:, ::-1, :, :], dtype=np.float64))
                units.append(str(getattr(variable, "units", "")))
            else:
                data[..., index].fill(_geos_chem_initial_vv(item.background_vv))
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


def _geos_chem_initial_vv(values) -> np.ndarray | np.float32:
    """Mirror GEOS-Chem restart/background initialization precision."""

    return np.asarray(values, dtype=np.float32).astype(np.float64)


def _load_tracer_variables(
    path: Path,
    prefix: str,
    species: Iterable[Species] | None = None,
) -> TracerField:
    with netCDF4.Dataset(path) as dataset:
        _assert_supported_grid(dataset)
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

    data = public_tracer5_to_canonical(np.stack(arrays, axis=0))
    return TracerField(names=names, data=data, units=tuple(units), coords=coords)


def _sorted_suffixes(variables: dict[str, object], prefix: str) -> tuple[str, ...]:
    return tuple(sorted(name.removeprefix(prefix) for name in variables if name.startswith(prefix)))


def _assert_supported_grid(dataset: netCDF4.Dataset) -> tuple[int, int, int]:
    lev = len(dataset.dimensions["lev"])
    if lev != MODEL_LEVELS:
        raise ValueError(f"expected lev={MODEL_LEVELS}, found {lev}")
    lat = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
    lon = np.asarray(dataset.variables["lon"][:], dtype=np.float64)
    geos_chem_horizontal_resolution(lat, lon)
    return lev, lat.size, lon.size


def _assert_tracer_shape(
    tracer_field: TracerField, expected: tuple[int, int, int]
) -> None:
    if tracer_field.shape[-1] != len(tracer_field.names):
        raise ValueError("tracer name count does not match logical tracer dimension")
    if tracer_field.shape[1:4] != expected:
        raise ValueError(f"expected tracer grid {expected}, found {tracer_field.shape[1:4]}")


def _read_coords(dataset: netCDF4.Dataset) -> dict[str, np.ndarray]:
    coords = {
        name: np.asarray(dataset.variables[name][:])
        for name in GRID_COORDS
        if name in dataset.variables
    }
    for name in ("lev", "hyam", "hybm"):
        if name in coords:
            coords[name] = coords[name][::-1]
    for name in ("ilev", "hyai", "hybi"):
        if name in coords:
            coords[name] = coords[name][::-1]
    return coords
