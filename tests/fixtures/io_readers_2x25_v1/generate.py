from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np


ROOT = Path(__file__).parent
TRACERS = tuple(f"r0002p001s{index:03d}" for index in range(1, 25))


def _create_grid(path: Path) -> netCDF4.Dataset:
    dataset = netCDF4.Dataset(path, "w", format="NETCDF4")
    dataset.createDimension("time", 1)
    dataset.createDimension("lev", 47)
    dataset.createDimension("ilev", 48)
    dataset.createDimension("lat", 91)
    dataset.createDimension("lon", 144)
    dataset.createDimension("nb", 2)
    dataset.createVariable("time", "f8", ("time",))[:] = [0.0]
    dataset.createVariable("lev", "f8", ("lev",))[:] = np.arange(47)
    dataset.createVariable("ilev", "f8", ("ilev",))[:] = np.arange(48)
    lat = np.concatenate(([-89.5], np.arange(-88.0, 90.0, 2.0), [89.5]))
    dataset.createVariable("lat", "f8", ("lat",))[:] = lat
    dataset.createVariable("lon", "f8", ("lon",))[:] = np.arange(-180.0, 180.0, 2.5)
    dataset.createVariable("hyam", "f8", ("lev",))[:] = np.linspace(0.0, 1.0, 47)
    dataset.createVariable("hybm", "f8", ("lev",))[:] = np.linspace(1.0, 0.0, 47)
    dataset.createVariable("hyai", "f8", ("ilev",))[:] = np.linspace(0.0, 1.0, 48)
    dataset.createVariable("hybi", "f8", ("ilev",))[:] = np.linspace(1.0, 0.0, 48)
    dataset.createVariable("P0", "f8").assignValue(100000.0)
    dataset.createVariable("AREA", "f4", ("lat", "lon"), zlib=True)[:] = 1.0
    return dataset


def _tracer_variable(dataset: netCDF4.Dataset, name: str, value: float) -> None:
    variable = dataset.createVariable(
        name,
        "f4",
        ("time", "lev", "lat", "lon"),
        zlib=True,
        complevel=1,
        shuffle=True,
    )
    variable.units = "mol mol-1 dry"
    variable[:] = value


def write_restart() -> None:
    with _create_grid(ROOT / "restart.nc4") as dataset:
        _tracer_variable(dataset, "SpeciesRst_CO2", 3.7e-4)
        for index, name in enumerate(TRACERS, start=1):
            _tracer_variable(dataset, f"SpeciesRst_{name}", 4.0e-4 + index * 1.0e-8)


def write_species_conc() -> None:
    with _create_grid(ROOT / "base_species_conc.nc4") as dataset:
        _tracer_variable(dataset, "SpeciesConcVV_CO2", 3.8e-4)
    with _create_grid(ROOT / "residual_species_conc.nc4") as dataset:
        for index, name in enumerate(TRACERS, start=1):
            _tracer_variable(dataset, f"SpeciesConcVV_{name}", 4.0e-4 + index * 1.0e-8)


def write_hemco() -> None:
    with _create_grid(ROOT / "hemco.nc4") as dataset:
        for index, name in enumerate(TRACERS, start=1):
            variable = dataset.createVariable(
                f"Emis_{name}",
                "f4",
                ("time", "lev", "lat", "lon"),
                zlib=True,
                complevel=1,
            )
            variable.units = "kg/m2/s"
            variable[:] = index * 1.0e-12
    with _create_grid(ROOT / "hemco_invalid.nc4") as dataset:
        variable = dataset.createVariable(
            "Emis_invalid",
            "f8",
            ("time", "lev", "lat", "lon"),
            zlib=True,
            complevel=1,
        )
        variable.units = "kg/m2/s"
        variable[:] = 1.0e30


def write_met() -> None:
    with _create_grid(ROOT / "met_diagnostics.nc4") as dataset:
        for name in ("Met_PEDGE", "Met_PEDGEDRY"):
            dataset.createVariable(name, "f4", ("time", "ilev", "lat", "lon"), zlib=True)[:] = 1.0
        for name in ("Met_BXHEIGHT", "Met_AVGW"):
            dataset.createVariable(name, "f4", ("time", "lev", "lat", "lon"), zlib=True)[:] = 1.0


if __name__ == "__main__":
    write_restart()
    write_species_conc()
    write_hemco()
    write_met()
