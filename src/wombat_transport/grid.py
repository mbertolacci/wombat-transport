from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import netCDF4
import numpy as np

from wombat_transport.constants import EARTH_RADIUS_M


MODEL_LEVELS = 47
GEOS_HORIZONTAL_GRIDS = {
    (91, 144): "2x25",
    (46, 72): "4x5",
}


@dataclass(frozen=True)
class TransportGrid:
    """Static target transport-grid metadata loaded from a template NetCDF file."""

    lat_deg: np.ndarray
    lon_deg: np.ndarray
    lev: np.ndarray
    area_m2: np.ndarray
    hyai_hpa: np.ndarray
    hybi: np.ndarray
    template_path: Path

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.hyai_hpa.size - 1, self.lat_deg.size, self.lon_deg.size)

    @property
    def horizontal_resolution(self) -> str:
        return geos_chem_horizontal_resolution(self.lat_deg, self.lon_deg)


def load_transport_grid(template_path: str | Path) -> TransportGrid:
    """Load static transport-grid metadata from a GEOS-Chem restart/template file."""

    path = Path(template_path)
    with netCDF4.Dataset(path) as template:
        lat = np.asarray(template.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(template.variables["lon"][:], dtype=np.float64)
        grid = TransportGrid(
            lat_deg=lat,
            lon_deg=lon,
            lev=np.asarray(template.variables["lev"][:], dtype=np.float64),
            area_m2=geos_chem_grid_cell_area_m2(lat, lon),
            hyai_hpa=np.asarray(template.variables["hyai"][:], dtype=np.float64),
            hybi=np.asarray(template.variables["hybi"][:], dtype=np.float64),
            template_path=path,
        )
    if grid.shape[0] != MODEL_LEVELS:
        raise ValueError(f"expected {MODEL_LEVELS} model levels, found {grid.shape[0]}")
    geos_chem_horizontal_resolution(grid.lat_deg, grid.lon_deg)
    return grid


def geos_chem_horizontal_resolution(lat_deg: np.ndarray, lon_deg: np.ndarray) -> str:
    """Return the GEOS filename tag for a supported global transport grid."""

    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = np.asarray(lon_deg, dtype=np.float64)
    resolution = GEOS_HORIZONTAL_GRIDS.get((lat.size, lon.size))
    if resolution is None:
        raise ValueError(f"unsupported GEOS horizontal grid {lat.size}x{lon.size}")
    return resolution


def geos_chem_horizontal_centers(resolution: str) -> tuple[np.ndarray, np.ndarray]:
    """Construct GEOS-Chem global grid centers, including half polar boxes."""

    if resolution == "2x25":
        lat = np.concatenate(([-89.5], np.arange(-88.0, 90.0, 2.0), [89.5]))
        lon = np.arange(-180.0, 180.0, 2.5)
    elif resolution == "4x5":
        lat = np.concatenate(([-89.0], np.arange(-86.0, 90.0, 4.0), [89.0]))
        lon = np.arange(-180.0, 180.0, 5.0)
    else:
        raise ValueError(f"unsupported GEOS horizontal resolution {resolution!r}")
    return lat, lon


def geos_chem_grid_cell_area_m2(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Compute GEOS-Chem Classic regular-grid surface area in m2."""

    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = np.asarray(lon_deg, dtype=np.float64)
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("lat_deg and lon_deg must be one-dimensional")
    if lat.size < 2 or lon.size < 2:
        raise ValueError("at least two latitudes and longitudes are required")

    dx_deg = 360.0 / lon.size
    lat_step = _regular_latitude_step_deg(lat)
    edges = geos_chem_latitude_edges_deg(lat, lat_step)
    area_1d = (dx_deg * np.pi / 180.0) * (EARTH_RADIUS_M**2) * (
        np.sin(np.deg2rad(edges[1:])) - np.sin(np.deg2rad(edges[:-1]))
    )
    return np.broadcast_to(area_1d[:, np.newaxis], (lat.size, lon.size)).copy()


def _regular_latitude_step_deg(lat_deg: np.ndarray) -> float:
    diffs = np.diff(np.asarray(lat_deg, dtype=np.float64))
    if np.any(diffs <= 0.0):
        raise ValueError("latitudes must be strictly increasing")
    return float(np.max(np.round(diffs, 10)))


def geos_chem_latitude_edges_deg(lat_deg: np.ndarray, lat_step_deg: float | None = None) -> np.ndarray:
    """Infer bounded latitude edges for regular GEOS grids."""

    lat = np.asarray(lat_deg, dtype=np.float64)
    lat_step_deg = _regular_latitude_step_deg(lat) if lat_step_deg is None else float(lat_step_deg)
    edges = np.empty(lat.size + 1, dtype=np.float64)
    edges[0] = -90.0
    edges[-1] = 90.0
    if np.isclose(lat[0], -90.0 + 0.25 * lat_step_deg) and np.isclose(lat[-1], 90.0 - 0.25 * lat_step_deg):
        edges[1:-1] = -90.0 - 0.5 * lat_step_deg + lat_step_deg * np.arange(1, lat.size)
    else:
        edges[1:-1] = 0.5 * (lat[:-1] + lat[1:])
    return edges
