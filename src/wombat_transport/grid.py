from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import netCDF4
import numpy as np

from wombat_transport.constants import EARTH_RADIUS_M


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


def load_transport_grid(template_path: str | Path) -> TransportGrid:
    """Load static transport-grid metadata from a GEOS-Chem restart/template file."""

    path = Path(template_path)
    with netCDF4.Dataset(path) as template:
        lat = np.asarray(template.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(template.variables["lon"][:], dtype=np.float64)
        return TransportGrid(
            lat_deg=lat,
            lon_deg=lon,
            lev=np.asarray(template.variables["lev"][:], dtype=np.float64),
            area_m2=geos_chem_grid_cell_area_m2(lat, lon),
            hyai_hpa=np.asarray(template.variables["hyai"][:], dtype=np.float64),
            hybi=np.asarray(template.variables["hybi"][:], dtype=np.float64),
            template_path=path,
        )


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
    edges = _regular_latitude_edges_deg(lat, lat_step)
    area_1d = (dx_deg * np.pi / 180.0) * (EARTH_RADIUS_M**2) * (
        np.sin(np.deg2rad(edges[1:])) - np.sin(np.deg2rad(edges[:-1]))
    )
    return np.broadcast_to(area_1d[:, np.newaxis], (lat.size, lon.size)).copy()


def _regular_latitude_step_deg(lat_deg: np.ndarray) -> float:
    diffs = np.diff(np.asarray(lat_deg, dtype=np.float64))
    if np.any(diffs <= 0.0):
        raise ValueError("latitudes must be strictly increasing")
    return float(np.max(np.round(diffs, 10)))


def _regular_latitude_edges_deg(lat_deg: np.ndarray, lat_step_deg: float) -> np.ndarray:
    lat = np.asarray(lat_deg, dtype=np.float64)
    edges = np.empty(lat.size + 1, dtype=np.float64)
    edges[0] = -90.0
    edges[-1] = 90.0
    if np.isclose(lat[0], -90.0 + 0.25 * lat_step_deg) and np.isclose(lat[-1], 90.0 - 0.25 * lat_step_deg):
        edges[1:-1] = -90.0 - 0.5 * lat_step_deg + lat_step_deg * np.arange(1, lat.size)
    else:
        edges[1:-1] = 0.5 * (lat[:-1] + lat[1:])
    return edges
