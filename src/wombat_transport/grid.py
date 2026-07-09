from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import netCDF4
import numpy as np


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
        return TransportGrid(
            lat_deg=np.asarray(template.variables["lat"][:], dtype=np.float64),
            lon_deg=np.asarray(template.variables["lon"][:], dtype=np.float64),
            lev=np.asarray(template.variables["lev"][:], dtype=np.float64),
            area_m2=np.asarray(template.variables["AREA"][:], dtype=np.float64),
            hyai_hpa=np.asarray(template.variables["hyai"][:], dtype=np.float64),
            hybi=np.asarray(template.variables["hybi"][:], dtype=np.float64),
            template_path=path,
        )
