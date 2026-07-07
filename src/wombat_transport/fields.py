from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TracerField:
    """Stacked tracer data in canonical transport layout.

    ``data`` is ordered as ``(time, lev_top, lat, lon, tracer)``. Boundary
    readers and writers convert to and from GEOS-Chem/NetCDF tracer-first,
    bottom-level order.
    """

    names: tuple[str, ...]
    data: np.ndarray
    units: tuple[str, ...]
    coords: dict[str, np.ndarray]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape


def public_tracer5_to_canonical(values: np.ndarray) -> np.ndarray:
    """Convert ``(tracer, time, lev_bottom, lat, lon)`` to canonical layout."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 5:
        raise ValueError(f"expected 5-D public tracer array, found shape {array.shape}")
    return np.ascontiguousarray(np.transpose(array[:, :, ::-1, :, :], (1, 2, 3, 4, 0)))


def canonical_to_public_tracer5(values: np.ndarray) -> np.ndarray:
    """Convert canonical layout to ``(tracer, time, lev_bottom, lat, lon)``."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 5:
        raise ValueError(f"expected 5-D canonical tracer array, found shape {array.shape}")
    return np.ascontiguousarray(np.transpose(array[:, ::-1, :, :, :], (4, 0, 1, 2, 3)))


def public_tracer4_to_transport(values: np.ndarray) -> np.ndarray:
    """Convert ``(tracer, lev_bottom, lat, lon)`` to ``(lev_top, lat, lon, tracer)``."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 4:
        raise ValueError(f"expected 4-D public tracer array, found shape {array.shape}")
    return np.ascontiguousarray(np.transpose(array[:, ::-1, :, :], (1, 2, 3, 0)))


def transport_tracer_to_public4(values: np.ndarray) -> np.ndarray:
    """Convert ``(lev_top, lat, lon, tracer)`` to ``(tracer, lev_bottom, lat, lon)``."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 4:
        raise ValueError(f"expected 4-D transport tracer array, found shape {array.shape}")
    return np.ascontiguousarray(np.transpose(array[::-1], (3, 0, 1, 2)))


def canonical_time_slice(values: np.ndarray, time_index: int = 0) -> np.ndarray:
    """Return one canonical time slice as ``(lev_top, lat, lon, tracer)``."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 5:
        raise ValueError(f"expected 5-D canonical tracer array, found shape {array.shape}")
    return np.ascontiguousarray(array[time_index])


def transport_tracer_to_canonical(values: np.ndarray) -> np.ndarray:
    """Add a length-one time axis to ``(lev_top, lat, lon, tracer)``."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 4:
        raise ValueError(f"expected 4-D transport tracer array, found shape {array.shape}")
    return np.ascontiguousarray(array[np.newaxis, ...])


def bottom_field3_to_top(values: np.ndarray) -> np.ndarray:
    """Convert ``(lev_bottom, lat, lon)`` to ``(lev_top, lat, lon)``."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"expected 3-D bottom-order field, found shape {array.shape}")
    return np.ascontiguousarray(array[::-1])


def top_field3_to_bottom(values: np.ndarray) -> np.ndarray:
    """Convert ``(lev_top, lat, lon)`` to ``(lev_bottom, lat, lon)``."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"expected 3-D top-order field, found shape {array.shape}")
    return np.ascontiguousarray(array[::-1])


def public_surface_flux_to_transport(values: np.ndarray) -> np.ndarray:
    """Convert ``(tracer, lat, lon)`` to ``(lat, lon, tracer)``."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"expected 3-D public surface flux, found shape {array.shape}")
    return np.ascontiguousarray(np.moveaxis(array, 0, -1))
