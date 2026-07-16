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

    @property
    def tracer_count(self) -> int:
        return len(self.names)

    @property
    def block_count(self) -> int:
        return 1

    @property
    def block_width(self) -> int:
        return self.tracer_count

    def tracer(self, tracer: int) -> np.ndarray:
        if tracer < 0 or tracer >= self.tracer_count:
            raise IndexError("tracer index out of range")
        return self.data[..., tracer]


@dataclass(frozen=True)
class BlockedTracerField:
    """Tracer state stored as contiguous fixed-width tracer blocks.

    ``data`` is ordered as ``(time, block, lev_top, lat, lon, lane)``.
    ``names`` and ``units`` describe only active tracers; unused lanes in the
    final block are storage padding.
    """

    names: tuple[str, ...]
    data: np.ndarray
    units: tuple[str, ...]
    coords: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        if self.data.ndim != 6:
            raise ValueError(f"blocked tracer data must be 6-D, found shape {self.data.shape}")
        if self.data.shape[1] < 1 or self.data.shape[-1] < 1:
            raise ValueError("blocked tracer data must contain at least one block and lane")
        if len(self.names) < 1 or len(self.names) > self.capacity:
            raise ValueError("tracer names must fit within blocked storage capacity")
        if len(self.units) not in (0, len(self.names)):
            raise ValueError("tracer units must be empty or match tracer names")

    @classmethod
    def from_tracer_field(cls, field: TracerField, block_width: int) -> BlockedTracerField:
        """Pack a canonical field into fixed-width block storage."""

        if field.data.ndim != 5:
            raise ValueError(f"tracer data must be 5-D, found shape {field.data.shape}")
        ntracer = len(field.names)
        if field.data.shape[-1] != ntracer:
            raise ValueError("tracer data width does not match tracer names")
        if block_width < 1:
            raise ValueError("block_width must be positive")
        if block_width == ntracer:
            data = field.data[:, np.newaxis, ...]
        else:
            ntime, nlev, nlat, nlon, _ = field.data.shape
            nblock = (ntracer + block_width - 1) // block_width
            data = np.zeros((ntime, nblock, nlev, nlat, nlon, block_width), dtype=field.data.dtype)
            for block in range(nblock):
                start = block * block_width
                stop = min(start + block_width, ntracer)
                data[:, block, :, :, :, : stop - start] = field.data[..., start:stop]
        return cls(names=field.names, data=data, units=field.units, coords=field.coords)

    @property
    def block_count(self) -> int:
        return self.data.shape[1]

    @property
    def block_width(self) -> int:
        return self.data.shape[-1]

    @property
    def tracer_count(self) -> int:
        return len(self.names)

    @property
    def capacity(self) -> int:
        return self.block_count * self.block_width

    @property
    def shape(self) -> tuple[int, ...]:
        """Logical canonical shape, excluding block padding."""

        return (*self.data.shape[:1], *self.data.shape[2:5], self.tracer_count)

    def block_bounds(self, block: int) -> tuple[int, int]:
        if block < 0 or block >= self.block_count:
            raise IndexError("tracer block index out of range")
        start = block * self.block_width
        return start, min(start + self.block_width, self.tracer_count)

    def block(self, block: int) -> TracerField:
        """Return active lanes in one block as a zero-copy canonical view."""

        start, stop = self.block_bounds(block)
        active = stop - start
        units = self.units[start:stop] if self.units else ()
        return TracerField(
            names=self.names[start:stop],
            data=self.data[:, block, :, :, :, :active],
            units=units,
            coords=self.coords,
        )

    def iter_blocks(self):
        for block in range(self.block_count):
            yield self.block(block)

    def tracer(self, tracer: int) -> np.ndarray:
        """Return one logical tracer as ``(time, lev, lat, lon)`` view."""

        if tracer < 0 or tracer >= self.tracer_count:
            raise IndexError("tracer index out of range")
        block, lane = divmod(tracer, self.block_width)
        return self.data[:, block, :, :, :, lane]

    def to_tracer_field(self) -> TracerField:
        """Return canonical storage, copying only when blocks must be joined."""

        if self.block_count == 1:
            canonical = self.data[:, 0, :, :, :, : self.tracer_count]
        else:
            canonical = np.empty(self.shape, dtype=self.data.dtype)
            for block in range(self.block_count):
                start, stop = self.block_bounds(block)
                canonical[..., start:stop] = self.data[:, block, :, :, :, : stop - start]
        return TracerField(names=self.names, data=canonical, units=self.units, coords=self.coords)


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
