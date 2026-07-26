from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Protocol

import numpy as np


class ArrayStorage(Protocol):
    """Array behavior needed by :class:`TracerField` storage."""

    ndim: int
    shape: tuple[int, ...]
    dtype: np.dtype[Any]

    def __getitem__(self, key: Any) -> Any: ...

    def __setitem__(self, key: Any, value: Any) -> None: ...


@dataclass(frozen=True, init=False)
class TracerField:
    """Logical tracer field backed by native fixed-width block storage.

    Physical storage is always ordered as
    ``(time, block, lev_top, lat, lon, lane)``. Names and units describe
    only active tracers; unused tail lanes are padding. Canonical storage is
    available as a zero-copy view only for a one-block field.
    """

    names: tuple[str, ...]
    _data: ArrayStorage
    units: tuple[str, ...]
    coords: dict[str, np.ndarray]

    def __init__(
        self,
        *,
        names: tuple[str, ...],
        data: ArrayStorage,
        units: tuple[str, ...],
        coords: dict[str, np.ndarray],
    ) -> None:
        storage = _coerce_storage(data)
        if storage.ndim == 5:
            if storage.shape[-1] != len(names):
                raise ValueError("canonical tracer width does not match tracer names")
            storage = storage[:, np.newaxis, ...]
        elif storage.ndim != 6:
            raise ValueError(f"tracer data must be 5-D or 6-D, found shape {storage.shape}")
        object.__setattr__(self, "names", tuple(names))
        object.__setattr__(self, "_data", storage)
        object.__setattr__(self, "units", tuple(units))
        object.__setattr__(self, "coords", coords)
        self._validate()

    def _validate(self) -> None:
        if self._data.shape[1] < 1 or self._data.shape[-1] < 1:
            raise ValueError("tracer data must contain at least one block and lane")
        if len(self.names) < 1 or len(self.names) > self.capacity:
            raise ValueError("tracer names must fit within block storage capacity")
        expected_blocks = (len(self.names) + self.block_width - 1) // self.block_width
        if self.block_count != expected_blocks:
            raise ValueError(
                f"expected {expected_blocks} tracer blocks for {len(self.names)} names, "
                f"found {self.block_count}"
            )
        if len(self.units) not in (0, len(self.names)):
            raise ValueError("tracer units must be empty or match tracer names")

    @classmethod
    def from_canonical(
        cls,
        *,
        names: tuple[str, ...],
        data: ArrayStorage,
        units: tuple[str, ...],
        coords: dict[str, np.ndarray],
        block_width: int | None = None,
    ) -> TracerField:
        field = cls(names=names, data=data, units=units, coords=coords)
        return field if block_width is None else field.reblock(block_width)

    @property
    def block_data(self) -> ArrayStorage:
        """Return physical ``(time, block, lev, lat, lon, lane)`` storage."""

        return self._data

    @property
    def data(self) -> ArrayStorage:
        """Return a zero-copy canonical view for a one-block field.

        Call :meth:`to_canonical` explicitly when multiple blocks must be
        joined.
        """

        return self.canonical_view()

    @property
    def shape(self) -> tuple[int, ...]:
        """Logical canonical shape, excluding block padding."""

        return (*self._data.shape[:1], *self._data.shape[2:5], self.tracer_count)

    @property
    def block_count(self) -> int:
        return self._data.shape[1]

    @property
    def block_width(self) -> int:
        return self._data.shape[-1]

    @property
    def tracer_count(self) -> int:
        return len(self.names)

    @property
    def capacity(self) -> int:
        return self.block_count * self.block_width

    def block_bounds(self, block: int) -> tuple[int, int]:
        if block < 0 or block >= self.block_count:
            raise IndexError("tracer block index out of range")
        start = block * self.block_width
        return start, min(start + self.block_width, self.tracer_count)

    def block(self, block: int) -> TracerField:
        """Return active lanes in one block as a zero-copy one-block field."""

        start, stop = self.block_bounds(block)
        active = stop - start
        units = self.units[start:stop] if self.units else ()
        return TracerField(
            names=self.names[start:stop],
            data=self._data[:, block : block + 1, :, :, :, :active],
            units=units,
            coords=self.coords,
        )

    def iter_blocks(self):
        for block in range(self.block_count):
            yield self.block(block)

    def tracer(self, tracer: int) -> ArrayStorage:
        """Return one logical tracer as a zero-copy ``(time, lev, lat, lon)`` view."""

        if tracer < 0 or tracer >= self.tracer_count:
            raise IndexError("tracer index out of range")
        block, lane = divmod(tracer, self.block_width)
        return self._data[:, block, :, :, :, lane]

    def canonical_view(self) -> ArrayStorage:
        """Return canonical storage without copying, requiring one block."""

        if self.block_count != 1:
            raise ValueError(
                "canonical storage is not contiguous across multiple blocks; "
                "use to_canonical() explicitly"
            )
        return self._data[:, 0, :, :, :, : self.tracer_count]

    def to_canonical(self) -> ArrayStorage:
        """Return canonical ``(time, lev, lat, lon, tracer)`` storage."""

        if self.block_count == 1:
            return self.canonical_view()
        array_module = _storage_array_module(self._data)
        canonical = array_module.empty(self.shape, dtype=self._data.dtype)
        for block in range(self.block_count):
            start, stop = self.block_bounds(block)
            canonical[..., start:stop] = self._data[
                :, block, :, :, :, : stop - start
            ]
        return canonical

    def reblock(self, block_width: int) -> TracerField:
        """Return the same logical field stored with a new block width."""

        if block_width < 1:
            raise ValueError("block_width must be positive")
        if self.block_width == block_width:
            return self
        ntime, nlev, nlat, nlon, _ = self.shape
        nblock = (self.tracer_count + block_width - 1) // block_width
        array_module = _storage_array_module(self._data)
        storage = array_module.zeros(
            (ntime, nblock, nlev, nlat, nlon, block_width),
            dtype=self._data.dtype,
        )
        start = 0
        while start < self.tracer_count:
            source_block, source_lane = divmod(start, self.block_width)
            target_block, target_lane = divmod(start, block_width)
            count = min(
                self.block_width - source_lane,
                block_width - target_lane,
                self.tracer_count - start,
            )
            storage[
                :, target_block, :, :, :, target_lane : target_lane + count
            ] = self._data[
                :, source_block, :, :, :, source_lane : source_lane + count
            ]
            start += count
        return TracerField(
            names=self.names,
            data=storage,
            units=self.units,
            coords=self.coords,
        )


def _coerce_storage(values: ArrayStorage) -> ArrayStorage:
    if isinstance(values, np.ndarray):
        return np.asarray(values)
    if hasattr(values, "__cuda_array_interface__"):
        _cupy_for_storage(values)
        return values
    return np.asarray(values)


def _storage_array_module(values: ArrayStorage) -> Any:
    if isinstance(values, np.ndarray):
        return np
    if hasattr(values, "__cuda_array_interface__"):
        return _cupy_for_storage(values)
    raise TypeError(f"unsupported tracer storage type {type(values).__name__}")


def _cupy_for_storage(values: ArrayStorage) -> Any:
    from wombat_transport.cuda.runtime import require_cupy

    cupy = require_cupy()
    if not isinstance(values, cupy.ndarray):
        raise TypeError("CUDA tracer storage must be a CuPy array")
    return cupy


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
