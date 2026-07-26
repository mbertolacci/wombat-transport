"""Minimal explicit ownership boundary for optional CuPy arrays."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np


class CudaUnavailableError(RuntimeError):
    """Raised when the optional CUDA runtime cannot be initialized."""


@dataclass(frozen=True)
class CudaDeviceInfo:
    device_id: int
    name: str
    compute_capability: str


@dataclass(frozen=True)
class TransferStats:
    host_to_device_count: int = 0
    host_to_device_bytes: int = 0
    device_to_host_count: int = 0
    device_to_host_bytes: int = 0
    explicit_synchronizations: int = 0


def require_cupy() -> Any:
    try:
        return import_module("cupy")
    except (ImportError, OSError) as exc:
        raise CudaUnavailableError(
            "CUDA execution requires the optional 'cuda' dependency extra"
        ) from exc


class CudaRuntime:
    """Own one CUDA device and account for explicit array transfers."""

    def __init__(self, device_id: int = 0) -> None:
        cupy = require_cupy()
        try:
            device_count = int(cupy.cuda.runtime.getDeviceCount())
            if device_id < 0 or device_id >= device_count:
                raise ValueError(
                    f"CUDA device {device_id} is outside the available range "
                    f"0..{device_count - 1}"
                )
            device = cupy.cuda.Device(device_id)
            device.use()
            properties = cupy.cuda.runtime.getDeviceProperties(device_id)
        except ValueError:
            raise
        except Exception as exc:
            raise CudaUnavailableError(
                f"could not initialize CUDA device {device_id}: {exc}"
            ) from exc

        raw_name = properties["name"]
        name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
        self._cupy = cupy
        self._device = device
        self._device_info = CudaDeviceInfo(
            device_id=device_id,
            name=name,
            compute_capability=str(device.compute_capability),
        )
        self._host_to_device_count = 0
        self._host_to_device_bytes = 0
        self._device_to_host_count = 0
        self._device_to_host_bytes = 0
        self._explicit_synchronizations = 0

    @property
    def array_module(self) -> Any:
        """Return CuPy for explicitly CUDA-specific array operations."""

        return self._cupy

    @property
    def device_info(self) -> CudaDeviceInfo:
        return self._device_info

    @property
    def transfer_stats(self) -> TransferStats:
        return TransferStats(
            host_to_device_count=self._host_to_device_count,
            host_to_device_bytes=self._host_to_device_bytes,
            device_to_host_count=self._device_to_host_count,
            device_to_host_bytes=self._device_to_host_bytes,
            explicit_synchronizations=self._explicit_synchronizations,
        )

    def reset_transfer_stats(self) -> None:
        self._host_to_device_count = 0
        self._host_to_device_bytes = 0
        self._device_to_host_count = 0
        self._device_to_host_bytes = 0
        self._explicit_synchronizations = 0

    def is_device_array(self, values: object) -> bool:
        return isinstance(values, self._cupy.ndarray)

    def to_device(
        self,
        values: np.ndarray,
        *,
        dtype: np.dtype[Any] | type[Any] | str | None = None,
    ) -> Any:
        """Copy one host array to this runtime's device."""

        if self.is_device_array(values):
            raise TypeError("to_device expects a host array, not a CuPy array")
        host = np.asarray(values, dtype=dtype, order="C")
        with self._device:
            device = self._cupy.asarray(host)
        self._host_to_device_count += 1
        self._host_to_device_bytes += host.nbytes
        return device

    def to_host(self, values: object) -> np.ndarray:
        """Copy one array from this runtime's device to the host."""

        if not self.is_device_array(values):
            raise TypeError("to_host expects a CuPy array")
        with self._device:
            host = self._cupy.asnumpy(values)
        self._device_to_host_count += 1
        self._device_to_host_bytes += host.nbytes
        return host

    def copy_to_device(
        self,
        destination: object,
        values: np.ndarray,
    ) -> None:
        """Replace a persistent device buffer from identically shaped host data."""

        if not self.is_device_array(destination):
            raise TypeError("copy_to_device destination must be a CuPy array")
        host = np.asarray(values, dtype=destination.dtype, order="C")
        if host.shape != destination.shape:
            raise ValueError(
                f"device destination {destination.shape} does not match "
                f"host source {host.shape}"
            )
        with self._device:
            destination.set(host)
        self._host_to_device_count += 1
        self._host_to_device_bytes += host.nbytes

    def empty(
        self,
        shape: tuple[int, ...],
        *,
        dtype: np.dtype[Any] | type[Any] | str,
    ) -> Any:
        with self._device:
            return self._cupy.empty(shape, dtype=dtype)

    def zeros(
        self,
        shape: tuple[int, ...],
        *,
        dtype: np.dtype[Any] | type[Any] | str,
    ) -> Any:
        with self._device:
            return self._cupy.zeros(shape, dtype=dtype)

    def synchronize(self) -> None:
        with self._device:
            self._device.synchronize()
        self._explicit_synchronizations += 1
