"""Optional parallel accumulation for time-averaged HISTORY output."""

from __future__ import annotations

import numpy as np

from wombat_transport.transport.numba_control import configure_numba_threads
from wombat_transport.transport.numba_control import numba_available_and_enabled

try:  # Optional acceleration dependency.
    from numba import njit
    from numba import prange
except ImportError:  # pragma: no cover - exercised in environments without numba.
    njit = None
    prange = range


_NUMBA_AVAILABLE = njit is not None


def accumulate_history_sum(accumulator: np.ndarray, values: np.ndarray) -> None:
    """Add one state to a HISTORY accumulator without changing addition order."""

    if _history_numba_enabled() and accumulator.flags.c_contiguous and values.flags.c_contiguous:
        configure_numba_threads(available=_NUMBA_AVAILABLE)
        _accumulate_history_sum_numba_kernel(accumulator.reshape(-1), values.reshape(-1))
        return
    np.add(accumulator, values, out=accumulator)


def accumulate_history_sums(accumulators: np.ndarray, values: np.ndarray) -> None:
    """Add one state to every collection in a stacked HISTORY accumulator."""

    if accumulators.ndim != values.ndim + 1 or accumulators.shape[1:] != values.shape:
        raise ValueError(
            f"HISTORY accumulator stack {accumulators.shape} does not match "
            f"tracer storage {values.shape}"
        )
    if accumulators.shape[0] == 1:
        accumulate_history_sum(accumulators[0], values)
        return
    if (
        _history_numba_enabled()
        and accumulators.flags.c_contiguous
        and values.flags.c_contiguous
    ):
        configure_numba_threads(available=_NUMBA_AVAILABLE)
        _accumulate_history_sums_numba_kernel(
            accumulators.reshape(accumulators.shape[0], -1),
            values.reshape(-1),
        )
        return
    for accumulator in accumulators:
        np.add(accumulator, values, out=accumulator)


def _history_numba_enabled() -> bool:
    return numba_available_and_enabled(available=_NUMBA_AVAILABLE)


if njit is not None:

    @njit(cache=True, parallel=True, nogil=True)
    def _accumulate_history_sum_numba_kernel(accumulator: np.ndarray, values: np.ndarray) -> None:
        for index in prange(accumulator.size):
            accumulator[index] += values[index]

    @njit(cache=True, parallel=True, nogil=True)
    def _accumulate_history_sums_numba_kernel(
        accumulators: np.ndarray,
        values: np.ndarray,
    ) -> None:
        for index in prange(values.size):
            value = values[index]
            for collection in range(accumulators.shape[0]):
                accumulators[collection, index] += value

else:

    def _accumulate_history_sum_numba_kernel(accumulator: np.ndarray, values: np.ndarray) -> None:
        raise RuntimeError("numba is not available")

    def _accumulate_history_sums_numba_kernel(
        accumulators: np.ndarray,
        values: np.ndarray,
    ) -> None:
        raise RuntimeError("numba is not available")
