"""Optional parallel accumulation for time-averaged HISTORY output."""

from __future__ import annotations

import numpy as np

from wombat_transport.transport.numba_control import apply_numba_thread_count
from wombat_transport.transport.numba_control import numba_enabled

try:  # Optional acceleration dependency.
    from numba import njit
    from numba import prange
except ImportError:  # pragma: no cover - exercised in environments without numba.
    njit = None
    prange = range


HISTORY_NUMBA_ENV = "WOMBAT_HISTORY_NUMBA"
_NUMBA_AVAILABLE = njit is not None


def accumulate_history_sum(accumulator: np.ndarray, values: np.ndarray) -> None:
    """Add one state to a HISTORY accumulator without changing addition order."""

    if _history_numba_enabled() and accumulator.flags.c_contiguous and values.flags.c_contiguous:
        apply_numba_thread_count(HISTORY_NUMBA_ENV, available=_NUMBA_AVAILABLE)
        _accumulate_history_sum_numba_kernel(accumulator.reshape(-1), values.reshape(-1))
        return
    np.add(accumulator, values, out=accumulator)


def _history_numba_enabled() -> bool:
    return numba_enabled(HISTORY_NUMBA_ENV, available=_NUMBA_AVAILABLE)


if njit is not None:

    @njit(cache=True, parallel=True, nogil=True)
    def _accumulate_history_sum_numba_kernel(accumulator: np.ndarray, values: np.ndarray) -> None:
        for index in prange(accumulator.size):
            accumulator[index] += values[index]

else:

    def _accumulate_history_sum_numba_kernel(accumulator: np.ndarray, values: np.ndarray) -> None:
        raise RuntimeError("numba is not available")
