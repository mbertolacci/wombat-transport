"""Resident HISTORY accumulation using ordinary CuPy operations."""

from __future__ import annotations

from typing import Any

from wombat_transport.cuda.runtime import require_cupy


def accumulate_history_sum(accumulator: Any, values: Any) -> None:
    """Add one resident state to one resident HISTORY accumulator."""

    cupy = require_cupy()
    _validate_device_array(cupy, accumulator, "accumulator")
    _validate_device_array(cupy, values, "values")
    if accumulator.shape != values.shape:
        raise ValueError(
            f"HISTORY accumulator {accumulator.shape} does not match "
            f"tracer storage {values.shape}"
        )
    if accumulator.dtype != values.dtype and str(accumulator.dtype) != "float64":
        raise TypeError(
            "HISTORY accumulation requires matching dtypes or a float64 "
            f"accumulator, found {accumulator.dtype} and {values.dtype}"
        )
    cupy.add(accumulator, values, out=accumulator)


def accumulate_history_sums(accumulators: Any, values: Any) -> None:
    """Add one resident state to every resident HISTORY accumulator."""

    cupy = require_cupy()
    _validate_device_array(cupy, accumulators, "accumulators")
    _validate_device_array(cupy, values, "values")
    if accumulators.ndim != values.ndim + 1 or accumulators.shape[1:] != values.shape:
        raise ValueError(
            f"HISTORY accumulator stack {accumulators.shape} does not match "
            f"tracer storage {values.shape}"
        )
    if accumulators.dtype != values.dtype and str(accumulators.dtype) != "float64":
        raise TypeError(
            "HISTORY accumulation requires matching dtypes or float64 "
            f"accumulators, found {accumulators.dtype} and {values.dtype}"
        )
    cupy.add(accumulators, values[None, ...], out=accumulators)


def _validate_device_array(cupy: Any, values: Any, label: str) -> None:
    if not isinstance(values, cupy.ndarray):
        raise TypeError(f"{label} must be a CuPy array")
