from __future__ import annotations

from datetime import datetime
import math
from typing import Any

import numpy as np

MICROSECONDS_PER_SECOND = 1_000_000

def _datetime_to_microseconds(value: datetime) -> int:
    epoch = datetime(1970, 1, 1)
    delta = value - epoch
    return (delta.days * 86400 + delta.seconds) * MICROSECONDS_PER_SECOND + delta.microseconds


def _seconds_to_microseconds(value: float, label: str) -> int:
    seconds = float(value)
    if not np.isfinite(seconds) or seconds <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    microseconds = int(round(seconds * MICROSECONDS_PER_SECOND))
    if not math.isclose(microseconds / MICROSECONDS_PER_SECOND, seconds, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"{label} must be representable to microsecond precision")
    return microseconds

def _nul_padded_matrix(values: Any, width: int, count: int) -> np.ndarray:
    output = np.full((count, width), b"\x00", dtype="S1")
    for index, value in enumerate(values):
        encoded = _fixed_width_utf8(value, width, "encoded string")
        output[index, : len(encoded)] = np.frombuffer(encoded, dtype="S1")
    return output


def _fixed_width_utf8(value: str, width: int, label: str) -> bytes:
    _validate_fixed_width_utf8(value, width, label)
    return value.encode("utf-8")


def _validate_fixed_width_utf8(value: str, width: int, label: str) -> None:
    if "\x00" in value:
        raise ValueError(f"{label} must not contain NUL characters")
    encoded_length = len(value) if value.isascii() else len(value.encode("utf-8"))
    if encoded_length > width:
        raise ValueError(f"{label} exceeds fixed UTF-8 width {width} bytes")


def _validate_vertical_values(
    values: np.ndarray,
    weights: np.ndarray,
    unit: str,
    nlev: int,
    label: str,
) -> None:
    if (
        values.size == 0
        or values.size != weights.size
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(weights))
    ):
        raise ValueError(f"{label} has invalid exact vertical values or weights")
    if unit == "pressure_level" and (
        np.any(values != np.floor(values)) or np.any(values < 1) or np.any(values > nlev)
    ):
        raise ValueError(f"{label} has invalid pressure levels")
    if unit != "pressure_level" and np.any(values < 0.0):
        raise ValueError(f"{label} has negative vertical values")


def _validate_vertical_bounds(
    start: float,
    end: float,
    unit: str,
    nlev: int,
    label: str,
) -> None:
    if not np.isfinite(start) or not np.isfinite(end) or start > end or start < 0.0:
        raise ValueError(f"{label} has invalid vertical bounds")
    if unit == "pressure_level" and (
        start != math.floor(start) or end != math.floor(end) or start < 1 or end > nlev
    ):
        raise ValueError(f"{label} has invalid pressure-level bounds")
