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
        encoded = value.encode("utf-8")
        if len(encoded) > width:
            raise ValueError(f"encoded string exceeds fixed width {width}")
        output[index, : len(encoded)] = np.frombuffer(encoded, dtype="S1")
    return output
