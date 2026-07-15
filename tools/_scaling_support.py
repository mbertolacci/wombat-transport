from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import resource
import sys
from typing import Any, Iterable


AUTO_MEMORY_FRACTION = 0.55


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def tracer_state_bytes(ntracer: int, grid_shape: tuple[int, int, int]) -> int:
    return int(ntracer) * math.prod(grid_shape) * 8


def estimate_peak_bytes(
    ntracer: int,
    grid_shape: tuple[int, int, int],
    *,
    multiplier: float,
    fixed_overhead_bytes: int,
) -> int:
    return int(tracer_state_bytes(ntracer, grid_shape) * multiplier + fixed_overhead_bytes)


def count_is_allowed(
    estimated_peak_bytes: int,
    memory_limit_bytes: int | None,
    *,
    detailed_reason: bool = False,
) -> tuple[bool, str]:
    if memory_limit_bytes is None:
        reason = "memory limit unavailable"
        if detailed_reason:
            reason += "; running without auto skip"
        return True, reason
    if estimated_peak_bytes <= memory_limit_bytes:
        return True, ""
    if detailed_reason:
        return (
            False,
            f"estimated peak {bytes_to_mib(estimated_peak_bytes):.1f} MiB exceeds memory limit "
            f"{bytes_to_mib(memory_limit_bytes):.1f} MiB",
        )
    return False, "estimated peak exceeds memory limit"


def memory_limit_bytes(value: str, *, fraction: float = AUTO_MEMORY_FRACTION) -> int | None:
    if value == "auto":
        physical = physical_memory_bytes()
        return None if physical is None else int(physical * fraction)
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError("memory limit must be positive or 'auto'")
    return int(parsed * 1024**3)


def physical_memory_bytes() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    if pages <= 0 or page_size <= 0:
        return None
    return int(pages) * int(page_size)


def peak_rss_mib() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss / 1024.0**2 if sys.platform == "darwin" else rss / 1024.0


def bytes_to_mib(value: int) -> float:
    return float(value) / 1024.0**2


def format_optional_fixed(value: float | None, *, precision: int = 8) -> str:
    if value is None:
        return ""
    return f"{value:.{precision}f}"


def format_optional_general(value: float | None, *, precision: int = 8) -> str:
    if value is None:
        return ""
    return f"{value:.{precision}g}"


def write_rows(rows: Iterable[Any], fieldnames: tuple[str, ...], output: Path | None) -> None:
    if output is None:
        _write_csv(rows, fieldnames, sys.stdout)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        _write_csv(rows, fieldnames, handle)


def _write_csv(rows: Iterable[Any], fieldnames: tuple[str, ...], handle: Any) -> None:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row.as_csv_row())
