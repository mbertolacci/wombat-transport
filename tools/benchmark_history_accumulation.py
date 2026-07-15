#!/usr/bin/env python3
"""Benchmark HISTORY accumulation at exact GEOS 2x2.5 state sizes."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import statistics
import time

import numpy as np

from wombat_transport.history_accumulation import HISTORY_NUMBA_ENV
from wombat_transport.history_accumulation import _NUMBA_AVAILABLE
from wombat_transport.history_accumulation import accumulate_history_sum


GEOS_SHAPE = (1, 47, 91, 144)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+", default=(60, 80, 100))
    parser.add_argument("--threads", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(*args.counts, *args.threads, args.repeat, args.iterations) < 1:
        parser.error("counts, threads, repeat, and iterations must be positive")

    rows: list[dict[str, object]] = []
    affinity = _cpu_affinity()
    print(f"CPU affinity: {affinity}")
    print("tracers,mode,threads,best_ms,mean_ms,speedup_mean,effective_GiB_s")
    for tracers in args.counts:
        shape = (*GEOS_SHAPE, tracers)
        values = np.full(shape, 1.0e-6, dtype=np.float64)
        accumulator = np.zeros_like(values)
        native_mean = 0.0
        configurations = [("native", 1), *(("numba", count) for count in args.threads)]
        for mode, threads in configurations:
            if mode == "numba" and not _NUMBA_AVAILABLE:
                continue
            os.environ[HISTORY_NUMBA_ENV] = "1" if mode == "numba" else "0"
            os.environ[f"{HISTORY_NUMBA_ENV}_THREADS"] = str(threads)
            accumulate_history_sum(accumulator, values)
            samples: list[float] = []
            for _ in range(args.repeat):
                accumulator.fill(0.0)
                start = time.perf_counter()
                for _ in range(args.iterations):
                    accumulate_history_sum(accumulator, values)
                samples.append((time.perf_counter() - start) / args.iterations)
            best = min(samples)
            mean = statistics.mean(samples)
            if mode == "native":
                native_mean = mean
            traffic_gib = 3 * accumulator.nbytes / 1024**3
            row = {
                "tracers": tracers,
                "mode": mode,
                "threads": threads,
                "best_s_per_update": best,
                "mean_s_per_update": mean,
                "speedup_vs_native_mean": native_mean / mean,
                "effective_gib_s_mean": traffic_gib / mean,
                "state_gib": accumulator.nbytes / 1024**3,
                "iterations": args.iterations,
                "repeats": args.repeat,
                "cpu_affinity": affinity,
            }
            rows.append(row)
            print(
                f"{tracers},{mode},{threads},{best * 1000:.3f},{mean * 1000:.3f},"
                f"{native_mean / mean:.2f},{traffic_gib / mean:.1f}"
            )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _cpu_affinity() -> str:
    if hasattr(os, "sched_getaffinity"):
        return ",".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0)))
    return "unknown"


if __name__ == "__main__":
    main()
