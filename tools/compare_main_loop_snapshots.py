#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


BOUNDARIES = (
    "before_do_transport",
    "before_tpcore_fvdas",
    "after_do_transport",
    "before_do_vdiff",
    "after_do_vdiff",
    "before_do_convection",
    "after_do_convection",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare GC and Wombat full-grid main-loop tracer snapshots.")
    parser.add_argument("gc_dir", type=Path)
    parser.add_argument("wombat_dir", type=Path)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--boundary", action="append", default=None)
    parser.add_argument("--air-mw-g-mol", type=float, default=28.965)
    parser.add_argument("--tracer-mw-g-mol", type=float, default=44.01)
    args = parser.parse_args()

    scale = args.air_mw_g_mol / args.tracer_mw_g_mol
    boundaries = tuple(args.boundary or BOUNDARIES)
    print("boundary,max_abs_ppm,mean_abs_ppm,bias_ppm,rms_ppm,max_index,actual_ppm,expected_ppm")
    for boundary in boundaries:
        gc_path = args.gc_dir / f"{boundary}_{args.step:06d}.bin"
        wombat_path = args.wombat_dir / f"{boundary}_{args.step:06d}.bin"
        if not gc_path.exists() or not wombat_path.exists():
            print(f"{boundary},missing,missing,missing,missing,missing,missing,missing")
            continue
        gc = _read_snapshot(gc_path) * scale
        wombat = _read_snapshot(wombat_path)
        if gc.shape != wombat.shape:
            raise ValueError(f"{boundary}: shape mismatch GC {gc.shape} vs Wombat {wombat.shape}")
        diff = wombat - gc
        abs_diff = np.abs(diff)
        idx = tuple(int(item) for item in np.unravel_index(int(np.argmax(abs_diff)), abs_diff.shape))
        print(
            f"{boundary},"
            f"{float(abs_diff[idx] * 1.0e6):.12e},"
            f"{float(np.mean(abs_diff) * 1.0e6):.12e},"
            f"{float(np.mean(diff) * 1.0e6):.12e},"
            f"{float(np.sqrt(np.mean(diff * diff)) * 1.0e6):.12e},"
            f"{idx},"
            f"{float(wombat[idx] * 1.0e6):.12e},"
            f"{float(gc[idx] * 1.0e6):.12e}"
        )
    return 0


def _read_snapshot(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        dims = np.fromfile(handle, dtype="<i4", count=4)
        if dims.size != 4:
            raise ValueError(f"{path}: missing snapshot dimensions")
        endian = "<"
        if np.any(dims <= 0) or int(np.prod(dims, dtype=np.int64)) > 1_000_000_000:
            handle.seek(0)
            dims = np.fromfile(handle, dtype=">i4", count=4)
            endian = ">"
        count = int(np.prod(dims, dtype=np.int64))
        values = np.fromfile(handle, dtype=f"{endian}f8", count=count)
    if values.size != count:
        raise ValueError(f"{path}: expected {count} values, got {values.size}")
    return values.reshape(tuple(int(item) for item in dims), order="F")


if __name__ == "__main__":
    raise SystemExit(main())
