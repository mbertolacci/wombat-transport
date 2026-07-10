#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


FIELDS = (
    ("p_tp1_hpa", "2d"),
    ("p_tp2_hpa", "2d"),
    ("xmass_hpa", "3d"),
    ("ymass_hpa", "3d"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare full-grid TPCORE input snapshots from GC and Wombat.")
    parser.add_argument("gc_dir", type=Path)
    parser.add_argument("wombat_dir", type=Path)
    parser.add_argument("--step", type=int, default=0)
    args = parser.parse_args()

    gc = _read_snapshot(args.gc_dir / f"before_tpcore_fvdas_inputs_{args.step:06d}.bin")
    wombat = _read_snapshot(args.wombat_dir / f"before_tpcore_fvdas_inputs_{args.step:06d}.bin")
    print("field,max_abs,mean_abs,bias,rms,max_index,actual,expected")
    for name, _kind in FIELDS:
        actual = wombat[name]
        expected = gc[name]
        if actual.shape != expected.shape:
            raise ValueError(f"{name}: shape mismatch {actual.shape} vs {expected.shape}")
        diff = actual - expected
        abs_diff = np.abs(diff)
        idx = tuple(int(item) for item in np.unravel_index(int(np.argmax(abs_diff)), abs_diff.shape))
        print(
            f"{name},"
            f"{float(abs_diff[idx]):.12e},"
            f"{float(np.mean(abs_diff)):.12e},"
            f"{float(np.mean(diff)):.12e},"
            f"{float(np.sqrt(np.mean(diff * diff))):.12e},"
            f"{idx},"
            f"{float(actual[idx]):.12e},"
            f"{float(expected[idx]):.12e}"
        )
    return 0


def _read_snapshot(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        dims = np.fromfile(handle, dtype="<i4", count=3)
        endian = "<"
        if dims.size != 3:
            raise ValueError(f"{path}: missing dimensions")
        if np.any(dims <= 0) or int(np.prod(dims, dtype=np.int64)) > 1_000_000_000:
            handle.seek(0)
            dims = np.fromfile(handle, dtype=">i4", count=3)
            endian = ">"
        nx, ny, nz = (int(item) for item in dims)
        p_count = nx * ny
        m_count = nx * ny * nz
        p1 = np.fromfile(handle, dtype=f"{endian}f8", count=p_count).reshape((nx, ny), order="F")
        p2 = np.fromfile(handle, dtype=f"{endian}f8", count=p_count).reshape((nx, ny), order="F")
        xmass = np.fromfile(handle, dtype=f"{endian}f8", count=m_count).reshape((nx, ny, nz), order="F")
        ymass = np.fromfile(handle, dtype=f"{endian}f8", count=m_count).reshape((nx, ny, nz), order="F")
    return {
        "p_tp1_hpa": p1,
        "p_tp2_hpa": p2,
        "xmass_hpa": xmass,
        "ymass_hpa": ymass,
    }


if __name__ == "__main__":
    raise SystemExit(main())
