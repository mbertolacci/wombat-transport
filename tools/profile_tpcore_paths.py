from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from wombat_transport.transport.tpcore import _numba as nb
from wombat_transport.transport.tpcore._native import setup_tpcore_terms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Count geometry-dependent TPCORE Numba hot-kernel paths.")
    parser.add_argument("--run-config", type=Path, default=Path("validation_runs/cases/realistic_restart_noemis/wombat/main/run.yml"))
    parser.add_argument("--tracers", type=int, default=24)
    parser.add_argument("--dt-s", type=float, default=600.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    bench = _load_benchmark_module()
    inputs = bench._build_synthetic_tpcore_inputs(args.run_config, args.tracers, dt_s=args.dt_s)
    setup = setup_tpcore_terms(
        p1_hpa=inputs.p1_hpa,
        p2_hpa=inputs.p2_hpa,
        u_m_s=inputs.u_m_s,
        v_m_s=inputs.v_m_s,
        area_m2=inputs.area_m2,
        hyai_hpa=inputs.hyai_hpa,
        hybi=inputs.hybi,
        lat_deg=inputs.lat_deg,
        dt_s=inputs.dt_s,
    )
    jn = np.empty(setup.cx.shape[0], dtype=np.int64)
    js = np.empty_like(jn)
    nb._set_jn_js_numba_kernel(setup.cx, jn, js)
    payload = _count_paths(setup, jn, js, args.tracers)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text, encoding="utf-8")
    return 0


def _load_benchmark_module() -> Any:
    path = Path(__file__).with_name("benchmark_tpcore_scaling.py").resolve()
    spec = importlib.util.spec_from_file_location("benchmark_tpcore_scaling_paths", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _count_paths(setup: Any, jn: np.ndarray, js: np.ndarray, ntracer: int) -> dict[str, Any]:
    nlev, nlat, nlon = setup.cx.shape
    j1p = 2
    j2p = nlat - 3
    jvan = max(1, nlat // 18)
    x_rows = {"edge": 0, "near_pole": 0, "ppm": 0, "large_courant": 0}
    x_large_cells = {"positive": 0, "negative": 0, "fractional": 0}
    x_flux_sign = {"positive": 0, "nonpositive": 0}
    for level in range(nlev):
        for j in range(j1p, j2p + 1):
            if j > int(js[level]) and j < int(jn[level]):
                if j == j1p or j == j2p:
                    x_rows["edge"] += 1
                elif j <= j1p + jvan or j >= j2p - jvan:
                    x_rows["near_pole"] += 1
                else:
                    x_rows["ppm"] += 1
            else:
                x_rows["large_courant"] += 1
                values = setup.cx[level, j]
                x_large_cells["positive"] += int(np.count_nonzero(values > 1.0))
                x_large_cells["negative"] += int(np.count_nonzero(values < -1.0))
                x_large_cells["fractional"] += int(np.count_nonzero((values >= -1.0) & (values <= 1.0)))
            values = setup.cx[level, j]
            x_flux_sign["positive"] += int(np.count_nonzero(values > 0.0))
            x_flux_sign["nonpositive"] += int(np.count_nonzero(values <= 0.0))

    y_values = setup.cy[:, j1p : j2p + 2, :]
    z_values = setup.vertical_mass_flux_hpa[:, np.r_[0, np.arange(2, nlat - 2), nlat - 1], :]
    y_positive = int(np.count_nonzero(y_values > 0.0))
    y_total = int(y_values.size)
    z_positive = int(np.count_nonzero(z_values[:-1] > 0.0))
    z_total = int(z_values[:-1].size)
    x_total_rows = sum(x_rows.values())
    return {
        "shape": {"levels": nlev, "latitudes": nlat, "longitudes": nlon, "tracers": ntracer},
        "xtp": {
            "rows": x_rows,
            "row_percent": {name: value / x_total_rows * 100.0 for name, value in x_rows.items()},
            "large_courant_cells": x_large_cells,
            "flux_sign_cells": x_flux_sign,
            "ppm_limiter_evaluations": x_rows["ppm"] * nlon * ntracer,
        },
        "ytp": {
            "positive_flux_cells": y_positive,
            "nonpositive_flux_cells": y_total - y_positive,
            "limiter_evaluations": nlev * nlon * (nlat - 2) * ntracer,
        },
        "fzppm": {
            "processed_latitude_rows": nlat - 2,
            "skipped_latitude_rows": 2,
            "positive_flux_interfaces": z_positive,
            "nonpositive_flux_interfaces": z_total - z_positive,
            "edge_limiter_evaluations": (nlat - 2) * nlon * 4 * ntracer,
            "interior_limiter_evaluations": (nlat - 2) * nlon * max(nlev - 4, 0) * ntracer,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
