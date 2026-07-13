from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _load_profile_module():
    path = Path(__file__).parents[1] / "tools" / "profile_tpcore_paths.py"
    spec = importlib.util.spec_from_file_location("profile_tpcore_paths", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


profile = _load_profile_module()


def test_count_paths_splits_common_and_fallback_branches():
    nlev, nlat, nlon, ntracer = 5, 7, 2, 4
    cx = np.full((nlev, nlat, nlon), 0.5)
    cx[:, 2, 0] = 2.0
    cx[:, 2, 1] = -2.0
    setup = SimpleNamespace(
        cx=cx,
        cy=np.ones((nlev, nlat, nlon)),
        vertical_mass_flux_hpa=np.ones((nlev, nlat, nlon)),
    )
    jn = np.full(nlev, nlat - 3, dtype=np.int64)
    js = np.full(nlev, 2, dtype=np.int64)

    counts = profile._count_paths(setup, jn, js, ntracer)

    assert counts["xtp"]["rows"] == {"edge": 0, "near_pole": 5, "ppm": 0, "large_courant": 10}
    assert counts["xtp"]["large_courant_cells"] == {"positive": 5, "negative": 5, "fractional": 10}
    assert counts["xtp"]["flux_sign_cells"] == {"positive": 25, "nonpositive": 5}
    assert counts["ytp"]["positive_flux_cells"] == 40
    assert counts["fzppm"]["positive_flux_interfaces"] == 40
    assert counts["fzppm"]["edge_limiter_evaluations"] == 160
    assert counts["fzppm"]["interior_limiter_evaluations"] == 40
