from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DATA = REPO_ROOT / "external_data"
BASE_CONFIG = REPO_ROOT / "validation_runs/cases/realistic_restart_noemis/wombat/main/run.yml"
RESIDUAL_CONFIG = REPO_ROOT / "validation_runs/cases/residual_24tracer_emissions_1day/wombat/main/run.yml"
INITIAL_RESTART = EXTERNAL_DATA / "restarts/GEOSChem.Restart.20140901_0000z.nc4"
MERRA2_ROOT = EXTERNAL_DATA / "geoschem/GEOS_2x2.5/MERRA2"
FOUR_BY_FIVE_RESTART = EXTERNAL_DATA / "restarts/4x5/GEOSChem.Restart.20140901_0000z.nc4"
MERRA2_4X5_ROOT = EXTERNAL_DATA / "geoschem/GEOS_4x5/MERRA2"
FLUX_ROOT = EXTERNAL_DATA / "fluxes"
SCALING_GRID_ROOT = EXTERNAL_DATA / "scaling-grids"
OBSOPERATOR_ROOT = EXTERNAL_DATA / "obsoperator"


def _skip_marker(label: str, *paths: Path):
    missing = [path for path in paths if not path.exists()]
    rendered = ", ".join(str(path.relative_to(REPO_ROOT)) for path in missing)
    return pytest.mark.skipif(bool(missing), reason=f"{label} requires unavailable external data: {rendered}")


requires_restart = _skip_marker("full-grid test", INITIAL_RESTART)
requires_transport_data = _skip_marker("real transport test", INITIAL_RESTART, MERRA2_ROOT)
requires_4x5_transport_data = _skip_marker(
    "real 4x5 transport test", FOUR_BY_FIVE_RESTART, MERRA2_4X5_ROOT
)
requires_residual_data = _skip_marker(
    "residual emissions test",
    INITIAL_RESTART,
    FLUX_ROOT,
    SCALING_GRID_ROOT,
)


def require_external_paths(*paths: Path, purpose: str = "real-data test") -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        rendered = ", ".join(str(path.relative_to(REPO_ROOT)) for path in missing)
        pytest.skip(f"{purpose} requires unavailable external data: {rendered}")
