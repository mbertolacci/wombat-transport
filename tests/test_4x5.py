from __future__ import annotations

from datetime import datetime
import importlib.util
from pathlib import Path
import sys

import netCDF4
import numpy as np
import pytest

from tests.data_paths import (
    FOUR_BY_FIVE_RESTART,
    MERRA2_4X5_ROOT,
    requires_4x5_transport_data,
)
from wombat_transport.constants import EARTH_RADIUS_M
from wombat_transport.emissions import conservative_regrid_horizontal
from wombat_transport.grid import (
    geos_chem_grid_cell_area_m2,
    geos_chem_horizontal_centers,
    geos_chem_latitude_edges_deg,
    load_transport_grid,
)
from wombat_transport.io import initialize_tracers, load_restart, write_restart_like
from wombat_transport.transport import (
    TransportExecutor,
    load_transport_forcing,
    merra2_filename,
    run_transport_one_step,
    run_transport_step_with_executor,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FIXTURE = REPO_ROOT / "tests/fixtures/io_readers_2x25_v1/restart.nc4"
BASE_SPECIES = REPO_ROOT / "validation_runs/cases/realistic_restart_noemis_2x25/wombat/main/species_database.yml"


def _load_restart_tool():
    path = REPO_ROOT / "tools/regrid_restart.py"
    spec = importlib.util.spec_from_file_location("regrid_restart", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


regrid_restart_tool = _load_restart_tool()


def test_geos_4x5_grid_has_half_height_polar_rows():
    lat, lon = geos_chem_horizontal_centers("4x5")
    edges = geos_chem_latitude_edges_deg(lat)
    area = geos_chem_grid_cell_area_m2(lat, lon)

    assert lat.size == 46
    assert lon.size == 72
    np.testing.assert_array_equal(lat[:3], [-89.0, -86.0, -82.0])
    np.testing.assert_array_equal(lat[-3:], [82.0, 86.0, 89.0])
    np.testing.assert_array_equal(edges[:4], [-90.0, -88.0, -84.0, -80.0])
    np.testing.assert_array_equal(edges[-4:], [80.0, 84.0, 88.0, 90.0])
    np.testing.assert_allclose(area.sum(), 4.0 * np.pi * EARTH_RADIUS_M**2, rtol=2.0e-15)


def test_conservative_regrid_to_4x5_preserves_constant_and_uniform_poles():
    source_lat, source_lon = geos_chem_horizontal_centers("2x25")
    target_lat, target_lon = geos_chem_horizontal_centers("4x5")
    values = np.ones((2, source_lat.size, source_lon.size), dtype=np.float64)
    values[:, 0, :] = np.arange(source_lon.size)
    values[:, -1, :] = -np.arange(source_lon.size)

    result = conservative_regrid_horizontal(
        values, source_lat, source_lon, target_lat, target_lon
    )

    assert result.shape == (2, 46, 72)
    np.testing.assert_array_equal(np.ptp(result[:, 0, :], axis=-1), 0.0)
    np.testing.assert_array_equal(np.ptp(result[:, -1, :], axis=-1), 0.0)
    np.testing.assert_allclose(result[:, 2:-2, :], 1.0, rtol=0.0, atol=0.0)

    source_edges = geos_chem_latitude_edges_deg(source_lat)
    target_edges = geos_chem_latitude_edges_deg(target_lat)
    source_area = geos_chem_grid_cell_area_m2(source_lat, source_lon)
    target_area = geos_chem_grid_cell_area_m2(target_lat, target_lon)
    for source_values, target_values in zip(values, result, strict=True):
        np.testing.assert_allclose(
            np.sum(target_values * target_area),
            np.sum(source_values * source_area),
            rtol=2.0e-15,
        )
        for target_row in (0, target_lat.size - 1):
            low = target_edges[target_row]
            high = target_edges[target_row + 1]
            overlap = np.maximum(
                0.0,
                np.sin(np.deg2rad(np.minimum(source_edges[1:], high)))
                - np.sin(np.deg2rad(np.maximum(source_edges[:-1], low))),
            )
            expected = np.sum(overlap * np.mean(source_values, axis=1)) / np.sum(overlap)
            np.testing.assert_allclose(target_values[target_row], expected, rtol=2.0e-15)


def test_restart_converter_produces_dynamic_4x5_template(tmp_path):
    output = tmp_path / "restart_4x5.nc4"
    regrid_restart_tool.regrid_restart(SOURCE_FIXTURE, output, target_grid="4x5", overwrite=False)

    grid = load_transport_grid(output)
    loaded = load_restart(output)
    assert grid.shape == (47, 46, 72)
    assert loaded.shape[1:4] == (47, 46, 72)
    with netCDF4.Dataset(output) as dataset:
        assert not any(name.startswith("Met_") for name in dataset.variables)
        assert dataset.variables["AREA"].dtype == np.dtype("float64")

    roundtrip = tmp_path / "roundtrip.nc4"
    write_restart_like(roundtrip, loaded, output)
    np.testing.assert_array_equal(load_restart(roundtrip).data, loaded.data)
    with pytest.raises(FileExistsError):
        regrid_restart_tool.regrid_restart(SOURCE_FIXTURE, output, target_grid="4x5", overwrite=False)


@requires_4x5_transport_data
def test_real_4x5_forcing_and_transport_chain(transport_numba_mode):
    grid = load_transport_grid(FOUR_BY_FIVE_RESTART)
    state = initialize_tracers(FOUR_BY_FIVE_RESTART, BASE_SPECIES)
    forcing = load_transport_forcing(MERRA2_4X5_ROOT, datetime(2014, 9, 1), grid)

    assert merra2_filename(datetime(2014, 9, 1), "A1", grid) == "MERRA2.20140901.A1.4x5.nc4"
    assert forcing.u_m_s.shape == (1, 47, 46, 72)
    result = run_transport_one_step(state, forcing, grid, dt_s=600.0)
    assert result.state.shape == (1, 47, 46, 72, 1)
    assert np.all(np.isfinite(result.state.data))
    if transport_numba_mode == "numba":
        blocked = state.reblock(1)
        executor = TransportExecutor.create(blocked)
        compiled = run_transport_step_with_executor(
            blocked,
            forcing,
            grid,
            executor,
            dt_s=600.0,
        )
        np.testing.assert_allclose(
            compiled.state.to_canonical(),
            result.state.data,
            rtol=5.0e-14,
            atol=2.0e-19,
        )
