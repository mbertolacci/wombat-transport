from __future__ import annotations

import json
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from wombat_transport.gc_harness import (
    PJC_SNAPSHOT_VERSION,
    PJC_INPUT_VERSION,
    PJC_OUTPUT_VERSION,
    SNAPSHOT_INPUT_NAME,
    SNAPSHOT_METADATA_NAME,
    SNAPSHOT_OUTPUT_NAME,
    TRANSPORT_INPUT_VERSION,
    TRANSPORT_OUTPUT_VERSION,
    append_transport_step_tracers,
    compare_pjc_output,
    read_transport_step_output,
    run_pjc_harness,
    write_synthetic_pjc_snapshot_input,
    write_pjc_input,
)
from wombat_transport.transport import pjc_mass_flux_hpa

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pjc_snapshot_v1"


def test_write_pjc_input_records_fixture_contract(tmp_path):
    input_path = _write_synthetic_pjc_input(tmp_path / "pjc_input.nc")

    with netCDF4.Dataset(input_path) as dataset:
        assert dataset.harness == PJC_INPUT_VERSION
        assert dataset.dt_s == 600.0
        assert dataset.dimensions["lon"].size == 3
        assert dataset.dimensions["lat"].size == 7
        assert dataset.dimensions["lev"].size == 4
        assert dataset.dimensions["ilev"].size == 5
        assert dataset.variables["u_m_s"].shape == (4, 7, 3)
        assert dataset.variables["area_m2"].shape == (7, 3)


def test_compare_pjc_output_reports_zero_for_matching_numpy_pjc_fluxes(tmp_path):
    input_path = _write_synthetic_pjc_input(tmp_path / "pjc_input.nc")
    with netCDF4.Dataset(input_path) as dataset:
        lat = np.asarray(dataset.variables["lat"][:])
        hyai = np.asarray(dataset.variables["hyai"][:])
        hybi = np.asarray(dataset.variables["hybi"][:])
        area = np.asarray(dataset.variables["area_m2"][:])
        p1 = np.asarray(dataset.variables["p1_hpa"][:])
        p2 = np.asarray(dataset.variables["p2_hpa"][:])
        u = np.asarray(dataset.variables["u_m_s"][:])
        v = np.asarray(dataset.variables["v_m_s"][:])
        dt_s = float(dataset.dt_s)
    xmass, ymass = pjc_mass_flux_hpa(
        p1_hpa=p1,
        p2_hpa=p2,
        u_m_s=u,
        v_m_s=v,
        area_m2=area,
        hyai_hpa=hyai,
        hybi=hybi,
        lat_deg=lat,
        dt_s=dt_s,
    )
    output_path = tmp_path / "pjc_output.nc"
    with netCDF4.Dataset(output_path, "w") as dataset:
        dataset.createDimension("lev", 4)
        dataset.createDimension("lat", xmass.shape[1])
        dataset.createDimension("lon", 3)
        dataset.harness = PJC_OUTPUT_VERSION
        dataset.createVariable("xmass_hpa", "f8", ("lev", "lat", "lon"))[:] = xmass
        dataset.createVariable("ymass_hpa", "f8", ("lev", "lat", "lon"))[:] = ymass

    comparison = compare_pjc_output(input_path, output_path)

    assert comparison.xmass_max_abs_error_hpa == 0.0
    assert comparison.xmass_mean_abs_error_hpa == 0.0
    assert comparison.ymass_max_abs_error_hpa == 0.0
    assert comparison.ymass_mean_abs_error_hpa == 0.0


def test_pjc_mass_flux_preserves_contract_on_geos_like_grid():
    nlon = 8
    nlat = 7
    nlev = 4
    lat = np.linspace(-90.0, 90.0, nlat)
    area = np.broadcast_to(np.cos(np.deg2rad(np.clip(lat, -89.0, 89.0)))[:, np.newaxis], (nlat, nlon)).copy()
    area *= 1.0e10
    hyai = np.array([0.0, 10.0, 40.0, 100.0, 0.01])
    hybi = np.array([1.0, 0.9, 0.5, 0.1, 0.0])
    p1 = np.full((nlat, nlon), 950.0)
    p2 = p1.copy()
    u = np.ones((nlev, nlat, nlon))
    v = np.zeros_like(u)

    xmass, ymass = pjc_mass_flux_hpa(
        p1_hpa=p1,
        p2_hpa=p2,
        u_m_s=u,
        v_m_s=v,
        area_m2=area,
        hyai_hpa=hyai,
        hybi=hybi,
        lat_deg=lat,
        dt_s=600.0,
    )

    assert xmass.shape == (nlev, nlat, nlon)
    assert ymass.shape == (nlev, nlat, nlon)
    assert np.all(np.isfinite(xmass))
    assert np.all(np.isfinite(ymass))
    np.testing.assert_allclose(ymass, 0.0, atol=1.0e-12)


def test_write_synthetic_pjc_snapshot_input_records_compact_47_level_contract(tmp_path):
    input_path = write_synthetic_pjc_snapshot_input(tmp_path / SNAPSHOT_INPUT_NAME)

    with netCDF4.Dataset(input_path) as dataset:
        assert dataset.harness == PJC_INPUT_VERSION
        assert dataset.dt_s == 600.0
        assert dataset.dimensions["lon"].size == 8
        assert dataset.dimensions["lat"].size == 7
        assert dataset.dimensions["lev"].size == 47
        assert dataset.dimensions["ilev"].size == 48
        assert dataset.variables["u_m_s"].shape == (47, 7, 8)
        assert dataset.variables["area_m2"].shape == (7, 8)


def test_pjc_mass_flux_matches_tracked_geos_chem_snapshot():
    with (FIXTURE_DIR / SNAPSHOT_METADATA_NAME).open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["snapshot"] == PJC_SNAPSHOT_VERSION
    assert metadata["shape"] == {"lev": 47, "lat": 7, "lon": 8}

    comparison = compare_pjc_output(FIXTURE_DIR / SNAPSHOT_INPUT_NAME, FIXTURE_DIR / SNAPSHOT_OUTPUT_NAME)

    assert comparison.xmass_max_abs_error_hpa < 1.0e-12
    assert comparison.xmass_mean_abs_error_hpa < 1.0e-13
    assert comparison.ymass_max_abs_error_hpa < 1.0e-12
    assert comparison.ymass_mean_abs_error_hpa < 1.0e-13


def test_append_transport_step_tracers_records_fixture_contract(tmp_path):
    input_path = _write_synthetic_pjc_input(tmp_path / "transport_input.nc")
    tracer_conc = np.arange(2 * 4 * 7 * 3, dtype=np.float64).reshape(2, 4, 7, 3)

    append_transport_step_tracers(input_path, tracer_conc, tracer_names=("A", "B"))

    with netCDF4.Dataset(input_path) as dataset:
        assert dataset.harness == TRANSPORT_INPUT_VERSION
        assert dataset.dimensions["tracer"].size == 2
        assert dataset.dimensions["name_strlen"].size == 1
        assert dataset.variables["tracer_conc"].shape == (2, 4, 7, 3)
        assert np.array_equal(np.asarray(dataset.variables["tracer_conc"][:]), tracer_conc)


def test_read_transport_step_output_records_oracle_fields(tmp_path):
    output_path = tmp_path / "transport_output.nc"
    with netCDF4.Dataset(output_path, "w") as dataset:
        dataset.createDimension("tracer", 1)
        dataset.createDimension("lev", 4)
        dataset.createDimension("lat", 2)
        dataset.createDimension("lon", 3)
        dataset.harness = TRANSPORT_OUTPUT_VERSION
        dataset.createVariable("tracer_conc_after", "f8", ("tracer", "lev", "lat", "lon"))[:] = 1.0
        dataset.createVariable("xmass_hpa", "f8", ("lev", "lat", "lon"))[:] = 2.0
        dataset.createVariable("ymass_hpa", "f8", ("lev", "lat", "lon"))[:] = 3.0
        dataset.createVariable("surface_pressure_hpa", "f8", ("lat", "lon"))[:] = 4.0

    output = read_transport_step_output(output_path)

    assert output.tracer_conc_after.shape == (1, 4, 2, 3)
    assert output.xmass_hpa.shape == (4, 2, 3)
    assert output.ymass_hpa.shape == (4, 2, 3)
    assert output.surface_pressure_hpa.shape == (2, 3)


def test_run_pjc_harness_missing_executable_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="GEOS-Chem harness executable not found"):
        run_pjc_harness(tmp_path / "missing_harness", tmp_path / "input.nc", tmp_path / "output.nc")


def test_write_pjc_input_rejects_bad_shapes(tmp_path):
    with pytest.raises(ValueError, match="u_m_s and v_m_s"):
        write_pjc_input(
            tmp_path / "bad.nc",
            lat_deg=np.array([-1.0, 1.0]),
            lon_deg=np.array([0.0, 2.5, 5.0]),
            area_m2=np.ones((2, 3)),
            hyai_hpa=np.arange(5.0),
            hybi=np.arange(5.0),
            p1_hpa=np.ones((2, 3)),
            p2_hpa=np.ones((2, 3)),
            u_m_s=np.ones((3, 2, 3)),
            v_m_s=np.ones((4, 2, 3)),
            dt_s=600.0,
        )


def test_append_transport_step_tracers_rejects_bad_shape(tmp_path):
    input_path = _write_synthetic_pjc_input(tmp_path / "transport_input.nc")

    with pytest.raises(ValueError, match="tracer_conc grid"):
        append_transport_step_tracers(input_path, np.ones((1, 3, 2, 3)))


def _write_synthetic_pjc_input(path):
    lat = np.array([-89.0, -87.0, -45.0, 0.0, 45.0, 87.0, 89.0])
    lon = np.array([0.0, 2.5, 5.0])
    area = np.broadcast_to(np.cos(np.deg2rad(lat))[:, np.newaxis], (lat.size, lon.size)).copy()
    area *= 1.0e10
    hyai = np.array([1000.0, 800.0, 500.0, 100.0, 0.01])
    hybi = np.zeros_like(hyai)
    p1 = np.full((lat.size, lon.size), 1000.0)
    p2 = p1.copy()
    u = np.arange(4 * lat.size * lon.size, dtype=np.float64).reshape(4, lat.size, lon.size) * 0.01
    v = u * -0.25
    return write_pjc_input(
        path,
        lat_deg=lat,
        lon_deg=lon,
        area_m2=area,
        hyai_hpa=hyai,
        hybi=hybi,
        p1_hpa=p1,
        p2_hpa=p2,
        u_m_s=u,
        v_m_s=v,
        dt_s=600.0,
    )
