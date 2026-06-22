from __future__ import annotations

import netCDF4
import numpy as np
import pytest

from wombat_transport.gc_harness import (
    PJC_INPUT_VERSION,
    PJC_OUTPUT_VERSION,
    TRANSPORT_INPUT_VERSION,
    TRANSPORT_OUTPUT_VERSION,
    append_transport_step_tracers,
    compare_pjc_output,
    read_transport_step_output,
    run_pjc_harness,
    write_pjc_input,
)
from wombat_transport.transport import dry_pressure_thickness_hpa, horizontal_mass_flux_hpa


def test_write_pjc_input_records_fixture_contract(tmp_path):
    input_path = _write_synthetic_pjc_input(tmp_path / "pjc_input.nc")

    with netCDF4.Dataset(input_path) as dataset:
        assert dataset.harness == PJC_INPUT_VERSION
        assert dataset.dt_s == 600.0
        assert dataset.dimensions["lon"].size == 3
        assert dataset.dimensions["lat"].size == 2
        assert dataset.dimensions["lev"].size == 4
        assert dataset.dimensions["ilev"].size == 5
        assert dataset.variables["u_m_s"].shape == (4, 2, 3)
        assert dataset.variables["area_m2"].shape == (2, 3)


def test_compare_pjc_output_reports_zero_for_matching_wombat_fluxes(tmp_path):
    input_path = _write_synthetic_pjc_input(tmp_path / "pjc_input.nc")
    with netCDF4.Dataset(input_path) as dataset:
        lat = np.asarray(dataset.variables["lat"][:])
        hyai = np.asarray(dataset.variables["hyai"][:])
        hybi = np.asarray(dataset.variables["hybi"][:])
        p1 = np.asarray(dataset.variables["p1_hpa"][:])
        u = np.asarray(dataset.variables["u_m_s"][:])
        v = np.asarray(dataset.variables["v_m_s"][:])
        dt_s = float(dataset.dt_s)
    delp = dry_pressure_thickness_hpa(p1[np.newaxis, :, :] * 100.0, hyai, hybi)
    xmass, ymass = horizontal_mass_flux_hpa(
        delp,
        u[np.newaxis, :, :, :],
        v[np.newaxis, :, :, :],
        lat,
        dt_s=dt_s,
    )
    output_path = tmp_path / "pjc_output.nc"
    with netCDF4.Dataset(output_path, "w") as dataset:
        dataset.createDimension("lev", 4)
        dataset.createDimension("lat", 2)
        dataset.createDimension("lon", 3)
        dataset.harness = PJC_OUTPUT_VERSION
        dataset.createVariable("xmass_hpa", "f8", ("lev", "lat", "lon"))[:] = xmass[0]
        dataset.createVariable("ymass_hpa", "f8", ("lev", "lat", "lon"))[:] = ymass[0]

    comparison = compare_pjc_output(input_path, output_path)

    assert comparison.xmass_max_abs_error_hpa == 0.0
    assert comparison.xmass_mean_abs_error_hpa == 0.0
    assert comparison.ymass_max_abs_error_hpa == 0.0
    assert comparison.ymass_mean_abs_error_hpa == 0.0


def test_append_transport_step_tracers_records_fixture_contract(tmp_path):
    input_path = _write_synthetic_pjc_input(tmp_path / "transport_input.nc")
    tracer_conc = np.arange(2 * 4 * 2 * 3, dtype=np.float64).reshape(2, 4, 2, 3)

    append_transport_step_tracers(input_path, tracer_conc, tracer_names=("A", "B"))

    with netCDF4.Dataset(input_path) as dataset:
        assert dataset.harness == TRANSPORT_INPUT_VERSION
        assert dataset.dimensions["tracer"].size == 2
        assert dataset.dimensions["name_strlen"].size == 1
        assert dataset.variables["tracer_conc"].shape == (2, 4, 2, 3)
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
    lat = np.array([-1.0, 1.0])
    lon = np.array([0.0, 2.5, 5.0])
    area = np.full((2, 3), 1.0e10)
    hyai = np.array([1000.0, 800.0, 500.0, 100.0, 0.01])
    hybi = np.zeros_like(hyai)
    p1 = np.full((2, 3), 1000.0)
    p2 = p1.copy()
    u = np.arange(4 * 2 * 3, dtype=np.float64).reshape(4, 2, 3) * 0.01
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
