from __future__ import annotations

import json
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from wombat_transport.gc_harness import (
    BASE_INITIAL_TPCORE_FIXTURE_ID,
    FULLGRID_SYNTHETIC_LOW_COURANT_TPCORE_FIXTURE_ID,
    PJC_SNAPSHOT_VERSION,
    PJC_INPUT_VERSION,
    PJC_OUTPUT_VERSION,
    SNAPSHOT_INPUT_NAME,
    SNAPSHOT_METADATA_NAME,
    SNAPSHOT_OUTPUT_NAME,
    TPCORE_SNAPSHOT_INPUT_NAME,
    TPCORE_SNAPSHOT_OUTPUT_NAME,
    TPCORE_SNAPSHOT_VERSION,
    TRANSPORT_INPUT_VERSION,
    TRANSPORT_OUTPUT_VERSION,
    append_transport_step_tracers,
    attribute_python_tpcore_error,
    check_large_oracle_fixture,
    compare_tpcore_trace_files,
    compare_pjc_output,
    compare_large_oracle_fixture,
    compare_python_tpcore_output,
    compare_transport_step_output,
    format_large_oracle_fixture_check,
    large_oracle_fixture_paths,
    read_transport_step_output,
    write_python_tpcore_trace,
    run_pjc_harness,
    sha256_file,
    write_synthetic_tpcore_branch_input,
    write_synthetic_pjc_snapshot_input,
    write_synthetic_tpcore_snapshot_input,
    write_pjc_input,
)
from wombat_transport.transport import pjc_mass_flux_hpa
from wombat_transport.transport.tpcore import (
    TpcoreSetup,
    analyze_tpcore_branches,
    run_tpcore_one_step,
    setup_tpcore_terms,
    trace_tpcore_one_step,
    validate_tpcore_branch_support,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pjc_snapshot_v1"
TPCORE_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tpcore_snapshot_v1"
TPCORE_FXPPM_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tpcore_x_fxppm_low_courant_v1"
TPCORE_LARGE_CX_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tpcore_x_large_courant_polar_v1"


def test_large_oracle_fixture_check_verifies_cached_payloads(tmp_path):
    cache_dir = tmp_path / "oracle_data"
    manifest_dir = cache_dir / "manifests"
    payload_dir = cache_dir / BASE_INITIAL_TPCORE_FIXTURE_ID
    manifest_dir.mkdir(parents=True)
    payload_dir.mkdir()
    input_path = payload_dir / "transport_step_input.nc"
    output_path = payload_dir / "transport_step_output.nc"
    input_path.write_bytes(b"input")
    output_path.write_bytes(b"output")
    _write_large_oracle_definition(
        manifest_dir / f"{BASE_INITIAL_TPCORE_FIXTURE_ID}.json",
        input_sha=sha256_file(input_path),
        output_sha=sha256_file(output_path),
        input_size=input_path.stat().st_size,
        output_size=output_path.stat().st_size,
    )

    check = check_large_oracle_fixture(BASE_INITIAL_TPCORE_FIXTURE_ID, cache_dir=cache_dir, manifest_dir=manifest_dir)

    assert check.is_available
    assert check.missing_files == ()
    assert check.checksum_failures == ()
    assert check.unchecked_files == ()


def test_large_oracle_fixture_check_reports_missing_payloads(tmp_path):
    cache_dir = tmp_path / "oracle_data"
    manifest_dir = cache_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    _write_large_oracle_definition(manifest_dir / f"{BASE_INITIAL_TPCORE_FIXTURE_ID}.json")

    check = check_large_oracle_fixture(BASE_INITIAL_TPCORE_FIXTURE_ID, cache_dir=cache_dir, manifest_dir=manifest_dir)

    assert not check.is_available
    assert check.missing_files == ("transport_step_input.nc", "transport_step_output.nc")
    assert "available,False" in format_large_oracle_fixture_check(check)


def test_large_base_oracle_fixture_if_cached_reports_pjc_and_tpcore_branches():
    check = check_large_oracle_fixture(BASE_INITIAL_TPCORE_FIXTURE_ID)
    if not check.is_available:
        pytest.skip(format_large_oracle_fixture_check(check))

    paths = large_oracle_fixture_paths(BASE_INITIAL_TPCORE_FIXTURE_ID)
    assert paths.input_path.exists()
    assert paths.output_path.exists()

    report = compare_large_oracle_fixture(BASE_INITIAL_TPCORE_FIXTURE_ID)

    assert "xmass_max_abs_error_hpa" in report
    assert "tpcore_shape,(47, 91, 144)" in report
    assert "tpcore_supported," in report


def test_fullgrid_synthetic_oracle_fixture_if_cached_matches_python_tpcore():
    check = check_large_oracle_fixture(FULLGRID_SYNTHETIC_LOW_COURANT_TPCORE_FIXTURE_ID)
    if not check.is_available:
        pytest.skip(format_large_oracle_fixture_check(check))

    paths = large_oracle_fixture_paths(FULLGRID_SYNTHETIC_LOW_COURANT_TPCORE_FIXTURE_ID)
    with netCDF4.Dataset(paths.input_path) as dataset:
        setup = setup_tpcore_terms(
            p1_hpa=np.asarray(dataset.variables["p1_hpa"][:], dtype=np.float64),
            p2_hpa=np.asarray(dataset.variables["p2_hpa"][:], dtype=np.float64),
            u_m_s=np.asarray(dataset.variables["u_m_s"][:], dtype=np.float64),
            v_m_s=np.asarray(dataset.variables["v_m_s"][:], dtype=np.float64),
            area_m2=np.asarray(dataset.variables["area_m2"][:], dtype=np.float64),
            hyai_hpa=np.asarray(dataset.variables["hyai"][:], dtype=np.float64),
            hybi=np.asarray(dataset.variables["hybi"][:], dtype=np.float64),
            lat_deg=np.asarray(dataset.variables["lat"][:], dtype=np.float64),
            dt_s=float(dataset.dt_s),
        )
    branch_report = analyze_tpcore_branches(setup)

    assert branch_report.is_supported
    assert branch_report.shape == (47, 91, 144)
    assert branch_report.needs_fxppm
    assert not branch_report.has_large_cx
    assert not branch_report.has_large_cy

    comparison = compare_python_tpcore_output(paths.input_path, paths.output_path)

    assert comparison.tracer_max_abs_error < 1.0e-12
    assert comparison.max_abs_cx < 1.0
    assert comparison.max_abs_cy < 1.0


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


def test_write_synthetic_tpcore_snapshot_input_records_compact_47_level_contract(tmp_path):
    input_path = write_synthetic_tpcore_snapshot_input(tmp_path / TPCORE_SNAPSHOT_INPUT_NAME)

    with netCDF4.Dataset(input_path) as dataset:
        assert dataset.harness == TRANSPORT_INPUT_VERSION
        assert dataset.dt_s == 600.0
        assert dataset.dimensions["tracer"].size == 2
        assert dataset.dimensions["lon"].size == 8
        assert dataset.dimensions["lat"].size == 7
        assert dataset.dimensions["lev"].size == 47
        assert dataset.dimensions["ilev"].size == 48
        assert dataset.variables["tracer_conc"].shape == (2, 47, 7, 8)
        tracer = np.asarray(dataset.variables["tracer_conc"][:])
        assert np.all(tracer > 0.0)
        assert float(np.max(tracer) - np.min(tracer)) > 0.0


def test_write_synthetic_tpcore_branch_input_records_fxppm_scenario(tmp_path):
    input_path = write_synthetic_tpcore_branch_input(tmp_path / TPCORE_SNAPSHOT_INPUT_NAME, scenario="x_fxppm_low_courant")

    with netCDF4.Dataset(input_path) as dataset:
        assert dataset.harness == TRANSPORT_INPUT_VERSION
        assert dataset.dimensions["lon"].size == 12
        assert dataset.dimensions["lat"].size == 11
        assert dataset.dimensions["lev"].size == 47
        setup = setup_tpcore_terms(
            p1_hpa=np.asarray(dataset.variables["p1_hpa"][:], dtype=np.float64),
            p2_hpa=np.asarray(dataset.variables["p2_hpa"][:], dtype=np.float64),
            u_m_s=np.asarray(dataset.variables["u_m_s"][:], dtype=np.float64),
            v_m_s=np.asarray(dataset.variables["v_m_s"][:], dtype=np.float64),
            area_m2=np.asarray(dataset.variables["area_m2"][:], dtype=np.float64),
            hyai_hpa=np.asarray(dataset.variables["hyai"][:], dtype=np.float64),
            hybi=np.asarray(dataset.variables["hybi"][:], dtype=np.float64),
            lat_deg=np.asarray(dataset.variables["lat"][:], dtype=np.float64),
            dt_s=float(dataset.dt_s),
        )

    report = analyze_tpcore_branches(setup)

    assert report.needs_fxppm
    assert not report.has_large_cx
    assert report.is_supported


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


def test_tpcore_step_snapshot_records_geos_chem_oracle_boundary():
    with (TPCORE_FIXTURE_DIR / SNAPSHOT_METADATA_NAME).open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["snapshot"] == TPCORE_SNAPSHOT_VERSION
    assert metadata["shape"] == {"tracer": 2, "lev": 47, "lat": 7, "lon": 8}

    comparison = compare_transport_step_output(
        TPCORE_FIXTURE_DIR / TPCORE_SNAPSHOT_INPUT_NAME,
        TPCORE_FIXTURE_DIR / TPCORE_SNAPSHOT_OUTPUT_NAME,
    )

    assert comparison.xmass_max_abs_error_hpa < 1.0e-12
    assert comparison.xmass_mean_abs_error_hpa < 1.0e-13
    assert comparison.ymass_max_abs_error_hpa < 1.0e-12
    assert comparison.ymass_mean_abs_error_hpa < 1.0e-13
    assert comparison.tracer_max_abs_change > 0.0
    assert comparison.negative_count_after == 0
    assert 0.0 < comparison.surface_pressure_min_hpa < comparison.surface_pressure_max_hpa


def test_python_tpcore_setup_matches_oracle_pressure_on_low_courant_fixture():
    with netCDF4.Dataset(TPCORE_FIXTURE_DIR / TPCORE_SNAPSHOT_INPUT_NAME) as dataset:
        setup = setup_tpcore_terms(
            p1_hpa=np.asarray(dataset.variables["p1_hpa"][:], dtype=np.float64),
            p2_hpa=np.asarray(dataset.variables["p2_hpa"][:], dtype=np.float64),
            u_m_s=np.asarray(dataset.variables["u_m_s"][:], dtype=np.float64),
            v_m_s=np.asarray(dataset.variables["v_m_s"][:], dtype=np.float64),
            area_m2=np.asarray(dataset.variables["area_m2"][:], dtype=np.float64),
            hyai_hpa=np.asarray(dataset.variables["hyai"][:], dtype=np.float64),
            hybi=np.asarray(dataset.variables["hybi"][:], dtype=np.float64),
            lat_deg=np.asarray(dataset.variables["lat"][:], dtype=np.float64),
            dt_s=float(dataset.dt_s),
        )
    oracle = read_transport_step_output(TPCORE_FIXTURE_DIR / TPCORE_SNAPSHOT_OUTPUT_NAME)

    np.testing.assert_array_equal(setup.surface_pressure_hpa, oracle.surface_pressure_hpa)
    assert float(np.max(np.abs(setup.cx))) < 1.0
    assert float(np.max(np.abs(setup.cy))) < 1.0


def test_tpcore_branch_report_accepts_low_courant_oracle_path():
    with netCDF4.Dataset(TPCORE_FIXTURE_DIR / TPCORE_SNAPSHOT_INPUT_NAME) as dataset:
        setup = setup_tpcore_terms(
            p1_hpa=np.asarray(dataset.variables["p1_hpa"][:], dtype=np.float64),
            p2_hpa=np.asarray(dataset.variables["p2_hpa"][:], dtype=np.float64),
            u_m_s=np.asarray(dataset.variables["u_m_s"][:], dtype=np.float64),
            v_m_s=np.asarray(dataset.variables["v_m_s"][:], dtype=np.float64),
            area_m2=np.asarray(dataset.variables["area_m2"][:], dtype=np.float64),
            hyai_hpa=np.asarray(dataset.variables["hyai"][:], dtype=np.float64),
            hybi=np.asarray(dataset.variables["hybi"][:], dtype=np.float64),
            lat_deg=np.asarray(dataset.variables["lat"][:], dtype=np.float64),
            dt_s=float(dataset.dt_s),
        )

    report = analyze_tpcore_branches(setup)

    assert report.is_supported
    assert report.shape == (47, 7, 8)
    assert not report.has_large_cx
    assert not report.has_large_cy
    assert not report.needs_fxppm


def test_tpcore_branch_report_identifies_fxppm_path():
    setup = _zero_tpcore_setup(nlev=4, nlat=11, nlon=8)

    report = analyze_tpcore_branches(setup)

    assert report.is_supported
    assert report.needs_fxppm


def test_tpcore_branch_report_identifies_large_e_w_path():
    setup = _zero_tpcore_setup(nlev=4, nlat=7, nlon=8)
    setup.cx[0, 2, 0] = 1.1

    report = analyze_tpcore_branches(setup)

    assert report.is_supported
    assert report.has_large_cx
    assert report.x_ffsl_active


def test_tpcore_branch_preflight_rejects_large_n_s_path():
    setup = _zero_tpcore_setup(nlev=4, nlat=7, nlon=8)
    setup.cy[0, 2, 0] = 1.1

    report = analyze_tpcore_branches(setup)

    assert not report.is_supported
    assert report.has_large_cy
    with pytest.raises(NotImplementedError, match="large-Courant N-S"):
        validate_tpcore_branch_support(setup)


def test_python_tpcore_matches_low_courant_oracle_tracer_step():
    comparison = compare_python_tpcore_output(
        TPCORE_FIXTURE_DIR / TPCORE_SNAPSHOT_INPUT_NAME,
        TPCORE_FIXTURE_DIR / TPCORE_SNAPSHOT_OUTPUT_NAME,
    )

    assert comparison.xmass_max_abs_error_hpa < 1.0e-12
    assert comparison.ymass_max_abs_error_hpa < 1.0e-12
    assert comparison.surface_pressure_max_abs_error_hpa == 0.0
    assert comparison.tracer_max_abs_error < 1.0e-11
    assert comparison.tracer_mean_abs_error < 1.0e-12
    assert comparison.max_abs_cx < 1.0
    assert comparison.max_abs_cy < 1.0


def test_python_tpcore_matches_fxppm_low_courant_branch_fixture():
    with (TPCORE_FXPPM_FIXTURE_DIR / SNAPSHOT_METADATA_NAME).open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["scenario"] == "x_fxppm_low_courant"
    assert metadata["branch_report"]["needs_fxppm"]
    assert not metadata["branch_report"]["has_large_cx"]

    comparison = compare_python_tpcore_output(
        TPCORE_FXPPM_FIXTURE_DIR / TPCORE_SNAPSHOT_INPUT_NAME,
        TPCORE_FXPPM_FIXTURE_DIR / TPCORE_SNAPSHOT_OUTPUT_NAME,
    )

    assert comparison.xmass_max_abs_error_hpa < 1.0e-12
    assert comparison.ymass_max_abs_error_hpa < 2.0e-14
    assert comparison.surface_pressure_max_abs_error_hpa < 5.0e-13
    assert comparison.tracer_max_abs_error < 1.0e-11
    assert comparison.max_abs_cx < 1.0
    assert comparison.max_abs_cy < 1.0


def test_python_tpcore_matches_large_courant_branch_fixture():
    with (TPCORE_LARGE_CX_FIXTURE_DIR / SNAPSHOT_METADATA_NAME).open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["scenario"] == "x_large_courant_polar"
    assert metadata["branch_report"]["has_large_cx"]
    assert not metadata["branch_report"]["needs_fxppm"]

    comparison = compare_python_tpcore_output(
        TPCORE_LARGE_CX_FIXTURE_DIR / TPCORE_SNAPSHOT_INPUT_NAME,
        TPCORE_LARGE_CX_FIXTURE_DIR / TPCORE_SNAPSHOT_OUTPUT_NAME,
    )

    assert comparison.xmass_max_abs_error_hpa < 1.0e-12
    assert comparison.ymass_max_abs_error_hpa < 1.0e-14
    assert comparison.surface_pressure_max_abs_error_hpa < 5.0e-13
    assert comparison.max_abs_cx > 1.0
    assert comparison.max_abs_cy < 1.0
    assert comparison.tracer_max_abs_error < 1.0e-12


def test_python_tpcore_preserves_constant_tracer_on_low_courant_fixture():
    with netCDF4.Dataset(TPCORE_FIXTURE_DIR / TPCORE_SNAPSHOT_INPUT_NAME) as dataset:
        tracer = np.asarray(dataset.variables["tracer_conc"][:], dtype=np.float64)
        tracer[:] = 4.0e-4
        state = run_tpcore_one_step(
            tracer_conc=tracer,
            p1_hpa=np.asarray(dataset.variables["p1_hpa"][:], dtype=np.float64),
            p2_hpa=np.asarray(dataset.variables["p2_hpa"][:], dtype=np.float64),
            u_m_s=np.asarray(dataset.variables["u_m_s"][:], dtype=np.float64),
            v_m_s=np.asarray(dataset.variables["v_m_s"][:], dtype=np.float64),
            area_m2=np.asarray(dataset.variables["area_m2"][:], dtype=np.float64),
            hyai_hpa=np.asarray(dataset.variables["hyai"][:], dtype=np.float64),
            hybi=np.asarray(dataset.variables["hybi"][:], dtype=np.float64),
            lat_deg=np.asarray(dataset.variables["lat"][:], dtype=np.float64),
            dt_s=float(dataset.dt_s),
        )

    np.testing.assert_allclose(state.tracer_conc_after, 4.0e-4, atol=1.0e-18, rtol=0.0)


def test_python_tpcore_trace_preserves_final_output_on_low_courant_fixture():
    with netCDF4.Dataset(TPCORE_FIXTURE_DIR / TPCORE_SNAPSHOT_INPUT_NAME) as dataset:
        kwargs = {
            "tracer_conc": np.asarray(dataset.variables["tracer_conc"][:], dtype=np.float64),
            "p1_hpa": np.asarray(dataset.variables["p1_hpa"][:], dtype=np.float64),
            "p2_hpa": np.asarray(dataset.variables["p2_hpa"][:], dtype=np.float64),
            "u_m_s": np.asarray(dataset.variables["u_m_s"][:], dtype=np.float64),
            "v_m_s": np.asarray(dataset.variables["v_m_s"][:], dtype=np.float64),
            "area_m2": np.asarray(dataset.variables["area_m2"][:], dtype=np.float64),
            "hyai_hpa": np.asarray(dataset.variables["hyai"][:], dtype=np.float64),
            "hybi": np.asarray(dataset.variables["hybi"][:], dtype=np.float64),
            "lat_deg": np.asarray(dataset.variables["lat"][:], dtype=np.float64),
            "dt_s": float(dataset.dt_s),
        }

    normal = run_tpcore_one_step(**kwargs)
    traced, trace = trace_tpcore_one_step(**kwargs)

    np.testing.assert_array_equal(traced.tracer_conc_after, normal.tracer_conc_after)
    np.testing.assert_array_equal(trace.tracer_conc_after, normal.tracer_conc_after)
    assert trace.dq_after_xtp.shape == normal.tracer_conc_after.shape
    assert trace.dq_after_fzppm.shape == normal.tracer_conc_after.shape


def test_write_python_tpcore_trace_records_stage_contract(tmp_path):
    input_path = TPCORE_FIXTURE_DIR / TPCORE_SNAPSHOT_INPUT_NAME
    trace_path = write_python_tpcore_trace(input_path, tmp_path / "python_tpcore_trace.nc")

    with netCDF4.Dataset(trace_path) as dataset:
        assert dataset.harness == "tpcore-trace-v1"
        assert dataset.variables["q_after_pole_average"].shape == (2, 47, 7, 8)
        assert dataset.variables["dq_after_ytp_hpa"].shape == (2, 47, 7, 8)
        assert dataset.variables["cx"].shape == (47, 7, 8)
        assert dataset.variables["vertical_mass_flux_hpa"].shape == (47, 7, 8)

    report = compare_tpcore_trace_files(trace_path, trace_path)

    assert "tracer_conc_after,0.00000000e+00" in report


def test_attribute_python_tpcore_error_reports_error_bins(tmp_path):
    input_path = TPCORE_FIXTURE_DIR / TPCORE_SNAPSHOT_INPUT_NAME
    with netCDF4.Dataset(input_path) as dataset:
        state = run_tpcore_one_step(
            tracer_conc=np.asarray(dataset.variables["tracer_conc"][:], dtype=np.float64),
            p1_hpa=np.asarray(dataset.variables["p1_hpa"][:], dtype=np.float64),
            p2_hpa=np.asarray(dataset.variables["p2_hpa"][:], dtype=np.float64),
            u_m_s=np.asarray(dataset.variables["u_m_s"][:], dtype=np.float64),
            v_m_s=np.asarray(dataset.variables["v_m_s"][:], dtype=np.float64),
            area_m2=np.asarray(dataset.variables["area_m2"][:], dtype=np.float64),
            hyai_hpa=np.asarray(dataset.variables["hyai"][:], dtype=np.float64),
            hybi=np.asarray(dataset.variables["hybi"][:], dtype=np.float64),
            lat_deg=np.asarray(dataset.variables["lat"][:], dtype=np.float64),
            dt_s=float(dataset.dt_s),
        )
    output_path = tmp_path / "transport_output.nc"
    perturbed = state.tracer_conc_after.copy()
    perturbed[0, 3, 2, 1] += 1.0e-9
    with netCDF4.Dataset(output_path, "w") as dataset:
        dataset.createDimension("tracer", perturbed.shape[0])
        dataset.createDimension("lev", perturbed.shape[1])
        dataset.createDimension("lat", perturbed.shape[2])
        dataset.createDimension("lon", perturbed.shape[3])
        dataset.harness = TRANSPORT_OUTPUT_VERSION
        dataset.createVariable("tracer_conc_after", "f8", ("tracer", "lev", "lat", "lon"))[:] = perturbed
        dataset.createVariable("xmass_hpa", "f8", ("lev", "lat", "lon"))[:] = state.xmass_hpa
        dataset.createVariable("ymass_hpa", "f8", ("lev", "lat", "lon"))[:] = state.ymass_hpa
        dataset.createVariable("surface_pressure_hpa", "f8", ("lat", "lon"))[:] = state.surface_pressure_hpa

    report = attribute_python_tpcore_error(input_path, output_path)

    assert "section,key,max_abs,mean_abs,count,extra" in report
    assert "top_cell,max,1.00000000e-09" in report
    assert "abs_cx" in report


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


def test_compare_transport_step_output_reports_pjc_flux_errors_and_tracer_summary(tmp_path):
    input_path = write_synthetic_tpcore_snapshot_input(tmp_path / "transport_input.nc")
    with netCDF4.Dataset(input_path) as input_dataset:
        lat = np.asarray(input_dataset.variables["lat"][:])
        hyai = np.asarray(input_dataset.variables["hyai"][:])
        hybi = np.asarray(input_dataset.variables["hybi"][:])
        area = np.asarray(input_dataset.variables["area_m2"][:])
        p1 = np.asarray(input_dataset.variables["p1_hpa"][:])
        p2 = np.asarray(input_dataset.variables["p2_hpa"][:])
        u = np.asarray(input_dataset.variables["u_m_s"][:])
        v = np.asarray(input_dataset.variables["v_m_s"][:])
        tracer = np.asarray(input_dataset.variables["tracer_conc"][:])
        dt_s = float(input_dataset.dt_s)
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
    output_path = tmp_path / "transport_output.nc"
    with netCDF4.Dataset(output_path, "w") as dataset:
        dataset.createDimension("tracer", tracer.shape[0])
        dataset.createDimension("lev", tracer.shape[1])
        dataset.createDimension("lat", tracer.shape[2])
        dataset.createDimension("lon", tracer.shape[3])
        dataset.harness = TRANSPORT_OUTPUT_VERSION
        dataset.createVariable("tracer_conc_after", "f8", ("tracer", "lev", "lat", "lon"))[:] = tracer + 1.0e-12
        dataset.createVariable("xmass_hpa", "f8", ("lev", "lat", "lon"))[:] = xmass
        dataset.createVariable("ymass_hpa", "f8", ("lev", "lat", "lon"))[:] = ymass
        dataset.createVariable("surface_pressure_hpa", "f8", ("lat", "lon"))[:] = 1000.0

    comparison = compare_transport_step_output(input_path, output_path)

    assert comparison.xmass_max_abs_error_hpa == 0.0
    assert comparison.ymass_max_abs_error_hpa == 0.0
    assert comparison.tracer_max_abs_change == pytest.approx(1.0e-12)
    assert comparison.negative_count_after == 0
    assert comparison.surface_pressure_min_hpa == 1000.0
    assert comparison.surface_pressure_max_hpa == 1000.0


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


def _zero_tpcore_setup(*, nlev: int, nlat: int, nlon: int) -> TpcoreSetup:
    field = np.zeros((nlev, nlat, nlon), dtype=np.float64)
    return TpcoreSetup(
        xmass_hpa=field.copy(),
        ymass_hpa=field.copy(),
        surface_pressure_hpa=np.zeros((nlat, nlon), dtype=np.float64),
        delp1_hpa=np.ones_like(field),
        delpm_hpa=np.ones_like(field),
        delp2_hpa=np.ones_like(field),
        pu_hpa=np.ones_like(field),
        vertical_mass_flux_hpa=field.copy(),
        cx=field.copy(),
        cy=field.copy(),
        geofac=np.ones(nlat, dtype=np.float64),
        geofac_pc=1.0,
    )


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


def _write_large_oracle_definition(
    path: Path,
    *,
    input_sha: str | None = None,
    output_sha: str | None = None,
    input_size: int | None = None,
    output_size: int | None = None,
) -> None:
    payload = {
        "fixture_id": BASE_INITIAL_TPCORE_FIXTURE_ID,
        "files": [
            {
                "name": "transport_step_input.nc",
                "sha256": input_sha,
                "size_bytes": input_size,
                "url": None,
            },
            {
                "name": "transport_step_output.nc",
                "sha256": output_sha,
                "size_bytes": output_size,
                "url": None,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
