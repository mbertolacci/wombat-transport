from __future__ import annotations

import json
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from wombat_transport.gc_harness import (
    BASE_INITIAL_CONVECTION_FULLGRID_FIXTURE_ID,
    BASE_INITIAL_TPCORE_FIXTURE_ID,
    BASE_INITIAL_TRANSPORT_CHAIN_FIXTURE_ID,
    BASE_INITIAL_VDIFF_AFTER_TPCORE_FIXTURE_ID,
    FULLGRID_SYNTHETIC_LOW_COURANT_TPCORE_FIXTURE_ID,
    RESIDUAL_INITIAL_TPCORE_FIXTURE_ID,
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
    CONVECTION_INPUT_VERSION,
    CONVECTION_OUTPUT_VERSION,
    DRY_PRESSURE_INPUT_VERSION,
    DRY_PRESSURE_OUTPUT_VERSION,
    DRY_PRESSURE_SNAPSHOT_INPUT_NAME,
    DRY_PRESSURE_SNAPSHOT_OUTPUT_NAME,
    DRY_PRESSURE_SNAPSHOT_VERSION,
    MET_AIRQNT_OUTPUT_VERSION,
    MET_AIRQNT_SNAPSHOT_INPUT_NAME,
    MET_AIRQNT_SNAPSHOT_OUTPUT_NAME,
    MET_AIRQNT_SNAPSHOT_VERSION,
    HISTORY_HARNESS_OUTPUT_NAME,
    VDIFF_INPUT_VERSION,
    VDIFF_OUTPUT_VERSION,
    append_transport_step_tracers,
    attribute_python_tpcore_error,
    check_large_oracle_fixture,
    compare_history_harness_output,
    compare_history_harness_to_wombat,
    compare_dry_pressure_output,
    compare_met_airqnt_output,
    compare_tpcore_trace_files,
    compare_pjc_output,
    compare_large_oracle_fixture,
    compare_python_tpcore_output,
    compare_transport_step_output,
    compare_convection_output,
    compare_transport_chain_handoffs,
    compare_vdiff_output,
    format_convection_comparison,
    format_dry_pressure_comparison,
    format_met_airqnt_comparison,
    format_history_harness_comparison,
    format_history_harness_wombat_comparison,
    format_large_oracle_fixture_check,
    format_vdiff_comparison,
    read_convection_output,
    read_dry_pressure_output,
    read_met_airqnt_output,
    large_oracle_fixture_paths,
    history_harness_scenario_config,
    read_vdiff_output,
    read_transport_step_output,
    write_python_convection_output,
    write_python_dry_pressure_output,
    write_python_met_airqnt_output,
    write_python_tpcore_trace,
    write_python_vdiff_output,
    run_history_harness,
    run_operator_harness,
    run_pjc_harness,
    write_history_harness_run_directory,
    write_synthetic_convection_input,
    write_synthetic_dry_pressure_input,
    write_real_convection_input_from_config,
    sha256_file,
    write_synthetic_tpcore_branch_input,
    write_synthetic_vdiff_input,
    write_synthetic_pjc_snapshot_input,
    write_synthetic_tpcore_snapshot_input,
    write_pjc_input,
    _tpcore_pressure_branch_gap,
)
from wombat_transport.transport import convection as convection_mod
from wombat_transport.transport.convection import G0_100, run_cloud_convection_one_step
from wombat_transport.transport import pjc_mass_flux_hpa
from wombat_transport.transport.tpcore import (
    TpcoreSetup,
    _calc_advec_cross_terms,
    analyze_tpcore_branches,
    run_tpcore_one_step,
    setup_tpcore_terms,
    trace_tpcore_one_step,
    validate_tpcore_branch_support,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pjc_snapshot_v1"
TPCORE_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tpcore_snapshot_v2"
TPCORE_FXPPM_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tpcore_x_fxppm_low_courant_v2"
TPCORE_LARGE_CX_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tpcore_x_large_courant_polar_v2"
VDIFF_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "vdiff_snapshot_v2"
VDIFF_NONZERO_FLUX_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "vdiff_nonzero_surface_flux_v2"
VDIFF_NEGATIVE_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "vdiff_negative_clipping_v2"
CONVECTION_REAL_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "convection_real_sampled_v2"
DRY_PRESSURE_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "dry_pressure_synthetic_v1"
MET_AIRQNT_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "met_airqnt_synthetic_v1"
HISTORY_DEFAULT_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "history_default_v1"
HISTORY_SIX_HOUR_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "history_six_hour_groups_v1"
HISTORY_HARNESS_EXE = Path("tools/gc_harness/build/history_harness")
DRY_PRESSURE_HARNESS_EXE = Path("tools/gc_harness/build/dry_pressure_harness")
MET_AIRQNT_HARNESS_EXE = Path("tools/gc_harness/build/met_airqnt_harness")


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


def test_history_harness_run_directory_contains_minimal_history_config(tmp_path):
    run_dir = write_history_harness_run_directory(
        tmp_path / "history",
        ntracer=3,
        acc_interval="00000000 010000",
    )

    history = (run_dir / "HISTORY.rc").read_text(encoding="utf-8")
    species = (run_dir / "species_database.yml").read_text(encoding="utf-8")
    config = (run_dir / "geoschem_config.yml").read_text(encoding="utf-8")

    assert "SpeciesConcThreeHourly.fields: 'SpeciesConcVV_?ADV?'" in history
    assert "SpeciesConcThreeHourly.acc_interval: 00000000 010000" in history
    assert "hist_001:" in species
    assert "hist_003:" in species
    assert "MW_g: 28.97" in species
    assert "name: TransportTracers" in config
    assert "aod_wavelengths_in_nm: [550]" in config
    assert "transported_species: hist_001, hist_002, hist_003" in config


def test_history_harness_compare_reconstructs_default_window(tmp_path):
    output_path = tmp_path / HISTORY_HARNESS_OUTPUT_NAME
    output_path.parent.mkdir(parents=True)
    with netCDF4.Dataset(output_path, "w") as dataset:
        dataset.createDimension("time", 8)
        dataset.createDimension("lev", 1)
        dataset.createDimension("lat", 1)
        dataset.createDimension("lon", 1)
        time = dataset.createVariable("time", "f8", ("time",))
        time[:] = np.arange(8) * 180.0
        for tracer_index in range(1, 3):
            variable = dataset.createVariable(
                f"SpeciesConcVV_hist_{tracer_index:03d}",
                "f8",
                ("time", "lev", "lat", "lon"),
            )
            expected_time = np.array(
                [tracer_index * 1000 + np.mean(np.arange(i * 18 + 1, (i + 1) * 18 + 1)) for i in range(8)]
            )
            variable[:] = expected_time[:, None, None, None] + 0.136

    comparison = compare_history_harness_output(output_path, ntracer=2)

    assert comparison.max_abs_error == 0.0
    assert comparison.max_time_error_min == 0.0
    assert comparison.first_record_expected == 1009.636
    assert comparison.first_record_actual == 1009.636
    assert comparison.boundary_included_in_previous
    assert "max_abs_error,0.00000000e+00" in format_history_harness_comparison(comparison)


@pytest.mark.skipif(not HISTORY_HARNESS_EXE.exists(), reason="GEOS-Chem HISTORY harness executable is not built")
def test_history_harness_optional_geos_chem_compare(tmp_path):
    output = run_history_harness(HISTORY_HARNESS_EXE, tmp_path / "history", ntracer=2)
    comparison = compare_history_harness_output(output, ntracer=2)

    assert comparison.n_records == 8
    assert comparison.n_tracers == 2
    assert comparison.max_abs_error < 1.0e-4
    assert comparison.max_time_error_min == 0.0
    assert comparison.first_record_expected == 1009.636
    assert comparison.boundary_included_in_previous


@pytest.mark.parametrize(
    ("scenario", "fixture_dir"),
    (
        ("default", HISTORY_DEFAULT_FIXTURE_DIR),
        ("six_hour_groups", HISTORY_SIX_HOUR_FIXTURE_DIR),
    ),
)
def test_history_harness_fixtures_match_wombat_output_manager(tmp_path, scenario, fixture_dir):
    config = history_harness_scenario_config(scenario)
    reference = fixture_dir / HISTORY_HARNESS_OUTPUT_NAME

    comparison = compare_history_harness_output(
        reference,
        ntracer=int(config["ntracer"]),
        nsteps=int(config["nsteps"]),
        dt_s=int(config["dt_s"]),
        frequency_s=_history_interval_seconds_for_test(str(config["frequency"])),
    )
    assert comparison.max_abs_error < 1.0e-4

    wombat_comparison = compare_history_harness_to_wombat(
        reference,
        tmp_path / scenario,
        ntracer=int(config["ntracer"]),
        nsteps=int(config["nsteps"]),
        dt_s=int(config["dt_s"]),
        frequency=str(config["frequency"]),
        duration=str(config["duration"]),
    )

    assert wombat_comparison.n_records == int(config["nsteps"]) // (
        _history_interval_seconds_for_test(str(config["frequency"])) // int(config["dt_s"])
    )
    assert wombat_comparison.n_tracers == int(config["ntracer"])
    assert wombat_comparison.max_abs_error == 0.0
    assert wombat_comparison.max_time_error_min == 0.0
    assert wombat_comparison.max_coord_error < 1.0e-5
    assert "max_abs_error,0.00000000e+00" in format_history_harness_wombat_comparison(wombat_comparison)


def test_history_harness_fixture_metadata_records_generation_contract():
    metadata = json.loads((HISTORY_DEFAULT_FIXTURE_DIR / SNAPSHOT_METADATA_NAME).read_text(encoding="utf-8"))

    assert metadata["version"] == "history-harness-v1"
    assert metadata["scenario"] == "default"
    assert metadata["output"]["name"] == Path(HISTORY_HARNESS_OUTPUT_NAME).name


def test_dry_pressure_fixture_matches_wombat_pressure_bookkeeping(tmp_path):
    input_path = DRY_PRESSURE_FIXTURE_DIR / DRY_PRESSURE_SNAPSHOT_INPUT_NAME
    output_path = DRY_PRESSURE_FIXTURE_DIR / DRY_PRESSURE_SNAPSHOT_OUTPUT_NAME
    metadata = json.loads((DRY_PRESSURE_FIXTURE_DIR / SNAPSHOT_METADATA_NAME).read_text(encoding="utf-8"))

    comparison = compare_dry_pressure_output(
        input_path,
        output_path,
        python_output_path=tmp_path / "python_dry_pressure_output.nc",
    )

    assert metadata["version"] == DRY_PRESSURE_SNAPSHOT_VERSION
    assert metadata["input"]["name"] == DRY_PRESSURE_SNAPSHOT_INPUT_NAME
    assert metadata["output"]["harness"] == DRY_PRESSURE_OUTPUT_VERSION
    assert comparison.ps1_wet_max_abs_error_hpa < 1.0e-12
    assert comparison.ps2_wet_max_abs_error_hpa < 1.0e-12
    assert comparison.ps1_dry_max_abs_error_hpa < 1.0e-12
    assert comparison.ps2_dry_max_abs_error_hpa < 1.0e-12
    assert comparison.psc2_dry_max_abs_error_hpa < 1.0e-12
    assert comparison.delp_dry_max_abs_error_hpa < 1.0e-12
    assert comparison.specific_humidity_max_abs_error < 1.0e-15
    assert comparison.temperature_max_abs_error < 1.0e-12


def test_met_airqnt_fixture_matches_wombat_air_quantity_bookkeeping(tmp_path):
    input_path = MET_AIRQNT_FIXTURE_DIR / MET_AIRQNT_SNAPSHOT_INPUT_NAME
    output_path = MET_AIRQNT_FIXTURE_DIR / MET_AIRQNT_SNAPSHOT_OUTPUT_NAME
    metadata = json.loads((MET_AIRQNT_FIXTURE_DIR / SNAPSHOT_METADATA_NAME).read_text(encoding="utf-8"))

    comparison = compare_met_airqnt_output(
        input_path,
        output_path,
        python_output_path=tmp_path / "python_met_airqnt_output.nc",
    )

    assert metadata["version"] == MET_AIRQNT_SNAPSHOT_VERSION
    assert metadata["input"]["name"] == MET_AIRQNT_SNAPSHOT_INPUT_NAME
    assert metadata["output"]["harness"] == MET_AIRQNT_OUTPUT_VERSION
    assert comparison.max_abs_errors["psc2_dry_hpa"] < 1.0e-12
    assert comparison.max_abs_errors["wet_pressure_edges_hpa"] < 1.0e-12
    assert comparison.max_abs_errors["wet_pressure_thickness_hpa"] < 1.0e-12
    assert comparison.max_abs_errors["dry_partial_pressure_edges_hpa"] < 1.0e-12
    assert comparison.max_abs_errors["delp_dry_hpa"] < 1.0e-12
    assert comparison.max_abs_errors["specific_humidity_kg_kg"] < 1.0e-15
    assert comparison.max_abs_errors["temperature_k"] < 1.0e-12
    assert comparison.max_abs_errors["virtual_temperature_k"] < 1.0e-12
    assert comparison.max_abs_errors["bxheight_m"] < 1.0e-9
    assert comparison.max_abs_errors["dry_air_mass_kg"] < 100.0
    assert "bxheight_m" in format_met_airqnt_comparison(comparison)


@pytest.mark.skipif(not DRY_PRESSURE_HARNESS_EXE.exists(), reason="GEOS-Chem dry-pressure harness executable is not built")
def test_dry_pressure_harness_optional_geos_chem_compare(tmp_path):
    input_path = write_synthetic_dry_pressure_input(tmp_path / DRY_PRESSURE_SNAPSHOT_INPUT_NAME)
    output_path = tmp_path / DRY_PRESSURE_SNAPSHOT_OUTPUT_NAME

    run_operator_harness(DRY_PRESSURE_HARNESS_EXE, input_path, output_path)
    comparison = compare_dry_pressure_output(input_path, output_path)

    assert comparison.ps1_dry_max_abs_error_hpa < 1.0e-12
    assert comparison.ps2_dry_max_abs_error_hpa < 1.0e-12
    assert comparison.delp_dry_max_abs_error_hpa < 1.0e-12


@pytest.mark.skipif(not MET_AIRQNT_HARNESS_EXE.exists(), reason="GEOS-Chem AIRQNT harness executable is not built")
def test_met_airqnt_harness_optional_geos_chem_compare(tmp_path):
    input_path = write_synthetic_dry_pressure_input(tmp_path / MET_AIRQNT_SNAPSHOT_INPUT_NAME)
    output_path = tmp_path / MET_AIRQNT_SNAPSHOT_OUTPUT_NAME

    run_operator_harness(MET_AIRQNT_HARNESS_EXE, input_path, output_path)
    comparison = compare_met_airqnt_output(input_path, output_path)

    assert comparison.max_abs_errors["delp_dry_hpa"] < 1.0e-12
    assert comparison.max_abs_errors["bxheight_m"] < 1.0e-9
    assert comparison.max_abs_errors["dry_air_mass_kg"] < 100.0


def test_large_oracle_fixture_check_reports_missing_payloads(tmp_path):
    cache_dir = tmp_path / "oracle_data"
    manifest_dir = cache_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    _write_large_oracle_definition(manifest_dir / f"{BASE_INITIAL_TPCORE_FIXTURE_ID}.json")

    check = check_large_oracle_fixture(BASE_INITIAL_TPCORE_FIXTURE_ID, cache_dir=cache_dir, manifest_dir=manifest_dir)

    assert not check.is_available
    assert check.missing_files == ("transport_step_input.nc", "transport_step_output.nc")
    assert "available,False" in format_large_oracle_fixture_check(check)


def test_fullgrid_operator_oracle_fixture_checks_report_operator_payload_names(tmp_path):
    cache_dir = tmp_path / "oracle_data"
    manifest_dir = cache_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    _write_large_oracle_definition(
        manifest_dir / f"{BASE_INITIAL_VDIFF_AFTER_TPCORE_FIXTURE_ID}.json",
        fixture_id=BASE_INITIAL_VDIFF_AFTER_TPCORE_FIXTURE_ID,
        input_name="vdiff_input.nc",
        output_name="vdiff_output.nc",
    )
    _write_large_oracle_definition(
        manifest_dir / f"{BASE_INITIAL_CONVECTION_FULLGRID_FIXTURE_ID}.json",
        fixture_id=BASE_INITIAL_CONVECTION_FULLGRID_FIXTURE_ID,
        input_name="convection_input.nc",
        output_name="convection_output.nc",
    )

    vdiff_check = check_large_oracle_fixture(
        BASE_INITIAL_VDIFF_AFTER_TPCORE_FIXTURE_ID,
        cache_dir=cache_dir,
        manifest_dir=manifest_dir,
    )
    convection_check = check_large_oracle_fixture(
        BASE_INITIAL_CONVECTION_FULLGRID_FIXTURE_ID,
        cache_dir=cache_dir,
        manifest_dir=manifest_dir,
    )

    assert vdiff_check.missing_files == ("vdiff_input.nc", "vdiff_output.nc")
    assert convection_check.missing_files == ("convection_input.nc", "convection_output.nc")


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


def test_fullgrid_vdiff_after_tpcore_oracle_fixture_if_cached_reports_vdiff_metrics():
    check = check_large_oracle_fixture(BASE_INITIAL_VDIFF_AFTER_TPCORE_FIXTURE_ID)
    if not check.is_available:
        pytest.skip(format_large_oracle_fixture_check(check))

    report = compare_large_oracle_fixture(BASE_INITIAL_VDIFF_AFTER_TPCORE_FIXTURE_ID)
    metrics = dict(line.split(",", 1) for line in report.splitlines()[1:])

    assert "specific_humidity_max_abs_error" in metrics
    assert "kvh_max_abs_error" in metrics
    assert "common_basis_final_mass_max_abs_error" in metrics
    assert "reported_final_mass_max_abs_error" in metrics
    assert float(metrics["tracer_max_abs_error"]) < 1.0e-15
    assert int(metrics["negative_count_after_clip_actual"]) == 0


def test_fullgrid_convection_oracle_fixture_if_cached_reports_convection_metrics():
    check = check_large_oracle_fixture(BASE_INITIAL_CONVECTION_FULLGRID_FIXTURE_ID)
    if not check.is_available:
        pytest.skip(format_large_oracle_fixture_check(check))

    report = compare_large_oracle_fixture(BASE_INITIAL_CONVECTION_FULLGRID_FIXTURE_ID)
    metrics = dict(line.split(",", 1) for line in report.splitlines()[1:])

    assert "diag14_max_abs_error" in metrics
    assert "internal_steps_actual" in metrics
    assert "common_basis_final_mass_max_abs_error" in metrics
    assert "reported_final_mass_max_abs_error" in metrics
    assert float(metrics["tracer_max_abs_error"]) < 1.0e-15
    assert int(metrics["negative_count_after_actual"]) == 0


def test_transport_chain_oracle_fixture_if_cached_reports_common_and_reported_mass_metrics():
    check = check_large_oracle_fixture(BASE_INITIAL_TRANSPORT_CHAIN_FIXTURE_ID)
    if not check.is_available:
        pytest.skip(format_large_oracle_fixture_check(check))

    report = compare_large_oracle_fixture(BASE_INITIAL_TRANSPORT_CHAIN_FIXTURE_ID)
    metrics = dict(line.split(",", 1) for line in report.splitlines()[1:])

    assert "common_basis_final_mass_max_abs_error" in metrics
    assert "reported_final_mass_max_abs_error" in metrics
    assert "common_basis_vdiff_stage_mass_change_max_abs" in metrics
    assert "reported_vdiff_stage_mass_change_max_abs" in metrics
    assert int(metrics["negative_count_actual"]) == 0


def test_transport_chain_handoff_report_if_cached_locates_stage_interfaces():
    required = (
        BASE_INITIAL_TRANSPORT_CHAIN_FIXTURE_ID,
        BASE_INITIAL_VDIFF_AFTER_TPCORE_FIXTURE_ID,
        BASE_INITIAL_CONVECTION_FULLGRID_FIXTURE_ID,
    )
    for fixture_id in required:
        check = check_large_oracle_fixture(fixture_id)
        if not check.is_available:
            pytest.skip(format_large_oracle_fixture_check(check))

    report = compare_transport_chain_handoffs(BASE_INITIAL_TRANSPORT_CHAIN_FIXTURE_ID)
    rows = {
        (section, field): (float(max_abs), float(mean_abs))
        for section, field, max_abs, mean_abs, *_ in (line.split(",") for line in report.splitlines()[1:])
    }

    assert ("tpcore_to_vdiff_input", "tracer_conc") in rows
    assert ("tpcore_to_vdiff_input", "dry_air_mass_kg") in rows
    assert ("vdiff_output", "tracer_conc_after") in rows
    assert ("vdiff_to_convection_input", "tracer_conc") in rows
    assert ("vdiff_to_convection_input", "cmfmc_kg_m2_s") in rows
    assert ("convection_output", "tracer_conc_after") in rows
    assert ("final_chain_output", "tracer_conc_after") in rows
    assert rows[("tpcore_to_vdiff_input", "virtual_temperature_k")][0] < 1.0e-15
    assert rows[("tpcore_to_vdiff_input", "bxheight_m")][0] < 1.0e-10
    assert rows[("vdiff_output", "tracer_conc_after")][0] < 1.0e-15
    assert rows[("final_chain_output", "tracer_conc_after")][0] < 1.0e-15


def test_residual_initial_oracle_fixture_if_cached_matches_python_tpcore():
    check = check_large_oracle_fixture(RESIDUAL_INITIAL_TPCORE_FIXTURE_ID)
    if not check.is_available:
        pytest.skip(format_large_oracle_fixture_check(check))

    paths = large_oracle_fixture_paths(RESIDUAL_INITIAL_TPCORE_FIXTURE_ID)
    with netCDF4.Dataset(paths.input_path) as dataset:
        tracer = np.asarray(dataset.variables["tracer_conc"][:], dtype=np.float64)
        names = tuple(str(value).strip() for value in netCDF4.chartostring(dataset.variables["tracer_name"][:]))
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

    assert tracer.shape == (47, 91, 144, 24)
    assert names[0] == "r0002p001s001"
    assert names[-1] == "r0002p001s024"
    assert branch_report.is_supported
    assert branch_report.shape == (47, 91, 144)

    report = compare_large_oracle_fixture(RESIDUAL_INITIAL_TPCORE_FIXTURE_ID)
    metrics = dict(line.split(",", 1) for line in report.splitlines()[1:])

    assert "tracer_count,24" in report
    assert "r0002p001s001" in report
    assert "r0002p001s024" in report
    assert "python_negative_count_after,0" in report
    assert float(metrics["tracer_max_abs_error"]) < 1.0e-16
    assert float(metrics["max_abs_cx"]) > 0.0
    assert float(metrics["max_abs_cy"]) > 0.0


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


def test_write_synthetic_vdiff_input_records_fixture_contract(tmp_path):
    input_path = write_synthetic_vdiff_input(tmp_path / "vdiff_input.nc")

    with netCDF4.Dataset(input_path) as dataset:
        assert dataset.harness == VDIFF_INPUT_VERSION
        assert dataset.scenario == "zero_surface_flux"
        assert dataset.dt_s == 600.0
        assert dataset.dimensions["lev"].size == 47
        assert dataset.dimensions["ilev"].size == 48
        assert dataset.dimensions["lat"].size == 3
        assert dataset.dimensions["lon"].size == 4
        assert dataset.dimensions["tracer"].size == 2
        assert dataset.variables["tracer_conc"].shape == (47, 3, 4, 2)
        assert dataset.variables["surface_flux_kg_m2_s"].shape == (3, 4, 2)


def test_write_synthetic_vdiff_input_can_exercise_nonzero_surface_flux(tmp_path):
    input_path = write_synthetic_vdiff_input(tmp_path / "vdiff_input.nc", scenario="nonzero_surface_flux")

    with netCDF4.Dataset(input_path) as dataset:
        surface_flux = np.asarray(dataset.variables["surface_flux_kg_m2_s"][:])
        assert dataset.scenario == "nonzero_surface_flux"
        assert np.count_nonzero(surface_flux) == surface_flux.size
        assert np.all(surface_flux > 0.0)


def test_write_synthetic_vdiff_input_can_exercise_negative_clipping(tmp_path):
    input_path = write_synthetic_vdiff_input(tmp_path / "vdiff_input.nc", scenario="negative_clipping")

    with netCDF4.Dataset(input_path) as dataset:
        tracer = np.asarray(dataset.variables["tracer_conc"][:])
        assert dataset.scenario == "negative_clipping"
        assert np.count_nonzero(tracer < 0.0) == 2


def test_write_synthetic_dry_pressure_input_records_fixture_contract(tmp_path):
    input_path = write_synthetic_dry_pressure_input(tmp_path / DRY_PRESSURE_SNAPSHOT_INPUT_NAME)

    with netCDF4.Dataset(input_path) as dataset:
        assert dataset.harness == DRY_PRESSURE_INPUT_VERSION
        assert dataset.ntime0_s == 0
        assert dataset.ntime1_s == 3600
        assert dataset.ntdt_s == 600
        assert dataset.dimensions["lev"].size == 47
        assert dataset.dimensions["ilev"].size == 48
        assert dataset.dimensions["lat"].size == 7
        assert dataset.dimensions["lon"].size == 8
        assert dataset.variables["ps1_wet_hpa"].shape == (7, 8)
        assert dataset.variables["sphu1_kg_kg"].shape == (47, 7, 8)


def test_python_dry_pressure_output_roundtrips_through_comparison_contract(tmp_path):
    input_path = write_synthetic_dry_pressure_input(tmp_path / DRY_PRESSURE_SNAPSHOT_INPUT_NAME)
    output_path = write_python_dry_pressure_output(input_path, tmp_path / DRY_PRESSURE_SNAPSHOT_OUTPUT_NAME)

    output = read_dry_pressure_output(output_path)
    assert output.delp_dry_hpa.shape == (47, 7, 8)
    assert output.ps1_wet_hpa.shape == (7, 8)
    assert output.ps1_dry_hpa.shape == (7, 8)
    assert np.all(output.delp_dry_hpa > 0.0)
    np.testing.assert_allclose(output.ps1_wet_hpa[0, :], output.ps1_wet_hpa[1, :])
    np.testing.assert_allclose(output.ps1_wet_hpa[-1, :], output.ps1_wet_hpa[-2, :])

    comparison = compare_dry_pressure_output(input_path, output_path)
    assert comparison.delp_dry_max_abs_error_hpa == 0.0
    assert "delp_dry_max_abs_error_hpa,0.00000000e+00" in format_dry_pressure_comparison(comparison)


def test_python_met_airqnt_output_roundtrips_through_comparison_contract(tmp_path):
    input_path = write_synthetic_dry_pressure_input(tmp_path / MET_AIRQNT_SNAPSHOT_INPUT_NAME)
    output_path = write_python_met_airqnt_output(input_path, tmp_path / MET_AIRQNT_SNAPSHOT_OUTPUT_NAME)

    output = read_met_airqnt_output(output_path)
    assert output.wet_pressure_edges_hpa.shape == (48, 7, 8)
    assert output.delp_dry_hpa.shape == (47, 7, 8)
    assert output.bxheight_m.shape == (47, 7, 8)
    assert output.dry_air_mass_kg.shape == (47, 7, 8)
    assert np.all(output.wet_pressure_thickness_hpa > 0.0)
    assert np.all(output.bxheight_m > 0.0)

    comparison = compare_met_airqnt_output(input_path, output_path)
    assert comparison.max_abs_errors["delp_dry_hpa"] == 0.0
    assert comparison.max_abs_errors["bxheight_m"] == 0.0


def test_python_vdiff_output_roundtrips_through_comparison_contract(tmp_path):
    input_path = write_synthetic_vdiff_input(tmp_path / "vdiff_input.nc")
    output_path = write_python_vdiff_output(input_path, tmp_path / "vdiff_output.nc")

    output = read_vdiff_output(output_path)
    assert output.tracer_conc_after.shape == (47, 3, 4, 2)
    assert output.specific_humidity_after.shape == (47, 3, 4)
    assert output.kvh_m2_s.shape == (48, 3, 4)
    assert output.kvm_m2_s.shape == (48, 3, 4)
    assert output.negative_count_after_clip == 0

    comparison = compare_vdiff_output(input_path, output_path)
    assert comparison.tracer_max_abs_error == 0.0
    assert comparison.specific_humidity_max_abs_error == 0.0
    assert comparison.kvh_max_abs_error == 0.0
    assert "tracer_max_abs_error,0.00000000e+00" in format_vdiff_comparison(comparison)
    assert "common_basis_final_mass_max_abs_error,0.00000000e+00" in format_vdiff_comparison(comparison)


def test_vdiff_common_basis_mass_ignores_perturbed_reported_scalar(tmp_path):
    input_path = write_synthetic_vdiff_input(tmp_path / "vdiff_input.nc")
    output_path = write_python_vdiff_output(input_path, tmp_path / "vdiff_output.nc")
    with netCDF4.Dataset(output_path, "a") as dataset:
        dataset.variables["final_tracer_mass"][:] = np.asarray(dataset.variables["final_tracer_mass"][:]) + 123.0

    comparison = compare_vdiff_output(
        input_path,
        output_path,
        python_output_path=tmp_path / "python_vdiff_output.nc",
    )

    assert comparison.tracer_max_abs_error == 0.0
    assert comparison.common_basis_final_mass_max_abs_error == 0.0
    assert comparison.common_basis_mass_change_max_abs_error == 0.0
    assert comparison.reported_final_mass_max_abs_error == pytest.approx(123.0)


def test_write_synthetic_convection_input_records_fixture_contract(tmp_path):
    input_path = write_synthetic_convection_input(tmp_path / "convection_input.nc")

    with netCDF4.Dataset(input_path) as dataset:
        assert dataset.harness == CONVECTION_INPUT_VERSION
        assert dataset.scenario == "active_cloud"
        assert dataset.dt_s == 600.0
        assert dataset.dimensions["lev"].size == 47
        assert dataset.dimensions["lat"].size == 2
        assert dataset.dimensions["lon"].size == 3
        assert dataset.dimensions["tracer"].size == 2
        assert dataset.variables["tracer_conc"].shape == (47, 2, 3, 2)
        assert dataset.variables["cmfmc_kg_m2_s"].shape == (47, 2, 3)
        assert dataset.variables["precccon_mm_day"].shape == (2, 3)
        names = tuple(str(value).strip() for value in netCDF4.chartostring(dataset.variables["tracer_name"][:]))
        assert names == ("conv_001", "conv_002")


def test_python_convection_output_roundtrips_through_comparison_contract(tmp_path):
    input_path = write_synthetic_convection_input(tmp_path / "convection_input.nc")
    output_path = write_python_convection_output(input_path, tmp_path / "convection_output.nc")

    output = read_convection_output(output_path)
    with netCDF4.Dataset(output_path) as dataset:
        assert dataset.harness == CONVECTION_OUTPUT_VERSION
    assert output.tracer_conc_after.shape == (47, 2, 3, 2)
    assert output.diag14_mass_flux.shape == (47, 2, 3, 2)
    assert output.internal_steps == 2
    assert output.internal_dt_s == 300.0
    assert output.negative_count_after == 0

    comparison = compare_convection_output(input_path, output_path)
    assert comparison.tracer_max_abs_error == 0.0
    assert comparison.diag14_max_abs_error == 0.0
    assert "tracer_max_abs_error,0.00000000e+00" in format_convection_comparison(comparison)
    assert "common_basis_final_mass_max_abs_error,0.00000000e+00" in format_convection_comparison(comparison)


def test_convection_common_basis_mass_ignores_perturbed_reported_scalar(tmp_path):
    input_path = write_synthetic_convection_input(tmp_path / "convection_input.nc")
    output_path = write_python_convection_output(input_path, tmp_path / "convection_output.nc")
    with netCDF4.Dataset(output_path, "a") as dataset:
        dataset.variables["initial_tracer_mass"][:] = np.asarray(dataset.variables["initial_tracer_mass"][:]) - 10.0
        dataset.variables["final_tracer_mass"][:] = np.asarray(dataset.variables["final_tracer_mass"][:]) + 15.0

    comparison = compare_convection_output(
        input_path,
        output_path,
        python_output_path=tmp_path / "python_convection_output.nc",
    )

    assert comparison.tracer_max_abs_error == 0.0
    assert comparison.common_basis_final_mass_max_abs_error == 0.0
    assert comparison.common_basis_mass_change_max_abs_error == 0.0
    assert comparison.reported_initial_mass_max_abs_error == pytest.approx(10.0)
    assert comparison.reported_final_mass_max_abs_error == pytest.approx(15.0)


def test_convection_no_cloud_leaves_tracer_unchanged(tmp_path):
    input_path = write_synthetic_convection_input(tmp_path / "convection_input.nc", scenario="no_cloud")

    with netCDF4.Dataset(input_path) as dataset:
        tracer = np.asarray(dataset.variables["tracer_conc"][:], dtype=np.float64)
        result = run_cloud_convection_one_step(
            tracer_conc=tracer,
            cmfmc_kg_m2_s=np.asarray(dataset.variables["cmfmc_kg_m2_s"][:], dtype=np.float64),
            dtrain_kg_m2_s=np.asarray(dataset.variables["dtrain_kg_m2_s"][:], dtype=np.float64),
            dqrcu_kg_kg_s=np.asarray(dataset.variables["dqrcu_kg_kg_s"][:], dtype=np.float64),
            reevapcn_kg_kg_s=np.asarray(dataset.variables["reevapcn_kg_kg_s"][:], dtype=np.float64),
            delp_dry_hpa=np.asarray(dataset.variables["delp_dry_hpa"][:], dtype=np.float64),
            delp_hpa=np.asarray(dataset.variables["delp_hpa"][:], dtype=np.float64),
            area_m2=np.asarray(dataset.variables["area_m2"][:], dtype=np.float64),
            bxheight_m=np.asarray(dataset.variables["bxheight_m"][:], dtype=np.float64),
            pficu_kg_m2_s=np.asarray(dataset.variables["pficu_kg_m2_s"][:], dtype=np.float64),
            pflcu_kg_m2_s=np.asarray(dataset.variables["pflcu_kg_m2_s"][:], dtype=np.float64),
            temperature_k=np.asarray(dataset.variables["temperature_k"][:], dtype=np.float64),
            precccon_mm_day=np.asarray(dataset.variables["precccon_mm_day"][:], dtype=np.float64),
            dt_s=float(dataset.dt_s),
        )

    np.testing.assert_array_equal(result.tracer_conc, tracer)
    np.testing.assert_array_equal(result.initial_tracer_mass, result.final_tracer_mass)


def test_convection_active_cloud_changes_tracer_and_conserves_mass(tmp_path):
    input_path = write_synthetic_convection_input(tmp_path / "convection_input.nc", scenario="active_cloud")
    output_path = write_python_convection_output(input_path, tmp_path / "convection_output.nc")
    output = read_convection_output(output_path)

    with netCDF4.Dataset(input_path) as dataset:
        before = np.asarray(dataset.variables["tracer_conc"][:], dtype=np.float64)

    assert float(np.max(np.abs(output.tracer_conc_after - before))) > 0.0
    np.testing.assert_allclose(output.final_tracer_mass, output.initial_tracer_mass, rtol=1.0e-14, atol=0.0)


def test_convection_diagnostics_light_preserves_tracer_update(tmp_path):
    input_path = write_synthetic_convection_input(tmp_path / "convection_input.nc", scenario="active_cloud")
    with netCDF4.Dataset(input_path) as dataset:
        kwargs = dict(
            tracer_conc=np.asarray(dataset.variables["tracer_conc"][:], dtype=np.float64),
            cmfmc_kg_m2_s=np.asarray(dataset.variables["cmfmc_kg_m2_s"][:], dtype=np.float64),
            dtrain_kg_m2_s=np.asarray(dataset.variables["dtrain_kg_m2_s"][:], dtype=np.float64),
            dqrcu_kg_kg_s=np.asarray(dataset.variables["dqrcu_kg_kg_s"][:], dtype=np.float64),
            reevapcn_kg_kg_s=np.asarray(dataset.variables["reevapcn_kg_kg_s"][:], dtype=np.float64),
            delp_dry_hpa=np.asarray(dataset.variables["delp_dry_hpa"][:], dtype=np.float64),
            delp_hpa=np.asarray(dataset.variables["delp_hpa"][:], dtype=np.float64),
            area_m2=np.asarray(dataset.variables["area_m2"][:], dtype=np.float64),
            bxheight_m=np.asarray(dataset.variables["bxheight_m"][:], dtype=np.float64),
            pficu_kg_m2_s=np.asarray(dataset.variables["pficu_kg_m2_s"][:], dtype=np.float64),
            pflcu_kg_m2_s=np.asarray(dataset.variables["pflcu_kg_m2_s"][:], dtype=np.float64),
            temperature_k=np.asarray(dataset.variables["temperature_k"][:], dtype=np.float64),
            precccon_mm_day=np.asarray(dataset.variables["precccon_mm_day"][:], dtype=np.float64),
            dt_s=float(dataset.dt_s),
        )

    full = run_cloud_convection_one_step(**kwargs, diagnostics=True)
    light = run_cloud_convection_one_step(**kwargs, diagnostics=False)
    reused = run_cloud_convection_one_step(**kwargs, diagnostics=False, reuse_output=True)

    np.testing.assert_array_equal(light.tracer_conc, full.tracer_conc)
    np.testing.assert_array_equal(reused.tracer_conc, full.tracer_conc)
    assert light.diag14_mass_flux.shape == (0,)
    assert light.initial_tracer_mass.shape == (0,)
    assert light.final_tracer_mass.shape == (0,)
    assert light.negative_count_before == 0
    assert light.negative_count_after == 0
    assert reused.diag14_mass_flux.shape == (0,)
    assert reused.initial_tracer_mass.shape == (0,)
    assert reused.final_tracer_mass.shape == (0,)
    assert reused.negative_count_before == 0
    assert reused.negative_count_after == 0


def test_convection_vectorized_batches_mixed_cloud_bases_and_inactive_columns():
    nlev, nlat, nlon, ntracer = 5, 2, 2, 2
    tracer = np.full((nlev, nlat, nlon, ntracer), 4.0e-4, dtype=np.float64)
    tracer += 1.0e-7 * np.arange(ntracer, dtype=np.float64)[np.newaxis, np.newaxis, np.newaxis, :]
    tracer += 1.0e-8 * np.arange(nlev, dtype=np.float64)[:, np.newaxis, np.newaxis, np.newaxis]
    cmfmc = np.zeros((nlev, nlat, nlon), dtype=np.float64)
    dtrain = np.zeros_like(cmfmc)
    dqrcu = np.zeros_like(cmfmc)
    reevapcn = np.zeros_like(cmfmc)
    delp_dry = np.broadcast_to(np.linspace(10.0, 50.0, nlev)[:, np.newaxis, np.newaxis], (nlev, nlat, nlon)).copy()
    delp = delp_dry * 1.01
    area = np.full((nlat, nlon), 2.0e10, dtype=np.float64)

    cmfmc[:, 0, 1] = [0.0, 0.003, 0.004, 0.005, 0.006]
    dtrain[:, 0, 1] = [0.0, 0.0002, 0.0002, 0.0002, 0.0002]
    dqrcu[3, 0, 1] = 1.0e-8
    cmfmc[:, 1, 0] = [0.0, 0.002, 0.003, 0.004, 0.0]
    dtrain[:, 1, 0] = [0.0, 0.0001, 0.0001, 0.0001, 0.0]
    dqrcu[2, 1, 0] = 1.0e-8
    cmfmc[:, 1, 1] = [0.0, 0.0, 0.0, 0.0, 0.004]
    dtrain[:, 1, 1] = [0.0, 0.0, 0.0, 0.0, 0.0001]

    result = run_cloud_convection_one_step(
        tracer_conc=tracer,
        cmfmc_kg_m2_s=cmfmc,
        dtrain_kg_m2_s=dtrain,
        dqrcu_kg_kg_s=dqrcu,
        reevapcn_kg_kg_s=reevapcn,
        delp_dry_hpa=delp_dry,
        delp_hpa=delp,
        area_m2=area,
        dt_s=600.0,
    )

    np.testing.assert_array_equal(result.tracer_conc[:, 0, 0, :], tracer[:, 0, 0, :])
    np.testing.assert_array_equal(result.diag14_mass_flux[:, 0, 0, :], np.zeros((nlev, ntracer)))
    assert float(np.max(np.abs(result.tracer_conc[:, 0, 1, :] - tracer[:, 0, 1, :]))) > 0.0
    assert float(np.max(np.abs(result.tracer_conc[:, 1, 0, :] - tracer[:, 1, 0, :]))) > 0.0
    assert float(np.max(np.abs(result.tracer_conc[:, 1, 1, :] - tracer[:, 1, 1, :]))) > 0.0
    assert np.all(np.isfinite(result.tracer_conc))
    assert result.negative_count_after == 0


def test_convection_column_chunking_preserves_results(monkeypatch):
    nlev, nlat, nlon, ntracer = 6, 2, 4, 3
    tracer = np.full((nlev, nlat, nlon, ntracer), 4.0e-4, dtype=np.float64)
    tracer += 1.0e-7 * np.arange(ntracer, dtype=np.float64)[np.newaxis, np.newaxis, np.newaxis, :]
    tracer += 1.0e-8 * np.arange(nlev, dtype=np.float64)[:, np.newaxis, np.newaxis, np.newaxis]
    tracer += 1.0e-9 * np.arange(nlat * nlon, dtype=np.float64).reshape(1, nlat, nlon, 1)
    cmfmc = np.zeros((nlev, nlat, nlon), dtype=np.float64)
    dtrain = np.zeros_like(cmfmc)
    dqrcu = np.zeros_like(cmfmc)
    reevapcn = np.zeros_like(cmfmc)
    delp_dry = np.broadcast_to(np.linspace(10.0, 55.0, nlev)[:, np.newaxis, np.newaxis], (nlev, nlat, nlon)).copy()
    delp = delp_dry * 1.01
    area = np.full((nlat, nlon), 2.0e10, dtype=np.float64)
    cmfmc[1:, :, :] = np.linspace(0.002, 0.006, nlev - 1, dtype=np.float64)[:, np.newaxis, np.newaxis]
    dtrain[1:, :, :] = 1.0e-4
    dqrcu[3, :, :] = 1.0e-8

    wide = run_cloud_convection_one_step(
        tracer_conc=tracer,
        cmfmc_kg_m2_s=cmfmc,
        dtrain_kg_m2_s=dtrain,
        dqrcu_kg_kg_s=dqrcu,
        reevapcn_kg_kg_s=reevapcn,
        delp_dry_hpa=delp_dry,
        delp_hpa=delp,
        area_m2=area,
        dt_s=600.0,
    )
    monkeypatch.setattr(convection_mod, "_MAX_GROUP_TRACER_BYTES", nlev * ntracer * np.dtype(np.float64).itemsize)
    chunked = run_cloud_convection_one_step(
        tracer_conc=tracer,
        cmfmc_kg_m2_s=cmfmc,
        dtrain_kg_m2_s=dtrain,
        dqrcu_kg_kg_s=dqrcu,
        reevapcn_kg_kg_s=reevapcn,
        delp_dry_hpa=delp_dry,
        delp_hpa=delp,
        area_m2=area,
        dt_s=600.0,
    )

    np.testing.assert_array_equal(chunked.tracer_conc, wide.tracer_conc)
    np.testing.assert_array_equal(chunked.diag14_mass_flux, wide.diag14_mass_flux)
    np.testing.assert_array_equal(chunked.final_tracer_mass, wide.final_tracer_mass)


def test_convection_multi_tracer_preserves_ordering_and_independent_updates(tmp_path):
    input_path = write_synthetic_convection_input(
        tmp_path / "convection_input.nc",
        scenario="multi_tracer",
        ntracer=3,
    )
    output_path = write_python_convection_output(input_path, tmp_path / "convection_output.nc")
    output = read_convection_output(output_path)

    with netCDF4.Dataset(input_path) as dataset:
        names = tuple(str(value).strip() for value in netCDF4.chartostring(dataset.variables["tracer_name"][:]))
        before = np.asarray(dataset.variables["tracer_conc"][:], dtype=np.float64)
        bmass = np.asarray(dataset.variables["delp_dry_hpa"][:], dtype=np.float64) * G0_100
        area = np.asarray(dataset.variables["area_m2"][:], dtype=np.float64)

    assert names == ("conv_001", "conv_002", "conv_003")
    assert output.tracer_conc_after.shape[-1] == 3
    assert not np.array_equal(output.tracer_conc_after[0], output.tracer_conc_after[1])
    expected_initial_mass = np.sum(before * bmass[:, :, :, np.newaxis] * area[np.newaxis, :, :, np.newaxis], axis=(0, 1, 2))
    np.testing.assert_allclose(output.initial_tracer_mass, expected_initial_mass)


def test_write_real_convection_input_records_sampled_47_level_contract(tmp_path):
    _require_real_convection_inputs()
    input_path = write_real_convection_input_from_config(
        Path("residual_20140901_part001_split01_wombat/run.yml"),
        tmp_path / "convection_input.nc",
        mode="sampled-columns",
    )

    with netCDF4.Dataset(input_path) as dataset:
        assert dataset.harness == CONVECTION_INPUT_VERSION
        assert dataset.scenario == "real-sampled-columns"
        assert dataset.vertical_mapping == "native_72_to_47_center_and_73_to_48_edge"
        assert dataset.dimensions["tracer"].size == 24
        assert dataset.dimensions["lev"].size == 47
        assert dataset.dimensions["lon"].size == 1
        assert dataset.variables["cmfmc_kg_m2_s"].shape[0] == 47
        assert np.max(np.asarray(dataset.variables["cmfmc_kg_m2_s"][:])) > 0.0
        column_activity = np.max(np.abs(np.asarray(dataset.variables["cmfmc_kg_m2_s"][:])), axis=0)
        assert np.count_nonzero(column_activity == 0.0) >= 1
        assert dataset.variables["source_lat_index"].shape == (dataset.dimensions["lat"].size, 1)
        names = tuple(str(value).strip() for value in netCDF4.chartostring(dataset.variables["tracer_name"][:]))
        assert names[0] == "r0002p001s001"
        assert names[-1] == "r0002p001s024"


def test_python_real_convection_sampled_output_has_finite_mass(tmp_path):
    _require_real_convection_inputs()
    input_path = write_real_convection_input_from_config(
        Path("residual_20140901_part001_split01_wombat/run.yml"),
        tmp_path / "convection_input.nc",
        mode="sampled-columns",
        max_tracers=3,
    )
    output_path = write_python_convection_output(input_path, tmp_path / "convection_output.nc")
    output = read_convection_output(output_path)

    assert output.tracer_conc_after.shape[-1] == 3
    assert np.all(np.isfinite(output.tracer_conc_after))
    assert output.negative_count_after == 0
    np.testing.assert_allclose(output.final_tracer_mass, output.initial_tracer_mass, rtol=0.0, atol=1.0e-6)


def test_tracked_real_convection_sampled_snapshot_matches_python_port(tmp_path, transport_numba_mode):
    with (CONVECTION_REAL_FIXTURE_DIR / SNAPSHOT_METADATA_NAME).open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["snapshot"] == "convection-real-sampled-v2"
    assert metadata["shape"] == {"tracer": 24, "lev": 47, "lat": 7, "lon": 1}

    comparison = compare_convection_output(
        CONVECTION_REAL_FIXTURE_DIR / "convection_input.nc",
        CONVECTION_REAL_FIXTURE_DIR / "convection_output.nc",
        python_output_path=tmp_path / "python_convection_output.nc",
    )

    assert comparison.tracer_max_abs_error == 0.0
    assert comparison.diag14_max_abs_error == 0.0
    assert comparison.negative_count_before_actual == comparison.negative_count_before_expected
    assert comparison.negative_count_after_actual == comparison.negative_count_after_expected
    assert comparison.common_basis_python_mass_change_max_abs == 0.0
    assert comparison.common_basis_oracle_mass_change_max_abs == 0.0


def test_tracked_vdiff_snapshot_matches_python_port(tmp_path, transport_numba_mode):
    comparison = compare_vdiff_output(
        VDIFF_FIXTURE_DIR / "vdiff_input.nc",
        VDIFF_FIXTURE_DIR / "vdiff_output.nc",
        python_output_path=tmp_path / "python_vdiff_output.nc",
    )

    assert comparison.tracer_max_abs_error < 1.0e-15
    assert comparison.specific_humidity_max_abs_error < 1.0e-15
    assert comparison.kvh_max_abs_error == 0.0
    assert comparison.kvm_max_abs_error == 0.0
    assert comparison.pbl_top_max_abs_error_m == 0.0
    assert comparison.tpert_max_abs_error == 0.0
    assert comparison.qpert_max_abs_error == 0.0
    assert comparison.negative_count_before_clip_actual == comparison.negative_count_before_clip_expected
    assert comparison.negative_count_after_clip_actual == comparison.negative_count_after_clip_expected
    assert comparison.common_basis_final_mass_max_abs_error < 1.0e-12


def test_tracked_vdiff_nonzero_surface_flux_snapshot_matches_python_port(tmp_path, transport_numba_mode):
    comparison = compare_vdiff_output(
        VDIFF_NONZERO_FLUX_FIXTURE_DIR / "vdiff_input.nc",
        VDIFF_NONZERO_FLUX_FIXTURE_DIR / "vdiff_output.nc",
        python_output_path=tmp_path / "python_vdiff_output.nc",
    )

    assert comparison.tracer_max_abs_error < 1.0e-15
    assert comparison.specific_humidity_max_abs_error < 1.0e-15
    assert comparison.kvh_max_abs_error == 0.0
    assert comparison.kvm_max_abs_error == 0.0
    assert comparison.negative_count_before_clip_actual == 0
    assert comparison.negative_count_after_clip_actual == 0
    assert comparison.common_basis_final_mass_max_abs_error < 1.0e-12


def test_tracked_vdiff_negative_clipping_snapshot_matches_python_port(tmp_path, transport_numba_mode):
    comparison = compare_vdiff_output(
        VDIFF_NEGATIVE_FIXTURE_DIR / "vdiff_input.nc",
        VDIFF_NEGATIVE_FIXTURE_DIR / "vdiff_output.nc",
        python_output_path=tmp_path / "python_vdiff_output.nc",
    )

    assert comparison.tracer_max_abs_error < 1.0e-15
    assert comparison.specific_humidity_max_abs_error < 1.0e-15
    assert comparison.kvh_max_abs_error == 0.0
    assert comparison.kvm_max_abs_error == 0.0
    assert comparison.negative_count_before_clip_expected == 2
    assert comparison.negative_count_before_clip_actual == 2
    assert comparison.negative_count_after_clip_expected == 0
    assert comparison.negative_count_after_clip_actual == 0
    assert comparison.common_basis_final_mass_max_abs_error < 1.0e-12


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
        assert dataset.variables["tracer_conc"].shape == (47, 7, 8, 2)
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
    assert metadata["pressure_branch_gap_max_hpa"] == 0.0

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


def test_tpcore_fixture_can_distinguish_raw_p2_from_pjc_adjusted_pressure_branch(tmp_path):
    input_path = write_synthetic_tpcore_snapshot_input(tmp_path / "tpcore_input.nc")
    with netCDF4.Dataset(input_path, "a") as dataset:
        dataset.variables["p2_hpa"][:] = np.asarray(dataset.variables["p2_hpa"][:], dtype=np.float64) + 0.25

    assert _tpcore_pressure_branch_gap(input_path) > 1.0e-3


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


def test_calc_advec_cross_terms_uses_geos_real_index_for_positive_fractional_offsets():
    rows = np.arange(7, dtype=np.float64)[:, None]
    cols = np.arange(8, dtype=np.float64)[None, :]
    q = 100.0 * rows**2 + cols**2
    ua = np.zeros_like(q)
    va = np.zeros_like(q)
    ua[3, 3] = 0.25
    va[3, 3] = 0.25

    qqu, qqv = _calc_advec_cross_terms(q, ua, va, jn=5, js=1)

    assert qqu[3, 3] == pytest.approx(q[3, 3] + 0.5 * 0.25 * (q[3, 2] - q[3, 3]))
    assert qqv[3, 3] == pytest.approx(q[3, 3] + 0.5 * 0.25 * (q[2, 3] - q[3, 3]))


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


def test_python_tpcore_matches_low_courant_oracle_tracer_step(transport_numba_mode):
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


def test_python_tpcore_matches_fxppm_low_courant_branch_fixture(transport_numba_mode):
    with (TPCORE_FXPPM_FIXTURE_DIR / SNAPSHOT_METADATA_NAME).open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["scenario"] == "x_fxppm_low_courant"
    assert metadata["pressure_branch_gap_max_hpa"] == 0.0
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


def test_python_tpcore_matches_large_courant_branch_fixture(transport_numba_mode):
    with (TPCORE_LARGE_CX_FIXTURE_DIR / SNAPSHOT_METADATA_NAME).open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["scenario"] == "x_large_courant_polar"
    assert metadata["pressure_branch_gap_max_hpa"] < 1.0e-12
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


def test_python_tpcore_preserves_constant_tracer_on_low_courant_fixture(transport_numba_mode):
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

    np.testing.assert_allclose(traced.tracer_conc_after, normal.tracer_conc_after, rtol=0.0, atol=1.0e-18)
    np.testing.assert_allclose(trace.tracer_conc_after, normal.tracer_conc_after, rtol=0.0, atol=1.0e-18)
    assert trace.dq_after_xtp.shape == normal.tracer_conc_after.shape
    assert trace.dq_after_fzppm.shape == normal.tracer_conc_after.shape


def test_write_python_tpcore_trace_records_stage_contract(tmp_path):
    input_path = TPCORE_FIXTURE_DIR / TPCORE_SNAPSHOT_INPUT_NAME
    trace_path = write_python_tpcore_trace(input_path, tmp_path / "python_tpcore_trace.nc")

    with netCDF4.Dataset(trace_path) as dataset:
        assert dataset.harness == "tpcore-trace-v2"
        assert dataset.variables["q_after_pole_average"].shape == (47, 7, 8, 2)
        assert dataset.variables["dq_after_ytp_hpa"].shape == (47, 7, 8, 2)
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
    perturbed[3, 2, 1, 0] += 1.0e-9
    with netCDF4.Dataset(output_path, "w") as dataset:
        dataset.createDimension("lev", perturbed.shape[0])
        dataset.createDimension("lat", perturbed.shape[1])
        dataset.createDimension("lon", perturbed.shape[2])
        dataset.createDimension("tracer", perturbed.shape[3])
        dataset.harness = TRANSPORT_OUTPUT_VERSION
        dataset.createVariable("tracer_conc_after", "f8", ("lev", "lat", "lon", "tracer"))[:] = perturbed
        dataset.createVariable("xmass_hpa", "f8", ("lev", "lat", "lon"))[:] = state.xmass_hpa
        dataset.createVariable("ymass_hpa", "f8", ("lev", "lat", "lon"))[:] = state.ymass_hpa
        dataset.createVariable("surface_pressure_hpa", "f8", ("lat", "lon"))[:] = state.surface_pressure_hpa

    report = attribute_python_tpcore_error(input_path, output_path)

    assert "section,key,max_abs,mean_abs,count,extra" in report
    assert "top_cell,max,1.00000000e-09" in report
    assert "abs_cx" in report


def test_append_transport_step_tracers_records_fixture_contract(tmp_path):
    input_path = _write_synthetic_pjc_input(tmp_path / "transport_input.nc")
    tracer_conc = np.arange(4 * 7 * 3 * 2, dtype=np.float64).reshape(4, 7, 3, 2)

    append_transport_step_tracers(input_path, tracer_conc, tracer_names=("A", "B"))

    with netCDF4.Dataset(input_path) as dataset:
        assert dataset.harness == TRANSPORT_INPUT_VERSION
        assert dataset.dimensions["tracer"].size == 2
        assert dataset.dimensions["name_strlen"].size == 1
        assert dataset.variables["tracer_conc"].shape == (4, 7, 3, 2)
        assert np.array_equal(np.asarray(dataset.variables["tracer_conc"][:]), tracer_conc)


def test_read_transport_step_output_records_oracle_fields(tmp_path):
    output_path = tmp_path / "transport_output.nc"
    with netCDF4.Dataset(output_path, "w") as dataset:
        dataset.createDimension("tracer", 1)
        dataset.createDimension("lev", 4)
        dataset.createDimension("lat", 2)
        dataset.createDimension("lon", 3)
        dataset.harness = TRANSPORT_OUTPUT_VERSION
        dataset.createVariable("tracer_conc_after", "f8", ("lev", "lat", "lon", "tracer"))[:] = 1.0
        dataset.createVariable("xmass_hpa", "f8", ("lev", "lat", "lon"))[:] = 2.0
        dataset.createVariable("ymass_hpa", "f8", ("lev", "lat", "lon"))[:] = 3.0
        dataset.createVariable("surface_pressure_hpa", "f8", ("lat", "lon"))[:] = 4.0

    output = read_transport_step_output(output_path)

    assert output.tracer_conc_after.shape == (4, 2, 3, 1)
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
        dataset.createDimension("lev", tracer.shape[0])
        dataset.createDimension("lat", tracer.shape[1])
        dataset.createDimension("lon", tracer.shape[2])
        dataset.createDimension("tracer", tracer.shape[3])
        dataset.harness = TRANSPORT_OUTPUT_VERSION
        dataset.createVariable("tracer_conc_after", "f8", ("lev", "lat", "lon", "tracer"))[:] = tracer + 1.0e-12
        dataset.createVariable("xmass_hpa", "f8", ("lev", "lat", "lon"))[:] = xmass[::-1]
        dataset.createVariable("ymass_hpa", "f8", ("lev", "lat", "lon"))[:] = ymass[::-1]
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


def _require_real_convection_inputs() -> None:
    required = (
        Path("residual_20140901_part001_split01_wombat/run.yml"),
        Path("ExtData/GEOS_2x2.5/MERRA2/2014/09/MERRA2.20140901.A1.2x25.nc4"),
        Path("ExtData/GEOS_2x2.5/MERRA2/2014/09/MERRA2.20140901.A3dyn.2x25.nc4"),
        Path("ExtData/GEOS_2x2.5/MERRA2/2014/09/MERRA2.20140901.A3mstC.2x25.nc4"),
        Path("ExtData/GEOS_2x2.5/MERRA2/2014/09/MERRA2.20140901.A3mstE.2x25.nc4"),
        Path("ExtData/GEOS_2x2.5/MERRA2/2014/09/MERRA2.20140901.I3.2x25.nc4"),
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        pytest.skip(f"real convection fixture inputs are not available: {', '.join(missing)}")


def _history_interval_seconds_for_test(value: str) -> int:
    date, clock = value.split()
    return int(date[6:8]) * 86400 + int(clock[0:2]) * 3600 + int(clock[2:4]) * 60 + int(clock[4:6])


def _write_large_oracle_definition(
    path: Path,
    *,
    fixture_id: str = BASE_INITIAL_TPCORE_FIXTURE_ID,
    input_name: str = "transport_step_input.nc",
    output_name: str = "transport_step_output.nc",
    input_sha: str | None = None,
    output_sha: str | None = None,
    input_size: int | None = None,
    output_size: int | None = None,
) -> None:
    payload = {
        "fixture_id": fixture_id,
        "files": [
            {
                "name": input_name,
                "sha256": input_sha,
                "size_bytes": input_size,
                "url": None,
            },
            {
                "name": output_name,
                "sha256": output_sha,
                "size_bytes": output_size,
                "url": None,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
