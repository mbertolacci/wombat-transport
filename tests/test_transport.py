from __future__ import annotations

from datetime import datetime

import netCDF4
import numpy as np

from wombat_transport.fields import TracerField
from wombat_transport.fields import (
    canonical_time_slice,
)
from wombat_transport.grid import load_transport_grid
from wombat_transport.io import FIXED_GRID, initialize_tracers
from wombat_transport.run_config import (
    load_run_config,
    meteorology_initial_time_index,
    meteorology_root,
    simulation_start,
    transport_timestep_s,
)
from wombat_transport.transport import (
    MERRA2_72_AP_HPA,
    MERRA2_72_TO_47_GROUPS,
    MERRA2_72_TO_47_MAPPING,
    compute_pbl_height,
    dry_air_mass_from_pressure,
    dry_pressure_edges_from_thickness_hpa,
    dry_pressure_thickness_hpa,
    load_transport_forcing,
    mix_full_pbl,
    run_transport_one_step,
    run_transport_window,
    trace_transport_one_step,
    _map_met_levels_to_47,
)
from wombat_transport.transport.pbl import (
    ZVIR,
    run_vdiffdr_one_step,
)
from wombat_transport.transport.driver import _load_window_forcing

BASE_CONFIG = "base_wombat/run.yml"
RESIDUAL_CONFIG = "residual_20140901_part001_split01_wombat/run.yml"


def test_transport_forcing_loads_merra2_on_47_level_grid():
    config = load_run_config(BASE_CONFIG)
    forcing = _load_forcing(config)

    expected_shape = (1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert forcing.u_m_s.shape == expected_shape
    assert forcing.v_m_s.shape == expected_shape
    assert forcing.omega_pa_s.shape == expected_shape
    assert forcing.specific_humidity_kg_kg.shape == expected_shape
    assert forcing.temperature_k.shape == expected_shape
    assert forcing.surface_pressure_pa.shape == (1, FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert forcing.pbl_height_m.shape == (1, FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert forcing.sensible_heat_flux_w_m2.shape == (1, FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert forcing.latent_heat_flux_w_m2.shape == (1, FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert forcing.friction_velocity_m_s.shape == (1, FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert forcing.vertical_mapping == MERRA2_72_TO_47_MAPPING
    assert forcing.a1_path.exists()
    assert forcing.a3dyn_path.exists()
    assert forcing.i3_path.exists()
    assert np.all(np.isfinite(forcing.u_m_s))


def test_load_transport_grid_reads_template_metadata():
    config = load_run_config(BASE_CONFIG)

    grid = load_transport_grid(config.grid_template)

    assert grid.shape == (FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert grid.area_m2.shape == (FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert grid.hyai_hpa.shape == (FIXED_GRID["lev"] + 1,)
    assert grid.hybi.shape == (FIXED_GRID["lev"] + 1,)
    assert grid.lat_deg.shape == (FIXED_GRID["lat"],)
    assert grid.lon_deg.shape == (FIXED_GRID["lon"],)
    assert grid.lev.shape == (FIXED_GRID["lev"],)
    assert np.all(grid.area_m2 > 0.0)


def test_transport_forcing_accepts_preloaded_grid():
    config = load_run_config(BASE_CONFIG)
    grid = load_transport_grid(config.grid_template)

    forcing = _load_forcing(config, grid=grid)

    np.testing.assert_array_equal(forcing.lat_deg, grid.lat_deg)
    np.testing.assert_array_equal(forcing.lon_deg, grid.lon_deg)


def test_transport_window_forcing_cache_keeps_only_current_met_slice(monkeypatch):
    calls = []

    def fake_load_transport_forcing(met_root, timestamp, grid, *, time_index=0):
        forcing = object()
        calls.append((timestamp, time_index, forcing))
        return forcing

    monkeypatch.setattr("wombat_transport.transport.driver.load_transport_forcing", fake_load_transport_forcing)
    cache = {}
    start = simulation_start(load_run_config(BASE_CONFIG))

    first = _load_window_forcing(cache, "met", start, None, step=0, dt_s=600.0, initial_met_time_index=0)
    same = _load_window_forcing(cache, "met", start, None, step=17, dt_s=600.0, initial_met_time_index=0)
    next_met = _load_window_forcing(cache, "met", start, None, step=18, dt_s=600.0, initial_met_time_index=0)

    assert same is first
    assert next_met is not first
    assert len(calls) == 2
    assert len(cache) == 1
    assert list(cache) == [(datetime(2014, 9, 1), 1)]


def test_met_level_mapping_returns_47_level_inputs_unchanged():
    data = np.arange(1 * 47 * 2 * 3, dtype=np.float64).reshape(1, 47, 2, 3)

    mapped = _map_met_levels_to_47(data)

    assert mapped is data


def test_met_level_mapping_collapses_72_levels_with_pressure_weights():
    levels = np.arange(72, dtype=np.float64)
    data = np.broadcast_to(levels[np.newaxis, :, np.newaxis, np.newaxis], (2, 72, 3, 4)).copy()
    data[1] += 100.0

    mapped = _map_met_levels_to_47(data)

    assert mapped.shape == (2, 47, 3, 4)
    np.testing.assert_array_equal(mapped[:, :36, :, :], data[:, :36, :, :])
    for target_level, (start, end) in enumerate(MERRA2_72_TO_47_GROUPS, start=36):
        weights = MERRA2_72_AP_HPA[start:end] - MERRA2_72_AP_HPA[start + 1 : end + 1]
        expected = np.average(data[:, start:end, :, :], axis=1, weights=weights)
        np.testing.assert_allclose(mapped[:, target_level, :, :], expected, rtol=1e-14)


def test_pressure_bookkeeping_returns_positive_dry_air_mass():
    config = load_run_config(BASE_CONFIG)
    forcing = _load_forcing(config)
    with netCDF4.Dataset(config.grid_template) as dataset:
        hyai = np.asarray(dataset.variables["hyai"][:])
        hybi = np.asarray(dataset.variables["hybi"][:])
        area = np.asarray(dataset.variables["AREA"][:])

    delp = dry_pressure_thickness_hpa(forcing.surface_pressure_pa, hyai, hybi)
    dry_air_mass = dry_air_mass_from_pressure(delp, area)

    assert delp.shape == (1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert dry_air_mass.shape == delp.shape
    assert np.all(delp > 0.0)
    assert np.all(dry_air_mass > 0.0)


def test_dry_pressure_edges_from_thickness_reconstructs_bottom_to_top_edges():
    delp = np.array([[[[100.0]], [[20.0]], [[5.0]]]])

    edges = dry_pressure_edges_from_thickness_hpa(delp, top_edge_hpa=0.01)

    np.testing.assert_allclose(edges[:, :, 0, 0], [[125.01, 25.01, 5.01, 0.01]])


def test_compute_pbl_height_matches_geos_chem_fractional_level_bookkeeping():
    bxheight = np.array(
        [
            [[100.0, 100.0]],
            [[100.0, 100.0]],
            [[100.0, 100.0]],
        ],
        dtype=np.float64,
    )
    pedge = np.array(
        [
            [[1000.0, 1000.0]],
            [[900.0, 900.0]],
            [[800.0, 800.0]],
            [[700.0, 700.0]],
        ],
        dtype=np.float64,
    )
    tv = np.full_like(bxheight, 280.0)
    pblh = np.array([[150.0, 60.0]], dtype=np.float64)

    state = compute_pbl_height(
        pbl_height_m=pblh,
        bxheight_m=bxheight,
        pressure_edges_hpa=pedge,
        virtual_temperature_k=tv,
    )

    expected_first_top = 900.0 * np.exp(-50.0 * 9.80665 / (287.0 * 280.0))
    expected_second_top = 1000.0 * np.exp(-60.0 * 9.80665 / (287.0 * 280.0))
    np.testing.assert_allclose(state.pbl_top_hpa, [[expected_first_top, expected_second_top]])
    np.testing.assert_allclose(state.pbl_thick_hpa, [[1000.0 - expected_first_top, 1000.0 - expected_second_top]])
    np.testing.assert_allclose(np.sum(state.f_of_pbl, axis=0), np.ones((1, 2)))
    np.testing.assert_array_equal(state.in_pbl[:, 0, 0], [True, False, False])
    np.testing.assert_array_equal(state.in_pbl[:, 0, 1], [False, False, False])
    np.testing.assert_allclose(state.pbl_top_l[0, 0], 1.0 + (900.0 - expected_first_top) / 100.0)
    np.testing.assert_allclose(state.pbl_top_l[0, 1], (1000.0 - expected_second_top) / 100.0)
    assert state.pbl_max_l == 2


def test_mix_full_pbl_mass_weights_full_and_fractional_levels():
    tracer = np.array(
        [
            [
                [[1.0]],
                [[3.0]],
                [[9.0]],
            ],
            [
                [[2.0]],
                [[6.0]],
                [[18.0]],
            ],
        ],
        dtype=np.float64,
    )
    dry_mass = np.array(
        [
            [[2.0]],
            [[6.0]],
            [[10.0]],
        ],
        dtype=np.float64,
    )
    pbl_top_l = np.array([[1.5]], dtype=np.float64)

    mixed = mix_full_pbl(tracer, dry_mass, pbl_top_l)

    expected_mean = np.array(
        [
            (1.0 * 2.0 + 3.0 * 6.0 * 0.5) / (2.0 + 6.0 * 0.5),
            (2.0 * 2.0 + 6.0 * 6.0 * 0.5) / (2.0 + 6.0 * 0.5),
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(mixed[:, 0, 0, 0], expected_mean)
    np.testing.assert_allclose(mixed[:, 1, 0, 0], tracer[:, 1, 0, 0] + 0.5 * (expected_mean - tracer[:, 1, 0, 0]))
    np.testing.assert_allclose(mixed[:, 2, 0, 0], tracer[:, 2, 0, 0])

    before_column_mass = np.sum(tracer[:, :, 0, 0] * dry_mass[:, 0, 0], axis=1)
    after_column_mass = np.sum(mixed[:, :, 0, 0] * dry_mass[:, 0, 0], axis=1)
    np.testing.assert_allclose(after_column_mass, before_column_mass)


def test_run_vdiffdr_one_step_preserves_long_lived_mass_with_zero_surface_flux():
    fixture = _synthetic_vdiff_fixture()

    result = run_vdiffdr_one_step(**fixture)

    tracer = fixture["tracer_conc"]
    nlev, nlat, nlon = fixture["u_m_s"].shape
    assert result.tracer_conc.shape == tracer.shape
    assert result.specific_humidity_kg_kg.shape == (nlev, nlat, nlon)
    assert result.kvh_m2_s.shape == (nlev + 1, nlat, nlon)
    assert result.kvm_m2_s.shape == (nlev + 1, nlat, nlon)
    assert result.tpert_k.shape == (nlat, nlon)
    assert result.qpert_kg_kg.shape == (nlat, nlon)
    assert result.negative_count_after_clip == 0
    assert np.all(np.isfinite(result.tracer_conc))
    assert result.tracer_conc.flags.c_contiguous
    assert np.all(result.kvh_m2_s >= 0.0)
    assert np.max(result.kvh_m2_s) > 0.0
    np.testing.assert_allclose(result.final_tracer_mass, result.initial_tracer_mass, rtol=2.0e-14)


def test_run_vdiffdr_one_step_rejects_non_wombat_shapes():
    fixture = _synthetic_vdiff_fixture()
    fixture["pedge_hpa"] = fixture["pedge_hpa"][:-1]

    try:
        run_vdiffdr_one_step(**fixture)
    except ValueError as exc:
        assert "pedge_hpa shape" in str(exc)
    else:
        raise AssertionError("run_vdiffdr_one_step accepted a malformed edge grid")


def test_transport_one_step_conserves_residual_scalar_mass():
    config = load_run_config(RESIDUAL_CONFIG)
    grid = load_transport_grid(config.grid_template)
    field = initialize_tracers(config.initial_restart, config.species_database, template_path=config.grid_template)
    field = TracerField(
        names=field.names[:1],
        data=field.data[..., :1],
        units=field.units[:1],
        coords=field.coords,
    )
    result = run_transport_one_step(field, _load_forcing(config, grid=grid), grid, dt_s=600.0)

    assert result.state.shape == field.shape
    assert result.transport_operators == ("tpcore", "vdiff", "convection")
    assert tuple(stage.operator for stage in result.stage_masses) == ("tpcore", "vdiff", "convection")
    assert result.delp_dry_hpa.shape == (1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert result.zmass_hpa.shape == (1, FIXED_GRID["lev"] + 1, FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert np.all(np.isfinite(result.state.data))
    np.testing.assert_allclose(result.final_scalar_mass, result.initial_scalar_mass, rtol=1e-13)


def test_trace_transport_one_step_captures_operator_handoffs():
    config = load_run_config(RESIDUAL_CONFIG)
    grid = load_transport_grid(config.grid_template)
    field = initialize_tracers(config.initial_restart, config.species_database, template_path=config.grid_template)
    field = TracerField(
        names=field.names[:1],
        data=field.data[..., :1],
        units=field.units[:1],
        coords=field.coords,
    )

    trace = trace_transport_one_step(field, _load_forcing(config, grid=grid), grid, dt_s=600.0)

    assert trace.result.transport_operators == ("tpcore", "vdiff", "convection")
    assert trace.tpcore_state.tracer_conc_after.shape == (FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"], 1)
    assert trace.vdiff_input.tracer_conc.shape == trace.tpcore_state.tracer_conc_after.shape
    assert trace.vdiff_output.tracer_conc.shape == trace.convection_input.tracer_conc.shape
    canonical_result = canonical_time_slice(trace.result.state.data)
    assert trace.convection_output.tracer_conc.shape == canonical_result.shape
    np.testing.assert_allclose(canonical_result, trace.convection_output.tracer_conc)


def test_transport_window_accumulates_average_state_and_conserves_mass():
    config = load_run_config(BASE_CONFIG)
    grid = load_transport_grid(config.grid_template)
    field = initialize_tracers(config.initial_restart, config.species_database)
    result = run_transport_window(
        field,
        meteorology_root(config),
        simulation_start(config),
        grid,
        steps=2,
        dt_s=transport_timestep_s(config),
    )

    assert result.steps == 2
    assert result.transport_operators == ("tpcore", "vdiff", "convection")
    assert tuple(stage.operator for stage in result.stage_masses) == ("tpcore", "vdiff", "convection")
    assert result.state.shape == field.shape
    assert result.average_state.shape == field.shape
    assert result.average_delp_dry_hpa.shape == (1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert np.all(np.isfinite(result.average_state.data))
    np.testing.assert_allclose(result.final_scalar_mass, result.initial_scalar_mass, rtol=1e-13)


def test_transport_forcing_loads_convection_fields_on_target_grid():
    config = load_run_config(BASE_CONFIG)
    forcing = _load_forcing(config)

    center_shape = (1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"])
    horizontal_shape = (1, FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert forcing.convective_mass_flux_kg_m2_s.shape == center_shape
    assert forcing.convective_detrainment_kg_m2_s.shape == center_shape
    assert forcing.convective_precip_prod_kg_kg_s.shape == center_shape
    assert forcing.convective_precip_reevap_kg_kg_s.shape == center_shape
    assert forcing.convective_ice_flux_kg_m2_s.shape == center_shape
    assert forcing.convective_liquid_flux_kg_m2_s.shape == center_shape
    assert forcing.convective_precip_mm_day.shape == horizontal_shape
    assert np.max(np.abs(forcing.convective_mass_flux_kg_m2_s)) > 0.0


def _load_forcing(config, *, grid=None):
    if grid is None:
        grid = load_transport_grid(config.grid_template)
    return load_transport_forcing(
        meteorology_root(config),
        simulation_start(config),
        grid,
        time_index=meteorology_initial_time_index(config),
    )


def _synthetic_vdiff_fixture():
    nlev = 47
    nlat = 3
    nlon = 4
    ntracer = 2
    lev = np.arange(nlev, dtype=np.float64)[:, np.newaxis, np.newaxis]
    lat = np.linspace(-1.0, 1.0, nlat, dtype=np.float64)[np.newaxis, :, np.newaxis]
    lon = np.linspace(0.0, 1.0, nlon, dtype=np.float64)[np.newaxis, np.newaxis, :]
    tracer_index = np.arange(ntracer, dtype=np.float64)[np.newaxis, np.newaxis, np.newaxis, :]

    pedge_profile = np.linspace(50.0, 1000.0, nlev + 1, dtype=np.float64)
    pedge = np.broadcast_to(pedge_profile[:, np.newaxis, np.newaxis], (nlev + 1, nlat, nlon)).copy()
    pmid = 0.5 * (pedge[:-1] + pedge[1:])
    temperature = 289.0 - 0.45 * lev + 1.5 * lat + 0.2 * lon
    sphu = 0.010 * np.exp(-lev / 18.0) * (1.0 + 0.03 * lat) * np.ones((1, 1, nlon), dtype=np.float64)
    tv = temperature * (1.0 + ZVIR * sphu)
    bxheight = np.full((nlev, nlat, nlon), 125.0, dtype=np.float64)
    dry_mass = (pedge[1:] - pedge[:-1]) * 100.0 / 9.80665
    dry_mass = dry_mass * np.ones((nlev, nlat, nlon), dtype=np.float64)
    u = (4.0 + 0.05 * lev + 0.2 * lon) * np.ones((1, nlat, 1), dtype=np.float64)
    v = (0.3 * np.sin((lev + 1.0) / nlev * np.pi) + 0.02 * lat) * np.ones((1, 1, nlon), dtype=np.float64)
    tracer = 4.0e-4 + 1.0e-7 * tracer_index
    tracer = tracer + 4.0e-9 * lev[..., np.newaxis]
    tracer = tracer + 2.0e-9 * lat[..., np.newaxis] + 1.0e-9 * lon[..., np.newaxis]

    return {
        "tracer_conc": tracer,
        "u_m_s": u,
        "v_m_s": v,
        "temperature_k": temperature,
        "specific_humidity_kg_kg": sphu,
        "pmid_hpa": pmid,
        "pedge_hpa": pedge,
        "virtual_temperature_k": tv,
        "bxheight_m": bxheight,
        "dry_air_mass_kg": dry_mass,
        "pbl_top_m": np.full((nlat, nlon), 950.0, dtype=np.float64),
        "hflux_w_m2": np.full((nlat, nlon), 65.0, dtype=np.float64),
        "eflux_w_m2": np.full((nlat, nlon), 90.0, dtype=np.float64),
        "ustar_m_s": np.full((nlat, nlon), 0.35, dtype=np.float64),
        "area_m2": np.ones((nlat, nlon), dtype=np.float64),
        "dt_s": 600.0,
    }
