from __future__ import annotations

from datetime import datetime

import netCDF4
import numpy as np

from wombat_transport.constants import G0_M_PER_S2
from wombat_transport.fields import TracerField
from wombat_transport.io import FIXED_GRID, initialize_tracers
from wombat_transport.run_config import load_run_config
from wombat_transport.transport import (
    MERRA2_72_AP_HPA,
    MERRA2_72_TO_47_GROUPS,
    MERRA2_72_TO_47_MAPPING,
    advect_horizontal_mass_flux,
    advect_horizontal_upwind,
    advect_vertical_mass_flux,
    dry_air_mass_from_pressure,
    dry_pressure_edges_from_thickness_hpa,
    dry_pressure_thickness_hpa,
    horizontal_mass_flux_hpa,
    load_transport_forcing,
    run_transport_one_step,
    run_transport_window,
    scalar_mass_by_tracer,
    vertical_mass_flux_hpa,
    _map_met_levels_to_47,
)

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
    assert forcing.vertical_mapping == MERRA2_72_TO_47_MAPPING
    assert forcing.a3dyn_path.exists()
    assert forcing.i3_path.exists()
    assert np.all(np.isfinite(forcing.u_m_s))


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


def test_horizontal_mass_fluxes_are_finite_with_closed_southern_edge():
    config = load_run_config(BASE_CONFIG)
    forcing = _load_forcing(config)
    with netCDF4.Dataset(config.grid_template) as dataset:
        hyai = np.asarray(dataset.variables["hyai"][:])
        hybi = np.asarray(dataset.variables["hybi"][:])
    delp = dry_pressure_thickness_hpa(forcing.surface_pressure_pa, hyai, hybi)

    xmass, ymass = horizontal_mass_flux_hpa(
        delp,
        forcing.u_m_s,
        forcing.v_m_s,
        forcing.lat_deg,
        dt_s=600.0,
    )

    assert xmass.shape == delp.shape
    assert ymass.shape == delp.shape
    assert np.all(np.isfinite(xmass))
    assert np.all(np.isfinite(ymass))
    np.testing.assert_array_equal(ymass[:, :, 0, :], 0.0)


def test_zero_wind_transport_leaves_field_and_air_mass_unchanged():
    config = load_run_config(BASE_CONFIG)
    field = initialize_tracers(config.initial_restart, config.species_database)
    forcing = _load_forcing(config)
    with netCDF4.Dataset(config.grid_template) as dataset:
        delp = np.asarray(dataset.variables["Met_DELPDRY"][:])
        area = np.asarray(dataset.variables["AREA"][:])
    dry_air_mass = dry_air_mass_from_pressure(delp, area)
    zero = np.zeros_like(forcing.u_m_s)

    transported, next_air_mass = advect_horizontal_upwind(
        field,
        dry_air_mass,
        zero,
        zero,
        forcing.lat_deg,
        forcing.lon_deg,
        dt_s=600.0,
    )

    np.testing.assert_array_equal(transported.data, field.data)
    np.testing.assert_array_equal(next_air_mass, dry_air_mass)


def test_zero_mass_flux_transport_leaves_field_and_air_mass_unchanged():
    config = load_run_config(BASE_CONFIG)
    field = initialize_tracers(config.initial_restart, config.species_database)
    with netCDF4.Dataset(config.grid_template) as dataset:
        delp = np.asarray(dataset.variables["Met_DELPDRY"][:])
        area = np.asarray(dataset.variables["AREA"][:])
    dry_air_mass = dry_air_mass_from_pressure(delp, area)
    zero = np.zeros_like(delp)

    transported, next_air_mass = advect_horizontal_mass_flux(
        field,
        dry_air_mass,
        zero,
        zero,
        area,
    )

    np.testing.assert_array_equal(transported.data, field.data)
    np.testing.assert_array_equal(next_air_mass, dry_air_mass)


def test_zero_vertical_flux_transport_leaves_field_and_air_mass_unchanged():
    config = load_run_config(BASE_CONFIG)
    field = initialize_tracers(config.initial_restart, config.species_database)
    with netCDF4.Dataset(config.grid_template) as dataset:
        delp = np.asarray(dataset.variables["Met_DELPDRY"][:])
        area = np.asarray(dataset.variables["AREA"][:])
    dry_air_mass = dry_air_mass_from_pressure(delp, area)
    zero = np.zeros((dry_air_mass.shape[0], dry_air_mass.shape[1] + 1, dry_air_mass.shape[2], dry_air_mass.shape[3]))

    transported, next_air_mass = advect_vertical_mass_flux(
        field,
        dry_air_mass,
        zero,
        area,
    )

    np.testing.assert_array_equal(transported.data, field.data)
    np.testing.assert_array_equal(next_air_mass, dry_air_mass)


def test_vertical_mass_flux_closes_boundaries_and_follows_horizontal_tendency():
    previous = np.array([[[[10.0]], [[20.0]], [[30.0]]]])
    horizontal = np.array([[[[12.0]], [[19.0]], [[31.0]]]])
    area = np.array([[1.0]])

    zflux_hpa = vertical_mass_flux_hpa(previous, horizontal, area)
    zflux_kg = zflux_hpa * 100.0 / G0_M_PER_S2

    assert zflux_hpa.shape == (1, 4, 1, 1)
    np.testing.assert_allclose(zflux_kg[:, 0, :, :], 0.0)
    np.testing.assert_allclose(zflux_kg[:, -1, :, :], 0.0)
    assert zflux_kg[0, 1, 0, 0] > 0.0
    np.testing.assert_allclose(zflux_kg[0, 2, 0, 0], 0.0, atol=1e-14)


def test_vertical_mass_flux_transport_reaches_column_fraction_target():
    field = TracerField(
        names=("CO2",),
        data=np.array([[[[[1.0]], [[2.0]], [[3.0]]]]]),
        units=("mol mol-1 dry",),
        coords={},
    )
    previous = np.array([[[[10.0]], [[20.0]], [[30.0]]]])
    horizontal = np.array([[[[12.0]], [[19.0]], [[31.0]]]])
    area = np.array([[1.0]])
    zflux_hpa = vertical_mass_flux_hpa(previous, horizontal, area)

    transported, next_air_mass = advect_vertical_mass_flux(field, horizontal, zflux_hpa, area)

    expected_mass = previous * (np.sum(horizontal, axis=1, keepdims=True) / np.sum(previous, axis=1, keepdims=True))
    np.testing.assert_allclose(next_air_mass, expected_mass, rtol=1e-14)
    np.testing.assert_allclose(
        scalar_mass_by_tracer(transported.data, next_air_mass),
        scalar_mass_by_tracer(field.data, horizontal),
        rtol=1e-14,
    )


def test_uniform_field_stays_uniform_and_conserves_mass_with_real_winds():
    config = load_run_config(BASE_CONFIG)
    field = initialize_tracers(config.initial_restart, config.species_database)
    uniform = TracerField(
        names=field.names,
        data=np.full_like(field.data, 0.0004),
        units=field.units,
        coords=field.coords,
    )
    forcing = _load_forcing(config)
    with netCDF4.Dataset(config.grid_template) as dataset:
        hyai = np.asarray(dataset.variables["hyai"][:])
        hybi = np.asarray(dataset.variables["hybi"][:])
        area = np.asarray(dataset.variables["AREA"][:])
    delp = dry_pressure_thickness_hpa(forcing.surface_pressure_pa, hyai, hybi)
    dry_air_mass = dry_air_mass_from_pressure(delp, area)

    transported, next_air_mass = advect_horizontal_upwind(
        uniform,
        dry_air_mass,
        forcing.u_m_s,
        forcing.v_m_s,
        forcing.lat_deg,
        forcing.lon_deg,
        dt_s=600.0,
    )

    np.testing.assert_allclose(transported.data, uniform.data, rtol=0.0, atol=1e-18)
    np.testing.assert_allclose(np.sum(next_air_mass), np.sum(dry_air_mass), rtol=1e-14)
    np.testing.assert_allclose(
        scalar_mass_by_tracer(transported.data, next_air_mass),
        scalar_mass_by_tracer(uniform.data, dry_air_mass),
        rtol=1e-14,
    )


def test_uniform_field_stays_uniform_with_mass_flux_transport():
    config = load_run_config(BASE_CONFIG)
    field = initialize_tracers(config.initial_restart, config.species_database)
    uniform = TracerField(
        names=field.names,
        data=np.full_like(field.data, 0.0004),
        units=field.units,
        coords=field.coords,
    )
    forcing = _load_forcing(config)
    with netCDF4.Dataset(config.grid_template) as dataset:
        hyai = np.asarray(dataset.variables["hyai"][:])
        hybi = np.asarray(dataset.variables["hybi"][:])
        area = np.asarray(dataset.variables["AREA"][:])
    delp = dry_pressure_thickness_hpa(forcing.surface_pressure_pa, hyai, hybi)
    dry_air_mass = dry_air_mass_from_pressure(delp, area)
    xmass, ymass = horizontal_mass_flux_hpa(
        delp,
        forcing.u_m_s,
        forcing.v_m_s,
        forcing.lat_deg,
        dt_s=600.0,
    )

    transported, next_air_mass = advect_horizontal_mass_flux(
        uniform,
        dry_air_mass,
        xmass,
        ymass,
        area,
    )

    np.testing.assert_allclose(transported.data, uniform.data, rtol=0.0, atol=1e-18)
    np.testing.assert_allclose(np.sum(next_air_mass), np.sum(dry_air_mass), rtol=1e-14)
    np.testing.assert_allclose(
        scalar_mass_by_tracer(transported.data, next_air_mass),
        scalar_mass_by_tracer(uniform.data, dry_air_mass),
        rtol=1e-14,
    )


def test_uniform_field_stays_uniform_with_vertical_mass_flux_transport():
    field = TracerField(
        names=("CO2",),
        data=np.full((1, 1, 3, 1, 1), 0.0004),
        units=("mol mol-1 dry",),
        coords={},
    )
    previous = np.array([[[[10.0]], [[20.0]], [[30.0]]]])
    horizontal = np.array([[[[12.0]], [[19.0]], [[31.0]]]])
    area = np.array([[1.0]])
    zflux_hpa = vertical_mass_flux_hpa(previous, horizontal, area)

    transported, next_air_mass = advect_vertical_mass_flux(field, horizontal, zflux_hpa, area)

    np.testing.assert_allclose(transported.data, field.data, rtol=0.0, atol=1e-18)
    np.testing.assert_allclose(np.sum(next_air_mass), np.sum(horizontal), rtol=1e-14)
    np.testing.assert_allclose(
        scalar_mass_by_tracer(transported.data, next_air_mass),
        scalar_mass_by_tracer(field.data, horizontal),
        rtol=1e-14,
    )


def test_dry_pressure_edges_from_thickness_reconstructs_bottom_to_top_edges():
    delp = np.array([[[[100.0]], [[20.0]], [[5.0]]]])

    edges = dry_pressure_edges_from_thickness_hpa(delp, top_edge_hpa=0.01)

    np.testing.assert_allclose(edges[:, :, 0, 0], [[125.01, 25.01, 5.01, 0.01]])


def test_transport_one_step_conserves_residual_scalar_mass():
    config = load_run_config(RESIDUAL_CONFIG)
    field = initialize_tracers(config.initial_restart, config.species_database, template_path=config.grid_template)
    result = run_transport_one_step(field, _load_forcing(config), config.grid_template, dt_s=600.0)

    assert result.state.shape == field.shape
    assert result.delp_dry_hpa.shape == (1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert result.zmass_hpa.shape == (1, FIXED_GRID["lev"] + 1, FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert np.all(np.isfinite(result.state.data))
    np.testing.assert_allclose(result.final_scalar_mass, result.initial_scalar_mass, rtol=1e-13)


def test_transport_window_accumulates_average_state_and_conserves_mass():
    config = load_run_config(BASE_CONFIG)
    field = initialize_tracers(config.initial_restart, config.species_database)
    result = run_transport_window(
        field,
        config.root / config.transport["met_root"],
        datetime.strptime(config.transport["start"], "%Y-%m-%d %H:%M"),
        config.grid_template,
        steps=2,
        dt_s=600.0,
    )

    assert result.steps == 2
    assert result.state.shape == field.shape
    assert result.average_state.shape == field.shape
    assert result.average_delp_dry_hpa.shape == (1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert np.all(np.isfinite(result.average_state.data))
    np.testing.assert_allclose(result.final_scalar_mass, result.initial_scalar_mass, rtol=1e-13)


def _load_forcing(config):
    from datetime import datetime

    return load_transport_forcing(
        config.root / config.transport["met_root"],
        datetime.strptime(config.transport["start"], "%Y-%m-%d %H:%M"),
        config.grid_template,
        time_index=int(config.transport["met_time_index"]),
    )
