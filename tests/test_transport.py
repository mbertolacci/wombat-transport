from __future__ import annotations

import netCDF4
import numpy as np

from wombat_transport.fields import TracerField
from wombat_transport.io import FIXED_GRID, initialize_tracers
from wombat_transport.run_config import load_run_config
from wombat_transport.transport import (
    advect_horizontal_upwind,
    dry_air_mass_from_pressure,
    dry_pressure_thickness_hpa,
    load_transport_forcing,
    run_transport_one_step,
    scalar_mass_by_tracer,
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
    assert forcing.vertical_mapping == "lowest_47_of_72"
    assert forcing.a3dyn_path.exists()
    assert forcing.i3_path.exists()
    assert np.all(np.isfinite(forcing.u_m_s))


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


def test_transport_one_step_conserves_residual_scalar_mass():
    config = load_run_config(RESIDUAL_CONFIG)
    field = initialize_tracers(config.initial_restart, config.species_database, template_path=config.grid_template)
    result = run_transport_one_step(field, _load_forcing(config), config.grid_template, dt_s=600.0)

    assert result.state.shape == field.shape
    assert result.delp_dry_hpa.shape == (1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert np.all(np.isfinite(result.state.data))
    np.testing.assert_allclose(result.final_scalar_mass, result.initial_scalar_mass, rtol=1e-13)


def _load_forcing(config):
    from datetime import datetime

    return load_transport_forcing(
        config.root / config.transport["met_root"],
        datetime.strptime(config.transport["start"], "%Y-%m-%d %H:%M"),
        config.grid_template,
        time_index=int(config.transport["met_time_index"]),
    )
