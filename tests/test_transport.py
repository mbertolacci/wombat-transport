from __future__ import annotations

from datetime import datetime
from datetime import timedelta
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from wombat_transport.fields import TracerField
from wombat_transport.emissions import SurfaceEmissions
from wombat_transport.fields import (
    canonical_time_slice,
)
from wombat_transport.grid import geos_chem_grid_cell_area_m2
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
    TransportExecutor,
    compute_transport_stage_masses,
    compute_pbl_height,
    dry_air_mass_from_pressure,
    dry_pressure_edges_from_thickness_hpa,
    dry_pressure_thickness_from_surface_hpa,
    dry_surface_pressure_hpa,
    load_transport_forcing,
    mix_full_pbl,
    run_transport_one_step,
    run_transport_step_with_executor,
    run_transport_window,
    trace_transport_one_step,
    wet_surface_pressure_hpa,
    _map_met_levels_to_47,
)
import wombat_transport.transport.forcing as forcing_module
import wombat_transport.transport.pbl._kernels as pbl_numba
import wombat_transport.transport.pbl._operator as pbl_operator
from tests.data_paths import BASE_CONFIG, RESIDUAL_CONFIG, requires_restart, requires_transport_data
from wombat_transport.transport.pbl import (
    ZVIR,
    run_vdiffdr_one_step,
)
from wombat_transport.transport.driver import _load_window_forcing, _surface_flux_from_active_emissions

@requires_transport_data
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
    assert forcing.surface_pressure_start_pa.shape == (1, FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert forcing.restart_surface_pressure_pa.shape == (1, FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert forcing.dry_surface_pressure_start_hpa.shape == (1, FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert forcing.dry_surface_pressure_hpa.shape == (1, FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert forcing.restart_dry_surface_pressure_hpa.shape == (1, FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert forcing.i3_start_wet_surface_pressure_hpa.shape == (1, FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert forcing.i3_start_dry_surface_pressure_hpa.shape == (1, FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert forcing.i3_start_specific_humidity_kg_kg.shape == expected_shape
    assert forcing.i3_start_temperature_k.shape == expected_shape
    assert forcing.pbl_height_m.shape == (1, FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert forcing.sensible_heat_flux_w_m2.shape == (1, FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert forcing.latent_heat_flux_w_m2.shape == (1, FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert forcing.friction_velocity_m_s.shape == (1, FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert forcing.vertical_mapping == MERRA2_72_TO_47_MAPPING
    assert forcing.a1_path.exists()
    assert forcing.a3dyn_path.exists()
    assert forcing.i3_path.exists()
    assert np.all(np.isfinite(forcing.u_m_s))


def test_transport_forcing_provider_uses_block_cadences(monkeypatch):
    calls = []
    grid = _fake_grid()

    def fake_a1_block(met_root, start_day, start_index, count, grid):
        calls.append(("A1", start_index, count))
        return _fake_a1_block(start_index, count)

    def fake_a3_block(met_root, start_day, start_index, count, grid):
        calls.append(("A3", start_index, count))
        return _fake_a3_block(start_index, count)

    def fake_i3_block(met_root, start_day, start_index, count, grid):
        calls.append(("I3", start_index, count))
        return _fake_i3_block(start_index, count)

    monkeypatch.setattr("wombat_transport.transport.forcing._load_a1_block", fake_a1_block)
    monkeypatch.setattr("wombat_transport.transport.forcing._load_a3_block", fake_a3_block)
    monkeypatch.setattr("wombat_transport.transport.forcing._load_i3_block", fake_i3_block)

    start = datetime(2014, 9, 1)
    provider = forcing_module.TransportForcingProvider(
        "met",
        start,
        grid,  # type: ignore[arg-type]
    )
    provider.forcing_for_step(start + timedelta(hours=3), dt_s=600.0)
    provider.forcing_for_step(start + timedelta(hours=3, minutes=10), dt_s=600.0)
    provider.forcing_for_step(start + timedelta(hours=23, minutes=50), dt_s=600.0)

    assert calls == [
        ("A1", 0, 24),
        ("A3", 0, 4),
        ("I3", 0, 4),
        ("A3", 4, 4),
        ("I3", 4, 4),
    ]


def test_transport_forcing_provider_interpolates_i3_like_geos_chem(monkeypatch):
    grid = _fake_grid()

    monkeypatch.setattr("wombat_transport.transport.forcing._load_a1_block", lambda *args: _fake_a1_block(0, 24))
    monkeypatch.setattr("wombat_transport.transport.forcing._load_a3_block", lambda *args: _fake_a3_block(0, 4))
    monkeypatch.setattr("wombat_transport.transport.forcing._load_i3_block", lambda *args: _fake_i3_block(0, 4))

    start = datetime(2014, 9, 1)
    forcing = forcing_module.TransportForcingProvider(
        "met",
        start,
        grid,  # type: ignore[arg-type]
    ).forcing_for_step(
        start + timedelta(hours=1),
        dt_s=600.0,
    )

    np.testing.assert_allclose(forcing.surface_pressure_start_pa, 1.0 / 3.0)
    np.testing.assert_allclose(forcing.surface_pressure_pa, 4200.0 / 10800.0)
    np.testing.assert_allclose(forcing.i3_start_specific_humidity_kg_kg, 0.0)
    np.testing.assert_allclose(forcing.specific_humidity_kg_kg, 3900.0 / 10800.0)
    np.testing.assert_allclose(forcing.i3_start_temperature_k, 0.0)
    np.testing.assert_allclose(forcing.temperature_k, 3900.0 / 10800.0)
    np.testing.assert_allclose(forcing.restart_surface_pressure_pa, 0.0)


def test_transport_forcing_rejects_step_crossing_i3_boundary(monkeypatch):
    grid = _fake_grid()
    monkeypatch.setattr("wombat_transport.transport.forcing._load_a1_block", lambda *args: _fake_a1_block(0, 24))
    monkeypatch.setattr("wombat_transport.transport.forcing._load_a3_block", lambda *args: _fake_a3_block(0, 4))
    monkeypatch.setattr("wombat_transport.transport.forcing._load_i3_block", lambda *args: _fake_i3_block(0, 4))
    start = datetime(2014, 9, 1)
    provider = forcing_module.TransportForcingProvider(
        "met",
        start,
        grid,  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="crosses a three-hour"):
        provider.forcing_for_step(start + timedelta(hours=2, minutes=55), dt_s=600.0)


@requires_restart
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


@requires_restart
def test_load_transport_grid_uses_geos_chem_area_formula():
    config = load_run_config(BASE_CONFIG)

    grid = load_transport_grid(config.grid_template)
    with netCDF4.Dataset(config.grid_template) as dataset:
        template_area = np.asarray(dataset.variables["AREA"][:], dtype=np.float64)

    expected = geos_chem_grid_cell_area_m2(grid.lat_deg, grid.lon_deg)
    np.testing.assert_allclose(grid.area_m2, expected, rtol=0.0, atol=0.0)
    assert not np.allclose(grid.area_m2, template_area, rtol=0.0, atol=1.0)
    np.testing.assert_allclose(grid.area_m2[45, 0], 6.18185596759564e10)
    np.testing.assert_allclose(grid.area_m2[0, 0], 2.697411986535481e8)


@requires_transport_data
def test_transport_forcing_accepts_preloaded_grid():
    config = load_run_config(BASE_CONFIG)
    grid = load_transport_grid(config.grid_template)

    forcing = _load_forcing(config, grid=grid)

    np.testing.assert_array_equal(forcing.lat_deg, grid.lat_deg)
    np.testing.assert_array_equal(forcing.lon_deg, grid.lon_deg)


@requires_transport_data
def test_transport_forcing_provider_chunk_multiple_is_numerically_equivalent():
    config = load_run_config(BASE_CONFIG)
    grid = load_transport_grid(config.grid_template)
    met_root = meteorology_root(config)
    start = simulation_start(config)
    dt_s = float(transport_timestep_s(config))
    provider_one = forcing_module.TransportForcingProvider(
        met_root,
        start,
        grid,
        initial_met_time_index=meteorology_initial_time_index(config),
        chunk_multiple=1,
    )
    provider_two = forcing_module.TransportForcingProvider(
        met_root,
        start,
        grid,
        initial_met_time_index=meteorology_initial_time_index(config),
        chunk_multiple=2,
    )

    for current in (
        start,
        start + timedelta(hours=1),
        start + timedelta(hours=12),
        start + timedelta(hours=23, minutes=50),
        start + timedelta(hours=24),
    ):
        forcing_one = provider_one.forcing_for_step(current, dt_s=dt_s)
        forcing_two = provider_two.forcing_for_step(current, dt_s=dt_s)
        for name in (
            "u_m_s",
            "v_m_s",
            "omega_pa_s",
            "surface_pressure_start_pa",
            "surface_pressure_pa",
            "dry_surface_pressure_start_hpa",
            "dry_surface_pressure_hpa",
            "specific_humidity_kg_kg",
            "temperature_k",
            "pbl_height_m",
            "convective_mass_flux_kg_m2_s",
        ):
            np.testing.assert_array_equal(getattr(forcing_one, name), getattr(forcing_two, name))


def test_transport_window_uses_forcing_provider_timestamps():
    calls = []

    class FakeProvider:
        def __init__(self, start):
            self.start = start

        def forcing_for_step(self, current, *, dt_s):
            forcing = object()
            calls.append((current, dt_s, forcing))
            return forcing

    start = simulation_start(load_run_config(BASE_CONFIG))
    provider = FakeProvider(start)

    first = _load_window_forcing(provider, step=0, dt_s=600.0)
    same = _load_window_forcing(provider, step=17, dt_s=600.0)
    next_met = _load_window_forcing(provider, step=18, dt_s=600.0)

    assert first is calls[0][2]
    assert same is calls[1][2]
    assert next_met is calls[2][2]
    assert [call[0] for call in calls] == [
        start,
        start + timedelta(minutes=170),
        start + timedelta(hours=3),
    ]


def _fake_a1_fields(value: float):
    data2 = np.full((1, 1, 1), float(value), dtype=np.float64)
    return SimpleNamespace(
        pblh=data2,
        hflux=data2,
        eflux=data2,
        ustar=data2,
        precccon=data2,
        path=Path("A1.nc4"),
    )


def _fake_a3_fields(value: float):
    data3 = np.full((1, 1, 1, 1), float(value), dtype=np.float64)
    edge = np.full((2, 1, 1), float(value), dtype=np.float64)
    return SimpleNamespace(
        u=data3,
        v=data3,
        omega=data3,
        dtrain=data3,
        dqrcu=data3,
        reevapcn=data3,
        cmfmc=edge,
        pficu=edge,
        pflcu=edge,
        a3dyn_path=Path("A3dyn.nc4"),
        a3mstc_path=Path("A3mstC.nc4"),
        a3mste_path=Path("A3mstE.nc4"),
    )


def _fake_i3_fields(value: float):
    data2 = np.full((1, 1, 1), float(value), dtype=np.float64)
    data3 = np.full((1, 1, 1, 1), float(value), dtype=np.float64)
    return SimpleNamespace(surface_pressure=data2, qv=data3, temperature=data3, path=Path("I3.nc4"))


def _fake_a1_block(start_index: int, count: int):
    values = np.arange(start_index, start_index + count, dtype=np.float64).reshape(count, 1, 1)
    paths = tuple(Path(f"A1_{index}.nc4") for index in range(start_index, start_index + count))
    return forcing_module._A1Block(
        start_index=start_index,
        count=count,
        pblh=values,
        hflux=values,
        eflux=values,
        ustar=values,
        precccon=values,
        paths=paths,
    )


def _fake_a3_block(start_index: int, count: int):
    values = np.arange(start_index, start_index + count, dtype=np.float64).reshape(count, 1, 1, 1)
    edges = np.arange(start_index, start_index + count, dtype=np.float64).reshape(count, 1, 1, 1)
    edges = np.repeat(edges, 2, axis=1)
    paths = tuple(Path(f"A3_{index}.nc4") for index in range(start_index, start_index + count))
    return forcing_module._A3Block(
        start_index=start_index,
        count=count,
        u=values,
        v=values,
        omega=values,
        dtrain=values,
        dqrcu=values,
        reevapcn=values,
        cmfmc=edges,
        pficu=edges,
        pflcu=edges,
        a3dyn_paths=paths,
        a3mstc_paths=paths,
        a3mste_paths=paths,
    )


def _fake_i3_block(start_index: int, count: int):
    read_count = count + 1
    values2 = np.arange(start_index, start_index + read_count, dtype=np.float64).reshape(read_count, 1, 1)
    values3 = np.arange(start_index, start_index + read_count, dtype=np.float64).reshape(read_count, 1, 1, 1)
    paths = tuple(Path(f"I3_{index}.nc4") for index in range(start_index, start_index + read_count))
    return forcing_module._I3Block(
        start_index=start_index,
        count=count,
        surface_pressure=values2,
        qv=values3,
        temperature=values3,
        dry_surface_pressure_hpa=values2,
        wet_surface_pressure_hpa=values2,
        paths=paths,
    )


def _fake_grid():
    return SimpleNamespace(
        lat_deg=np.array([0.0]),
        lon_deg=np.array([0.0]),
        area_m2=np.array([[1.0]]),
        hyai_hpa=np.array([0.0, 0.0]),
        hybi=np.array([1.0, 0.0]),
    )


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


def test_dry_surface_pressure_reconstructs_geos_chem_style_column():
    wet_ps = np.array([[[100000.0, 90000.0], [80000.0, 70000.0], [60000.0, 50000.0]]])
    q = np.full((1, 2, 3, 2), 0.01)
    hyai = np.array([0.0, 0.0, 0.0])
    hybi = np.array([1.0, 0.5, 0.0])

    dry_ps = dry_surface_pressure_hpa(wet_ps, q, hyai, hybi)
    delp = dry_pressure_thickness_from_surface_hpa(dry_ps, hyai, hybi)
    expected = wet_ps / 100.0 * 0.99
    expected_wet = wet_ps / 100.0

    np.testing.assert_allclose(dry_ps, expected)
    np.testing.assert_allclose(np.sum(delp, axis=1), dry_ps)
    np.testing.assert_allclose(wet_surface_pressure_hpa(wet_ps), expected_wet)


@requires_restart
def test_pressure_bookkeeping_returns_positive_dry_air_mass():
    config = load_run_config(BASE_CONFIG)
    forcing = _load_forcing(config)
    with netCDF4.Dataset(config.grid_template) as dataset:
        hyai = np.asarray(dataset.variables["hyai"][:])
        hybi = np.asarray(dataset.variables["hybi"][:])
        area = np.asarray(dataset.variables["AREA"][:])

    delp = dry_pressure_thickness_from_surface_hpa(forcing.dry_surface_pressure_hpa, hyai, hybi)
    dry_air_mass = dry_air_mass_from_pressure(delp, area)

    assert delp.shape == (1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert dry_air_mass.shape == delp.shape
    assert np.all(delp > 0.0)
    assert np.all(dry_air_mass > 0.0)
    assert np.all(forcing.dry_surface_pressure_hpa <= forcing.wet_surface_pressure_hpa)


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


def test_run_vdiffdr_one_step_preserves_long_lived_mass_with_zero_surface_flux(transport_numba_mode):
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


@pytest.mark.parametrize(("workers", "offset", "negative_count"), ((1, 1.0, 7), (2, 3.0, 11)))
def test_run_vdiffdr_one_step_diagnostics_light_uses_shared_block_path(
    monkeypatch, workers, offset, negative_count
):
    fixture = _synthetic_vdiff_fixture()
    calls = []

    def fake_block_path(**kwargs):
        calls.append(kwargs)
        empty = np.empty((0,), dtype=np.float64)
        return SimpleNamespace(
            tracer_conc=kwargs["tracer_top"] + offset,
            specific_humidity_kg_kg=kwargs["sphu_top"] + offset + 1.0,
            kvh_m2_s=empty,
            kvm_m2_s=empty,
            pbl_top_m=kwargs["pblh_m"].copy(),
            tpert_k=empty,
            qpert_kg_kg=empty,
            negative_count_before_clip=negative_count,
            negative_count_after_clip=0,
            initial_tracer_mass=empty,
            final_tracer_mass=empty,
        )

    monkeypatch.setattr(pbl_numba, "_NUMBA_AVAILABLE", True)
    monkeypatch.setenv("WOMBAT_NUMBA_THREADS", str(workers))
    monkeypatch.setattr(pbl_operator, "run_vdiff_one_block_compiled", fake_block_path)

    result = run_vdiffdr_one_step(**fixture, diagnostics=False)

    assert len(calls) == 1
    assert calls[0]["workers"] == workers
    assert calls[0]["diagnostics"] is False
    np.testing.assert_array_equal(calls[0]["surface_flux_kg_m2_s"], 0.0)
    np.testing.assert_allclose(result.tracer_conc, fixture["tracer_conc"] + offset)
    np.testing.assert_allclose(
        result.specific_humidity_kg_kg,
        fixture["specific_humidity_kg_kg"] + offset + 1.0,
    )
    assert result.negative_count_before_clip == negative_count
    assert result.negative_count_after_clip == 0
    assert result.kvh_m2_s.shape == (0,)
    assert result.kvm_m2_s.shape == (0,)
    assert result.initial_tracer_mass.shape == (0,)
    assert result.final_tracer_mass.shape == (0,)


def test_run_vdiffdr_one_step_reuses_light_output_only_when_requested(monkeypatch):
    fixture = _synthetic_vdiff_fixture()
    monkeypatch.setenv("WOMBAT_NUMBA_THREADS", "1")

    first = run_vdiffdr_one_step(**fixture, diagnostics=False, reuse_output=True)
    second = run_vdiffdr_one_step(**fixture, diagnostics=False, reuse_output=True)
    fresh = run_vdiffdr_one_step(**fixture, diagnostics=False)

    assert second.tracer_conc is first.tracer_conc
    assert second.specific_humidity_kg_kg is first.specific_humidity_kg_kg
    assert fresh.tracer_conc is not second.tracer_conc
    assert fresh.specific_humidity_kg_kg is not second.specific_humidity_kg_kg


def test_run_vdiffdr_one_step_avoids_aliasing_reused_input_and_output(monkeypatch):
    fixture = _synthetic_vdiff_fixture()
    monkeypatch.setenv("WOMBAT_NUMBA_THREADS", "1")

    first = run_vdiffdr_one_step(**fixture, diagnostics=False, reuse_output=True)
    second = run_vdiffdr_one_step(
        **{**fixture, "tracer_conc": first.tracer_conc},
        diagnostics=False,
        reuse_output=True,
    )

    assert not np.shares_memory(second.tracer_conc, first.tracer_conc)


def test_run_vdiffdr_one_step_uses_owned_output_buffer(monkeypatch):
    fixture = _synthetic_vdiff_fixture()
    monkeypatch.setenv("WOMBAT_NUMBA_THREADS", "1")
    output = np.empty_like(fixture["tracer_conc"])

    expected = run_vdiffdr_one_step(**fixture, diagnostics=False)
    result = run_vdiffdr_one_step(
        **fixture,
        diagnostics=False,
        reuse_output=True,
        output_buffer=output,
    )

    assert result.tracer_conc is output
    np.testing.assert_array_equal(result.tracer_conc, expected.tracer_conc)


def test_run_vdiffdr_one_step_accepts_deferred_tpcore_pressure_mass(monkeypatch):
    fixture = _synthetic_vdiff_fixture()
    monkeypatch.setenv("WOMBAT_NUMBA_THREADS", "1")
    tracer = fixture["tracer_conc"]
    pressure_mass = fixture["pmid_hpa"] * 0.031 + 1.0
    tracer_mass = tracer * pressure_mass[..., np.newaxis]
    finalized = tracer_mass * (1.0 / pressure_mass)[..., np.newaxis]
    finalized[finalized < 0.0] = 1.0e-26
    finalized[:, 1, :, :] = finalized[:, 0, :, :]
    finalized[:, -2, :, :] = finalized[:, -1, :, :]

    expected = run_vdiffdr_one_step(
        **{**fixture, "tracer_conc": finalized},
        diagnostics=False,
    )
    result = run_vdiffdr_one_step(
        **{**fixture, "tracer_conc": tracer_mass},
        diagnostics=False,
        input_mass_pressure_hpa=pressure_mass,
    )

    np.testing.assert_array_equal(result.tracer_conc, expected.tracer_conc)
    np.testing.assert_array_equal(result.specific_humidity_kg_kg, expected.specific_humidity_kg_kg)

    surface_flux = np.full((tracer.shape[1], tracer.shape[2], tracer.shape[3]), 1.0e-15)
    expected_flux = run_vdiffdr_one_step(
        **{**fixture, "tracer_conc": finalized, "surface_flux_kg_m2_s": surface_flux},
        diagnostics=False,
    )
    result_flux = run_vdiffdr_one_step(
        **{**fixture, "tracer_conc": tracer_mass, "surface_flux_kg_m2_s": surface_flux},
        diagnostics=False,
        input_mass_pressure_hpa=pressure_mass,
    )
    np.testing.assert_array_equal(result_flux.tracer_conc, expected_flux.tracer_conc)


def test_run_vdiffdr_one_step_rejects_non_wombat_shapes():
    fixture = _synthetic_vdiff_fixture()
    fixture["pedge_hpa"] = fixture["pedge_hpa"][:-1]

    try:
        run_vdiffdr_one_step(**fixture)
    except ValueError as exc:
        assert "pedge_hpa shape" in str(exc)
    else:
        raise AssertionError("run_vdiffdr_one_step accepted a malformed edge grid")


@requires_transport_data
def test_transport_one_step_runs_residual_operator_chain(transport_numba_mode):
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
    assert result.delp_dry_hpa.shape == (1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert result.specific_humidity_kg_kg.shape == result.delp_dry_hpa.shape
    assert result.xmass_hpa is None
    assert result.ymass_hpa is None
    assert result.zmass_hpa is None
    assert np.all(np.isfinite(result.state.data))

    diagnostic_result = run_transport_one_step(
        field,
        _load_forcing(config, grid=grid),
        grid,
        dt_s=600.0,
        include_flux_diagnostics=True,
    )
    assert diagnostic_result.zmass_hpa.shape == (1, FIXED_GRID["lev"] + 1, FIXED_GRID["lat"], FIXED_GRID["lon"])


@requires_transport_data
@pytest.mark.parametrize("block_width", (8, 24))
def test_spatial_transport_over_blocks_matches_single_field(
    monkeypatch, block_width, transport_numba_mode
):
    monkeypatch.setenv("WOMBAT_NUMBA_THREADS", "2")
    config = load_run_config(RESIDUAL_CONFIG)
    grid = load_transport_grid(config.grid_template)
    field = initialize_tracers(
        config.initial_restart,
        config.species_database,
        template_path=config.grid_template,
    )
    forcing = _load_forcing(config, grid=grid)

    expected = run_transport_one_step(field, forcing, grid, dt_s=600.0)
    blocked = field.reblock(block_width)
    actual = run_transport_one_step(blocked, forcing, grid, dt_s=600.0)

    np.testing.assert_array_equal(actual.state.to_canonical(), expected.state.data)
    np.testing.assert_array_equal(actual.dry_air_mass_kg, expected.dry_air_mass_kg)
    np.testing.assert_array_equal(actual.specific_humidity_kg_kg, expected.specific_humidity_kg_kg)

@requires_transport_data
@pytest.mark.parametrize("execution", ("spatial", "blocks"))
def test_transport_executor_matches_single_field(monkeypatch, execution):
    monkeypatch.setenv("WOMBAT_NUMBA", "1")
    monkeypatch.setenv("WOMBAT_NUMBA_THREADS", "2")
    config = load_run_config(RESIDUAL_CONFIG)
    grid = load_transport_grid(config.grid_template)
    field = initialize_tracers(
        config.initial_restart,
        config.species_database,
        template_path=config.grid_template,
    )
    forcing = _load_forcing(config, grid=grid)

    expected = run_transport_one_step(field, forcing, grid, dt_s=600.0)
    blocked = field.reblock(8)
    executor = TransportExecutor.create(blocked)
    actual = run_transport_step_with_executor(
        blocked, forcing, grid, executor, dt_s=600.0, execution=execution
    )

    np.testing.assert_allclose(
        actual.state.to_canonical(),
        expected.state.data,
        rtol=3.0e-16,
        atol=2.0e-19,
    )
    np.testing.assert_array_equal(actual.dry_air_mass_kg, expected.dry_air_mass_kg)
    np.testing.assert_array_equal(actual.specific_humidity_kg_kg, expected.specific_humidity_kg_kg)

    expected_next = run_transport_one_step(
        expected.state,
        forcing,
        grid,
        dt_s=600.0,
        dry_air_mass_kg=expected.dry_air_mass_kg,
    )
    actual_next = run_transport_step_with_executor(
        actual.state,
        forcing,
        grid,
        executor,
        dt_s=600.0,
        dry_air_mass_kg=actual.dry_air_mass_kg,
        execution=execution,
    )
    np.testing.assert_allclose(
        actual_next.state.to_canonical(),
        expected_next.state.data,
        rtol=3.0e-16,
        atol=2.0e-19,
    )


@pytest.mark.parametrize("tracer_count", (1, 24))
@requires_transport_data
def test_transport_one_step_numpy_numba_parity(monkeypatch, tracer_count):
    if importlib.util.find_spec("numba") is None:
        pytest.skip("numba is not available")
    config = load_run_config(RESIDUAL_CONFIG)
    grid = load_transport_grid(config.grid_template)
    initialized = initialize_tracers(
        config.initial_restart,
        config.species_database,
        template_path=config.grid_template,
    )
    field = TracerField(
        names=initialized.names[:tracer_count],
        data=np.ascontiguousarray(initialized.data[..., :tracer_count]),
        units=initialized.units[:tracer_count],
        coords=initialized.coords,
    )
    forcing = _load_forcing(config, grid=grid)

    monkeypatch.setenv("WOMBAT_NUMBA", "0")
    numpy_result = run_transport_one_step(field, forcing, grid, dt_s=600.0)
    monkeypatch.setenv("WOMBAT_NUMBA", "1")
    monkeypatch.setenv("WOMBAT_NUMBA_THREADS", "1")
    numba_result = run_transport_one_step(field, forcing, grid, dt_s=600.0)

    np.testing.assert_allclose(numba_result.state.data, numpy_result.state.data, rtol=2.0e-14, atol=1.0e-20)
    np.testing.assert_array_equal(numba_result.delp_dry_hpa, numpy_result.delp_dry_hpa)


@requires_transport_data
def test_transport_one_step_consumes_input_only_when_requested(monkeypatch):
    monkeypatch.setenv("WOMBAT_NUMBA", "1")
    monkeypatch.setenv("WOMBAT_NUMBA_THREADS", "1")
    config = load_run_config(RESIDUAL_CONFIG)
    grid = load_transport_grid(config.grid_template)
    initialized = initialize_tracers(
        config.initial_restart,
        config.species_database,
        template_path=config.grid_template,
    )
    original = np.ascontiguousarray(initialized.data[..., :1])
    forcing = _load_forcing(config, grid=grid)

    safe_field = TracerField(
        names=initialized.names[:1],
        data=original.copy(),
        units=initialized.units[:1],
        coords=initialized.coords,
    )
    safe_before = safe_field.data.copy()
    safe_result = run_transport_one_step(safe_field, forcing, grid, dt_s=600.0)
    np.testing.assert_array_equal(safe_field.data, safe_before)
    safe_output = safe_result.state.data.copy()

    consumed_field = TracerField(
        names=initialized.names[:1],
        data=original.copy(),
        units=initialized.units[:1],
        coords=initialized.coords,
    )
    consumed_before = consumed_field.data.copy()
    consumed_result = run_transport_one_step(
        consumed_field,
        forcing,
        grid,
        dt_s=600.0,
        consume_input=True,
    )

    assert not np.array_equal(consumed_field.data, consumed_before)
    assert np.shares_memory(consumed_result.state.data, consumed_field.data)
    np.testing.assert_array_equal(consumed_result.state.data, safe_output)


@requires_transport_data
def test_transport_one_step_keeps_pure_reference_path_non_destructive(monkeypatch):
    monkeypatch.setenv("WOMBAT_NUMBA", "0")
    config = load_run_config(RESIDUAL_CONFIG)
    grid = load_transport_grid(config.grid_template)
    initialized = initialize_tracers(
        config.initial_restart,
        config.species_database,
        template_path=config.grid_template,
    )
    original = np.ascontiguousarray(initialized.data[..., :1])
    forcing = _load_forcing(config, grid=grid)

    safe_field = TracerField(
        names=initialized.names[:1],
        data=original.copy(),
        units=initialized.units[:1],
        coords=initialized.coords,
    )
    safe_result = run_transport_one_step(safe_field, forcing, grid, dt_s=600.0)

    reference_field = TracerField(
        names=initialized.names[:1],
        data=original.copy(),
        units=initialized.units[:1],
        coords=initialized.coords,
    )
    reference_before = reference_field.data.copy()
    reference_result = run_transport_one_step(
        reference_field,
        forcing,
        grid,
        dt_s=600.0,
        consume_input=True,
    )

    np.testing.assert_array_equal(reference_field.data, reference_before)
    assert not np.shares_memory(reference_result.state.data, reference_field.data)
    np.testing.assert_array_equal(reference_result.state.data, safe_result.state.data)


@requires_transport_data
def test_trace_transport_one_step_captures_operator_handoffs(transport_numba_mode):
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
    stage_masses = compute_transport_stage_masses(trace, field, grid.area_m2)
    assert tuple(stage.operator for stage in stage_masses) == ("tpcore", "vdiff", "convection")
    assert stage_masses[0].initial_scalar_mass.shape == (1,)
    np.testing.assert_allclose(stage_masses[-1].final_scalar_mass, stage_masses[0].initial_scalar_mass, rtol=1e-13)


@requires_transport_data
def test_trace_transport_one_step_passes_active_surface_emissions_to_vdiff():
    config = load_run_config(RESIDUAL_CONFIG)
    grid = load_transport_grid(config.grid_template)
    field = initialize_tracers(config.initial_restart, config.species_database, template_path=config.grid_template)
    field = TracerField(
        names=field.names[:1],
        data=field.data[..., :1],
        units=field.units[:1],
        coords=field.coords,
    )
    emissions_data = np.zeros_like(field.data)
    expected_surface_flux = np.full((FIXED_GRID["lat"], FIXED_GRID["lon"], 1), 1.0e-12)
    emissions_data[0, -1, :, :, :] = expected_surface_flux
    emissions = TracerField(names=field.names, data=emissions_data, units=("kg/m2/s",), coords=field.coords)

    trace = trace_transport_one_step(field, _load_forcing(config, grid=grid), grid, dt_s=600.0, active_emissions=emissions)

    np.testing.assert_array_equal(trace.vdiff_input.surface_flux_kg_m2_s, expected_surface_flux)


@requires_transport_data
def test_trace_transport_one_step_scales_active_surface_emissions_for_vdiff_solver():
    config = load_run_config(RESIDUAL_CONFIG)
    grid = load_transport_grid(config.grid_template)
    field = initialize_tracers(config.initial_restart, config.species_database, template_path=config.grid_template)
    field = TracerField(
        names=field.names[:1],
        data=field.data[..., :1],
        units=field.units[:1],
        coords=field.coords,
    )
    emissions_data = np.zeros_like(field.data)
    raw_surface_flux = np.full((FIXED_GRID["lat"], FIXED_GRID["lon"], 1), 1.0e-12)
    emissions_data[0, -1, :, :, :] = raw_surface_flux
    emissions = TracerField(names=field.names, data=emissions_data, units=("kg/m2/s",), coords=field.coords)
    factor = np.array([0.25], dtype=np.float64)

    trace = trace_transport_one_step(
        field,
        _load_forcing(config, grid=grid),
        grid,
        dt_s=600.0,
        active_emissions=emissions,
        surface_flux_to_vmr_factor=factor,
    )

    np.testing.assert_array_equal(trace.vdiff_input.surface_flux_kg_m2_s, raw_surface_flux)
    np.testing.assert_array_equal(trace.vdiff_input.surface_flux_for_vdiff, raw_surface_flux * factor)


def test_active_emissions_rejects_vertically_distributed_flux_until_supported():
    tracer = TracerField(
        names=("CO2",),
        data=np.zeros((1, 3, 2, 2, 1), dtype=np.float64),
        units=("mol mol-1 dry",),
        coords={},
    )
    emissions_data = np.zeros_like(tracer.data)
    emissions_data[0, 1, 0, 0, 0] = 1.0e-12
    emissions = TracerField(names=tracer.names, data=emissions_data, units=("kg/m2/s",), coords={})

    try:
        _surface_flux_from_active_emissions(tracer, emissions, nlat=2, nlon=2, ntracer=1)
    except ValueError as exc:
        assert "vertically distributed emissions are not yet supported" in str(exc)
    else:
        raise AssertionError("vertically distributed emissions were accepted")


def test_surface_emissions_are_accepted_without_full_5d_field():
    tracer = TracerField(
        names=("A",),
        data=np.zeros((1, 3, 2, 2, 1), dtype=np.float64),
        units=("mol mol-1 dry",),
        coords={},
    )
    values = np.arange(4.0).reshape(2, 2, 1)
    emissions = SurfaceEmissions(
        names=("A",),
        data=values,
        units=("kg/m2/s",),
        coords={"AREA": np.ones((2, 2), dtype=np.float64)},
    )

    surface = _surface_flux_from_active_emissions(tracer, emissions, nlat=2, nlon=2, ntracer=1)

    np.testing.assert_array_equal(surface, values)


@requires_transport_data
def test_transport_window_accumulates_average_state():
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
    assert result.state.shape == field.shape
    assert result.average_state.shape == field.shape
    assert result.average_delp_dry_hpa.shape == (1, FIXED_GRID["lev"], FIXED_GRID["lat"], FIXED_GRID["lon"])
    assert np.all(np.isfinite(result.average_state.data))


@requires_transport_data
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
