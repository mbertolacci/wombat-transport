from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wombat_transport.cuda import CudaRuntime, CudaUnavailableError
from wombat_transport.fields import TracerField
from wombat_transport.grid import TransportGrid
from wombat_transport.obsoperator.input import _obs_plan_from_raw_entries
from wombat_transport.obsoperator.sampling import (
    _sample_obs_plan_block_range_kernel,
)
from wombat_transport.obsoperator.sampling_cuda import CudaObsSampler
from wombat_transport.obsoperator.state import ObsPlan
from wombat_transport.snapshot import CompletedStepSnapshot


def _runtime_or_skip() -> CudaRuntime:
    try:
        return CudaRuntime()
    except CudaUnavailableError as exc:
        pytest.skip(str(exc))


@pytest.mark.cuda
def test_cuda_obsoperator_exact_sampling_matches_cpu():
    runtime = _runtime_or_skip()
    nlev, nlat, nlon = 3, 3, 4
    bottom = np.zeros((nlev, nlat, nlon, 2), dtype=np.float64)
    for level in range(nlev):
        bottom[level, ..., 0] = level + 1.0
        bottom[level, ..., 1] = 10.0 * (level + 1.0)
    blocks = bottom[::-1][None, ...]
    grid = TransportGrid(
        lat_deg=np.array([-45.0, 0.0, 45.0]),
        lon_deg=np.array([-180.0, -90.0, 0.0, 90.0]),
        lev=np.arange(1.0, 4.0),
        area_m2=np.ones((nlat, nlon)),
        hyai_hpa=np.array([1000.0, 700.0, 300.0, 1.0]),
        hybi=np.zeros(4),
        template_path=Path("unused.nc4"),
    )
    start_us = int(datetime(2014, 9, 1).timestamp() * 1_000_000)
    plan = ObsPlan(
        ids=("sample",),
        field_names=("SpeciesConcVV_A", "SpeciesConcVV_B"),
        accumulator=np.zeros(2),
        entry_field_start=np.array([0], dtype=np.int64),
        entry_field_count=np.array([2], dtype=np.int32),
        field_tracer=np.array([0, 1], dtype=np.int64),
        field_to_accumulator=np.array([0, 1], dtype=np.int64),
        time_operator_start=np.array([0], dtype=np.int64),
        time_operator_count=np.array([1], dtype=np.int32),
        time_operator_bounds_us=np.array(
            [[start_us, start_us + 600_000_000]],
            dtype=np.int64,
        ),
        time_operator_weight=np.array([1.0]),
        horizontal_operator_start=np.array([0], dtype=np.int64),
        horizontal_operator_count=np.array([1], dtype=np.int32),
        horizontal_operator_bounds=np.array([[[0, 1], [0, 1]]], dtype=np.int32),
        horizontal_weight_type=np.array([4], dtype=np.int8),
        horizontal_weight=np.array([1.0]),
        horizontal_normalization=np.array([1.0]),
        vertical_operator_start=np.array([0], dtype=np.int64),
        vertical_operator_count=np.array([1], dtype=np.int32),
        vertical_operator_type=np.array([1], dtype=np.int8),
        vertical_operator_unit=np.array([2], dtype=np.int8),
        vertical_operator_bounds=np.array([[1.0, 1.0]]),
        vertical_weight_type=np.array([4], dtype=np.int8),
        vertical_weight=np.array([1.0]),
        entry_end_us=np.array([start_us + 600_000_000], dtype=np.int64),
    )
    forcing = SimpleNamespace(
        wet_surface_pressure_hpa=np.full((1, nlat, nlon), 1000.0),
        specific_humidity_kg_kg=np.zeros((1, nlev, nlat, nlon)),
        temperature_k=np.full((1, nlev, nlat, nlon), 280.0),
    )
    snapshot = CompletedStepSnapshot(
        timestamp=datetime(2014, 9, 1, 0, 10),
        state=TracerField(
            names=("A", "B"),
            data=runtime.to_device(blocks)[None, ...],
            units=("", ""),
            coords={},
        ),
        delp_dry_hpa=np.ones((1, nlev, nlat, nlon)),
        forcing=forcing,  # type: ignore[arg-type]
    )
    expected = np.zeros(2)
    _sample_obs_plan_block_range_kernel(
        blocks[:, ::-1],
        2,
        0,
        1,
        start_us,
        forcing.wet_surface_pressure_hpa[0],
        forcing.specific_humidity_kg_kg[0],
        forcing.temperature_k[0],
        grid.area_m2,
        grid.hyai_hpa,
        grid.hybi,
        0,
        plan.entry_field_start,
        plan.entry_field_count,
        plan.field_tracer,
        plan.field_to_accumulator,
        plan.time_operator_start,
        plan.time_operator_count,
        plan.time_operator_bounds_us,
        plan.time_operator_weight,
        plan.horizontal_operator_start,
        plan.horizontal_operator_count,
        plan.horizontal_operator_bounds,
        plan.horizontal_weight_type,
        plan.horizontal_weight,
        plan.horizontal_normalization,
        plan.vertical_operator_start,
        plan.vertical_operator_count,
        plan.vertical_operator_type,
        plan.vertical_operator_unit,
        plan.vertical_operator_bounds,
        plan.vertical_weight_type,
        plan.vertical_weight,
        np.empty(2),
        expected,
    )

    sampler = CudaObsSampler(runtime, dtype=np.float64, grid=grid)
    sampler.sample(plan, step_time_us=start_us, snapshot=snapshot)
    sampler.sync_to_host(plan)

    np.testing.assert_array_equal(plan.accumulator, expected)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_cuda_obsoperator_weighted_modes_match_cpu(dtype):
    runtime = _runtime_or_skip()
    grid = TransportGrid(
        lat_deg=np.array([-45.0, 0.0, 45.0]),
        lon_deg=np.array([-180.0, -90.0, 0.0, 90.0]),
        lev=np.arange(1.0, 4.0),
        area_m2=np.arange(1.0, 13.0).reshape(3, 4),
        hyai_hpa=np.array([1000.0, 700.0, 300.0, 1.0]),
        hybi=np.zeros(4),
        template_path=Path("unused.nc4"),
    )
    operators = (
        (
            {
                "type": "box",
                "unit": "grid_index",
                "longitude_start": 1,
                "longitude_end": 2,
                "latitude_start": 1,
                "latitude_end": 3,
                "weights": "normalized_area",
            },
            {
                "type": "range",
                "unit": "pressure",
                "start": 100.0,
                "end": 900.0,
                "weights": "normalized_pressure",
            },
        ),
        (
            {
                "type": "box",
                "unit": "grid_index",
                "longitude_start": 1,
                "longitude_end": 2,
                "latitude_start": 1,
                "latitude_end": 2,
                "weights": "equal",
            },
            {
                "type": "range",
                "unit": "altitude",
                "start": 0.0,
                "end": 6000.0,
                "weights": "normalized",
            },
        ),
        (
            {
                "type": "point",
                "unit": "degrees",
                "longitude": 0.0,
                "latitude": 0.0,
            },
            {
                "type": "exact",
                "unit": "pressure",
                "values": [900.0, 500.0],
                "weights": [0.25, 0.75],
            },
        ),
    )
    entries = []
    for index, (horizontal, vertical) in enumerate(operators):
        entries.append(
            {
                "id": f"operator-{index}",
                "fields": ["SpeciesConcVV_A", "SpeciesConcVV_B"],
                "time_operator": {
                    "type": "range",
                    "unit": "time_index",
                    "start": 0,
                    "end": 1,
                    "weights": "equal",
                },
                "horizontal_operator": horizontal,
                "vertical_operator": vertical,
            }
        )
    start = datetime(2014, 9, 1)
    plan = _obs_plan_from_raw_entries(
        entries,
        tracer_names=("A", "B"),
        grid=grid,
        simulation_start=start,
        transport_dt_s=600.0,
    )
    nlev, nlat, nlon = grid.shape
    bottom = np.zeros((nlev, nlat, nlon, 2), dtype=dtype)
    for level in range(nlev):
        for lat in range(nlat):
            for lon in range(nlon):
                bottom[level, lat, lon, 0] = level + 1 + lat + 10 * lon
                bottom[level, lat, lon, 1] = 10 * (level + 1) + lat + lon
    forcing = SimpleNamespace(
        wet_surface_pressure_hpa=np.full((1, nlat, nlon), 1000.0),
        specific_humidity_kg_kg=np.full((1, nlev, nlat, nlon), 0.001),
        temperature_k=np.full((1, nlev, nlat, nlon), 280.0),
    )
    snapshot = CompletedStepSnapshot(
        timestamp=start,
        state=TracerField(
            names=("A", "B"),
            data=runtime.to_device(bottom[::-1][None, ...])[None, ...],
            units=("", ""),
            coords={},
        ),
        delp_dry_hpa=np.ones((1, nlev, nlat, nlon)),
        forcing=forcing,  # type: ignore[arg-type]
    )
    expected = np.zeros_like(plan.accumulator)
    step_time_us = int(plan.time_operator_bounds_us[0, 0])
    _sample_obs_plan_block_range_kernel(
        bottom[None, ...],
        2,
        0,
        1,
        step_time_us,
        forcing.wet_surface_pressure_hpa[0],
        forcing.specific_humidity_kg_kg[0],
        forcing.temperature_k[0],
        grid.area_m2,
        grid.hyai_hpa,
        grid.hybi,
        0,
        plan.entry_field_start,
        plan.entry_field_count,
        plan.field_tracer,
        plan.field_to_accumulator,
        plan.time_operator_start,
        plan.time_operator_count,
        plan.time_operator_bounds_us,
        plan.time_operator_weight,
        plan.horizontal_operator_start,
        plan.horizontal_operator_count,
        plan.horizontal_operator_bounds,
        plan.horizontal_weight_type,
        plan.horizontal_weight,
        plan.horizontal_normalization,
        plan.vertical_operator_start,
        plan.vertical_operator_count,
        plan.vertical_operator_type,
        plan.vertical_operator_unit,
        plan.vertical_operator_bounds,
        plan.vertical_weight_type,
        plan.vertical_weight,
        np.empty(2),
        expected,
    )

    sampler = CudaObsSampler(runtime, dtype=dtype, grid=grid)
    sampler.sample(plan, step_time_us=step_time_us, snapshot=snapshot)
    sampler.sync_to_host(plan)

    np.testing.assert_allclose(plan.accumulator, expected, rtol=2e-15, atol=0.0)
