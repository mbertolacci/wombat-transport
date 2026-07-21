from __future__ import annotations

import math
from typing import Callable

import numpy as np

from wombat_transport.constants import AIRMW_G_PER_MOL, G0_M_PER_S2, H2OMW_G_PER_MOL
from wombat_transport.met_diagnostics import RD_J_PER_KG_K
from wombat_transport.obsoperator.state import (
    _HORIZONTAL_AREA,
    _HORIZONTAL_EXACT,
    _HORIZONTAL_NORMALIZED,
    _HORIZONTAL_NORMALIZED_AREA,
    _VERTICAL_EXACT,
    _VERTICAL_NORMALIZED,
    _VERTICAL_NORMALIZED_PRESSURE,
    _VERTICAL_PRESSURE,
    _VERTICAL_PRESSURE_LEVEL,
    _VERTICAL_PRESSURE_WEIGHT,
)
from wombat_transport.transport.numba_control import configure_numba_threads
from wombat_transport.transport.numba_control import numba_available_and_enabled

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised without the optional accelerator.
    njit = None


def _sample_obs_plan_block_range_kernel(
    state_blocks: np.ndarray,
    block_width: int,
    start_block: int,
    end_block: int,
    step_time_us: int,
    wet_surface_pressure_hpa: np.ndarray,
    specific_humidity_kg_kg: np.ndarray,
    temperature_k: np.ndarray,
    area_m2: np.ndarray,
    hyai_hpa: np.ndarray,
    hybi: np.ndarray,
    first_entry: int,
    entry_field_start: np.ndarray,
    entry_field_count: np.ndarray,
    field_tracer: np.ndarray,
    field_to_accumulator: np.ndarray,
    time_operator_start: np.ndarray,
    time_operator_count: np.ndarray,
    time_operator_bounds_us: np.ndarray,
    time_operator_weight: np.ndarray,
    horizontal_operator_start: np.ndarray,
    horizontal_operator_count: np.ndarray,
    horizontal_operator_bounds: np.ndarray,
    horizontal_weight_type: np.ndarray,
    horizontal_weight: np.ndarray,
    horizontal_normalization: np.ndarray,
    vertical_operator_start: np.ndarray,
    vertical_operator_count: np.ndarray,
    vertical_operator_type: np.ndarray,
    vertical_operator_unit: np.ndarray,
    vertical_operator_bounds: np.ndarray,
    vertical_weight_type: np.ndarray,
    vertical_weight: np.ndarray,
    sample_scratch: np.ndarray,
    accumulator: np.ndarray,
) -> None:
    nlev = state_blocks.shape[1]
    tracer_start = start_block * block_width
    tracer_stop = end_block * block_width
    all_blocks = start_block == 0 and end_block == state_blocks.shape[0]
    for entry_index in range(first_entry, entry_field_start.size):
        time_weight = 0.0
        time_start = time_operator_start[entry_index]
        time_end = time_start + time_operator_count[entry_index]
        for time_index in range(time_start, time_end):
            if (
                time_operator_bounds_us[time_index, 0] <= step_time_us
                and step_time_us < time_operator_bounds_us[time_index, 1]
            ):
                time_weight += time_operator_weight[time_index]
        if time_weight == 0.0:
            continue

        field_start = entry_field_start[entry_index]
        field_end = field_start + entry_field_count[entry_index]
        if all_blocks:
            owned_fields = field_end - field_start
            for field_index in range(field_start, field_end):
                tracer = field_tracer[field_index]
                sample_scratch[tracer] = 0.0
        else:
            owned_fields = 0
            for field_index in range(field_start, field_end):
                tracer = field_tracer[field_index]
                if tracer_start <= tracer < tracer_stop:
                    sample_scratch[tracer - tracer_start] = 0.0
                    owned_fields += 1
        if owned_fields == 0:
            continue

        horizontal_start = horizontal_operator_start[entry_index]
        horizontal_end = horizontal_start + horizontal_operator_count[entry_index]
        vertical_start = vertical_operator_start[entry_index]
        vertical_end = vertical_start + vertical_operator_count[entry_index]
        for horizontal_index in range(horizontal_start, horizontal_end):
            lat_start = horizontal_operator_bounds[horizontal_index, 0, 0]
            lat_end = horizontal_operator_bounds[horizontal_index, 0, 1]
            lon_start = horizontal_operator_bounds[horizontal_index, 1, 0]
            lon_end = horizontal_operator_bounds[horizontal_index, 1, 1]
            horizontal_type = horizontal_weight_type[horizontal_index]
            normalization = horizontal_normalization[horizontal_index]

            for lon in range(lon_start, lon_end):
                for lat in range(lat_start, lat_end):
                    if horizontal_type in (_HORIZONTAL_AREA, _HORIZONTAL_NORMALIZED_AREA):
                        horizontal_factor = area_m2[lat, lon]
                    else:
                        horizontal_factor = 1.0
                    if horizontal_type in (_HORIZONTAL_NORMALIZED, _HORIZONTAL_NORMALIZED_AREA):
                        horizontal_factor /= normalization
                    elif horizontal_type == _HORIZONTAL_EXACT:
                        horizontal_factor *= horizontal_weight[horizontal_index]

                    surface_pressure = wet_surface_pressure_hpa[lat, lon]
                    for vertical_index in range(vertical_start, vertical_end):
                        vertical_type = vertical_operator_type[vertical_index]
                        vertical_unit = vertical_operator_unit[vertical_index]
                        lower = vertical_operator_bounds[vertical_index, 0]
                        upper = vertical_operator_bounds[vertical_index, 1]
                        weighting = vertical_weight_type[vertical_index]

                        if vertical_type == _VERTICAL_EXACT:
                            if vertical_unit == _VERTICAL_PRESSURE_LEVEL:
                                level_start = int(lower) - 1
                            elif vertical_unit == _VERTICAL_PRESSURE:
                                match = -1
                                for candidate in range(nlev):
                                    edge = hyai_hpa[candidate] + hybi[candidate] * surface_pressure
                                    if edge <= lower:
                                        match = candidate
                                        break
                                level_start = nlev - 1 if match < 0 else max(match - 1, 0)
                            else:
                                cumulative_height = 0.0
                                level_start = -1
                                for candidate in range(nlev):
                                    cumulative_height += _layer_height_numba(
                                        candidate,
                                        lat,
                                        lon,
                                        surface_pressure,
                                        specific_humidity_kg_kg,
                                        temperature_k,
                                        hyai_hpa,
                                        hybi,
                                    )
                                    if cumulative_height >= lower:
                                        level_start = candidate
                                        break
                                if level_start < 0:
                                    raise ValueError("vertical altitude exceeds the modeled column")
                            exact_factor = horizontal_factor * vertical_weight[vertical_index]
                            if all_blocks:
                                for field_index in range(field_start, field_end):
                                    tracer = field_tracer[field_index]
                                    block = tracer // block_width
                                    lane = tracer - block * block_width
                                    sample_scratch[tracer] += exact_factor * state_blocks[
                                        block, level_start, lat, lon, lane
                                    ]
                            else:
                                for field_index in range(field_start, field_end):
                                    tracer = field_tracer[field_index]
                                    if tracer_start <= tracer < tracer_stop:
                                        block = tracer // block_width
                                        lane = tracer - block * block_width
                                        sample_scratch[tracer - tracer_start] += (
                                            exact_factor
                                            * state_blocks[block, level_start, lat, lon, lane]
                                        )
                            continue

                        if vertical_unit == _VERTICAL_PRESSURE_LEVEL:
                            level_start = int(lower) - 1
                            level_end = int(upper) - 1
                        elif vertical_unit == _VERTICAL_PRESSURE:
                            start_match = -1
                            end_match = -1
                            for candidate in range(nlev):
                                edge = hyai_hpa[candidate] + hybi[candidate] * surface_pressure
                                if start_match < 0 and edge <= upper:
                                    start_match = candidate
                                if end_match < 0 and edge <= lower:
                                    end_match = candidate
                            level_start = nlev - 1 if start_match < 0 else max(start_match - 1, 0)
                            level_end = nlev - 1 if end_match < 0 else max(end_match - 1, 0)
                        else:
                            cumulative_height = 0.0
                            level_start = -1
                            level_end = -1
                            for candidate in range(nlev):
                                cumulative_height += _layer_height_numba(
                                    candidate,
                                    lat,
                                    lon,
                                    surface_pressure,
                                    specific_humidity_kg_kg,
                                    temperature_k,
                                    hyai_hpa,
                                    hybi,
                                )
                                if level_start < 0 and cumulative_height >= lower:
                                    level_start = candidate
                                if level_end < 0 and cumulative_height >= upper:
                                    level_end = candidate
                                    break
                            if level_start < 0 or level_end < 0:
                                raise ValueError("vertical altitude exceeds the modeled column")

                        vertical_normalization = 0.0
                        for level in range(level_start, level_end + 1):
                            if weighting in (
                                _VERTICAL_NORMALIZED_PRESSURE,
                                _VERTICAL_PRESSURE_WEIGHT,
                            ):
                                edge_lower = hyai_hpa[level] + hybi[level] * surface_pressure
                                edge_upper = hyai_hpa[level + 1] + hybi[level + 1] * surface_pressure
                                vertical_normalization += edge_lower - edge_upper
                            else:
                                vertical_normalization += 1.0

                        if all_blocks:
                            for field_index in range(field_start, field_end):
                                tracer = field_tracer[field_index]
                                block = tracer // block_width
                                lane = tracer - block * block_width
                                operator_sum = 0.0
                                for level in range(level_start, level_end + 1):
                                    if weighting in (
                                        _VERTICAL_NORMALIZED_PRESSURE,
                                        _VERTICAL_PRESSURE_WEIGHT,
                                    ):
                                        edge_lower = hyai_hpa[level] + hybi[level] * surface_pressure
                                        edge_upper = hyai_hpa[level + 1] + hybi[level + 1] * surface_pressure
                                        weight = edge_lower - edge_upper
                                    else:
                                        weight = 1.0
                                    operator_sum += weight * state_blocks[
                                        block, level, lat, lon, lane
                                    ]
                                if weighting in (
                                    _VERTICAL_NORMALIZED_PRESSURE,
                                    _VERTICAL_NORMALIZED,
                                ):
                                    operator_sum /= vertical_normalization
                                sample_scratch[tracer] += horizontal_factor * operator_sum
                        else:
                            for field_index in range(field_start, field_end):
                                tracer = field_tracer[field_index]
                                if tracer < tracer_start or tracer >= tracer_stop:
                                    continue
                                block = tracer // block_width
                                lane = tracer - block * block_width
                                operator_sum = 0.0
                                for level in range(level_start, level_end + 1):
                                    if weighting in (
                                        _VERTICAL_NORMALIZED_PRESSURE,
                                        _VERTICAL_PRESSURE_WEIGHT,
                                    ):
                                        edge_lower = hyai_hpa[level] + hybi[level] * surface_pressure
                                        edge_upper = hyai_hpa[level + 1] + hybi[level + 1] * surface_pressure
                                        weight = edge_lower - edge_upper
                                    else:
                                        weight = 1.0
                                    operator_sum += weight * state_blocks[
                                        block, level, lat, lon, lane
                                    ]
                                if weighting in (
                                    _VERTICAL_NORMALIZED_PRESSURE,
                                    _VERTICAL_NORMALIZED,
                                ):
                                    operator_sum /= vertical_normalization
                                sample_scratch[tracer - tracer_start] += (
                                    horizontal_factor * operator_sum
                                )

        if all_blocks:
            for field_index in range(field_start, field_end):
                tracer = field_tracer[field_index]
                accumulator[field_to_accumulator[field_index]] += (
                    time_weight * sample_scratch[tracer]
                )
        else:
            for field_index in range(field_start, field_end):
                tracer = field_tracer[field_index]
                if tracer_start <= tracer < tracer_stop:
                    accumulator[field_to_accumulator[field_index]] += (
                        time_weight * sample_scratch[tracer - tracer_start]
                    )


def _layer_height(
    level: int,
    lat: int,
    lon: int,
    surface_pressure: float,
    specific_humidity_kg_kg: np.ndarray,
    temperature_k: np.ndarray,
    hyai_hpa: np.ndarray,
    hybi: np.ndarray,
) -> float:
    edge_lower = hyai_hpa[level] + hybi[level] * surface_pressure
    edge_upper = hyai_hpa[level + 1] + hybi[level + 1] * surface_pressure
    q = specific_humidity_kg_kg[level, lat, lon]
    avgw = AIRMW_G_PER_MOL * q / (H2OMW_G_PER_MOL * (1.0 - q))
    xh2o = avgw / (1.0 + avgw)
    virtual_temperature = temperature_k[level, lat, lon] / (
        1.0 - xh2o * (1.0 - H2OMW_G_PER_MOL / AIRMW_G_PER_MOL)
    )
    return (
        RD_J_PER_KG_K
        / G0_M_PER_S2
        * virtual_temperature
        * math.log(edge_lower / edge_upper)
    )


_layer_height_numba = njit(inline="always")(_layer_height) if njit is not None else _layer_height


if njit is not None:
    _sample_obs_plan_block_range_numba = njit(cache=True, nogil=True)(
        _sample_obs_plan_block_range_kernel
    )
else:  # pragma: no cover - exercised without the optional accelerator.
    _sample_obs_plan_block_range_numba = None


# Reference name retained for lightweight instrumentation in tests.
_sample_prepared_entries_kernel = _sample_obs_plan_block_range_kernel


def select_sampling_kernel() -> Callable[..., None]:
    use_numba = numba_available_and_enabled(available=njit is not None)
    if use_numba:
        configure_numba_threads(available=True)
    return (
        _sample_obs_plan_block_range_numba
        if use_numba
        else _sample_prepared_entries_kernel
    )
