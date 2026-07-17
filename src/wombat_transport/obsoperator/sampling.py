from __future__ import annotations

import math
from typing import Callable

import numpy as np

from wombat_transport.constants import AIRMW_G_PER_MOL, G0_M_PER_S2, H2OMW_G_PER_MOL
from wombat_transport.met_diagnostics import RD_J_PER_KG_K
from wombat_transport.obsoperator.state import (
    _VERTICAL_EXACT,
    _VERTICAL_NORMALIZED,
    _VERTICAL_NORMALIZED_PRESSURE,
    _VERTICAL_PRESSURE,
    _VERTICAL_PRESSURE_LEVEL,
    _VERTICAL_PRESSURE_WEIGHT,
)
from wombat_transport.transport.numba_control import configure_numba_threads
from wombat_transport.transport.numba_control import numba_available_and_enabled

try:  # Optional acceleration path; the same array kernel runs in Python as the reference fallback.
    from numba import njit
except ImportError:  # pragma: no cover - exercised in environments without numba.
    njit = None

def _sample_prepared_entries_kernel(
    state_bottom: np.ndarray,
    block_width: int,
    wet_surface_pressure_hpa: np.ndarray,
    specific_humidity_kg_kg: np.ndarray,
    temperature_k: np.ndarray,
    hyai_hpa: np.ndarray,
    hybi: np.ndarray,
    scheduled_entries: np.ndarray,
    entry_field_start: np.ndarray,
    entry_field_count: np.ndarray,
    entry_horizontal_start: np.ndarray,
    entry_horizontal_count: np.ndarray,
    entry_vertical_type: np.ndarray,
    entry_vertical_unit: np.ndarray,
    entry_vertical_weighting: np.ndarray,
    entry_vertical_lower: np.ndarray,
    entry_vertical_upper: np.ndarray,
    entry_exact_start: np.ndarray,
    entry_exact_count: np.ndarray,
    field_indices: np.ndarray,
    horizontal_lat: np.ndarray,
    horizontal_lon: np.ndarray,
    horizontal_weight: np.ndarray,
    exact_value: np.ndarray,
    exact_weight: np.ndarray,
    samples: np.ndarray,
) -> None:
    nlev = state_bottom.shape[1]
    for schedule_index in range(scheduled_entries.size):
        entry_index = scheduled_entries[schedule_index]
        field_start = entry_field_start[entry_index]
        field_count = entry_field_count[entry_index]
        for field_offset in range(field_count):
            samples[schedule_index, field_offset] = 0.0

        horizontal_start = entry_horizontal_start[entry_index]
        horizontal_end = horizontal_start + entry_horizontal_count[entry_index]
        vertical_type = entry_vertical_type[entry_index]
        vertical_unit = entry_vertical_unit[entry_index]
        vertical_weighting = entry_vertical_weighting[entry_index]
        lower = entry_vertical_lower[entry_index]
        upper = entry_vertical_upper[entry_index]

        for horizontal_index in range(horizontal_start, horizontal_end):
            lat = horizontal_lat[horizontal_index]
            lon = horizontal_lon[horizontal_index]
            horizontal_factor = horizontal_weight[horizontal_index]
            surface_pressure = wet_surface_pressure_hpa[lat, lon]

            if vertical_type == _VERTICAL_EXACT:
                exact_start = entry_exact_start[entry_index]
                exact_end = exact_start + entry_exact_count[entry_index]
                for field_offset in range(field_count):
                    tracer = field_indices[field_start + field_offset]
                    block = tracer // block_width
                    lane = tracer - block * block_width
                    vertical_sum = 0.0
                    for exact_index in range(exact_start, exact_end):
                        value = exact_value[exact_index]
                        if vertical_unit == _VERTICAL_PRESSURE_LEVEL:
                            level = int(value) - 1
                        elif vertical_unit == _VERTICAL_PRESSURE:
                            match = -1
                            for candidate in range(nlev):
                                edge = hyai_hpa[candidate] + hybi[candidate] * surface_pressure
                                if edge <= value:
                                    match = candidate
                                    break
                            level = nlev - 1 if match < 0 else max(match - 1, 0)
                        else:
                            cumulative_height = 0.0
                            level = -1
                            for candidate in range(nlev):
                                edge_lower = hyai_hpa[candidate] + hybi[candidate] * surface_pressure
                                edge_upper = hyai_hpa[candidate + 1] + hybi[candidate + 1] * surface_pressure
                                q = specific_humidity_kg_kg[candidate, lat, lon]
                                avgw = AIRMW_G_PER_MOL * q / (H2OMW_G_PER_MOL * (1.0 - q))
                                xh2o = avgw / (1.0 + avgw)
                                virtual_temperature = temperature_k[candidate, lat, lon] / (
                                    1.0 - xh2o * (1.0 - H2OMW_G_PER_MOL / AIRMW_G_PER_MOL)
                                )
                                cumulative_height += (
                                    RD_J_PER_KG_K
                                    / G0_M_PER_S2
                                    * virtual_temperature
                                    * math.log(edge_lower / edge_upper)
                                )
                                if cumulative_height >= value:
                                    level = candidate
                                    break
                            if level < 0:
                                raise ValueError("vertical altitude exceeds the modeled column")
                        vertical_sum += (
                            exact_weight[exact_index] * state_bottom[block, level, lat, lon, lane]
                        )
                    samples[schedule_index, field_offset] += horizontal_factor * vertical_sum
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
                    edge_lower = hyai_hpa[candidate] + hybi[candidate] * surface_pressure
                    edge_upper = hyai_hpa[candidate + 1] + hybi[candidate + 1] * surface_pressure
                    q = specific_humidity_kg_kg[candidate, lat, lon]
                    avgw = AIRMW_G_PER_MOL * q / (H2OMW_G_PER_MOL * (1.0 - q))
                    xh2o = avgw / (1.0 + avgw)
                    virtual_temperature = temperature_k[candidate, lat, lon] / (
                        1.0 - xh2o * (1.0 - H2OMW_G_PER_MOL / AIRMW_G_PER_MOL)
                    )
                    cumulative_height += (
                        RD_J_PER_KG_K
                        / G0_M_PER_S2
                        * virtual_temperature
                        * math.log(edge_lower / edge_upper)
                    )
                    if level_start < 0 and cumulative_height >= lower:
                        level_start = candidate
                    if level_end < 0 and cumulative_height >= upper:
                        level_end = candidate
                        break
                if level_start < 0 or level_end < 0:
                    raise ValueError("vertical altitude exceeds the modeled column")

            normalization = 0.0
            for level in range(level_start, level_end + 1):
                if vertical_weighting in (_VERTICAL_NORMALIZED_PRESSURE, _VERTICAL_PRESSURE_WEIGHT):
                    edge_lower = hyai_hpa[level] + hybi[level] * surface_pressure
                    edge_upper = hyai_hpa[level + 1] + hybi[level + 1] * surface_pressure
                    normalization += edge_lower - edge_upper
                else:
                    normalization += 1.0

            for field_offset in range(field_count):
                tracer = field_indices[field_start + field_offset]
                block = tracer // block_width
                lane = tracer - block * block_width
                vertical_sum = 0.0
                for level in range(level_start, level_end + 1):
                    if vertical_weighting in (_VERTICAL_NORMALIZED_PRESSURE, _VERTICAL_PRESSURE_WEIGHT):
                        edge_lower = hyai_hpa[level] + hybi[level] * surface_pressure
                        edge_upper = hyai_hpa[level + 1] + hybi[level + 1] * surface_pressure
                        weight = edge_lower - edge_upper
                    else:
                        weight = 1.0
                    vertical_sum += weight * state_bottom[block, level, lat, lon, lane]
                if vertical_weighting in (_VERTICAL_NORMALIZED_PRESSURE, _VERTICAL_NORMALIZED):
                    vertical_sum /= normalization
                samples[schedule_index, field_offset] += horizontal_factor * vertical_sum


if njit is not None:
    _sample_prepared_entries_numba = njit(cache=True, nogil=True)(_sample_prepared_entries_kernel)
else:  # pragma: no cover - exercised in environments without numba.
    _sample_prepared_entries_numba = None


def _accumulate_prepared_samples_kernel(
    scheduled_entries: np.ndarray,
    time_weights: np.ndarray,
    entry_field_start: np.ndarray,
    entry_field_count: np.ndarray,
    samples: np.ndarray,
    field_accumulator: np.ndarray,
) -> None:
    for schedule_index in range(scheduled_entries.size):
        entry_index = scheduled_entries[schedule_index]
        field_start = entry_field_start[entry_index]
        field_count = entry_field_count[entry_index]
        time_weight = time_weights[schedule_index]
        for field_offset in range(field_count):
            field_accumulator[field_start + field_offset] += time_weight * samples[schedule_index, field_offset]


if njit is not None:
    _accumulate_prepared_samples_numba = njit(cache=True, nogil=True)(_accumulate_prepared_samples_kernel)
else:  # pragma: no cover - exercised in environments without numba.
    _accumulate_prepared_samples_numba = _accumulate_prepared_samples_kernel

def select_sampling_kernel() -> Callable[..., None]:
    use_numba = numba_available_and_enabled(available=njit is not None)
    if use_numba:
        configure_numba_threads(available=True)
    return _sample_prepared_entries_numba if use_numba else _sample_prepared_entries_kernel


def accumulate_prepared_samples(
    scheduled_entries: np.ndarray,
    time_weights: np.ndarray,
    entry_field_start: np.ndarray,
    entry_field_count: np.ndarray,
    samples: np.ndarray,
    field_accumulator: np.ndarray,
) -> None:
    use_numba = numba_available_and_enabled(available=njit is not None)
    if use_numba:
        configure_numba_threads(available=True)
    kernel = _accumulate_prepared_samples_numba if use_numba else _accumulate_prepared_samples_kernel
    kernel(
        scheduled_entries,
        time_weights,
        entry_field_start,
        entry_field_count,
        samples,
        field_accumulator,
    )
