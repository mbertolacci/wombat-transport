from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_ID_LENGTH = 255
MAX_FIELD_NAME_LENGTH = 64
FIELD_PREFIX = "SpeciesConcVV_"
FIELD_ALL = "SpeciesConcVV_?ALL?"
FIELD_ADVECTED = "SpeciesConcVV_?ADV?"

HORIZONTAL_WEIGHTING_CODES = {"area": 0, "normalized_area": 1, "normalized": 2, "equal": 3}
VERTICAL_TYPE_CODES = {"range": 0, "exact": 1}
VERTICAL_UNIT_CODES = {"pressure": 0, "altitude": 1, "pressure_level": 2}
VERTICAL_WEIGHTING_CODES = {"normalized_pressure": 0, "pressure": 1, "normalized": 2, "equal": 3}
_VERTICAL_EXACT = 1
_VERTICAL_PRESSURE = 0
_VERTICAL_ALTITUDE = 1
_VERTICAL_PRESSURE_LEVEL = 2
_VERTICAL_NORMALIZED_PRESSURE = 0
_VERTICAL_PRESSURE_WEIGHT = 1
_VERTICAL_NORMALIZED = 2

@dataclass(frozen=True)
class _PreparedObsOperators:
    entry_field_start: np.ndarray
    entry_field_count: np.ndarray
    entry_horizontal_start: np.ndarray
    entry_horizontal_count: np.ndarray
    entry_vertical_type: np.ndarray
    entry_vertical_unit: np.ndarray
    entry_vertical_weighting: np.ndarray
    entry_vertical_lower: np.ndarray
    entry_vertical_upper: np.ndarray
    entry_exact_start: np.ndarray
    entry_exact_count: np.ndarray
    field_indices: np.ndarray
    horizontal_lat: np.ndarray
    horizontal_lon: np.ndarray
    horizontal_weight: np.ndarray
    exact_value: np.ndarray
    exact_weight: np.ndarray
    max_field_count: int


@dataclass
class _ObsOperatorArrayState:
    ids: tuple[str, ...]
    field_names: tuple[tuple[str, ...], ...]
    prepared: _PreparedObsOperators
    field_accumulator: np.ndarray
    horizontal_weighting: np.ndarray
    time_start: np.ndarray
    time_count: np.ndarray
    time_consumed: np.ndarray
    remaining_time_us: np.ndarray
    remaining_time_weight: np.ndarray
    active: np.ndarray
    schedule_times_us: np.ndarray
    schedule_start: np.ndarray
    schedule_count: np.ndarray
    schedule_entry: np.ndarray
    schedule_weight: np.ndarray

    @property
    def entry_count(self) -> int:
        return len(self.ids)


def _array_state_from_components(
    rows: list[
        tuple[
            str,
            tuple[tuple[str, ...], np.ndarray],
            tuple[np.ndarray, np.ndarray, np.ndarray],
            tuple[np.ndarray, np.ndarray, np.ndarray, int],
            tuple[int, int, int, float, float, np.ndarray, np.ndarray],
        ]
    ],
) -> _ObsOperatorArrayState:
    entry_count = len(rows)
    field_total = sum(len(row[1][0]) for row in rows)
    time_total = sum(row[2][2].size for row in rows)
    horizontal_total = sum(row[3][0].size for row in rows)
    exact_total = sum(row[4][5].size for row in rows)

    entry_field_start = np.empty(entry_count, dtype=np.int64)
    entry_field_count = np.empty(entry_count, dtype=np.int32)
    field_indices = np.empty(field_total, dtype=np.int64)
    field_accumulator = np.zeros(field_total, dtype=np.float64)
    time_start = np.empty(entry_count, dtype=np.int64)
    time_count = np.empty(entry_count, dtype=np.int32)
    remaining_time_us = np.empty(time_total, dtype=np.int64)
    remaining_time_weight = np.empty(time_total, dtype=np.float64)
    schedule_entry = np.empty(time_total, dtype=np.int64)
    entry_horizontal_start = np.empty(entry_count, dtype=np.int64)
    entry_horizontal_count = np.empty(entry_count, dtype=np.int32)
    horizontal_lat = np.empty(horizontal_total, dtype=np.int32)
    horizontal_lon = np.empty(horizontal_total, dtype=np.int32)
    horizontal_weight = np.empty(horizontal_total, dtype=np.float64)
    horizontal_weighting = np.empty(entry_count, dtype=np.int8)
    entry_vertical_type = np.empty(entry_count, dtype=np.int8)
    entry_vertical_unit = np.empty(entry_count, dtype=np.int8)
    entry_vertical_weighting = np.empty(entry_count, dtype=np.int8)
    entry_vertical_lower = np.empty(entry_count, dtype=np.float64)
    entry_vertical_upper = np.empty(entry_count, dtype=np.float64)
    entry_exact_start = np.empty(entry_count, dtype=np.int64)
    entry_exact_count = np.empty(entry_count, dtype=np.int32)
    exact_value = np.empty(exact_total, dtype=np.float64)
    exact_weight = np.empty(exact_total, dtype=np.float64)

    field_offset = 0
    time_offset = 0
    horizontal_offset = 0
    exact_offset = 0
    max_field_count = 0
    for entry_index, (_, fields, time, horizontal, vertical) in enumerate(rows):
        field_names, entry_field_indices = fields
        count = len(field_names)
        field_slice = slice(field_offset, field_offset + count)
        entry_field_start[entry_index] = field_offset
        entry_field_count[entry_index] = count
        field_indices[field_slice] = entry_field_indices
        field_offset += count
        max_field_count = max(max_field_count, count)

        _, time_weights, time_values_us = time
        count = time_values_us.size
        time_slice = slice(time_offset, time_offset + count)
        time_start[entry_index] = time_offset
        time_count[entry_index] = count
        remaining_time_us[time_slice] = time_values_us
        remaining_time_weight[time_slice] = time_weights
        schedule_entry[time_slice] = entry_index
        time_offset += count

        horizontal_lats, horizontal_lons, horizontal_weights, horizontal_code = horizontal
        count = horizontal_lats.size
        horizontal_slice = slice(horizontal_offset, horizontal_offset + count)
        entry_horizontal_start[entry_index] = horizontal_offset
        entry_horizontal_count[entry_index] = count
        horizontal_lat[horizontal_slice] = horizontal_lats
        horizontal_lon[horizontal_slice] = horizontal_lons
        horizontal_weight[horizontal_slice] = horizontal_weights
        horizontal_weighting[entry_index] = horizontal_code
        horizontal_offset += count

        vertical_type, vertical_unit, vertical_weighting, vertical_lower, vertical_upper, values, weights = vertical
        entry_vertical_type[entry_index] = vertical_type
        entry_vertical_unit[entry_index] = vertical_unit
        entry_exact_start[entry_index] = exact_offset
        if vertical_type == _VERTICAL_EXACT:
            count = values.size
            exact_slice = slice(exact_offset, exact_offset + count)
            entry_vertical_weighting[entry_index] = -1
            entry_vertical_lower[entry_index] = np.nan
            entry_vertical_upper[entry_index] = np.nan
            entry_exact_count[entry_index] = count
            exact_value[exact_slice] = values
            exact_weight[exact_slice] = weights
            exact_offset += count
        else:
            entry_vertical_weighting[entry_index] = vertical_weighting
            entry_vertical_lower[entry_index] = vertical_lower
            entry_vertical_upper[entry_index] = vertical_upper
            entry_exact_count[entry_index] = 0

    order = np.argsort(remaining_time_us, kind="stable")
    sorted_times = remaining_time_us[order]
    schedule_entry = schedule_entry[order]
    schedule_weight = remaining_time_weight[order]
    schedule_times_us, schedule_start, schedule_count = np.unique(
        sorted_times,
        return_index=True,
        return_counts=True,
    )
    prepared = _PreparedObsOperators(
        entry_field_start=entry_field_start,
        entry_field_count=entry_field_count,
        entry_horizontal_start=entry_horizontal_start,
        entry_horizontal_count=entry_horizontal_count,
        entry_vertical_type=entry_vertical_type,
        entry_vertical_unit=entry_vertical_unit,
        entry_vertical_weighting=entry_vertical_weighting,
        entry_vertical_lower=entry_vertical_lower,
        entry_vertical_upper=entry_vertical_upper,
        entry_exact_start=entry_exact_start,
        entry_exact_count=entry_exact_count,
        field_indices=field_indices,
        horizontal_lat=horizontal_lat,
        horizontal_lon=horizontal_lon,
        horizontal_weight=horizontal_weight,
        exact_value=exact_value,
        exact_weight=exact_weight,
        max_field_count=max_field_count,
    )
    return _ObsOperatorArrayState(
        ids=tuple(row[0] for row in rows),
        field_names=tuple(row[1][0] for row in rows),
        prepared=prepared,
        field_accumulator=field_accumulator,
        horizontal_weighting=horizontal_weighting,
        time_start=time_start,
        time_count=time_count,
        time_consumed=np.zeros(entry_count, dtype=np.int32),
        remaining_time_us=remaining_time_us,
        remaining_time_weight=remaining_time_weight,
        active=np.ones(entry_count, dtype=bool),
        schedule_times_us=schedule_times_us,
        schedule_start=np.asarray(schedule_start, dtype=np.int64),
        schedule_count=np.asarray(schedule_count, dtype=np.int32),
        schedule_entry=schedule_entry,
        schedule_weight=schedule_weight,
    )
