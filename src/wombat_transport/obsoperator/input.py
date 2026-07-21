from __future__ import annotations

from datetime import datetime
import gzip
import math
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from yaml12 import read_yaml

from wombat_transport.grid import TransportGrid
from wombat_transport.obsoperator.state import (
    FIELD_ADVECTED,
    FIELD_ALL,
    FIELD_PREFIX,
    HORIZONTAL_WEIGHTING_CODES,
    MAX_FIELD_NAME_LENGTH,
    MAX_ID_LENGTH,
    VERTICAL_TYPE_CODES,
    VERTICAL_UNIT_CODES,
    VERTICAL_WEIGHTING_CODES,
    ObsPlan,
    _allocate_numeric,
)
from wombat_transport.obsoperator.utils import (
    _datetime_to_microseconds,
    _validate_fixed_width_utf8,
    _seconds_to_microseconds,
    _validate_vertical_bounds,
    _validate_vertical_values,
)

def _load_obsoperator_raw_entries(input_path: Path) -> list[Any]:
    with _open_yaml_text(input_path) as handle:
        raw = read_yaml(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"ObsOperator input {input_path} must contain a mapping")
    if "entries" not in raw:
        raise KeyError(f"ObsOperator input {input_path} is missing entries")
    entries_raw = raw["entries"]
    if entries_raw is None:
        entries_raw = []
    if not isinstance(entries_raw, list):
        raise TypeError(f"ObsOperator input {input_path} entries must be a sequence")
    return entries_raw


def _load_obs_plan(
    path: str | Path,
    *,
    tracer_names: tuple[str, ...],
    grid: TransportGrid,
    simulation_start: datetime,
    transport_dt_s: float,
) -> ObsPlan:
    entries_raw = _load_obsoperator_raw_entries(Path(path))
    return _obs_plan_from_raw_entries(
        entries_raw,
        tracer_names=tracer_names,
        grid=grid,
        simulation_start=simulation_start,
        transport_dt_s=transport_dt_s,
    )


def _obs_plan_from_raw_entries(
    entries_raw: list[Any],
    *,
    tracer_names: tuple[str, ...],
    grid: TransportGrid,
    simulation_start: datetime,
    transport_dt_s: float,
) -> ObsPlan:
    entry_count = len(entries_raw)
    field_total = 0
    time_total = 0
    horizontal_total = 0
    vertical_total = 0
    for index, raw_entry in enumerate(entries_raw):
        label = f"entries[{index}]"
        mapping = _require_mapping(raw_entry, label)
        if "species" in mapping:
            raise ValueError(f"{label}.species is obsolete; use fields")
        field_raw = _required(mapping, "fields", label)
        requested = [field_raw] if isinstance(field_raw, str) else field_raw
        if not isinstance(requested, list) or not requested:
            raise TypeError(f"{label}.fields must be a field name or nonempty sequence")
        selected: set[int] = set()
        tracer_by_name = {name: tracer for tracer, name in enumerate(tracer_names)}
        for value in requested:
            if not isinstance(value, str):
                raise TypeError(f"{label}.fields values must be strings")
            if value in {FIELD_ALL, FIELD_ADVECTED}:
                selected.update(range(len(tracer_names)))
            elif value.startswith(FIELD_PREFIX) and value[len(FIELD_PREFIX) :] in tracer_by_name:
                selected.add(tracer_by_name[value[len(FIELD_PREFIX) :]])
            else:
                # Use the full parser in the fill pass for the established error text.
                _parse_fields(field_raw, tracer_names, label)
        field_total += len(selected)
        time_total += len(
            _operator_sequence(_required(mapping, "time_operator", label), f"{label}.time_operator")
        )
        horizontal_total += len(
            _operator_sequence(
                _required(mapping, "horizontal_operator", label),
                f"{label}.horizontal_operator",
            )
        )
        vertical_raw = _operator_sequence(
            _required(mapping, "vertical_operator", label), f"{label}.vertical_operator"
        )
        for vertical in vertical_raw:
            vertical_total += (
                len(_required(vertical, "values", f"{label}.vertical_operator"))
                if vertical.get("type") == "exact"
                else 1
            )

    arrays = _allocate_numeric(
        entry_count,
        field_total,
        time_total,
        horizontal_total,
        vertical_total,
        field_total,
    )
    ids: list[str] = []
    field_names: list[str] = []
    seen_ids: set[str] = set()
    field_offset = time_offset = horizontal_offset = vertical_offset = 0
    for index, raw_entry in enumerate(entries_raw):
        label = f"entries[{index}]"
        mapping = _require_mapping(raw_entry, label)
        entry_id = str(_required(mapping, "id", label))
        if not entry_id:
            raise ValueError(f"{label}.id must not be empty")
        _validate_fixed_width_utf8(entry_id, MAX_ID_LENGTH, f"{label}.id")
        if entry_id in seen_ids:
            raise ValueError(f"duplicate active ObsOperator id {entry_id!r}")
        seen_ids.add(entry_id)
        if "species" in mapping:
            raise ValueError(f"{label}.species is obsolete; use fields")

        field_raw = _required(mapping, "fields", label)
        names, tracers = _parse_fields(field_raw, tracer_names, label)
        field_count = tracers.size
        field_slice = slice(field_offset, field_offset + field_count)
        arrays["entry_field_start"][index] = field_offset
        arrays["entry_field_count"][index] = field_count
        arrays["field_tracer"][field_slice] = tracers
        arrays["field_to_accumulator"][field_slice] = np.arange(
            field_offset, field_offset + field_count, dtype=np.int64
        )
        arrays["accumulator"][field_slice] = 0.0
        field_names.extend(names)
        field_offset += field_count

        time_raw = _operator_sequence(
            _required(mapping, "time_operator", label), f"{label}.time_operator"
        )
        arrays["time_operator_start"][index] = time_offset
        arrays["time_operator_count"][index] = len(time_raw)
        entry_end_us = np.iinfo(np.int64).min
        for component_index, time_component in enumerate(time_raw):
            time_bounds, time_weight = _parse_time_operator(
                time_component,
                label=f"{label}.time_operator[{component_index}]",
                simulation_start=simulation_start,
                transport_dt_s=transport_dt_s,
            )
            arrays["time_operator_bounds_us"][time_offset] = time_bounds
            arrays["time_operator_weight"][time_offset] = time_weight
            entry_end_us = max(entry_end_us, int(time_bounds[1]))
            time_offset += 1
        arrays["entry_end_us"][index] = entry_end_us

        horizontal_raw = _operator_sequence(
            _required(mapping, "horizontal_operator", label),
            f"{label}.horizontal_operator",
        )
        arrays["horizontal_operator_start"][index] = horizontal_offset
        arrays["horizontal_operator_count"][index] = len(horizontal_raw)
        for component_index, horizontal_component in enumerate(horizontal_raw):
            (
                horizontal_bounds,
                horizontal_type,
                horizontal_weight,
                horizontal_normalization,
            ) = _parse_horizontal_operator(
                horizontal_component,
                label=f"{label}.horizontal_operator[{component_index}]",
                grid=grid,
            )
            arrays["horizontal_operator_bounds"][horizontal_offset] = horizontal_bounds
            arrays["horizontal_weight_type"][horizontal_offset] = horizontal_type
            arrays["horizontal_weight"][horizontal_offset] = horizontal_weight
            arrays["horizontal_normalization"][horizontal_offset] = horizontal_normalization
            horizontal_offset += 1

        vertical_raw = _operator_sequence(
            _required(mapping, "vertical_operator", label), f"{label}.vertical_operator"
        )
        arrays["vertical_operator_start"][index] = vertical_offset
        entry_vertical_count = 0
        for component_index, vertical_component in enumerate(vertical_raw):
            vertical_type, vertical_unit, vertical_bounds, vertical_weight_type, vertical_weight = (
                _parse_vertical_operators(
                    vertical_component,
                    label=f"{label}.vertical_operator[{component_index}]",
                    nlev=grid.shape[0],
                )
            )
            vertical_count = vertical_weight.size
            vertical_slice = slice(vertical_offset, vertical_offset + vertical_count)
            arrays["vertical_operator_type"][vertical_slice] = vertical_type
            arrays["vertical_operator_unit"][vertical_slice] = vertical_unit
            arrays["vertical_operator_bounds"][vertical_slice] = vertical_bounds
            arrays["vertical_weight_type"][vertical_slice] = vertical_weight_type
            arrays["vertical_weight"][vertical_slice] = vertical_weight
            vertical_offset += vertical_count
            entry_vertical_count += vertical_count
        arrays["vertical_operator_count"][index] = entry_vertical_count
        ids.append(entry_id)

    order = np.argsort(arrays["entry_end_us"], kind="stable")
    plan = ObsPlan(ids=tuple(ids), field_names=tuple(field_names), **arrays)
    if not np.array_equal(order, np.arange(entry_count)):
        from wombat_transport.obsoperator.state import _copy_ordered_plans, empty_obs_plan

        plan = _copy_ordered_plans(
            plan,
            empty_obs_plan(),
            np.zeros(entry_count, dtype=np.int8),
            order.astype(np.int64, copy=False),
            boundary_us=-1,
        )
    plan.validate()
    return plan


# Private compatibility alias for callers outside this package during the transition.
_load_obsoperator_array_state = _load_obs_plan


def _operator_sequence(raw: Any, label: str) -> tuple[dict[str, Any], ...]:
    values = raw if isinstance(raw, list) else [raw]
    if not values:
        raise TypeError(f"{label} must be a mapping or nonempty sequence of mappings")
    return tuple(_require_mapping(value, label) for value in values)

def _parse_fields(raw: Any, tracer_names: tuple[str, ...], label: str) -> tuple[tuple[str, ...], np.ndarray]:
    requested = [raw] if isinstance(raw, str) else raw
    if not isinstance(requested, list) or not requested:
        raise TypeError(f"{label}.fields must be a field name or nonempty sequence")
    tracer_by_name = {name: index for index, name in enumerate(tracer_names)}
    selected: list[int] = []
    seen: set[int] = set()
    for value in requested:
        if not isinstance(value, str):
            raise TypeError(f"{label}.fields values must be strings")
        if value in {FIELD_ALL, FIELD_ADVECTED}:
            indices = range(len(tracer_names))
        elif value.startswith(FIELD_PREFIX):
            tracer_name = value[len(FIELD_PREFIX) :]
            if tracer_name not in tracer_by_name:
                raise ValueError(f"{label}.fields contains unknown tracer {tracer_name!r}")
            indices = (tracer_by_name[tracer_name],)
        else:
            raise ValueError(f"{label}.fields value {value!r} must start with {FIELD_PREFIX}")
        for tracer_index in indices:
            if tracer_index not in seen:
                selected.append(tracer_index)
                seen.add(tracer_index)
    names = tuple(f"{FIELD_PREFIX}{tracer_names[index]}" for index in selected)
    for name in names:
        _validate_fixed_width_utf8(name, MAX_FIELD_NAME_LENGTH, f"{label}.fields value")
    return names, np.asarray(selected, dtype=np.int64)


def _parse_time_operator(
    raw: Any,
    *,
    label: str,
    simulation_start: datetime,
    transport_dt_s: float,
) -> tuple[np.ndarray, float]:
    mapping = _require_mapping(raw, label)
    operator_type = str(_required(mapping, "type", label))
    if operator_type not in {"point", "range"}:
        raise ValueError(f"{label}.type must be 'point' or 'range'")
    unit = str(_required(mapping, "unit", label))
    if unit not in {"time", "time_index"}:
        raise ValueError(f"{label}.unit must be 'time' or 'time_index'")
    if operator_type == "point":
        start = end = _parse_time_value(
            _required(mapping, "time", label),
            unit,
            label,
            simulation_start,
            transport_dt_s,
            interval_end=True,
        )
        weighting = "normalized"
    else:
        raw_start = _required(mapping, "start", label)
        raw_end = _required(mapping, "end", label)
        start = _parse_time_value(
            raw_start,
            unit,
            label,
            simulation_start,
            transport_dt_s,
            interval_end=False,
        )
        end = _parse_time_value(
            raw_end,
            unit,
            label,
            simulation_start,
            transport_dt_s,
            interval_end=True,
        )
        weighting = str(mapping.get("weights", "normalized"))
        if weighting not in {"normalized", "equal"}:
            raise ValueError(f"{label}.weights must be 'normalized' or 'equal'")
        if unit == "time" and raw_start == raw_end:
            start = end
    if start > end:
        raise ValueError(f"{label} start must not exceed end")
    start_us = _datetime_to_microseconds(simulation_start)
    dt_us = _seconds_to_microseconds(transport_dt_s, "transport timestep")
    count = end - start + 1
    weight = 1.0 / count if weighting == "normalized" else 1.0
    bounds = np.asarray(
        [start_us + start * dt_us, start_us + (end + 1) * dt_us],
        dtype=np.int64,
    )
    return bounds, weight


def _parse_time_value(
    raw: Any,
    unit: str,
    label: str,
    simulation_start: datetime,
    transport_dt_s: float,
    *,
    interval_end: bool,
) -> int:
    if unit == "time_index":
        return _integer(raw, f"{label} time index")
    if not isinstance(raw, list) or len(raw) != 2:
        raise TypeError(f"{label} time values must be [YYYYMMDD, HHMM]")
    date_value = _integer(raw[0], f"{label} date")
    clock_value = _integer(raw[1], f"{label} clock")
    try:
        timestamp = datetime.strptime(f"{date_value:08d} {clock_value:04d}", "%Y%m%d %H%M")
    except ValueError as exc:
        raise ValueError(f"{label} contains invalid date/time {raw!r}") from exc
    elapsed = (timestamp - simulation_start).total_seconds()
    if interval_end:
        return max(math.ceil(elapsed / float(transport_dt_s)) - 1, 0)
    return math.floor(elapsed / float(transport_dt_s))


def _parse_horizontal_operator(
    raw: Any,
    *,
    label: str,
    grid: TransportGrid,
) -> tuple[np.ndarray, int, float, float]:
    mapping = _require_mapping(raw, label)
    operator_type = str(_required(mapping, "type", label))
    if operator_type not in {"point", "box"}:
        raise ValueError(f"{label}.type must be 'point' or 'box'")
    unit = str(mapping.get("unit", "degrees"))
    if unit not in {"degrees", "grid_index"}:
        raise ValueError(f"{label}.unit must be 'degrees' or 'grid_index'")
    weighting = str(mapping.get("weights", "normalized_area"))
    if weighting not in {"area", "normalized_area", "normalized", "equal"}:
        raise ValueError(f"{label}.weights is invalid")

    if operator_type == "point":
        lon_start = lon_end = _horizontal_index(mapping, "longitude", unit, grid.lon_deg, "longitude", label)
        lat_start = lat_end = _horizontal_index(mapping, "latitude", unit, grid.lat_deg, "latitude", label)
    else:
        lon_start = _horizontal_index(mapping, "longitude_start", unit, grid.lon_deg, "longitude", label)
        lon_end = _horizontal_index(mapping, "longitude_end", unit, grid.lon_deg, "longitude", label)
        lat_start = _horizontal_index(mapping, "latitude_start", unit, grid.lat_deg, "latitude", label)
        lat_end = _horizontal_index(mapping, "latitude_end", unit, grid.lat_deg, "latitude", label)
    if lon_start > lon_end or lat_start > lat_end:
        raise ValueError(f"{label} box start must not exceed end")

    bounds = np.asarray(
        [[lat_start, lat_end + 1], [lon_start, lon_end + 1]],
        dtype=np.int32,
    )
    if weighting == "normalized_area":
        normalization = float(
            np.sum(grid.area_m2[lat_start : lat_end + 1, lon_start : lon_end + 1])
        )
    elif weighting == "normalized":
        normalization = float((lat_end - lat_start + 1) * (lon_end - lon_start + 1))
    else:
        normalization = 1.0
    return bounds, HORIZONTAL_WEIGHTING_CODES[weighting], 1.0, normalization


def _horizontal_index(
    mapping: dict[str, Any],
    key: str,
    unit: str,
    centers: np.ndarray,
    axis: str,
    label: str,
) -> int:
    raw = _required(mapping, key, label)
    if unit == "grid_index":
        index = _integer(raw, f"{label}.{key}")
        if index < 1 or index > centers.size:
            raise ValueError(f"{label}.{key} must be between 1 and {centers.size}")
        return index - 1
    value = float(raw)
    lower, upper = (-180.0, 180.0) if axis == "longitude" else (-90.0, 90.0)
    if value < lower or value > upper:
        raise ValueError(f"{label}.{key} must be between {lower:g} and {upper:g} degrees")
    spacing = float(np.max(np.diff(centers)))
    origin = -180.0 if axis == "longitude" else -90.0
    index = math.floor((value - origin) / spacing + 1.5) - 1
    if axis == "longitude" and index >= centers.size:
        index -= centers.size
    return min(max(index, 0), centers.size - 1)


def _parse_vertical_operators(
    raw: Any,
    *,
    label: str,
    nlev: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mapping = _require_mapping(raw, label)
    operator_type = str(_required(mapping, "type", label))
    if operator_type not in {"point", "range", "exact"}:
        raise ValueError(f"{label}.type must be 'point', 'range', or 'exact'")
    unit = str(mapping.get("unit", "pressure"))
    if unit not in {"pressure", "altitude", "pressure_level"}:
        raise ValueError(f"{label}.unit is invalid")

    if operator_type == "exact":
        values_raw = _required(mapping, "values", label)
        weights_raw = _required(mapping, "weights", label)
        if not isinstance(values_raw, list) or not isinstance(weights_raw, list) or not values_raw:
            raise TypeError(f"{label}.values and weights must be nonempty sequences")
        if len(values_raw) != len(weights_raw):
            raise ValueError(f"{label}.values and weights must have the same length")
        values = _vertical_values(values_raw, unit, nlev, f"{label}.values")
        weights = np.asarray([float(value) for value in weights_raw], dtype=np.float64)
        _validate_vertical_values(values, weights, unit, nlev, label)
        count = values.size
        bounds = np.column_stack((values, values)).astype(np.float64, copy=False)
        return (
            np.full(count, VERTICAL_TYPE_CODES["exact"], dtype=np.int8),
            np.full(count, VERTICAL_UNIT_CODES[unit], dtype=np.int8),
            bounds,
            np.full(count, VERTICAL_WEIGHTING_CODES["exact"], dtype=np.int8),
            weights,
        )

    weighting = str(mapping.get("weights", "normalized_pressure"))
    if weighting not in {"normalized_pressure", "pressure", "normalized", "equal"}:
        raise ValueError(f"{label}.weights is invalid")
    if operator_type == "point":
        start = end = _vertical_value(_required(mapping, "value", label), unit, nlev, f"{label}.value")
    else:
        start = _vertical_value(_required(mapping, "start", label), unit, nlev, f"{label}.start")
        end = _vertical_value(_required(mapping, "end", label), unit, nlev, f"{label}.end")
    _validate_vertical_bounds(float(start), float(end), unit, nlev, label)
    return (
        np.asarray([VERTICAL_TYPE_CODES["range"]], dtype=np.int8),
        np.asarray([VERTICAL_UNIT_CODES[unit]], dtype=np.int8),
        np.asarray([[float(start), float(end)]], dtype=np.float64),
        np.asarray([VERTICAL_WEIGHTING_CODES[weighting]], dtype=np.int8),
        np.ones(1, dtype=np.float64),
    )


def _vertical_values(raw: list[Any], unit: str, nlev: int, label: str) -> np.ndarray:
    values = [_vertical_value(value, unit, nlev, label) for value in raw]
    dtype = np.int64 if unit == "pressure_level" else np.float64
    return np.asarray(values, dtype=dtype)


def _vertical_value(raw: Any, unit: str, nlev: int, label: str) -> float | int:
    if unit == "pressure_level":
        value = _integer(raw, label)
        if value < 1 or value > nlev:
            raise ValueError(f"{label} must be between 1 and {nlev}")
        return value
    value = float(raw)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return value



def _open_yaml_text(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")

def _require_mapping(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError(f"{label} must be a mapping")
    return raw


def _required(raw: dict[str, Any], key: str, label: str) -> Any:
    if key not in raw:
        raise KeyError(f"{label}.{key} is required")
    return raw[key]


def _integer(raw: Any, label: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    return int(raw)
