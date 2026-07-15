from __future__ import annotations

from datetime import datetime
import hashlib
import logging
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import netCDF4
import numpy as np

from wombat_transport.grid import TransportGrid
from wombat_transport.obsoperator.state import (
    FIELD_PREFIX,
    HORIZONTAL_WEIGHTING_CODES,
    MAX_FIELD_NAME_LENGTH,
    MAX_ID_LENGTH,
    VERTICAL_TYPE_CODES,
    VERTICAL_UNIT_CODES,
    VERTICAL_WEIGHTING_CODES,
    _ObsOperatorArrayState,
    _PreparedObsOperators,
)
from wombat_transport.obsoperator.utils import (
    _datetime_to_microseconds,
    _nul_padded_matrix,
    _seconds_to_microseconds,
)

logger = logging.getLogger(__name__)

RESTART_FORMAT = "Wombat ObsOperator restart"
RESTART_FORMAT_VERSION = 2
RESTART_VARIABLE_SPECS = {
    "id": ("S1", ("entries", "id_chars")),
    "field_start": ("i8", ("entries",)),
    "field_count": ("i4", ("entries",)),
    "time_start": ("i8", ("entries",)),
    "time_count": ("i4", ("entries",)),
    "horizontal_bounds": ("i4", ("entries", "horizontal_bound")),
    "horizontal_weighting": ("i1", ("entries",)),
    "vertical_type": ("i1", ("entries",)),
    "vertical_unit": ("i1", ("entries",)),
    "vertical_weighting": ("i1", ("entries",)),
    "vertical_bounds": ("f8", ("entries", "vertical_bound")),
    "vertical_start": ("i8", ("entries",)),
    "vertical_count": ("i4", ("entries",)),
    "field_name": ("S1", ("entry_fields", "field_chars")),
    "field_accumulator": ("f8", ("entry_fields",)),
    "remaining_time_us": ("i8", ("remaining_times",)),
    "remaining_time_weight": ("f8", ("remaining_times",)),
    "vertical_value": ("f8", ("vertical_values",)),
    "vertical_weight": ("f8", ("vertical_values",)),
}

def _grid_signature(grid: TransportGrid) -> str:
    digest = hashlib.sha256()
    for name, values in (
        ("lat_deg", grid.lat_deg),
        ("lon_deg", grid.lon_deg),
        ("lev", grid.lev),
        ("area_m2", grid.area_m2),
        ("hyai_hpa", grid.hyai_hpa),
        ("hybi", grid.hybi),
    ):
        array = np.ascontiguousarray(values, dtype=np.float64)
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


def _write_obsoperator_restart_states(
    path: Path,
    *,
    states: list[_ObsOperatorArrayState],
    restart_time: datetime,
    transport_dt_s: float,
    grid: TransportGrid,
) -> None:
    arrays = _restart_snapshot_arrays_from_states(states)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        _write_obsoperator_restart_arrays_file(
            temporary_path,
            arrays=arrays,
            restart_time=restart_time,
            transport_dt_s=transport_dt_s,
            grid=grid,
        )
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    logger.info("obsoperator_restart_written path=%s entries=%d", path, arrays["field_count"].size)


def _write_obsoperator_restart_arrays_file(
    path: Path,
    *,
    arrays: dict[str, np.ndarray],
    restart_time: datetime,
    transport_dt_s: float,
    grid: TransportGrid,
) -> None:
    field_count = arrays["field_accumulator"].size
    time_count = arrays["remaining_time_us"].size
    exact_count = arrays["vertical_value"].size
    entry_count = arrays["field_count"].size
    with netCDF4.Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.set_fill_off()
        dataset.setncattr("format", RESTART_FORMAT)
        dataset.setncattr("format_version", np.int32(RESTART_FORMAT_VERSION))
        dataset.setncattr("restart_time_us", np.int64(_datetime_to_microseconds(restart_time)))
        dataset.setncattr("transport_timestep_seconds", np.float64(transport_dt_s))
        dataset.setncattr("grid_signature", _grid_signature(grid))
        _create_restart_dimensions(dataset, entry_count, field_count, time_count, exact_count)
        variables = _create_restart_variables(dataset)
        for name, values in arrays.items():
            variables[name][:] = values


def _restart_snapshot_arrays_from_states(states: list[_ObsOperatorArrayState]) -> dict[str, np.ndarray]:
    active_entries = [
        (state, int(entry_index))
        for state in states
        for entry_index in np.flatnonzero(state.active)
    ]
    entry_count = len(active_entries)
    field_count = sum(int(state.prepared.entry_field_count[index]) for state, index in active_entries)
    time_count = sum(
        int(state.time_count[index] - state.time_consumed[index]) for state, index in active_entries
    )
    exact_count = sum(int(state.prepared.entry_exact_count[index]) for state, index in active_entries)
    arrays = {
        "id": _nul_padded_matrix((state.ids[index] for state, index in active_entries), MAX_ID_LENGTH, entry_count),
        "field_start": np.empty(entry_count, dtype=np.int64),
        "field_count": np.empty(entry_count, dtype=np.int32),
        "time_start": np.empty(entry_count, dtype=np.int64),
        "time_count": np.empty(entry_count, dtype=np.int32),
        "horizontal_bounds": np.empty((entry_count, 4), dtype=np.int32),
        "horizontal_weighting": np.empty(entry_count, dtype=np.int8),
        "vertical_type": np.empty(entry_count, dtype=np.int8),
        "vertical_unit": np.empty(entry_count, dtype=np.int8),
        "vertical_weighting": np.empty(entry_count, dtype=np.int8),
        "vertical_bounds": np.empty((entry_count, 2), dtype=np.float64),
        "vertical_start": np.empty(entry_count, dtype=np.int64),
        "vertical_count": np.empty(entry_count, dtype=np.int32),
        "field_name": np.empty((field_count, MAX_FIELD_NAME_LENGTH), dtype="S1"),
        "field_accumulator": np.empty(field_count, dtype=np.float64),
        "remaining_time_us": np.empty(time_count, dtype=np.int64),
        "remaining_time_weight": np.empty(time_count, dtype=np.float64),
        "vertical_value": np.empty(exact_count, dtype=np.float64),
        "vertical_weight": np.empty(exact_count, dtype=np.float64),
    }

    field_offset = 0
    time_offset = 0
    exact_offset = 0
    for output_index, (state, entry_index) in enumerate(active_entries):
        prepared = state.prepared
        source_field_start = int(prepared.entry_field_start[entry_index])
        count = int(prepared.entry_field_count[entry_index])
        source_field_slice = slice(source_field_start, source_field_start + count)
        output_field_slice = slice(field_offset, field_offset + count)
        arrays["field_start"][output_index] = field_offset
        arrays["field_count"][output_index] = count
        arrays["field_name"][output_field_slice, :] = _nul_padded_matrix(
            state.field_names[entry_index], MAX_FIELD_NAME_LENGTH, count
        )
        arrays["field_accumulator"][output_field_slice] = state.field_accumulator[source_field_slice]
        field_offset += count

        source_time_start = int(state.time_start[entry_index] + state.time_consumed[entry_index])
        source_time_end = int(state.time_start[entry_index] + state.time_count[entry_index])
        count = source_time_end - source_time_start
        output_time_slice = slice(time_offset, time_offset + count)
        arrays["time_start"][output_index] = time_offset
        arrays["time_count"][output_index] = count
        arrays["remaining_time_us"][output_time_slice] = state.remaining_time_us[
            source_time_start:source_time_end
        ]
        arrays["remaining_time_weight"][output_time_slice] = state.remaining_time_weight[
            source_time_start:source_time_end
        ]
        time_offset += count

        horizontal_start = int(prepared.entry_horizontal_start[entry_index])
        horizontal_end = horizontal_start + int(prepared.entry_horizontal_count[entry_index])
        latitudes = prepared.horizontal_lat[horizontal_start:horizontal_end]
        longitudes = prepared.horizontal_lon[horizontal_start:horizontal_end]
        arrays["horizontal_bounds"][output_index, :] = np.asarray(
            [longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], dtype=np.int32
        )
        arrays["horizontal_weighting"][output_index] = state.horizontal_weighting[entry_index]

        arrays["vertical_type"][output_index] = prepared.entry_vertical_type[entry_index]
        arrays["vertical_unit"][output_index] = prepared.entry_vertical_unit[entry_index]
        count = int(prepared.entry_exact_count[entry_index])
        arrays["vertical_start"][output_index] = exact_offset
        arrays["vertical_count"][output_index] = count
        arrays["vertical_weighting"][output_index] = prepared.entry_vertical_weighting[entry_index]
        arrays["vertical_bounds"][output_index, :] = np.asarray(
            [prepared.entry_vertical_lower[entry_index], prepared.entry_vertical_upper[entry_index]],
            dtype=np.float64,
        )
        if count:
            source_exact_start = int(prepared.entry_exact_start[entry_index])
            source_exact_slice = slice(source_exact_start, source_exact_start + count)
            output_exact_slice = slice(exact_offset, exact_offset + count)
            arrays["vertical_value"][output_exact_slice] = prepared.exact_value[source_exact_slice]
            arrays["vertical_weight"][output_exact_slice] = prepared.exact_weight[source_exact_slice]
            exact_offset += count
    return arrays


def _create_restart_dimensions(
    dataset: netCDF4.Dataset,
    entries: int,
    fields: int,
    times: int,
    vertical_values: int,
) -> None:
    dataset.createDimension("entries", entries or None)
    dataset.createDimension("entry_fields", fields or None)
    dataset.createDimension("remaining_times", times or None)
    dataset.createDimension("vertical_values", vertical_values or None)
    dataset.createDimension("id_chars", MAX_ID_LENGTH)
    dataset.createDimension("field_chars", MAX_FIELD_NAME_LENGTH)
    dataset.createDimension("horizontal_bound", 4)
    dataset.createDimension("vertical_bound", 2)


def _create_restart_variables(dataset: netCDF4.Dataset) -> dict[str, netCDF4.Variable]:
    variables = {
        name: dataset.createVariable(name, dtype, dimensions, zlib=True, complevel=1, shuffle=True)
        for name, (dtype, dimensions) in RESTART_VARIABLE_SPECS.items()
    }
    variables["horizontal_weighting"].codes = _enum_description(HORIZONTAL_WEIGHTING_CODES)
    variables["vertical_type"].codes = _enum_description(VERTICAL_TYPE_CODES)
    variables["vertical_unit"].codes = _enum_description(VERTICAL_UNIT_CODES)
    variables["vertical_weighting"].codes = "-1=exact," + _enum_description(VERTICAL_WEIGHTING_CODES)
    variables["remaining_time_us"].units = "microseconds since 1970-01-01 00:00:00"
    variables["horizontal_bounds"].description = "zero-based lon_start, lon_end, lat_start, lat_end"
    return variables


def _enum_description(values: dict[str, int]) -> str:
    return ",".join(f"{code}={name}" for name, code in values.items())


def _read_obsoperator_restart(
    path: Path,
    *,
    restart_time: datetime,
    transport_dt_s: float,
    tracer_names: tuple[str, ...],
    grid: TransportGrid,
) -> _ObsOperatorArrayState:
    try:
        with netCDF4.Dataset(path) as dataset:
            dataset.set_auto_mask(False)
            return _read_obsoperator_restart_dataset(
                dataset,
                path=path,
                restart_time=restart_time,
                transport_dt_s=transport_dt_s,
                tracer_names=tracer_names,
                grid=grid,
            )
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"cannot read ObsOperator restart {path}: {exc}") from exc


def _read_obsoperator_restart_dataset(
    dataset: netCDF4.Dataset,
    *,
    path: Path,
    restart_time: datetime,
    transport_dt_s: float,
    tracer_names: tuple[str, ...],
    grid: TransportGrid,
) -> _ObsOperatorArrayState:
    if getattr(dataset, "format", None) != RESTART_FORMAT:
        raise ValueError(f"ObsOperator restart {path} has an invalid format")
    if int(getattr(dataset, "format_version", -1)) != RESTART_FORMAT_VERSION:
        raise ValueError(f"ObsOperator restart {path} has an unsupported format version")
    expected_time_us = _datetime_to_microseconds(restart_time)
    if int(getattr(dataset, "restart_time_us", -1)) != expected_time_us:
        raise ValueError(f"ObsOperator restart {path} does not match simulation start {restart_time.isoformat()}")
    stored_dt = float(getattr(dataset, "transport_timestep_seconds", np.nan))
    if stored_dt != float(transport_dt_s):
        raise ValueError(
            f"ObsOperator restart is incompatible: transport timestep changed from {stored_dt:g} s "
            f"to {float(transport_dt_s):g} s"
        )
    if getattr(dataset, "grid_signature", None) != _grid_signature(grid):
        raise ValueError("ObsOperator restart is incompatible: transport grid changed")

    required_dimensions = {
        "entries", "entry_fields", "remaining_times", "vertical_values", "id_chars", "field_chars",
        "horizontal_bound", "vertical_bound",
    }
    required_variables = set(RESTART_VARIABLE_SPECS)
    if not required_dimensions.issubset(dataset.dimensions) or not required_variables.issubset(dataset.variables):
        raise ValueError(f"ObsOperator restart {path} is missing required dimensions or variables")
    if (
        len(dataset.dimensions["id_chars"]) != MAX_ID_LENGTH
        or len(dataset.dimensions["field_chars"]) != MAX_FIELD_NAME_LENGTH
    ):
        raise ValueError(f"ObsOperator restart {path} has invalid string dimensions")
    if len(dataset.dimensions["horizontal_bound"]) != 4 or len(dataset.dimensions["vertical_bound"]) != 2:
        raise ValueError(f"ObsOperator restart {path} has invalid operator dimensions")
    for name, (dtype, dimensions) in RESTART_VARIABLE_SPECS.items():
        variable = dataset.variables[name]
        if variable.dtype != np.dtype(dtype) or variable.dimensions != dimensions:
            raise ValueError(f"ObsOperator restart variable {name!r} has an invalid dtype or dimension order")

    entry_count = len(dataset.dimensions["entries"])
    field_total = len(dataset.dimensions["entry_fields"])
    time_total = len(dataset.dimensions["remaining_times"])
    vertical_total = len(dataset.dimensions["vertical_values"])
    ids = _decode_nul_padded_rows(dataset.variables["id"][:], MAX_ID_LENGTH, "id")
    field_names = _decode_nul_padded_rows(dataset.variables["field_name"][:], MAX_FIELD_NAME_LENGTH, "field_name")
    if len(ids) != entry_count or len(field_names) != field_total:
        raise ValueError(f"ObsOperator restart {path} has inconsistent string arrays")

    field_starts = _restart_array(dataset, "field_start", np.int64, (entry_count,))
    field_counts = _restart_array(dataset, "field_count", np.int64, (entry_count,))
    time_starts = _restart_array(dataset, "time_start", np.int64, (entry_count,))
    time_counts = _restart_array(dataset, "time_count", np.int64, (entry_count,))
    vertical_starts = _restart_array(dataset, "vertical_start", np.int64, (entry_count,))
    vertical_counts = _restart_array(dataset, "vertical_count", np.int64, (entry_count,))
    _validate_contiguous_ragged(field_starts, field_counts, field_total, "fields", require_nonempty=True)
    _validate_contiguous_ragged(time_starts, time_counts, time_total, "remaining times", require_nonempty=True)
    _validate_contiguous_ragged(vertical_starts, vertical_counts, vertical_total, "vertical values", require_nonempty=False)

    accumulators = _restart_array(dataset, "field_accumulator", np.float64, (field_total,))
    remaining_times = _restart_array(dataset, "remaining_time_us", np.int64, (time_total,))
    remaining_weights = _restart_array(dataset, "remaining_time_weight", np.float64, (time_total,))
    horizontal_bounds = _restart_array(dataset, "horizontal_bounds", np.int64, (entry_count, 4))
    horizontal_codes = _restart_array(dataset, "horizontal_weighting", np.int64, (entry_count,))
    vertical_types = _restart_array(dataset, "vertical_type", np.int64, (entry_count,))
    vertical_units = _restart_array(dataset, "vertical_unit", np.int64, (entry_count,))
    vertical_weightings = _restart_array(dataset, "vertical_weighting", np.int64, (entry_count,))
    vertical_bounds = _restart_array(dataset, "vertical_bounds", np.float64, (entry_count, 2))
    vertical_values = _restart_array(dataset, "vertical_value", np.float64, (vertical_total,))
    vertical_weights = _restart_array(dataset, "vertical_weight", np.float64, (vertical_total,))
    if not np.all(np.isfinite(accumulators)) or not np.all(np.isfinite(remaining_weights)):
        raise ValueError(f"ObsOperator restart {path} contains non-finite accumulators or time weights")
    if np.any(remaining_weights <= 0.0):
        raise ValueError(f"ObsOperator restart {path} contains non-positive time weights")

    horizontal_names = _reverse_codes(HORIZONTAL_WEIGHTING_CODES)
    vertical_type_names = _reverse_codes(VERTICAL_TYPE_CODES)
    vertical_unit_names = _reverse_codes(VERTICAL_UNIT_CODES)
    vertical_weighting_names = _reverse_codes(VERTICAL_WEIGHTING_CODES)
    tracer_by_field = {f"{FIELD_PREFIX}{name}": index for index, name in enumerate(tracer_names)}
    dt_us = _seconds_to_microseconds(transport_dt_s, "transport timestep")
    seen_ids: set[str] = set()
    entry_field_name_rows: list[tuple[str, ...]] = []
    field_indices = np.empty(field_total, dtype=np.int64)
    entry_horizontal_start = np.empty(entry_count, dtype=np.int64)
    entry_horizontal_count = np.empty(entry_count, dtype=np.int32)
    horizontal_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    entry_vertical_type = np.empty(entry_count, dtype=np.int8)
    entry_vertical_unit = np.empty(entry_count, dtype=np.int8)
    entry_vertical_weighting = np.empty(entry_count, dtype=np.int8)
    entry_vertical_lower = np.empty(entry_count, dtype=np.float64)
    entry_vertical_upper = np.empty(entry_count, dtype=np.float64)
    max_field_count = 0
    horizontal_offset = 0
    for index, entry_id in enumerate(ids):
        if not entry_id or entry_id in seen_ids:
            raise ValueError(f"ObsOperator restart {path} contains an empty or duplicate id {entry_id!r}")
        seen_ids.add(entry_id)
        field_slice = _ragged_slice(field_starts, field_counts, index)
        entry_field_names = tuple(field_names[field_slice])
        if len(set(entry_field_names)) != len(entry_field_names):
            raise ValueError(f"ObsOperator restart entry {entry_id!r} contains duplicate fields")
        try:
            field_indices[field_slice] = [tracer_by_field[name] for name in entry_field_names]
        except KeyError as exc:
            raise ValueError(f"ObsOperator restart entry {entry_id!r} requires missing field {exc.args[0]!r}") from exc
        entry_field_name_rows.append(entry_field_names)
        max_field_count = max(max_field_count, len(entry_field_names))

        time_slice = _ragged_slice(time_starts, time_counts, index)
        entry_times = remaining_times[time_slice].copy()
        entry_weights = remaining_weights[time_slice].copy()
        if np.any(entry_times < expected_time_us) or np.any(np.diff(entry_times) <= 0):
            raise ValueError(f"ObsOperator restart entry {entry_id!r} has invalid remaining timestamps")
        deltas = entry_times - expected_time_us
        if np.any(deltas % dt_us != 0):
            raise ValueError(f"ObsOperator restart entry {entry_id!r} has times not aligned to the transport timestep")
        horizontal_name = _code_name(horizontal_codes[index], horizontal_names, "horizontal weighting", entry_id)
        horizontal_lat, horizontal_lon, horizontal_weight = _horizontal_arrays_from_bounds(
            horizontal_bounds[index], horizontal_name, grid, entry_id
        )
        horizontal_count = horizontal_lat.size
        entry_horizontal_start[index] = horizontal_offset
        entry_horizontal_count[index] = horizontal_count
        horizontal_parts.append((horizontal_lat, horizontal_lon, horizontal_weight))
        horizontal_offset += horizontal_count

        vertical_type = _code_name(vertical_types[index], vertical_type_names, "vertical type", entry_id)
        vertical_unit = _code_name(vertical_units[index], vertical_unit_names, "vertical unit", entry_id)
        entry_vertical_type[index] = int(vertical_types[index])
        entry_vertical_unit[index] = int(vertical_units[index])
        vertical_slice = _ragged_slice(vertical_starts, vertical_counts, index)
        if vertical_type == "exact":
            if vertical_counts[index] <= 0 or vertical_weightings[index] != -1:
                raise ValueError(f"ObsOperator restart entry {entry_id!r} has invalid exact vertical state")
            values = vertical_values[vertical_slice].copy()
            weights = vertical_weights[vertical_slice].copy()
            _validate_restart_vertical_values(values, weights, vertical_unit, grid.shape[0], entry_id)
            entry_vertical_weighting[index] = -1
            entry_vertical_lower[index] = np.nan
            entry_vertical_upper[index] = np.nan
        else:
            if vertical_counts[index] != 0:
                raise ValueError(f"ObsOperator restart entry {entry_id!r} has unexpected exact vertical values")
            weighting = _code_name(
                vertical_weightings[index], vertical_weighting_names, "vertical weighting", entry_id
            )
            start, end = vertical_bounds[index]
            _validate_restart_vertical_bounds(start, end, vertical_unit, grid.shape[0], entry_id)
            entry_vertical_weighting[index] = int(vertical_weightings[index])
            entry_vertical_lower[index] = start
            entry_vertical_upper[index] = end

    if horizontal_parts:
        horizontal_lat = np.concatenate([part[0] for part in horizontal_parts])
        horizontal_lon = np.concatenate([part[1] for part in horizontal_parts])
        horizontal_weight = np.concatenate([part[2] for part in horizontal_parts])
    else:
        horizontal_lat = np.empty(0, dtype=np.int32)
        horizontal_lon = np.empty(0, dtype=np.int32)
        horizontal_weight = np.empty(0, dtype=np.float64)

    schedule_entry = np.repeat(np.arange(entry_count, dtype=np.int64), time_counts)
    order = np.argsort(remaining_times, kind="stable")
    sorted_times = remaining_times[order]
    schedule_entry = schedule_entry[order]
    schedule_weight = remaining_weights[order]
    schedule_times_us, schedule_start, schedule_count = np.unique(
        sorted_times,
        return_index=True,
        return_counts=True,
    )
    prepared = _PreparedObsOperators(
        entry_field_start=field_starts,
        entry_field_count=field_counts.astype(np.int32, copy=False),
        entry_horizontal_start=entry_horizontal_start,
        entry_horizontal_count=entry_horizontal_count,
        entry_vertical_type=entry_vertical_type,
        entry_vertical_unit=entry_vertical_unit,
        entry_vertical_weighting=entry_vertical_weighting,
        entry_vertical_lower=entry_vertical_lower,
        entry_vertical_upper=entry_vertical_upper,
        entry_exact_start=vertical_starts,
        entry_exact_count=vertical_counts.astype(np.int32, copy=False),
        field_indices=field_indices,
        horizontal_lat=horizontal_lat,
        horizontal_lon=horizontal_lon,
        horizontal_weight=horizontal_weight,
        exact_value=vertical_values,
        exact_weight=vertical_weights,
        max_field_count=max_field_count,
    )
    return _ObsOperatorArrayState(
        ids=tuple(ids),
        field_names=tuple(entry_field_name_rows),
        prepared=prepared,
        field_accumulator=accumulators,
        horizontal_weighting=horizontal_codes.astype(np.int8, copy=False),
        time_start=time_starts,
        time_count=time_counts.astype(np.int32, copy=False),
        time_consumed=np.zeros(entry_count, dtype=np.int32),
        remaining_time_us=remaining_times,
        remaining_time_weight=remaining_weights,
        active=np.ones(entry_count, dtype=bool),
        schedule_times_us=schedule_times_us,
        schedule_start=np.asarray(schedule_start, dtype=np.int64),
        schedule_count=np.asarray(schedule_count, dtype=np.int32),
        schedule_entry=schedule_entry,
        schedule_weight=schedule_weight,
    )


def _restart_array(
    dataset: netCDF4.Dataset,
    name: str,
    dtype: np.dtype[Any] | type[Any],
    shape: tuple[int, ...],
) -> np.ndarray:
    values = np.asarray(dataset.variables[name][:], dtype=dtype)
    if values.shape != shape:
        raise ValueError(f"ObsOperator restart variable {name!r} has shape {values.shape}, expected {shape}")
    return values


def _decode_nul_padded_rows(values: Any, width: int, label: str) -> list[str]:
    array = np.asarray(values, dtype="S1")
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"ObsOperator restart {label} has invalid shape")
    result: list[str] = []
    for row in array:
        encoded = row.tobytes()
        nul = encoded.find(b"\x00")
        if nul < 0:
            nul = len(encoded)
        elif any(encoded[nul:]):
            raise ValueError(f"ObsOperator restart {label} is not NUL padded")
        try:
            result.append(encoded[:nul].decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ValueError(f"ObsOperator restart {label} is not valid UTF-8") from exc
    return result


def _validate_contiguous_ragged(
    starts: np.ndarray,
    counts: np.ndarray,
    total: int,
    label: str,
    *,
    require_nonempty: bool,
) -> None:
    offset = 0
    for start, count in zip(starts, counts, strict=True):
        if int(start) != offset or int(count) < (1 if require_nonempty else 0):
            raise ValueError(f"ObsOperator restart has invalid contiguous {label} offsets")
        offset += int(count)
    if offset != total:
        raise ValueError(f"ObsOperator restart has inconsistent {label} length")


def _ragged_slice(starts: np.ndarray, counts: np.ndarray, index: int) -> slice:
    start = int(starts[index])
    return slice(start, start + int(counts[index]))


def _reverse_codes(codes: dict[str, int]) -> dict[int, str]:
    return {code: name for name, code in codes.items()}


def _code_name(code: int, names: dict[int, str], label: str, entry_id: str) -> str:
    try:
        return names[int(code)]
    except KeyError as exc:
        raise ValueError(f"ObsOperator restart entry {entry_id!r} has invalid {label} code {int(code)}") from exc


def _horizontal_arrays_from_bounds(
    bounds: np.ndarray,
    weighting: str,
    grid: TransportGrid,
    entry_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon_start, lon_end, lat_start, lat_end = (int(value) for value in bounds)
    if not (0 <= lon_start <= lon_end < grid.lon_deg.size and 0 <= lat_start <= lat_end < grid.lat_deg.size):
        raise ValueError(f"ObsOperator restart entry {entry_id!r} has invalid horizontal bounds")
    lon_count = lon_end - lon_start + 1
    lat_count = lat_end - lat_start + 1
    lon = np.repeat(np.arange(lon_start, lon_end + 1, dtype=np.int32), lat_count)
    lat = np.tile(np.arange(lat_start, lat_end + 1, dtype=np.int32), lon_count)
    if weighting in {"area", "normalized_area"}:
        weights = grid.area_m2[lat, lon].astype(np.float64, copy=True)
    else:
        weights = np.ones(lat.size, dtype=np.float64)
    if weighting in {"normalized_area", "normalized"}:
        weights /= np.sum(weights)
    return lat, lon, weights


def _validate_restart_vertical_values(
    values: np.ndarray,
    weights: np.ndarray,
    unit: str,
    nlev: int,
    entry_id: str,
) -> None:
    if values.size == 0 or values.size != weights.size or not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ValueError(f"ObsOperator restart entry {entry_id!r} has invalid exact vertical values")
    if unit == "pressure_level" and (
        np.any(values != np.floor(values)) or np.any(values < 1) or np.any(values > nlev)
    ):
        raise ValueError(f"ObsOperator restart entry {entry_id!r} has invalid pressure levels")
    if unit != "pressure_level" and np.any(values < 0.0):
        raise ValueError(f"ObsOperator restart entry {entry_id!r} has negative vertical values")


def _validate_restart_vertical_bounds(start: float, end: float, unit: str, nlev: int, entry_id: str) -> None:
    if not np.isfinite(start) or not np.isfinite(end) or start > end or start < 0.0:
        raise ValueError(f"ObsOperator restart entry {entry_id!r} has invalid vertical bounds")
    if unit == "pressure_level" and (
        start != math.floor(start) or end != math.floor(end) or start < 1 or end > nlev
    ):
        raise ValueError(f"ObsOperator restart entry {entry_id!r} has invalid pressure-level bounds")
