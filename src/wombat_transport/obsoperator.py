from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import gzip
import hashlib
import logging
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, TextIO

import netCDF4
import numpy as np
import yaml

from wombat_transport.grid import TransportGrid
from wombat_transport.met_diagnostics import airqnt_diagnostics_from_forcing
from wombat_transport.output import OutputSnapshot
from wombat_transport.run_config import RunConfig, simulation_start

logger = logging.getLogger(__name__)

MAX_ID_LENGTH = 255
MAX_FIELD_NAME_LENGTH = 64
FIELD_PREFIX = "SpeciesConcVV_"
FIELD_ALL = "SpeciesConcVV_?ALL?"
FIELD_ADVECTED = "SpeciesConcVV_?ADV?"
RESTART_FORMAT = "Wombat ObsOperator restart"
RESTART_FORMAT_VERSION = 1
MICROSECONDS_PER_SECOND = 1_000_000

HORIZONTAL_WEIGHTING_CODES = {"area": 0, "normalized_area": 1, "normalized": 2, "equal": 3}
VERTICAL_TYPE_CODES = {"range": 0, "exact": 1}
VERTICAL_UNIT_CODES = {"pressure": 0, "altitude": 1, "pressure_level": 2}
VERTICAL_WEIGHTING_CODES = {"normalized_pressure": 0, "pressure": 1, "normalized": 2, "equal": 3}
RESTART_VARIABLE_SPECS = {
    "id": ("S1", ("entries", "id_chars")),
    "definition_hash": ("S1", ("entries", "hash_chars")),
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


class _ObsOperatorYamlLoader(yaml.SafeLoader):
    yaml_implicit_resolvers = {
        key: list(value) for key, value in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }


def _construct_decimal_int(loader: yaml.SafeLoader, node: yaml.Node) -> int:
    return int(loader.construct_scalar(node).replace("_", ""), 10)


for _resolver_key, _resolvers in _ObsOperatorYamlLoader.yaml_implicit_resolvers.items():
    _ObsOperatorYamlLoader.yaml_implicit_resolvers[_resolver_key] = [
        resolver for resolver in _resolvers if resolver[0] != "tag:yaml.org,2002:int"
    ]
_ObsOperatorYamlLoader.add_implicit_resolver(
    "tag:yaml.org,2002:int",
    re.compile(r"^[-+]?[0-9][0-9_]*$"),
    list("-+0123456789"),
)
_ObsOperatorYamlLoader.add_constructor("tag:yaml.org,2002:int", _construct_decimal_int)


@dataclass(frozen=True)
class ObsOperatorConfig:
    activate: bool = False
    verbose: bool = False
    input_file: str | None = None
    output_file: str | None = None
    restart_file: str | None = None
    restart_missing: str = "warn"


@dataclass(frozen=True)
class TimeOperator:
    indices: np.ndarray
    weights: np.ndarray
    times_us: np.ndarray


@dataclass(frozen=True)
class HorizontalOperator:
    indices: np.ndarray
    weights: np.ndarray
    weighting: str


@dataclass(frozen=True)
class VerticalOperator:
    operator_type: str
    unit: str
    weights: str | np.ndarray
    start: float | int | None = None
    end: float | int | None = None
    values: np.ndarray | None = None


@dataclass
class ObsOperatorEntry:
    id: str
    definition_hash: str
    field_names: tuple[str, ...]
    field_indices: np.ndarray
    time: TimeOperator
    horizontal: HorizontalOperator
    vertical: VerticalOperator
    field_values: np.ndarray = field(init=False)
    active: bool = True

    def __post_init__(self) -> None:
        self.field_values = np.zeros(len(self.field_names), dtype=np.float64)


class ObsOperatorManager:
    def __init__(
        self,
        *,
        root: Path,
        config: ObsOperatorConfig,
        start: datetime,
        transport_dt_s: float,
        tracer_names: tuple[str, ...],
        grid: TransportGrid,
    ) -> None:
        if (
            not config.activate
            or config.input_file is None
            or config.output_file is None
            or config.restart_file is None
        ):
            raise ValueError("an active ObsOperator manager requires input_file, output_file, and restart_file")
        self._root = root
        self._config = config
        self._start = start
        self._transport_dt_s = float(transport_dt_s)
        _seconds_to_microseconds(transport_dt_s, "transport timestep")
        self._tracer_names = tracer_names
        self._grid = grid
        self._previous_input_path: Path | None = None
        self._current_output_path: Path | None = None
        self._writer: _ObsOperatorNetCDFWriter | None = None
        self._active_entries: list[ObsOperatorEntry] = []
        self._schedule: dict[int, list[tuple[ObsOperatorEntry, float]]] = {}
        self._entry_ids: set[str] = set()
        self._restart_entry_ids: set[str] = set()
        self._closed = False
        self._load_restart()

    @classmethod
    def from_run_config(
        cls,
        config: RunConfig,
        *,
        tracer_names: tuple[str, ...],
        grid: TransportGrid,
        transport_dt_s: float,
    ) -> ObsOperatorManager | None:
        parsed = parse_obsoperator_config(config.outputs)
        if not parsed.activate:
            return None
        return cls(
            root=config.root,
            config=parsed,
            start=simulation_start(config),
            transport_dt_s=transport_dt_s,
            tracer_names=tracer_names,
            grid=grid,
        )

    def sample(self, *, step_start: datetime, time_index: int, snapshot: OutputSnapshot) -> None:
        self._ensure_open()
        self._initialize_for_date(step_start)
        step_time_us = _datetime_to_microseconds(step_start)
        for entry, time_weight in self._schedule.pop(step_time_us, ()):
            if not entry.active:
                continue
            sampled = sample_obsoperator_entry(entry, snapshot, self._grid)
            entry.field_values += time_weight * sampled
            _consume_entry_time(entry, step_time_us)
            if self._config.verbose:
                logger.info("obsoperator_sample id=%s time_index=%d", entry.id, time_index)
            if entry.time.times_us.size == 0:
                self._finalize_entry(entry)

    def close(self, *, boundary_time: datetime) -> None:
        if self._closed:
            return
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        boundary_us = _datetime_to_microseconds(boundary_time)
        unfinished = tuple(entry for entry in self._active_entries if entry.active)
        for entry in unfinished:
            if entry.time.times_us.size == 0 or np.any(entry.time.times_us < boundary_us):
                raise ValueError(
                    f"ObsOperator entry {entry.id!r} has an invalid remaining schedule at restart boundary "
                    f"{boundary_time.isoformat()}"
                )
        restart_path = _resolve_template_path(self._root, self._config.restart_file, boundary_time)
        _write_obsoperator_restart(
            restart_path,
            entries=unfinished,
            restart_time=boundary_time,
            transport_dt_s=self._transport_dt_s,
            grid=self._grid,
        )
        self._closed = True

    def _initialize_for_date(self, timestamp: datetime) -> None:
        input_path = _resolve_template_path(self._root, self._config.input_file, timestamp)
        if input_path == self._previous_input_path:
            return
        self._previous_input_path = input_path

        if input_path.is_file():
            entries = load_obsoperator_entries(
                input_path,
                tracer_names=self._tracer_names,
                grid=self._grid,
                simulation_start=self._start,
                transport_dt_s=self._transport_dt_s,
            )
            current_time_us = _datetime_to_microseconds(timestamp)
            for entry in entries:
                if entry.id in self._restart_entry_ids:
                    existing = next(value for value in self._active_entries if value.active and value.id == entry.id)
                    if existing.definition_hash != entry.definition_hash:
                        raise ValueError(f"daily ObsOperator entry {entry.id!r} conflicts with its restart state")
                    logger.debug("obsoperator_restart_daily_duplicate_skipped id=%s", entry.id)
                    continue
                self._register_entry(entry, earliest_time_us=current_time_us)
            logger.info("obsoperator_input_loaded path=%s entries=%d", input_path, len(entries))
        else:
            logger.info("obsoperator_input_missing path=%s", input_path)

        output_path = _resolve_template_path(self._root, self._config.output_file, timestamp)
        if output_path != self._current_output_path:
            if self._writer is not None:
                self._writer.close()
            self._writer = None
            self._current_output_path = output_path

    def _register_entry(self, entry: ObsOperatorEntry, *, earliest_time_us: int) -> None:
        if entry.id in self._entry_ids:
            raise ValueError(f"duplicate active ObsOperator id {entry.id!r}")
        if entry.time.times_us.size == 0:
            raise ValueError(f"ObsOperator entry {entry.id!r} has no remaining sampling times")
        if np.any(entry.time.times_us < earliest_time_us):
            raise ValueError(
                f"ObsOperator entry {entry.id!r} has sampling times before the current run position; "
                "a matching ObsOperator restart is required"
            )
        self._active_entries.append(entry)
        self._entry_ids.add(entry.id)
        for time_us, time_weight in zip(entry.time.times_us, entry.time.weights, strict=True):
            self._schedule.setdefault(int(time_us), []).append((entry, float(time_weight)))

    def _finalize_entry(self, entry: ObsOperatorEntry) -> None:
        if self._current_output_path is None:
            raise ValueError(f"cannot finalize ObsOperator entry {entry.id!r} without an output path")
        if self._writer is None:
            self._writer = _ObsOperatorNetCDFWriter(self._current_output_path)
        self._writer.write_entry(entry)
        entry.active = False
        self._entry_ids.discard(entry.id)
        self._restart_entry_ids.discard(entry.id)

    def _ensure_open(self) -> None:
        if self._closed:
            raise ValueError("cannot sample with a closed ObsOperator manager")

    def _load_restart(self) -> None:
        restart_path = _resolve_template_path(self._root, self._config.restart_file, self._start)
        if not restart_path.is_file():
            message = f"ObsOperator restart missing: {restart_path}"
            if self._config.restart_missing == "error":
                raise FileNotFoundError(message)
            if self._config.restart_missing == "warn":
                logger.warning(message)
            return
        entries = _read_obsoperator_restart(
            restart_path,
            restart_time=self._start,
            transport_dt_s=self._transport_dt_s,
            tracer_names=self._tracer_names,
            grid=self._grid,
        )
        start_us = _datetime_to_microseconds(self._start)
        for entry in entries:
            self._register_entry(entry, earliest_time_us=start_us)
            self._restart_entry_ids.add(entry.id)
        logger.info("obsoperator_restart_loaded path=%s entries=%d", restart_path, len(entries))


def parse_obsoperator_config(outputs: dict[str, Any]) -> ObsOperatorConfig:
    raw = outputs.get("obsoperator", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("outputs.obsoperator must be a mapping")
    activate = bool(raw.get("activate", False))
    verbose = bool(raw.get("verbose", False))
    input_file = _optional_config_string(raw, "input_file")
    output_file = _optional_config_string(raw, "output_file")
    restart_file = _optional_config_string(raw, "restart_file")
    restart_missing = str(raw.get("restart_missing", "warn"))
    if restart_missing not in {"warn", "error", "ignore"}:
        raise ValueError("outputs.obsoperator.restart_missing must be 'warn', 'error', or 'ignore'")
    if activate and input_file is None:
        raise KeyError("outputs.obsoperator.input_file is required when ObsOperator is active")
    if activate and output_file is None:
        raise KeyError("outputs.obsoperator.output_file is required when ObsOperator is active")
    if activate and restart_file is None:
        raise KeyError("outputs.obsoperator.restart_file is required when ObsOperator is active")
    return ObsOperatorConfig(
        activate=activate,
        verbose=verbose,
        input_file=input_file,
        output_file=output_file,
        restart_file=restart_file,
        restart_missing=restart_missing,
    )


def expand_obsoperator_template(template: str, timestamp: datetime) -> str:
    return (
        str(template)
        .replace("YYYY", f"{timestamp.year:04d}")
        .replace("MM", f"{timestamp.month:02d}")
        .replace("DD", f"{timestamp.day:02d}")
        .replace("hh", f"{timestamp.hour:02d}")
        .replace("mm", f"{timestamp.minute:02d}")
        .replace("ss", f"{timestamp.second:02d}")
    )


def load_obsoperator_entries(
    path: str | Path,
    *,
    tracer_names: tuple[str, ...],
    grid: TransportGrid,
    simulation_start: datetime,
    transport_dt_s: float,
) -> tuple[ObsOperatorEntry, ...]:
    input_path = Path(path)
    with _open_yaml_text(input_path) as handle:
        raw = yaml.load(handle, Loader=_ObsOperatorYamlLoader) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"ObsOperator input {input_path} must contain a mapping")
    if "entries" not in raw:
        raise KeyError(f"ObsOperator input {input_path} is missing entries")
    entries_raw = raw["entries"]
    if entries_raw is None:
        entries_raw = []
    if not isinstance(entries_raw, list):
        raise TypeError(f"ObsOperator input {input_path} entries must be a sequence")
    return tuple(
        _parse_entry(
            value,
            entry_index=index,
            tracer_names=tracer_names,
            grid=grid,
            simulation_start=simulation_start,
            transport_dt_s=transport_dt_s,
        )
        for index, value in enumerate(entries_raw)
    )


def sample_obsoperator_entry(
    entry: ObsOperatorEntry,
    snapshot: OutputSnapshot,
    grid: TransportGrid,
) -> np.ndarray:
    state_bottom = np.asarray(snapshot.state.data[0, ::-1, :, :, :], dtype=np.float64)
    diagnostics = airqnt_diagnostics_from_forcing(snapshot.forcing, grid)
    wet_edges = diagnostics.wet_pressure_edges_hpa
    wet_delp = wet_edges[:-1] - wet_edges[1:]
    box_height = diagnostics.box_height_m
    result = np.zeros(len(entry.field_names), dtype=np.float64)

    for horizontal_offset, (lat_index, lon_index) in enumerate(entry.horizontal.indices):
        vertical_value = _sample_vertical(
            entry.vertical,
            state_bottom[:, lat_index, lon_index, :][:, entry.field_indices],
            wet_edges[:, lat_index, lon_index],
            wet_delp[:, lat_index, lon_index],
            box_height[:, lat_index, lon_index],
        )
        result += entry.horizontal.weights[horizontal_offset] * vertical_value
    return result


def _sample_vertical(
    operator: VerticalOperator,
    field_values: np.ndarray,
    pressure_edges_hpa: np.ndarray,
    pressure_thickness_hpa: np.ndarray,
    box_height_m: np.ndarray,
) -> np.ndarray:
    if operator.operator_type == "exact":
        if operator.values is None or not isinstance(operator.weights, np.ndarray):
            raise ValueError("invalid exact vertical operator")
        result = np.zeros(field_values.shape[1], dtype=np.float64)
        for value, weight in zip(operator.values, operator.weights, strict=True):
            level = _vertical_level(operator.unit, value, pressure_edges_hpa, box_height_m)
            result += float(weight) * field_values[level]
        return result

    if operator.start is None or operator.end is None or not isinstance(operator.weights, str):
        raise ValueError("invalid range vertical operator")
    if operator.unit == "pressure":
        level_start = _pressure_level(float(operator.end), pressure_edges_hpa)
        level_end = _pressure_level(float(operator.start), pressure_edges_hpa)
    else:
        level_start = _vertical_level(operator.unit, operator.start, pressure_edges_hpa, box_height_m)
        level_end = _vertical_level(operator.unit, operator.end, pressure_edges_hpa, box_height_m)
    if level_start > level_end:
        raise ValueError("vertical operator resolves to an inverted level range")

    selected = field_values[level_start : level_end + 1]
    if operator.weights in {"pressure", "normalized_pressure"}:
        weights = pressure_thickness_hpa[level_start : level_end + 1]
    else:
        weights = np.ones(selected.shape[0], dtype=np.float64)
    result = np.sum(selected * weights[:, np.newaxis], axis=0)
    if operator.weights in {"normalized", "normalized_pressure"}:
        result /= np.sum(weights)
    return result


def _vertical_level(
    unit: str,
    value: float | int,
    pressure_edges_hpa: np.ndarray,
    box_height_m: np.ndarray,
) -> int:
    if unit == "pressure":
        return _pressure_level(float(value), pressure_edges_hpa)
    if unit == "altitude":
        cumulative = np.cumsum(box_height_m)
        matches = np.flatnonzero(cumulative >= float(value))
        if matches.size == 0:
            raise ValueError(f"vertical altitude {value} m exceeds the modeled column")
        return int(matches[0])
    return int(value) - 1


def _pressure_level(pressure_hpa: float, pressure_edges_hpa: np.ndarray) -> int:
    matches = np.flatnonzero(pressure_edges_hpa[:-1] <= pressure_hpa)
    if matches.size == 0:
        return pressure_edges_hpa.size - 2
    return max(int(matches[0]) - 1, 0)


def _parse_entry(
    raw: Any,
    *,
    entry_index: int,
    tracer_names: tuple[str, ...],
    grid: TransportGrid,
    simulation_start: datetime,
    transport_dt_s: float,
) -> ObsOperatorEntry:
    label = f"entries[{entry_index}]"
    mapping = _require_mapping(raw, label)
    entry_id = str(_required(mapping, "id", label))
    if not entry_id:
        raise ValueError(f"{label}.id must not be empty")
    if len(entry_id) > MAX_ID_LENGTH:
        raise ValueError(f"{label}.id exceeds {MAX_ID_LENGTH} characters")
    if "species" in mapping:
        raise ValueError(f"{label}.species is obsolete; use fields")
    field_names, field_indices = _parse_fields(_required(mapping, "fields", label), tracer_names, label)
    return ObsOperatorEntry(
        id=entry_id,
        definition_hash=hashlib.sha256(yaml.safe_dump(mapping, sort_keys=True).encode("utf-8")).hexdigest(),
        field_names=field_names,
        field_indices=field_indices,
        time=_parse_time_operator(
            _required(mapping, "time_operator", label),
            label=f"{label}.time_operator",
            simulation_start=simulation_start,
            transport_dt_s=transport_dt_s,
        ),
        horizontal=_parse_horizontal_operator(
            _required(mapping, "horizontal_operator", label),
            label=f"{label}.horizontal_operator",
            grid=grid,
        ),
        vertical=_parse_vertical_operator(
            _required(mapping, "vertical_operator", label),
            label=f"{label}.vertical_operator",
            nlev=grid.shape[0],
        ),
    )


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
    if any(len(name) > MAX_FIELD_NAME_LENGTH for name in names):
        raise ValueError(f"{label}.fields contains a name exceeding {MAX_FIELD_NAME_LENGTH} characters")
    return names, np.asarray(selected, dtype=np.int64)


def _parse_time_operator(
    raw: Any,
    *,
    label: str,
    simulation_start: datetime,
    transport_dt_s: float,
) -> TimeOperator:
    mapping = _require_mapping(raw, label)
    operator_type = str(_required(mapping, "type", label))
    if operator_type not in {"point", "range"}:
        raise ValueError(f"{label}.type must be 'point' or 'range'")
    unit = str(_required(mapping, "unit", label))
    if unit not in {"time", "time_index"}:
        raise ValueError(f"{label}.unit must be 'time' or 'time_index'")
    if operator_type == "point":
        start = end = _parse_time_value(
            _required(mapping, "time", label), unit, label, simulation_start, transport_dt_s
        )
        weighting = "normalized"
    else:
        start = _parse_time_value(
            _required(mapping, "start", label), unit, label, simulation_start, transport_dt_s
        )
        end = _parse_time_value(_required(mapping, "end", label), unit, label, simulation_start, transport_dt_s)
        weighting = str(mapping.get("weights", "normalized"))
        if weighting not in {"normalized", "equal"}:
            raise ValueError(f"{label}.weights must be 'normalized' or 'equal'")
    if start > end:
        raise ValueError(f"{label} start must not exceed end")
    indices = np.arange(start, end + 1, dtype=np.int64)
    if weighting == "normalized":
        weights = np.full(indices.size, 1.0 / indices.size, dtype=np.float64)
    else:
        weights = np.ones(indices.size, dtype=np.float64)
    start_us = _datetime_to_microseconds(simulation_start)
    dt_us = _seconds_to_microseconds(transport_dt_s, "transport timestep")
    times_us = start_us + indices * dt_us
    return TimeOperator(indices=indices, weights=weights, times_us=times_us)


def _parse_time_value(
    raw: Any,
    unit: str,
    label: str,
    simulation_start: datetime,
    transport_dt_s: float,
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
    return math.floor((timestamp - simulation_start).total_seconds() / float(transport_dt_s))


def _parse_horizontal_operator(raw: Any, *, label: str, grid: TransportGrid) -> HorizontalOperator:
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

    indices = np.asarray(
        [(lat_index, lon_index) for lon_index in range(lon_start, lon_end + 1) for lat_index in range(lat_start, lat_end + 1)],
        dtype=np.int64,
    )
    if weighting in {"area", "normalized_area"}:
        weights = grid.area_m2[indices[:, 0], indices[:, 1]].astype(np.float64, copy=True)
    else:
        weights = np.ones(indices.shape[0], dtype=np.float64)
    if weighting in {"normalized_area", "normalized"}:
        weights /= np.sum(weights)
    return HorizontalOperator(indices=indices, weights=weights, weighting=weighting)


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


def _parse_vertical_operator(raw: Any, *, label: str, nlev: int) -> VerticalOperator:
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
        return VerticalOperator(operator_type="exact", unit=unit, values=values, weights=weights)

    weighting = str(mapping.get("weights", "normalized_pressure"))
    if weighting not in {"normalized_pressure", "pressure", "normalized", "equal"}:
        raise ValueError(f"{label}.weights is invalid")
    if operator_type == "point":
        start = end = _vertical_value(_required(mapping, "value", label), unit, nlev, f"{label}.value")
    else:
        start = _vertical_value(_required(mapping, "start", label), unit, nlev, f"{label}.start")
        end = _vertical_value(_required(mapping, "end", label), unit, nlev, f"{label}.end")
    if start > end:
        raise ValueError(f"{label} start must not exceed end")
    return VerticalOperator(operator_type="range", unit=unit, start=start, end=end, weights=weighting)


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
    if value < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _consume_entry_time(entry: ObsOperatorEntry, time_us: int) -> None:
    matches = np.flatnonzero(entry.time.times_us == time_us)
    if matches.size != 1:
        raise ValueError(f"ObsOperator entry {entry.id!r} does not contain scheduled time {time_us}")
    keep = np.ones(entry.time.times_us.size, dtype=bool)
    keep[int(matches[0])] = False
    entry.time = TimeOperator(
        indices=entry.time.indices[keep],
        weights=entry.time.weights[keep],
        times_us=entry.time.times_us[keep],
    )


def _datetime_to_microseconds(value: datetime) -> int:
    epoch = datetime(1970, 1, 1)
    delta = value - epoch
    return (delta.days * 86400 + delta.seconds) * MICROSECONDS_PER_SECOND + delta.microseconds


def _seconds_to_microseconds(value: float, label: str) -> int:
    seconds = float(value)
    if not np.isfinite(seconds) or seconds <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    microseconds = int(round(seconds * MICROSECONDS_PER_SECOND))
    if not math.isclose(microseconds / MICROSECONDS_PER_SECOND, seconds, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"{label} must be representable to microsecond precision")
    return microseconds


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


def _write_obsoperator_restart(
    path: Path,
    *,
    entries: tuple[ObsOperatorEntry, ...],
    restart_time: datetime,
    transport_dt_s: float,
    grid: TransportGrid,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        _write_obsoperator_restart_file(
            temporary_path,
            entries=entries,
            restart_time=restart_time,
            transport_dt_s=transport_dt_s,
            grid=grid,
        )
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    logger.info("obsoperator_restart_written path=%s entries=%d", path, len(entries))


def _write_obsoperator_restart_file(
    path: Path,
    *,
    entries: tuple[ObsOperatorEntry, ...],
    restart_time: datetime,
    transport_dt_s: float,
    grid: TransportGrid,
) -> None:
    field_count = sum(len(entry.field_names) for entry in entries)
    time_count = sum(entry.time.times_us.size for entry in entries)
    exact_count = sum(
        0 if entry.vertical.operator_type != "exact" or entry.vertical.values is None else entry.vertical.values.size
        for entry in entries
    )
    with netCDF4.Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.set_fill_off()
        dataset.setncattr("format", RESTART_FORMAT)
        dataset.setncattr("format_version", np.int32(RESTART_FORMAT_VERSION))
        dataset.setncattr("restart_time_us", np.int64(_datetime_to_microseconds(restart_time)))
        dataset.setncattr("transport_timestep_seconds", np.float64(transport_dt_s))
        dataset.setncattr("grid_signature", _grid_signature(grid))
        _create_restart_dimensions(dataset, len(entries), field_count, time_count, exact_count)
        variables = _create_restart_variables(dataset)

        field_offset = 0
        time_offset = 0
        exact_offset = 0
        for entry_index, entry in enumerate(entries):
            variables["id"][entry_index, :] = _nul_padded_chars(entry.id, MAX_ID_LENGTH)
            variables["definition_hash"][entry_index, :] = _nul_padded_chars(entry.definition_hash, 64)
            variables["field_start"][entry_index] = field_offset
            variables["field_count"][entry_index] = len(entry.field_names)
            for field_name, accumulator in zip(entry.field_names, entry.field_values, strict=True):
                variables["field_name"][field_offset, :] = _nul_padded_chars(field_name, MAX_FIELD_NAME_LENGTH)
                variables["field_accumulator"][field_offset] = accumulator
                field_offset += 1

            variables["time_start"][entry_index] = time_offset
            variables["time_count"][entry_index] = entry.time.times_us.size
            time_slice = slice(time_offset, time_offset + entry.time.times_us.size)
            variables["remaining_time_us"][time_slice] = entry.time.times_us
            variables["remaining_time_weight"][time_slice] = entry.time.weights
            time_offset += entry.time.times_us.size

            lat_indices = entry.horizontal.indices[:, 0]
            lon_indices = entry.horizontal.indices[:, 1]
            variables["horizontal_bounds"][entry_index, :] = np.asarray(
                [lon_indices.min(), lon_indices.max(), lat_indices.min(), lat_indices.max()], dtype=np.int32
            )
            variables["horizontal_weighting"][entry_index] = HORIZONTAL_WEIGHTING_CODES[entry.horizontal.weighting]

            vertical = entry.vertical
            variables["vertical_type"][entry_index] = VERTICAL_TYPE_CODES[vertical.operator_type]
            variables["vertical_unit"][entry_index] = VERTICAL_UNIT_CODES[vertical.unit]
            variables["vertical_start"][entry_index] = exact_offset
            if vertical.operator_type == "exact":
                assert vertical.values is not None and isinstance(vertical.weights, np.ndarray)
                count = vertical.values.size
                variables["vertical_weighting"][entry_index] = -1
                variables["vertical_bounds"][entry_index, :] = np.asarray([np.nan, np.nan])
                variables["vertical_count"][entry_index] = count
                exact_slice = slice(exact_offset, exact_offset + count)
                variables["vertical_value"][exact_slice] = vertical.values
                variables["vertical_weight"][exact_slice] = vertical.weights
                exact_offset += count
            else:
                assert vertical.start is not None and vertical.end is not None and isinstance(vertical.weights, str)
                variables["vertical_weighting"][entry_index] = VERTICAL_WEIGHTING_CODES[vertical.weights]
                variables["vertical_bounds"][entry_index, :] = np.asarray([vertical.start, vertical.end])
                variables["vertical_count"][entry_index] = 0


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
    dataset.createDimension("hash_chars", 64)
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
) -> tuple[ObsOperatorEntry, ...]:
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
) -> tuple[ObsOperatorEntry, ...]:
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
        "entries", "entry_fields", "remaining_times", "vertical_values", "id_chars", "field_chars", "hash_chars",
        "horizontal_bound", "vertical_bound",
    }
    required_variables = set(RESTART_VARIABLE_SPECS)
    if not required_dimensions.issubset(dataset.dimensions) or not required_variables.issubset(dataset.variables):
        raise ValueError(f"ObsOperator restart {path} is missing required dimensions or variables")
    if (
        len(dataset.dimensions["id_chars"]) != MAX_ID_LENGTH
        or len(dataset.dimensions["field_chars"]) != MAX_FIELD_NAME_LENGTH
        or len(dataset.dimensions["hash_chars"]) != 64
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
    definition_hashes = _decode_nul_padded_rows(dataset.variables["definition_hash"][:], 64, "definition_hash")
    field_names = _decode_nul_padded_rows(dataset.variables["field_name"][:], MAX_FIELD_NAME_LENGTH, "field_name")
    if len(ids) != entry_count or len(definition_hashes) != entry_count or len(field_names) != field_total:
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
    entries: list[ObsOperatorEntry] = []
    seen_ids: set[str] = set()
    for index, entry_id in enumerate(ids):
        if not entry_id or entry_id in seen_ids:
            raise ValueError(f"ObsOperator restart {path} contains an empty or duplicate id {entry_id!r}")
        seen_ids.add(entry_id)
        definition_hash = definition_hashes[index]
        if not re.fullmatch(r"[0-9a-f]{64}", definition_hash):
            raise ValueError(f"ObsOperator restart entry {entry_id!r} has an invalid definition hash")
        field_slice = _ragged_slice(field_starts, field_counts, index)
        entry_field_names = tuple(field_names[field_slice])
        if len(set(entry_field_names)) != len(entry_field_names):
            raise ValueError(f"ObsOperator restart entry {entry_id!r} contains duplicate fields")
        try:
            field_indices = np.asarray([tracer_by_field[name] for name in entry_field_names], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"ObsOperator restart entry {entry_id!r} requires missing field {exc.args[0]!r}") from exc

        time_slice = _ragged_slice(time_starts, time_counts, index)
        entry_times = remaining_times[time_slice].copy()
        entry_weights = remaining_weights[time_slice].copy()
        if np.any(entry_times < expected_time_us) or np.any(np.diff(entry_times) <= 0):
            raise ValueError(f"ObsOperator restart entry {entry_id!r} has invalid remaining timestamps")
        deltas = entry_times - expected_time_us
        if np.any(deltas % dt_us != 0):
            raise ValueError(f"ObsOperator restart entry {entry_id!r} has times not aligned to the transport timestep")
        time = TimeOperator(indices=deltas // dt_us, weights=entry_weights, times_us=entry_times)

        horizontal_name = _code_name(horizontal_codes[index], horizontal_names, "horizontal weighting", entry_id)
        horizontal = _horizontal_from_bounds(horizontal_bounds[index], horizontal_name, grid, entry_id)
        vertical_type = _code_name(vertical_types[index], vertical_type_names, "vertical type", entry_id)
        vertical_unit = _code_name(vertical_units[index], vertical_unit_names, "vertical unit", entry_id)
        vertical_slice = _ragged_slice(vertical_starts, vertical_counts, index)
        if vertical_type == "exact":
            if vertical_counts[index] <= 0 or vertical_weightings[index] != -1:
                raise ValueError(f"ObsOperator restart entry {entry_id!r} has invalid exact vertical state")
            values = vertical_values[vertical_slice].copy()
            weights = vertical_weights[vertical_slice].copy()
            _validate_restart_vertical_values(values, weights, vertical_unit, grid.shape[0], entry_id)
            vertical = VerticalOperator(operator_type="exact", unit=vertical_unit, values=values, weights=weights)
        else:
            if vertical_counts[index] != 0:
                raise ValueError(f"ObsOperator restart entry {entry_id!r} has unexpected exact vertical values")
            weighting = _code_name(
                vertical_weightings[index], vertical_weighting_names, "vertical weighting", entry_id
            )
            start, end = vertical_bounds[index]
            _validate_restart_vertical_bounds(start, end, vertical_unit, grid.shape[0], entry_id)
            vertical = VerticalOperator(
                operator_type="range", unit=vertical_unit, weights=weighting, start=float(start), end=float(end)
            )

        entry = ObsOperatorEntry(
            id=entry_id,
            definition_hash=definition_hash,
            field_names=entry_field_names,
            field_indices=field_indices,
            time=time,
            horizontal=horizontal,
            vertical=vertical,
        )
        entry.field_values[:] = accumulators[field_slice]
        entries.append(entry)
    return tuple(entries)


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


def _horizontal_from_bounds(
    bounds: np.ndarray,
    weighting: str,
    grid: TransportGrid,
    entry_id: str,
) -> HorizontalOperator:
    lon_start, lon_end, lat_start, lat_end = (int(value) for value in bounds)
    if not (0 <= lon_start <= lon_end < grid.lon_deg.size and 0 <= lat_start <= lat_end < grid.lat_deg.size):
        raise ValueError(f"ObsOperator restart entry {entry_id!r} has invalid horizontal bounds")
    indices = np.asarray(
        [(lat, lon) for lon in range(lon_start, lon_end + 1) for lat in range(lat_start, lat_end + 1)],
        dtype=np.int64,
    )
    if weighting in {"area", "normalized_area"}:
        weights = grid.area_m2[indices[:, 0], indices[:, 1]].astype(np.float64, copy=True)
    else:
        weights = np.ones(indices.shape[0], dtype=np.float64)
    if weighting in {"normalized_area", "normalized"}:
        weights /= np.sum(weights)
    return HorizontalOperator(indices=indices, weights=weights, weighting=weighting)


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


class _ObsOperatorNetCDFWriter:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._dataset: netCDF4.Dataset | None = None
        self._field_indices: dict[str, int] = {}
        self._field_names: list[str] = []
        self._entry_index = 0
        self._sample_index = 0

    def write_entry(self, entry: ObsOperatorEntry) -> None:
        self._ensure_created()
        assert self._dataset is not None
        field_indices: list[int] = []
        for name in entry.field_names:
            if name not in self._field_indices:
                self._field_indices[name] = len(self._field_names) + 1
                self._field_names.append(name)
            field_indices.append(self._field_indices[name])

        self._dataset.variables["id"][self._entry_index, :] = _nul_padded_chars(entry.id, MAX_ID_LENGTH)
        sample_slice = slice(self._sample_index, self._sample_index + len(entry.field_names))
        self._dataset.variables["id_index"][sample_slice] = self._entry_index + 1
        self._dataset.variables["field_index"][sample_slice] = np.asarray(field_indices, dtype=np.int32)
        self._dataset.variables["sample"][sample_slice] = entry.field_values.astype(np.float32)
        self._entry_index += 1
        self._sample_index += len(entry.field_names)

    def close(self) -> None:
        if self._dataset is None:
            return
        field_variable = self._dataset.variables["field"]
        for index, name in enumerate(self._field_names):
            field_variable[index, :] = _nul_padded_chars(name, MAX_FIELD_NAME_LENGTH)
        self._dataset.close()
        self._dataset = None

    def _ensure_created(self) -> None:
        if self._dataset is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        dataset = netCDF4.Dataset(self._path, "w", format="NETCDF4")
        dataset.set_fill_off()
        dataset.createDimension("entries", None)
        dataset.createDimension("id_chars", MAX_ID_LENGTH)
        dataset.createDimension("fields", None)
        dataset.createDimension("field_chars", MAX_FIELD_NAME_LENGTH)
        dataset.createDimension("samples", None)
        _create_variable(
            dataset,
            "id",
            "S1",
            ("entries", "id_chars"),
            long_name="ids",
            description="id",
        )
        _create_variable(
            dataset,
            "field",
            "S1",
            ("fields", "field_chars"),
            long_name="fields",
            description="field name",
        )
        _create_variable(
            dataset,
            "id_index",
            "i4",
            ("samples",),
            long_name="id_index",
            description="index of the id in the id list",
        )
        _create_variable(
            dataset,
            "field_index",
            "i4",
            ("samples",),
            long_name="field_index",
            description="index of the field in the field list",
        )
        _create_variable(
            dataset,
            "sample",
            "f4",
            ("samples",),
            long_name="samples",
            description="sample of the id and field",
        )
        self._dataset = dataset


def _create_variable(
    dataset: netCDF4.Dataset,
    name: str,
    dtype: str,
    dimensions: tuple[str, ...],
    *,
    long_name: str,
    description: str,
) -> netCDF4.Variable:
    variable = dataset.createVariable(name, dtype, dimensions, zlib=True, complevel=1, shuffle=True)
    variable.long_name = long_name
    variable.units = "1"
    variable.description = description
    return variable


def _nul_padded_chars(value: str, length: int) -> np.ndarray:
    encoded = value.encode("utf-8")
    if len(encoded) > length:
        raise ValueError(f"encoded string exceeds fixed width {length}")
    output = np.full(length, b"\x00", dtype="S1")
    output[: len(encoded)] = np.frombuffer(encoded, dtype="S1")
    return output


def _resolve_template_path(root: Path, template: str | None, timestamp: datetime) -> Path:
    if template is None:
        raise ValueError("ObsOperator path template is missing")
    path = Path(expand_obsoperator_template(template, timestamp))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _open_yaml_text(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _optional_config_string(raw: dict[str, Any], key: str) -> str | None:
    if key not in raw or raw[key] is None:
        return None
    value = str(raw[key]).strip()
    return value or None


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
