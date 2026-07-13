from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import gzip
import heapq
import logging
import math
from pathlib import Path
import re
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


@dataclass(frozen=True)
class TimeOperator:
    indices: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class HorizontalOperator:
    indices: np.ndarray
    weights: np.ndarray


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
    field_names: tuple[str, ...]
    field_indices: np.ndarray
    time: TimeOperator
    horizontal: HorizontalOperator
    vertical: VerticalOperator
    field_values: np.ndarray = field(init=False)
    active: bool = True

    def __post_init__(self) -> None:
        self.field_values = np.zeros(len(self.field_names), dtype=np.float64)

    @property
    def max_time_index(self) -> int:
        return int(self.time.indices[-1])


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
        if not config.activate or config.input_file is None or config.output_file is None:
            raise ValueError("an active ObsOperator manager requires input_file and output_file")
        self._root = root
        self._config = config
        self._start = start
        self._transport_dt_s = float(transport_dt_s)
        self._tracer_names = tracer_names
        self._grid = grid
        self._previous_input_path: Path | None = None
        self._current_output_path: Path | None = None
        self._writer: _ObsOperatorNetCDFWriter | None = None
        self._active_entries: list[ObsOperatorEntry] = []
        self._schedule: dict[int, list[ObsOperatorEntry]] = {}
        self._completion_heap: list[tuple[int, int, ObsOperatorEntry]] = []
        self._entry_sequence = 0
        self._closed = False

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
        for entry in self._schedule.pop(int(time_index), ()):
            if not entry.active:
                continue
            offset = int(time_index) - int(entry.time.indices[0])
            sampled = sample_obsoperator_entry(entry, snapshot, self._grid)
            entry.field_values += entry.time.weights[offset] * sampled
            if self._config.verbose:
                logger.info("obsoperator_sample id=%s time_index=%d", entry.id, time_index)

        while self._completion_heap and self._completion_heap[0][0] <= int(time_index):
            _, _, entry = heapq.heappop(self._completion_heap)
            if entry.active:
                self._finalize_entry(entry)

    def close(self) -> None:
        if self._closed:
            return
        for entry in self._active_entries:
            if entry.active:
                self._finalize_entry(entry)
        if self._writer is not None:
            self._writer.close()
            self._writer = None
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
            for entry in entries:
                self._register_entry(entry)
            logger.info("obsoperator_input_loaded path=%s entries=%d", input_path, len(entries))
        else:
            logger.info("obsoperator_input_missing path=%s", input_path)

        output_path = _resolve_template_path(self._root, self._config.output_file, timestamp)
        if output_path != self._current_output_path and any(entry.active for entry in self._active_entries):
            if self._writer is not None:
                self._writer.close()
            self._writer = None
            self._current_output_path = output_path

    def _register_entry(self, entry: ObsOperatorEntry) -> None:
        self._active_entries.append(entry)
        for time_index in entry.time.indices:
            self._schedule.setdefault(int(time_index), []).append(entry)
        heapq.heappush(self._completion_heap, (entry.max_time_index, self._entry_sequence, entry))
        self._entry_sequence += 1

    def _finalize_entry(self, entry: ObsOperatorEntry) -> None:
        if self._current_output_path is None:
            raise ValueError(f"cannot finalize ObsOperator entry {entry.id!r} without an output path")
        if self._writer is None:
            self._writer = _ObsOperatorNetCDFWriter(self._current_output_path)
        self._writer.write_entry(entry)
        entry.active = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise ValueError("cannot sample with a closed ObsOperator manager")


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
    if activate and input_file is None:
        raise KeyError("outputs.obsoperator.input_file is required when ObsOperator is active")
    if activate and output_file is None:
        raise KeyError("outputs.obsoperator.output_file is required when ObsOperator is active")
    return ObsOperatorConfig(
        activate=activate,
        verbose=verbose,
        input_file=input_file,
        output_file=output_file,
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
    return TimeOperator(indices=indices, weights=weights)


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
    return HorizontalOperator(indices=indices, weights=weights)


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
