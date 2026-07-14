from __future__ import annotations

from dataclasses import dataclass
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
from yaml12 import read_yaml

from wombat_transport.constants import AIRMW_G_PER_MOL, G0_M_PER_S2, H2OMW_G_PER_MOL
from wombat_transport.grid import TransportGrid
from wombat_transport.met_diagnostics import RD_J_PER_KG_K
from wombat_transport.output import OutputSnapshot
from wombat_transport.run_config import RunConfig, simulation_start
from wombat_transport.transport.numba_control import numba_enabled

try:  # Optional acceleration path; the same array kernel runs in Python as the reference fallback.
    from numba import njit
except ImportError:  # pragma: no cover - exercised in environments without numba.
    njit = None

logger = logging.getLogger(__name__)

MAX_ID_LENGTH = 255
MAX_FIELD_NAME_LENGTH = 64
FIELD_PREFIX = "SpeciesConcVV_"
FIELD_ALL = "SpeciesConcVV_?ALL?"
FIELD_ADVECTED = "SpeciesConcVV_?ADV?"
RESTART_FORMAT = "Wombat ObsOperator restart"
RESTART_FORMAT_VERSION = 2
MICROSECONDS_PER_SECOND = 1_000_000
OBSOPERATOR_NUMBA_ENV = "WOMBAT_OBSOPERATOR_NUMBA"
SCIENCE_ENTRY_CHUNK = 256
SCIENCE_FIELD_CHUNK = 64
SCIENCE_SAMPLE_CHUNK = 16_384
SCIENCE_STAGE_ENTRIES = SCIENCE_ENTRY_CHUNK
SCIENCE_STAGE_SAMPLES = SCIENCE_SAMPLE_CHUNK

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


@dataclass(frozen=True)
class ObsOperatorConfig:
    activate: bool = False
    verbose: bool = False
    input_file: str | None = None
    output_file: str | None = None
    restart_file: str | None = None
    restart_missing: str = "warn"


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
        self._states: list[_ObsOperatorArrayState] = []
        self._schedule: dict[int, list[tuple[_ObsOperatorArrayState, int, int]]] = {}
        self._entry_ids: set[str] = set()
        self._sample_workspace = np.empty((0, 0), dtype=np.float64)
        use_numba = numba_enabled(OBSOPERATOR_NUMBA_ENV, available=njit is not None)
        self._sampling_kernel = (
            _sample_prepared_entries_numba if use_numba else _sample_prepared_entries_kernel
        )
        self._writer: _ObsOperatorNetCDFWriter | None = None
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
        scheduled_groups = self._schedule.pop(step_time_us, ())
        if not scheduled_groups:
            return
        completed: list[tuple[_ObsOperatorArrayState, np.ndarray]] = []
        for state, schedule_start, schedule_end in scheduled_groups:
            scheduled_entries = state.schedule_entry[schedule_start:schedule_end]
            scheduled_weights = state.schedule_weight[schedule_start:schedule_end]
            active_mask = state.active[scheduled_entries]
            if not np.all(active_mask):
                scheduled_entries = scheduled_entries[active_mask]
                scheduled_weights = scheduled_weights[active_mask]
            if scheduled_entries.size == 0:
                continue
            samples = self._evaluate_entries(state, scheduled_entries, snapshot)
            _accumulate_prepared_samples_numba(
                scheduled_entries,
                scheduled_weights,
                state.prepared.entry_field_start,
                state.prepared.entry_field_count,
                samples,
                state.field_accumulator,
            )
            state.time_consumed[scheduled_entries] += 1
            if self._config.verbose:
                for entry_index in scheduled_entries:
                    logger.info("obsoperator_sample id=%s time_index=%d", state.ids[int(entry_index)], time_index)
            finished = scheduled_entries[
                state.time_consumed[scheduled_entries] == state.time_count[scheduled_entries]
            ]
            finished = finished[state.active[finished]]
            if finished.size:
                state.active[finished] = False
                for entry_index in finished:
                    self._entry_ids.discard(state.ids[int(entry_index)])
                completed.append((state, finished.copy()))
        if completed:
            assert self._current_output_path is not None
            if self._writer is None:
                self._writer = _ObsOperatorNetCDFWriter(self._current_output_path)
            self._writer.write_array_entries(tuple(completed))

    def _evaluate_entries(
        self,
        state: _ObsOperatorArrayState,
        scheduled_indices: np.ndarray,
        snapshot: OutputSnapshot,
    ) -> np.ndarray:
        prepared = state.prepared
        required_shape = (scheduled_indices.size, prepared.max_field_count)
        if (
            self._sample_workspace.shape[0] < required_shape[0]
            or self._sample_workspace.shape[1] < required_shape[1]
        ):
            self._sample_workspace = np.empty(required_shape, dtype=np.float64)
        samples = self._sample_workspace[: required_shape[0], : required_shape[1]]
        self._sampling_kernel(
            np.asarray(snapshot.state.data[0, ::-1, :, :, :], dtype=np.float64),
            np.asarray(snapshot.forcing.wet_surface_pressure_hpa[0], dtype=np.float64),
            np.asarray(snapshot.forcing.specific_humidity_kg_kg[0], dtype=np.float64),
            np.asarray(snapshot.forcing.temperature_k[0], dtype=np.float64),
            self._grid.hyai_hpa,
            self._grid.hybi,
            scheduled_indices,
            prepared.entry_field_start,
            prepared.entry_field_count,
            prepared.entry_horizontal_start,
            prepared.entry_horizontal_count,
            prepared.entry_vertical_type,
            prepared.entry_vertical_unit,
            prepared.entry_vertical_weighting,
            prepared.entry_vertical_lower,
            prepared.entry_vertical_upper,
            prepared.entry_exact_start,
            prepared.entry_exact_count,
            prepared.field_indices,
            prepared.horizontal_lat,
            prepared.horizontal_lon,
            prepared.horizontal_weight,
            prepared.exact_value,
            prepared.exact_weight,
            samples,
        )
        return samples

    def close(self, *, boundary_time: datetime) -> None:
        if self._closed:
            return
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        boundary_us = _datetime_to_microseconds(boundary_time)
        for state in self._states:
            for entry_index_value in np.flatnonzero(state.active):
                entry_index = int(entry_index_value)
                time_start = int(state.time_start[entry_index] + state.time_consumed[entry_index])
                time_end = int(state.time_start[entry_index] + state.time_count[entry_index])
                remaining_times = state.remaining_time_us[time_start:time_end]
                if remaining_times.size == 0 or np.any(remaining_times < boundary_us):
                    raise ValueError(
                        f"ObsOperator entry {state.ids[entry_index]!r} has an invalid remaining schedule at "
                        f"restart boundary {boundary_time.isoformat()}"
                    )
        restart_path = _resolve_template_path(self._root, self._config.restart_file, boundary_time)
        _write_obsoperator_restart_states(
            restart_path,
            states=self._states,
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

        state = None
        if input_path.is_file():
            state = _load_obsoperator_array_state(
                input_path,
                tracer_names=self._tracer_names,
                grid=self._grid,
                simulation_start=self._start,
                transport_dt_s=self._transport_dt_s,
            )
        if state is not None:
            current_time_us = _datetime_to_microseconds(timestamp)
            self._register_state(state, earliest_time_us=current_time_us)
            logger.info("obsoperator_input_loaded path=%s entries=%d", input_path, state.entry_count)
        else:
            logger.info("obsoperator_input_missing path=%s", input_path)
        output_path = _resolve_template_path(self._root, self._config.output_file, timestamp)
        if output_path != self._current_output_path:
            if self._writer is not None:
                self._writer.close()
                self._writer = None
            self._current_output_path = output_path

    def _register_state(self, state: _ObsOperatorArrayState, *, earliest_time_us: int) -> None:
        duplicates = self._entry_ids.intersection(state.ids)
        if duplicates:
            duplicate = next(entry_id for entry_id in state.ids if entry_id in duplicates)
            raise ValueError(f"duplicate active ObsOperator id {duplicate!r}")
        if state.entry_count and (
            np.any(state.time_count <= 0) or np.any(state.remaining_time_us < earliest_time_us)
        ):
            raise ValueError(
                "ObsOperator entries have no remaining times or sampling times before the current run position; "
                "a matching ObsOperator restart is required"
            )
        self._states.append(state)
        self._entry_ids.update(state.ids)
        for time_index, time_value in enumerate(state.schedule_times_us):
            start = int(state.schedule_start[time_index])
            end = start + int(state.schedule_count[time_index])
            self._schedule.setdefault(int(time_value), []).append((state, start, end))

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
        state = _read_obsoperator_restart(
            restart_path,
            restart_time=self._start,
            transport_dt_s=self._transport_dt_s,
            tracer_names=self._tracer_names,
            grid=self._grid,
        )
        start_us = _datetime_to_microseconds(self._start)
        self._register_state(state, earliest_time_us=start_us)
        logger.info("obsoperator_restart_loaded path=%s entries=%d", restart_path, state.entry_count)


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
    if "input_mode" in raw or "writer" in raw:
        raise ValueError("outputs.obsoperator async input_mode/writer options are no longer supported")
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


def _freeze_operator_spec(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((key, _freeze_operator_spec(item)) for key, item in value.items())
    if isinstance(value, list):
        return tuple(_freeze_operator_spec(item) for item in value)
    return value


def _load_obsoperator_array_state(
    path: str | Path,
    *,
    tracer_names: tuple[str, ...],
    grid: TransportGrid,
    simulation_start: datetime,
    transport_dt_s: float,
) -> _ObsOperatorArrayState:
    entries_raw = _load_obsoperator_raw_entries(Path(path))
    return _array_state_from_raw_entries(
        entries_raw,
        tracer_names=tracer_names,
        grid=grid,
        simulation_start=simulation_start,
        transport_dt_s=transport_dt_s,
    )


def _array_state_from_raw_entries(
    entries_raw: list[Any],
    *,
    tracer_names: tuple[str, ...],
    grid: TransportGrid,
    simulation_start: datetime,
    transport_dt_s: float,
) -> _ObsOperatorArrayState:
    field_cache: dict[Any, tuple[tuple[str, ...], np.ndarray]] = {}
    time_cache: dict[Any, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    horizontal_cache: dict[Any, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    vertical_cache: dict[Any, tuple[int, int, int, float, float, np.ndarray, np.ndarray]] = {}
    rows: list[
        tuple[
            str,
            tuple[tuple[str, ...], np.ndarray],
            tuple[np.ndarray, np.ndarray, np.ndarray],
            tuple[np.ndarray, np.ndarray, np.ndarray, int],
            tuple[int, int, int, float, float, np.ndarray, np.ndarray],
        ]
    ] = []
    seen_ids: set[str] = set()
    for index, raw_entry in enumerate(entries_raw):
        label = f"entries[{index}]"
        mapping = _require_mapping(raw_entry, label)
        entry_id = str(_required(mapping, "id", label))
        if not entry_id:
            raise ValueError(f"{label}.id must not be empty")
        if len(entry_id) > MAX_ID_LENGTH:
            raise ValueError(f"{label}.id exceeds {MAX_ID_LENGTH} characters")
        if entry_id in seen_ids:
            raise ValueError(f"duplicate active ObsOperator id {entry_id!r}")
        seen_ids.add(entry_id)
        if "species" in mapping:
            raise ValueError(f"{label}.species is obsolete; use fields")

        field_raw = _required(mapping, "fields", label)
        field_key = _freeze_operator_spec(field_raw)
        fields = field_cache.get(field_key)
        if fields is None:
            fields = _parse_fields(field_raw, tracer_names, label)
            field_cache[field_key] = fields

        time_raw = _required(mapping, "time_operator", label)
        time_key = _freeze_operator_spec(time_raw)
        time = time_cache.get(time_key)
        if time is None:
            time = _parse_time_arrays(
                time_raw,
                label=f"{label}.time_operator",
                simulation_start=simulation_start,
                transport_dt_s=transport_dt_s,
            )
            time_cache[time_key] = time

        horizontal_raw = _required(mapping, "horizontal_operator", label)
        horizontal_key = _freeze_operator_spec(horizontal_raw)
        horizontal = horizontal_cache.get(horizontal_key)
        if horizontal is None:
            horizontal = _parse_horizontal_arrays(
                horizontal_raw,
                label=f"{label}.horizontal_operator",
                grid=grid,
            )
            horizontal_cache[horizontal_key] = horizontal

        vertical_raw = _required(mapping, "vertical_operator", label)
        vertical_key = _freeze_operator_spec(vertical_raw)
        vertical = vertical_cache.get(vertical_key)
        if vertical is None:
            vertical = _parse_vertical_arrays(
                vertical_raw,
                label=f"{label}.vertical_operator",
                nlev=grid.shape[0],
            )
            vertical_cache[vertical_key] = vertical
        rows.append((entry_id, fields, time, horizontal, vertical))
    return _array_state_from_components(rows)


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


def _sample_prepared_entries_kernel(
    state_bottom: np.ndarray,
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
    nlev = state_bottom.shape[0]
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
                        vertical_sum += exact_weight[exact_index] * state_bottom[level, lat, lon, tracer]
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
                vertical_sum = 0.0
                for level in range(level_start, level_end + 1):
                    if vertical_weighting in (_VERTICAL_NORMALIZED_PRESSURE, _VERTICAL_PRESSURE_WEIGHT):
                        edge_lower = hyai_hpa[level] + hybi[level] * surface_pressure
                        edge_upper = hyai_hpa[level + 1] + hybi[level + 1] * surface_pressure
                        weight = edge_lower - edge_upper
                    else:
                        weight = 1.0
                    vertical_sum += weight * state_bottom[level, lat, lon, tracer]
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


def _parse_time_arrays(
    raw: Any,
    *,
    label: str,
    simulation_start: datetime,
    transport_dt_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    indices = np.arange(start, end + 1, dtype=np.int64)
    if weighting == "normalized":
        weights = np.full(indices.size, 1.0 / indices.size, dtype=np.float64)
    else:
        weights = np.ones(indices.size, dtype=np.float64)
    start_us = _datetime_to_microseconds(simulation_start)
    dt_us = _seconds_to_microseconds(transport_dt_s, "transport timestep")
    times_us = start_us + indices * dt_us
    return indices, weights, times_us


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


def _parse_horizontal_arrays(
    raw: Any,
    *,
    label: str,
    grid: TransportGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
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
    return lat, lon, weights, HORIZONTAL_WEIGHTING_CODES[weighting]


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


def _parse_vertical_arrays(
    raw: Any,
    *,
    label: str,
    nlev: int,
) -> tuple[int, int, int, float, float, np.ndarray, np.ndarray]:
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
        return (
            VERTICAL_TYPE_CODES["exact"],
            VERTICAL_UNIT_CODES[unit],
            -1,
            np.nan,
            np.nan,
            values,
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
    if start > end:
        raise ValueError(f"{label} start must not exceed end")
    return (
        VERTICAL_TYPE_CODES["range"],
        VERTICAL_UNIT_CODES[unit],
        VERTICAL_WEIGHTING_CODES[weighting],
        float(start),
        float(end),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
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
    if value < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return value


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


class _ObsOperatorNetCDFWriter:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._dataset: netCDF4.Dataset | None = None
        self._field_indices: dict[str, int] = {}
        self._field_names: list[str] = []
        self._entry_index = 0
        self._sample_index = 0
        self._pending_array_batches: list[tuple[_ObsOperatorArrayState, np.ndarray]] = []
        self._pending_entry_count = 0
        self._pending_samples = 0

    def write_array_entries(
        self,
        batches: tuple[tuple[_ObsOperatorArrayState, np.ndarray], ...],
    ) -> None:
        for state, entry_indices in batches:
            indices = np.asarray(entry_indices, dtype=np.int64)
            offset = 0
            while offset < indices.size:
                entry_capacity = SCIENCE_STAGE_ENTRIES - self._pending_entry_count
                sample_capacity = SCIENCE_STAGE_SAMPLES - self._pending_samples
                if entry_capacity <= 0 or sample_capacity <= 0:
                    self.flush()
                    entry_capacity = SCIENCE_STAGE_ENTRIES
                    sample_capacity = SCIENCE_STAGE_SAMPLES
                remaining = indices[offset:]
                field_counts = state.prepared.entry_field_count[remaining].astype(np.int64, copy=False)
                cumulative_samples = np.cumsum(field_counts)
                sample_limit = int(np.searchsorted(cumulative_samples, sample_capacity, side="right"))
                if sample_limit == 0:
                    if self._pending_entry_count:
                        self.flush()
                        continue
                    sample_limit = 1
                take = min(remaining.size, entry_capacity, sample_limit)
                selected = remaining[:take].copy()
                selected_samples = int(np.sum(field_counts[:take]))
                self._pending_array_batches.append((state, selected))
                self._pending_entry_count += take
                self._pending_samples += selected_samples
                offset += take
                if (
                    self._pending_entry_count >= SCIENCE_STAGE_ENTRIES
                    or self._pending_samples >= SCIENCE_STAGE_SAMPLES
                    or offset < indices.size
                ):
                    self.flush()

    def flush(self) -> None:
        if not self._pending_array_batches:
            return
        entry_count = self._pending_entry_count
        sample_count = self._pending_samples
        previous_field_count = len(self._field_names)

        field_indices = np.empty(sample_count, dtype=np.int32)
        id_indices = np.empty(sample_count, dtype=np.int32)
        samples = np.empty(sample_count, dtype=np.float32)
        sample_offset = 0
        array_entry_offset = 0
        array_ids: list[str] = []
        for state, entry_indices in self._pending_array_batches:
            for entry_index_value in entry_indices:
                entry_index = int(entry_index_value)
                field_start = int(state.prepared.entry_field_start[entry_index])
                entry_sample_count = int(state.prepared.entry_field_count[entry_index])
                field_end = field_start + entry_sample_count
                entry_slice = slice(sample_offset, sample_offset + entry_sample_count)
                for field_offset, name in enumerate(state.field_names[entry_index]):
                    if name not in self._field_indices:
                        self._field_indices[name] = len(self._field_names) + 1
                        self._field_names.append(name)
                    field_indices[sample_offset + field_offset] = self._field_indices[name]
                id_indices[entry_slice] = self._entry_index + array_entry_offset + 1
                samples[entry_slice] = state.field_accumulator[field_start:field_end]
                array_ids.append(state.ids[entry_index])
                array_entry_offset += 1
                sample_offset += entry_sample_count

        self._ensure_created()
        assert self._dataset is not None
        if len(self._field_names) > previous_field_count:
            new_field_names = self._field_names[previous_field_count:]
            self._dataset.variables["field"][previous_field_count : len(self._field_names), :] = (
                _nul_padded_matrix(new_field_names, MAX_FIELD_NAME_LENGTH, len(new_field_names))
            )
        entry_slice = slice(self._entry_index, self._entry_index + entry_count)
        sample_slice = slice(self._sample_index, self._sample_index + sample_count)
        self._dataset.variables["id"][entry_slice, :] = _nul_padded_matrix(
            array_ids, MAX_ID_LENGTH, entry_count
        )
        self._dataset.variables["id_index"][sample_slice] = id_indices
        self._dataset.variables["field_index"][sample_slice] = field_indices
        self._dataset.variables["sample"][sample_slice] = samples
        self._entry_index += entry_count
        self._sample_index += sample_count
        self._pending_array_batches = []
        self._pending_entry_count = 0
        self._pending_samples = 0

    def close(self) -> None:
        self.flush()
        if self._dataset is None:
            return
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
            chunksizes=(SCIENCE_ENTRY_CHUNK, MAX_ID_LENGTH),
            long_name="ids",
            description="id",
        )
        _create_variable(
            dataset,
            "field",
            "S1",
            ("fields", "field_chars"),
            chunksizes=(SCIENCE_FIELD_CHUNK, MAX_FIELD_NAME_LENGTH),
            long_name="fields",
            description="field name",
        )
        _create_variable(
            dataset,
            "id_index",
            "i4",
            ("samples",),
            chunksizes=(SCIENCE_SAMPLE_CHUNK,),
            long_name="id_index",
            description="index of the id in the id list",
        )
        _create_variable(
            dataset,
            "field_index",
            "i4",
            ("samples",),
            chunksizes=(SCIENCE_SAMPLE_CHUNK,),
            long_name="field_index",
            description="index of the field in the field list",
        )
        _create_variable(
            dataset,
            "sample",
            "f4",
            ("samples",),
            chunksizes=(SCIENCE_SAMPLE_CHUNK,),
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
    chunksizes: tuple[int, ...],
    long_name: str,
    description: str,
) -> netCDF4.Variable:
    variable = dataset.createVariable(
        name,
        dtype,
        dimensions,
        zlib=True,
        complevel=1,
        shuffle=True,
        chunksizes=chunksizes,
    )
    variable.long_name = long_name
    variable.units = "1"
    variable.description = description
    return variable


def _nul_padded_matrix(values: Any, width: int, count: int) -> np.ndarray:
    output = np.full((count, width), b"\x00", dtype="S1")
    for index, value in enumerate(values):
        encoded = value.encode("utf-8")
        if len(encoded) > width:
            raise ValueError(f"encoded string exceeds fixed width {width}")
        output[index, : len(encoded)] = np.frombuffer(encoded, dtype="S1")
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
