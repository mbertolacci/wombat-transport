from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path

import numpy as np

from wombat_transport.grid import TransportGrid
from wombat_transport.obsoperator.config import (
    ObsOperatorConfig,
    _resolve_template_path,
    parse_obsoperator_config,
)
from wombat_transport.obsoperator.input import _load_obsoperator_array_state
from wombat_transport.obsoperator.restart import (
    _read_obsoperator_restart,
    _write_obsoperator_restart_states,
)
from wombat_transport.obsoperator.sampling import (
    accumulate_prepared_samples,
    select_sampling_kernel,
)
from wombat_transport.obsoperator.state import _ObsOperatorArrayState
from wombat_transport.obsoperator.utils import (
    _datetime_to_microseconds,
    _seconds_to_microseconds,
)
from wombat_transport.obsoperator.writer import _ObsOperatorNetCDFWriter
from wombat_transport.output import OutputSnapshot
from wombat_transport.run_config import RunConfig, simulation_start

logger = logging.getLogger(__name__)

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
        self._sampling_kernel = select_sampling_kernel()
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
            accumulate_prepared_samples(
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
