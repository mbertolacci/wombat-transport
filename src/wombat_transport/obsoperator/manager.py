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
from wombat_transport.obsoperator.input import _load_obs_plan
from wombat_transport.obsoperator.restart import _read_obsoperator_restart, _write_obsoperator_restart
from wombat_transport.obsoperator.sampling import select_sampling_kernel
from wombat_transport.obsoperator.state import (
    _completed_batch_range,
    compact_obs_plan,
    completed_prefix,
    empty_obs_plan,
    merge_obs_plans,
)
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
        self._transport_dt_us = _seconds_to_microseconds(transport_dt_s, "transport timestep")
        self._position_us = _datetime_to_microseconds(start)
        self._tracer_names = tracer_names
        self._grid = grid
        self._previous_input_path: Path | None = None
        self._current_output_path: Path | None = None
        self._plan = empty_obs_plan()
        self._sample_scratch = np.empty(0, dtype=np.float64)
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
        step_time_us = _datetime_to_microseconds(step_start)
        if step_time_us != self._position_us:
            raise ValueError(
                "ObsOperator sampling must advance contiguously from the current model position"
            )
        self._initialize_for_date(step_start)
        if self._plan.first_unexpired < self._plan.entry_count:
            state_bottom = np.asarray(
                snapshot.state.block_data[0, :, ::-1, :, :, :], dtype=np.float64
            )
            wet_surface_pressure = np.asarray(
                snapshot.forcing.wet_surface_pressure_hpa[0], dtype=np.float64
            )
            specific_humidity = np.asarray(
                snapshot.forcing.specific_humidity_kg_kg[0], dtype=np.float64
            )
            temperature = np.asarray(snapshot.forcing.temperature_k[0], dtype=np.float64)
            width = snapshot.state.block_width
            plan = self._plan
            required_scratch = state_bottom.shape[0] * width
            if self._sample_scratch.size < required_scratch:
                self._sample_scratch = np.empty(required_scratch, dtype=np.float64)
            self._sampling_kernel(
                state_bottom,
                width,
                0,
                state_bottom.shape[0],
                step_time_us,
                wet_surface_pressure,
                specific_humidity,
                temperature,
                self._grid.area_m2,
                self._grid.hyai_hpa,
                self._grid.hybi,
                plan.first_unexpired,
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
                self._sample_scratch,
                plan.accumulator,
            )
            if self._config.verbose:
                for entry_index in range(plan.first_unexpired, plan.entry_count):
                    time_slice = _ragged_slice(
                        plan.time_operator_start, plan.time_operator_count, entry_index
                    )
                    bounds = plan.time_operator_bounds_us[time_slice]
                    if np.any((bounds[:, 0] <= step_time_us) & (step_time_us < bounds[:, 1])):
                        logger.info(
                            "obsoperator_sample id=%s time_index=%d",
                            plan.ids[entry_index],
                            time_index,
                        )

        boundary_us = step_time_us + self._transport_dt_us
        complete = completed_prefix(self._plan, boundary_us)
        if complete > self._plan.first_unexpired:
            batch = _completed_batch_range(
                self._plan,
                self._plan.first_unexpired,
                complete,
            )
            assert self._current_output_path is not None
            if self._writer is None:
                self._writer = _ObsOperatorNetCDFWriter(self._current_output_path)
            self._writer.write_completed(batch)
            self._plan.first_unexpired = complete
        self._position_us = boundary_us

    def close(self, *, boundary_time: datetime) -> None:
        if self._closed:
            return
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        boundary_us = _datetime_to_microseconds(boundary_time)
        if self._plan.entry_count and boundary_us != self._position_us:
            raise ValueError(
                "ObsOperator plan has an invalid model position at restart boundary "
                f"{boundary_time.isoformat()}"
            )
        self._plan = compact_obs_plan(self._plan, boundary_us)
        restart_path = _resolve_template_path(self._root, self._config.restart_file, boundary_time)
        _write_obsoperator_restart(
            restart_path,
            plan=self._plan,
            restart_time=boundary_time,
            transport_dt_s=self._transport_dt_s,
            grid=self._grid,
        )
        self._closed = True

    def _initialize_for_date(self, timestamp: datetime) -> None:
        input_path = _resolve_template_path(self._root, self._config.input_file, timestamp)
        output_path = _resolve_template_path(self._root, self._config.output_file, timestamp)
        if output_path != self._current_output_path:
            if self._writer is not None:
                self._writer.close()
                self._writer = None
            self._current_output_path = output_path
        if input_path == self._previous_input_path:
            return
        current_time_us = _datetime_to_microseconds(timestamp)
        if self._plan.entry_count and current_time_us != self._position_us:
            raise ValueError(
                "ObsOperator plan cannot skip model timesteps while changing daily input"
            )
        candidate_plan = compact_obs_plan(self._plan, current_time_us)
        if input_path.is_file():
            incoming = _load_obs_plan(
                input_path,
                tracer_names=self._tracer_names,
                grid=self._grid,
                simulation_start=self._start,
                transport_dt_s=self._transport_dt_s,
            )
            if incoming.time_operator_bounds_us.size and np.any(
                incoming.time_operator_bounds_us[:, 0] < current_time_us
            ):
                raise ValueError(
                    "ObsOperator entries have sampling times before the current run position; "
                    "a matching ObsOperator restart is required"
                )
            candidate_plan = merge_obs_plans(candidate_plan, incoming)
            logger.info("obsoperator_input_loaded path=%s entries=%d", input_path, incoming.entry_count)
        else:
            logger.info("obsoperator_input_missing path=%s", input_path)
        self._plan = candidate_plan
        self._previous_input_path = input_path

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
        self._plan = _read_obsoperator_restart(
            restart_path,
            restart_time=self._start,
            transport_dt_s=self._transport_dt_s,
            tracer_names=self._tracer_names,
            grid=self._grid,
        )
        logger.info("obsoperator_restart_loaded path=%s entries=%d", restart_path, self._plan.entry_count)


def _ragged_slice(starts: np.ndarray, counts: np.ndarray, index: int) -> slice:
    start = int(starts[index])
    return slice(start, start + int(counts[index]))
