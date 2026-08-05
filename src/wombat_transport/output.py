from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np

from wombat_transport.fields import TracerField
from wombat_transport.history_accumulation import accumulate_history_sums
from wombat_transport.io import GRID_COORDS
from wombat_transport.run_config import RunConfig, simulation_end, simulation_start
from wombat_transport.snapshot import CompletedStepSnapshot

SUPPORTED_RESTART_MET_FIELDS = {"Met_DELPDRY", "Met_PS1DRY", "Met_PS1WET", "Met_SPHU1", "Met_TMPU1"}
SUPPORTED_FIELD_TOKENS = {"SpeciesConcVV_?ADV?", "SpeciesRst_?ALL?", *SUPPORTED_RESTART_MET_FIELDS}


@dataclass(frozen=True)
class OutputCompressionConfig:
    enabled: bool = True
    algorithm: str = "zlib"
    level: int = 1
    shuffle: bool = True


@dataclass(frozen=True)
class OutputChunkingConfig:
    rank1: tuple[int, ...] | None = None
    rank2: tuple[int, ...] | None = None
    rank3: tuple[int, ...] | None = None
    rank4: tuple[int, ...] | None = None


@dataclass(frozen=True)
class OutputStorageConfig:
    dtype: str = "float32"
    compression: OutputCompressionConfig = field(default_factory=OutputCompressionConfig)
    chunking: OutputChunkingConfig = field(default_factory=OutputChunkingConfig)

    @property
    def netcdf_dtype(self) -> str:
        if self.dtype == "float32":
            return "f4"
        if self.dtype == "float64":
            return "f8"
        raise ValueError(f"unsupported output dtype {self.dtype!r}")


@dataclass(frozen=True)
class HistoryInterval:
    months: int = 0
    days: int = 0
    seconds: int = 0

    def add_to(self, timestamp: datetime) -> datetime:
        result = _add_months(timestamp, self.months) if self.months else timestamp
        return result + timedelta(days=self.days, seconds=self.seconds)

    @property
    def is_zero(self) -> bool:
        return self.months == 0 and self.days == 0 and self.seconds == 0


@dataclass(frozen=True)
class OutputCollectionConfig:
    name: str
    filename: str | None
    template: str | None
    frequency: HistoryInterval
    duration: HistoryInterval
    mode: str
    fields: tuple[str, ...]
    storage: OutputStorageConfig | None = None


OutputSnapshot = CompletedStepSnapshot


@dataclass(frozen=True)
class _DetachedAverageOutput:
    owner: _AverageCollection
    timestamp: datetime
    values: np.ndarray
    denominator: float
    metadata: TracerField
    group_start: datetime
    close_file_after: bool


@dataclass(frozen=True)
class _DetachedRestartOutput:
    owner: _InstantaneousRestartWriter
    path: Path
    snapshot: OutputSnapshot


class HistoryOutputManager:
    def __init__(
        self,
        *,
        root: Path,
        template_path: Path,
        expid: str,
        collections: tuple[OutputCollectionConfig, ...],
        start: datetime,
    ) -> None:
        validate_output_collections(collections)
        self._averages = [
            _AverageCollection(
                root=root,
                template_path=template_path,
                expid=expid,
                collection=collection,
                start=start,
                accumulator_index=index,
                materialize_average=self._materialize_average,
            )
            for index, collection in enumerate(
                collection
                for collection in collections
                if collection.mode == "time-averaged"
            )
        ]
        self._restarts = [
            _InstantaneousRestartWriter(
                root=root,
                template_path=template_path,
                expid=expid,
                collection=collection,
                start=start,
                materialize_snapshot=self._materialize_snapshot,
            )
            for collection in collections
            if collection.mode == "instantaneous"
        ]
        self._sums: Any | None = None
        self._prepared_timestamp: datetime | None = None
        self._last_state: TracerField | None = None
        self._detached_cuda_outputs: list[
            _DetachedAverageOutput | _DetachedRestartOutput
        ] = []
        self._zeros = lambda shape: np.zeros(shape, dtype=np.float64)
        self._average_materializer = lambda values, count, dtype: (
            values,
            float(count),
        )
        self._state_materializer = lambda state: state
        self._snapshot_materializer = lambda snapshot: snapshot

    @classmethod
    def from_run_config(
        cls,
        config: RunConfig,
        *,
        transport_dt_s: float,
    ) -> HistoryOutputManager | None:
        if not config.outputs:
            return None
        _validate_output_writer(config.outputs)
        collections = parse_output_collections(config.outputs)
        if not collections:
            return None
        start = simulation_start(config)
        validate_restart_output_alignment(
            collections,
            start=start,
            end=simulation_end(config),
            transport_dt_s=transport_dt_s,
        )
        return cls(
            root=config.root,
            template_path=config.grid_template,
            expid=str(config.outputs.get("expid", "OutputDir/GEOSChem")),
            collections=collections,
            start=start,
        )

    def record_step(self, snapshot: OutputSnapshot) -> None:
        sums = self.prepare_step(snapshot.timestamp, snapshot.state)
        if sums is not None:
            accumulate_history_sums(sums, snapshot.state.block_data[0])
        self.complete_step(snapshot)

    def use_cuda(self, runtime: Any) -> None:
        """Keep HISTORY sums resident and materialize only writer boundaries."""

        if self._sums is not None:
            raise ValueError("cannot change HISTORY storage after accumulation starts")
        self._zeros = lambda shape: runtime.zeros(shape, dtype=np.float64)
        from wombat_transport.cuda.history import (
            CudaHistoryAverageMaterializer,
        )

        materializer = CudaHistoryAverageMaterializer(runtime)

        def materialize_average(
            values: Any,
            count: int,
            dtype: str,
        ) -> tuple[np.ndarray, float]:
            return (
                materializer.materialize(values, count, dtype=dtype),
                1.0,
            )

        self._average_materializer = materialize_average

        def materialize_state(state: TracerField) -> TracerField:
            return TracerField(
                names=state.names,
                data=runtime.to_host(state.block_data),
                units=state.units,
                coords=state.coords,
            )

        self._state_materializer = materialize_state

        def materialize_array(values: Any) -> Any:
            return runtime.to_host(values) if runtime.is_device_array(values) else values

        def materialize_snapshot(snapshot: OutputSnapshot) -> OutputSnapshot:
            forcing = snapshot.forcing
            return replace(
                snapshot,
                state=materialize_state(snapshot.state),
                delp_dry_hpa=materialize_array(snapshot.delp_dry_hpa),
                forcing=replace(
                    forcing,
                    wet_surface_pressure_hpa=materialize_array(
                        forcing.wet_surface_pressure_hpa
                    ),
                    dry_surface_pressure_hpa=materialize_array(
                        forcing.dry_surface_pressure_hpa
                    ),
                    specific_humidity_kg_kg=materialize_array(
                        forcing.specific_humidity_kg_kg
                    ),
                    temperature_k=materialize_array(forcing.temperature_k),
                ),
            )

        self._snapshot_materializer = materialize_snapshot

    def prepare_step(
        self,
        timestamp: datetime,
        state: TracerField,
    ) -> Any | None:
        if self._prepared_timestamp is not None:
            raise ValueError("an output transport step is already prepared")
        self._ensure_accumulators(state)
        assert self._sums is not None or not self._averages
        if self._sums is not None:
            for average in self._averages:
                average.prepare(
                    timestamp,
                    state,
                    self._sums[average.accumulator_index],
                )
        self._prepared_timestamp = timestamp
        return self._sums

    def complete_step(self, snapshot: OutputSnapshot) -> None:
        if self._prepared_timestamp is None:
            raise ValueError("no output transport step is prepared")
        if snapshot.timestamp != self._prepared_timestamp:
            raise ValueError("output snapshot timestamp does not match the prepared step")
        if self._sums is not None:
            for average in self._averages:
                average.complete(
                    snapshot.timestamp,
                    snapshot.state,
                    self._sums[average.accumulator_index],
                )
        for restart in self._restarts:
            restart.record_step(snapshot)
        self._last_state = snapshot.state
        self._prepared_timestamp = None

    def detach_cuda_step(self, snapshot: OutputSnapshot) -> None:
        """Complete a CUDA step while deferring only its host file writes."""

        if self._prepared_timestamp is None:
            raise ValueError("no output transport step is prepared")
        if snapshot.timestamp != self._prepared_timestamp:
            raise ValueError(
                "output snapshot timestamp does not match the prepared step"
            )
        if self._sums is not None:
            for average in self._averages:
                detached = average.detach_complete(
                    snapshot.timestamp,
                    snapshot.state,
                    self._sums[average.accumulator_index],
                )
                if detached is not None:
                    self._detached_cuda_outputs.append(detached)
        for restart in self._restarts:
            self._detached_cuda_outputs.extend(
                restart.detach_step(snapshot)
            )
        self._last_state = snapshot.state
        self._prepared_timestamp = None

    def write_detached_cuda_outputs(self) -> None:
        """Write host payloads detached at an earlier CUDA boundary."""

        for detached in self._detached_cuda_outputs:
            detached.owner.write_detached(detached)
        self._detached_cuda_outputs = []

    def requires_host_completion(self, timestamp: datetime) -> bool:
        """Return whether completing this CUDA step materializes host output."""

        return any(
            average.requires_host_completion(timestamp)
            for average in self._averages
        ) or any(
            restart.requires_host_completion(timestamp)
            for restart in self._restarts
        )

    def close(self) -> None:
        self._prepared_timestamp = None
        self.write_detached_cuda_outputs()
        if self._sums is not None:
            for average in self._averages:
                average.close(
                    self._sums[average.accumulator_index],
                    self._last_state,
                )
        for restart in self._restarts:
            restart.close()

    def _ensure_accumulators(self, state: TracerField) -> None:
        if not self._averages:
            return
        expected = (len(self._averages), *state.block_data.shape[1:])
        if self._sums is None:
            self._sums = self._zeros(expected)
            return
        if self._sums.shape != expected:
            raise ValueError(
                f"HISTORY tracer storage changed from {self._sums.shape[1:]} "
                f"to {expected[1:]}"
            )

    def _materialize_average(
        self,
        values: Any,
        count: int,
        dtype: str,
    ) -> tuple[np.ndarray, float]:
        return self._average_materializer(values, count, dtype)

    def _materialize_state(self, state: TracerField) -> TracerField:
        return self._state_materializer(state)

    def _materialize_snapshot(
        self,
        snapshot: OutputSnapshot,
    ) -> OutputSnapshot:
        return self._snapshot_materializer(snapshot)


class _AverageCollection:
    def __init__(
        self,
        *,
        root: Path,
        template_path: Path,
        expid: str,
        collection: OutputCollectionConfig,
        start: datetime,
        accumulator_index: int,
        materialize_average: Any,
    ) -> None:
        self._root = root
        self._template_path = template_path
        self._expid = expid
        self._collection = collection
        self._start = start
        self.accumulator_index = accumulator_index
        self._materialize_average = materialize_average
        self._window_start: datetime | None = None
        self._window_end: datetime | None = None
        self._group_start: datetime | None = None
        self._group_end: datetime | None = None
        self._count = 0
        self._file: _StreamingSpeciesConcFile | None = None

    def prepare(
        self,
        timestamp: datetime,
        state: TracerField,
        summed: np.ndarray,
    ) -> None:
        if self._window_start is None:
            self._initialize_schedule(timestamp)
        assert self._window_end is not None
        while timestamp > self._window_end:
            self._finish_and_advance(summed, state)

    def complete(
        self,
        timestamp: datetime,
        state: TracerField,
        summed: np.ndarray,
    ) -> None:
        if self._window_end is None:
            raise ValueError("cannot complete an unprepared SpeciesConc step")
        self._count += 1
        if timestamp == self._window_end:
            self._finish_and_advance(summed, state)
        elif timestamp > self._window_end:
            raise ValueError("SpeciesConc sample advanced beyond its prepared window")

    def detach_complete(
        self,
        timestamp: datetime,
        state: TracerField,
        summed: np.ndarray,
    ) -> _DetachedAverageOutput | None:
        """Finalize a CUDA average into host memory without writing it."""

        if self._window_end is None:
            raise ValueError("cannot complete an unprepared SpeciesConc step")
        self._count += 1
        if timestamp < self._window_end:
            return None
        if timestamp > self._window_end:
            raise ValueError(
                "SpeciesConc sample advanced beyond its prepared window"
            )
        if self._window_start is None or self._group_start is None:
            raise ValueError("SpeciesConc schedule is not initialized")
        storage = _collection_storage(self._collection)
        materialized, denominator = self._materialize_average(
            summed,
            self._count,
            storage.netcdf_dtype,
        )
        output = _DetachedAverageOutput(
            owner=self,
            timestamp=self._window_start,
            values=materialized,
            denominator=denominator,
            metadata=state,
            group_start=self._group_start,
            close_file_after=False,
        )
        summed.fill(0.0)
        self._count = 0
        close_file_after = self._advance_schedule(close_group_file=False)
        if close_file_after:
            output = replace(output, close_file_after=True)
        return output

    def write_detached(self, output: _DetachedAverageOutput) -> None:
        self._ensure_file(
            group_start=output.group_start,
            first_timestamp=output.timestamp,
            first_state=output.metadata,
        )
        assert self._file is not None
        self._file.append_average(
            output.timestamp,
            output.values,
            output.denominator,
            output.metadata,
        )
        if output.close_file_after:
            self._close_file()

    def requires_host_completion(self, timestamp: datetime) -> bool:
        if self._window_end is None:
            self._initialize_schedule(timestamp)
        assert self._window_end is not None
        return timestamp >= self._window_end

    def close(
        self,
        summed: np.ndarray,
        metadata: TracerField | None,
    ) -> None:
        if self._count:
            if metadata is None:
                raise ValueError(
                    "cannot finish SpeciesConc output window without tracer metadata"
                )
            self._append_average(summed, metadata)
            summed.fill(0.0)
            self._count = 0
        self._close_file()

    def _initialize_schedule(self, timestamp: datetime) -> None:
        window_start = self._start
        window_end = self._collection.frequency.add_to(window_start)
        while timestamp > window_end:
            window_start = window_end
            window_end = self._collection.frequency.add_to(window_start)
        group_start = self._start
        group_end = self._collection.duration.add_to(group_start)
        while window_start >= group_end:
            group_start = group_end
            group_end = self._collection.duration.add_to(group_start)
        self._window_start = window_start
        self._window_end = window_end
        self._group_start = group_start
        self._group_end = group_end

    def _finish_and_advance(
        self,
        summed: np.ndarray,
        metadata: TracerField,
    ) -> None:
        if self._window_end is None or self._group_end is None:
            raise ValueError("SpeciesConc schedule is not initialized")
        if self._count:
            self._append_average(summed, metadata)
        summed.fill(0.0)
        self._count = 0

        self._advance_schedule(close_group_file=True)

    def _advance_schedule(self, *, close_group_file: bool) -> bool:
        """Advance one average window and report a file-group transition."""

        if self._window_end is None or self._group_end is None:
            raise ValueError("SpeciesConc schedule is not initialized")
        next_window_start = self._window_end
        next_window_end = self._collection.frequency.add_to(next_window_start)
        next_group_start = self._group_start
        next_group_end = self._group_end
        if next_group_start is None:
            raise ValueError("SpeciesConc group schedule is not initialized")
        while next_window_start >= next_group_end:
            next_group_start = next_group_end
            next_group_end = self._collection.duration.add_to(next_group_start)
        group_changed = next_group_start != self._group_start
        if group_changed and close_group_file:
            self._close_file()
        self._window_start = next_window_start
        self._window_end = next_window_end
        self._group_start = next_group_start
        self._group_end = next_group_end
        return group_changed

    def _append_average(
        self,
        summed: np.ndarray,
        metadata: TracerField,
    ) -> None:
        if self._window_start is None or self._group_start is None:
            raise ValueError("SpeciesConc schedule is not initialized")
        if self._count <= 0:
            return
        storage = _collection_storage(self._collection)
        self._ensure_file(
            group_start=self._group_start,
            first_timestamp=self._window_start,
            first_state=metadata,
        )
        materialized, denominator = self._materialize_average(
            summed,
            self._count,
            storage.netcdf_dtype,
        )
        self._file.append_average(
            self._window_start,
            materialized,
            denominator,
            metadata,
        )

    def _ensure_file(
        self,
        *,
        group_start: datetime,
        first_timestamp: datetime,
        first_state: TracerField,
    ) -> None:
        if self._file is not None:
            return
        self._file = _StreamingSpeciesConcFile(
            path=_collection_path(
                self._root,
                self._expid,
                self._collection,
                group_start,
            ),
            template_path=self._template_path,
            title=(
                "GEOS-Chem diagnostic collection: "
                f"{self._collection.name}"
            ),
            storage=_collection_storage(self._collection),
            first_timestamp=first_timestamp,
            first_state=first_state,
        )

    def _close_file(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def _validate_output_writer(raw: dict[str, Any]) -> None:
    mode = str(raw.get("writer", "sync")).lower()
    if mode != "sync":
        raise ValueError(
            f"outputs.writer={mode!r} is not supported; "
            "HISTORY output requires 'sync'"
        )


def parse_output_writer(raw: dict[str, Any]) -> str:
    """Validate the retained synchronous writer setting."""

    _validate_output_writer(raw)
    return "sync"


class _StreamingSpeciesConcFile:
    def __init__(
        self,
        *,
        path: Path,
        template_path: Path,
        title: str,
        storage: OutputStorageConfig,
        first_timestamp: datetime,
        first_state: TracerField,
    ) -> None:
        self._path = path
        self._storage = storage
        self._base_time = first_timestamp
        self._sample_index = 0
        self._names = first_state.names
        self._shape = first_state.shape
        self._storage_shape = first_state.block_data.shape[1:]
        self._dataset: netCDF4.Dataset | None = None
        self._time_variable = None
        self._variables: list[netCDF4.Variable] = []
        self._write_buffer = np.empty(self._shape[1:4], dtype=np.float64)
        self._open(template_path, title, first_state)

    def append(self, timestamp: datetime, sample: TracerField) -> None:
        self._validate_open_sample(sample)
        self._write_time_sample(timestamp)
        self._write_state(sample)
        self._sample_index += 1

    def append_average(
        self,
        timestamp: datetime,
        summed: np.ndarray,
        denominator: float,
        metadata: TracerField,
    ) -> None:
        self._validate_open_sample(metadata)
        if summed.shape != self._storage_shape:
            raise ValueError("all SpeciesConc samples must have the same shape")
        if denominator <= 0:
            raise ValueError("SpeciesConc average denominator must be positive")

        self._write_time_sample(timestamp)
        if denominator == 1.0:
            self._write_preaveraged(summed, metadata)
        else:
            self._write_average(summed, metadata, denominator)
        self._sample_index += 1

    def _write_state(self, sample: TracerField) -> None:
        for tracer_index, variable in enumerate(self._variables):
            variable[self._sample_index, :, :, :] = sample.tracer(tracer_index)[
                0, ::-1, :, :
            ]

    def _write_average(
        self,
        summed: np.ndarray,
        metadata: TracerField,
        denominator: float,
    ) -> None:
        for tracer_index, variable in enumerate(self._variables):
            np.divide(
                _summed_tracer(summed, metadata, tracer_index)[::-1, :, :],
                denominator,
                out=self._write_buffer,
            )
            variable[self._sample_index, :, :, :] = self._write_buffer

    def _write_preaveraged(
        self,
        values: np.ndarray,
        metadata: TracerField,
    ) -> None:
        for tracer_index, variable in enumerate(self._variables):
            variable[self._sample_index, :, :, :] = _summed_tracer(
                values,
                metadata,
                tracer_index,
            )[::-1, :, :]

    def close(self) -> None:
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None
            self._time_variable = None
            self._variables = []

    def _validate_open_sample(self, sample: TracerField) -> None:
        if self._dataset is None or self._time_variable is None:
            raise ValueError("cannot write to closed SpeciesConc output file")
        if sample.names != self._names:
            raise ValueError("all SpeciesConc samples must have the same tracer names")
        if (
            sample.shape != self._shape
            or sample.block_data.shape[1:] != self._storage_shape
        ):
            raise ValueError("all SpeciesConc samples must have the same shape")

    def _write_time_sample(self, timestamp: datetime) -> None:
        if self._time_variable is None:
            raise ValueError("cannot write time to closed SpeciesConc output file")
        self._time_variable[self._sample_index] = (timestamp - self._base_time).total_seconds() / 60.0

    def _open(
        self, template_path: Path, title: str, first_state: TracerField
    ) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with netCDF4.Dataset(template_path) as template:
                self._dataset = netCDF4.Dataset(self._path, "w")
                _create_common_dimensions(self._dataset, template, include_bounds=True)
                _assert_compatible_samples(
                    [first_state],
                    expected_shape=(
                        1,
                        len(template.dimensions["lev"]),
                        len(template.dimensions["lat"]),
                        len(template.dimensions["lon"]),
                    ),
                )
                _copy_common_coordinates(self._dataset, template, include_bounds=True, storage=self._storage)
                self._time_variable = _create_time_variable(
                    self._dataset,
                    base=self._base_time,
                    storage=self._storage,
                )
                self._dataset.title = title
                self._dataset.format = "NetCDF-4"
                self._variables = []
                for tracer_index, tracer_name in enumerate(first_state.names):
                    variable = _create_output_variable(
                        self._dataset,
                        f"SpeciesConcVV_{tracer_name}",
                        ("time", "lev", "lat", "lon"),
                        self._storage,
                    )
                    variable.units = (
                        first_state.units[tracer_index]
                        if tracer_index < len(first_state.units)
                        else "mol mol-1 dry"
                    )
                    variable.long_name = f"Dry mixing ratio of species {tracer_name}"
                    self._variables.append(variable)
        except Exception:
            self.close()
            raise


class _InstantaneousRestartWriter:
    def __init__(
        self,
        *,
        root: Path,
        template_path: Path,
        expid: str,
        collection: OutputCollectionConfig,
        start: datetime,
        materialize_snapshot: Any,
    ) -> None:
        self._root = root
        self._template_path = template_path
        self._expid = expid
        self._collection = collection
        self._next_output = collection.frequency.add_to(start)
        self._materialize_snapshot = materialize_snapshot

    def requires_host_completion(self, timestamp: datetime) -> bool:
        return timestamp >= self._next_output

    def record_step(self, snapshot: OutputSnapshot) -> None:
        while snapshot.timestamp >= self._next_output:
            path = _collection_path(self._root, self._expid, self._collection, self._next_output)
            host_snapshot = self._materialize_snapshot(snapshot)
            write_restart_collection(
                path,
                host_snapshot,
                self._template_path,
                fields=self._collection.fields,
                title=f"GEOS-Chem diagnostic collection: {self._collection.name}",
                storage=_collection_storage(self._collection),
            )
            self._next_output = self._collection.frequency.add_to(self._next_output)

    def detach_step(
        self,
        snapshot: OutputSnapshot,
    ) -> list[_DetachedRestartOutput]:
        outputs = []
        while snapshot.timestamp >= self._next_output:
            outputs.append(
                _DetachedRestartOutput(
                    owner=self,
                    path=_collection_path(
                        self._root,
                        self._expid,
                        self._collection,
                        self._next_output,
                    ),
                    snapshot=self._materialize_snapshot(snapshot),
                )
            )
            self._next_output = self._collection.frequency.add_to(
                self._next_output
            )
        return outputs

    def write_detached(self, output: _DetachedRestartOutput) -> None:
        write_restart_collection(
            output.path,
            output.snapshot,
            self._template_path,
            fields=self._collection.fields,
            title=(
                "GEOS-Chem diagnostic collection: "
                f"{self._collection.name}"
            ),
            storage=_collection_storage(self._collection),
        )

    def close(self) -> None:
        return None


def parse_output_collections(raw: dict[str, Any]) -> tuple[OutputCollectionConfig, ...]:
    storage = parse_output_storage(raw)
    collections_raw = raw.get("collections", {})
    if not isinstance(collections_raw, dict):
        raise TypeError("outputs.collections must be a mapping")
    collections: list[OutputCollectionConfig] = []
    for name, value in collections_raw.items():
        if not isinstance(value, dict):
            raise TypeError(f"outputs.collections.{name} must be a mapping")
        fields = _parse_fields(value.get("fields", ()))
        unsupported = [field for field in fields if field not in SUPPORTED_FIELD_TOKENS]
        if unsupported:
            raise ValueError(f"output collection {name} contains unsupported fields: {', '.join(unsupported)}")
        collections.append(
            OutputCollectionConfig(
                name=str(name),
                filename=str(value["filename"]) if "filename" in value else None,
                template=str(value["template"]) if "template" in value else None,
                frequency=parse_history_interval(str(value["frequency"])),
                duration=parse_history_interval(str(value.get("duration", value["frequency"]))),
                mode=str(value["mode"]),
                fields=fields,
                storage=parse_output_storage(
                    value,
                    defaults=storage,
                    label=f"outputs.collections.{name}",
                ),
            )
        )
    return tuple(collections)


def validate_output_collections(collections: tuple[OutputCollectionConfig, ...]) -> None:
    """Validate collection semantics without constructing any output resources."""

    for collection in collections:
        if collection.filename is None and collection.template is None:
            raise ValueError(f"output collection {collection.name} requires filename or template")
        if collection.frequency.is_zero:
            raise ValueError(f"output collection {collection.name} frequency must be nonzero")
        if collection.duration.is_zero:
            raise ValueError(f"output collection {collection.name} duration must be nonzero")
        time_average = (
            collection.mode == "time-averaged"
            and collection.fields == ("SpeciesConcVV_?ADV?",)
        )
        restart = (
            collection.mode == "instantaneous"
            and "SpeciesRst_?ALL?" in collection.fields
        )
        if not time_average and not restart:
            raise ValueError(
                f"unsupported output collection {collection.name}: "
                f"mode={collection.mode}, fields={collection.fields}"
            )


def parse_output_storage(
    raw: dict[str, Any],
    *,
    defaults: OutputStorageConfig | None = None,
    label: str = "outputs",
) -> OutputStorageConfig:
    defaults = defaults or OutputStorageConfig()
    dtype = str(raw.get("dtype", defaults.dtype)).lower()
    if dtype not in {"float32", "float64"}:
        raise ValueError(f"{label}.dtype must be 'float32' or 'float64'")

    compression_raw = raw.get("compression")
    if compression_raw is None:
        compression_raw = {}
    if not isinstance(compression_raw, dict):
        raise TypeError(f"{label}.compression must be a mapping")
    algorithm = str(
        compression_raw.get("algorithm", defaults.compression.algorithm)
    ).lower()
    supported_algorithms = {"zlib", "zstd", "blosc_lz4", "blosc_zstd"}
    if algorithm not in supported_algorithms:
        raise ValueError(
            f"{label}.compression.algorithm must be one of "
            f"{', '.join(sorted(supported_algorithms))}"
        )
    level = int(compression_raw.get("level", defaults.compression.level))
    if level < 0 or level > 9:
        raise ValueError(f"{label}.compression.level must be between 0 and 9")
    compression = OutputCompressionConfig(
        enabled=bool(
            compression_raw.get("enabled", defaults.compression.enabled)
        ),
        algorithm=algorithm,
        level=level,
        shuffle=bool(
            compression_raw.get("shuffle", defaults.compression.shuffle)
        ),
    )

    chunking_raw = raw.get("chunking")
    if chunking_raw is None:
        chunking_raw = {}
    if not isinstance(chunking_raw, dict):
        raise TypeError(f"{label}.chunking must be a mapping")
    chunking = OutputChunkingConfig(
        rank1=_parse_chunk_array(
            chunking_raw.get("rank1", defaults.chunking.rank1),
            1,
            f"{label}.chunking.rank1",
        ),
        rank2=_parse_chunk_array(
            chunking_raw.get("rank2", defaults.chunking.rank2),
            2,
            f"{label}.chunking.rank2",
        ),
        rank3=_parse_chunk_array(
            chunking_raw.get("rank3", defaults.chunking.rank3),
            3,
            f"{label}.chunking.rank3",
        ),
        rank4=_parse_chunk_array(
            chunking_raw.get("rank4", defaults.chunking.rank4),
            4,
            f"{label}.chunking.rank4",
        ),
    )
    return OutputStorageConfig(dtype=dtype, compression=compression, chunking=chunking)


def parse_history_interval(value: str) -> HistoryInterval:
    pieces = value.split()
    if len(pieces) != 2 or len(pieces[0]) != 8 or len(pieces[1]) != 6:
        raise ValueError(f"invalid HISTORY interval {value!r}")
    date, clock = pieces
    if not (date.isdigit() and clock.isdigit()):
        raise ValueError(f"invalid HISTORY interval {value!r}")
    years = int(date[0:4])
    months = int(date[4:6])
    days = int(date[6:8])
    hours = int(clock[0:2])
    minutes = int(clock[2:4])
    seconds = int(clock[4:6])
    if years:
        months += years * 12
    return HistoryInterval(months=months, days=days, seconds=hours * 3600 + minutes * 60 + seconds)


def validate_restart_output_alignment(
    collections: tuple[OutputCollectionConfig, ...],
    *,
    start: datetime,
    end: datetime,
    transport_dt_s: float,
) -> None:
    """Require instantaneous restart boundaries to coincide with transport steps."""

    for collection in collections:
        if collection.mode != "instantaneous" or "SpeciesRst_?ALL?" not in collection.fields:
            continue
        boundary = collection.frequency.add_to(start)
        while boundary <= end:
            step_index = (boundary - start).total_seconds() / float(transport_dt_s)
            if not np.isclose(step_index, round(step_index), rtol=0.0, atol=1.0e-9):
                raise ValueError(
                    f"instantaneous restart collection {collection.name!r} has an output "
                    "boundary that is not aligned with transport_timestep_s"
                )
            next_boundary = collection.frequency.add_to(boundary)
            if next_boundary <= boundary:
                raise ValueError(
                    f"instantaneous restart collection {collection.name!r} frequency must be positive"
                )
            boundary = next_boundary


def expand_history_template(template: str, timestamp: datetime) -> str:
    return (
        template.replace("%y4", f"{timestamp.year:04d}")
        .replace("%m2", f"{timestamp.month:02d}")
        .replace("%d2", f"{timestamp.day:02d}")
        .replace("%h2", f"{timestamp.hour:02d}")
        .replace("%n2", f"{timestamp.minute:02d}")
    )


def write_species_conc_collection(
    path: str | Path,
    samples: list[tuple[datetime, TracerField]],
    template_path: str | Path,
    *,
    title: str,
    storage: OutputStorageConfig | None = None,
) -> Path:
    storage = storage or OutputStorageConfig()
    if not samples:
        raise ValueError("cannot write a SpeciesConc collection with no samples")
    output_path = Path(path)
    writer = _StreamingSpeciesConcFile(
        path=output_path,
        template_path=Path(template_path),
        title=title,
        storage=storage,
        first_timestamp=samples[0][0],
        first_state=samples[0][1],
    )
    try:
        for timestamp, sample in samples:
            writer.append(timestamp, sample)
    finally:
        writer.close()
    return output_path


def write_restart_collection(
    path: str | Path,
    snapshot: OutputSnapshot,
    template_path: str | Path,
    *,
    fields: tuple[str, ...],
    title: str,
    storage: OutputStorageConfig | None = None,
) -> Path:
    storage = storage or OutputStorageConfig()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(template_path) as template, netCDF4.Dataset(output_path, "w") as output:
        _create_common_dimensions(output, template, include_bounds=False)
        _copy_common_coordinates(output, template, include_bounds=False, storage=storage)
        _write_time(output, [snapshot.timestamp], base=snapshot.timestamp, utc=True, storage=storage)
        output.title = title
        output.format = "CFIO"
        if "SpeciesRst_?ALL?" in fields:
            for tracer_index, tracer_name in enumerate(snapshot.state.names):
                variable = _create_output_variable(
                    output,
                    f"SpeciesRst_{tracer_name}",
                    ("time", "lev", "lat", "lon"),
                    storage,
                )
                variable.units = (
                    snapshot.state.units[tracer_index]
                    if tracer_index < len(snapshot.state.units)
                    else "mol mol-1 dry"
                )
                variable.long_name = (
                    f"Wombat restart concentration of species {tracer_name}"
                )
                variable[:] = snapshot.state.tracer(tracer_index)[:, ::-1, :, :]
        for field in fields:
            if field in SUPPORTED_RESTART_MET_FIELDS:
                _write_restart_met_field(output, field, snapshot, storage)
    return output_path


def _parse_fields(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, (list, tuple)):
        return tuple(str(item) for item in raw)
    raise TypeError("output collection fields must be a string or list of strings")


def _parse_chunk_array(raw: Any, rank: int, label: str) -> tuple[int, ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise TypeError(f"{label} must be a list of {rank} positive integers")
    values = tuple(int(item) for item in raw)
    if len(values) != rank:
        raise ValueError(f"{label} must contain exactly {rank} values")
    if any(value <= 0 for value in values):
        raise ValueError(f"{label} values must be positive")
    return values


def _collection_storage(collection: OutputCollectionConfig) -> OutputStorageConfig:
    return collection.storage or OutputStorageConfig()


def _collection_path(root: Path, expid: str, collection: OutputCollectionConfig, timestamp: datetime) -> Path:
    if collection.filename is not None:
        value = expand_history_template(collection.filename, timestamp)
    elif collection.template is not None:
        value = f"{expid}.{collection.name}.{expand_history_template(collection.template, timestamp)}"
    else:
        raise ValueError(f"output collection {collection.name} requires filename or template")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _create_common_dimensions(
    output: netCDF4.Dataset,
    template: netCDF4.Dataset,
    *,
    include_bounds: bool,
) -> None:
    output.createDimension("time", None)
    for dim_name in ("lev", "ilev", "lat", "lon"):
        output.createDimension(dim_name, len(template.dimensions[dim_name]))
    if include_bounds:
        output.createDimension("nb", 2)


def _create_output_variable(
    output: netCDF4.Dataset,
    name: str,
    dimensions: tuple[str, ...],
    storage: OutputStorageConfig,
):
    kwargs: dict[str, Any] = {}
    if dimensions:
        chunks = _chunks_for_variable(output, dimensions, storage)
        if chunks is not None:
            kwargs["chunksizes"] = chunks
        kwargs.update(netcdf_compression_kwargs(storage.compression))
    return output.createVariable(name, storage.netcdf_dtype, dimensions, **kwargs)


def _create_template_variable(
    output: netCDF4.Dataset,
    source: netCDF4.Variable,
    storage: OutputStorageConfig,
):
    storage = _metadata_storage(storage)
    kwargs: dict[str, Any] = {}
    dimensions = source.dimensions
    if dimensions:
        chunks = _chunks_for_variable(output, dimensions, storage)
        if chunks is not None:
            kwargs["chunksizes"] = chunks
        kwargs.update(netcdf_compression_kwargs(storage.compression))
    return output.createVariable(source.name, source.datatype, dimensions, **kwargs)


def netcdf_compression_kwargs(
    compression: OutputCompressionConfig,
) -> dict[str, Any]:
    if not compression.enabled:
        return {}
    if (
        compression.algorithm == "zstd"
        and not getattr(netCDF4, "__has_zstandard_support__", False)
    ):
        raise RuntimeError(
            "zstd output requires the netCDF4 HDF5 zstandard filter plugin"
        )
    if (
        compression.algorithm.startswith("blosc_")
        and not getattr(netCDF4, "__has_blosc_support__", False)
    ):
        raise RuntimeError(
            "Blosc output requires the netCDF4 HDF5 Blosc filter plugin"
        )
    kwargs: dict[str, Any] = {
        "compression": compression.algorithm,
        "complevel": compression.level,
    }
    if compression.algorithm.startswith("blosc_"):
        kwargs["blosc_shuffle"] = 1 if compression.shuffle else 0
    else:
        kwargs["shuffle"] = compression.shuffle
    return kwargs


def _chunks_for_variable(
    output: netCDF4.Dataset,
    dimensions: tuple[str, ...],
    storage: OutputStorageConfig,
) -> tuple[int, ...] | None:
    shape = tuple(len(output.dimensions[dimension]) for dimension in dimensions)
    configured = {
        1: storage.chunking.rank1,
        2: storage.chunking.rank2,
        3: storage.chunking.rank3,
        4: storage.chunking.rank4,
    }.get(len(dimensions))
    if configured is not None:
        return _fit_chunks_to_dimensions(output, dimensions, configured)
    if len(dimensions) == 1:
        if dimensions == ("time",):
            return (512,)
        return shape
    if len(dimensions) == 2:
        return shape
    if len(dimensions) == 3:
        if dimensions == ("time", "lat", "lon"):
            return (1, shape[1], shape[2])
        return shape
    if len(dimensions) == 4:
        if dimensions == ("time", "lev", "lat", "lon"):
            return (1, 1, shape[2], shape[3])
        return shape
    return None


def _fit_chunks_to_dimensions(
    output: netCDF4.Dataset,
    dimensions: tuple[str, ...],
    chunks: tuple[int, ...],
) -> tuple[int, ...]:
    fitted: list[int] = []
    for dimension, chunk in zip(dimensions, chunks, strict=True):
        output_dimension = output.dimensions[dimension]
        if output_dimension.isunlimited():
            fitted.append(chunk)
        else:
            fitted.append(min(chunk, len(output_dimension)))
    return tuple(fitted)


def _copy_common_coordinates(
    output: netCDF4.Dataset,
    template: netCDF4.Dataset,
    *,
    include_bounds: bool,
    storage: OutputStorageConfig,
) -> None:
    for coord_name in GRID_COORDS:
        if coord_name == "time" or coord_name not in template.variables:
            continue
        source = template.variables[coord_name]
        variable = _create_template_variable(output, source, storage)
        variable.setncatts({name: source.getncattr(name) for name in source.ncattrs()})
        if source.dimensions:
            variable[:] = np.asarray(source[:])
        else:
            variable.assignValue(source.getValue())
    if include_bounds:
        for coord_name in ("lat_bnds", "lon_bnds"):
            if coord_name in template.variables:
                source = template.variables[coord_name]
                variable = _create_template_variable(output, source, storage)
                variable.setncatts({name: source.getncattr(name) for name in source.ncattrs()})
                variable[:] = np.asarray(source[:])


def _write_time(
    output: netCDF4.Dataset,
    times: list[datetime],
    *,
    base: datetime,
    storage: OutputStorageConfig,
    utc: bool = False,
) -> None:
    variable = _create_time_variable(output, base=base, storage=storage, utc=utc)
    variable[:] = np.asarray([(timestamp - base).total_seconds() / 60.0 for timestamp in times], dtype=np.float64)


def _create_time_variable(
    output: netCDF4.Dataset,
    *,
    base: datetime,
    storage: OutputStorageConfig,
    utc: bool = False,
):
    variable = _create_output_variable(
        output,
        "time",
        ("time",),
        _metadata_storage(storage),
    )
    variable.long_name = "Time"
    suffix = " UTC" if utc else ""
    variable.units = f"minutes since {base:%Y-%m-%d %H:%M:%S}{suffix}"
    variable.calendar = "gregorian"
    variable.axis = "T"
    return variable


def _metadata_storage(storage: OutputStorageConfig) -> OutputStorageConfig:
    if not storage.compression.algorithm.startswith("blosc_"):
        return storage
    return replace(
        storage,
        compression=replace(storage.compression, algorithm="zlib"),
    )


def _assert_compatible_samples(
    fields: list[TracerField], *, expected_shape: tuple[int, int, int, int]
) -> None:
    first = fields[0]
    for sample in fields:
        if sample.names != first.names:
            raise ValueError("all SpeciesConc samples must have the same tracer names")
        if sample.shape != first.shape:
            raise ValueError("all SpeciesConc samples must have the same shape")
        if sample.shape[0:4] != expected_shape:
            raise ValueError(f"SpeciesConc sample shape {sample.shape} does not match template shape {expected_shape}")


def _summed_tracer(
    summed: np.ndarray, metadata: TracerField, tracer_index: int
) -> np.ndarray:
    block, lane = divmod(tracer_index, metadata.block_width)
    return summed[block, :, :, :, lane]


def _write_restart_met_field(
    output: netCDF4.Dataset,
    field: str,
    snapshot: OutputSnapshot,
    storage: OutputStorageConfig,
) -> None:
    if field == "Met_DELPDRY":
        variable = _create_output_variable(output, field, ("time", "lev", "lat", "lon"), storage)
        variable.units = "hPa"
        variable[:] = snapshot.delp_dry_hpa
        return
    if field == "Met_PS1DRY":
        variable = _create_output_variable(output, field, ("time", "lat", "lon"), storage)
        variable.units = "hPa"
        dry_surface_pressure = getattr(snapshot.forcing, "i3_start_dry_surface_pressure_hpa", None)
        if dry_surface_pressure is None:
            dry_surface_pressure = np.sum(snapshot.delp_dry_hpa, axis=1)
        variable[:] = dry_surface_pressure
        return
    if field == "Met_PS1WET":
        variable = _create_output_variable(output, field, ("time", "lat", "lon"), storage)
        variable.units = "hPa"
        wet_surface_pressure = getattr(snapshot.forcing, "i3_start_wet_surface_pressure_hpa", None)
        if wet_surface_pressure is None:
            surface_pressure = getattr(
                snapshot.forcing,
                "surface_pressure_start_pa",
                snapshot.forcing.surface_pressure_pa,
            )
            wet_surface_pressure = surface_pressure / 100.0
        variable[:] = wet_surface_pressure
        return
    if field == "Met_SPHU1":
        variable = _create_output_variable(output, field, ("time", "lev", "lat", "lon"), storage)
        variable.units = "g kg-1"
        specific_humidity = getattr(
            snapshot.forcing,
            "i3_start_specific_humidity_kg_kg",
            snapshot.forcing.specific_humidity_kg_kg,
        )
        variable[:] = specific_humidity * 1000.0
        return
    if field == "Met_TMPU1":
        variable = _create_output_variable(output, field, ("time", "lev", "lat", "lon"), storage)
        variable.units = "K"
        temperature = getattr(snapshot.forcing, "i3_start_temperature_k", snapshot.forcing.temperature_k)
        variable[:] = temperature
        return
    raise ValueError(f"unsupported restart met field {field}")


def _add_months(timestamp: datetime, months: int) -> datetime:
    month_index = timestamp.month - 1 + months
    year = timestamp.year + month_index // 12
    month = month_index % 12 + 1
    day = min(timestamp.day, _days_in_month(year, month))
    return timestamp.replace(year=year, month=month, day=day)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    return int((next_month - datetime(year, month, 1)).days)
