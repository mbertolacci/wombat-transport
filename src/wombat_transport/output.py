from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np

from wombat_transport.fields import TracerField
from wombat_transport.io import GRID_COORDS
from wombat_transport.run_config import RunConfig, simulation_start
from wombat_transport.transport.forcing import TransportForcing

SUPPORTED_RESTART_MET_FIELDS = {"Met_DELPDRY", "Met_PS1DRY", "Met_PS1WET", "Met_SPHU1", "Met_TMPU1"}
SUPPORTED_FIELD_TOKENS = {"SpeciesConcVV_?ADV?", "SpeciesRst_?ALL?", *SUPPORTED_RESTART_MET_FIELDS}


@dataclass(frozen=True)
class OutputCompressionConfig:
    enabled: bool = True
    level: int = 5
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


@dataclass(frozen=True)
class OutputSnapshot:
    timestamp: datetime
    state: TracerField
    delp_dry_hpa: np.ndarray
    forcing: TransportForcing


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
        self._root = root
        self._template_path = template_path
        self._expid = expid
        self._start = start
        self._writers: list[_CollectionWriter] = [
            _writer_for_collection(root, template_path, expid, collection, start) for collection in collections
        ]

    @classmethod
    def from_run_config(cls, config: RunConfig) -> HistoryOutputManager | None:
        if not config.outputs:
            return None
        return cls(
            root=config.root,
            template_path=config.grid_template,
            expid=str(config.outputs.get("expid", "OutputDir/GEOSChem")),
            collections=parse_output_collections(config.outputs),
            start=simulation_start(config),
        )

    def record_step(self, snapshot: OutputSnapshot) -> None:
        for writer in self._writers:
            writer.record_step(snapshot)

    def close(self) -> None:
        for writer in self._writers:
            writer.close()


class _CollectionWriter:
    def record_step(self, snapshot: OutputSnapshot) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class _TimeAverageSpeciesWriter(_CollectionWriter):
    def __init__(
        self,
        *,
        root: Path,
        template_path: Path,
        expid: str,
        collection: OutputCollectionConfig,
        start: datetime,
    ) -> None:
        self._root = root
        self._template_path = template_path
        self._expid = expid
        self._collection = collection
        self._start = start
        self._window_start: datetime | None = None
        self._group_start: datetime | None = None
        self._sum: np.ndarray | None = None
        self._count = 0
        self._samples: list[tuple[datetime, TracerField]] = []
        self._last_state: TracerField | None = None

    def record_step(self, snapshot: OutputSnapshot) -> None:
        self._last_state = snapshot.state
        effective = snapshot.timestamp
        if effective > self._start:
            effective = effective - timedelta(microseconds=1)
        window_start = _floor_to_interval(self._start, effective, self._collection.frequency)
        if self._window_start is None:
            self._start_window(window_start)
        elif window_start != self._window_start:
            self._finish_window(snapshot.state)
            group_start = _floor_to_interval(self._start, window_start, self._collection.duration)
            if self._group_start is not None and group_start != self._group_start:
                self._write_group()
            self._start_window(window_start)

        if self._sum is None:
            self._sum = np.zeros_like(snapshot.state.data, dtype=np.float64)
        self._sum += snapshot.state.data
        self._count += 1

    def close(self) -> None:
        if self._window_start is not None and self._count:
            self._finish_window(self._last_state)
        if self._samples:
            self._write_group()

    def _start_window(self, window_start: datetime) -> None:
        self._window_start = window_start
        self._group_start = _floor_to_interval(self._start, window_start, self._collection.duration)
        self._sum = None
        self._count = 0

    def _finish_window(self, fallback_state: TracerField | None) -> None:
        if self._window_start is None or self._sum is None or self._count == 0:
            return
        if fallback_state is None:
            names = self._samples[-1][1].names if self._samples else ()
            units = self._samples[-1][1].units if self._samples else ()
            coords = self._samples[-1][1].coords if self._samples else {}
        else:
            names = fallback_state.names
            units = fallback_state.units
            coords = fallback_state.coords
        sample = TracerField(
            names=names,
            units=units,
            coords=coords,
            data=self._sum / float(self._count),
        )
        self._samples.append((self._window_start, sample))
        self._sum = None
        self._count = 0

    def _write_group(self) -> None:
        if self._group_start is None or not self._samples:
            return
        path = _collection_path(
            self._root,
            self._expid,
            self._collection,
            self._group_start,
        )
        write_species_conc_collection(
            path,
            self._samples,
            self._template_path,
            title=f"GEOS-Chem diagnostic collection: {self._collection.name}",
            storage=_collection_storage(self._collection),
        )
        self._samples = []


class _InstantaneousRestartWriter(_CollectionWriter):
    def __init__(
        self,
        *,
        root: Path,
        template_path: Path,
        expid: str,
        collection: OutputCollectionConfig,
        start: datetime,
    ) -> None:
        self._root = root
        self._template_path = template_path
        self._expid = expid
        self._collection = collection
        self._next_output = collection.frequency.add_to(start)

    def record_step(self, snapshot: OutputSnapshot) -> None:
        while snapshot.timestamp >= self._next_output:
            path = _collection_path(self._root, self._expid, self._collection, self._next_output)
            write_restart_collection(
                path,
                snapshot,
                self._template_path,
                fields=self._collection.fields,
                title=f"GEOS-Chem diagnostic collection: {self._collection.name}",
                storage=_collection_storage(self._collection),
            )
            self._next_output = self._collection.frequency.add_to(self._next_output)

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
                storage=storage,
            )
        )
    return tuple(collections)


def parse_output_storage(raw: dict[str, Any]) -> OutputStorageConfig:
    dtype = str(raw.get("dtype", "float32")).lower()
    if dtype not in {"float32", "float64"}:
        raise ValueError("outputs.dtype must be 'float32' or 'float64'")

    compression_raw = raw.get("compression", {})
    if compression_raw is None:
        compression_raw = {}
    if not isinstance(compression_raw, dict):
        raise TypeError("outputs.compression must be a mapping")
    level = int(compression_raw.get("level", 5))
    if level < 0 or level > 9:
        raise ValueError("outputs.compression.level must be between 0 and 9")
    compression = OutputCompressionConfig(
        enabled=bool(compression_raw.get("enabled", True)),
        level=level,
        shuffle=bool(compression_raw.get("shuffle", True)),
    )

    chunking_raw = raw.get("chunking", {})
    if chunking_raw is None:
        chunking_raw = {}
    if not isinstance(chunking_raw, dict):
        raise TypeError("outputs.chunking must be a mapping")
    chunking = OutputChunkingConfig(
        rank1=_parse_chunk_array(chunking_raw.get("rank1"), 1, "outputs.chunking.rank1"),
        rank2=_parse_chunk_array(chunking_raw.get("rank2"), 2, "outputs.chunking.rank2"),
        rank3=_parse_chunk_array(chunking_raw.get("rank3"), 3, "outputs.chunking.rank3"),
        rank4=_parse_chunk_array(chunking_raw.get("rank4"), 4, "outputs.chunking.rank4"),
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    times = [timestamp for timestamp, _ in samples]
    fields = [field for _, field in samples]
    with netCDF4.Dataset(template_path) as template, netCDF4.Dataset(output_path, "w") as output:
        _create_common_dimensions(output, template, time_size=len(samples), include_bounds=True)
        _assert_compatible_samples(
            fields,
            expected_shape=(
                1,
                len(template.dimensions["lev"]),
                len(template.dimensions["lat"]),
                len(template.dimensions["lon"]),
            ),
        )
        _copy_common_coordinates(output, template, include_bounds=True, storage=storage)
        _write_time(output, times, base=times[0], storage=storage)
        output.title = title
        output.format = "NetCDF-4"
        first = fields[0]
        for tracer_index, tracer_name in enumerate(first.names):
            variable = _create_output_variable(
                output,
                f"SpeciesConcVV_{tracer_name}",
                ("time", "lev", "lat", "lon"),
                storage,
            )
            variable.units = first.units[tracer_index] if tracer_index < len(first.units) else "mol mol-1 dry"
            variable.long_name = f"Dry mixing ratio of species {tracer_name}"
            variable[:] = np.stack([field.data[0, ::-1, :, :, tracer_index] for field in fields], axis=0)
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
        _create_common_dimensions(output, template, time_size=1, include_bounds=False)
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
                variable.long_name = f"Wombat restart concentration of species {tracer_name}"
                variable[:] = snapshot.state.data[:, ::-1, :, :, tracer_index]
        for field in fields:
            if field in SUPPORTED_RESTART_MET_FIELDS:
                _write_restart_met_field(output, field, snapshot, storage)
    return output_path


def _writer_for_collection(
    root: Path,
    template_path: Path,
    expid: str,
    collection: OutputCollectionConfig,
    start: datetime,
) -> _CollectionWriter:
    if collection.mode == "time-averaged" and collection.fields == ("SpeciesConcVV_?ADV?",):
        return _TimeAverageSpeciesWriter(
            root=root,
            template_path=template_path,
            expid=expid,
            collection=collection,
            start=start,
        )
    if collection.mode == "instantaneous" and "SpeciesRst_?ALL?" in collection.fields:
        if collection.frequency.is_zero:
            raise ValueError(f"output collection {collection.name} frequency must be nonzero")
        return _InstantaneousRestartWriter(
            root=root,
            template_path=template_path,
            expid=expid,
            collection=collection,
            start=start,
        )
    raise ValueError(f"unsupported output collection {collection.name}: mode={collection.mode}, fields={collection.fields}")


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
    time_size: int,
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
        if storage.compression.enabled:
            kwargs.update(
                {
                    "zlib": True,
                    "complevel": storage.compression.level,
                    "shuffle": storage.compression.shuffle,
                }
            )
    return output.createVariable(name, storage.netcdf_dtype, dimensions, **kwargs)


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
        variable = _create_output_variable(output, coord_name, source.dimensions, storage)
        variable.setncatts({name: source.getncattr(name) for name in source.ncattrs()})
        if source.dimensions:
            variable[:] = np.asarray(source[:])
        else:
            variable.assignValue(source.getValue())
    if include_bounds:
        for coord_name in ("lat_bnds", "lon_bnds"):
            if coord_name in template.variables:
                source = template.variables[coord_name]
                variable = _create_output_variable(output, coord_name, source.dimensions, storage)
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
    variable = _create_output_variable(output, "time", ("time",), storage)
    variable.long_name = "Time"
    suffix = " UTC" if utc else ""
    variable.units = f"minutes since {base:%Y-%m-%d %H:%M:%S}{suffix}"
    variable.calendar = "gregorian"
    variable.axis = "T"
    variable[:] = np.asarray([(timestamp - base).total_seconds() / 60.0 for timestamp in times], dtype=np.float64)


def _assert_compatible_samples(fields: list[TracerField], *, expected_shape: tuple[int, int, int, int]) -> None:
    first = fields[0]
    for field in fields:
        if field.names != first.names:
            raise ValueError("all SpeciesConc samples must have the same tracer names")
        if field.data.shape != first.data.shape:
            raise ValueError("all SpeciesConc samples must have the same shape")
        if field.data.shape[0:4] != expected_shape:
            raise ValueError(f"SpeciesConc sample shape {field.data.shape} does not match template shape {expected_shape}")


def _write_restart_met_field(
    output: netCDF4.Dataset,
    field: str,
    snapshot: OutputSnapshot,
    storage: OutputStorageConfig,
) -> None:
    if field == "Met_DELPDRY":
        variable = _create_output_variable(output, field, ("time", "lev", "lat", "lon"), storage)
        variable.units = "hPa"
        variable[:] = snapshot.delp_dry_hpa[:, ::-1, :, :]
        return
    if field == "Met_PS1DRY":
        variable = _create_output_variable(output, field, ("time", "lat", "lon"), storage)
        variable.units = "hPa"
        variable[:] = np.sum(snapshot.delp_dry_hpa, axis=1)
        return
    if field == "Met_PS1WET":
        variable = _create_output_variable(output, field, ("time", "lat", "lon"), storage)
        variable.units = "hPa"
        surface_pressure = getattr(
            snapshot.forcing,
            "restart_surface_pressure_pa",
            snapshot.forcing.surface_pressure_pa,
        )
        variable[:] = surface_pressure / 100.0
        return
    if field == "Met_SPHU1":
        variable = _create_output_variable(output, field, ("time", "lev", "lat", "lon"), storage)
        variable.units = "kg kg-1"
        specific_humidity = getattr(
            snapshot.forcing,
            "restart_specific_humidity_kg_kg",
            snapshot.forcing.specific_humidity_kg_kg,
        )
        variable[:] = specific_humidity[:, ::-1, :, :]
        return
    if field == "Met_TMPU1":
        variable = _create_output_variable(output, field, ("time", "lev", "lat", "lon"), storage)
        variable.units = "K"
        temperature = getattr(snapshot.forcing, "restart_temperature_k", snapshot.forcing.temperature_k)
        variable[:] = temperature[:, ::-1, :, :]
        return
    raise ValueError(f"unsupported restart met field {field}")


def _floor_to_interval(start: datetime, timestamp: datetime, interval: HistoryInterval) -> datetime:
    if interval.is_zero:
        raise ValueError("cannot floor to a zero HISTORY interval")
    current = start
    while True:
        next_time = interval.add_to(current)
        if next_time > timestamp:
            return current
        current = next_time


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
