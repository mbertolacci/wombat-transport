from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
from yaml12 import read_yaml

from wombat_transport.fields import TracerField
from wombat_transport.grid import TransportGrid, geos_chem_latitude_edges_deg
from wombat_transport.species import Species


@dataclass(frozen=True)
class EmissionOperatorConfiguration:
    scales: dict[str, dict[str, Any]]
    fields: tuple[dict[str, Any], ...]
    unit_conversion: str
    missing_species: str


@dataclass(frozen=True)
class SurfaceEmissions:
    """Surface-only emissions in transport layout ``(lat, lon, tracer)``."""

    names: tuple[str, ...]
    data: np.ndarray
    units: tuple[str, ...]
    coords: dict[str, np.ndarray]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def to_tracer_field(self, nlev: int) -> TracerField:
        full = np.zeros((1, int(nlev), self.data.shape[0], self.data.shape[1], self.data.shape[2]), dtype=np.float64)
        full[0, -1, :, :, :] = self.data
        return TracerField(names=self.names, data=full, units=self.units, coords=self.coords)


@dataclass(frozen=True)
class _SourceSlice:
    values: np.ndarray
    dims: tuple[str, ...]
    lat: np.ndarray
    lon: np.ndarray


_HORIZONTAL_DIMENSION_ALIASES = {
    "lat": "lat",
    "latitude": "lat",
    "lon": "lon",
    "longitude": "lon",
}


class EmissionsOperator:
    """Evaluate explicitly configured raw emissions fields for one timestep."""

    def __init__(
        self,
        config: EmissionOperatorConfiguration,
        *,
        root: Path,
        species: list[Species] | tuple[Species, ...],
        grid: TransportGrid,
    ) -> None:
        if config.unit_conversion != "none":
            raise ValueError("configured emissions only support unit_conversion: none")
        if config.missing_species != "zero":
            raise ValueError("configured emissions only support missing_species: zero")
        self.config = config
        self.root = root
        self.species = tuple(species)
        self.grid = grid
        self._species_index = {item.name: index for index, item in enumerate(self.species)}
        self._field_cache: dict[tuple[object, ...], np.ndarray] = {}
        self._scale_cache: dict[tuple[str, datetime], np.ndarray | float] = {}
        self._source_cache: dict[tuple[object, ...], _SourceSlice] = {}
        self._regrid_cache: dict[tuple[bytes, bytes], ConservativeRemappingWeights] = {}
        self._surface_cache_key: tuple[datetime, ...] | None = None
        self._surface_cache_value: SurfaceEmissions | None = None

        for field in self.config.fields:
            species_name = str(field["species"])
            if species_name not in self._species_index:
                raise ValueError(f"configured emissions field references unknown species {species_name}")
            for scale_name in field.get("scales", ()):
                if scale_name not in self.config.scales:
                    raise ValueError(f"configured emissions field {field['name']} references unknown scale {scale_name}")

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        root: Path,
        species: list[Species] | tuple[Species, ...],
        grid: TransportGrid,
    ) -> "EmissionsOperator":
        config_path = _resolve_path(root, str(path))
        raw = read_yaml(config_path) or {}
        return cls.from_mapping(raw, root=config_path.parent, species=species, grid=grid)

    @classmethod
    def from_mapping(
        cls,
        raw: dict[str, Any],
        *,
        root: Path,
        species: list[Species] | tuple[Species, ...],
        grid: TransportGrid,
    ) -> "EmissionsOperator":
        config = EmissionOperatorConfiguration(
            scales=dict(raw.get("scales", {})),
            fields=tuple(dict(item) for item in raw.get("fields", ())),
            unit_conversion=str(raw.get("unit_conversion", "none")),
            missing_species=str(raw.get("missing_species", "zero")),
        )
        return cls(config, root=root, species=species, grid=grid)

    @property
    def emitted_species(self) -> tuple[str, ...]:
        return tuple(str(field["species"]) for field in self.config.fields)

    def evaluate(self, valid_time: datetime) -> TracerField:
        return self.evaluate_surface_flux(valid_time).to_tracer_field(self.grid.shape[0])

    def evaluate_surface_flux(self, valid_time: datetime) -> SurfaceEmissions:
        cache_key = self._surface_emissions_key(valid_time)
        if (
            cache_key == self._surface_cache_key
            and self._surface_cache_value is not None
        ):
            return self._surface_cache_value
        _nlev, nlat, nlon = self.grid.shape
        data = np.zeros((nlat, nlon, len(self.species)), dtype=np.float64)
        units = [""] * len(self.species)

        for field in self.config.fields:
            species_name = str(field["species"])
            tracer_index = self._species_index[species_name]
            surface_flux = self._evaluate_field(field, valid_time).copy()
            for scale_name in field.get("scales", ()):
                scale = self._evaluate_scale(str(scale_name), valid_time)
                surface_flux *= scale

            data[:, :, tracer_index] += surface_flux
            units[tracer_index] = str(field.get("units", units[tracer_index]))

        self._prune_temporal_caches(valid_time)
        result = SurfaceEmissions(
            names=tuple(item.name for item in self.species),
            data=data,
            units=tuple(units),
            coords={
                "lev": self.grid.lev,
                "lat": self.grid.lat_deg,
                "lon": self.grid.lon_deg,
                "AREA": self.grid.area_m2,
            },
        )
        self._surface_cache_key = cache_key
        self._surface_cache_value = result
        return result

    def _surface_emissions_key(
        self,
        valid_time: datetime,
    ) -> tuple[datetime, ...]:
        selections: list[datetime] = []
        for field in self.config.fields:
            selections.append(
                _selection_time(
                    valid_time,
                    str(field.get("frequency", "constant")),
                )
            )
            for scale_name in field.get("scales", ()):
                scale = self.config.scales[str(scale_name)]
                selections.append(
                    _selection_time(
                        valid_time,
                        str(scale.get("frequency", "constant")),
                    )
                )
        return tuple(selections)

    def _evaluate_field(self, field: dict[str, Any], valid_time: datetime) -> np.ndarray:
        selection_time = _selection_time(valid_time, str(field.get("frequency", "constant")))
        key = self._configured_array_key(field, selection_time)
        if key not in self._field_cache:
            self._field_cache[key] = self._read_configured_array(field, selection_time)
        return self._field_cache[key]

    def _evaluate_scale(self, name: str, valid_time: datetime) -> np.ndarray | float:
        spec = self.config.scales[name]
        if "value" in spec:
            return float(spec["value"])

        key = (name, _selection_time(valid_time, str(spec.get("frequency", "constant"))))
        if key not in self._scale_cache:
            self._scale_cache[key] = self._read_configured_array(spec, key[1])
        return self._scale_cache[key]

    def _read_configured_array(self, spec: dict[str, Any], selection_time: datetime) -> np.ndarray:
        source = self._read_source_slice(spec, selection_time)
        values = source.values
        dims = list(source.dims)
        path = _resolve_template_path(self.root, str(spec["path_template"]), selection_time)
        variable_name = str(spec["variable"])

        select = spec.get("select")
        if select:
            dim_name = str(select["dimension"])
            if dim_name not in dims:
                raise ValueError(f"{path}:{variable_name} has no dimension {dim_name}")
            axis = dims.index(dim_name)
            index = _select_dimension_index_from_size(dims, dim_name, int(select["value"]), values.shape[axis])
            values = np.take(values, index, axis=axis)
            dims.pop(axis)

        if "lat" not in dims or "lon" not in dims:
            raise ValueError(f"{path}:{variable_name} must have lat and lon dimensions after selection")
        lat_axis = dims.index("lat")
        lon_axis = dims.index("lon")
        values = np.moveaxis(np.asarray(values, dtype=np.float64), (lat_axis, lon_axis), (0, 1))
        if values.ndim != 2:
            raise ValueError(f"{path}:{variable_name} must reduce to a 2-D horizontal field")

        if _same_grid(source.lat, self.grid.lat_deg) and _same_grid(source.lon, self.grid.lon_deg):
            return np.ascontiguousarray(values)
        return self._regrid_to_target(values, source.lat, source.lon)

    def _read_source_slice(self, spec: dict[str, Any], selection_time: datetime) -> _SourceSlice:
        key = self._source_slice_key(spec, selection_time)
        if key in self._source_cache:
            return self._source_cache[key]

        path = _resolve_template_path(self.root, str(spec["path_template"]), selection_time)
        variable_name = str(spec["variable"])
        with netCDF4.Dataset(path) as dataset:
            if variable_name not in dataset.variables:
                raise KeyError(f"{path} is missing variable {variable_name}")
            variable = dataset.variables[variable_name]
            dims = [_HORIZONTAL_DIMENSION_ALIASES.get(name, name) for name in variable.dimensions]
            slices: list[object] = [slice(None)] * len(dims)
            if "time" in dims:
                axis = dims.index("time")
                index = _select_time_index(dataset, selection_time, frequency=str(spec.get("frequency", "constant")))
                slices[axis] = index
                dims.pop(axis)
            values = np.ma.filled(variable[tuple(slices)], np.nan)
            source = _SourceSlice(
                values=np.asarray(values, dtype=np.float64),
                dims=tuple(dims),
                lat=_read_horizontal_coordinate(dataset, ("lat", "latitude"), path=path),
                lon=_read_horizontal_coordinate(dataset, ("lon", "longitude"), path=path),
            )
        self._source_cache[key] = source
        return source

    def _regrid_to_target(self, values: np.ndarray, source_lat: np.ndarray, source_lon: np.ndarray) -> np.ndarray:
        key = (np.ascontiguousarray(source_lat).tobytes(), np.ascontiguousarray(source_lon).tobytes())
        if key not in self._regrid_cache:
            self._regrid_cache[key] = ConservativeRemappingWeights(
                source_lat=source_lat,
                source_lon=source_lon,
                target_lat=self.grid.lat_deg,
                target_lon=self.grid.lon_deg,
            )
        return self._regrid_cache[key].apply(values)

    def _configured_array_key(self, spec: dict[str, Any], selection_time: datetime) -> tuple[object, ...]:
        select = spec.get("select") or {}
        return (
            "array",
            str(_resolve_template_path(self.root, str(spec["path_template"]), selection_time)),
            str(spec["variable"]),
            selection_time,
            str(select.get("dimension", "")),
            int(select["value"]) if "value" in select else None,
        )

    def _source_slice_key(self, spec: dict[str, Any], selection_time: datetime) -> tuple[object, ...]:
        return (
            "source",
            str(_resolve_template_path(self.root, str(spec["path_template"]), selection_time)),
            str(spec["variable"]),
            selection_time,
        )

    def _prune_temporal_caches(self, valid_time: datetime) -> None:
        keep_times = {datetime(2000, 1, 1)}
        for field in self.config.fields:
            keep_times.add(_selection_time(valid_time, str(field.get("frequency", "constant"))))
            for scale_name in field.get("scales", ()):
                spec = self.config.scales[str(scale_name)]
                keep_times.add(_selection_time(valid_time, str(spec.get("frequency", "constant"))))

        self._field_cache = {key: value for key, value in self._field_cache.items() if key[3] in keep_times}
        self._source_cache = {key: value for key, value in self._source_cache.items() if key[3] in keep_times}
        self._scale_cache = {key: value for key, value in self._scale_cache.items() if key[1] in keep_times}

def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _read_horizontal_coordinate(
    dataset: netCDF4.Dataset,
    names: tuple[str, str],
    *,
    path: Path,
) -> np.ndarray:
    matches = [name for name in names if name in dataset.variables]
    if not matches:
        raise KeyError(f"{path} is missing horizontal coordinate {names[0]!r} or {names[1]!r}")
    if len(matches) > 1:
        left = np.asarray(dataset.variables[matches[0]][:], dtype=np.float64)
        right = np.asarray(dataset.variables[matches[1]][:], dtype=np.float64)
        if not _same_grid(left, right):
            raise ValueError(f"{path} has conflicting {matches[0]!r} and {matches[1]!r} coordinates")
    values = np.asarray(dataset.variables[matches[0]][:], dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{path}:{matches[0]} must be a 1-D coordinate")
    return values


def _resolve_template_path(root: Path, template: str, timestamp: datetime) -> Path:
    value = (
        template.replace("$YYYY", f"{timestamp.year:04d}")
        .replace("$MM", f"{timestamp.month:02d}")
        .replace("$DD", f"{timestamp.day:02d}")
        .replace("$HH", f"{timestamp.hour:02d}")
    )
    return _resolve_path(root, value)


def _selection_time(valid_time: datetime, frequency: str) -> datetime:
    if frequency == "hourly":
        return valid_time.replace(minute=0, second=0, microsecond=0)
    if frequency == "monthly":
        return valid_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if frequency == "daily":
        return valid_time.replace(hour=0, minute=0, second=0, microsecond=0)
    if frequency == "constant":
        return datetime(2000, 1, 1)
    raise ValueError(f"unsupported emissions frequency {frequency}")


def _select_time_index(dataset: netCDF4.Dataset, timestamp: datetime, *, frequency: str) -> int:
    if "time" not in dataset.variables:
        raise ValueError("time dimension is present but no time coordinate variable exists")
    time_var = dataset.variables["time"]
    units = getattr(time_var, "units")
    calendar = getattr(time_var, "calendar", "standard")
    target = netCDF4.date2num(timestamp, units, calendar=calendar)
    values = np.asarray(time_var[:], dtype=np.float64)
    index = int(np.argmin(np.abs(values - target)))
    tolerance_hours = {"hourly": 0.51, "daily": 12.01, "monthly": 24.01, "constant": np.inf}[frequency]
    tolerance = (
        np.inf
        if not np.isfinite(tolerance_hours)
        else abs(
            float(netCDF4.date2num(timestamp + timedelta(hours=tolerance_hours), units, calendar=calendar))
            - float(target)
        )
    )
    if abs(float(values[index]) - float(target)) > tolerance:
        raise ValueError(f"no {frequency} emissions time found for {timestamp:%Y-%m-%d %H:%M}")
    return index


def _select_dimension_index(dataset: netCDF4.Dataset, dimension: str, value: int) -> int:
    return _select_dimension_index_from_size([dimension], dimension, value, len(dataset.dimensions[dimension]))


def _select_dimension_index_from_size(dims: list[str], dimension: str, value: int, size: int) -> int:
    if dimension not in dims:
        raise ValueError(f"dimension {dimension!r} is not present")
    index = value - 1
    if index < 0 or index >= size:
        raise IndexError(f"{dimension}={value} is outside dimension length {size}")
    return index


def _same_grid(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and bool(np.allclose(left, right, rtol=0.0, atol=1.0e-12))


def _latitude_overlap_weights(source_lat: np.ndarray, target_lat: np.ndarray) -> np.ndarray:
    source_low, source_high = _noncyclic_bounds(source_lat, lower=-90.0, upper=90.0)
    target_low, target_high = _noncyclic_bounds(target_lat, lower=-90.0, upper=90.0)
    weights = np.zeros((target_lat.size, source_lat.size), dtype=np.float64)
    for target_index, (low, high) in enumerate(zip(target_low, target_high, strict=True)):
        overlap_low = np.maximum(source_low, low)
        overlap_high = np.minimum(source_high, high)
        active = overlap_high > overlap_low
        weights[target_index, active] = (
            np.sin(np.deg2rad(overlap_high[active])) - np.sin(np.deg2rad(overlap_low[active]))
        )
    return weights


@dataclass(frozen=True, eq=False)
class ConservativeRemappingWeights:
    """Immutable conservative horizontal-remapping geometry."""

    source_lat: np.ndarray
    source_lon: np.ndarray
    target_lat: np.ndarray
    target_lon: np.ndarray
    latitude_overlap: np.ndarray = dataclass_field(init=False, repr=False)
    longitude_overlap: np.ndarray = dataclass_field(init=False, repr=False)
    latitude_denominator: np.ndarray = dataclass_field(init=False, repr=False)
    longitude_denominator: np.ndarray = dataclass_field(init=False, repr=False)

    def __post_init__(self) -> None:
        source_lat = _immutable_float64_copy(self.source_lat)
        source_lon = _immutable_float64_copy(self.source_lon)
        target_lat = _immutable_float64_copy(self.target_lat)
        target_lon = _immutable_float64_copy(self.target_lon)
        latitude_overlap = _latitude_overlap_weights(source_lat, target_lat)
        longitude_overlap = _longitude_overlap_weights(source_lon, target_lon)
        latitude_denominator = latitude_overlap.sum(axis=1)
        longitude_denominator = longitude_overlap.sum(axis=1)
        if np.any(latitude_denominator <= 0.0) or np.any(longitude_denominator <= 0.0):
            raise ValueError("source grid does not overlap target grid")

        object.__setattr__(self, "source_lat", source_lat)
        object.__setattr__(self, "source_lon", source_lon)
        object.__setattr__(self, "target_lat", target_lat)
        object.__setattr__(self, "target_lon", target_lon)
        object.__setattr__(self, "latitude_overlap", _immutable_float64_copy(latitude_overlap))
        object.__setattr__(self, "longitude_overlap", _immutable_float64_copy(longitude_overlap))
        object.__setattr__(self, "latitude_denominator", _immutable_float64_copy(latitude_denominator))
        object.__setattr__(self, "longitude_denominator", _immutable_float64_copy(longitude_denominator))

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Apply cached weights to arrays ending in source latitude/longitude."""

        array = np.asarray(values, dtype=np.float64)
        source_shape = (self.source_lat.size, self.source_lon.size)
        if array.shape[-2:] != source_shape:
            raise ValueError(
                f"source field shape {array.shape[-2:]} does not match coordinates {source_shape}"
            )

        leading_shape = array.shape[:-2]
        planes = array.reshape((-1, *source_shape))
        lat_regridded = np.matmul(self.latitude_overlap[np.newaxis, :, :], planes)
        lat_regridded /= self.latitude_denominator[np.newaxis, :, np.newaxis]
        for plane in lat_regridded:
            _average_regridded_polar_rows(
                plane,
                self.source_lat,
                self.target_lat,
                self.source_lon,
            )
        regridded = np.matmul(lat_regridded, self.longitude_overlap.T)
        regridded /= self.longitude_denominator[np.newaxis, np.newaxis, :]
        target_shape = (self.target_lat.size, self.target_lon.size)
        return np.ascontiguousarray(regridded.reshape((*leading_shape, *target_shape)))


def conservative_regrid_horizontal(
    values: np.ndarray,
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> np.ndarray:
    """Conservatively remap arrays whose final dimensions are latitude/longitude."""

    weights = ConservativeRemappingWeights(
        source_lat=source_lat,
        source_lon=source_lon,
        target_lat=target_lat,
        target_lon=target_lon,
    )
    return weights.apply(values)


def _longitude_overlap_weights(source_lon: np.ndarray, target_lon: np.ndarray) -> np.ndarray:
    source_intervals = _cyclic_cell_intervals(source_lon)
    target_intervals = _cyclic_cell_intervals(target_lon)
    weights = np.zeros((len(target_intervals), len(source_intervals)), dtype=np.float64)
    for target_index, target_parts in enumerate(target_intervals):
        for source_index, source_parts in enumerate(source_intervals):
            for target_low, target_high in target_parts:
                for source_low, source_high in source_parts:
                    weights[target_index, source_index] += max(
                        0.0,
                        min(target_high, source_high) - max(target_low, source_low),
                    )
    return weights


def _average_regridded_polar_rows(
    lat_regridded: np.ndarray,
    source_lat: np.ndarray,
    target_lat: np.ndarray,
    source_lon: np.ndarray,
) -> None:
    source_low, source_high = _noncyclic_bounds(source_lat, lower=-90.0, upper=90.0)
    target_low, target_high = _noncyclic_bounds(target_lat, lower=-90.0, upper=90.0)
    lon_widths = _longitude_cell_widths(source_lon)
    if (
        source_lat[0] <= -89.0
        and target_lat[0] <= -89.0
        and np.isclose(source_low[0], -90.0, rtol=0.0, atol=1.0e-12)
        and np.isclose(target_low[0], -90.0, rtol=0.0, atol=1.0e-12)
    ):
        lat_regridded[0, :] = np.average(lat_regridded[0, :], weights=lon_widths)
    if (
        source_lat[-1] >= 89.0
        and target_lat[-1] >= 89.0
        and np.isclose(source_high[-1], 90.0, rtol=0.0, atol=1.0e-12)
        and np.isclose(target_high[-1], 90.0, rtol=0.0, atol=1.0e-12)
    ):
        lat_regridded[-1, :] = np.average(lat_regridded[-1, :], weights=lon_widths)


def _longitude_cell_widths(lon: np.ndarray) -> np.ndarray:
    return np.asarray(
        [sum(high - low for low, high in intervals) for intervals in _cyclic_cell_intervals(lon)],
        dtype=np.float64,
    )


def _immutable_float64_copy(values: np.ndarray) -> np.ndarray:
    snapshot = np.array(values, dtype=np.float64, copy=True, order="C")
    snapshot.flags.writeable = False
    return snapshot


def _cyclic_cell_intervals(lon: np.ndarray) -> list[tuple[tuple[float, float], ...]]:
    values = np.asarray(lon, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("longitude centers must be a 1-D array with at least two values")
    normalized = values % 360.0
    order = np.argsort(normalized)
    sorted_values = normalized[order]
    gaps = np.diff(np.concatenate((sorted_values, sorted_values[:1] + 360.0)))
    if np.any(gaps <= 0.0):
        raise ValueError("longitude centers must be unique on the cyclic grid")

    sorted_intervals: list[tuple[tuple[float, float], ...]] = []
    for index, center in enumerate(sorted_values):
        low = center - gaps[index - 1] / 2.0
        high = center + gaps[index] / 2.0
        if low < 0.0:
            parts = ((low + 360.0, 360.0), (0.0, high))
        elif high > 360.0:
            parts = ((low, 360.0), (0.0, high - 360.0))
        else:
            parts = ((low, high),)
        sorted_intervals.append(parts)

    intervals: list[tuple[tuple[float, float], ...]] = [()] * values.size
    for sorted_index, original_index in enumerate(order):
        intervals[int(original_index)] = sorted_intervals[sorted_index]
    return intervals


def _noncyclic_bounds(centers: np.ndarray, *, lower: float, upper: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(centers, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("grid coordinate centers must be a 1-D array with at least two values")
    if lower == -90.0 and upper == 90.0:
        bounds = geos_chem_latitude_edges_deg(values)
        return bounds[:-1], bounds[1:]
    midpoints = (values[:-1] + values[1:]) / 2.0
    bounds = np.empty(values.size + 1, dtype=np.float64)
    bounds[1:-1] = midpoints
    bounds[0] = max(lower, values[0] - (midpoints[0] - values[0]))
    bounds[-1] = min(upper, values[-1] + (values[-1] - midpoints[-1]))
    return bounds[:-1], bounds[1:]
