from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
import yaml

from wombat_transport.constants import AIRMW_G_PER_MOL, G0_M_PER_S2
from wombat_transport.fields import TracerField
from wombat_transport.grid import TransportGrid
from wombat_transport.species import Species


@dataclass(frozen=True)
class EmissionOperatorConfiguration:
    scales: dict[str, dict[str, Any]]
    fields: tuple[dict[str, Any], ...]
    unit_conversion: str
    missing_species: str


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
        self._field_cache: dict[tuple[str, datetime], np.ndarray] = {}
        self._scale_cache: dict[tuple[str, datetime], np.ndarray | float] = {}
        self._regrid_cache: dict[tuple[bytes, bytes], tuple[np.ndarray, np.ndarray]] = {}

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
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
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
        nlev, nlat, nlon = self.grid.shape
        data = np.zeros((1, nlev, nlat, nlon, len(self.species)), dtype=np.float64)
        units = [""] * len(self.species)

        for field in self.config.fields:
            species_name = str(field["species"])
            tracer_index = self._species_index[species_name]
            surface_flux = self._evaluate_field(field, valid_time).copy()
            for scale_name in field.get("scales", ()):
                scale = self._evaluate_scale(str(scale_name), valid_time)
                surface_flux *= scale

            data[0, -1, :, :, tracer_index] += surface_flux
            units[tracer_index] = str(field.get("units", units[tracer_index]))

        return TracerField(
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

    def _evaluate_field(self, field: dict[str, Any], valid_time: datetime) -> np.ndarray:
        key = (str(field["name"]), _selection_time(valid_time, str(field.get("frequency", "constant"))))
        if key not in self._field_cache:
            self._field_cache[key] = self._read_configured_array(field, key[1])
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
        path = _resolve_template_path(self.root, str(spec["path_template"]), selection_time)
        variable_name = str(spec["variable"])
        with netCDF4.Dataset(path) as dataset:
            if variable_name not in dataset.variables:
                raise KeyError(f"{path} is missing variable {variable_name}")
            variable = dataset.variables[variable_name]
            values = np.ma.filled(variable[:], np.nan)
            dims = list(variable.dimensions)

            if "time" in dims:
                axis = dims.index("time")
                index = _select_time_index(dataset, selection_time, frequency=str(spec.get("frequency", "constant")))
                values = np.take(values, index, axis=axis)
                dims.pop(axis)

            select = spec.get("select")
            if select:
                dim_name = str(select["dimension"])
                if dim_name not in dims:
                    raise ValueError(f"{path}:{variable_name} has no dimension {dim_name}")
                axis = dims.index(dim_name)
                index = _select_dimension_index(dataset, dim_name, int(select["value"]))
                values = np.take(values, index, axis=axis)
                dims.pop(axis)

            if "lat" not in dims or "lon" not in dims:
                raise ValueError(f"{path}:{variable_name} must have lat and lon dimensions after selection")
            lat_axis = dims.index("lat")
            lon_axis = dims.index("lon")
            values = np.moveaxis(np.asarray(values, dtype=np.float64), (lat_axis, lon_axis), (0, 1))
            if values.ndim != 2:
                raise ValueError(f"{path}:{variable_name} must reduce to a 2-D horizontal field")

            source_lat = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
            source_lon = np.asarray(dataset.variables["lon"][:], dtype=np.float64)

        if _same_grid(source_lat, self.grid.lat_deg) and _same_grid(source_lon, self.grid.lon_deg):
            return np.ascontiguousarray(values)
        return self._regrid_to_target(values, source_lat, source_lon)

    def _regrid_to_target(self, values: np.ndarray, source_lat: np.ndarray, source_lon: np.ndarray) -> np.ndarray:
        key = (np.ascontiguousarray(source_lat).tobytes(), np.ascontiguousarray(source_lon).tobytes())
        if key not in self._regrid_cache:
            self._regrid_cache[key] = (
                _latitude_overlap_weights(source_lat, self.grid.lat_deg),
                _longitude_overlap_weights(source_lon, self.grid.lon_deg),
            )
        lat_weights, lon_weights = self._regrid_cache[key]
        lat_denominator = lat_weights.sum(axis=1)
        lon_denominator = lon_weights.sum(axis=1)
        if np.any(lat_denominator <= 0.0) or np.any(lon_denominator <= 0.0):
            raise ValueError("source grid does not overlap target grid")

        lat_regridded = lat_weights @ values
        lat_regridded /= lat_denominator[:, np.newaxis]
        regridded = lat_regridded @ lon_weights.T
        regridded /= lon_denominator[np.newaxis, :]
        return np.ascontiguousarray(regridded)


def dry_air_mass_per_area(delp_dry_hpa: np.ndarray) -> np.ndarray:
    """Convert dry pressure thickness from hPa to kg dry air per m2."""

    return np.asarray(delp_dry_hpa, dtype=np.float64) * 100.0 / G0_M_PER_S2


def emission_increment_vv(
    emis_kg_m2_s: np.ndarray,
    delp_dry_hpa: np.ndarray,
    species: list[Species] | tuple[Species, ...],
    dt_s: float,
) -> np.ndarray:
    """Convert emissions flux into a dry volume mixing-ratio increment."""

    emissions = np.asarray(emis_kg_m2_s, dtype=np.float64)
    dry_air = dry_air_mass_per_area(delp_dry_hpa)
    if dry_air.ndim == 4:
        dry_air = dry_air[:, ::-1, :, :]
    elif dry_air.ndim == 3:
        dry_air = dry_air[::-1][np.newaxis, ...]
    else:
        raise ValueError(f"delp_dry_hpa must be 3-D or 4-D, found shape {dry_air.shape}")

    if emissions.ndim != 5:
        raise ValueError(f"emissions must be 5-D canonical tracer data, found shape {emissions.shape}")
    if emissions.shape[-1] != len(species):
        raise ValueError(
            f"emissions last dimension has {emissions.shape[-1]} tracers, "
            f"but {len(species)} species were supplied"
        )
    if dry_air.shape != emissions.shape[:-1]:
        raise ValueError(f"dry pressure shape {dry_air.shape} does not match emissions grid {emissions.shape[:-1]}")

    kgkg_dry = emissions * float(dt_s) / dry_air[..., np.newaxis]
    mw = np.asarray([item.molecular_weight_g for item in species], dtype=np.float64)
    return kgkg_dry * (AIRMW_G_PER_MOL / mw[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :])


def apply_emissions(
    tracer_field: TracerField,
    emissions: TracerField,
    delp_dry_hpa: np.ndarray,
    species: list[Species] | tuple[Species, ...],
    dt_s: float,
) -> TracerField:
    """Return a new tracer field after adding emissions increments."""

    species_names = tuple(item.name for item in species)
    if tracer_field.names != species_names:
        raise ValueError("tracer field names do not match species order")
    if emissions.names != species_names:
        raise ValueError("emission field names do not match species order")

    increment = emission_increment_vv(emissions.data, delp_dry_hpa, species, dt_s)
    return TracerField(
        names=tracer_field.names,
        data=tracer_field.data + increment,
        units=tracer_field.units,
        coords=tracer_field.coords,
    )


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


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
    index = value - 1
    if index < 0 or index >= len(dataset.dimensions[dimension]):
        raise IndexError(f"{dimension}={value} is outside dimension length {len(dataset.dimensions[dimension])}")
    return index


def _same_grid(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and bool(np.allclose(left, right, rtol=0.0, atol=1.0e-12))


def _latitude_overlap_weights(source_lat: np.ndarray, target_lat: np.ndarray) -> np.ndarray:
    source_low, source_high = _noncyclic_bounds(source_lat, lower=-90.0, upper=90.0)
    target_low, target_high = _noncyclic_bounds(target_lat, lower=-90.0, upper=90.0)
    weights = np.zeros((target_lat.size, source_lat.size), dtype=np.float64)
    for target_index, (low, high) in enumerate(zip(target_low, target_high)):
        overlap_low = np.maximum(source_low, low)
        overlap_high = np.minimum(source_high, high)
        active = overlap_high > overlap_low
        weights[target_index, active] = (
            np.sin(np.deg2rad(overlap_high[active])) - np.sin(np.deg2rad(overlap_low[active]))
        )
    return weights


def _longitude_overlap_weights(source_lon: np.ndarray, target_lon: np.ndarray) -> np.ndarray:
    source = np.asarray(source_lon, dtype=np.float64)
    order = np.argsort((source + 360.0) % 360.0)
    sorted_source = ((source + 360.0) % 360.0)[order]
    step = float(np.median(np.diff(sorted_source)))
    source_low = sorted_source - step / 2.0
    source_high = sorted_source + step / 2.0
    source_low[0] = 0.0
    source_high[-1] = 360.0

    weights_sorted = np.zeros((target_lon.size, source_lon.size), dtype=np.float64)
    target_step = float(np.median(np.diff(np.sort((np.asarray(target_lon, dtype=np.float64) + 360.0) % 360.0))))
    for target_index, center in enumerate(target_lon):
        normalized = (float(center) + 360.0) % 360.0
        low = normalized - target_step / 2.0
        high = normalized + target_step / 2.0
        intervals = [(low, high)]
        if low < 0.0:
            intervals = [(low + 360.0, 360.0), (0.0, high)]
        elif high > 360.0:
            intervals = [(low, 360.0), (0.0, high - 360.0)]
        for interval_low, interval_high in intervals:
            overlap_low = np.maximum(source_low, interval_low)
            overlap_high = np.minimum(source_high, interval_high)
            weights_sorted[target_index] += np.maximum(0.0, overlap_high - overlap_low)

    weights = np.empty_like(weights_sorted)
    weights[:, order] = weights_sorted
    return weights


def _noncyclic_bounds(centers: np.ndarray, *, lower: float, upper: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(centers, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("grid coordinate centers must be a 1-D array with at least two values")
    midpoints = (values[:-1] + values[1:]) / 2.0
    bounds = np.empty(values.size + 1, dtype=np.float64)
    bounds[1:-1] = midpoints
    bounds[0] = max(lower, values[0] - (midpoints[0] - values[0]))
    bounds[-1] = min(upper, values[-1] + (values[-1] - midpoints[-1]))
    return bounds[:-1], bounds[1:]
