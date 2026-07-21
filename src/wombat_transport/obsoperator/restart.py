from __future__ import annotations

from datetime import datetime
import hashlib
import logging
import os
from pathlib import Path
import tempfile
from typing import Any

import netCDF4
import numpy as np

from wombat_transport.grid import TransportGrid
from wombat_transport.obsoperator.state import (
    FIELD_PREFIX,
    MAX_FIELD_NAME_LENGTH,
    MAX_ID_LENGTH,
    ObsPlan,
    VERTICAL_TYPE_CODES,
    VERTICAL_UNIT_CODES,
)
from wombat_transport.obsoperator.utils import (
    _datetime_to_microseconds,
    _nul_padded_matrix,
    _seconds_to_microseconds,
    _validate_vertical_bounds,
    _validate_vertical_values,
)

logger = logging.getLogger(__name__)

RESTART_FORMAT = "Wombat ObsOperator restart"
RESTART_FORMAT_VERSION = 3

_VARIABLE_SPECS = {
    "id": ("S1", ("entries", "id_chars")),
    "field_name": ("S1", ("fields", "field_chars")),
    "accumulator": ("f8", ("accumulators",)),
    "entry_field_start": ("i8", ("entries",)),
    "entry_field_count": ("i4", ("entries",)),
    "field_tracer": ("i8", ("fields",)),
    "field_to_accumulator": ("i8", ("fields",)),
    "time_operator_start": ("i8", ("entries",)),
    "time_operator_count": ("i4", ("entries",)),
    "time_operator_bounds_us": ("i8", ("time_operators", "bound")),
    "time_operator_weight": ("f8", ("time_operators",)),
    "horizontal_operator_start": ("i8", ("entries",)),
    "horizontal_operator_count": ("i4", ("entries",)),
    "horizontal_operator_bounds": (
        "i4",
        ("horizontal_operators", "horizontal_dimension", "bound"),
    ),
    "horizontal_weight_type": ("i1", ("horizontal_operators",)),
    "horizontal_weight": ("f8", ("horizontal_operators",)),
    "horizontal_normalization": ("f8", ("horizontal_operators",)),
    "vertical_operator_start": ("i8", ("entries",)),
    "vertical_operator_count": ("i4", ("entries",)),
    "vertical_operator_type": ("i1", ("vertical_operators",)),
    "vertical_operator_unit": ("i1", ("vertical_operators",)),
    "vertical_operator_bounds": ("f8", ("vertical_operators", "bound")),
    "vertical_weight_type": ("i1", ("vertical_operators",)),
    "vertical_weight": ("f8", ("vertical_operators",)),
    "entry_end_us": ("i8", ("entries",)),
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


def _write_obsoperator_restart(
    path: Path,
    *,
    plan: ObsPlan,
    restart_time: datetime,
    transport_dt_s: float,
    grid: TransportGrid,
) -> None:
    plan.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        _write_restart_file(
            temporary_path,
            plan=plan,
            restart_time=restart_time,
            transport_dt_s=transport_dt_s,
            grid=grid,
        )
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    logger.info("obsoperator_restart_written path=%s entries=%d", path, plan.entry_count)


def _write_restart_file(
    path: Path,
    *,
    plan: ObsPlan,
    restart_time: datetime,
    transport_dt_s: float,
    grid: TransportGrid,
) -> None:
    dimensions = {
        "entries": plan.entry_count,
        "fields": len(plan.field_names),
        "accumulators": plan.accumulator.size,
        "time_operators": plan.time_operator_weight.size,
        "horizontal_operators": plan.horizontal_weight.size,
        "vertical_operators": plan.vertical_weight.size,
        "id_chars": MAX_ID_LENGTH,
        "field_chars": MAX_FIELD_NAME_LENGTH,
        "bound": 2,
        "horizontal_dimension": 2,
    }
    with netCDF4.Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.set_fill_off()
        dataset.setncattr("format", RESTART_FORMAT)
        dataset.setncattr("format_version", np.int32(RESTART_FORMAT_VERSION))
        dataset.setncattr("restart_time_us", np.int64(_datetime_to_microseconds(restart_time)))
        dataset.setncattr("transport_timestep_seconds", np.float64(transport_dt_s))
        dataset.setncattr("grid_signature", _grid_signature(grid))
        for name, size in dimensions.items():
            dataset.createDimension(name, int(size))
        variables = {
            name: dataset.createVariable(
                name, dtype, dims, zlib=True, complevel=1, shuffle=True
            )
            for name, (dtype, dims) in _VARIABLE_SPECS.items()
        }
        variables["id"][:] = _nul_padded_matrix(plan.ids, MAX_ID_LENGTH, plan.entry_count)
        variables["field_name"][:] = _nul_padded_matrix(
            plan.field_names, MAX_FIELD_NAME_LENGTH, len(plan.field_names)
        )
        for name in _VARIABLE_SPECS:
            if name in {"id", "field_name"}:
                continue
            variables[name][:] = getattr(plan, name)
        variables["time_operator_bounds_us"].units = "microseconds since 1970-01-01 00:00:00"
        variables["horizontal_operator_bounds"].description = (
            "zero-based half-open bounds ordered as latitude, longitude"
        )


def _read_obsoperator_restart(
    path: Path,
    *,
    restart_time: datetime,
    transport_dt_s: float,
    tracer_names: tuple[str, ...],
    grid: TransportGrid,
) -> ObsPlan:
    try:
        with netCDF4.Dataset(path) as dataset:
            dataset.set_auto_mask(False)
            if getattr(dataset, "format", None) != RESTART_FORMAT:
                raise ValueError(f"ObsOperator restart {path} has an invalid format")
            version = int(getattr(dataset, "format_version", -1))
            if version != RESTART_FORMAT_VERSION:
                raise ValueError(
                    f"ObsOperator restart {path} has unsupported format version {version}; "
                    f"version {RESTART_FORMAT_VERSION} is required"
                )
            expected_time_us = _datetime_to_microseconds(restart_time)
            if int(getattr(dataset, "restart_time_us", -1)) != expected_time_us:
                raise ValueError(
                    f"ObsOperator restart {path} does not match simulation start "
                    f"{restart_time.isoformat()}"
                )
            stored_dt = float(getattr(dataset, "transport_timestep_seconds", np.nan))
            if stored_dt != float(transport_dt_s):
                raise ValueError(
                    "ObsOperator restart is incompatible: transport timestep changed from "
                    f"{stored_dt:g} s to {float(transport_dt_s):g} s"
                )
            if getattr(dataset, "grid_signature", None) != _grid_signature(grid):
                raise ValueError("ObsOperator restart is incompatible: transport grid changed")
            if not set(_VARIABLE_SPECS).issubset(dataset.variables):
                raise ValueError(f"ObsOperator restart {path} is missing required variables")
            for name, (dtype, dimensions) in _VARIABLE_SPECS.items():
                variable = dataset.variables[name]
                if variable.dtype != np.dtype(dtype) or variable.dimensions != dimensions:
                    raise ValueError(
                        f"ObsOperator restart variable {name!r} has an invalid dtype or dimension order"
                    )
            ids = tuple(_decode_rows(dataset.variables["id"][:], MAX_ID_LENGTH, "id"))
            field_names = tuple(
                _decode_rows(
                    dataset.variables["field_name"][:],
                    MAX_FIELD_NAME_LENGTH,
                    "field_name",
                )
            )
            values: dict[str, np.ndarray | tuple[str, ...] | int] = {
                "ids": ids,
                "field_names": field_names,
                "first_unexpired": 0,
            }
            for name, (dtype, _) in _VARIABLE_SPECS.items():
                if name in {"id", "field_name"}:
                    continue
                values[name] = np.asarray(dataset.variables[name][:], dtype=np.dtype(dtype))
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"cannot read ObsOperator restart {path}: {exc}") from exc

    plan = ObsPlan(**values)  # type: ignore[arg-type]
    plan.validate()
    dt_us = _seconds_to_microseconds(transport_dt_s, "transport timestep")
    if plan.time_operator_bounds_us.size and np.any(
        (plan.time_operator_bounds_us - expected_time_us) % dt_us != 0
    ):
        raise ValueError("ObsOperator restart contains times not aligned to the transport timestep")
    nlat = grid.lat_deg.size
    nlon = grid.lon_deg.size
    horizontal = plan.horizontal_operator_bounds
    if horizontal.size and (
        np.any(horizontal[:, 0, 0] < 0)
        or np.any(horizontal[:, 0, 1] > nlat)
        or np.any(horizontal[:, 1, 0] < 0)
        or np.any(horizontal[:, 1, 1] > nlon)
    ):
        raise ValueError("ObsOperator restart contains horizontal bounds outside the transport grid")
    unit_names = {code: name for name, code in VERTICAL_UNIT_CODES.items()}
    for vertical_index in range(plan.vertical_weight.size):
        unit = unit_names[int(plan.vertical_operator_unit[vertical_index])]
        bounds = plan.vertical_operator_bounds[vertical_index]
        if int(plan.vertical_operator_type[vertical_index]) == VERTICAL_TYPE_CODES["exact"]:
            _validate_vertical_values(
                np.asarray([bounds[0]]),
                np.asarray([plan.vertical_weight[vertical_index]]),
                unit,
                grid.shape[0],
                "ObsOperator restart vertical operator",
            )
        else:
            _validate_vertical_bounds(
                float(bounds[0]),
                float(bounds[1]),
                unit,
                grid.shape[0],
                "ObsOperator restart vertical operator",
            )
    tracer_by_field = {f"{FIELD_PREFIX}{name}": index for index, name in enumerate(tracer_names)}
    remapped_tracers = np.empty_like(plan.field_tracer)
    for field_index, field_name in enumerate(plan.field_names):
        if field_name not in tracer_by_field:
            raise ValueError(
                f"ObsOperator restart requires missing field {field_name!r}"
            )
        remapped_tracers[field_index] = tracer_by_field[field_name]
    plan.field_tracer = remapped_tracers
    if plan.time_operator_bounds_us.size and np.any(
        plan.time_operator_bounds_us[:, 0] < expected_time_us
    ):
        raise ValueError("ObsOperator restart contains sampling times before its restart boundary")
    return plan


def _decode_rows(values: Any, width: int, label: str) -> list[str]:
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
