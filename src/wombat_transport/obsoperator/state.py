from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wombat_transport.transport.numba_control import numba_available_and_enabled

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised without the optional accelerator.
    njit = None


MAX_ID_LENGTH = 255
MAX_FIELD_NAME_LENGTH = 64
FIELD_PREFIX = "SpeciesConcVV_"
FIELD_ALL = "SpeciesConcVV_?ALL?"
FIELD_ADVECTED = "SpeciesConcVV_?ADV?"

HORIZONTAL_WEIGHTING_CODES = {
    "equal": 0,
    "normalized": 1,
    "area": 2,
    "normalized_area": 3,
    "exact": 4,
}
VERTICAL_TYPE_CODES = {"range": 0, "exact": 1}
VERTICAL_UNIT_CODES = {"pressure": 0, "altitude": 1, "pressure_level": 2}
VERTICAL_WEIGHTING_CODES = {
    "equal": 0,
    "normalized": 1,
    "pressure": 2,
    "normalized_pressure": 3,
    "exact": 4,
}

_HORIZONTAL_NORMALIZED = HORIZONTAL_WEIGHTING_CODES["normalized"]
_HORIZONTAL_AREA = HORIZONTAL_WEIGHTING_CODES["area"]
_HORIZONTAL_NORMALIZED_AREA = HORIZONTAL_WEIGHTING_CODES["normalized_area"]
_HORIZONTAL_EXACT = HORIZONTAL_WEIGHTING_CODES["exact"]
_VERTICAL_EXACT = VERTICAL_TYPE_CODES["exact"]
_VERTICAL_PRESSURE = VERTICAL_UNIT_CODES["pressure"]
_VERTICAL_PRESSURE_LEVEL = VERTICAL_UNIT_CODES["pressure_level"]
_VERTICAL_NORMALIZED = VERTICAL_WEIGHTING_CODES["normalized"]
_VERTICAL_PRESSURE_WEIGHT = VERTICAL_WEIGHTING_CODES["pressure"]
_VERTICAL_NORMALIZED_PRESSURE = VERTICAL_WEIGHTING_CODES["normalized_pressure"]


@dataclass
class ObsPlan:
    ids: tuple[str, ...]
    field_names: tuple[str, ...]
    accumulator: np.ndarray
    entry_field_start: np.ndarray
    entry_field_count: np.ndarray
    field_tracer: np.ndarray
    field_to_accumulator: np.ndarray
    time_operator_start: np.ndarray
    time_operator_count: np.ndarray
    time_operator_bounds_us: np.ndarray
    time_operator_weight: np.ndarray
    horizontal_operator_start: np.ndarray
    horizontal_operator_count: np.ndarray
    horizontal_operator_bounds: np.ndarray
    horizontal_weight_type: np.ndarray
    horizontal_weight: np.ndarray
    horizontal_normalization: np.ndarray
    vertical_operator_start: np.ndarray
    vertical_operator_count: np.ndarray
    vertical_operator_type: np.ndarray
    vertical_operator_unit: np.ndarray
    vertical_operator_bounds: np.ndarray
    vertical_weight_type: np.ndarray
    vertical_weight: np.ndarray
    entry_end_us: np.ndarray
    first_unexpired: int = 0

    @property
    def entry_count(self) -> int:
        return len(self.ids)

    @property
    def accumulator_count(self) -> int:
        return int(self.accumulator.size)

    def field_slice(self, entry_index: int) -> slice:
        start = int(self.entry_field_start[entry_index])
        return slice(start, start + int(self.entry_field_count[entry_index]))

    def entry_field_names(self, entry_index: int) -> tuple[str, ...]:
        field_slice = self.field_slice(entry_index)
        return self.field_names[field_slice]

    def validate(self) -> None:
        entry_count = self.entry_count
        if len(set(self.ids)) != entry_count or any(not value for value in self.ids):
            raise ValueError("ObsPlan ids must be nonempty and unique")
        if self.entry_end_us.shape != (entry_count,):
            raise ValueError("ObsPlan entry_end_us has an invalid shape")
        if entry_count > 1 and np.any(np.diff(self.entry_end_us) < 0):
            raise ValueError("ObsPlan entries must be sorted by completion time")
        _validate_ragged(
            self.entry_field_start,
            self.entry_field_count,
            self.field_tracer.size,
            "field",
            require_nonempty=True,
        )
        if len(self.field_names) != self.field_tracer.size:
            raise ValueError("ObsPlan field metadata has an invalid length")
        if self.field_to_accumulator.shape != self.field_tracer.shape:
            raise ValueError("ObsPlan field mapping has an invalid shape")
        if np.any(self.field_tracer < 0):
            raise ValueError("ObsPlan tracer indices must be nonnegative")
        if np.any(self.field_to_accumulator < 0) or np.any(
            self.field_to_accumulator >= self.accumulator.size
        ):
            raise ValueError("ObsPlan accumulator mapping is out of range")
        owner = np.full(self.accumulator.size, -1, dtype=np.int64)
        for field_index, accumulator_index in enumerate(self.field_to_accumulator):
            accumulator_index = int(accumulator_index)
            tracer = int(self.field_tracer[field_index])
            if owner[accumulator_index] not in (-1, tracer):
                raise ValueError("ObsPlan accumulator slots cannot have multiple tracer owners")
            owner[accumulator_index] = tracer
        if np.any(owner < 0):
            raise ValueError("ObsPlan contains an unreferenced accumulator slot")
        if not np.all(np.isfinite(self.accumulator)):
            raise ValueError("ObsPlan accumulator values must be finite")
        for entry_index in range(entry_count):
            names = self.entry_field_names(entry_index)
            if len(set(names)) != len(names):
                raise ValueError(f"ObsPlan entry {self.ids[entry_index]!r} contains duplicate fields")
        _validate_ragged(
            self.time_operator_start,
            self.time_operator_count,
            self.time_operator_weight.size,
            "time operator",
            require_nonempty=True,
        )
        if self.time_operator_bounds_us.shape != (self.time_operator_weight.size, 2):
            raise ValueError("ObsPlan time bounds have an invalid shape")
        if np.any(self.time_operator_bounds_us[:, 0] >= self.time_operator_bounds_us[:, 1]):
            raise ValueError("ObsPlan time bounds must be nonempty half-open intervals")
        if not np.all(np.isfinite(self.time_operator_weight)):
            raise ValueError("ObsPlan time weights must be finite")
        _validate_ragged(
            self.horizontal_operator_start,
            self.horizontal_operator_count,
            self.horizontal_weight.size,
            "horizontal operator",
            require_nonempty=True,
        )
        if self.horizontal_operator_bounds.shape != (self.horizontal_weight.size, 2, 2):
            raise ValueError("ObsPlan horizontal bounds have an invalid shape")
        if (
            self.horizontal_weight_type.shape != self.horizontal_weight.shape
            or self.horizontal_normalization.shape != self.horizontal_weight.shape
        ):
            raise ValueError("ObsPlan horizontal weight arrays have inconsistent shapes")
        if np.any(self.horizontal_weight_type < 0) or np.any(
            self.horizontal_weight_type > max(HORIZONTAL_WEIGHTING_CODES.values())
        ):
            raise ValueError("ObsPlan contains an invalid horizontal weight type")
        if not np.all(np.isfinite(self.horizontal_weight)):
            raise ValueError("ObsPlan horizontal weights must be finite")
        if not np.all(np.isfinite(self.horizontal_normalization)) or np.any(
            self.horizontal_normalization <= 0.0
        ):
            raise ValueError("ObsPlan horizontal normalizations must be finite and positive")
        if np.any(self.horizontal_operator_bounds[:, :, 0] >= self.horizontal_operator_bounds[:, :, 1]):
            raise ValueError("ObsPlan horizontal bounds must be nonempty half-open boxes")
        _validate_ragged(
            self.vertical_operator_start,
            self.vertical_operator_count,
            self.vertical_weight.size,
            "vertical operator",
            require_nonempty=True,
        )
        vertical_count = self.vertical_weight.size
        if self.vertical_operator_bounds.shape != (vertical_count, 2):
            raise ValueError("ObsPlan vertical bounds have an invalid shape")
        for array in (
            self.vertical_operator_type,
            self.vertical_operator_unit,
            self.vertical_weight_type,
        ):
            if array.shape != (vertical_count,):
                raise ValueError("ObsPlan vertical arrays have inconsistent shapes")
        if not np.all(np.isfinite(self.vertical_operator_bounds)) or not np.all(
            np.isfinite(self.vertical_weight)
        ):
            raise ValueError("ObsPlan vertical bounds and weights must be finite")
        if np.any(self.vertical_operator_type < 0) or np.any(
            self.vertical_operator_type > max(VERTICAL_TYPE_CODES.values())
        ):
            raise ValueError("ObsPlan contains an invalid vertical operator type")
        if np.any(self.vertical_operator_unit < 0) or np.any(
            self.vertical_operator_unit > max(VERTICAL_UNIT_CODES.values())
        ):
            raise ValueError("ObsPlan contains an invalid vertical unit")
        if np.any(self.vertical_weight_type < 0) or np.any(
            self.vertical_weight_type > max(VERTICAL_WEIGHTING_CODES.values())
        ):
            raise ValueError("ObsPlan contains an invalid vertical weight type")
        for entry_index in range(entry_count):
            start = int(self.time_operator_start[entry_index])
            end = start + int(self.time_operator_count[entry_index])
            if int(np.max(self.time_operator_bounds_us[start:end, 1])) != int(
                self.entry_end_us[entry_index]
            ):
                raise ValueError("ObsPlan completion times do not match time operators")
        if not 0 <= self.first_unexpired <= entry_count:
            raise ValueError("ObsPlan first-unexpired index is out of range")


@dataclass(frozen=True)
class CompletedObsBatch:
    ids: tuple[str, ...]
    field_names: tuple[str, ...]
    entry_field_start: np.ndarray
    entry_field_count: np.ndarray
    samples: np.ndarray

    @property
    def entry_count(self) -> int:
        return len(self.ids)


def empty_obs_plan() -> ObsPlan:
    return ObsPlan(
        ids=(),
        field_names=(),
        accumulator=np.empty(0, dtype=np.float64),
        entry_field_start=np.empty(0, dtype=np.int64),
        entry_field_count=np.empty(0, dtype=np.int32),
        field_tracer=np.empty(0, dtype=np.int64),
        field_to_accumulator=np.empty(0, dtype=np.int64),
        time_operator_start=np.empty(0, dtype=np.int64),
        time_operator_count=np.empty(0, dtype=np.int32),
        time_operator_bounds_us=np.empty((0, 2), dtype=np.int64),
        time_operator_weight=np.empty(0, dtype=np.float64),
        horizontal_operator_start=np.empty(0, dtype=np.int64),
        horizontal_operator_count=np.empty(0, dtype=np.int32),
        horizontal_operator_bounds=np.empty((0, 2, 2), dtype=np.int32),
        horizontal_weight_type=np.empty(0, dtype=np.int8),
        horizontal_weight=np.empty(0, dtype=np.float64),
        horizontal_normalization=np.empty(0, dtype=np.float64),
        vertical_operator_start=np.empty(0, dtype=np.int64),
        vertical_operator_count=np.empty(0, dtype=np.int32),
        vertical_operator_type=np.empty(0, dtype=np.int8),
        vertical_operator_unit=np.empty(0, dtype=np.int8),
        vertical_operator_bounds=np.empty((0, 2), dtype=np.float64),
        vertical_weight_type=np.empty(0, dtype=np.int8),
        vertical_weight=np.empty(0, dtype=np.float64),
        entry_end_us=np.empty(0, dtype=np.int64),
    )


def completed_prefix(plan: ObsPlan, boundary_us: int) -> int:
    return int(np.searchsorted(plan.entry_end_us, boundary_us, side="right"))


def completed_batch(plan: ObsPlan, entry_count: int) -> CompletedObsBatch:
    return _completed_batch_range(plan, 0, entry_count)


def _completed_batch_range(
    plan: ObsPlan,
    first_entry: int,
    stop_entry: int,
) -> CompletedObsBatch:
    entry_count = stop_entry - first_entry
    if entry_count == 0:
        return CompletedObsBatch(
            ids=(),
            field_names=(),
            entry_field_start=np.empty(0, dtype=np.int64),
            entry_field_count=np.empty(0, dtype=np.int32),
            samples=np.empty(0, dtype=np.float64),
        )
    counts = plan.entry_field_count[first_entry:stop_entry].copy()
    starts = np.empty(entry_count, dtype=np.int64)
    total = int(np.sum(counts, dtype=np.int64))
    names: list[str] = []
    values = np.empty(total, dtype=np.float64)
    offset = 0
    for output_index, plan_index in enumerate(range(first_entry, stop_entry)):
        starts[output_index] = offset
        source = plan.field_slice(plan_index)
        count = int(counts[output_index])
        names.extend(plan.field_names[source])
        mappings = plan.field_to_accumulator[source]
        values[offset : offset + count] = plan.accumulator[mappings]
        offset += count
    return CompletedObsBatch(
        plan.ids[first_entry:stop_entry],
        tuple(names),
        starts,
        counts,
        values,
    )


def merge_obs_plans(left: ObsPlan, right: ObsPlan) -> ObsPlan:
    duplicates = set(left.ids).intersection(right.ids)
    if duplicates:
        duplicate = next(value for value in right.ids if value in duplicates)
        raise ValueError(f"duplicate active ObsOperator id {duplicate!r}")
    if not left.entry_count:
        right.validate()
        return right
    if not right.entry_count:
        left.validate()
        return left
    source, indices = _merge_order(
        left.entry_end_us,
        right.entry_end_us,
    )
    merged = _copy_ordered_plans(left, right, source, indices, boundary_us=-1)
    merged.validate()
    return merged


def compact_obs_plan(plan: ObsPlan, boundary_us: int) -> ObsPlan:
    keep = np.flatnonzero(plan.entry_end_us > boundary_us).astype(np.int64)
    if keep.size == plan.entry_count and not np.any(plan.time_operator_bounds_us[:, 0] < boundary_us):
        plan.first_unexpired = 0
        return plan
    source = np.zeros(keep.size, dtype=np.int8)
    compacted = _copy_ordered_plans(plan, empty_obs_plan(), source, keep, boundary_us=boundary_us)
    compacted.validate()
    return compacted


def _copy_ordered_plans(
    left: ObsPlan,
    right: ObsPlan,
    source: np.ndarray,
    indices: np.ndarray,
    *,
    boundary_us: int,
) -> ObsPlan:
    entry_count = source.size
    count_kernel = _select_structural_kernel(_count_ordered_kernel, _count_ordered_numba)
    field_total, time_total, horizontal_total, vertical_total, accumulator_total = count_kernel(
        source,
        indices,
        boundary_us,
        left.entry_field_count,
        left.time_operator_start,
        left.time_operator_count,
        left.time_operator_bounds_us,
        left.horizontal_operator_count,
        left.vertical_operator_count,
        right.entry_field_count,
        right.time_operator_start,
        right.time_operator_count,
        right.time_operator_bounds_us,
        right.horizontal_operator_count,
        right.vertical_operator_count,
    )
    ids: list[str] = []
    field_names: list[str] = []
    for output_index in range(entry_count):
        plan = left if source[output_index] == 0 else right
        entry = int(indices[output_index])
        ids.append(plan.ids[entry])
        field_names.extend(plan.field_names[plan.field_slice(entry)])
    arrays = _allocate_numeric(entry_count, field_total, time_total, horizontal_total, vertical_total, accumulator_total)
    _fill_ordered_numeric(left, right, source, indices, boundary_us, arrays)
    result = ObsPlan(ids=tuple(ids), field_names=tuple(field_names), **arrays)
    return result


def _allocate_numeric(
    entry_count: int,
    field_total: int,
    time_total: int,
    horizontal_total: int,
    vertical_total: int,
    accumulator_total: int,
) -> dict[str, np.ndarray | int]:
    return {
        "accumulator": np.empty(accumulator_total, dtype=np.float64),
        "entry_field_start": np.empty(entry_count, dtype=np.int64),
        "entry_field_count": np.empty(entry_count, dtype=np.int32),
        "field_tracer": np.empty(field_total, dtype=np.int64),
        "field_to_accumulator": np.empty(field_total, dtype=np.int64),
        "time_operator_start": np.empty(entry_count, dtype=np.int64),
        "time_operator_count": np.empty(entry_count, dtype=np.int32),
        "time_operator_bounds_us": np.empty((time_total, 2), dtype=np.int64),
        "time_operator_weight": np.empty(time_total, dtype=np.float64),
        "horizontal_operator_start": np.empty(entry_count, dtype=np.int64),
        "horizontal_operator_count": np.empty(entry_count, dtype=np.int32),
        "horizontal_operator_bounds": np.empty((horizontal_total, 2, 2), dtype=np.int32),
        "horizontal_weight_type": np.empty(horizontal_total, dtype=np.int8),
        "horizontal_weight": np.empty(horizontal_total, dtype=np.float64),
        "horizontal_normalization": np.empty(horizontal_total, dtype=np.float64),
        "vertical_operator_start": np.empty(entry_count, dtype=np.int64),
        "vertical_operator_count": np.empty(entry_count, dtype=np.int32),
        "vertical_operator_type": np.empty(vertical_total, dtype=np.int8),
        "vertical_operator_unit": np.empty(vertical_total, dtype=np.int8),
        "vertical_operator_bounds": np.empty((vertical_total, 2), dtype=np.float64),
        "vertical_weight_type": np.empty(vertical_total, dtype=np.int8),
        "vertical_weight": np.empty(vertical_total, dtype=np.float64),
        "entry_end_us": np.empty(entry_count, dtype=np.int64),
        "first_unexpired": 0,
    }


def _count_ordered_kernel(
    source: np.ndarray,
    indices: np.ndarray,
    boundary_us: int,
    left_field_count: np.ndarray,
    left_time_start: np.ndarray,
    left_time_count: np.ndarray,
    left_time_bounds: np.ndarray,
    left_horizontal_count: np.ndarray,
    left_vertical_count: np.ndarray,
    right_field_count: np.ndarray,
    right_time_start: np.ndarray,
    right_time_count: np.ndarray,
    right_time_bounds: np.ndarray,
    right_horizontal_count: np.ndarray,
    right_vertical_count: np.ndarray,
) -> tuple[int, int, int, int, int]:
    field_total = time_total = horizontal_total = vertical_total = accumulator_total = 0
    for output_index in range(source.size):
        entry = indices[output_index]
        if source[output_index] == 0:
            field_count = left_field_count[entry]
            time_start = left_time_start[entry]
            time_count = left_time_count[entry]
            time_bounds = left_time_bounds
            horizontal_count = left_horizontal_count[entry]
            vertical_count = left_vertical_count[entry]
        else:
            field_count = right_field_count[entry]
            time_start = right_time_start[entry]
            time_count = right_time_count[entry]
            time_bounds = right_time_bounds
            horizontal_count = right_horizontal_count[entry]
            vertical_count = right_vertical_count[entry]
        field_total += field_count
        accumulator_total += field_count
        horizontal_total += horizontal_count
        vertical_total += vertical_count
        for time_index in range(time_start, time_start + time_count):
            if time_bounds[time_index, 1] > boundary_us:
                time_total += 1
    return field_total, time_total, horizontal_total, vertical_total, accumulator_total


def _copy_fields_kernel(
    source: np.ndarray,
    indices: np.ndarray,
    left_start: np.ndarray,
    left_count: np.ndarray,
    left_tracer: np.ndarray,
    left_mapping: np.ndarray,
    left_accumulator: np.ndarray,
    right_start: np.ndarray,
    right_count: np.ndarray,
    right_tracer: np.ndarray,
    right_mapping: np.ndarray,
    right_accumulator: np.ndarray,
    output_start: np.ndarray,
    output_count: np.ndarray,
    output_tracer: np.ndarray,
    output_mapping: np.ndarray,
    output_accumulator: np.ndarray,
) -> None:
    field_offset = 0
    accumulator_offset = 0
    for output_index in range(source.size):
        entry = indices[output_index]
        if source[output_index] == 0:
            start = left_start[entry]
            count = left_count[entry]
            tracer = left_tracer
            mapping = left_mapping
            accumulator = left_accumulator
        else:
            start = right_start[entry]
            count = right_count[entry]
            tracer = right_tracer
            mapping = right_mapping
            accumulator = right_accumulator
        output_start[output_index] = field_offset
        output_count[output_index] = count
        for offset in range(count):
            output_tracer[field_offset] = tracer[start + offset]
            output_mapping[field_offset] = accumulator_offset
            output_accumulator[accumulator_offset] = accumulator[mapping[start + offset]]
            field_offset += 1
            accumulator_offset += 1


def _copy_time_kernel(
    source: np.ndarray,
    indices: np.ndarray,
    boundary_us: int,
    left_start: np.ndarray,
    left_count: np.ndarray,
    left_bounds: np.ndarray,
    left_weight: np.ndarray,
    right_start: np.ndarray,
    right_count: np.ndarray,
    right_bounds: np.ndarray,
    right_weight: np.ndarray,
    output_start: np.ndarray,
    output_count: np.ndarray,
    output_bounds: np.ndarray,
    output_weight: np.ndarray,
) -> None:
    output_offset = 0
    for output_index in range(source.size):
        entry = indices[output_index]
        if source[output_index] == 0:
            start = left_start[entry]
            count = left_count[entry]
            bounds = left_bounds
            weight = left_weight
        else:
            start = right_start[entry]
            count = right_count[entry]
            bounds = right_bounds
            weight = right_weight
        output_start[output_index] = output_offset
        output_count[output_index] = 0
        for source_index in range(start, start + count):
            end = bounds[source_index, 1]
            if end <= boundary_us:
                continue
            interval_start = bounds[source_index, 0]
            output_bounds[output_offset, 0] = max(interval_start, boundary_us)
            output_bounds[output_offset, 1] = end
            output_weight[output_offset] = weight[source_index]
            output_count[output_index] += 1
            output_offset += 1


def _copy_horizontal_kernel(
    source: np.ndarray,
    indices: np.ndarray,
    left_start: np.ndarray,
    left_count: np.ndarray,
    left_bounds: np.ndarray,
    left_type: np.ndarray,
    left_weight: np.ndarray,
    left_normalization: np.ndarray,
    right_start: np.ndarray,
    right_count: np.ndarray,
    right_bounds: np.ndarray,
    right_type: np.ndarray,
    right_weight: np.ndarray,
    right_normalization: np.ndarray,
    output_start: np.ndarray,
    output_count: np.ndarray,
    output_bounds: np.ndarray,
    output_type: np.ndarray,
    output_weight: np.ndarray,
    output_normalization: np.ndarray,
) -> None:
    output_offset = 0
    for output_index in range(source.size):
        entry = indices[output_index]
        if source[output_index] == 0:
            start = left_start[entry]
            count = left_count[entry]
            bounds = left_bounds
            weighting = left_type
            weight = left_weight
            normalization = left_normalization
        else:
            start = right_start[entry]
            count = right_count[entry]
            bounds = right_bounds
            weighting = right_type
            weight = right_weight
            normalization = right_normalization
        output_start[output_index] = output_offset
        output_count[output_index] = count
        for source_index in range(start, start + count):
            for dimension in range(2):
                output_bounds[output_offset, dimension, 0] = bounds[source_index, dimension, 0]
                output_bounds[output_offset, dimension, 1] = bounds[source_index, dimension, 1]
            output_type[output_offset] = weighting[source_index]
            output_weight[output_offset] = weight[source_index]
            output_normalization[output_offset] = normalization[source_index]
            output_offset += 1


def _copy_vertical_kernel(
    source: np.ndarray,
    indices: np.ndarray,
    left_start: np.ndarray,
    left_count: np.ndarray,
    left_type: np.ndarray,
    left_unit: np.ndarray,
    left_bounds: np.ndarray,
    left_weight_type: np.ndarray,
    left_weight: np.ndarray,
    right_start: np.ndarray,
    right_count: np.ndarray,
    right_type: np.ndarray,
    right_unit: np.ndarray,
    right_bounds: np.ndarray,
    right_weight_type: np.ndarray,
    right_weight: np.ndarray,
    output_start: np.ndarray,
    output_count: np.ndarray,
    output_type: np.ndarray,
    output_unit: np.ndarray,
    output_bounds: np.ndarray,
    output_weight_type: np.ndarray,
    output_weight: np.ndarray,
    output_end_us: np.ndarray,
    left_end_us: np.ndarray,
    right_end_us: np.ndarray,
) -> None:
    output_offset = 0
    for output_index in range(source.size):
        entry = indices[output_index]
        if source[output_index] == 0:
            start = left_start[entry]
            count = left_count[entry]
            operator_type = left_type
            unit = left_unit
            bounds = left_bounds
            weight_type = left_weight_type
            weight = left_weight
            output_end_us[output_index] = left_end_us[entry]
        else:
            start = right_start[entry]
            count = right_count[entry]
            operator_type = right_type
            unit = right_unit
            bounds = right_bounds
            weight_type = right_weight_type
            weight = right_weight
            output_end_us[output_index] = right_end_us[entry]
        output_start[output_index] = output_offset
        output_count[output_index] = count
        for source_index in range(start, start + count):
            output_type[output_offset] = operator_type[source_index]
            output_unit[output_offset] = unit[source_index]
            output_bounds[output_offset, 0] = bounds[source_index, 0]
            output_bounds[output_offset, 1] = bounds[source_index, 1]
            output_weight_type[output_offset] = weight_type[source_index]
            output_weight[output_offset] = weight[source_index]
            output_offset += 1


if njit is not None:
    _count_ordered_numba = njit(cache=True, nogil=True)(_count_ordered_kernel)
    _copy_fields_numba = njit(cache=True, nogil=True)(_copy_fields_kernel)
    _copy_time_numba = njit(cache=True, nogil=True)(_copy_time_kernel)
    _copy_horizontal_numba = njit(cache=True, nogil=True)(_copy_horizontal_kernel)
    _copy_vertical_numba = njit(cache=True, nogil=True)(_copy_vertical_kernel)
else:  # pragma: no cover - exercised without the optional accelerator.
    _count_ordered_numba = None
    _copy_fields_numba = None
    _copy_time_numba = None
    _copy_horizontal_numba = None
    _copy_vertical_numba = None


def _select_structural_kernel(reference, compiled):
    if numba_available_and_enabled(available=compiled is not None):
        return compiled
    return reference


def _fill_ordered_numeric(
    left: ObsPlan,
    right: ObsPlan,
    source: np.ndarray,
    indices: np.ndarray,
    boundary_us: int,
    arrays: dict[str, np.ndarray | int],
) -> None:
    field_kernel = _select_structural_kernel(_copy_fields_kernel, _copy_fields_numba)
    field_kernel(
        source, indices,
        left.entry_field_start, left.entry_field_count, left.field_tracer,
        left.field_to_accumulator, left.accumulator,
        right.entry_field_start, right.entry_field_count, right.field_tracer,
        right.field_to_accumulator, right.accumulator,
        arrays["entry_field_start"], arrays["entry_field_count"], arrays["field_tracer"],
        arrays["field_to_accumulator"], arrays["accumulator"],
    )
    time_kernel = _select_structural_kernel(_copy_time_kernel, _copy_time_numba)
    time_kernel(
        source, indices, boundary_us,
        left.time_operator_start, left.time_operator_count, left.time_operator_bounds_us,
        left.time_operator_weight,
        right.time_operator_start, right.time_operator_count, right.time_operator_bounds_us,
        right.time_operator_weight,
        arrays["time_operator_start"], arrays["time_operator_count"],
        arrays["time_operator_bounds_us"], arrays["time_operator_weight"],
    )
    horizontal_kernel = _select_structural_kernel(
        _copy_horizontal_kernel, _copy_horizontal_numba
    )
    horizontal_kernel(
        source, indices,
        left.horizontal_operator_start, left.horizontal_operator_count,
        left.horizontal_operator_bounds, left.horizontal_weight_type, left.horizontal_weight,
        left.horizontal_normalization,
        right.horizontal_operator_start, right.horizontal_operator_count,
        right.horizontal_operator_bounds, right.horizontal_weight_type, right.horizontal_weight,
        right.horizontal_normalization,
        arrays["horizontal_operator_start"], arrays["horizontal_operator_count"],
        arrays["horizontal_operator_bounds"], arrays["horizontal_weight_type"],
        arrays["horizontal_weight"],
        arrays["horizontal_normalization"],
    )
    vertical_kernel = _select_structural_kernel(_copy_vertical_kernel, _copy_vertical_numba)
    vertical_kernel(
        source, indices,
        left.vertical_operator_start, left.vertical_operator_count,
        left.vertical_operator_type, left.vertical_operator_unit, left.vertical_operator_bounds,
        left.vertical_weight_type, left.vertical_weight,
        right.vertical_operator_start, right.vertical_operator_count,
        right.vertical_operator_type, right.vertical_operator_unit, right.vertical_operator_bounds,
        right.vertical_weight_type, right.vertical_weight,
        arrays["vertical_operator_start"], arrays["vertical_operator_count"],
        arrays["vertical_operator_type"], arrays["vertical_operator_unit"],
        arrays["vertical_operator_bounds"], arrays["vertical_weight_type"],
        arrays["vertical_weight"], arrays["entry_end_us"], left.entry_end_us,
        right.entry_end_us,
    )


def _merge_order_kernel(left_end: np.ndarray, right_end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = left_end.size + right_end.size
    source = np.empty(total, dtype=np.int8)
    indices = np.empty(total, dtype=np.int64)
    left_index = right_index = 0
    for output_index in range(total):
        take_left = right_index >= right_end.size or (
            left_index < left_end.size and left_end[left_index] <= right_end[right_index]
        )
        if take_left:
            source[output_index] = 0
            indices[output_index] = left_index
            left_index += 1
        else:
            source[output_index] = 1
            indices[output_index] = right_index
            right_index += 1
    return source, indices


_merge_order_numba = njit(cache=True, nogil=True)(_merge_order_kernel) if njit is not None else None


def _merge_order(left_end: np.ndarray, right_end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if _merge_order_numba is not None:
        return _merge_order_numba(left_end, right_end)
    return _merge_order_kernel(left_end, right_end)


def _slice(starts: np.ndarray, counts: np.ndarray, index: int) -> slice:
    start = int(starts[index])
    return slice(start, start + int(counts[index]))


def _validate_ragged(
    starts: np.ndarray,
    counts: np.ndarray,
    total: int,
    label: str,
    *,
    require_nonempty: bool,
) -> None:
    if starts.shape != counts.shape:
        raise ValueError(f"ObsPlan {label} offsets have inconsistent shapes")
    offset = 0
    for start, count in zip(starts, counts, strict=True):
        minimum = 1 if require_nonempty else 0
        if int(start) != offset or int(count) < minimum:
            raise ValueError(f"ObsPlan has invalid contiguous {label} offsets")
        offset += int(count)
    if offset != total:
        raise ValueError(f"ObsPlan has an inconsistent {label} payload length")
