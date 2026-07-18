from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np

from wombat_transport.obsoperator.state import (
    MAX_FIELD_NAME_LENGTH,
    MAX_ID_LENGTH,
    _ObsOperatorArrayState,
)
from wombat_transport.obsoperator.utils import _nul_padded_matrix

SCIENCE_ENTRY_CHUNK = 256
SCIENCE_FIELD_CHUNK = 64
SCIENCE_SAMPLE_CHUNK = 16_384
SCIENCE_STAGE_ENTRIES = SCIENCE_ENTRY_CHUNK
SCIENCE_STAGE_SAMPLES = SCIENCE_SAMPLE_CHUNK

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
        field_indices_by_name = self._field_indices.copy()
        field_names = self._field_names.copy()
        for state, entry_indices in self._pending_array_batches:
            for entry_index_value in entry_indices:
                entry_index = int(entry_index_value)
                field_start = int(state.prepared.entry_field_start[entry_index])
                entry_sample_count = int(state.prepared.entry_field_count[entry_index])
                field_end = field_start + entry_sample_count
                entry_slice = slice(sample_offset, sample_offset + entry_sample_count)
                for field_offset, name in enumerate(state.field_names[entry_index]):
                    if name not in field_indices_by_name:
                        field_indices_by_name[name] = len(field_names) + 1
                        field_names.append(name)
                    field_indices[sample_offset + field_offset] = field_indices_by_name[name]
                id_indices[entry_slice] = self._entry_index + array_entry_offset + 1
                samples[entry_slice] = state.field_accumulator[field_start:field_end]
                array_ids.append(state.ids[entry_index])
                array_entry_offset += 1
                sample_offset += entry_sample_count

        self._ensure_created()
        assert self._dataset is not None
        if len(field_names) > previous_field_count:
            new_field_names = field_names[previous_field_count:]
            self._dataset.variables["field"][previous_field_count : len(field_names), :] = (
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
        self._field_indices = field_indices_by_name
        self._field_names = field_names
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
