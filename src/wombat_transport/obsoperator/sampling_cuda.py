"""Resident CUDA ObsOperator sampling companion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from wombat_transport.cuda.modules import load_raw_module
from wombat_transport.cuda.runtime import CudaRuntime
from wombat_transport.grid import TransportGrid
from wombat_transport.obsoperator.state import ObsPlan
from wombat_transport.snapshot import CompletedStepSnapshot


@dataclass
class _DevicePlan:
    identity: int
    arrays: dict[str, Any]
    accumulator: Any


class CudaObsSampler:
    """Own a device plan and accumulate observation fields in float64."""

    def __init__(
        self,
        runtime: CudaRuntime,
        *,
        dtype: np.dtype[Any] | type[Any],
        grid: TransportGrid,
    ) -> None:
        resolved = np.dtype(dtype)
        if resolved == np.dtype(np.float32):
            cuda_type = "float"
        elif resolved == np.dtype(np.float64):
            cuda_type = "double"
        else:
            raise ValueError("CUDA ObsOperator supports float32 and float64 state")
        expression = f"sample_obsoperator<{cuda_type}>"
        module = load_raw_module(
            "obsoperator.cu",
            name_expressions=(expression,),
        )
        self._runtime = runtime
        self._kernel = module.get_function(expression)
        self._grid = grid
        self._area = runtime.to_device(grid.area_m2, dtype=np.float64)
        self._hyai = runtime.to_device(grid.hyai_hpa, dtype=np.float64)
        self._hybi = runtime.to_device(grid.hybi, dtype=np.float64)
        self._error_flag = runtime.zeros((1,), dtype=np.int32)
        self._plan: _DevicePlan | None = None

    def sample(
        self,
        plan: ObsPlan,
        *,
        step_time_us: int,
        snapshot: CompletedStepSnapshot,
    ) -> None:
        device = self._ensure_plan(plan)
        if not plan.field_tracer.size:
            return
        blocks = snapshot.state.block_data[0]
        nblock, nlev, nlat, nlon, width = blocks.shape
        _ = nblock
        wet_ps = self._device_meteorology(
            snapshot.forcing.wet_surface_pressure_hpa
        )
        sphu = self._device_meteorology(
            snapshot.forcing.specific_humidity_kg_kg
        )
        temperature = self._device_meteorology(
            snapshot.forcing.temperature_k
        )
        field_count = int(plan.field_tracer.size)
        threads = 128
        self._kernel(
            ((field_count + threads - 1) // threads,),
            (threads,),
            (
                blocks,
                wet_ps,
                sphu,
                temperature,
                np.int64(sphu.strides[0] // sphu.itemsize),
                np.int64(temperature.strides[0] // temperature.itemsize),
                self._area,
                self._hyai,
                self._hybi,
                np.int64(step_time_us),
                np.int32(plan.first_unexpired),
                device.arrays["field_entry"],
                device.arrays["field_tracer"],
                device.arrays["field_to_accumulator"],
                device.arrays["time_operator_start"],
                device.arrays["time_operator_count"],
                device.arrays["time_operator_bounds_us"],
                device.arrays["time_operator_weight"],
                device.arrays["horizontal_operator_start"],
                device.arrays["horizontal_operator_count"],
                device.arrays["horizontal_operator_bounds"],
                device.arrays["horizontal_weight_type"],
                device.arrays["horizontal_weight"],
                device.arrays["horizontal_normalization"],
                device.arrays["vertical_operator_start"],
                device.arrays["vertical_operator_count"],
                device.arrays["vertical_operator_type"],
                device.arrays["vertical_operator_unit"],
                device.arrays["vertical_operator_bounds"],
                device.arrays["vertical_weight_type"],
                device.arrays["vertical_weight"],
                device.accumulator,
                self._error_flag,
                np.int32(field_count),
                np.int32(nlev),
                np.int32(nlat),
                np.int32(nlon),
                np.int32(width),
            ),
        )

    def sync_to_host(self, plan: ObsPlan) -> None:
        if self._plan is None or self._plan.identity != id(plan):
            return
        plan.accumulator[:] = self._runtime.to_host(self._plan.accumulator)
        if int(self._runtime.to_host(self._error_flag)[0]):
            raise ValueError("vertical altitude exceeds the modeled column")

    def invalidate(self) -> None:
        self._plan = None

    def _device_meteorology(self, values: Any) -> Any:
        step_values = values[0]
        if self._runtime.is_device_array(step_values):
            if step_values.dtype != np.dtype(np.float64):
                raise ValueError(
                    "resident CUDA ObsOperator meteorology must be float64"
                )
            return step_values
        return self._runtime.to_device(step_values, dtype=np.float64)

    def _ensure_plan(self, plan: ObsPlan) -> _DevicePlan:
        if self._plan is not None and self._plan.identity == id(plan):
            return self._plan
        if np.unique(plan.field_to_accumulator).size != plan.field_to_accumulator.size:
            raise ValueError(
                "CUDA ObsOperator requires one accumulator per observation field"
            )
        field_entry = np.empty(plan.field_tracer.size, dtype=np.int32)
        for entry in range(plan.entry_count):
            start = int(plan.entry_field_start[entry])
            stop = start + int(plan.entry_field_count[entry])
            field_entry[start:stop] = entry
        names = (
            "field_tracer",
            "field_to_accumulator",
            "time_operator_start",
            "time_operator_count",
            "time_operator_bounds_us",
            "time_operator_weight",
            "horizontal_operator_start",
            "horizontal_operator_count",
            "horizontal_operator_bounds",
            "horizontal_weight_type",
            "horizontal_weight",
            "horizontal_normalization",
            "vertical_operator_start",
            "vertical_operator_count",
            "vertical_operator_type",
            "vertical_operator_unit",
            "vertical_operator_bounds",
            "vertical_weight_type",
            "vertical_weight",
        )
        arrays = {
            name: self._runtime.to_device(getattr(plan, name))
            for name in names
        }
        arrays["field_entry"] = self._runtime.to_device(field_entry)
        self._plan = _DevicePlan(
            identity=id(plan),
            arrays=arrays,
            accumulator=self._runtime.to_device(
                plan.accumulator,
                dtype=np.float64,
            ),
        )
        return self._plan
