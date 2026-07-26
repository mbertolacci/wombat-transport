"""CUDA application of a prepared VDIFF/PBL tracer plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from wombat_transport.cuda.modules import load_raw_module
from wombat_transport.cuda.runtime import CudaRuntime
from wombat_transport.transport.pbl._plan import VdiffPlan


@dataclass(frozen=True)
class CudaVdiffPlan:
    cch: Any
    zeh: Any
    termh: Any
    cgs: Any
    kvh: Any
    potbar: Any
    rpdel: Any
    rrho: Any
    tmp1: Any
    dry_mass: Any
    area_m2: Any
    dt_s: float
    start_level: Any
    specific_humidity_after: Any


@dataclass(frozen=True)
class CudaVdiffResult:
    tracer_conc: Any
    specific_humidity_kg_kg: Any
    negative_count_before_clip: Any


class CudaVdiffExecutor:
    """Own the strict CUDA VDIFF kernel and its reusable tracer workspace."""

    def __init__(self, runtime: CudaRuntime, *, dtype: np.dtype[Any] | type[Any]) -> None:
        resolved_dtype = np.dtype(dtype)
        if resolved_dtype == np.dtype(np.float32):
            cuda_type = "float"
        elif resolved_dtype == np.dtype(np.float64):
            cuda_type = "double"
        else:
            raise ValueError("CUDA VDIFF supports only float32 and float64")
        expression = f"apply_vdiff<{cuda_type}>"
        module = load_raw_module(
            "vdiff.cu",
            name_expressions=(expression,),
        )
        self._runtime = runtime
        self._cupy = runtime.array_module
        self._dtype = resolved_dtype
        self._kernel = module.get_function(expression)
        self._qmx: Any | None = None
        self._negative_count = runtime.zeros((), dtype=np.int32)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    def upload_plan(self, plan: VdiffPlan) -> CudaVdiffPlan:
        """Copy one shared host meteorology plan to the device."""

        return CudaVdiffPlan(
            cch=self._runtime.to_device(plan.cch, dtype=self._dtype),
            zeh=self._runtime.to_device(plan.zeh, dtype=self._dtype),
            termh=self._runtime.to_device(plan.termh, dtype=self._dtype),
            cgs=self._runtime.to_device(plan.cgs, dtype=self._dtype),
            kvh=self._runtime.to_device(plan.kvh, dtype=self._dtype),
            potbar=self._runtime.to_device(plan.potbar, dtype=self._dtype),
            rpdel=self._runtime.to_device(plan.rpdel, dtype=self._dtype),
            rrho=self._runtime.to_device(plan.rrho, dtype=self._dtype),
            tmp1=self._runtime.to_device(plan.tmp1, dtype=self._dtype),
            dry_mass=self._runtime.to_device(plan.dry_mass, dtype=self._dtype),
            area_m2=self._runtime.to_device(plan.area_m2, dtype=self._dtype),
            dt_s=float(plan.dt_s),
            start_level=self._runtime.to_device(
                np.array([int(plan.start_level)], dtype=np.int32)
            ),
            specific_humidity_after=self._runtime.to_device(
                plan.specific_humidity_after,
                dtype=self._dtype,
            ),
        )

    def apply(
        self,
        tracer_conc: Any,
        plan: CudaVdiffPlan,
        surface_flux_kg_m2_s: Any,
        *,
        has_flux: bool,
        output: Any | None = None,
    ) -> CudaVdiffResult:
        """Apply a prepared VDIFF plan without transferring resident arrays."""

        self._validate_inputs(tracer_conc, plan, surface_flux_kg_m2_s)
        nlev, nlat, nlon, nlane = tracer_conc.shape
        if nlev < 2:
            raise ValueError("CUDA VDIFF requires at least two vertical levels")
        output = self._resolve_output(tracer_conc, output)

        if self._qmx is None or self._qmx.shape != tracer_conc.shape:
            self._qmx = self._runtime.empty(tracer_conc.shape, dtype=self._dtype)
        self._negative_count.fill(0)
        self._launch(
            tracer_conc,
            output,
            plan,
            surface_flux_kg_m2_s,
            has_flux=has_flux,
            shape=(1, nlev, nlat, nlon, nlane),
            tracer_count=nlane,
        )
        return CudaVdiffResult(
            tracer_conc=output,
            specific_humidity_kg_kg=plan.specific_humidity_after,
            negative_count_before_clip=self._negative_count,
        )

    def apply_blocks(
        self,
        tracer_blocks: Any,
        plan: CudaVdiffPlan,
        surface_flux_blocks: Any,
        *,
        has_flux: bool,
        tracer_count: int | None = None,
        output: Any | None = None,
    ) -> CudaVdiffResult:
        """Apply VDIFF directly to ``(block, lev, lat, lon, lane)`` storage."""

        self._validate_device_array(tracer_blocks, "tracer_blocks")
        self._validate_device_array(surface_flux_blocks, "surface_flux_blocks")
        if tracer_blocks.ndim != 5:
            raise ValueError("CUDA VDIFF block storage must be 5-D")
        nblock, nlev, nlat, nlon, nlane = tracer_blocks.shape
        if nblock < 1:
            raise ValueError("CUDA VDIFF requires at least one tracer block")
        if nlev < 2:
            raise ValueError("CUDA VDIFF requires at least two vertical levels")
        self._validate_inputs(tracer_blocks[0], plan, surface_flux_blocks[0])
        if surface_flux_blocks.shape != (nblock, nlat, nlon, nlane):
            raise ValueError(
                "CUDA VDIFF block surface flux shape does not match tracer storage"
            )
        capacity = nblock * nlane
        active_tracers = capacity if tracer_count is None else int(tracer_count)
        if active_tracers < 1 or active_tracers > capacity:
            raise ValueError("CUDA VDIFF tracer count must fit block storage")
        if active_tracers <= (nblock - 1) * nlane:
            raise ValueError("CUDA VDIFF tracer count leaves an empty final block")
        if not tracer_blocks.flags.c_contiguous:
            raise ValueError("CUDA VDIFF tracer blocks must be C-contiguous")
        output = self._resolve_output(tracer_blocks, output)

        if self._qmx is None or self._qmx.shape != tracer_blocks.shape:
            self._qmx = self._runtime.empty(tracer_blocks.shape, dtype=self._dtype)
        self._negative_count.fill(0)
        if active_tracers != capacity:
            output.fill(0)
        self._launch(
            tracer_blocks,
            output,
            plan,
            surface_flux_blocks,
            has_flux=has_flux,
            shape=(nblock, nlev, nlat, nlon, nlane),
            tracer_count=active_tracers,
        )
        return CudaVdiffResult(
            tracer_conc=output,
            specific_humidity_kg_kg=plan.specific_humidity_after,
            negative_count_before_clip=self._negative_count,
        )

    def _launch(
        self,
        tracer: Any,
        output: Any,
        plan: CudaVdiffPlan,
        surface_flux: Any,
        *,
        has_flux: bool,
        shape: tuple[int, int, int, int, int],
        tracer_count: int,
    ) -> None:
        _nblock, nlev, nlat, nlon, nlane = shape
        work_size = nlat * nlon * tracer_count
        threads = 128
        blocks = (work_size + threads - 1) // threads
        scalar_type = self._dtype.type
        self._kernel(
            (blocks,),
            (threads,),
            (
                tracer,
                output,
                plan.cch,
                plan.zeh,
                plan.termh,
                plan.dry_mass,
                plan.area_m2,
                plan.cgs,
                plan.kvh,
                plan.potbar,
                plan.rpdel,
                plan.rrho,
                plan.tmp1,
                scalar_type(plan.dt_s),
                plan.start_level,
                surface_flux,
                np.int32(bool(has_flux)),
                self._qmx,
                self._negative_count,
                np.int32(tracer_count),
                np.int32(nlev),
                np.int32(nlat),
                np.int32(nlon),
                np.int32(nlane),
            ),
        )

    def _resolve_output(self, tracer: Any, output: Any | None) -> Any:
        if output is None:
            return self._runtime.empty(tracer.shape, dtype=self._dtype)
        self._validate_device_array(output, "output")
        if output.shape != tracer.shape or output.dtype != self._dtype:
            raise ValueError("CUDA VDIFF output must match tracer shape and dtype")
        if not output.flags.c_contiguous:
            raise ValueError("CUDA VDIFF output must be C-contiguous")
        if self._cupy.shares_memory(output, tracer):
            raise ValueError("CUDA VDIFF output must not overlap tracer input")
        return output

    def _validate_inputs(
        self,
        tracer: Any,
        plan: CudaVdiffPlan,
        surface_flux: Any,
    ) -> None:
        self._validate_device_array(tracer, "tracer_conc")
        self._validate_device_array(surface_flux, "surface_flux_kg_m2_s")
        if tracer.ndim != 4:
            raise ValueError("CUDA VDIFF tracer storage must be 4-D")
        nlev, nlat, nlon, nlane = tracer.shape
        center_shape = (nlev, nlat, nlon)
        edge_shape = (nlev + 1, nlat, nlon)
        horizontal_shape = (nlat, nlon)
        expected_shapes = (
            ("cch", plan.cch, center_shape),
            ("zeh", plan.zeh, center_shape),
            ("termh", plan.termh, center_shape),
            ("cgs", plan.cgs, edge_shape),
            ("kvh", plan.kvh, edge_shape),
            ("potbar", plan.potbar, edge_shape),
            ("rpdel", plan.rpdel, center_shape),
            ("dry_mass", plan.dry_mass, center_shape),
            ("specific_humidity_after", plan.specific_humidity_after, center_shape),
            ("rrho", plan.rrho, horizontal_shape),
            ("tmp1", plan.tmp1, horizontal_shape),
            ("area_m2", plan.area_m2, horizontal_shape),
        )
        for name, values, expected in expected_shapes:
            self._validate_device_array(values, name)
            if values.shape != expected:
                raise ValueError(
                    f"CUDA VDIFF {name} shape {values.shape} does not match {expected}"
                )
            if values.dtype != self._dtype:
                raise TypeError(f"CUDA VDIFF {name} dtype must be {self._dtype}")
        self._validate_device_array(plan.start_level, "start_level")
        if (
            plan.start_level.shape != (1,)
            or plan.start_level.dtype != np.dtype(np.int32)
        ):
            raise TypeError(
                "CUDA VDIFF start_level must have shape (1,) and dtype int32"
            )
        if tracer.dtype != self._dtype:
            raise TypeError(f"CUDA VDIFF tracer dtype must be {self._dtype}")
        if surface_flux.shape != (nlat, nlon, nlane):
            raise ValueError("CUDA VDIFF surface flux shape does not match tracer storage")
        if surface_flux.dtype != self._dtype:
            raise TypeError(f"CUDA VDIFF surface flux dtype must be {self._dtype}")
        if not tracer.flags.c_contiguous or not surface_flux.flags.c_contiguous:
            raise ValueError("CUDA VDIFF inputs must be C-contiguous")

    def _validate_device_array(self, values: Any, label: str) -> None:
        if not self._runtime.is_device_array(values):
            raise TypeError(f"CUDA VDIFF {label} must be a CuPy array")
