"""CUDA application of a prepared TPCORE transport plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from wombat_transport.cuda.modules import load_raw_module
from wombat_transport.cuda.runtime import CudaRuntime
from wombat_transport.transport.tpcore._plan import TpcorePlan


_CUDA_WARP_SIZE = 32
_DEFAULT_ZONAL_WARPS_PER_BLOCK = 4
_WIDE_FLOAT32_ZONAL_WARPS_PER_BLOCK = 8
_WIDE_FLOAT32_TRACER_THRESHOLD = 32
_ZONAL_SHARED_ARRAYS = 4
_MAX_LONGITUDES = 144


def _zonal_warps_per_block(dtype: np.dtype[Any], tracer_count: int) -> int:
    if (
        dtype == np.dtype(np.float32)
        and tracer_count >= _WIDE_FLOAT32_TRACER_THRESHOLD
    ):
        return _WIDE_FLOAT32_ZONAL_WARPS_PER_BLOCK
    return _DEFAULT_ZONAL_WARPS_PER_BLOCK


@dataclass(frozen=True)
class CudaTpcorePlan:
    delp1: Any
    delp2: Any
    pu: Any
    xmass: Any
    ymass: Any
    vertical_mass_flux: Any
    normalized_vertical_courant: Any
    cx: Any
    cy: Any
    geofac: Any
    geofac_pc: float
    ua: Any
    va: Any
    jn: Any
    js: Any
    area_1d: Any


@dataclass(frozen=True)
class CudaTpcoreResult:
    tracer_blocks: Any
    q_after_horizontal: Any | None
    dq_after_horizontal: Any | None
    dq_after_vertical: Any | None


class CudaTpcoreExecutor:
    """Own the strict, staged CUDA TPCORE implementation."""

    def __init__(self, runtime: CudaRuntime, *, dtype: np.dtype[Any] | type[Any]) -> None:
        resolved_dtype = np.dtype(dtype)
        if resolved_dtype == np.dtype(np.float32):
            cuda_type = "float"
        elif resolved_dtype == np.dtype(np.float64):
            cuda_type = "double"
        else:
            raise ValueError("CUDA TPCORE supports only float32 and float64")
        expressions = tuple(
            f"{name}<{cuda_type}>"
            for name in (
                "tpcore_horizontal_poles",
                "tpcore_horizontal_initialize",
                "tpcore_horizontal_zonal_warp",
                "tpcore_horizontal_meridional",
                "tpcore_horizontal_finalize_poles",
                "tpcore_vertical",
            )
        )
        module = load_raw_module("tpcore.cu", name_expressions=expressions)
        self._runtime = runtime
        self._dtype = resolved_dtype
        self._horizontal_poles = module.get_function(expressions[0])
        self._horizontal_initialize = module.get_function(expressions[1])
        self._horizontal_zonal_warp = module.get_function(expressions[2])
        self._horizontal_meridional = module.get_function(expressions[3])
        self._horizontal_finalize_poles = module.get_function(expressions[4])
        self._vertical = module.get_function(expressions[5])
        self._qqu: Any | None = None
        self._qqv: Any | None = None

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    @property
    def expired_horizontal_workspace(self) -> Any:
        """Return scratch whose lifetime ends after a completed TPCORE call."""

        if self._qqu is None:
            raise RuntimeError("CUDA TPCORE workspace is not initialized")
        return self._qqu

    def upload_plan(self, plan: TpcorePlan) -> CudaTpcorePlan:
        """Upload one tracer-independent TPCORE plan."""

        setup = plan.setup
        return CudaTpcorePlan(
            delp1=self._runtime.to_device(setup.delp1_hpa, dtype=self._dtype),
            delp2=self._runtime.to_device(setup.delp2_hpa, dtype=self._dtype),
            pu=self._runtime.to_device(setup.pu_hpa, dtype=self._dtype),
            xmass=self._runtime.to_device(setup.xmass_hpa, dtype=self._dtype),
            ymass=self._runtime.to_device(setup.ymass_hpa, dtype=self._dtype),
            vertical_mass_flux=self._runtime.to_device(
                setup.vertical_mass_flux_hpa,
                dtype=self._dtype,
            ),
            normalized_vertical_courant=self._runtime.to_device(
                plan.normalized_vertical_courant,
                dtype=self._dtype,
            ),
            cx=self._runtime.to_device(setup.cx, dtype=self._dtype),
            cy=self._runtime.to_device(setup.cy, dtype=self._dtype),
            geofac=self._runtime.to_device(setup.geofac, dtype=self._dtype),
            geofac_pc=float(setup.geofac_pc),
            ua=self._runtime.to_device(plan.ua, dtype=self._dtype),
            va=self._runtime.to_device(plan.va, dtype=self._dtype),
            jn=self._runtime.to_device(plan.jn, dtype=np.int64),
            js=self._runtime.to_device(plan.js, dtype=np.int64),
            area_1d=self._runtime.to_device(plan.area_1d_m2, dtype=self._dtype),
        )

    def apply_blocks(
        self,
        tracer_blocks: Any,
        plan: CudaTpcorePlan,
        *,
        tracer_count: int,
        fill: bool = True,
        finalize_output: bool = True,
        output: Any | None = None,
        capture_handoffs: bool = False,
    ) -> CudaTpcoreResult:
        """Consume resident concentration and produce mass or concentration."""

        nblock, nlev, nlat, nlon, lane_width = self._validate(
            tracer_blocks,
            plan,
            tracer_count,
        )
        output = self._resolve_output(tracer_blocks, output)
        if tracer_count != nblock * lane_width:
            output.fill(0)
        if self._qqu is None or self._qqu.shape != tracer_blocks.shape:
            self._qqu = self._runtime.empty(tracer_blocks.shape, dtype=self._dtype)
            self._qqv = self._runtime.empty(tracer_blocks.shape, dtype=self._dtype)

        scalar_type = self._dtype.type
        horizontal_work = nlev * tracer_count
        serial_horizontal_threads = _CUDA_WARP_SIZE
        serial_horizontal_blocks = (
            horizontal_work + serial_horizontal_threads - 1
        ) // serial_horizontal_threads
        self._horizontal_poles(
            (serial_horizontal_blocks,),
            (serial_horizontal_threads,),
            (
                tracer_blocks,
                plan.delp1,
                plan.area_1d,
                np.int32(tracer_count),
                np.int32(nlev),
                np.int32(nlat),
                np.int32(nlon),
                np.int32(lane_width),
            ),
        )
        initialize_work = nlev * nlat * nlon * tracer_count
        initialize_threads = 128
        self._horizontal_initialize(
            (
                (initialize_work + initialize_threads - 1)
                // initialize_threads,
            ),
            (initialize_threads,),
            (
                tracer_blocks,
                output,
                self._qqu,
                self._qqv,
                plan.delp1,
                plan.ua,
                plan.va,
                plan.jn,
                plan.js,
                np.int32(tracer_count),
                np.int32(nlev),
                np.int32(nlat),
                np.int32(nlon),
                np.int32(lane_width),
            ),
        )
        zonal_warps_per_block = _zonal_warps_per_block(
            self._dtype,
            tracer_count,
        )
        self._horizontal_zonal_warp(
            (
                (horizontal_work + zonal_warps_per_block - 1)
                // zonal_warps_per_block,
            ),
            (_CUDA_WARP_SIZE * zonal_warps_per_block,),
            (
                tracer_blocks,
                output,
                self._qqu,
                self._qqv,
                plan.pu,
                plan.xmass,
                plan.cx,
                plan.ua,
                plan.va,
                plan.jn,
                plan.js,
                np.int32(tracer_count),
                np.int32(nlev),
                np.int32(nlat),
                np.int32(nlon),
                np.int32(lane_width),
            ),
            shared_mem=(
                zonal_warps_per_block
                * _ZONAL_SHARED_ARRAYS
                * _MAX_LONGITUDES
                * self._dtype.itemsize
            ),
        )
        meridional_work = horizontal_work * nlon
        meridional_threads = 128
        self._horizontal_meridional(
            (
                (meridional_work + meridional_threads - 1)
                // meridional_threads,
            ),
            (meridional_threads,),
            (
                output,
                self._qqu,
                self._qqv,
                plan.ymass,
                plan.cy,
                plan.geofac,
                np.int32(tracer_count),
                np.int32(nlev),
                np.int32(nlat),
                np.int32(nlon),
                np.int32(lane_width),
            ),
        )
        self._horizontal_finalize_poles(
            (serial_horizontal_blocks,),
            (serial_horizontal_threads,),
            (
                output,
                self._qqv,
                scalar_type(plan.geofac_pc),
                np.int32(tracer_count),
                np.int32(nlev),
                np.int32(nlat),
                np.int32(nlon),
                np.int32(lane_width),
            ),
        )
        q_after_horizontal = tracer_blocks.copy() if capture_handoffs else None
        dq_after_horizontal = output.copy() if capture_handoffs else None

        column_work = nlat * nlon * tracer_count
        column_threads = 128
        column_blocks = (column_work + column_threads - 1) // column_threads
        self._vertical(
            (column_blocks,),
            (column_threads,),
            (
                tracer_blocks,
                output,
                plan.delp1,
                plan.delp2,
                plan.vertical_mass_flux,
                plan.normalized_vertical_courant,
                np.int32(fill),
                np.int32(finalize_output),
                np.int32(tracer_count),
                np.int32(nlev),
                np.int32(nlat),
                np.int32(nlon),
                np.int32(lane_width),
            ),
        )
        dq_after_vertical = output.copy() if capture_handoffs else None
        return CudaTpcoreResult(
            tracer_blocks=output,
            q_after_horizontal=q_after_horizontal,
            dq_after_horizontal=dq_after_horizontal,
            dq_after_vertical=dq_after_vertical,
        )

    def _validate(
        self,
        tracer: Any,
        plan: CudaTpcorePlan,
        tracer_count: int,
    ) -> tuple[int, int, int, int, int]:
        if not self._runtime.is_device_array(tracer):
            raise TypeError("CUDA TPCORE tracer storage must be a CuPy array")
        if tracer.ndim != 5:
            raise ValueError("CUDA TPCORE tracer block storage must be 5-D")
        if tracer.dtype != self._dtype or not tracer.flags.c_contiguous:
            raise ValueError("CUDA TPCORE tracer storage has the wrong dtype or layout")
        nblock, nlev, nlat, nlon, lane_width = tracer.shape
        if nlev != 47 or nlat not in {46, 91} or nlon not in {72, 144}:
            if nlev > 47 or nlat > 91 or nlon > 144:
                raise ValueError("CUDA TPCORE grid exceeds compiled workspace bounds")
        if tracer_count <= (nblock - 1) * lane_width or tracer_count > nblock * lane_width:
            raise ValueError("CUDA TPCORE tracer count does not match block storage")
        center_shape = (nlev, nlat, nlon)
        for name in (
            "delp1",
            "delp2",
            "pu",
            "xmass",
            "ymass",
            "vertical_mass_flux",
            "normalized_vertical_courant",
            "cx",
            "cy",
            "ua",
            "va",
        ):
            self._validate_array(getattr(plan, name), center_shape, name)
        self._validate_array(plan.geofac, (nlat,), "geofac")
        self._validate_array(plan.area_1d, (nlat,), "area_1d")
        self._validate_array(plan.jn, (nlev,), "jn", dtype=np.dtype(np.int64))
        self._validate_array(plan.js, (nlev,), "js", dtype=np.dtype(np.int64))
        return nblock, nlev, nlat, nlon, lane_width

    def _validate_array(
        self,
        values: Any,
        shape: tuple[int, ...],
        label: str,
        *,
        dtype: np.dtype[Any] | None = None,
    ) -> None:
        expected_dtype = self._dtype if dtype is None else dtype
        if not self._runtime.is_device_array(values):
            raise TypeError(f"CUDA TPCORE {label} must be a CuPy array")
        if values.shape != shape or values.dtype != expected_dtype:
            raise ValueError(
                f"CUDA TPCORE {label} must have shape {shape} and dtype {expected_dtype}"
            )
        if not values.flags.c_contiguous:
            raise ValueError(f"CUDA TPCORE {label} must be C-contiguous")

    def _resolve_output(self, tracer: Any, output: Any | None) -> Any:
        if output is None:
            return self._runtime.empty(tracer.shape, dtype=self._dtype)
        if not self._runtime.is_device_array(output):
            raise TypeError("CUDA TPCORE output must be a CuPy array")
        if output.shape != tracer.shape or output.dtype != self._dtype:
            raise ValueError("CUDA TPCORE output must match tracer storage")
        if not output.flags.c_contiguous:
            raise ValueError("CUDA TPCORE output must be C-contiguous")
        if self._runtime.shares_memory(output, tracer):
            raise ValueError("CUDA TPCORE output must not overlap tracer input")
        return output
