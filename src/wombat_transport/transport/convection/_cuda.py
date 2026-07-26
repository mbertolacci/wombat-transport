"""CUDA application of the cloud-convection transport operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from wombat_transport.cuda.modules import load_raw_module
from wombat_transport.cuda.runtime import CudaRuntime
from wombat_transport.transport.convection import G0_100


@dataclass(frozen=True)
class CudaConvectionPlan:
    cmfmc: Any
    dtrain: Any
    delp_hpa: Any
    delp_dry: Any
    bmass: Any
    dqrcu: Any
    reevapcn: Any
    area_m2: Any
    reconstruct_conv_precip_flux: bool
    internal_steps: int
    internal_dt_s: float


@dataclass(frozen=True)
class CudaConvectionResult:
    tracer_blocks: Any
    diag14_mass_flux: Any | None


class CudaConvectionExecutor:
    """Own the strict, block-native CUDA convection kernel."""

    def __init__(self, runtime: CudaRuntime, *, dtype: np.dtype[Any] | type[Any]) -> None:
        resolved_dtype = np.dtype(dtype)
        if resolved_dtype == np.dtype(np.float32):
            cuda_type = "float"
        elif resolved_dtype == np.dtype(np.float64):
            cuda_type = "double"
        else:
            raise ValueError("CUDA convection supports only float32 and float64")
        expression = f"apply_convection<{cuda_type}>"
        module = load_raw_module("convection.cu", name_expressions=(expression,))
        self._runtime = runtime
        self._dtype = resolved_dtype
        self._kernel = module.get_function(expression)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    def upload_plan(
        self,
        *,
        cmfmc_kg_m2_s: np.ndarray,
        dtrain_kg_m2_s: np.ndarray,
        dqrcu_kg_kg_s: np.ndarray,
        reevapcn_kg_kg_s: np.ndarray,
        delp_dry_hpa: np.ndarray,
        delp_hpa: np.ndarray,
        area_m2: np.ndarray,
        dt_s: float,
        reconstruct_conv_precip_flux: bool,
    ) -> CudaConvectionPlan:
        """Upload tracer-independent convection inputs for one transport step."""

        delp_dry = np.asarray(delp_dry_hpa, dtype=np.float64)
        if dt_s <= 0.0:
            raise ValueError("convection dt_s must be positive")
        if np.any(delp_dry <= 0.0):
            raise ValueError("convection dry pressure must be positive")
        internal_steps = max(int(dt_s) // 300, 1)
        return CudaConvectionPlan(
            cmfmc=self._runtime.to_device(cmfmc_kg_m2_s, dtype=self._dtype),
            dtrain=self._runtime.to_device(dtrain_kg_m2_s, dtype=self._dtype),
            delp_hpa=self._runtime.to_device(delp_hpa, dtype=self._dtype),
            delp_dry=self._runtime.to_device(delp_dry, dtype=self._dtype),
            bmass=self._runtime.to_device(delp_dry * G0_100, dtype=self._dtype),
            dqrcu=self._runtime.to_device(dqrcu_kg_kg_s, dtype=self._dtype),
            reevapcn=self._runtime.to_device(reevapcn_kg_kg_s, dtype=self._dtype),
            area_m2=self._runtime.to_device(area_m2, dtype=self._dtype),
            reconstruct_conv_precip_flux=bool(reconstruct_conv_precip_flux),
            internal_steps=internal_steps,
            internal_dt_s=float(dt_s) / float(internal_steps),
        )

    def apply_blocks(
        self,
        tracer_blocks: Any,
        plan: CudaConvectionPlan,
        *,
        tracer_count: int,
        diagnostics: bool = False,
        diag14_mass_flux: Any | None = None,
    ) -> CudaConvectionResult:
        """Apply convection in-place to resident block storage."""

        self._validate(tracer_blocks, plan, tracer_count)
        nblock, nlev, nlat, nlon, lane_width = tracer_blocks.shape
        _ = nblock
        if diagnostics:
            if diag14_mass_flux is None:
                diag14_mass_flux = self._runtime.zeros(
                    tracer_blocks.shape,
                    dtype=self._dtype,
                )
            else:
                self._validate_storage(diag14_mass_flux, tracer_blocks.shape, "diag14")
        elif diag14_mass_flux is not None:
            raise ValueError("CUDA convection diagnostics output requires diagnostics=True")

        work_size = nlat * nlon * tracer_count
        threads = 128
        blocks = (work_size + threads - 1) // threads
        scalar_type = self._dtype.type
        self._kernel(
            (blocks,),
            (threads,),
            (
                tracer_blocks,
                diag14_mass_flux if diagnostics else tracer_blocks,
                plan.cmfmc,
                plan.dtrain,
                plan.delp_hpa,
                plan.delp_dry,
                plan.bmass,
                plan.dqrcu,
                plan.reevapcn,
                plan.area_m2,
                np.int32(diagnostics),
                np.int32(plan.reconstruct_conv_precip_flux),
                np.int32(plan.internal_steps),
                scalar_type(plan.internal_dt_s),
                np.int32(tracer_count),
                np.int32(nlev),
                np.int32(nlat),
                np.int32(nlon),
                np.int32(lane_width),
            ),
        )
        return CudaConvectionResult(
            tracer_blocks=tracer_blocks,
            diag14_mass_flux=diag14_mass_flux,
        )

    def _validate(
        self,
        tracer: Any,
        plan: CudaConvectionPlan,
        tracer_count: int,
    ) -> None:
        if not self._runtime.is_device_array(tracer):
            raise TypeError("CUDA convection tracer storage must be a CuPy array")
        if tracer.ndim != 5:
            raise ValueError("CUDA convection tracer block storage must be 5-D")
        nblock, nlev, nlat, nlon, lane_width = tracer.shape
        if nlev < 2:
            raise ValueError("CUDA convection requires at least two levels")
        if tracer_count <= (nblock - 1) * lane_width or tracer_count > nblock * lane_width:
            raise ValueError("CUDA convection tracer count does not match block storage")
        self._validate_storage(tracer, tracer.shape, "tracer")
        center_shape = (nlev, nlat, nlon)
        for name in (
            "cmfmc",
            "dtrain",
            "delp_hpa",
            "delp_dry",
            "bmass",
            "dqrcu",
            "reevapcn",
        ):
            self._validate_storage(getattr(plan, name), center_shape, name)
        self._validate_storage(plan.area_m2, (nlat, nlon), "area_m2")

    def _validate_storage(
        self,
        values: Any,
        shape: tuple[int, ...],
        label: str,
    ) -> None:
        if not self._runtime.is_device_array(values):
            raise TypeError(f"CUDA convection {label} must be a CuPy array")
        if values.shape != shape or values.dtype != self._dtype:
            raise ValueError(
                f"CUDA convection {label} must have shape {shape} and dtype {self._dtype}"
            )
        if not values.flags.c_contiguous:
            raise ValueError(f"CUDA convection {label} must be C-contiguous")
