"""Resident composition of the three CUDA transport operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from wombat_transport.cuda.runtime import CudaRuntime
from wombat_transport.transport.convection._cuda import CudaConvectionExecutor
from wombat_transport.transport.convection._cuda import CudaConvectionPlan
from wombat_transport.transport.pbl._cuda import CudaVdiffExecutor
from wombat_transport.transport.pbl._cuda import CudaVdiffPlan
from wombat_transport.transport.tpcore._cuda import CudaTpcoreExecutor
from wombat_transport.transport.tpcore._cuda import CudaTpcorePlan


@dataclass(frozen=True)
class CudaTransportStepPlans:
    tpcore: CudaTpcorePlan
    vdiff: CudaVdiffPlan
    convection: CudaConvectionPlan
    surface_flux_blocks: Any
    has_surface_flux: bool


@dataclass(frozen=True)
class CudaTransportStepResult:
    tracer_blocks: Any
    tpcore_tracer_blocks: Any
    vdiff_tracer_blocks: Any | None
    specific_humidity_kg_kg: Any
    negative_count_before_vdiff_clip: Any


class CudaTransportStepExecutor:
    """Apply TPCORE, VDIFF, and convection without staging tracer state."""

    def __init__(self, runtime: CudaRuntime, *, dtype: np.dtype[Any] | type[Any]) -> None:
        self._runtime = runtime
        self._dtype = np.dtype(dtype)
        self.tpcore = CudaTpcoreExecutor(runtime, dtype=self._dtype)
        self.vdiff = CudaVdiffExecutor(runtime, dtype=self._dtype)
        self.convection = CudaConvectionExecutor(runtime, dtype=self._dtype)
        self._tpcore_output: Any | None = None

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    def apply(
        self,
        tracer_blocks: Any,
        plans: CudaTransportStepPlans,
        *,
        tracer_count: int,
        capture_vdiff_handoff: bool = False,
    ) -> CudaTransportStepResult:
        """Consume one resident state and return executor-owned final storage."""

        if self._tpcore_output is None or self._tpcore_output.shape != tracer_blocks.shape:
            self._tpcore_output = self._runtime.empty(
                tracer_blocks.shape,
                dtype=self._dtype,
            )

        tpcore_result = self.tpcore.apply_blocks(
            tracer_blocks,
            plans.tpcore,
            tracer_count=tracer_count,
            output=self._tpcore_output,
        )
        vdiff_result = self.vdiff.apply_blocks(
            tpcore_result.tracer_blocks,
            plans.vdiff,
            plans.surface_flux_blocks,
            has_flux=plans.has_surface_flux,
            tracer_count=tracer_count,
            output=tracer_blocks,
            workspace=self.tpcore.expired_horizontal_workspace,
        )
        vdiff_handoff = (
            vdiff_result.tracer_conc.copy()
            if capture_vdiff_handoff
            else None
        )
        convection_result = self.convection.apply_blocks(
            vdiff_result.tracer_conc,
            plans.convection,
            tracer_count=tracer_count,
        )
        return CudaTransportStepResult(
            tracer_blocks=convection_result.tracer_blocks,
            tpcore_tracer_blocks=tpcore_result.tracer_blocks,
            vdiff_tracer_blocks=vdiff_handoff,
            specific_humidity_kg_kg=vdiff_result.specific_humidity_kg_kg,
            negative_count_before_vdiff_clip=(
                vdiff_result.negative_count_before_clip
            ),
        )
