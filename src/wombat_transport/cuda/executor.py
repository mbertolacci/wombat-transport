"""Long-lived CUDA ownership for a runner transport state."""

from __future__ import annotations

from typing import Any

import numpy as np

from wombat_transport.cuda.runtime import CudaRuntime
from wombat_transport.cuda.transport_step import CudaTransportStepExecutor
from wombat_transport.cuda.transport_step import CudaTransportStepPlans
from wombat_transport.fields import TracerField
from wombat_transport.transport.driver import PreparedTransportStep
from wombat_transport.transport.driver import TransportStepResult
from wombat_transport.transport.driver import transport_result_from_prepared


class CudaRunExecutor:
    """Own resident tracer state and apply CPU-prepared transport steps."""

    def __init__(
        self,
        state: TracerField,
        *,
        dtype: np.dtype[Any] | type[Any],
        device_id: int = 0,
    ) -> None:
        if state.block_data.shape[0] != 1:
            raise ValueError("CUDA transport requires exactly one time slice")
        self.runtime = CudaRuntime(device_id)
        self.dtype = np.dtype(dtype)
        device_blocks = self.runtime.to_device(
            state.block_data[0],
            dtype=self.dtype,
        )
        self.state = TracerField(
            names=state.names,
            data=device_blocks[None, ...],
            units=state.units,
            coords=state.coords,
        )
        self.transport = CudaTransportStepExecutor(
            self.runtime,
            dtype=self.dtype,
        )

    def apply(self, prepared: PreparedTransportStep) -> TransportStepResult:
        """Upload one prepared step and apply it without staging tracer state."""

        surface_flux_blocks = _to_horizontal_blocks(
            prepared.surface_flux_kg_m2_s,
            self.state.block_width,
        )
        plans = CudaTransportStepPlans(
            tpcore=self.transport.tpcore.upload_plan(prepared.tpcore_plan),
            vdiff=self.transport.vdiff.upload_plan(prepared.vdiff_plan),
            convection=self.transport.convection.upload_plan(
                cmfmc_kg_m2_s=prepared.cmfmc,
                dtrain_kg_m2_s=prepared.dtrain,
                dqrcu_kg_kg_s=prepared.dqrcu,
                reevapcn_kg_kg_s=prepared.reevapcn,
                delp_dry_hpa=prepared.delp_dry,
                delp_hpa=prepared.delp_hpa,
                area_m2=prepared.vdiff_plan.area_m2,
                dt_s=prepared.internal_steps * prepared.internal_dt_s,
                reconstruct_conv_precip_flux=(
                    prepared.reconstruct_conv_precip_flux
                ),
            ),
            surface_flux_blocks=self.runtime.to_device(
                surface_flux_blocks,
                dtype=self.dtype,
            ),
            has_surface_flux=bool(
                np.any(prepared.surface_flux_kg_m2_s != 0.0)
            ),
        )
        result = self.transport.apply(
            self.state.block_data[0],
            plans,
            tracer_count=self.state.tracer_count,
        )
        self.state = TracerField(
            names=self.state.names,
            data=result.tracer_blocks[None, ...],
            units=self.state.units,
            coords=self.state.coords,
        )
        return transport_result_from_prepared(self.state, prepared)

    def to_host_state(self) -> TracerField:
        """Materialize the current tracer state at an explicit boundary."""

        return TracerField(
            names=self.state.names,
            data=self.runtime.to_host(self.state.block_data),
            units=self.state.units,
            coords=self.state.coords,
        )


def _to_horizontal_blocks(values: np.ndarray, width: int) -> np.ndarray:
    nlat, nlon, ntracer = values.shape
    result = np.zeros(
        ((ntracer + width - 1) // width, nlat, nlon, width),
        dtype=values.dtype,
    )
    for tracer in range(ntracer):
        block, lane = divmod(tracer, width)
        result[block, ..., lane] = values[..., tracer]
    return result
