"""Long-lived CUDA ownership for a runner transport state."""

from __future__ import annotations

from typing import Any

import numpy as np

from wombat_transport.cuda.forcing import CudaForcingChunks
from wombat_transport.cuda.preparation import CudaPlanPreparation
from wombat_transport.cuda.runtime import CudaRuntime
from wombat_transport.cuda.transport_step import CudaTransportStepExecutor
from wombat_transport.cuda.transport_step import CudaTransportStepPlans
from wombat_transport.emissions import SurfaceEmissions
from wombat_transport.fields import TracerField
from wombat_transport.grid import TransportGrid
from wombat_transport.transport.driver import PreparedTransportStep
from wombat_transport.transport.driver import TransportStepResult
from wombat_transport.transport.driver import transport_result_from_prepared
from wombat_transport.transport.forcing import TransportForcingChunkSelection
from wombat_transport.transport.tpcore.types import TpcoreStaticTerms


class CudaRunExecutor:
    """Own resident tracer state and apply CPU-prepared transport steps."""

    def __init__(
        self,
        state: TracerField,
        *,
        dtype: np.dtype[Any] | type[Any],
        device_id: int = 0,
        grid: TransportGrid | None = None,
        tpcore_static_terms: TpcoreStaticTerms | None = None,
        initial_dry_surface_pressure_hpa: np.ndarray | None = None,
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
        resident_arguments = (
            grid,
            tpcore_static_terms,
            initial_dry_surface_pressure_hpa,
        )
        if any(value is not None for value in resident_arguments) and not all(
            value is not None for value in resident_arguments
        ):
            raise ValueError(
                "resident CUDA preparation requires grid, static terms, and "
                "initial dry surface pressure"
            )
        self.forcing_chunks: CudaForcingChunks | None = None
        self.preparation: CudaPlanPreparation | None = None
        if grid is not None:
            assert tpcore_static_terms is not None
            assert initial_dry_surface_pressure_hpa is not None
            self.forcing_chunks = CudaForcingChunks(
                self.runtime,
                dtype=np.float64,
            )
            self.preparation = CudaPlanPreparation(
                self.runtime,
                dtype=self.dtype,
                grid=grid,
                tpcore_static_terms=tpcore_static_terms,
                initial_dry_surface_pressure_hpa=(
                    initial_dry_surface_pressure_hpa
                ),
            )
        self._surface_flux_identity: int | None = None
        self._surface_flux_blocks: Any | None = None
        self._has_surface_flux = False

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

    def apply_resident(
        self,
        selection: TransportForcingChunkSelection,
        *,
        dt_s: float,
        active_emissions: SurfaceEmissions | None,
        surface_flux_to_vmr_factor: np.ndarray | None,
    ) -> TransportStepResult:
        """Prepare and apply a step entirely from resident forcing chunks."""

        if self.forcing_chunks is None or self.preparation is None:
            raise ValueError("resident CUDA forcing preparation is not configured")
        forcing = self.forcing_chunks.select(selection)
        tpcore = self.preparation.prepare_tpcore_step(forcing, dt_s=dt_s)
        vdiff, convection = self.preparation.prepare_vdiff_and_convection(
            forcing,
            dt_s=dt_s,
        )
        surface_flux = self._resident_surface_flux(
            active_emissions,
            surface_flux_to_vmr_factor,
        )
        result = self.transport.apply(
            self.state.block_data[0],
            CudaTransportStepPlans(
                tpcore=tpcore,
                vdiff=vdiff,
                convection=convection,
                surface_flux_blocks=surface_flux,
                has_surface_flux=self._has_surface_flux,
            ),
            tracer_count=self.state.tracer_count,
        )
        self.state = TracerField(
            names=self.state.names,
            data=result.tracer_blocks[None, ...],
            units=self.state.units,
            coords=self.state.coords,
        )
        return TransportStepResult(
            state=self.state,
            dry_air_mass_kg=self.preparation.next_dry_air_mass_kg,
            delp_dry_hpa=self.preparation.next_delp_dry_hpa,
            specific_humidity_kg_kg=(
                self.preparation.specific_humidity_after_kg_kg
            ),
            xmass_hpa=None,
            ymass_hpa=None,
            zmass_hpa=None,
            transport_operators=("tpcore", "vdiff", "convection"),
        )

    def snapshot_forcing(self, template: Any) -> Any:
        """Attach current resident output meteorology to host metadata."""

        if self.preparation is None:
            raise ValueError("resident CUDA forcing preparation is not configured")
        from dataclasses import replace

        meteorology = self.preparation.meteorology
        return replace(
            template,
            wet_surface_pressure_hpa=(
                meteorology.wet_surface_pressure_hpa[None, ...]
            ),
            dry_surface_pressure_hpa=(
                meteorology.dry_surface_pressure_hpa[None, ...]
            ),
            specific_humidity_kg_kg=(
                self.preparation.specific_humidity_after_kg_kg
            ),
            temperature_k=meteorology.temperature_k[None, ...],
        )

    def _resident_surface_flux(
        self,
        active_emissions: SurfaceEmissions | None,
        surface_flux_to_vmr_factor: np.ndarray | None,
    ) -> Any:
        identity = None if active_emissions is None else id(active_emissions)
        if self._surface_flux_blocks is not None and (
            self._surface_flux_identity == identity
        ):
            return self._surface_flux_blocks
        nlat = self.state.block_data.shape[3]
        nlon = self.state.block_data.shape[4]
        ntracer = self.state.tracer_count
        if active_emissions is None:
            values = np.zeros((nlat, nlon, ntracer), dtype=np.float64)
        else:
            if active_emissions.names != self.state.names:
                raise ValueError(
                    "active emissions names do not match tracer field names"
                )
            values = np.asarray(active_emissions.data, dtype=np.float64)
            if values.shape != (nlat, nlon, ntracer):
                raise ValueError(
                    f"active surface emissions shape {values.shape} does not "
                    f"match {(nlat, nlon, ntracer)}"
                )
        if surface_flux_to_vmr_factor is not None:
            factor = np.asarray(surface_flux_to_vmr_factor, dtype=np.float64)
            if factor.shape != (ntracer,):
                raise ValueError(
                    "surface flux conversion factor does not match tracer count"
                )
            values = values * factor[None, None, :]
        blocks = _to_horizontal_blocks(values, self.state.block_width)
        if (
            self._surface_flux_blocks is None
            or self._surface_flux_blocks.shape != blocks.shape
        ):
            self._surface_flux_blocks = self.runtime.to_device(
                blocks,
                dtype=self.dtype,
            )
        else:
            self.runtime.copy_to_device(self._surface_flux_blocks, blocks)
        self._surface_flux_identity = identity
        self._has_surface_flux = bool(np.any(values != 0.0))
        return self._surface_flux_blocks

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
    nblock = (ntracer + width - 1) // width
    result = np.zeros(
        (nblock, nlat, nlon, width),
        dtype=values.dtype,
    )
    for block in range(nblock):
        start = block * width
        stop = min(start + width, ntracer)
        result[block, ..., : stop - start] = values[..., start:stop]
    return result
