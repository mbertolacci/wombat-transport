"""Persistent CUDA mirrors of block-oriented meteorological forcing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from wombat_transport.cuda.runtime import CudaRuntime
from wombat_transport.transport.forcing import TransportForcingChunkSelection


@dataclass(frozen=True)
class CudaForcingStep:
    """Resident source fields and interpolation coordinates for one step."""

    pblh_m: Any
    hflux_w_m2: Any
    eflux_w_m2: Any
    ustar_m_s: Any
    u_m_s: Any
    v_m_s: Any
    dtrain_kg_m2_s: Any
    dqrcu_kg_kg_s: Any
    reevapcn_kg_kg_s: Any
    cmfmc_kg_m2_s: Any
    surface_pressure_start_pa: Any
    surface_pressure_end_pa: Any
    qv_start: Any
    qv_end: Any
    temperature_start_k: Any
    temperature_end_k: Any
    start_fraction: float
    end_fraction: float
    midpoint_fraction: float


class CudaForcingChunks:
    """Upload each A1/A3/I3 source chunk once and return zero-copy step views."""

    _A1_NAMES = ("pblh", "hflux", "eflux", "ustar")
    _A3_NAMES = ("u", "v", "dtrain", "dqrcu", "reevapcn", "cmfmc")
    _I3_NAMES = ("surface_pressure", "qv", "temperature")

    def __init__(
        self,
        runtime: CudaRuntime,
        *,
        dtype: np.dtype[Any] | type[Any],
    ) -> None:
        self._runtime = runtime
        self._dtype = np.dtype(dtype)
        self._identities: dict[str, int] = {}
        self._arrays: dict[str, dict[str, Any]] = {
            "a1": {},
            "a3": {},
            "i3": {},
        }

    def select(self, selection: TransportForcingChunkSelection) -> CudaForcingStep:
        self._ensure_chunk("a1", selection.a1_block, self._A1_NAMES)
        self._ensure_chunk("a3", selection.a3_block, self._A3_NAMES)
        self._ensure_chunk("i3", selection.i3_block, self._I3_NAMES)
        a1 = self._arrays["a1"]
        a3 = self._arrays["a3"]
        i3 = self._arrays["i3"]
        a1_offset = selection.a1_offset
        a3_offset = selection.a3_offset
        i3_start = selection.i3_start_offset
        i3_end = selection.i3_end_offset
        return CudaForcingStep(
            pblh_m=a1["pblh"][a1_offset],
            hflux_w_m2=a1["hflux"][a1_offset],
            eflux_w_m2=a1["eflux"][a1_offset],
            ustar_m_s=a1["ustar"][a1_offset],
            u_m_s=a3["u"][a3_offset],
            v_m_s=a3["v"][a3_offset],
            dtrain_kg_m2_s=a3["dtrain"][a3_offset],
            dqrcu_kg_kg_s=a3["dqrcu"][a3_offset],
            reevapcn_kg_kg_s=a3["reevapcn"][a3_offset],
            cmfmc_kg_m2_s=a3["cmfmc"][a3_offset, 1:],
            surface_pressure_start_pa=i3["surface_pressure"][i3_start],
            surface_pressure_end_pa=i3["surface_pressure"][i3_end],
            qv_start=i3["qv"][i3_start],
            qv_end=i3["qv"][i3_end],
            temperature_start_k=i3["temperature"][i3_start],
            temperature_end_k=i3["temperature"][i3_end],
            start_fraction=float(selection.start_fraction),
            end_fraction=float(selection.end_fraction),
            midpoint_fraction=float(selection.midpoint_fraction),
        )

    def _ensure_chunk(
        self,
        label: str,
        chunk: Any,
        names: tuple[str, ...],
    ) -> None:
        identity = id(chunk)
        if self._identities.get(label) == identity:
            return
        buffers = self._arrays[label]
        for name in names:
            host = np.asarray(getattr(chunk, name))
            existing = buffers.get(name)
            if (
                existing is None
                or existing.shape != host.shape
                or existing.dtype != self._dtype
            ):
                buffers[name] = self._runtime.to_device(
                    host,
                    dtype=self._dtype,
                )
            else:
                self._runtime.copy_to_device(existing, host)
        self._identities[label] = identity
