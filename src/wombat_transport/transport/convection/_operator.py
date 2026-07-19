"""Runtime convection operator and its compiled execution variants."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import wraps

import numpy as np

from wombat_transport.transport.convection import _reference
from wombat_transport.transport.convection._reference import (
    _CONVECTION_SCRATCH_PAD_TRACERS,
    _TINYNUM,
)
from wombat_transport.transport.numba_control import (
    configure_numba_threads,
    NoAliasCompiler,
    numba_available_and_enabled,
    numba_mode,
    synchronized_transport_numba,
)

try:  # Optional acceleration path; NumPy remains the reference fallback.
    from numba import get_thread_id, njit, prange
except ImportError:  # pragma: no cover - exercised in environments without numba.
    get_thread_id = None
    njit = None
    prange = range

_NUMBA_AVAILABLE = njit is not None

@dataclass(frozen=True)
class _ConvectionKernelWorkspace:
    shape: tuple[int, int]
    qc: np.ndarray
    qb_num: np.ndarray
    delq_work: np.ndarray
    current_work: np.ndarray


_CONVECTION_KERNEL_WORKSPACES = threading.local()


def _get_convection_kernel_workspace(nthreads: int, ntracer: int) -> _ConvectionKernelWorkspace:
    stride = max(ntracer, _CONVECTION_SCRATCH_PAD_TRACERS)
    shape = (nthreads, stride)
    existing = getattr(_CONVECTION_KERNEL_WORKSPACES, "workspace", None)
    if existing is not None and existing.shape == shape:
        return existing
    workspace = _ConvectionKernelWorkspace(
        shape=shape,
        qc=np.empty(shape, dtype=np.float64),
        qb_num=np.empty(shape, dtype=np.float64),
        delq_work=np.empty(shape, dtype=np.float64),
        current_work=np.empty(shape, dtype=np.float64),
    )
    _CONVECTION_KERNEL_WORKSPACES.workspace = workspace
    return workspace



def _numba_convection_mode() -> str:
    return numba_mode()


def _numba_convection_enabled() -> bool:
    return numba_available_and_enabled(available=_NUMBA_AVAILABLE)


@synchronized_transport_numba
def _convect_fullgrid_top_numba(
    q_all: np.ndarray,
    diag_all: np.ndarray,
    cmfmc_all: np.ndarray,
    dtrain_all: np.ndarray,
    delp_hpa_all: np.ndarray,
    delp_dry_all: np.ndarray,
    bmass_all: np.ndarray,
    dqrcu_met_all: np.ndarray,
    reevapcn_met_all: np.ndarray,
    area_all: np.ndarray,
    *,
    diagnostics: bool,
    reconstruct_conv_precip_flux: bool,
    internal_steps: int,
    internal_dt_s: float,
) -> None:
    if not _NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available")
    ntracer = q_all.shape[2]
    nthreads = configure_numba_threads(available=_NUMBA_AVAILABLE)
    workspace = _get_convection_kernel_workspace(nthreads, ntracer)
    _convect_fullgrid_top_numba_kernel(
        q_all,
        diag_all,
        cmfmc_all,
        dtrain_all,
        delp_hpa_all,
        delp_dry_all,
        bmass_all,
        dqrcu_met_all,
        reevapcn_met_all,
        area_all,
        diagnostics,
        reconstruct_conv_precip_flux,
        internal_steps,
        internal_dt_s,
        workspace.qc,
        workspace.qb_num,
        workspace.delq_work,
        workspace.current_work,
    )


if njit is not None:

    @njit(
        cache=True,
        parallel=True,
        nogil=True,
        fastmath={"contract"},
        pipeline_class=NoAliasCompiler,
    )
    def _convect_fullgrid_top_numba_kernel(
        q_all: np.ndarray,
        diag_all: np.ndarray,
        cmfmc_all: np.ndarray,
        dtrain_all: np.ndarray,
        delp_hpa_all: np.ndarray,
        delp_dry_all: np.ndarray,
        bmass_all: np.ndarray,
        dqrcu_met_all: np.ndarray,
        reevapcn_met_all: np.ndarray,
        area_all: np.ndarray,
        diagnostics: bool,
        reconstruct_conv_precip_flux: bool,
        internal_steps: int,
        internal_dt_s: float,
        qc_workspace: np.ndarray,
        qb_num_workspace: np.ndarray,
        delq_work_workspace: np.ndarray,
        current_work_workspace: np.ndarray,
    ) -> None:
        nlev = q_all.shape[0]
        ncol = q_all.shape[1]
        ntracer = q_all.shape[2]
        bottom_index = nlev - 1
        dns = float(internal_steps)

        for col in prange(ncol):
            thread_id = get_thread_id()
            qc = qc_workspace[thread_id]
            qb_num = qb_num_workspace[thread_id]
            delq_work = delq_work_workspace[thread_id]
            current_work = current_work_workspace[thread_id]
            cloud_base = bottom_index
            for level in range(bottom_index, -1, -1):
                dqrcu_value = 0.0
                if reconstruct_conv_precip_flux:
                    if level == 0:
                        dqrcu_value = dqrcu_met_all[level, col] + reevapcn_met_all[level, col]
                    elif level < bottom_index:
                        dqrcu_value = dqrcu_met_all[level, col] + (
                            reevapcn_met_all[level, col] * delp_hpa_all[level, col]
                            - reevapcn_met_all[level - 1, col] * delp_hpa_all[level - 1, col]
                        ) / delp_hpa_all[level, col]
                else:
                    dqrcu_value = dqrcu_met_all[level, col]
                if dqrcu_value > 0.0:
                    cloud_base = level
                    break

            mixes_below_base = cloud_base < bottom_index and cmfmc_all[cloud_base + 1, col] > _TINYNUM
            denominator = 1.0
            mass_below_base = 0.0
            cmfmc_base = 0.0
            inv_denominator = 1.0
            inv_denom_qc = 1.0
            if mixes_below_base:
                denominator = 0.0
                for level in range(cloud_base + 1, nlev):
                    denominator += delp_dry_all[level, col]
                    mass_below_base += bmass_all[level, col]
                if denominator <= 0.0:
                    denominator = 1.0
                cmfmc_base = cmfmc_all[cloud_base + 1, col]
                denom_qc = mass_below_base + cmfmc_base * internal_dt_s
                inv_denominator = 1.0 / denominator
                inv_denom_qc = 1.0 / denom_qc

            for step in range(internal_steps):
                _ = step
                for tracer in range(ntracer):
                    qc[tracer] = q_all[cloud_base, col, tracer]

                if mixes_below_base:
                    for tracer in range(ntracer):
                        qb_num[tracer] = 0.0
                    for level in range(cloud_base + 1, nlev):
                        delp_dry = delp_dry_all[level, col]
                        for tracer in range(ntracer):
                            qb_num[tracer] += q_all[level, col, tracer] * delp_dry
                    for tracer in range(ntracer):
                        qb = qb_num[tracer] * inv_denominator
                        plume = (
                            mass_below_base * qb
                            + cmfmc_base * q_all[cloud_base, col, tracer] * internal_dt_s
                        ) * inv_denom_qc
                        qc[tracer] = plume
                    for level in range(cloud_base + 1, nlev):
                        for tracer in range(ntracer):
                            q_all[level, col, tracer] = qc[tracer]

                for level in range(cloud_base, 0, -1):
                    if level == bottom_index:
                        cmfmc_below = 0.0
                    else:
                        cmfmc_below = cmfmc_all[level + 1, col]

                    if cmfmc_below > _TINYNUM:
                        cmfmc_current = cmfmc_all[level, col]
                        cmout = cmfmc_current + dtrain_all[level, col]
                        entrn = cmout - cmfmc_below
                        entrains = entrn >= 0.0 and cmout > 0.0
                        tendency_scale = internal_dt_s / bmass_all[level, col]
                        area_scale = area_all[col] / dns

                        if entrains:
                            inv_cmout = 1.0 / cmout
                            for tracer in range(ntracer):
                                qc_pres = qc[tracer]
                                current = q_all[level, col, tracer]
                                qc_next = (cmfmc_below * qc_pres + entrn * current) * inv_cmout

                                delq = cmfmc_below * qc_pres
                                temp = -(cmfmc_current * qc_next)
                                delq += temp
                                qc[tracer] = qc_next

                                upward = cmfmc_current * q_all[level - 1, col, tracer]
                                delq += upward
                                if diagnostics:
                                    diag_all[level, col, tracer] += (-temp - upward) * area_scale
                                delq -= cmfmc_below * current
                                current_work[tracer] = current
                                delq_work[tracer] = delq * tendency_scale
                        else:
                            for tracer in range(ntracer):
                                qc_pres = qc[tracer]
                                current = q_all[level, col, tracer]
                                delq = cmfmc_below * qc_pres
                                temp = -(cmfmc_current * qc_pres)
                                delq += temp

                                upward = cmfmc_current * q_all[level - 1, col, tracer]
                                delq += upward
                                if diagnostics:
                                    diag_all[level, col, tracer] += (-temp - upward) * area_scale
                                delq -= cmfmc_below * current
                                current_work[tracer] = current
                                delq_work[tracer] = delq * tendency_scale

                        for tracer in range(ntracer):
                            current = current_work[tracer]
                            delq = delq_work[tracer]
                            if current + delq < 0.0:
                                delq = -current
                            q_all[level, col, tracer] = current + delq
                    else:
                        cmfmc_current = cmfmc_all[level, col]
                        has_current_flux = cmfmc_current > _TINYNUM
                        tendency_scale = internal_dt_s / bmass_all[level, col]
                        for tracer in range(ntracer):
                            qc[tracer] = q_all[level, col, tracer]
                            if has_current_flux:
                                delq = -(cmfmc_current * qc[tracer])
                                delq += cmfmc_current * q_all[level - 1, col, tracer]
                                delq *= tendency_scale
                                current = q_all[level, col, tracer]
                                if current + delq < 0.0:
                                    delq = -current
                                q_all[level, col, tracer] = current + delq

else:

    def _convect_fullgrid_top_numba_kernel(
        q_all: np.ndarray,
        diag_all: np.ndarray,
        cmfmc_all: np.ndarray,
        dtrain_all: np.ndarray,
        delp_hpa_all: np.ndarray,
        delp_dry_all: np.ndarray,
        bmass_all: np.ndarray,
        dqrcu_met_all: np.ndarray,
        reevapcn_met_all: np.ndarray,
        area_all: np.ndarray,
        diagnostics: bool,
        reconstruct_conv_precip_flux: bool,
        internal_steps: int,
        internal_dt_s: float,
        qc_workspace: np.ndarray,
        qb_num_workspace: np.ndarray,
        delq_work_workspace: np.ndarray,
        current_work_workspace: np.ndarray,
    ) -> None:
        raise RuntimeError("numba is not available")

def _column_mass_transport_numba(tracer: np.ndarray, bmass_kg_m2: np.ndarray, area_m2: np.ndarray) -> np.ndarray:
    if not _NUMBA_AVAILABLE:
        return np.sum(
            tracer * bmass_kg_m2[:, :, :, np.newaxis] * area_m2[np.newaxis, :, :, np.newaxis],
            axis=(0, 1, 2),
        )
    return _column_mass_transport_numba_kernel(tracer, bmass_kg_m2, area_m2)


if njit is not None:

    @njit(cache=True, nogil=True)
    def _column_mass_transport_numba_kernel(
        tracer: np.ndarray,
        bmass_kg_m2: np.ndarray,
        area_m2: np.ndarray,
    ) -> np.ndarray:
        nlev = tracer.shape[0]
        nlat = tracer.shape[1]
        nlon = tracer.shape[2]
        ntracer = tracer.shape[3]
        total = np.zeros(ntracer, dtype=np.float64)
        for lev in range(nlev):
            for lat in range(nlat):
                for lon in range(nlon):
                    mass = bmass_kg_m2[lev, lat, lon] * area_m2[lat, lon]
                    for tracer_index in range(ntracer):
                        total[tracer_index] += tracer[lev, lat, lon, tracer_index] * mass
        return total

else:

    def _column_mass_transport_numba_kernel(
        tracer: np.ndarray,
        bmass_kg_m2: np.ndarray,
        area_m2: np.ndarray,
    ) -> np.ndarray:
        raise RuntimeError("numba is not available")


def _convect_compiled(
    tracer: np.ndarray,
    diag14: np.ndarray,
    cmfmc: np.ndarray,
    dtrain: np.ndarray,
    dqrcu_met: np.ndarray,
    reevapcn_met: np.ndarray,
    delp_hpa: np.ndarray,
    delp_dry: np.ndarray,
    bmass: np.ndarray,
    area_m2: np.ndarray,
    *,
    reconstruct_conv_precip_flux: bool,
    diagnostics: bool,
    internal_steps: int,
    internal_dt_s: float,
) -> None:
    """Adapt canonical operator arrays to the compiled column layout."""

    nlev, nlat, nlon, ntracer = tracer.shape
    ncol = nlat * nlon
    _convect_fullgrid_top_numba(
        tracer.reshape(nlev, ncol, ntracer),
        diag14.reshape(nlev, ncol, ntracer) if diagnostics else diag14.reshape(0, 0, 0),
        cmfmc.reshape(nlev, ncol),
        dtrain.reshape(nlev, ncol),
        delp_hpa.reshape(nlev, ncol),
        delp_dry.reshape(nlev, ncol),
        bmass.reshape(nlev, ncol),
        dqrcu_met.reshape(nlev, ncol),
        reevapcn_met.reshape(nlev, ncol),
        area_m2.reshape(ncol),
        diagnostics=diagnostics,
        reconstruct_conv_precip_flux=reconstruct_conv_precip_flux,
        internal_steps=internal_steps,
        internal_dt_s=internal_dt_s,
    )


if njit is not None:
    _convect_block_serial = njit(
        nogil=True,
        fastmath={"contract"},
        pipeline_class=NoAliasCompiler,
    )(
        _convect_fullgrid_top_numba_kernel.py_func
    )
    _convect_block_spatial = _convect_fullgrid_top_numba_kernel
else:
    _convect_block_serial = None
    _convect_block_spatial = None


@wraps(_reference.run_cloud_convection_one_step)
def run_cloud_convection_one_step(*args, **kwargs):
    """Run convection through the reference or compiled operator implementation."""

    if _numba_convection_enabled():
        kwargs["_convect_impl"] = _convect_compiled
        kwargs["_mass_impl"] = _column_mass_transport_numba
    return _reference.run_cloud_convection_one_step(*args, **kwargs)
