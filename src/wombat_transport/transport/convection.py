from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

try:  # Optional acceleration path; NumPy remains the reference fallback.
    from numba import njit
except ImportError:  # pragma: no cover - exercised in environments without numba.
    njit = None


G0_100 = 100.0 / 9.80665
_TINYNUM = 1.0e-14
_MAX_GROUP_TRACER_BYTES = 1024**3
_NUMBA_AVAILABLE = njit is not None
_EMPTY_FLOAT64 = np.empty(0, dtype=np.float64)


@dataclass(frozen=True)
class _ConvectionLightWorkspace:
    tracer_out: np.ndarray
    bmass: np.ndarray


_CONVECTION_LIGHT_WORKSPACE: _ConvectionLightWorkspace | None = None


def _get_convection_light_workspace(
    nlev: int,
    nlat: int,
    nlon: int,
    ntracer: int,
) -> _ConvectionLightWorkspace:
    global _CONVECTION_LIGHT_WORKSPACE
    existing = _CONVECTION_LIGHT_WORKSPACE
    if (
        existing is not None
        and existing.tracer_out.shape == (nlev, nlat, nlon, ntracer)
        and existing.bmass.shape == (nlev, nlat, nlon)
    ):
        return existing
    _CONVECTION_LIGHT_WORKSPACE = _ConvectionLightWorkspace(
        tracer_out=np.empty((nlev, nlat, nlon, ntracer), dtype=np.float64),
        bmass=np.empty((nlev, nlat, nlon), dtype=np.float64),
    )
    return _CONVECTION_LIGHT_WORKSPACE


@dataclass(frozen=True)
class ConvectionResult:
    """One-step GEOS-Chem cloud-convection result in canonical transport order."""

    tracer_conc: np.ndarray
    diag14_mass_flux: np.ndarray
    negative_count_before: int
    negative_count_after: int
    initial_tracer_mass: np.ndarray
    final_tracer_mass: np.ndarray
    internal_steps: int
    internal_dt_s: float


def run_cloud_convection_one_step(
    *,
    tracer_conc: np.ndarray,
    cmfmc_kg_m2_s: np.ndarray,
    dtrain_kg_m2_s: np.ndarray,
    dqrcu_kg_kg_s: np.ndarray,
    reevapcn_kg_kg_s: np.ndarray,
    delp_dry_hpa: np.ndarray,
    delp_hpa: np.ndarray,
    area_m2: np.ndarray,
    bxheight_m: np.ndarray | None = None,
    pficu_kg_m2_s: np.ndarray | None = None,
    pflcu_kg_m2_s: np.ndarray | None = None,
    temperature_k: np.ndarray | None = None,
    precccon_mm_day: np.ndarray | None = None,
    dt_s: float = 600.0,
    reconstruct_conv_precip_flux: bool = False,
    diagnostics: bool = True,
    reuse_output: bool = False,
) -> ConvectionResult:
    """Port GEOS-Chem ``DO_CLOUD_CONVECTION`` for transport-only tracers.

    Arrays use canonical transport order ``(lev_top, lat, lon, tracer)``. Wet
    scavenging is intentionally disabled by using zero soluble fractions, which
    keeps washout arrays as inert plumbing while preserving the long-lived
    tracer mass transport path.
    """

    tracer = np.asarray(tracer_conc, dtype=np.float64)
    cmfmc = np.asarray(cmfmc_kg_m2_s, dtype=np.float64)
    dtrain = np.asarray(dtrain_kg_m2_s, dtype=np.float64)
    dqrcu_met = np.asarray(dqrcu_kg_kg_s, dtype=np.float64)
    reevapcn_met = np.asarray(reevapcn_kg_kg_s, dtype=np.float64)
    delp_dry = np.asarray(delp_dry_hpa, dtype=np.float64)
    delp = np.asarray(delp_hpa, dtype=np.float64)
    area = np.asarray(area_m2, dtype=np.float64)

    if tracer.ndim != 4:
        raise ValueError(f"tracer_conc must be 4-D (lev, lat, lon, tracer), found {tracer.shape}")
    nlev, nlat, nlon, ntracer = tracer.shape
    grid_shape = (nlev, nlat, nlon)
    horizontal_shape = (nlat, nlon)
    for name, value in (
        ("cmfmc_kg_m2_s", cmfmc),
        ("dtrain_kg_m2_s", dtrain),
        ("dqrcu_kg_kg_s", dqrcu_met),
        ("reevapcn_kg_kg_s", reevapcn_met),
        ("delp_dry_hpa", delp_dry),
        ("delp_hpa", delp),
    ):
        if value.shape != grid_shape:
            raise ValueError(f"{name} shape {value.shape} does not match tracer grid {grid_shape}")
    if area.shape != horizontal_shape:
        raise ValueError(f"area_m2 shape {area.shape} does not match horizontal grid {horizontal_shape}")
    for name, value in (
        ("bxheight_m", bxheight_m),
        ("pficu_kg_m2_s", pficu_kg_m2_s),
        ("pflcu_kg_m2_s", pflcu_kg_m2_s),
        ("temperature_k", temperature_k),
    ):
        if value is not None and np.asarray(value).shape != grid_shape:
            raise ValueError(f"{name} shape {np.asarray(value).shape} does not match tracer grid {grid_shape}")
    if precccon_mm_day is not None and np.asarray(precccon_mm_day).shape != horizontal_shape:
        raise ValueError(
            f"precccon_mm_day shape {np.asarray(precccon_mm_day).shape} does not match horizontal grid {horizontal_shape}"
        )
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")

    internal_steps = max(int(dt_s) // 300, 1)
    internal_dt = float(dt_s) / float(internal_steps)
    if diagnostics or not reuse_output:
        bmass = delp_dry * G0_100
        tracer_after_top = np.ascontiguousarray(tracer).copy()
    else:
        workspace = _get_convection_light_workspace(nlev, nlat, nlon, ntracer)
        bmass = workspace.bmass
        tracer_after_top = workspace.tracer_out
        np.multiply(delp_dry, G0_100, out=bmass)
        np.copyto(tracer_after_top, tracer)
    diag14_top = np.zeros_like(tracer_after_top) if diagnostics else _EMPTY_FLOAT64
    initial_mass = _column_mass_transport(tracer_after_top, bmass, area) if diagnostics else _EMPTY_FLOAT64
    negative_before = int(np.count_nonzero(tracer_after_top < 0.0)) if diagnostics else 0

    _convect_active_columns_top(
        tracer_after_top,
        diag14_top,
        cmfmc,
        dtrain,
        dqrcu_met,
        reevapcn_met,
        delp,
        delp_dry,
        bmass,
        area,
        reconstruct_conv_precip_flux=reconstruct_conv_precip_flux,
        diagnostics=diagnostics,
        internal_steps=internal_steps,
        internal_dt_s=internal_dt,
    )

    final_mass = _column_mass_transport(tracer_after_top, bmass, area) if diagnostics else _EMPTY_FLOAT64
    return ConvectionResult(
        tracer_conc=tracer_after_top,
        diag14_mass_flux=diag14_top,
        negative_count_before=negative_before,
        negative_count_after=int(np.count_nonzero(tracer_after_top < 0.0)) if diagnostics else 0,
        initial_tracer_mass=initial_mass,
        final_tracer_mass=final_mass,
        internal_steps=internal_steps,
        internal_dt_s=internal_dt,
    )


def _convect_active_columns_top(
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
    nlev, nlat, nlon, ntracer = tracer.shape
    ncol = nlat * nlon
    q = tracer.reshape(nlev, ncol, -1)
    diag = diag14.reshape(nlev, ncol, -1) if diagnostics else diag14.reshape(0, 0, 0)
    cmf = cmfmc.reshape(nlev, ncol)
    detrain = dtrain.reshape(nlev, ncol)
    delp = delp_hpa.reshape(nlev, ncol)
    delp_d = delp_dry.reshape(nlev, ncol)
    bm = bmass.reshape(nlev, ncol)
    area = area_m2.reshape(ncol)

    if _numba_convection_enabled() and not diagnostics:
        _convect_fullgrid_top_numba(
            q,
            cmf,
            detrain,
            delp,
            delp_d,
            bm,
            dqrcu_met.reshape(nlev, ncol),
            reevapcn_met.reshape(nlev, ncol),
            reconstruct_conv_precip_flux=reconstruct_conv_precip_flux,
            internal_steps=internal_steps,
            internal_dt_s=internal_dt_s,
        )
        return

    active = (np.max(np.abs(cmf), axis=0) > _TINYNUM) | (np.max(np.abs(detrain), axis=0) > _TINYNUM)
    if not np.any(active):
        return

    dqrcu = _convective_precip_rates_columns(
        dqrcu_met.reshape(nlev, ncol),
        reevapcn_met.reshape(nlev, ncol),
        delp,
        reconstruct_conv_precip_flux=reconstruct_conv_precip_flux,
    )
    cloud_base_by_column = _cloud_base_indices_top(dqrcu)
    active_columns = np.flatnonzero(active)

    for cloud_base in np.unique(cloud_base_by_column[active_columns]):
        columns = active_columns[cloud_base_by_column[active_columns] == cloud_base]
        if _numba_convection_enabled():
            _convect_column_group_top_numba(
                q,
                diag,
                cmf,
                detrain,
                delp_d,
                bm,
                area,
                int(cloud_base),
                columns,
                diagnostics=diagnostics,
                internal_steps=internal_steps,
                internal_dt_s=internal_dt_s,
            )
        else:
            max_chunk_columns = _max_convection_group_columns(nlev, ntracer)
            for column_chunk in _iter_column_chunks(columns, max_chunk_columns):
                _convect_column_group_top(
                    q,
                    diag,
                    cmf,
                    detrain,
                    delp_d,
                    bm,
                    area,
                    int(cloud_base),
                    column_chunk,
                    diagnostics=diagnostics,
                    internal_steps=internal_steps,
                    internal_dt_s=internal_dt_s,
                )


def _convective_precip_rates_columns(
    dqrcu_met: np.ndarray,
    reevapcn_met: np.ndarray,
    delp_hpa: np.ndarray,
    *,
    reconstruct_conv_precip_flux: bool,
) -> np.ndarray:
    if not reconstruct_conv_precip_flux:
        return dqrcu_met

    nlev = dqrcu_met.shape[0]
    bottom_index = nlev - 1
    reevapcn = np.zeros_like(reevapcn_met)
    dqrcu = np.zeros_like(dqrcu_met)
    reevapcn[0] = reevapcn_met[0]
    for level in range(1, bottom_index):
        reevapcn[level] = (
            reevapcn_met[level] * delp_hpa[level] - reevapcn_met[level - 1] * delp_hpa[level - 1]
        ) / delp_hpa[level]
    dqrcu[:bottom_index] = dqrcu_met[:bottom_index] + reevapcn[:bottom_index]
    return dqrcu


def _cloud_base_indices_top(dqrcu: np.ndarray) -> np.ndarray:
    return dqrcu.shape[0] - 1 - np.argmax(dqrcu[::-1] > 0.0, axis=0)


def _max_convection_group_columns(nlev: int, ntracer: int) -> int:
    bytes_per_column = max(nlev, 1) * max(ntracer, 1) * np.dtype(np.float64).itemsize
    return max(int(_MAX_GROUP_TRACER_BYTES // bytes_per_column), 1)


def _iter_column_chunks(columns: np.ndarray, max_columns: int):
    if max_columns <= 0:
        raise ValueError("max_columns must be positive")
    for start in range(0, columns.size, max_columns):
        yield columns[start : start + max_columns]


def _convect_column_group_top(
    q_all: np.ndarray,
    diag_all: np.ndarray,
    cmfmc_all: np.ndarray,
    dtrain_all: np.ndarray,
    delp_dry_all: np.ndarray,
    bmass_all: np.ndarray,
    area_all: np.ndarray,
    cloud_base: int,
    columns: np.ndarray,
    *,
    diagnostics: bool,
    internal_steps: int,
    internal_dt_s: float,
) -> None:
    q = q_all[:, columns, :].copy()
    diag = np.zeros_like(q) if diagnostics else np.empty((0,), dtype=np.float64)
    cmfmc = cmfmc_all[:, columns]
    dtrain = dtrain_all[:, columns]
    delp_dry = delp_dry_all[:, columns]
    bmass = bmass_all[:, columns]
    area = area_all[columns]
    nlev = q.shape[0]
    ncol = q.shape[1]
    ntracer = q.shape[2]
    dns = float(internal_steps)
    bottom_index = nlev - 1
    flux_work = np.empty((ncol, ntracer), dtype=np.float64)
    temp_work = np.empty((ncol, ntracer), dtype=np.float64)
    next_work = np.empty((ncol, ntracer), dtype=np.float64)

    for _step in range(internal_steps):
        if cloud_base < bottom_index:
            cmfmc_below_base = cmfmc[cloud_base + 1]
            below_base_active = cmfmc_below_base > _TINYNUM
            qc = q[cloud_base].copy()
            if np.any(below_base_active):
                denominator = np.sum(delp_dry[cloud_base + 1 :, below_base_active], axis=0)
                if np.any(denominator <= 0.0):
                    raise ValueError("dry pressure below cloud base must be positive")
                qb = (
                    np.sum(q[cloud_base + 1 :, below_base_active, :] * delp_dry[cloud_base + 1 :, below_base_active, np.newaxis], axis=0)
                    / denominator[:, np.newaxis]
                )
                mass_below_base = np.sum(bmass[cloud_base + 1 :, below_base_active], axis=0)
                cmfmc_base = cmfmc_below_base[below_base_active]
                qc[below_base_active] = (
                    mass_below_base[:, np.newaxis] * qb
                    + cmfmc_base[:, np.newaxis] * q[cloud_base, below_base_active, :] * internal_dt_s
                ) / (mass_below_base + cmfmc_base * internal_dt_s)[:, np.newaxis]
                q[cloud_base + 1 :, below_base_active, :] = qc[below_base_active][np.newaxis, :, :]
        else:
            qc = q[cloud_base].copy()

        for level in range(cloud_base, 0, -1):
            cmfmc_below = np.zeros(columns.size, dtype=np.float64) if level == bottom_index else cmfmc[level + 1]
            has_below_flux = cmfmc_below > _TINYNUM
            if np.any(has_below_flux):
                local = np.flatnonzero(has_below_flux)
                local_count = local.size
                cmout = cmfmc[level, local] + dtrain[level, local]
                entrn = cmout - cmfmc_below[local]
                qc_pres = qc[local].copy()
                qc_next = next_work[:local_count, :]
                temp = temp_work[:local_count, :]
                qc_next[:] = qc_pres
                entrains = (entrn >= 0.0) & (cmout > 0.0)
                if np.any(entrains):
                    entrain_local = local[entrains]
                    qc_next[entrains] = (
                        cmfmc_below[entrain_local, np.newaxis] * qc_pres[entrains]
                        + entrn[entrains, np.newaxis] * q[level, entrain_local, :]
                    ) / cmout[entrains, np.newaxis]

                delq = flux_work[:local_count, :]
                np.multiply(cmfmc_below[local, np.newaxis], qc_pres, out=delq)
                np.multiply(cmfmc[level, local, np.newaxis], qc_next, out=temp)
                np.negative(temp, out=temp)
                delq += temp
                qc[local] = qc_next

                np.multiply(cmfmc[level, local, np.newaxis], q[level - 1, local, :], out=qc_next)
                delq += qc_next
                if diagnostics:
                    np.negative(temp, out=temp)
                    temp -= qc_next
                    temp *= area[local, np.newaxis] / dns
                    diag[level, local, :] += temp

                np.multiply(cmfmc_below[local, np.newaxis], q[level, local, :], out=qc_next)
                delq -= qc_next
                delq *= internal_dt_s / bmass[level, local, np.newaxis]
                current = q[level, local, :]
                np.add(current, delq, out=qc_next)
                negative = qc_next < 0.0
                if np.any(negative):
                    delq[negative] = -current[negative]
                np.add(current, delq, out=current)
                q[level, local, :] = current

            no_below_flux = ~has_below_flux
            if np.any(no_below_flux):
                local = np.flatnonzero(no_below_flux)
                qc[local] = q[level, local, :]
                has_current_flux = cmfmc[level, local] > _TINYNUM
                if np.any(has_current_flux):
                    flux_local = local[has_current_flux]
                    flux_count = flux_local.size
                    delq = flux_work[:flux_count, :]
                    temp = temp_work[:flux_count, :]
                    np.multiply(cmfmc[level, flux_local, np.newaxis], qc[flux_local], out=delq)
                    np.negative(delq, out=delq)
                    np.multiply(
                        cmfmc[level, flux_local, np.newaxis],
                        q[level - 1, flux_local, :],
                        out=temp,
                    )
                    delq += temp
                    delq *= internal_dt_s / bmass[level, flux_local, np.newaxis]
                    current = q[level, flux_local, :]
                    np.add(current, delq, out=temp)
                    negative = temp < 0.0
                    if np.any(negative):
                        delq[negative] = -current[negative]
                    np.add(current, delq, out=current)
                    q[level, flux_local, :] = current

    q_all[:, columns, :] = q
    if diagnostics:
        diag_all[:, columns, :] = diag


def _numba_convection_mode() -> str:
    return os.environ.get("WOMBAT_CONVECTION_NUMBA", "1").lower()


def _numba_convection_enabled() -> bool:
    if not _NUMBA_AVAILABLE:
        return False
    return _numba_convection_mode() not in {"0", "false", "no", "off", "none"}


def _convect_fullgrid_top_numba(
    q_all: np.ndarray,
    cmfmc_all: np.ndarray,
    dtrain_all: np.ndarray,
    delp_hpa_all: np.ndarray,
    delp_dry_all: np.ndarray,
    bmass_all: np.ndarray,
    dqrcu_met_all: np.ndarray,
    reevapcn_met_all: np.ndarray,
    *,
    reconstruct_conv_precip_flux: bool,
    internal_steps: int,
    internal_dt_s: float,
) -> None:
    if not _NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available")
    _convect_fullgrid_top_numba_kernel(
        q_all,
        cmfmc_all,
        dtrain_all,
        delp_hpa_all,
        delp_dry_all,
        bmass_all,
        dqrcu_met_all,
        reevapcn_met_all,
        reconstruct_conv_precip_flux,
        internal_steps,
        internal_dt_s,
    )


def _convect_column_group_top_numba(
    q_all: np.ndarray,
    diag_all: np.ndarray,
    cmfmc_all: np.ndarray,
    dtrain_all: np.ndarray,
    delp_dry_all: np.ndarray,
    bmass_all: np.ndarray,
    area_all: np.ndarray,
    cloud_base: int,
    columns: np.ndarray,
    *,
    diagnostics: bool,
    internal_steps: int,
    internal_dt_s: float,
) -> None:
    if not _NUMBA_AVAILABLE:
        _convect_column_group_top(
            q_all,
            diag_all,
            cmfmc_all,
            dtrain_all,
            delp_dry_all,
            bmass_all,
            area_all,
            cloud_base,
            columns,
            diagnostics=diagnostics,
            internal_steps=internal_steps,
            internal_dt_s=internal_dt_s,
        )
        return
    _convect_column_group_top_numba_kernel(
        q_all,
        diag_all,
        cmfmc_all,
        dtrain_all,
        delp_dry_all,
        bmass_all,
        area_all,
        cloud_base,
        columns,
        diagnostics,
        internal_steps,
        internal_dt_s,
    )


if njit is not None:

    @njit(cache=True)
    def _convect_fullgrid_top_numba_kernel(
        q_all: np.ndarray,
        cmfmc_all: np.ndarray,
        dtrain_all: np.ndarray,
        delp_hpa_all: np.ndarray,
        delp_dry_all: np.ndarray,
        bmass_all: np.ndarray,
        dqrcu_met_all: np.ndarray,
        reevapcn_met_all: np.ndarray,
        reconstruct_conv_precip_flux: bool,
        internal_steps: int,
        internal_dt_s: float,
    ) -> None:
        nlev = q_all.shape[0]
        ncol = q_all.shape[1]
        ntracer = q_all.shape[2]
        bottom_index = nlev - 1
        qc = np.empty(ntracer, dtype=np.float64)
        qb_num = np.empty(ntracer, dtype=np.float64)

        for col in range(ncol):
            active = False
            for level in range(nlev):
                cmfmc_value = cmfmc_all[level, col]
                dtrain_value = dtrain_all[level, col]
                if (
                    cmfmc_value > _TINYNUM
                    or cmfmc_value < -_TINYNUM
                    or dtrain_value > _TINYNUM
                    or dtrain_value < -_TINYNUM
                ):
                    active = True
                    break
            if not active:
                continue

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

            for step in range(internal_steps):
                _ = step
                for tracer in range(ntracer):
                    qc[tracer] = q_all[cloud_base, col, tracer]

                if cloud_base < bottom_index and cmfmc_all[cloud_base + 1, col] > _TINYNUM:
                    denominator = 0.0
                    mass_below_base = 0.0
                    for level in range(cloud_base + 1, nlev):
                        denominator += delp_dry_all[level, col]
                        mass_below_base += bmass_all[level, col]
                    if denominator <= 0.0:
                        raise ValueError("dry pressure below cloud base must be positive")

                    cmfmc_base = cmfmc_all[cloud_base + 1, col]
                    denom_qc = mass_below_base + cmfmc_base * internal_dt_s
                    for tracer in range(ntracer):
                        qb_num[tracer] = 0.0
                    for level in range(cloud_base + 1, nlev):
                        delp_dry = delp_dry_all[level, col]
                        for tracer in range(ntracer):
                            qb_num[tracer] += q_all[level, col, tracer] * delp_dry
                    for tracer in range(ntracer):
                        qb = qb_num[tracer] / denominator
                        plume = (
                            mass_below_base * qb
                            + cmfmc_base * q_all[cloud_base, col, tracer] * internal_dt_s
                        ) / denom_qc
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
                        cmout = cmfmc_all[level, col] + dtrain_all[level, col]
                        entrn = cmout - cmfmc_below
                        entrains = entrn >= 0.0 and cmout > 0.0
                        tendency_scale = internal_dt_s / bmass_all[level, col]

                        for tracer in range(ntracer):
                            qc_pres = qc[tracer]
                            qc_next = qc_pres
                            if entrains:
                                qc_next = (
                                    cmfmc_below * qc_pres + entrn * q_all[level, col, tracer]
                                ) / cmout

                            delq = cmfmc_below * qc_pres
                            temp = -(cmfmc_all[level, col] * qc_next)
                            delq += temp
                            qc[tracer] = qc_next

                            upward = cmfmc_all[level, col] * q_all[level - 1, col, tracer]
                            delq += upward
                            delq -= cmfmc_below * q_all[level, col, tracer]
                            delq *= tendency_scale
                            current = q_all[level, col, tracer]
                            if current + delq < 0.0:
                                delq = -current
                            q_all[level, col, tracer] = current + delq
                    else:
                        has_current_flux = cmfmc_all[level, col] > _TINYNUM
                        tendency_scale = internal_dt_s / bmass_all[level, col]
                        for tracer in range(ntracer):
                            qc[tracer] = q_all[level, col, tracer]
                            if has_current_flux:
                                delq = -(cmfmc_all[level, col] * qc[tracer])
                                delq += cmfmc_all[level, col] * q_all[level - 1, col, tracer]
                                delq *= tendency_scale
                                current = q_all[level, col, tracer]
                                if current + delq < 0.0:
                                    delq = -current
                                q_all[level, col, tracer] = current + delq

    @njit(cache=True)
    def _convect_column_group_top_numba_kernel(
        q_all: np.ndarray,
        diag_all: np.ndarray,
        cmfmc_all: np.ndarray,
        dtrain_all: np.ndarray,
        delp_dry_all: np.ndarray,
        bmass_all: np.ndarray,
        area_all: np.ndarray,
        cloud_base: int,
        columns: np.ndarray,
        diagnostics: bool,
        internal_steps: int,
        internal_dt_s: float,
    ) -> None:
        nlev = q_all.shape[0]
        ntracer = q_all.shape[2]
        ncolumns = columns.size
        bottom_index = nlev - 1
        dns = float(internal_steps)
        qc = np.empty(ntracer, dtype=np.float64)
        qb_num = np.empty(ntracer, dtype=np.float64)

        for step in range(internal_steps):
            _ = step
            for column_index in range(ncolumns):
                col = columns[column_index]

                for tracer in range(ntracer):
                    qc[tracer] = q_all[cloud_base, col, tracer]

                if cloud_base < bottom_index and cmfmc_all[cloud_base + 1, col] > _TINYNUM:
                    denominator = 0.0
                    mass_below_base = 0.0
                    for level in range(cloud_base + 1, nlev):
                        denominator += delp_dry_all[level, col]
                        mass_below_base += bmass_all[level, col]
                    if denominator <= 0.0:
                        raise ValueError("dry pressure below cloud base must be positive")

                    cmfmc_base = cmfmc_all[cloud_base + 1, col]
                    denom_qc = mass_below_base + cmfmc_base * internal_dt_s
                    for tracer in range(ntracer):
                        qb_num[tracer] = 0.0
                    for level in range(cloud_base + 1, nlev):
                        delp_dry = delp_dry_all[level, col]
                        for tracer in range(ntracer):
                            qb_num[tracer] += q_all[level, col, tracer] * delp_dry
                    for tracer in range(ntracer):
                        qb = qb_num[tracer] / denominator
                        plume = (
                            mass_below_base * qb
                            + cmfmc_base * q_all[cloud_base, col, tracer] * internal_dt_s
                        ) / denom_qc
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
                        cmout = cmfmc_all[level, col] + dtrain_all[level, col]
                        entrn = cmout - cmfmc_below
                        entrains = entrn >= 0.0 and cmout > 0.0
                        area_scale = area_all[col] / dns
                        tendency_scale = internal_dt_s / bmass_all[level, col]

                        for tracer in range(ntracer):
                            qc_pres = qc[tracer]
                            qc_next = qc_pres
                            if entrains:
                                qc_next = (
                                    cmfmc_below * qc_pres + entrn * q_all[level, col, tracer]
                                ) / cmout

                            delq = cmfmc_below * qc_pres
                            temp = -(cmfmc_all[level, col] * qc_next)
                            delq += temp
                            qc[tracer] = qc_next

                            upward = cmfmc_all[level, col] * q_all[level - 1, col, tracer]
                            delq += upward
                            if diagnostics:
                                diag_all[level, col, tracer] += (-temp - upward) * area_scale

                            delq -= cmfmc_below * q_all[level, col, tracer]
                            delq *= tendency_scale
                            current = q_all[level, col, tracer]
                            if current + delq < 0.0:
                                delq = -current
                            q_all[level, col, tracer] = current + delq
                    else:
                        has_current_flux = cmfmc_all[level, col] > _TINYNUM
                        tendency_scale = internal_dt_s / bmass_all[level, col]
                        for tracer in range(ntracer):
                            qc[tracer] = q_all[level, col, tracer]
                            if has_current_flux:
                                delq = -(cmfmc_all[level, col] * qc[tracer])
                                delq += cmfmc_all[level, col] * q_all[level - 1, col, tracer]
                                delq *= tendency_scale
                                current = q_all[level, col, tracer]
                                if current + delq < 0.0:
                                    delq = -current
                                q_all[level, col, tracer] = current + delq

else:

    def _convect_fullgrid_top_numba_kernel(
        q_all: np.ndarray,
        cmfmc_all: np.ndarray,
        dtrain_all: np.ndarray,
        delp_hpa_all: np.ndarray,
        delp_dry_all: np.ndarray,
        bmass_all: np.ndarray,
        dqrcu_met_all: np.ndarray,
        reevapcn_met_all: np.ndarray,
        reconstruct_conv_precip_flux: bool,
        internal_steps: int,
        internal_dt_s: float,
    ) -> None:
        raise RuntimeError("numba is not available")

    def _convect_column_group_top_numba_kernel(
        q_all: np.ndarray,
        diag_all: np.ndarray,
        cmfmc_all: np.ndarray,
        dtrain_all: np.ndarray,
        delp_dry_all: np.ndarray,
        bmass_all: np.ndarray,
        area_all: np.ndarray,
        cloud_base: int,
        columns: np.ndarray,
        diagnostics: bool,
        internal_steps: int,
        internal_dt_s: float,
    ) -> None:
        raise RuntimeError("numba is not available")


def _column_mass_transport(tracer: np.ndarray, bmass_kg_m2: np.ndarray, area_m2: np.ndarray) -> np.ndarray:
    if _NUMBA_AVAILABLE and _numba_convection_enabled():
        return _column_mass_transport_numba(tracer, bmass_kg_m2, area_m2)
    return np.sum(tracer * bmass_kg_m2[:, :, :, np.newaxis] * area_m2[np.newaxis, :, :, np.newaxis], axis=(0, 1, 2))


def _column_mass_transport_numba(tracer: np.ndarray, bmass_kg_m2: np.ndarray, area_m2: np.ndarray) -> np.ndarray:
    if not _NUMBA_AVAILABLE:
        return np.sum(
            tracer * bmass_kg_m2[:, :, :, np.newaxis] * area_m2[np.newaxis, :, :, np.newaxis],
            axis=(0, 1, 2),
        )
    return _column_mass_transport_numba_kernel(tracer, bmass_kg_m2, area_m2)


if njit is not None:

    @njit(cache=True)
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
