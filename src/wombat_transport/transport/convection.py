from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G0_100 = 100.0 / 9.80665
_TINYNUM = 1.0e-14


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
    nlev, nlat, nlon, _ntracer = tracer.shape
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
    bmass = delp_dry * G0_100
    tracer_after_top = np.ascontiguousarray(tracer).copy()
    diag14_top = np.zeros_like(tracer_after_top)
    initial_mass = _column_mass_transport(tracer_after_top, bmass, area)
    negative_before = int(np.count_nonzero(tracer_after_top < 0.0))

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
        internal_steps=internal_steps,
        internal_dt_s=internal_dt,
    )

    final_mass = _column_mass_transport(tracer_after_top, bmass, area)
    return ConvectionResult(
        tracer_conc=tracer_after_top,
        diag14_mass_flux=diag14_top,
        negative_count_before=negative_before,
        negative_count_after=int(np.count_nonzero(tracer_after_top < 0.0)),
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
    internal_steps: int,
    internal_dt_s: float,
) -> None:
    nlev, nlat, nlon, _ntracer = tracer.shape
    ncol = nlat * nlon
    q = tracer.reshape(nlev, ncol, -1)
    diag = diag14.reshape(nlev, ncol, -1)
    cmf = cmfmc.reshape(nlev, ncol)
    detrain = dtrain.reshape(nlev, ncol)
    delp = delp_hpa.reshape(nlev, ncol)
    delp_d = delp_dry.reshape(nlev, ncol)
    bm = bmass.reshape(nlev, ncol)
    area = area_m2.reshape(ncol)

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
        _convect_column_group_top(
            q,
            diag,
            cmf,
            detrain,
            delp_d,
            bm,
            area,
            int(cloud_base),
            columns,
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
    internal_steps: int,
    internal_dt_s: float,
) -> None:
    q = q_all[:, columns, :].copy()
    diag = np.zeros_like(q)
    cmfmc = cmfmc_all[:, columns]
    dtrain = dtrain_all[:, columns]
    delp_dry = delp_dry_all[:, columns]
    bmass = bmass_all[:, columns]
    area = area_all[columns]
    nlev = q.shape[0]
    dns = float(internal_steps)
    bottom_index = nlev - 1

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
                cmout = cmfmc[level, local] + dtrain[level, local]
                entrn = cmout - cmfmc_below[local]
                qc_pres = qc[local].copy()
                qc_next = qc_pres.copy()
                entrains = (entrn >= 0.0) & (cmout > 0.0)
                if np.any(entrains):
                    entrain_local = local[entrains]
                    qc_next[entrains] = (
                        cmfmc_below[entrain_local, np.newaxis] * qc_pres[entrains]
                        + entrn[entrains, np.newaxis] * q[level, entrain_local, :]
                    ) / cmout[entrains, np.newaxis]

                t1 = cmfmc_below[local, np.newaxis] * qc_pres
                t2 = -cmfmc[level, local, np.newaxis] * qc_next
                t3 = cmfmc[level, local, np.newaxis] * q[level - 1, local, :]
                t4 = -cmfmc_below[local, np.newaxis] * q[level, local, :]
                delq = (internal_dt_s / bmass[level, local, np.newaxis]) * (t1 + t2 + t3 + t4)
                current = q[level, local, :]
                delq = np.where(current + delq < 0.0, -current, delq)
                q[level, local, :] = current + delq
                diag[level, local, :] += (-t2 - t3) * area[local, np.newaxis] / dns
                qc[local] = qc_next

            no_below_flux = ~has_below_flux
            if np.any(no_below_flux):
                local = np.flatnonzero(no_below_flux)
                qc[local] = q[level, local, :].copy()
                has_current_flux = cmfmc[level, local] > _TINYNUM
                if np.any(has_current_flux):
                    flux_local = local[has_current_flux]
                    t2 = -cmfmc[level, flux_local, np.newaxis] * qc[flux_local]
                    t3 = cmfmc[level, flux_local, np.newaxis] * q[level - 1, flux_local, :]
                    delq = (internal_dt_s / bmass[level, flux_local, np.newaxis]) * (t2 + t3)
                    current = q[level, flux_local, :]
                    delq = np.where(current + delq < 0.0, -current, delq)
                    q[level, flux_local, :] = current + delq

    q_all[:, columns, :] = q
    diag_all[:, columns, :] = diag


def _column_mass_transport(tracer: np.ndarray, bmass_kg_m2: np.ndarray, area_m2: np.ndarray) -> np.ndarray:
    return np.sum(tracer * bmass_kg_m2[:, :, :, np.newaxis] * area_m2[np.newaxis, :, :, np.newaxis], axis=(0, 1, 2))
