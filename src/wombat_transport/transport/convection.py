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

    for lat_index in range(nlat):
        for lon_index in range(nlon):
            if not np.any(np.abs(cmfmc[:, lat_index, lon_index]) > _TINYNUM) and not np.any(
                np.abs(dtrain[:, lat_index, lon_index]) > _TINYNUM
            ):
                continue
            dqrcu, reevapcn = _convective_precip_rates_top(
                dqrcu_met[:, lat_index, lon_index],
                reevapcn_met[:, lat_index, lon_index],
                delp[:, lat_index, lon_index],
                reconstruct_conv_precip_flux=reconstruct_conv_precip_flux,
            )
            cloud_base = _cloud_base_index_top(dqrcu)
            mass_below_base = float(np.sum(bmass[cloud_base + 1 :, lat_index, lon_index]))
            column, diag_column = _convect_column_top(
                tracer_after_top[:, lat_index, lon_index, :],
                cmfmc[:, lat_index, lon_index],
                dtrain[:, lat_index, lon_index],
                delp_dry[:, lat_index, lon_index],
                bmass[:, lat_index, lon_index],
                area[lat_index, lon_index],
                cloud_base=cloud_base,
                mass_below_base=mass_below_base,
                internal_steps=internal_steps,
                internal_dt_s=internal_dt,
            )
            tracer_after_top[:, lat_index, lon_index, :] = column
            diag14_top[:, lat_index, lon_index, :] = diag_column
            _ = reevapcn

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


def _convective_precip_rates_top(
    dqrcu_met: np.ndarray,
    reevapcn_met: np.ndarray,
    delp_hpa: np.ndarray,
    *,
    reconstruct_conv_precip_flux: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not reconstruct_conv_precip_flux:
        return dqrcu_met.copy(), reevapcn_met.copy()

    nlev = dqrcu_met.size
    bottom_index = nlev - 1
    dqrcu = np.zeros(nlev, dtype=np.float64)
    reevapcn = np.zeros(nlev, dtype=np.float64)
    reevapcn[0] = reevapcn_met[0]
    for level in range(1, bottom_index):
        reevapcn[level] = (
            reevapcn_met[level] * delp_hpa[level] - reevapcn_met[level - 1] * delp_hpa[level - 1]
        ) / delp_hpa[level]
    reevapcn[bottom_index] = 0.0
    dqrcu[:bottom_index] = dqrcu_met[:bottom_index] + reevapcn[:bottom_index]
    dqrcu[bottom_index] = 0.0
    return dqrcu, reevapcn


def _cloud_base_index_top(dqrcu: np.ndarray) -> int:
    found = np.flatnonzero(dqrcu[::-1] > 0.0)
    if found.size == 0:
        return dqrcu.size - 1
    return int(dqrcu.size - 1 - found[0])


def _convect_column_top(
    q_in: np.ndarray,
    cmfmc: np.ndarray,
    dtrain: np.ndarray,
    delp_dry: np.ndarray,
    bmass: np.ndarray,
    area_m2: float,
    *,
    cloud_base: int,
    mass_below_base: float,
    internal_steps: int,
    internal_dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    q = q_in.copy()
    nlev, _ntracer = q.shape
    diag14 = np.zeros_like(q)
    dns = float(internal_steps)
    bottom_index = nlev - 1

    for _step in range(internal_steps):
        if cloud_base < bottom_index:
            cmfmc_below_base = cmfmc[cloud_base + 1]
            if cmfmc_below_base > _TINYNUM:
                denominator = np.sum(delp_dry[cloud_base + 1 :])
                if denominator <= 0.0:
                    raise ValueError("dry pressure below cloud base must be positive")
                qb = np.sum(q[cloud_base + 1 :, :] * delp_dry[cloud_base + 1 :, np.newaxis], axis=0) / denominator
                qc = (mass_below_base * qb + cmfmc_below_base * q[cloud_base, :] * internal_dt_s) / (
                    mass_below_base + cmfmc_below_base * internal_dt_s
                )
                q[cloud_base + 1 :, :] = qc[np.newaxis, :]
            else:
                qc = q[cloud_base, :].copy()
        else:
            qc = q[cloud_base, :].copy()

        for level in range(cloud_base, 0, -1):
            cmfmc_below = 0.0 if level == bottom_index else cmfmc[level + 1]
            if cmfmc_below > _TINYNUM:
                cmout = cmfmc[level] + dtrain[level]
                entrn = cmout - cmfmc_below
                qc_pres = qc.copy()
                qc_scav = 0.0
                if entrn >= 0.0 and cmout > 0.0:
                    qc = (cmfmc_below * qc_pres + entrn * q[level, :]) / cmout

                t1 = cmfmc_below * qc_pres
                t2 = -cmfmc[level] * qc
                t3 = cmfmc[level] * q[level - 1, :]
                t4 = -cmfmc_below * q[level, :]
                delq = (internal_dt_s / bmass[level]) * (t1 + t2 + t3 + t4)
                delq = np.where(q[level, :] + delq < 0.0, -q[level, :], delq)
                q[level, :] = q[level, :] + delq
                diag14[level, :] += (-t2 - t3) * area_m2 / dns
                _ = qc_scav
            else:
                qc = q[level, :].copy()
                if cmfmc[level] > _TINYNUM:
                    t2 = -cmfmc[level] * qc
                    t3 = cmfmc[level] * q[level - 1, :]
                    delq = (internal_dt_s / bmass[level]) * (t2 + t3)
                    delq = np.where(q[level, :] + delq < 0.0, -q[level, :], delq)
                    q[level, :] = q[level, :] + delq
    return q, diag14


def _column_mass_transport(tracer: np.ndarray, bmass_kg_m2: np.ndarray, area_m2: np.ndarray) -> np.ndarray:
    return np.sum(tracer * bmass_kg_m2[:, :, :, np.newaxis] * area_m2[np.newaxis, :, :, np.newaxis], axis=(0, 1, 2))
