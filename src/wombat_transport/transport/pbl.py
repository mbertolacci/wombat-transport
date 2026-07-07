from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G0_M_PER_S2 = 9.80665
RD_J_PER_KG_K = 287.0


@dataclass(frozen=True)
class PblHeightState:
    """GEOS-Chem PBL-top bookkeeping in bottom-to-top vertical order."""

    pbl_top_m: np.ndarray
    pbl_top_hpa: np.ndarray
    pbl_top_l: np.ndarray
    pbl_thick_hpa: np.ndarray
    in_pbl: np.ndarray
    f_of_pbl: np.ndarray
    f_under_pbl_top: np.ndarray
    pbl_max_l: int


def compute_pbl_height(
    *,
    pbl_height_m: np.ndarray,
    bxheight_m: np.ndarray,
    pressure_edges_hpa: np.ndarray,
    virtual_temperature_k: np.ndarray,
) -> PblHeightState:
    """Port GEOS-Chem ``Compute_Pbl_Height`` for the fixed-grid array layout.

    Arrays use Wombat/NetCDF order with vertical level 0 nearest the surface.
    ``pressure_edges_hpa`` has one more vertical edge than ``bxheight_m``.
    """

    pbl_height = np.asarray(pbl_height_m, dtype=np.float64)
    bxheight = np.asarray(bxheight_m, dtype=np.float64)
    pedge = np.asarray(pressure_edges_hpa, dtype=np.float64)
    tv = np.asarray(virtual_temperature_k, dtype=np.float64)
    if bxheight.ndim != 3:
        raise ValueError(f"bxheight_m must be 3-D (lev, lat, lon), found {bxheight.shape}")
    if tv.shape != bxheight.shape:
        raise ValueError(f"virtual_temperature_k shape {tv.shape} does not match bxheight_m {bxheight.shape}")
    if pedge.shape != (bxheight.shape[0] + 1, bxheight.shape[1], bxheight.shape[2]):
        raise ValueError(f"pressure_edges_hpa shape {pedge.shape} is incompatible with bxheight_m {bxheight.shape}")
    if pbl_height.shape != bxheight.shape[1:]:
        raise ValueError(f"pbl_height_m shape {pbl_height.shape} does not match horizontal grid {bxheight.shape[1:]}")
    if np.any(pbl_height <= 0.0):
        raise ValueError("pbl_height_m must be positive")

    nlev, nlat, nlon = bxheight.shape
    in_pbl = np.zeros((nlev, nlat, nlon), dtype=bool)
    f_of_pbl = np.zeros((nlev, nlat, nlon), dtype=np.float64)
    f_under = np.zeros((nlev, nlat, nlon), dtype=np.float64)
    pbl_top_hpa = np.empty((nlat, nlon), dtype=np.float64)
    pbl_top_l = np.empty((nlat, nlon), dtype=np.float64)
    pbl_thick = np.empty((nlat, nlon), dtype=np.float64)

    for j in range(nlat):
        for i in range(nlon):
            lower_edge_height = 0.0
            found = False
            for lev in range(nlev):
                next_edge_height = lower_edge_height + bxheight[lev, j, i]
                if next_edge_height >= pbl_height[j, i]:
                    top_hpa = pedge[lev, j, i] * np.exp(
                        -(pbl_height[j, i] - lower_edge_height) * G0_M_PER_S2 / (RD_J_PER_KG_K * tv[lev, j, i])
                    )
                    layer_thick = pedge[lev, j, i] - pedge[lev + 1, j, i]
                    pbl_mass_thick = pedge[lev, j, i] - top_hpa
                    pbl_top_hpa[j, i] = top_hpa
                    pbl_thick[j, i] = pedge[0, j, i] - top_hpa
                    f_of_pbl[lev, j, i] = pbl_mass_thick
                    f_under[lev, j, i] = pbl_mass_thick / layer_thick
                    pbl_top_l[j, i] = float(lev) + f_under[lev, j, i]
                    found = True
                    break

                in_pbl[lev, j, i] = True
                f_under[lev, j, i] = 1.0
                f_of_pbl[lev, j, i] = pedge[lev, j, i] - pedge[lev + 1, j, i]
                lower_edge_height = next_edge_height

            if not found:
                raise ValueError(f"PBL height {pbl_height[j, i]} m exceeds modeled column height at lat/lon index {j}/{i}")
            f_of_pbl[:, j, i] /= pbl_thick[j, i]

    sums = np.sum(f_of_pbl, axis=0)
    if np.any(np.abs(sums - 1.0) > 1.0e-3):
        raise ValueError("computed F_of_PBL does not sum to 1 within GEOS-Chem tolerance")

    return PblHeightState(
        pbl_top_m=pbl_height.copy(),
        pbl_top_hpa=pbl_top_hpa,
        pbl_top_l=pbl_top_l,
        pbl_thick_hpa=pbl_thick,
        in_pbl=in_pbl,
        f_of_pbl=f_of_pbl,
        f_under_pbl_top=f_under,
        pbl_max_l=int(np.max(np.ceil(pbl_top_l))),
    )


def mix_full_pbl(
    tracer_conc: np.ndarray,
    dry_air_mass_kg: np.ndarray,
    pbl_top_l: np.ndarray,
) -> np.ndarray:
    """Port the compact mass-weighted mixing core used by GEOS-Chem ``TurbDay``.

    This is the full-PBL mixer, not the configured non-local VDIFF scheme. It is
    useful as the first PBL bookkeeping/mass-conservation target and as a small
    oracle surface before porting the larger non-local path.
    """

    tracer = np.asarray(tracer_conc, dtype=np.float64)
    dry_mass = np.asarray(dry_air_mass_kg, dtype=np.float64)
    top_l = np.asarray(pbl_top_l, dtype=np.float64)
    if tracer.ndim != 4:
        raise ValueError(f"tracer_conc must be 4-D (tracer, lev, lat, lon), found {tracer.shape}")
    if dry_mass.shape != tracer.shape[1:]:
        raise ValueError(f"dry_air_mass_kg shape {dry_mass.shape} does not match tracer grid {tracer.shape[1:]}")
    if top_l.shape != tracer.shape[2:]:
        raise ValueError(f"pbl_top_l shape {top_l.shape} does not match horizontal grid {tracer.shape[2:]}")

    mixed = tracer.copy()
    ntracer, nlev, nlat, nlon = tracer.shape
    for j in range(nlat):
        for i in range(nlon):
            imix = int(np.ceil(top_l[j, i]))
            if imix < 1 or imix > nlev:
                raise ValueError(f"pbl_top_l at lat/lon index {j}/{i} is outside model levels: {top_l[j, i]}")
            top_index = imix - 1
            fpbl = top_l[j, i] - float(imix - 1)
            if fpbl <= 0.0 or fpbl > 1.0:
                raise ValueError(f"invalid fractional PBL top at lat/lon index {j}/{i}: {top_l[j, i]}")

            full_mass = dry_mass[:top_index, j, i]
            top_mass = dry_mass[top_index, j, i] * fpbl
            air_mass = np.sum(full_mass) + top_mass
            if air_mass <= 0.0:
                raise ValueError(f"non-positive PBL air mass at lat/lon index {j}/{i}")

            tracer_mass = np.sum(tracer[:, :top_index, j, i] * full_mass[np.newaxis, :], axis=1)
            tracer_mass = tracer_mass + tracer[:, top_index, j, i] * top_mass
            mean = tracer_mass / air_mass
            if top_index > 0:
                mixed[:, :top_index, j, i] = mean[:, np.newaxis]
            mixed[:, top_index, j, i] = tracer[:, top_index, j, i] + fpbl * (mean - tracer[:, top_index, j, i])

    return mixed
