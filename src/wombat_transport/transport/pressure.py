from __future__ import annotations

import numpy as np

from wombat_transport.constants import G0_M_PER_S2

def pressure_edges_hpa(surface_pressure_pa: np.ndarray, hyai_hpa: np.ndarray, hybi: np.ndarray) -> np.ndarray:
    """Compute hybrid pressure edges in hPa from surface pressure in Pa."""

    return pressure_edges_from_surface_hpa(np.asarray(surface_pressure_pa, dtype=np.float64) / 100.0, hyai_hpa, hybi)

def pressure_edges_from_surface_hpa(surface_pressure_hpa: np.ndarray, hyai_hpa: np.ndarray, hybi: np.ndarray) -> np.ndarray:
    """Compute hybrid pressure edges in hPa from surface pressure in hPa."""

    ps_hpa = np.asarray(surface_pressure_hpa, dtype=np.float64)
    hyai = np.asarray(hyai_hpa, dtype=np.float64)
    hybi = np.asarray(hybi, dtype=np.float64)
    return hyai[np.newaxis, :, np.newaxis, np.newaxis] + (
        hybi[np.newaxis, :, np.newaxis, np.newaxis] * ps_hpa[:, np.newaxis, :, :]
    )

def dry_pressure_thickness_hpa(surface_pressure_pa: np.ndarray, hyai_hpa: np.ndarray, hybi: np.ndarray) -> np.ndarray:
    """Return positive pressure thickness on the prototype level order."""

    edges = pressure_edges_hpa(surface_pressure_pa, hyai_hpa, hybi)
    return np.abs(edges[:, :-1, :, :] - edges[:, 1:, :, :])

def dry_surface_pressure_hpa(
    wet_surface_pressure_pa: np.ndarray,
    specific_humidity_kg_kg: np.ndarray,
    hyai_hpa: np.ndarray,
    hybi: np.ndarray,
    area_m2: np.ndarray | None = None,
) -> np.ndarray:
    """Compute GEOS-Chem-style dry surface pressure from canonical humidity."""

    wet_edges = pressure_edges_hpa(wet_surface_pressure_pa, hyai_hpa, hybi)
    wet_delta = wet_edges[:, :-1, :, :] - wet_edges[:, 1:, :, :]
    sphu = np.asarray(specific_humidity_kg_kg, dtype=np.float64)
    if sphu.shape != wet_delta.shape:
        raise ValueError(f"specific humidity shape {sphu.shape} does not match wet pressure layers {wet_delta.shape}")
    dry_ps = float(np.asarray(hyai_hpa, dtype=np.float64)[-1]) + np.sum(wet_delta * (1.0 - sphu), axis=1)
    wet_ps = np.asarray(wet_surface_pressure_pa, dtype=np.float64) / 100.0
    dry_ps = np.where(dry_ps < 0.0, wet_ps, dry_ps)
    return average_geos_chem_poles_2d(dry_ps, area_m2=area_m2)

def wet_surface_pressure_hpa(wet_surface_pressure_pa: np.ndarray, area_m2: np.ndarray | None = None) -> np.ndarray:
    """Return wet surface pressure in hPa with GEOS-Chem polar averaging."""

    return average_geos_chem_poles_2d(np.asarray(wet_surface_pressure_pa, dtype=np.float64) / 100.0, area_m2=area_m2)

def dry_pressure_thickness_from_surface_hpa(
    dry_surface_pressure_hpa: np.ndarray,
    hyai_hpa: np.ndarray,
    hybi: np.ndarray,
) -> np.ndarray:
    """Return GEOS-Chem dry pressure thickness from dry surface pressure in hPa."""

    ps = np.asarray(dry_surface_pressure_hpa, dtype=np.float64)
    hyai = np.asarray(hyai_hpa, dtype=np.float64)
    hybi = np.asarray(hybi, dtype=np.float64)
    edges = hyai[np.newaxis, :, np.newaxis, np.newaxis] + hybi[np.newaxis, :, np.newaxis, np.newaxis] * ps[
        :, np.newaxis, :, :
    ]
    return edges[:, :-1, :, :] - edges[:, 1:, :, :]

def dry_air_mass_from_pressure(delp_dry_hpa: np.ndarray, area_m2: np.ndarray) -> np.ndarray:
    """Convert dry pressure thickness to grid-box dry air mass in kg."""

    delp = np.asarray(delp_dry_hpa, dtype=np.float64)
    area = np.asarray(area_m2, dtype=np.float64)
    return delp * 100.0 / G0_M_PER_S2 * area[np.newaxis, np.newaxis, :, :]

def dry_pressure_edges_from_thickness_hpa(delp_dry_hpa: np.ndarray, top_edge_hpa: float | np.ndarray = 0.01) -> np.ndarray:
    """Reconstruct dry pressure edges from layer thickness on bottom-to-top levels."""

    delp = np.asarray(delp_dry_hpa, dtype=np.float64)
    edges = np.zeros((delp.shape[0], delp.shape[1] + 1, delp.shape[2], delp.shape[3]), dtype=np.float64)
    edges[:, -1:, :, :] = np.asarray(top_edge_hpa, dtype=np.float64)
    edges[:, :-1, :, :] = edges[:, -1:, :, :] + np.flip(np.cumsum(np.flip(delp, axis=1), axis=1), axis=1)
    return edges

def _dry_air_mass_to_pressure(dry_air_mass_kg: np.ndarray, area_m2: np.ndarray) -> np.ndarray:
    return np.asarray(dry_air_mass_kg, dtype=np.float64) / area_m2[np.newaxis, np.newaxis, :, :] * G0_M_PER_S2 / 100.0


def _dry_surface_pressure_from_mass_hpa(
    dry_air_mass_kg: np.ndarray,
    area_m2: np.ndarray,
    top_edge_hpa: float,
) -> np.ndarray:
    """Recover surface pressure from bottom-to-top dry-air layer mass."""

    return np.sum(_dry_air_mass_to_pressure(dry_air_mass_kg, area_m2), axis=1)[0] + float(
        top_edge_hpa
    )


def _dry_pressure_and_mass_from_surface_hpa(
    dry_surface_pressure_hpa: np.ndarray,
    area_m2: np.ndarray,
    hyai_hpa: np.ndarray,
    hybi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the paired dry-pressure thickness and mass arrays for one boundary."""

    delp_dry_hpa = dry_pressure_thickness_from_surface_hpa(
        dry_surface_pressure_hpa,
        hyai_hpa,
        hybi,
    )
    return delp_dry_hpa, dry_air_mass_from_pressure(delp_dry_hpa, area_m2)

def _pressure_flux_to_mass_kg(pressure_flux_hpa: np.ndarray, area_m2: np.ndarray) -> np.ndarray:
    return np.asarray(pressure_flux_hpa, dtype=np.float64) * 100.0 / G0_M_PER_S2 * area_m2[np.newaxis, np.newaxis, :, :]

def _mass_flux_to_pressure_hpa(mass_flux_kg: np.ndarray, area_m2: np.ndarray) -> np.ndarray:
    return np.asarray(mass_flux_kg, dtype=np.float64) / area_m2[np.newaxis, np.newaxis, :, :] * G0_M_PER_S2 / 100.0

def _meridional_pressure_flux_to_mass_kg(
    pressure_flux_hpa: np.ndarray,
    area_m2: np.ndarray,
    *,
    positive: bool,
) -> np.ndarray:
    source_area = np.zeros_like(area_m2, dtype=np.float64)
    if positive:
        source_area[1:, :] = area_m2[:-1, :]
    else:
        source_area[1:, :] = area_m2[1:, :]
    return np.asarray(pressure_flux_hpa, dtype=np.float64) * 100.0 / G0_M_PER_S2 * source_area[np.newaxis, np.newaxis, :, :]

def average_geos_chem_poles_2d(field: np.ndarray, area_m2: np.ndarray | None = None) -> np.ndarray:
    averaged = np.asarray(field, dtype=np.float64).copy()
    if averaged.ndim != 3:
        raise ValueError(f"surface pressure must have shape (time, lat, lon), found {averaged.shape}")
    if averaged.shape[1] < 4:
        return averaged
    if area_m2 is None:
        area = np.ones(averaged.shape[1:], dtype=np.float64)
    else:
        area = np.asarray(area_m2, dtype=np.float64)
        if area.shape != averaged.shape[1:]:
            raise ValueError(f"area_m2 shape {area.shape} does not match surface field shape {averaged.shape[1:]}")

    south_area = area[0:2, :]
    north_area = area[-2:, :]
    south = np.sum(averaged[:, 0:2, :] * south_area[np.newaxis, :, :], axis=(1, 2)) / np.sum(south_area)
    north = np.sum(averaged[:, -2:, :] * north_area[np.newaxis, :, :], axis=(1, 2)) / np.sum(north_area)
    averaged[:, 0, :] = south[:, np.newaxis]
    averaged[:, 1, :] = south[:, np.newaxis]
    averaged[:, -2, :] = north[:, np.newaxis]
    averaged[:, -1, :] = north[:, np.newaxis]
    return averaged
