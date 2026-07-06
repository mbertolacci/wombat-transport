from __future__ import annotations

import numpy as np

from wombat_transport.constants import G0_M_PER_S2

def pressure_edges_hpa(surface_pressure_pa: np.ndarray, hyai_hpa: np.ndarray, hybi: np.ndarray) -> np.ndarray:
    """Compute hybrid pressure edges in hPa from surface pressure in Pa."""

    ps_hpa = np.asarray(surface_pressure_pa, dtype=np.float64) / 100.0
    hyai = np.asarray(hyai_hpa, dtype=np.float64)
    hybi = np.asarray(hybi, dtype=np.float64)
    return hyai[np.newaxis, :, np.newaxis, np.newaxis] + (
        hybi[np.newaxis, :, np.newaxis, np.newaxis] * ps_hpa[:, np.newaxis, :, :]
    )

def dry_pressure_thickness_hpa(surface_pressure_pa: np.ndarray, hyai_hpa: np.ndarray, hybi: np.ndarray) -> np.ndarray:
    """Return positive pressure thickness on the prototype level order."""

    edges = pressure_edges_hpa(surface_pressure_pa, hyai_hpa, hybi)
    return np.abs(edges[:, :-1, :, :] - edges[:, 1:, :, :])

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
