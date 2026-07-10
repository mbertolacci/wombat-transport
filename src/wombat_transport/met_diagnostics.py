from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wombat_transport.constants import AIRMW_G_PER_MOL, G0_M_PER_S2
from wombat_transport.grid import TransportGrid
from wombat_transport.transport.forcing import TransportForcing

H2OMW_G_PER_MOL = 18.016
RD_J_PER_KG_K = 287.0


@dataclass(frozen=True)
class AirQuantityDiagnostics:
    """AIRQNT-style met diagnostics in public GEOS-Chem bottom-level order."""

    wet_pressure_edges_hpa: np.ndarray
    dry_partial_pressure_edges_hpa: np.ndarray
    water_vapor_volume_mixing_ratio: np.ndarray
    box_height_m: np.ndarray


def airqnt_diagnostics_from_forcing(
    forcing: TransportForcing,
    grid: TransportGrid,
) -> AirQuantityDiagnostics:
    """Compute GEOS-Chem AIRQNT diagnostic fields from Wombat forcing.

    GEOS-Chem writes these fields in public bottom-to-top order.  The dry edge
    diagnostic is the local dry-air partial pressure, not the transport dry
    pressure coordinate derived from dry surface pressure.
    """

    return airqnt_diagnostics_from_fields(
        wet_surface_pressure_hpa=forcing.wet_surface_pressure_hpa[0],
        specific_humidity_kg_kg=forcing.specific_humidity_kg_kg[0],
        temperature_k=forcing.temperature_k[0],
        grid=grid,
    )


def airqnt_diagnostics_from_fields(
    *,
    wet_surface_pressure_hpa: np.ndarray,
    specific_humidity_kg_kg: np.ndarray,
    temperature_k: np.ndarray,
    grid: TransportGrid,
) -> AirQuantityDiagnostics:
    """Compute AIRQNT diagnostics from bottom-order met fields."""

    wet_edges = _pressure_edges_from_surface_hpa(
        np.asarray(wet_surface_pressure_hpa, dtype=np.float64)[np.newaxis, :, :],
        grid.hyai_hpa,
        grid.hybi,
    )[0]
    q = np.asarray(specific_humidity_kg_kg, dtype=np.float64)
    temperature = np.asarray(temperature_k, dtype=np.float64)
    if q.shape != wet_edges[:-1].shape:
        raise ValueError(f"specific humidity shape {q.shape} does not match grid layers {wet_edges[:-1].shape}")
    if temperature.shape != q.shape:
        raise ValueError(f"temperature shape {temperature.shape} does not match humidity {q.shape}")

    avgw = AIRMW_G_PER_MOL * q / (H2OMW_G_PER_MOL * (1.0 - q))
    xh2o = avgw / (1.0 + avgw)
    virtual_temperature = temperature / (1.0 - xh2o * (1.0 - H2OMW_G_PER_MOL / AIRMW_G_PER_MOL))
    bxheight = (RD_J_PER_KG_K / G0_M_PER_S2) * virtual_temperature * np.log(wet_edges[:-1] / wet_edges[1:])

    pedge_dry = np.empty_like(wet_edges)
    pedge_dry[:-1] = wet_edges[:-1] * (1.0 - xh2o)
    pedge_dry[-1] = wet_edges[-1] * (1.0 - xh2o[-1])

    return AirQuantityDiagnostics(
        wet_pressure_edges_hpa=wet_edges,
        dry_partial_pressure_edges_hpa=pedge_dry,
        water_vapor_volume_mixing_ratio=avgw,
        box_height_m=bxheight,
    )


def _pressure_edges_from_surface_hpa(surface_pressure_hpa: np.ndarray, hyai_hpa: np.ndarray, hybi: np.ndarray) -> np.ndarray:
    ps = np.asarray(surface_pressure_hpa, dtype=np.float64)
    hyai = np.asarray(hyai_hpa, dtype=np.float64)
    hyb = np.asarray(hybi, dtype=np.float64)
    return hyai[np.newaxis, :, np.newaxis, np.newaxis] + hyb[np.newaxis, :, np.newaxis, np.newaxis] * ps[
        :, np.newaxis, :, :
    ]
