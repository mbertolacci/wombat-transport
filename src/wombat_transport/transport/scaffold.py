from __future__ import annotations

import numpy as np

from wombat_transport.constants import EARTH_RADIUS_M
from wombat_transport.fields import TracerField
from wombat_transport.transport.pressure import (
    _dry_air_mass_to_pressure,
    _mass_flux_to_pressure_hpa,
    _meridional_pressure_flux_to_mass_kg,
    _pressure_flux_to_mass_kg,
)

def horizontal_mass_flux_hpa(
    delp_hpa: np.ndarray,
    u_m_s: np.ndarray,
    v_m_s: np.ndarray,
    lat_deg: np.ndarray,
    *,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate GEOS-Chem/PJC horizontal mass fluxes in hPa.

    Positive x-flux enters a grid box from its western edge. Positive y-flux
    enters a grid box from its southern edge.
    """

    delp = np.asarray(delp_hpa, dtype=np.float64)
    u = np.asarray(u_m_s, dtype=np.float64)
    v = np.asarray(v_m_s, dtype=np.float64)
    lat = np.asarray(lat_deg, dtype=np.float64)
    nlon = delp.shape[-1]
    nlat = delp.shape[-2]

    dlambda = 2.0 * np.pi / float(nlon)
    dphi = np.pi / float(nlat - 1)
    cos_center = np.cos(np.deg2rad(lat))
    safe_cos_center = np.where(np.abs(cos_center) > 1.0e-12, cos_center, np.nan)
    factx = 0.5 * float(dt_s) / (dlambda * EARTH_RADIUS_M * safe_cos_center)
    xmass = factx[np.newaxis, np.newaxis, :, np.newaxis] * (
        (u * delp) + (np.roll(u, 1, axis=-1) * np.roll(delp, 1, axis=-1))
    )
    xmass = np.where(np.isfinite(xmass), xmass, 0.0)

    edge_lat = np.clip(lat - np.rad2deg(dphi) / 2.0, -90.0, 90.0)
    cos_edge = np.cos(np.deg2rad(edge_lat))
    facty = 0.5 * float(dt_s) / (EARTH_RADIUS_M * dphi)
    ymass = np.zeros_like(delp)
    ymass[:, :, 1:, :] = facty * cos_edge[np.newaxis, np.newaxis, 1:, np.newaxis] * (
        (v[:, :, 1:, :] * delp[:, :, 1:, :]) + (v[:, :, :-1, :] * delp[:, :, :-1, :])
    )
    return xmass, ymass

def advect_horizontal_mass_flux(
    tracer_field: TracerField,
    dry_air_mass_kg: np.ndarray,
    xmass_hpa: np.ndarray,
    ymass_hpa: np.ndarray,
    area_m2: np.ndarray,
    *,
    max_courant: float = 0.95,
) -> tuple[TracerField, np.ndarray]:
    """Conservative first-order upwind transport from signed mass fluxes."""

    air_mass = np.asarray(dry_air_mass_kg, dtype=np.float64).copy()
    if np.all(xmass_hpa == 0.0) and np.all(ymass_hpa == 0.0):
        return (
            TracerField(
                names=tracer_field.names,
                data=tracer_field.data.copy(),
                units=tracer_field.units,
                coords=tracer_field.coords,
            ),
            air_mass,
        )
    tracer_mass = tracer_field.data * air_mass[np.newaxis, ...]
    area = np.asarray(area_m2, dtype=np.float64)
    delp = _dry_air_mass_to_pressure(air_mass, area)

    x_positive = np.minimum(np.clip(xmass_hpa, 0.0, None), np.roll(delp, 1, axis=-1) * max_courant)
    x_negative = np.minimum(np.clip(-xmass_hpa, 0.0, None), delp * max_courant)
    air_mass, tracer_mass = _apply_x_edge_flux_kg(
        air_mass,
        tracer_mass,
        _pressure_flux_to_mass_kg(x_positive, area),
        positive=True,
    )
    air_mass, tracer_mass = _apply_x_edge_flux_kg(
        air_mass,
        tracer_mass,
        _pressure_flux_to_mass_kg(x_negative, area),
        positive=False,
    )

    delp = _dry_air_mass_to_pressure(air_mass, area)
    y_positive = np.zeros_like(ymass_hpa)
    y_positive[:, :, 1:, :] = np.minimum(
        np.clip(ymass_hpa[:, :, 1:, :], 0.0, None),
        delp[:, :, :-1, :] * max_courant,
    )
    y_negative = np.zeros_like(ymass_hpa)
    y_negative[:, :, 1:, :] = np.minimum(
        np.clip(-ymass_hpa[:, :, 1:, :], 0.0, None),
        delp[:, :, 1:, :] * max_courant,
    )
    air_mass, tracer_mass = _apply_y_edge_flux_kg(
        air_mass,
        tracer_mass,
        _meridional_pressure_flux_to_mass_kg(y_positive, area, positive=True),
        positive=True,
    )
    air_mass, tracer_mass = _apply_y_edge_flux_kg(
        air_mass,
        tracer_mass,
        _meridional_pressure_flux_to_mass_kg(y_negative, area, positive=False),
        positive=False,
    )

    data = np.divide(
        tracer_mass,
        air_mass[np.newaxis, ...],
        out=np.zeros_like(tracer_mass),
        where=air_mass[np.newaxis, ...] > 0.0,
    )
    return (
        TracerField(
            names=tracer_field.names,
            data=data,
            units=tracer_field.units,
            coords=tracer_field.coords,
        ),
        air_mass,
    )

def vertical_mass_flux_hpa(
    previous_dry_air_mass_kg: np.ndarray,
    horizontal_dry_air_mass_kg: np.ndarray,
    area_m2: np.ndarray,
) -> np.ndarray:
    """Infer closed-boundary vertical edge fluxes from horizontal mass divergence.

    Positive flux moves upward from level k-1 into level k across edge k.
    """

    previous = np.asarray(previous_dry_air_mass_kg, dtype=np.float64)
    horizontal = np.asarray(horizontal_dry_air_mass_kg, dtype=np.float64)
    column_previous = np.sum(previous, axis=1, keepdims=True)
    column_horizontal = np.sum(horizontal, axis=1, keepdims=True)
    target = np.divide(
        previous * column_horizontal,
        column_previous,
        out=horizontal.copy(),
        where=column_previous > 0.0,
    )
    vertical_change = target - horizontal
    flux_kg = np.zeros((previous.shape[0], previous.shape[1] + 1, previous.shape[2], previous.shape[3]), dtype=np.float64)
    flux_kg[:, 1:, :, :] = -np.cumsum(vertical_change, axis=1)
    flux_kg[:, -1, :, :] = 0.0
    return _mass_flux_to_pressure_hpa(flux_kg, area_m2)

def advect_vertical_mass_flux(
    tracer_field: TracerField,
    dry_air_mass_kg: np.ndarray,
    zmass_hpa: np.ndarray,
    area_m2: np.ndarray,
    *,
    max_courant: float = 0.95,
) -> tuple[TracerField, np.ndarray]:
    """Conservative first-order vertical upwind transport from edge mass fluxes."""

    air_mass = np.asarray(dry_air_mass_kg, dtype=np.float64).copy()
    zmass = np.asarray(zmass_hpa, dtype=np.float64).copy()
    zmass[:, 0, :, :] = 0.0
    zmass[:, -1, :, :] = 0.0
    if np.all(zmass == 0.0):
        return (
            TracerField(
                names=tracer_field.names,
                data=tracer_field.data.copy(),
                units=tracer_field.units,
                coords=tracer_field.coords,
            ),
            air_mass,
        )

    tracer_mass = tracer_field.data * air_mass[np.newaxis, ...]
    area = np.asarray(area_m2, dtype=np.float64)
    delp = _dry_air_mass_to_pressure(air_mass, area)
    upward = np.minimum(np.clip(zmass[:, 1:-1, :, :], 0.0, None), delp[:, :-1, :, :] * max_courant)
    air_mass, tracer_mass = _apply_vertical_edge_flux_kg(
        air_mass,
        tracer_mass,
        _pressure_flux_to_mass_kg(upward, area),
        upward=True,
    )

    delp = _dry_air_mass_to_pressure(air_mass, area)
    downward = np.minimum(np.clip(-zmass[:, 1:-1, :, :], 0.0, None), delp[:, 1:, :, :] * max_courant)
    air_mass, tracer_mass = _apply_vertical_edge_flux_kg(
        air_mass,
        tracer_mass,
        _pressure_flux_to_mass_kg(downward, area),
        upward=False,
    )

    data = np.divide(
        tracer_mass,
        air_mass[np.newaxis, ...],
        out=np.zeros_like(tracer_mass),
        where=air_mass[np.newaxis, ...] > 0.0,
    )
    return (
        TracerField(
            names=tracer_field.names,
            data=data,
            units=tracer_field.units,
            coords=tracer_field.coords,
        ),
        air_mass,
    )

def advect_horizontal_upwind(
    tracer_field: TracerField,
    dry_air_mass_kg: np.ndarray,
    u_m_s: np.ndarray,
    v_m_s: np.ndarray,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    *,
    dt_s: float,
    max_courant: float = 0.95,
) -> tuple[TracerField, np.ndarray]:
    """Conservative first-order horizontal upwind step over all tracers.

    This updates dry air mass and tracer scalar mass consistently. Vertical
    transport is intentionally absent in this scaffold; top and bottom are
    therefore no-flux boundaries for this first slice.
    """

    air_mass = np.asarray(dry_air_mass_kg, dtype=np.float64).copy()
    tracer_mass = tracer_field.data * air_mass[np.newaxis, ...]

    dx, dy = horizontal_grid_spacing(lat_deg, lon_deg)
    courant_x = np.asarray(u_m_s, dtype=np.float64) * float(dt_s) / dx[np.newaxis, np.newaxis, :, np.newaxis]
    courant_y = np.asarray(v_m_s, dtype=np.float64) * float(dt_s) / dy
    if np.all(courant_x == 0.0) and np.all(courant_y == 0.0):
        return (
            TracerField(
                names=tracer_field.names,
                data=tracer_field.data.copy(),
                units=tracer_field.units,
                coords=tracer_field.coords,
            ),
            air_mass,
        )

    air_mass, tracer_mass = _apply_periodic_x_flux(
        air_mass,
        tracer_mass,
        np.clip(courant_x, 0.0, max_courant),
        destination_shift=1,
    )
    air_mass, tracer_mass = _apply_periodic_x_flux(
        air_mass,
        tracer_mass,
        np.clip(-courant_x, 0.0, max_courant),
        destination_shift=-1,
    )
    north_fraction = np.clip(courant_y, 0.0, max_courant)
    north_fraction[:, :, -1, :] = 0.0
    air_mass, tracer_mass = _apply_y_flux(
        air_mass,
        tracer_mass,
        north_fraction,
        northward=True,
    )
    south_fraction = np.clip(-courant_y, 0.0, max_courant)
    south_fraction[:, :, 0, :] = 0.0
    air_mass, tracer_mass = _apply_y_flux(
        air_mass,
        tracer_mass,
        south_fraction,
        northward=False,
    )

    data = np.divide(
        tracer_mass,
        air_mass[np.newaxis, ...],
        out=np.zeros_like(tracer_mass),
        where=air_mass[np.newaxis, ...] > 0.0,
    )
    return (
        TracerField(
            names=tracer_field.names,
            data=data,
            units=tracer_field.units,
            coords=tracer_field.coords,
        ),
        air_mass,
    )

def horizontal_grid_spacing(lat_deg: np.ndarray, lon_deg: np.ndarray) -> tuple[np.ndarray, float]:
    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = np.asarray(lon_deg, dtype=np.float64)
    dlon = np.deg2rad(float(np.mean(np.diff(lon))))
    dlat = np.deg2rad(float(np.mean(np.diff(lat))))
    dx = EARTH_RADIUS_M * np.cos(np.deg2rad(lat)) * dlon
    dx = np.maximum(dx, 1.0)
    dy = EARTH_RADIUS_M * dlat
    return dx, dy

def _apply_x_edge_flux_kg(
    air_mass: np.ndarray,
    tracer_mass: np.ndarray,
    flux_kg: np.ndarray,
    *,
    positive: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if np.all(flux_kg == 0.0):
        return air_mass, tracer_mass
    if positive:
        outgoing = np.roll(flux_kg, -1, axis=-1)
        incoming = flux_kg
        incoming_tracer = np.roll(tracer_mass / air_mass[np.newaxis, ...], 1, axis=-1) * incoming[np.newaxis, ...]
        outgoing_tracer = (tracer_mass / air_mass[np.newaxis, ...]) * outgoing[np.newaxis, ...]
    else:
        outgoing = flux_kg
        incoming = np.roll(flux_kg, -1, axis=-1)
        mixing_ratio = tracer_mass / air_mass[np.newaxis, ...]
        incoming_tracer = np.roll(mixing_ratio * flux_kg[np.newaxis, ...], -1, axis=-1)
        outgoing_tracer = mixing_ratio * outgoing[np.newaxis, ...]
    return (
        air_mass - outgoing + incoming,
        tracer_mass - outgoing_tracer + incoming_tracer,
    )

def _apply_y_edge_flux_kg(
    air_mass: np.ndarray,
    tracer_mass: np.ndarray,
    flux_kg: np.ndarray,
    *,
    positive: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if np.all(flux_kg == 0.0):
        return air_mass, tracer_mass

    mixing_ratio = tracer_mass / air_mass[np.newaxis, ...]
    next_air_mass = air_mass.copy()
    next_tracer_mass = tracer_mass.copy()

    if positive:
        edge_flux = flux_kg[:, :, 1:, :]
        next_air_mass[:, :, :-1, :] -= edge_flux
        next_air_mass[:, :, 1:, :] += edge_flux
        tracer_flux = mixing_ratio[:, :, :, :-1, :] * edge_flux[np.newaxis, ...]
        next_tracer_mass[:, :, :, :-1, :] -= tracer_flux
        next_tracer_mass[:, :, :, 1:, :] += tracer_flux
    else:
        edge_flux = flux_kg[:, :, 1:, :]
        next_air_mass[:, :, 1:, :] -= edge_flux
        next_air_mass[:, :, :-1, :] += edge_flux
        tracer_flux = mixing_ratio[:, :, :, 1:, :] * edge_flux[np.newaxis, ...]
        next_tracer_mass[:, :, :, 1:, :] -= tracer_flux
        next_tracer_mass[:, :, :, :-1, :] += tracer_flux

    return next_air_mass, next_tracer_mass

def _apply_vertical_edge_flux_kg(
    air_mass: np.ndarray,
    tracer_mass: np.ndarray,
    flux_kg: np.ndarray,
    *,
    upward: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if np.all(flux_kg == 0.0):
        return air_mass, tracer_mass

    mixing_ratio = tracer_mass / air_mass[np.newaxis, ...]
    next_air_mass = air_mass.copy()
    next_tracer_mass = tracer_mass.copy()

    if upward:
        next_air_mass[:, :-1, :, :] -= flux_kg
        next_air_mass[:, 1:, :, :] += flux_kg
        tracer_flux = mixing_ratio[:, :, :-1, :, :] * flux_kg[np.newaxis, ...]
        next_tracer_mass[:, :, :-1, :, :] -= tracer_flux
        next_tracer_mass[:, :, 1:, :, :] += tracer_flux
    else:
        next_air_mass[:, 1:, :, :] -= flux_kg
        next_air_mass[:, :-1, :, :] += flux_kg
        tracer_flux = mixing_ratio[:, :, 1:, :, :] * flux_kg[np.newaxis, ...]
        next_tracer_mass[:, :, 1:, :, :] -= tracer_flux
        next_tracer_mass[:, :, :-1, :, :] += tracer_flux

    return next_air_mass, next_tracer_mass

def _apply_periodic_x_flux(
    air_mass: np.ndarray,
    tracer_mass: np.ndarray,
    fraction: np.ndarray,
    *,
    destination_shift: int,
) -> tuple[np.ndarray, np.ndarray]:
    flux_air = air_mass * fraction
    flux_tracer = tracer_mass * fraction[np.newaxis, ...]
    return (
        air_mass - flux_air + np.roll(flux_air, destination_shift, axis=-1),
        tracer_mass - flux_tracer + np.roll(flux_tracer, destination_shift, axis=-1),
    )

def _apply_y_flux(
    air_mass: np.ndarray,
    tracer_mass: np.ndarray,
    fraction: np.ndarray,
    *,
    northward: bool,
) -> tuple[np.ndarray, np.ndarray]:
    flux_air = air_mass * fraction
    flux_tracer = tracer_mass * fraction[np.newaxis, ...]
    next_air_mass = air_mass - flux_air
    next_tracer_mass = tracer_mass - flux_tracer
    if northward:
        next_air_mass[:, :, 1:, :] += flux_air[:, :, :-1, :]
        next_tracer_mass[:, :, :, 1:, :] += flux_tracer[:, :, :, :-1, :]
    else:
        next_air_mass[:, :, :-1, :] += flux_air[:, :, 1:, :]
        next_tracer_mass[:, :, :, :-1, :] += flux_tracer[:, :, :, 1:, :]
    return next_air_mass, next_tracer_mass
