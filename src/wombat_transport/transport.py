from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import netCDF4
import numpy as np

from wombat_transport.constants import EARTH_RADIUS_M, G0_M_PER_S2
from wombat_transport.fields import TracerField
from wombat_transport.io import FIXED_GRID

MERRA2_FILENAME = "MERRA2.{date}.{collection}.2x25.nc4"


@dataclass(frozen=True)
class TransportForcing:
    """Meteorological forcing mapped onto the prototype 47-level grid."""

    u_m_s: np.ndarray
    v_m_s: np.ndarray
    omega_pa_s: np.ndarray
    surface_pressure_pa: np.ndarray
    specific_humidity_kg_kg: np.ndarray
    temperature_k: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    vertical_mapping: str
    a3dyn_path: Path
    i3_path: Path


@dataclass(frozen=True)
class TransportStepResult:
    state: TracerField
    dry_air_mass_kg: np.ndarray
    delp_dry_hpa: np.ndarray
    initial_scalar_mass: np.ndarray
    final_scalar_mass: np.ndarray


def load_transport_forcing(
    met_root: str | Path,
    timestamp: datetime,
    template_path: str | Path,
    *,
    time_index: int = 0,
) -> TransportForcing:
    """Load MERRA2 forcing for one day and map 72 met levels to 47 levels.

    The current 72-to-47 mapping is deliberately simple: use the lowest 47
    MERRA2 levels. This is a transport scaffold, not a TPCORE parity claim.
    """

    met_root = Path(met_root)
    day_dir = met_root / f"{timestamp.year:04d}" / f"{timestamp.month:02d}"
    date = timestamp.strftime("%Y%m%d")
    a3dyn_path = day_dir / MERRA2_FILENAME.format(date=date, collection="A3dyn")
    i3_path = day_dir / MERRA2_FILENAME.format(date=date, collection="I3")

    with netCDF4.Dataset(a3dyn_path) as a3dyn, netCDF4.Dataset(i3_path) as i3, netCDF4.Dataset(template_path) as template:
        lat = np.asarray(template.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(template.variables["lon"][:], dtype=np.float64)
        u = _read_3d_time_slice(a3dyn, "U", time_index)
        v = _read_3d_time_slice(a3dyn, "V", time_index)
        omega = _read_3d_time_slice(a3dyn, "OMEGA", time_index)
        qv = _read_3d_time_slice(i3, "QV", time_index)
        temperature = _read_3d_time_slice(i3, "T", time_index)
        surface_pressure = np.asarray(i3.variables["PS"][time_index : time_index + 1], dtype=np.float64)

    return TransportForcing(
        u_m_s=_map_met_levels_to_47(u),
        v_m_s=_map_met_levels_to_47(v),
        omega_pa_s=_map_met_levels_to_47(omega),
        surface_pressure_pa=surface_pressure,
        specific_humidity_kg_kg=_map_met_levels_to_47(qv),
        temperature_k=_map_met_levels_to_47(temperature),
        lat_deg=lat,
        lon_deg=lon,
        vertical_mapping="lowest_47_of_72",
        a3dyn_path=a3dyn_path.resolve(),
        i3_path=i3_path.resolve(),
    )


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


def run_transport_one_step(
    tracer_field: TracerField,
    forcing: TransportForcing,
    template_path: str | Path,
    *,
    dt_s: float = 600.0,
    max_courant: float = 0.95,
) -> TransportStepResult:
    """Run one conservative horizontal upwind transport scaffold step."""

    with netCDF4.Dataset(template_path) as template:
        hyai = np.asarray(template.variables["hyai"][:], dtype=np.float64)
        hybi = np.asarray(template.variables["hybi"][:], dtype=np.float64)
        area = np.asarray(template.variables["AREA"][:], dtype=np.float64)

    delp = dry_pressure_thickness_hpa(forcing.surface_pressure_pa, hyai, hybi)
    dry_air_mass = dry_air_mass_from_pressure(delp, area)
    state, next_dry_air_mass = advect_horizontal_upwind(
        tracer_field,
        dry_air_mass,
        forcing.u_m_s,
        forcing.v_m_s,
        forcing.lat_deg,
        forcing.lon_deg,
        dt_s=dt_s,
        max_courant=max_courant,
    )
    next_delp = next_dry_air_mass / area[np.newaxis, np.newaxis, :, :] * G0_M_PER_S2 / 100.0
    return TransportStepResult(
        state=state,
        dry_air_mass_kg=next_dry_air_mass,
        delp_dry_hpa=next_delp,
        initial_scalar_mass=scalar_mass_by_tracer(tracer_field.data, dry_air_mass),
        final_scalar_mass=scalar_mass_by_tracer(state.data, next_dry_air_mass),
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


def scalar_mass_by_tracer(field_data: np.ndarray, dry_air_mass_kg: np.ndarray) -> np.ndarray:
    return np.sum(np.asarray(field_data) * np.asarray(dry_air_mass_kg)[np.newaxis, ...], axis=(1, 2, 3, 4))


def _read_3d_time_slice(dataset: netCDF4.Dataset, variable_name: str, time_index: int) -> np.ndarray:
    return np.asarray(dataset.variables[variable_name][time_index : time_index + 1], dtype=np.float64)


def _map_met_levels_to_47(data: np.ndarray) -> np.ndarray:
    if data.shape[1] == FIXED_GRID["lev"]:
        return data
    if data.shape[1] == 72:
        return data[:, -FIXED_GRID["lev"] :, :, :]
    raise ValueError(f"cannot map {data.shape[1]} met levels to {FIXED_GRID['lev']}")


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
