from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4
import numpy as np

from wombat_transport.constants import EARTH_RADIUS_M, G0_M_PER_S2
from wombat_transport.fields import TracerField
from wombat_transport.io import FIXED_GRID

MERRA2_FILENAME = "MERRA2.{date}.{collection}.2x25.nc4"
MERRA2_72_AP_HPA = np.array(
    [
        0.000000e00,
        4.804826e-02,
        6.593752e00,
        1.313480e01,
        1.961311e01,
        2.609201e01,
        3.257081e01,
        3.898201e01,
        4.533901e01,
        5.169611e01,
        5.805321e01,
        6.436264e01,
        7.062198e01,
        7.883422e01,
        8.909992e01,
        9.936521e01,
        1.091817e02,
        1.189586e02,
        1.286959e02,
        1.429100e02,
        1.562600e02,
        1.696090e02,
        1.816190e02,
        1.930970e02,
        2.032590e02,
        2.121500e02,
        2.187760e02,
        2.238980e02,
        2.243630e02,
        2.168650e02,
        2.011920e02,
        1.769300e02,
        1.503930e02,
        1.278370e02,
        1.086630e02,
        9.236572e01,
        7.851231e01,
        6.660341e01,
        5.638791e01,
        4.764391e01,
        4.017541e01,
        3.381001e01,
        2.836781e01,
        2.373041e01,
        1.979160e01,
        1.645710e01,
        1.364340e01,
        1.127690e01,
        9.292942e00,
        7.619842e00,
        6.216801e00,
        5.046801e00,
        4.076571e00,
        3.276431e00,
        2.620211e00,
        2.084970e00,
        1.650790e00,
        1.300510e00,
        1.019440e00,
        7.951341e-01,
        6.167791e-01,
        4.758061e-01,
        3.650411e-01,
        2.785261e-01,
        2.113490e-01,
        1.594950e-01,
        1.197030e-01,
        8.934502e-02,
        6.600001e-02,
        4.758501e-02,
        3.270000e-02,
        2.000000e-02,
        1.000000e-02,
    ],
    dtype=np.float64,
)
MERRA2_72_TO_47_GROUPS = (
    (36, 38),
    (38, 40),
    (40, 42),
    (42, 44),
    (44, 48),
    (48, 52),
    (52, 56),
    (56, 60),
    (60, 64),
    (64, 68),
    (68, 72),
)
MERRA2_72_TO_47_MAPPING = "collapse_72_to_47_pressure_weighted"


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
    xmass_hpa: np.ndarray
    ymass_hpa: np.ndarray
    zmass_hpa: np.ndarray
    initial_scalar_mass: np.ndarray
    final_scalar_mass: np.ndarray


@dataclass(frozen=True)
class TransportWindowResult:
    state: TracerField
    average_state: TracerField
    dry_air_mass_kg: np.ndarray
    average_dry_air_mass_kg: np.ndarray
    delp_dry_hpa: np.ndarray
    average_delp_dry_hpa: np.ndarray
    initial_scalar_mass: np.ndarray
    final_scalar_mass: np.ndarray
    steps: int
    dt_s: float


def load_transport_forcing(
    met_root: str | Path,
    timestamp: datetime,
    template_path: str | Path,
    *,
    time_index: int = 0,
) -> TransportForcing:
    """Load MERRA2 forcing for one day and map 72 met levels to 47 levels."""

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
        vertical_mapping=MERRA2_72_TO_47_MAPPING,
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


def dry_pressure_edges_from_thickness_hpa(delp_dry_hpa: np.ndarray, top_edge_hpa: float | np.ndarray = 0.01) -> np.ndarray:
    """Reconstruct dry pressure edges from layer thickness on bottom-to-top levels."""

    delp = np.asarray(delp_dry_hpa, dtype=np.float64)
    edges = np.zeros((delp.shape[0], delp.shape[1] + 1, delp.shape[2], delp.shape[3]), dtype=np.float64)
    edges[:, -1:, :, :] = np.asarray(top_edge_hpa, dtype=np.float64)
    edges[:, :-1, :, :] = edges[:, -1:, :, :] + np.flip(np.cumsum(np.flip(delp, axis=1), axis=1), axis=1)
    return edges


def run_transport_one_step(
    tracer_field: TracerField,
    forcing: TransportForcing,
    template_path: str | Path,
    *,
    dt_s: float = 600.0,
    max_courant: float = 0.95,
) -> TransportStepResult:
    """Run one conservative horizontal mass-flux transport scaffold step."""

    with netCDF4.Dataset(template_path) as template:
        hyai = np.asarray(template.variables["hyai"][:], dtype=np.float64)
        hybi = np.asarray(template.variables["hybi"][:], dtype=np.float64)
        area = np.asarray(template.variables["AREA"][:], dtype=np.float64)

    delp = dry_pressure_thickness_hpa(forcing.surface_pressure_pa, hyai, hybi)
    dry_air_mass = dry_air_mass_from_pressure(delp, area)
    xmass, ymass = horizontal_mass_flux_hpa(
        delp,
        forcing.u_m_s,
        forcing.v_m_s,
        forcing.lat_deg,
        dt_s=dt_s,
    )
    horizontal_state, horizontal_dry_air_mass = advect_horizontal_mass_flux(
        tracer_field,
        dry_air_mass,
        xmass,
        ymass,
        area,
        max_courant=max_courant,
    )
    zmass = vertical_mass_flux_hpa(dry_air_mass, horizontal_dry_air_mass, area)
    state, next_dry_air_mass = advect_vertical_mass_flux(
        horizontal_state,
        horizontal_dry_air_mass,
        zmass,
        area,
        max_courant=max_courant,
    )
    next_delp = next_dry_air_mass / area[np.newaxis, np.newaxis, :, :] * G0_M_PER_S2 / 100.0
    return TransportStepResult(
        state=state,
        dry_air_mass_kg=next_dry_air_mass,
        delp_dry_hpa=next_delp,
        xmass_hpa=xmass,
        ymass_hpa=ymass,
        zmass_hpa=zmass,
        initial_scalar_mass=scalar_mass_by_tracer(tracer_field.data, dry_air_mass),
        final_scalar_mass=scalar_mass_by_tracer(state.data, next_dry_air_mass),
    )


def run_transport_window(
    tracer_field: TracerField,
    met_root: str | Path,
    start: datetime,
    template_path: str | Path,
    *,
    steps: int = 18,
    dt_s: float = 600.0,
    initial_met_time_index: int = 0,
    max_courant: float = 0.95,
) -> TransportWindowResult:
    """Run a short transport window and accumulate arithmetic mean state."""

    if steps <= 0:
        raise ValueError("steps must be positive")

    state = tracer_field
    dry_mass_sum = None
    state_sum = None
    delp_sum = None
    initial_scalar_mass = None
    final_scalar_mass = None
    forcing_cache: dict[tuple[datetime, int], TransportForcing] = {}
    with netCDF4.Dataset(template_path) as template:
        hyai = np.asarray(template.variables["hyai"][:], dtype=np.float64)
        hybi = np.asarray(template.variables["hybi"][:], dtype=np.float64)
        area = np.asarray(template.variables["AREA"][:], dtype=np.float64)
    first_forcing = _load_window_forcing(
        forcing_cache,
        met_root,
        start,
        template_path,
        step=0,
        dt_s=dt_s,
        initial_met_time_index=initial_met_time_index,
    )
    dry_air_mass = dry_air_mass_from_pressure(
        dry_pressure_thickness_hpa(first_forcing.surface_pressure_pa, hyai, hybi),
        area,
    )

    for step in range(steps):
        forcing = _load_window_forcing(
            forcing_cache,
            met_root,
            start,
            template_path,
            step=step,
            dt_s=dt_s,
            initial_met_time_index=initial_met_time_index,
        )
        step_result = _run_transport_one_step_with_mass(
            state,
            forcing,
            dry_air_mass,
            area,
            dt_s=dt_s,
            max_courant=max_courant,
        )
        if initial_scalar_mass is None:
            initial_scalar_mass = step_result.initial_scalar_mass
        final_scalar_mass = step_result.final_scalar_mass
        state = step_result.state
        dry_air_mass = step_result.dry_air_mass_kg

        if state_sum is None:
            state_sum = np.zeros_like(state.data)
            dry_mass_sum = np.zeros_like(step_result.dry_air_mass_kg)
            delp_sum = np.zeros_like(step_result.delp_dry_hpa)
        state_sum += state.data
        dry_mass_sum += step_result.dry_air_mass_kg
        delp_sum += step_result.delp_dry_hpa

    assert state_sum is not None
    assert dry_mass_sum is not None
    assert delp_sum is not None
    assert initial_scalar_mass is not None
    assert final_scalar_mass is not None

    average_state = TracerField(
        names=state.names,
        data=state_sum / float(steps),
        units=state.units,
        coords=state.coords,
    )
    return TransportWindowResult(
        state=state,
        average_state=average_state,
        dry_air_mass_kg=step_result.dry_air_mass_kg,
        average_dry_air_mass_kg=dry_mass_sum / float(steps),
        delp_dry_hpa=step_result.delp_dry_hpa,
        average_delp_dry_hpa=delp_sum / float(steps),
        initial_scalar_mass=initial_scalar_mass,
        final_scalar_mass=final_scalar_mass,
        steps=steps,
        dt_s=dt_s,
    )


def _run_transport_one_step_with_mass(
    tracer_field: TracerField,
    forcing: TransportForcing,
    dry_air_mass: np.ndarray,
    area: np.ndarray,
    *,
    dt_s: float,
    max_courant: float,
) -> TransportStepResult:
    delp = _dry_air_mass_to_pressure(dry_air_mass, area)
    xmass, ymass = horizontal_mass_flux_hpa(
        delp,
        forcing.u_m_s,
        forcing.v_m_s,
        forcing.lat_deg,
        dt_s=dt_s,
    )
    horizontal_state, horizontal_dry_air_mass = advect_horizontal_mass_flux(
        tracer_field,
        dry_air_mass,
        xmass,
        ymass,
        area,
        max_courant=max_courant,
    )
    zmass = vertical_mass_flux_hpa(dry_air_mass, horizontal_dry_air_mass, area)
    state, next_dry_air_mass = advect_vertical_mass_flux(
        horizontal_state,
        horizontal_dry_air_mass,
        zmass,
        area,
        max_courant=max_courant,
    )
    next_delp = _dry_air_mass_to_pressure(next_dry_air_mass, area)
    return TransportStepResult(
        state=state,
        dry_air_mass_kg=next_dry_air_mass,
        delp_dry_hpa=next_delp,
        xmass_hpa=xmass,
        ymass_hpa=ymass,
        zmass_hpa=zmass,
        initial_scalar_mass=scalar_mass_by_tracer(tracer_field.data, dry_air_mass),
        final_scalar_mass=scalar_mass_by_tracer(state.data, next_dry_air_mass),
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


def scalar_mass_by_tracer(field_data: np.ndarray, dry_air_mass_kg: np.ndarray) -> np.ndarray:
    return np.sum(np.asarray(field_data) * np.asarray(dry_air_mass_kg)[np.newaxis, ...], axis=(1, 2, 3, 4))


def _read_3d_time_slice(dataset: netCDF4.Dataset, variable_name: str, time_index: int) -> np.ndarray:
    return np.asarray(dataset.variables[variable_name][time_index : time_index + 1], dtype=np.float64)


def _map_met_levels_to_47(data: np.ndarray) -> np.ndarray:
    if data.shape[1] == FIXED_GRID["lev"]:
        return data
    if data.shape[1] == 72:
        mapped = np.empty((data.shape[0], FIXED_GRID["lev"], data.shape[2], data.shape[3]), dtype=np.float64)
        mapped[:, :36, :, :] = data[:, :36, :, :]
        for target_level, (start, end) in enumerate(MERRA2_72_TO_47_GROUPS, start=36):
            weights = MERRA2_72_AP_HPA[start:end] - MERRA2_72_AP_HPA[start + 1 : end + 1]
            mapped[:, target_level, :, :] = np.sum(
                data[:, start:end, :, :] * weights[np.newaxis, :, np.newaxis, np.newaxis],
                axis=1,
            ) / np.sum(weights)
        return mapped
    raise ValueError(f"cannot map {data.shape[1]} met levels to {FIXED_GRID['lev']}")


def _load_window_forcing(
    cache: dict[tuple[datetime, int], TransportForcing],
    met_root: str | Path,
    start: datetime,
    template_path: str | Path,
    *,
    step: int,
    dt_s: float,
    initial_met_time_index: int,
) -> TransportForcing:
    met_step = int((step * float(dt_s)) // (3.0 * 60.0 * 60.0))
    absolute_index = int(initial_met_time_index) + met_step
    timestamp = start + timedelta(days=absolute_index // 8)
    time_index = absolute_index % 8
    key = (datetime(timestamp.year, timestamp.month, timestamp.day), time_index)
    if key not in cache:
        cache[key] = load_transport_forcing(met_root, key[0], template_path, time_index=time_index)
    return cache[key]


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
