from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4
import numpy as np

from wombat_transport.fields import TracerField
from wombat_transport.transport.forcing import TransportForcing, load_transport_forcing
from wombat_transport.transport.metrics import scalar_mass_by_tracer
from wombat_transport.transport.pjc import pjc_mass_flux_hpa
from wombat_transport.transport.pressure import (
    _dry_air_mass_to_pressure,
    dry_air_mass_from_pressure,
    dry_pressure_thickness_hpa,
)
from wombat_transport.transport.scaffold import (
    advect_horizontal_mass_flux,
    advect_vertical_mass_flux,
    horizontal_mass_flux_hpa,
    vertical_mass_flux_hpa,
)

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

    surface_pressure_hpa = forcing.surface_pressure_pa[0] / 100.0
    delp = dry_pressure_thickness_hpa(forcing.surface_pressure_pa, hyai, hybi)
    dry_air_mass = dry_air_mass_from_pressure(delp, area)
    xmass_single, ymass_single = pjc_mass_flux_hpa(
        p1_hpa=surface_pressure_hpa,
        p2_hpa=surface_pressure_hpa,
        u_m_s=forcing.u_m_s[0],
        v_m_s=forcing.v_m_s[0],
        area_m2=area,
        hyai_hpa=hyai,
        hybi=hybi,
        lat_deg=forcing.lat_deg,
        dt_s=dt_s,
    )
    xmass = xmass_single[np.newaxis, ...]
    ymass = ymass_single[np.newaxis, ...]
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
