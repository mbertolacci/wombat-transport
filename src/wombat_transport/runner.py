from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from wombat_transport.emissions import EmissionsOperator, apply_emissions
from wombat_transport.fields import TracerField
from wombat_transport.grid import load_transport_grid
from wombat_transport.io import initialize_tracers
from wombat_transport.output import HistoryOutputManager, OutputSnapshot
from wombat_transport.run_config import (
    RunConfig,
    emissions_timestep_s,
    meteorology_initial_time_index,
    meteorology_root,
    simulation_end,
    simulation_start,
    transport_timestep_s,
)
from wombat_transport.species import load_species_database
from wombat_transport.transport import (
    TransportStageMass,
    dry_pressure_thickness_hpa,
    load_transport_forcing,
    run_transport_one_step,
)

@dataclass(frozen=True)
class EmissionsStep:
    timestamp: datetime


@dataclass(frozen=True)
class TracerSimulationResult:
    state: TracerField
    emissions_processed: tuple[EmissionsStep, ...]
    emitted_mass_by_tracer: np.ndarray
    transport_steps: int
    emissions_steps: int
    transport_dt_s: float
    emissions_dt_s: float
    stage_masses: tuple[TransportStageMass, ...]
    final_delp_dry_hpa: np.ndarray | None

    @property
    def total_emitted_mass(self) -> float:
        return float(np.sum(self.emitted_mass_by_tracer))

    @property
    def transport_operators(self) -> tuple[str, str, str]:
        return ("tpcore", "vdiff", "convection")


def run_tracer_simulation(config: RunConfig, *, max_steps: int | None = None) -> TracerSimulationResult:
    species = load_species_database(config.species_database)
    state = initialize_tracers(
        config.initial_restart,
        config.species_database,
        template_path=config.grid_template,
    )
    grid = load_transport_grid(config.grid_template)
    met_root = meteorology_root(config)
    start = simulation_start(config)
    end = simulation_end(config)
    transport_dt_s = float(transport_timestep_s(config))
    emissions_dt_s = float(emissions_timestep_s(config))
    _validate_timestep_schedule(transport_dt_s, emissions_dt_s)

    configured_emissions = _load_emissions_operator(config, species, grid)
    output_manager = HistoryOutputManager.from_run_config(config)

    forcing_cache = {}
    emitted_mass_by_tracer = np.zeros(len(species), dtype=np.float64)
    emissions_processed: list[EmissionsStep] = []
    stage_masses: list[TransportStageMass] = []
    final_delp_dry_hpa = None
    transport_steps = 0
    emissions_steps = 0

    current = start
    while current < end:
        if max_steps is not None and transport_steps >= max_steps:
            break

        forcing = _load_simulation_forcing(
            forcing_cache,
            met_root,
            start,
            grid,
            current,
            transport_dt_s=transport_dt_s,
            initial_met_time_index=meteorology_initial_time_index(config),
        )
        delp_dry_hpa = dry_pressure_thickness_hpa(forcing.surface_pressure_pa, grid.hyai_hpa, grid.hybi)

        elapsed_s = int(round((current - start).total_seconds()))
        if _is_time_for_emissions(elapsed_s, transport_dt_s, emissions_dt_s):
            emission_midpoint = current + timedelta(seconds=emissions_dt_s / 2.0)
            emissions = configured_emissions.evaluate(emission_midpoint)
            if has_invalid_emissions(emissions):
                raise ValueError(f"configured emissions contain invalid values at {emission_midpoint:%Y-%m-%d %H:%M}")
            emitted_mass_by_tracer += emitted_mass_by_tracer_for_step(emissions, emissions_dt_s)
            state = apply_emissions(state, emissions, delp_dry_hpa, species, emissions_dt_s)
            emissions_processed.append(EmissionsStep(timestamp=emission_midpoint))
            emissions_steps += 1

        transport_result = run_transport_one_step(state, forcing, grid, dt_s=transport_dt_s)
        state = transport_result.state
        stage_masses.extend(transport_result.stage_masses)
        final_delp_dry_hpa = transport_result.delp_dry_hpa
        transport_steps += 1
        step_end = current + timedelta(seconds=transport_dt_s)
        if output_manager is not None:
            output_manager.record_step(
                OutputSnapshot(
                    timestamp=step_end,
                    state=state,
                    delp_dry_hpa=transport_result.delp_dry_hpa,
                    forcing=forcing,
                )
            )
        current = step_end

    if output_manager is not None:
        output_manager.close()

    return TracerSimulationResult(
        state=state,
        emissions_processed=tuple(emissions_processed),
        emitted_mass_by_tracer=emitted_mass_by_tracer,
        transport_steps=transport_steps,
        emissions_steps=emissions_steps,
        transport_dt_s=transport_dt_s,
        emissions_dt_s=emissions_dt_s,
        stage_masses=tuple(stage_masses),
        final_delp_dry_hpa=final_delp_dry_hpa,
    )


def _load_emissions_operator(config: RunConfig, species, grid) -> EmissionsOperator:
    if isinstance(config.emissions, str):
        return EmissionsOperator.from_yaml(config.emissions, root=config.root, species=species, grid=grid)
    if isinstance(config.emissions, dict):
        raw: dict[str, Any] = dict(config.emissions)
        return EmissionsOperator.from_mapping(raw, root=config.root, species=species, grid=grid)
    raise TypeError("emissions must be a path string or an inline emissions mapping")


def has_invalid_emissions(emissions: TracerField) -> bool:
    """Return true when an emissions field contains fill values as data."""

    data = emissions.data
    return bool(np.any(~np.isfinite(data)) or np.any(np.abs(data) > 1.0e20))


def emitted_mass_by_tracer_for_step(emissions: TracerField, dt_s: float) -> np.ndarray:
    area = emissions.coords["AREA"]
    area_5d = area[np.newaxis, np.newaxis, :, :, np.newaxis]
    return np.sum(emissions.data * float(dt_s) * area_5d, axis=(0, 1, 2, 3))


def _validate_timestep_schedule(transport_dt_s: float, emissions_dt_s: float) -> None:
    transport = int(round(float(transport_dt_s)))
    emissions = int(round(float(emissions_dt_s)))
    if transport <= 0 or emissions <= 0:
        raise ValueError("transport and emissions timesteps must be positive")
    if not np.isclose(transport_dt_s, transport) or not np.isclose(emissions_dt_s, emissions):
        raise ValueError("transport and emissions timesteps must be whole seconds")
    if emissions % transport != 0:
        raise ValueError("emissions_timestep_s must be an integer multiple of transport_timestep_s")


def _is_time_for_emissions(elapsed_s: int, transport_dt_s: float, emissions_dt_s: float) -> bool:
    transport = int(round(float(transport_dt_s)))
    emissions = int(round(float(emissions_dt_s)))
    _validate_timestep_schedule(float(transport), float(emissions))
    multiplier = emissions // transport
    center = max(multiplier // 2, 1)
    return elapsed_s % emissions == (center - 1) * transport


def _load_simulation_forcing(
    cache: dict[tuple[datetime, int], object],
    met_root: Path,
    start: datetime,
    grid,
    current: datetime,
    *,
    transport_dt_s: float,
    initial_met_time_index: int,
):
    elapsed_s = (current - start).total_seconds()
    step = int(elapsed_s // float(transport_dt_s))
    met_step = int((step * float(transport_dt_s)) // (3.0 * 60.0 * 60.0))
    absolute_index = int(initial_met_time_index) + met_step
    timestamp = start + timedelta(days=absolute_index // 8)
    time_index = absolute_index % 8
    key = (datetime(timestamp.year, timestamp.month, timestamp.day), time_index)
    if key not in cache:
        cache[key] = load_transport_forcing(met_root, key[0], grid, time_index=time_index)
    return cache[key]
