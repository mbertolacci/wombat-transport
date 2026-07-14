from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from datetime import timezone
import json
import logging
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from wombat_transport.constants import AIRMW_G_PER_MOL
from wombat_transport.emissions import EmissionsOperator, SurfaceEmissions
from wombat_transport.fields import TracerField
from wombat_transport.grid import load_transport_grid
from wombat_transport.io import initialize_tracers
from wombat_transport.obsoperator import ObsOperatorManager
from wombat_transport.output import HistoryOutputManager, OutputSnapshot
from wombat_transport.run_config import (
    RunConfig,
    emissions_timestep_s,
    meteorology_chunk_multiple,
    meteorology_initial_time_index,
    meteorology_root,
    simulation_end,
    simulation_start,
    transport_timestep_s,
)
from wombat_transport.species import load_species_database
from wombat_transport.transport import (
    build_tpcore_static_terms,
    dry_air_mass_from_pressure,
    dry_pressure_thickness_from_surface_hpa,
    TransportForcingProvider,
    run_transport_one_step,
)
from wombat_transport.transport.numba_control import warn_if_transport_numba_disabled

logger = logging.getLogger(__name__)
RUN_METADATA_NAME = "wombat_run_metadata.json"


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
    final_delp_dry_hpa: np.ndarray | None

    @property
    def total_emitted_mass(self) -> float:
        return float(np.sum(self.emitted_mass_by_tracer))

    @property
    def transport_operators(self) -> tuple[str, str, str]:
        return ("tpcore", "vdiff", "convection")


def run_tracer_simulation(config: RunConfig, *, max_steps: int | None = None) -> TracerSimulationResult:
    warn_if_transport_numba_disabled(logger)
    logger.info("simulation_start name=%s max_steps=%s", config.name, max_steps)
    _write_run_metadata(config)
    species = load_species_database(config.species_database)
    logger.debug("loaded_species count=%d", len(species))
    state = initialize_tracers(
        config.initial_restart,
        config.species_database,
        template_path=config.grid_template,
    )
    surface_flux_to_vmr_factor = _surface_flux_to_vmr_factor(state, species)
    logger.debug("initialized_tracers shape=%s", state.shape)
    grid = load_transport_grid(config.grid_template)
    tpcore_static_terms = build_tpcore_static_terms(
        area_m2=grid.area_m2,
        hyai_hpa=grid.hyai_hpa,
        hybi=grid.hybi,
        lat_deg=grid.lat_deg,
    )
    met_root = meteorology_root(config)
    start = simulation_start(config)
    end = simulation_end(config)
    transport_dt_s = float(transport_timestep_s(config))
    emissions_dt_s = float(emissions_timestep_s(config))
    _validate_timestep_schedule(transport_dt_s, emissions_dt_s)

    configured_emissions = _load_emissions_operator(config, species, grid)
    logger.debug("loaded_emissions_operator")
    output_manager = HistoryOutputManager.from_run_config(config)
    logger.debug("output_manager enabled=%s", output_manager is not None)
    obsoperator_manager = ObsOperatorManager.from_run_config(
        config,
        tracer_names=state.names,
        grid=grid,
        transport_dt_s=transport_dt_s,
    )
    logger.debug("obsoperator_manager enabled=%s", obsoperator_manager is not None)

    forcing_provider = TransportForcingProvider(
        met_root,
        start,
        grid,
        initial_met_time_index=meteorology_initial_time_index(config),
        chunk_multiple=meteorology_chunk_multiple(config),
    )
    first_forcing = _load_simulation_forcing(
        forcing_provider,
        start,
        transport_dt_s=transport_dt_s,
    )
    dry_air_mass = _initial_dry_air_mass(config, first_forcing, grid)
    emitted_mass_by_tracer = np.zeros(len(species), dtype=np.float64)
    emissions_processed: list[EmissionsStep] = []
    final_delp_dry_hpa = None
    transport_steps = 0
    emissions_steps = 0
    active_emissions: SurfaceEmissions | None = None

    current = start
    while current < end:
        if max_steps is not None and transport_steps >= max_steps:
            logger.info("max_steps_stop steps=%d time=%s", transport_steps, current.isoformat())
            break

        logger.info("transport_timestep step=%d time=%s", transport_steps + 1, current.isoformat())
        logger.debug("loading_forcing step=%d time=%s", transport_steps + 1, current.isoformat())
        forcing = _load_simulation_forcing(
            forcing_provider,
            current,
            transport_dt_s=transport_dt_s,
        )
        elapsed_s = int(round((current - start).total_seconds()))
        if _is_time_for_emissions(elapsed_s, transport_dt_s, emissions_dt_s):
            emission_midpoint = current + timedelta(seconds=emissions_dt_s / 2.0)
            logger.debug("evaluating_emissions step=%d midpoint=%s", transport_steps + 1, emission_midpoint.isoformat())
            emissions = configured_emissions.evaluate_surface_flux(emission_midpoint)
            if has_invalid_emissions(emissions):
                raise ValueError(f"configured emissions contain invalid values at {emission_midpoint:%Y-%m-%d %H:%M}")
            emitted_mass_by_tracer += emitted_mass_by_tracer_for_step(emissions, emissions_dt_s)
            active_emissions = emissions
            emissions_processed.append(EmissionsStep(timestamp=emission_midpoint))
            emissions_steps += 1
            logger.debug("refreshed_emissions step=%d emissions_steps=%d", transport_steps + 1, emissions_steps)

        logger.debug("running_transport step=%d", transport_steps + 1)
        transport_result = run_transport_one_step(
            state,
            forcing,
            grid,
            dt_s=transport_dt_s,
            active_emissions=active_emissions,
            surface_flux_to_vmr_factor=surface_flux_to_vmr_factor,
            dry_air_mass_kg=dry_air_mass,
            tpcore_static_terms=tpcore_static_terms,
            validate_tpcore_branches=elapsed_s == 0 or elapsed_s % 10800 < int(round(transport_dt_s)),
            consume_input=True,
        )
        state = transport_result.state
        dry_air_mass = transport_result.dry_air_mass_kg
        final_delp_dry_hpa = transport_result.delp_dry_hpa
        step_end = current + timedelta(seconds=transport_dt_s)
        snapshot: OutputSnapshot | None = None
        if output_manager is not None or obsoperator_manager is not None:
            snapshot = OutputSnapshot(
                timestamp=step_end,
                state=state,
                delp_dry_hpa=transport_result.delp_dry_hpa,
                forcing=forcing,
            )
        if obsoperator_manager is not None:
            assert snapshot is not None
            logger.debug("sampling_obsoperator step=%d time_index=%d", transport_steps + 1, transport_steps)
            obsoperator_manager.sample(step_start=current, time_index=transport_steps, snapshot=snapshot)
        transport_steps += 1
        logger.debug("completed_transport step=%d", transport_steps)
        if output_manager is not None and snapshot is not None:
            logger.debug("recording_outputs step=%d timestamp=%s", transport_steps, step_end.isoformat())
            output_manager.record_step(snapshot)
        current = step_end

    if output_manager is not None:
        logger.debug("closing_outputs")
        output_manager.close()
    if obsoperator_manager is not None:
        logger.debug("closing_obsoperator")
        obsoperator_manager.close(boundary_time=current)

    logger.info(
        "simulation_complete transport_steps=%d emissions_steps=%d total_emitted_mass_kg=%.8e",
        transport_steps,
        emissions_steps,
        float(np.sum(emitted_mass_by_tracer)),
    )
    return TracerSimulationResult(
        state=state,
        emissions_processed=tuple(emissions_processed),
        emitted_mass_by_tracer=emitted_mass_by_tracer,
        transport_steps=transport_steps,
        emissions_steps=emissions_steps,
        transport_dt_s=transport_dt_s,
        emissions_dt_s=emissions_dt_s,
        final_delp_dry_hpa=final_delp_dry_hpa,
    )


def _write_run_metadata(config: RunConfig) -> None:
    metadata = {
        "schema_version": 1,
        "kind": "wombat-run",
        "run_name": config.name,
        "run_directory": str(config.root),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_provenance(config.root),
    }
    path = config.root / RUN_METADATA_NAME
    try:
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.warning("failed_to_write_run_metadata path=%s error=%s", path, exc)


def _git_provenance(path: Path) -> dict[str, Any]:
    root = _git_output(path, "rev-parse", "--show-toplevel")
    if root is None:
        fallback = Path(__file__).resolve().parents[2]
        root = _git_output(fallback, "rev-parse", "--show-toplevel")
    if root is None:
        return {"available": False}
    commit = _git_output(Path(root), "rev-parse", "HEAD")
    tracked_dirty = _git_returncode(Path(root), "diff-index", "--quiet", "HEAD", "--") == 1
    untracked_output = _git_output(Path(root), "ls-files", "--others", "--exclude-standard")
    untracked_present = bool(untracked_output)
    return {
        "available": True,
        "root": root,
        "commit": commit,
        "dirty": bool(tracked_dirty or untracked_present),
        "tracked_dirty": bool(tracked_dirty),
        "untracked_present": untracked_present,
    }


def _git_output(path: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), *args],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_returncode(path: Path, *args: str) -> int:
    completed = subprocess.run(
        ["git", "-C", str(path), *args],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode


def _load_emissions_operator(config: RunConfig, species, grid) -> EmissionsOperator:
    if isinstance(config.emissions, str):
        return EmissionsOperator.from_yaml(config.emissions, root=config.root, species=species, grid=grid)
    if isinstance(config.emissions, dict):
        raw: dict[str, Any] = dict(config.emissions)
        return EmissionsOperator.from_mapping(raw, root=config.root, species=species, grid=grid)
    raise TypeError("emissions must be a path string or an inline emissions mapping")


def _surface_flux_to_vmr_factor(state: TracerField, species) -> np.ndarray:
    species_by_name = {item.name: item for item in species}
    factors = []
    for name in state.names:
        if name not in species_by_name:
            raise ValueError(f"tracer {name!r} is missing from the species database")
        factors.append(AIRMW_G_PER_MOL / species_by_name[name].molecular_weight_g)
    return np.asarray(factors, dtype=np.float64)


def _initial_dry_air_mass(config: RunConfig, forcing, grid) -> np.ndarray:
    delp = dry_pressure_thickness_from_surface_hpa(
        forcing.dry_surface_pressure_start_hpa,
        grid.hyai_hpa,
        grid.hybi,
    )
    return dry_air_mass_from_pressure(delp, grid.area_m2)


def has_invalid_emissions(emissions: TracerField | SurfaceEmissions) -> bool:
    """Return true when an emissions field contains fill values as data."""

    data = emissions.data
    return bool(np.any(~np.isfinite(data)) or np.any(np.abs(data) > 1.0e20))


def emitted_mass_by_tracer_for_step(emissions: TracerField | SurfaceEmissions, dt_s: float) -> np.ndarray:
    area = emissions.coords["AREA"]
    if emissions.data.ndim == 3:
        area_3d = area[:, :, np.newaxis]
        return np.sum(emissions.data * float(dt_s) * area_3d, axis=(0, 1))
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
    forcing_provider: TransportForcingProvider,
    current: datetime,
    *,
    transport_dt_s: float,
):
    return forcing_provider.forcing_for_step(current, dt_s=transport_dt_s)
