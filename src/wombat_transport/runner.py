from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
import re

import netCDF4
import numpy as np

from wombat_transport.emissions import apply_emissions
from wombat_transport.fields import TracerField, public_tracer5_to_canonical
from wombat_transport.grid import load_transport_grid
from wombat_transport.io import initialize_tracers, load_hemco_emissions
from wombat_transport.run_config import RunConfig
from wombat_transport.species import load_species_database
from wombat_transport.transport import (
    TransportStageMass,
    dry_pressure_thickness_hpa,
    load_transport_forcing,
    run_transport_one_step,
)

HEMCO_DIAGNOSTIC_RE = re.compile(r"HEMCO_diagnostics\.(\d{12})\.nc$")
CONFIG_TIME_FORMAT = "%Y-%m-%d %H:%M"
FILE_TIME_FORMAT = "%Y%m%d%H%M"


@dataclass(frozen=True)
class HemcoDiagnosticFile:
    path: Path
    timestamp: datetime


@dataclass(frozen=True)
class EmissionsReplayResult:
    state: TracerField
    discovered_files: tuple[HemcoDiagnosticFile, ...]
    processed_files: tuple[HemcoDiagnosticFile, ...]
    skipped_files: tuple[HemcoDiagnosticFile, ...]
    emitted_mass_by_tracer: np.ndarray

    @property
    def total_emitted_mass(self) -> float:
        return float(np.sum(self.emitted_mass_by_tracer))


@dataclass(frozen=True)
class TracerSimulationResult:
    state: TracerField
    discovered_files: tuple[HemcoDiagnosticFile, ...]
    processed_files: tuple[HemcoDiagnosticFile, ...]
    skipped_files: tuple[HemcoDiagnosticFile, ...]
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


def discover_hemco_diagnostics(config: RunConfig) -> tuple[HemcoDiagnosticFile, ...]:
    pattern = _resolve_config_path(config.root, _hemco_glob(config))
    start = _hemco_discovery_start(config)
    end = _hemco_discovery_end(config)

    files: list[HemcoDiagnosticFile] = []
    for raw_path in glob(str(pattern)):
        path = Path(raw_path)
        timestamp = parse_hemco_timestamp(path)
        if start <= timestamp <= end:
            files.append(HemcoDiagnosticFile(path=path.resolve(), timestamp=timestamp))
    return tuple(sorted(files, key=lambda item: item.timestamp))


def parse_hemco_timestamp(path: str | Path) -> datetime:
    match = HEMCO_DIAGNOSTIC_RE.search(Path(path).name)
    if match is None:
        raise ValueError(f"not a HEMCO diagnostic filename: {path}")
    return datetime.strptime(match.group(1), FILE_TIME_FORMAT)


def run_emissions_replay(config: RunConfig, *, max_steps: int | None = None) -> EmissionsReplayResult:
    species = load_species_database(config.species_database)
    state = initialize_tracers(
        config.initial_restart,
        config.species_database,
        template_path=config.grid_template,
    )
    with netCDF4.Dataset(config.grid_template) as dataset:
        delp_dry_hpa = np.asarray(dataset.variables["Met_DELPDRY"][:])

    discovered = discover_hemco_diagnostics(config)
    selected = discovered if max_steps is None else discovered[:max_steps]
    dt_s = float(config.replay.get("dt_s", 3600.0))
    emitted_mass_by_tracer = np.zeros(len(species), dtype=np.float64)
    processed: list[HemcoDiagnosticFile] = []
    skipped: list[HemcoDiagnosticFile] = []

    for diagnostic in selected:
        emissions = load_hemco_emissions(diagnostic.path)
        if has_invalid_emissions(emissions):
            skipped.append(diagnostic)
            continue

        emitted_mass_by_tracer += emitted_mass_by_tracer_for_step(emissions, dt_s)
        state = apply_emissions(state, emissions, delp_dry_hpa, species, dt_s)
        processed.append(diagnostic)

    return EmissionsReplayResult(
        state=state,
        discovered_files=discovered,
        processed_files=tuple(processed),
        skipped_files=tuple(skipped),
        emitted_mass_by_tracer=emitted_mass_by_tracer,
    )


def run_tracer_simulation(config: RunConfig, *, max_steps: int | None = None) -> TracerSimulationResult:
    species = load_species_database(config.species_database)
    state = initialize_tracers(
        config.initial_restart,
        config.species_database,
        template_path=config.grid_template,
    )
    grid = load_transport_grid(config.grid_template)
    met_root = _resolve_config_path(config.root, _meteorology_root(config))
    start = _simulation_start(config)
    end = _simulation_end(config)
    transport_dt_s = float(_transport_timestep_s(config))
    emissions_dt_s = float(_emissions_timestep_s(config))
    _validate_timestep_schedule(transport_dt_s, emissions_dt_s)

    diagnostics = discover_hemco_diagnostics(config)
    emissions_cache: dict[Path, TracerField] = {}
    forcing_cache = {}
    emitted_mass_by_tracer = np.zeros(len(species), dtype=np.float64)
    processed: list[HemcoDiagnosticFile] = []
    skipped: list[HemcoDiagnosticFile] = []
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
            initial_met_time_index=int(config.transport.get("met_time_index", 0)),
        )
        delp_dry_hpa = dry_pressure_thickness_hpa(forcing.surface_pressure_pa, grid.hyai_hpa, grid.hybi)

        elapsed_s = int(round((current - start).total_seconds()))
        if _is_time_for_emissions(elapsed_s, transport_dt_s, emissions_dt_s):
            emission_midpoint = current + timedelta(seconds=emissions_dt_s / 2.0)
            diagnostic = _select_hemco_diagnostic(diagnostics, emission_midpoint)
            emissions = _load_cached_emissions(diagnostic, emissions_cache, species)
            if has_invalid_emissions(emissions):
                skipped.append(diagnostic)
            else:
                emitted_mass_by_tracer += emitted_mass_by_tracer_for_step(emissions, emissions_dt_s)
                state = apply_emissions(state, emissions, delp_dry_hpa, species, emissions_dt_s)
                processed.append(diagnostic)
            emissions_steps += 1

        transport_result = run_transport_one_step(state, forcing, grid, dt_s=transport_dt_s)
        state = transport_result.state
        stage_masses.extend(transport_result.stage_masses)
        final_delp_dry_hpa = transport_result.delp_dry_hpa
        transport_steps += 1
        current += timedelta(seconds=transport_dt_s)

    return TracerSimulationResult(
        state=state,
        discovered_files=diagnostics,
        processed_files=tuple(processed),
        skipped_files=tuple(skipped),
        emitted_mass_by_tracer=emitted_mass_by_tracer,
        transport_steps=transport_steps,
        emissions_steps=emissions_steps,
        transport_dt_s=transport_dt_s,
        emissions_dt_s=emissions_dt_s,
        stage_masses=tuple(stage_masses),
        final_delp_dry_hpa=final_delp_dry_hpa,
    )


def has_invalid_emissions(emissions: TracerField) -> bool:
    """Return true when a HEMCO diagnostic contains fill values as data."""

    data = emissions.data
    return bool(np.any(~np.isfinite(data)) or np.any(np.abs(data) > 1.0e20))


def emitted_mass_by_tracer_for_step(emissions: TracerField, dt_s: float) -> np.ndarray:
    area = emissions.coords["AREA"]
    area_5d = area[np.newaxis, np.newaxis, :, :, np.newaxis]
    return np.sum(emissions.data * float(dt_s) * area_5d, axis=(0, 1, 2, 3))


def _resolve_config_path(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def _hemco_glob(config: RunConfig) -> str:
    if "glob" in config.emissions:
        return str(config.emissions["glob"])
    return str(config.replay["hemco_glob"])


def _hemco_discovery_start(config: RunConfig) -> datetime:
    if "start" in config.replay:
        return datetime.strptime(config.replay["start"], CONFIG_TIME_FORMAT)
    return _simulation_start(config)


def _hemco_discovery_end(config: RunConfig) -> datetime:
    if "end" in config.replay:
        return datetime.strptime(config.replay["end"], CONFIG_TIME_FORMAT)
    return _simulation_end(config)


def _simulation_start(config: RunConfig) -> datetime:
    value = config.simulation.get("start", config.transport.get("start"))
    if value is None:
        raise KeyError("simulation.start is required")
    return datetime.strptime(str(value), CONFIG_TIME_FORMAT)


def _simulation_end(config: RunConfig) -> datetime:
    value = config.simulation.get("end", config.replay.get("end"))
    if value is None:
        raise KeyError("simulation.end is required")
    return datetime.strptime(str(value), CONFIG_TIME_FORMAT)


def _transport_timestep_s(config: RunConfig) -> float:
    return float(config.simulation.get("transport_timestep_s", config.transport.get("dt_s", 600.0)))


def _emissions_timestep_s(config: RunConfig) -> float:
    return float(config.simulation.get("emissions_timestep_s", config.replay.get("dt_s", _transport_timestep_s(config))))


def _meteorology_root(config: RunConfig) -> str:
    return str(config.meteorology.get("root", config.transport["met_root"]))


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


def _select_hemco_diagnostic(
    diagnostics: tuple[HemcoDiagnosticFile, ...],
    valid_time: datetime,
) -> HemcoDiagnosticFile:
    if not diagnostics:
        raise FileNotFoundError("no HEMCO diagnostic files were discovered")
    if len(diagnostics) == 1:
        return diagnostics[0]
    nearest = min(diagnostics, key=lambda item: abs(item.timestamp - valid_time))
    cadence = min(
        abs(diagnostics[index + 1].timestamp - diagnostics[index].timestamp)
        for index in range(len(diagnostics) - 1)
    )
    if abs(nearest.timestamp - valid_time) > cadence / 2:
        raise FileNotFoundError(f"no HEMCO diagnostic is valid for {valid_time:%Y-%m-%d %H:%M}")
    return nearest


def _load_cached_emissions(
    diagnostic: HemcoDiagnosticFile,
    cache: dict[Path, TracerField],
    species,
) -> TracerField:
    if diagnostic.path not in cache:
        cache[diagnostic.path] = _load_hemco_emissions_for_species(diagnostic.path, species)
    return cache[diagnostic.path]


def _load_hemco_emissions_for_species(path: Path, species) -> TracerField:
    try:
        emissions = load_hemco_emissions(path)
    except KeyError:
        emissions = _load_total_hemco_emissions(path, species)
    species_names = tuple(item.name for item in species)
    if emissions.names != species_names:
        raise ValueError(f"emission field names {emissions.names} do not match species order {species_names}")
    return emissions


def _load_total_hemco_emissions(path: Path, species) -> TracerField:
    with netCDF4.Dataset(path) as dataset:
        lat = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(dataset.variables["lon"][:], dtype=np.float64)
        lev = np.asarray(dataset.variables["lev"][:], dtype=np.float64)
        area = np.asarray(dataset.variables["AREA"][:], dtype=np.float64)
        ntime = len(dataset.dimensions["time"])
        nlev = lev.size
        public = np.zeros((len(species), ntime, nlev, lat.size, lon.size), dtype=np.float64)
        units: list[str] = []
        for index, item in enumerate(species):
            variable_name = f"Emis{item.name}_Total"
            if variable_name not in dataset.variables:
                raise KeyError(f"{path} is missing variable {variable_name}")
            variable = dataset.variables[variable_name]
            values = np.asarray(variable[:], dtype=np.float64)
            if values.ndim == 3:
                public[index, :, 0, :, :] = values
            elif values.ndim == 4:
                public[index, :, :, :, :] = values
            else:
                raise ValueError(f"{variable_name} must be 3-D or 4-D, found shape {values.shape}")
            units.append(str(getattr(variable, "units", "")))
    return TracerField(
        names=tuple(item.name for item in species),
        data=public_tracer5_to_canonical(public),
        units=tuple(units),
        coords={"lev": lev, "lat": lat, "lon": lon, "AREA": area},
    )


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
