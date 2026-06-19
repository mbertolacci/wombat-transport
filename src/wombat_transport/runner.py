from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from glob import glob
from pathlib import Path
import re

import netCDF4
import numpy as np

from wombat_transport.emissions import apply_emissions
from wombat_transport.fields import TracerField
from wombat_transport.io import initialize_tracers, load_hemco_emissions
from wombat_transport.run_config import RunConfig
from wombat_transport.species import load_species_database

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


def discover_hemco_diagnostics(config: RunConfig) -> tuple[HemcoDiagnosticFile, ...]:
    replay = config.replay
    pattern = _resolve_config_path(config.root, replay["hemco_glob"])
    start = datetime.strptime(replay["start"], CONFIG_TIME_FORMAT)
    end = datetime.strptime(replay["end"], CONFIG_TIME_FORMAT)

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


def has_invalid_emissions(emissions: TracerField) -> bool:
    """Return true when a HEMCO diagnostic contains fill values as data."""

    data = emissions.data
    return bool(np.any(~np.isfinite(data)) or np.any(np.abs(data) > 1.0e20))


def emitted_mass_by_tracer_for_step(emissions: TracerField, dt_s: float) -> np.ndarray:
    area = emissions.coords["AREA"]
    area_5d = area[np.newaxis, np.newaxis, np.newaxis, :, :]
    return np.sum(emissions.data * float(dt_s) * area_5d, axis=(1, 2, 3, 4))


def _resolve_config_path(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path
