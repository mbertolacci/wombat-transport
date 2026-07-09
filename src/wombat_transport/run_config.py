from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

CONFIG_TIME_FORMAT = "%Y-%m-%d %H:%M"


@dataclass(frozen=True)
class RunConfig:
    name: str
    root: Path
    source_run_dir: Path
    species_database: Path
    initial_restart: Path | None
    grid_template: Path
    output_dir: Path
    diagnostics: dict[str, str]
    comparison: dict[str, Any]
    simulation: dict[str, Any]
    meteorology: dict[str, Any]
    emissions: str | dict[str, Any]
    validation: dict[str, Any]


def load_run_config(path: str | Path) -> RunConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    root = path.parent

    def resolve(value: str | None) -> Path | None:
        if value is None:
            return None
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = root / candidate
        return candidate.resolve()

    return RunConfig(
        name=str(raw["name"]),
        root=root.resolve(),
        source_run_dir=resolve(raw["source_run_dir"]),  # type: ignore[arg-type]
        species_database=resolve(raw["species_database"]),  # type: ignore[arg-type]
        initial_restart=resolve(raw.get("initial_restart")),
        grid_template=resolve(raw["grid_template"]),  # type: ignore[arg-type]
        output_dir=resolve(raw["output_dir"]),  # type: ignore[arg-type]
        diagnostics=dict(raw.get("diagnostics", {})),
        comparison=dict(raw.get("comparison", {})),
        simulation=dict(raw.get("simulation", {})),
        meteorology=dict(raw.get("meteorology", {})),
        emissions=raw.get("emissions", {}),
        validation=dict(raw.get("validation", {})),
    )


def resolve_config_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def simulation_start(config: RunConfig) -> datetime:
    value = config.simulation.get("start")
    if value is None:
        raise KeyError("simulation.start is required")
    return datetime.strptime(str(value), CONFIG_TIME_FORMAT)


def simulation_end(config: RunConfig) -> datetime:
    value = config.simulation.get("end")
    if value is None:
        raise KeyError("simulation.end is required")
    return datetime.strptime(str(value), CONFIG_TIME_FORMAT)


def transport_timestep_s(config: RunConfig) -> float:
    return float(config.simulation.get("transport_timestep_s", 600.0))


def emissions_timestep_s(config: RunConfig) -> float:
    return float(config.simulation.get("emissions_timestep_s", transport_timestep_s(config)))


def meteorology_root(config: RunConfig) -> Path:
    if "root" not in config.meteorology:
        raise KeyError("meteorology.root is required")
    return resolve_config_path(config.root, str(config.meteorology["root"]))


def meteorology_initial_time_index(config: RunConfig) -> int:
    return int(config.meteorology.get("initial_time_index", 0))
