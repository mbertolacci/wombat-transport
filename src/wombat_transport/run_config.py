from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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
    replay: dict[str, Any]
    comparison: dict[str, Any]


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
        replay=dict(raw.get("replay", {})),
        comparison=dict(raw.get("comparison", {})),
    )
