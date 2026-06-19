from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Species:
    name: str
    molecular_weight_g: float
    background_vv: float
    full_name: str


def load_species_database(path: str | Path) -> list[Species]:
    """Load the subset of GEOS-Chem species metadata needed for transport."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw: dict[str, dict[str, Any]] = yaml.safe_load(handle) or {}

    species: list[Species] = []
    for name, attrs in raw.items():
        if not attrs.get("Is_Tracer", False):
            continue
        species.append(
            Species(
                name=name,
                molecular_weight_g=float(attrs["MW_g"]),
                background_vv=float(attrs["Background_VV"]),
                full_name=str(attrs.get("FullName", name)),
            )
        )
    return species
