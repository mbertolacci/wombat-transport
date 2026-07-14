from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yaml12 import read_yaml


@dataclass(frozen=True)
class Species:
    name: str
    molecular_weight_g: float
    background_vv: float
    full_name: str


def load_species_database(path: str | Path) -> list[Species]:
    """Load the subset of GEOS-Chem species metadata needed for transport."""

    path = Path(path)
    raw: dict[str, dict[str, Any]] = read_yaml(path) or {}

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
