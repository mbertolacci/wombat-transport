from __future__ import annotations

import numpy as np

from wombat_transport.constants import AIRMW_G_PER_MOL, G0_M_PER_S2
from wombat_transport.fields import TracerField
from wombat_transport.species import Species


def dry_air_mass_per_area(delp_dry_hpa: np.ndarray) -> np.ndarray:
    """Convert dry pressure thickness from hPa to kg dry air per m2."""

    return np.asarray(delp_dry_hpa, dtype=np.float64) * 100.0 / G0_M_PER_S2


def emission_increment_vv(
    emis_kg_m2_s: np.ndarray,
    delp_dry_hpa: np.ndarray,
    species: list[Species] | tuple[Species, ...],
    dt_s: float,
) -> np.ndarray:
    """Convert emissions flux into a dry volume mixing-ratio increment."""

    emissions = np.asarray(emis_kg_m2_s, dtype=np.float64)
    dry_air = dry_air_mass_per_area(delp_dry_hpa)
    if dry_air.ndim == 4:
        dry_air = dry_air[:, ::-1, :, :]
    elif dry_air.ndim == 3:
        dry_air = dry_air[::-1][np.newaxis, ...]
    else:
        raise ValueError(f"delp_dry_hpa must be 3-D or 4-D, found shape {dry_air.shape}")

    if emissions.ndim != 5:
        raise ValueError(f"emissions must be 5-D canonical tracer data, found shape {emissions.shape}")
    if emissions.shape[-1] != len(species):
        raise ValueError(
            f"emissions last dimension has {emissions.shape[-1]} tracers, "
            f"but {len(species)} species were supplied"
        )
    if dry_air.shape != emissions.shape[:-1]:
        raise ValueError(f"dry pressure shape {dry_air.shape} does not match emissions grid {emissions.shape[:-1]}")

    kgkg_dry = emissions * float(dt_s) / dry_air[..., np.newaxis]
    mw = np.asarray([item.molecular_weight_g for item in species], dtype=np.float64)
    return kgkg_dry * (AIRMW_G_PER_MOL / mw[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :])


def apply_emissions(
    tracer_field: TracerField,
    emissions: TracerField,
    delp_dry_hpa: np.ndarray,
    species: list[Species] | tuple[Species, ...],
    dt_s: float,
) -> TracerField:
    """Return a new tracer field after adding emissions increments."""

    species_names = tuple(item.name for item in species)
    if tracer_field.names != species_names:
        raise ValueError("tracer field names do not match species order")
    if emissions.names != species_names:
        raise ValueError("emission field names do not match species order")

    increment = emission_increment_vv(emissions.data, delp_dry_hpa, species, dt_s)
    return TracerField(
        names=tracer_field.names,
        data=tracer_field.data + increment,
        units=tracer_field.units,
        coords=tracer_field.coords,
    )
