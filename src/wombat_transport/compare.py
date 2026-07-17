from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wombat_transport.constants import AIRMW_G_PER_MOL, G0_M_PER_S2
from wombat_transport.fields import TracerField
from wombat_transport.species import Species


@dataclass(frozen=True)
class ComparisonMetrics:
    names: tuple[str, ...]
    max_abs_error: np.ndarray
    mean_abs_error: np.ndarray
    candidate_mass_kg: np.ndarray | None = None
    reference_mass_kg: np.ndarray | None = None
    mass_error_kg: np.ndarray | None = None
    max_abs_column_error_kg: np.ndarray | None = None
    mean_abs_column_error_kg: np.ndarray | None = None


def compare_to_time_slice(
    candidate: TracerField,
    reference: TracerField,
    *,
    reference_time_index: int = -1,
    species: list[Species] | tuple[Species, ...] | None = None,
    delp_dry_hpa: np.ndarray | None = None,
    area_m2: np.ndarray | None = None,
) -> ComparisonMetrics:
    """Compare a single-time candidate field to one reference time slice."""

    if candidate.names != reference.names:
        raise ValueError("candidate and reference tracer names do not match")
    candidate_data = candidate.to_canonical()
    reference_data = reference.to_canonical()
    if reference_data.shape[0] == 0:
        raise ValueError("reference field has no time records")

    reference_slice = reference_data[[reference_time_index], ...]
    if candidate_data.shape != reference_slice.shape:
        raise ValueError(
            f"candidate shape {candidate_data.shape} does not match "
            f"reference slice shape {reference_slice.shape}"
        )

    abs_error = np.abs(candidate_data - reference_slice)
    reduce_axes = tuple(range(0, abs_error.ndim - 1))
    mass_metrics = _mass_metrics(candidate_data, reference_slice, species, delp_dry_hpa, area_m2)
    return ComparisonMetrics(
        names=candidate.names,
        max_abs_error=np.max(abs_error, axis=reduce_axes),
        mean_abs_error=np.mean(abs_error, axis=reduce_axes),
        **mass_metrics,
    )


def format_metrics(metrics: ComparisonMetrics, *, limit: int = 8) -> str:
    has_mass = metrics.mass_error_kg is not None
    header = ["tracer", "max_abs_error", "mean_abs_error"]
    if has_mass:
        header.extend(
            [
                "candidate_mass_kg",
                "reference_mass_kg",
                "mass_error_kg",
                "max_abs_column_error_kg",
                "mean_abs_column_error_kg",
            ]
        )
    lines = [",".join(header)]
    for index, name in enumerate(metrics.names[:limit]):
        row = [
            name,
            f"{metrics.max_abs_error[index]:.8e}",
            f"{metrics.mean_abs_error[index]:.8e}",
        ]
        if has_mass:
            assert metrics.candidate_mass_kg is not None
            assert metrics.reference_mass_kg is not None
            assert metrics.mass_error_kg is not None
            assert metrics.max_abs_column_error_kg is not None
            assert metrics.mean_abs_column_error_kg is not None
            row.extend(
                [
                    f"{metrics.candidate_mass_kg[index]:.8e}",
                    f"{metrics.reference_mass_kg[index]:.8e}",
                    f"{metrics.mass_error_kg[index]:.8e}",
                    f"{metrics.max_abs_column_error_kg[index]:.8e}",
                    f"{metrics.mean_abs_column_error_kg[index]:.8e}",
                ]
            )
        lines.append(",".join(row))
    if len(metrics.names) > limit:
        lines.append(f"... {len(metrics.names) - limit} more tracers")
    return "\n".join(lines)


def tracer_mass_kg(
    field_data: np.ndarray,
    species: list[Species] | tuple[Species, ...],
    delp_dry_hpa: np.ndarray,
    area_m2: np.ndarray,
) -> np.ndarray:
    """Return total species mass by tracer for dry mixing-ratio fields."""

    gridbox_mass = tracer_gridbox_mass_kg(field_data, species, delp_dry_hpa, area_m2)
    return np.sum(gridbox_mass, axis=(0, 1, 2, 3))


def tracer_gridbox_mass_kg(
    field_data: np.ndarray,
    species: list[Species] | tuple[Species, ...],
    delp_dry_hpa: np.ndarray,
    area_m2: np.ndarray,
) -> np.ndarray:
    """Convert dry volume mixing ratio fields to grid-box species mass."""

    data = np.asarray(field_data, dtype=np.float64)
    if data.ndim != 5:
        raise ValueError(f"expected field data to be 5-D, found {data.ndim}-D")
    if data.shape[-1] != len(species):
        raise ValueError("field tracer count does not match species count")

    dry_air = np.asarray(delp_dry_hpa, dtype=np.float64) * 100.0 / G0_M_PER_S2
    if dry_air.ndim == 4:
        dry_air = dry_air[:, ::-1, :, :]
    elif dry_air.ndim == 3:
        dry_air = dry_air[::-1][np.newaxis, ...]
    else:
        raise ValueError(f"dry pressure must be 3-D or 4-D, found shape {dry_air.shape}")
    area = np.asarray(area_m2, dtype=np.float64)
    if dry_air.shape != data.shape[:-1]:
        raise ValueError(f"dry pressure shape {dry_air.shape} does not match field shape {data.shape[:-1]}")
    if area.shape != data.shape[2:4]:
        raise ValueError(f"area shape {area.shape} does not match horizontal grid {data.shape[2:4]}")

    species_mw = np.asarray([item.molecular_weight_g for item in species], dtype=np.float64)
    species_mw = species_mw[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    dry_air = dry_air[..., np.newaxis]
    area = area[np.newaxis, np.newaxis, :, :, np.newaxis]
    return data * (species_mw / AIRMW_G_PER_MOL) * dry_air * area


def _mass_metrics(
    candidate_data: np.ndarray,
    reference_data: np.ndarray,
    species: list[Species] | tuple[Species, ...] | None,
    delp_dry_hpa: np.ndarray | None,
    area_m2: np.ndarray | None,
) -> dict[str, np.ndarray | None]:
    if species is None and delp_dry_hpa is None and area_m2 is None:
        return {}
    if species is None or delp_dry_hpa is None or area_m2 is None:
        raise ValueError("species, delp_dry_hpa, and area_m2 are all required for mass metrics")

    candidate_gridbox = tracer_gridbox_mass_kg(candidate_data, species, delp_dry_hpa, area_m2)
    reference_gridbox = tracer_gridbox_mass_kg(reference_data, species, delp_dry_hpa, area_m2)
    gridbox_error = candidate_gridbox - reference_gridbox
    column_error = np.sum(gridbox_error, axis=1)
    column_reduce_axes = (0, 1, 2)
    return {
        "candidate_mass_kg": np.sum(candidate_gridbox, axis=(0, 1, 2, 3)),
        "reference_mass_kg": np.sum(reference_gridbox, axis=(0, 1, 2, 3)),
        "mass_error_kg": np.sum(gridbox_error, axis=(0, 1, 2, 3)),
        "max_abs_column_error_kg": np.max(np.abs(column_error), axis=column_reduce_axes),
        "mean_abs_column_error_kg": np.mean(np.abs(column_error), axis=column_reduce_axes),
    }
