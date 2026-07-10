from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np

from wombat_transport.grid import TransportGrid
from wombat_transport.transport.pressure import dry_surface_pressure_hpa, wet_surface_pressure_hpa

from wombat_transport.io import FIXED_GRID

MERRA2_FILENAME = "MERRA2.{date}.{collection}.2x25.nc4"

MERRA2_72_AP_HPA = np.array(
    [
        0.000000e00,
        4.804826e-02,
        6.593752e00,
        1.313480e01,
        1.961311e01,
        2.609201e01,
        3.257081e01,
        3.898201e01,
        4.533901e01,
        5.169611e01,
        5.805321e01,
        6.436264e01,
        7.062198e01,
        7.883422e01,
        8.909992e01,
        9.936521e01,
        1.091817e02,
        1.189586e02,
        1.286959e02,
        1.429100e02,
        1.562600e02,
        1.696090e02,
        1.816190e02,
        1.930970e02,
        2.032590e02,
        2.121500e02,
        2.187760e02,
        2.238980e02,
        2.243630e02,
        2.168650e02,
        2.011920e02,
        1.769300e02,
        1.503930e02,
        1.278370e02,
        1.086630e02,
        9.236572e01,
        7.851231e01,
        6.660341e01,
        5.638791e01,
        4.764391e01,
        4.017541e01,
        3.381001e01,
        2.836781e01,
        2.373041e01,
        1.979160e01,
        1.645710e01,
        1.364340e01,
        1.127690e01,
        9.292942e00,
        7.619842e00,
        6.216801e00,
        5.046801e00,
        4.076571e00,
        3.276431e00,
        2.620211e00,
        2.084970e00,
        1.650790e00,
        1.300510e00,
        1.019440e00,
        7.951341e-01,
        6.167791e-01,
        4.758061e-01,
        3.650411e-01,
        2.785261e-01,
        2.113490e-01,
        1.594950e-01,
        1.197030e-01,
        8.934502e-02,
        6.600001e-02,
        4.758501e-02,
        3.270000e-02,
        2.000000e-02,
        1.000000e-02,
    ],
    dtype=np.float64,
)

MERRA2_72_TO_47_GROUPS = (
    (36, 38),
    (38, 40),
    (40, 42),
    (42, 44),
    (44, 48),
    (48, 52),
    (52, 56),
    (56, 60),
    (60, 64),
    (64, 68),
    (68, 72),
)

MERRA2_72_TO_47_MAPPING = "collapse_72_to_47_pressure_weighted"

@dataclass(frozen=True)
class TransportForcing:
    """Meteorological forcing mapped onto the prototype 47-level grid."""

    u_m_s: np.ndarray
    v_m_s: np.ndarray
    omega_pa_s: np.ndarray
    surface_pressure_start_pa: np.ndarray
    surface_pressure_pa: np.ndarray
    restart_surface_pressure_pa: np.ndarray
    wet_surface_pressure_start_hpa: np.ndarray
    wet_surface_pressure_hpa: np.ndarray
    restart_wet_surface_pressure_hpa: np.ndarray
    dry_surface_pressure_start_hpa: np.ndarray
    dry_surface_pressure_hpa: np.ndarray
    restart_dry_surface_pressure_hpa: np.ndarray
    specific_humidity_kg_kg: np.ndarray
    restart_specific_humidity_kg_kg: np.ndarray
    temperature_k: np.ndarray
    restart_temperature_k: np.ndarray
    pbl_height_m: np.ndarray
    sensible_heat_flux_w_m2: np.ndarray
    latent_heat_flux_w_m2: np.ndarray
    friction_velocity_m_s: np.ndarray
    convective_mass_flux_kg_m2_s: np.ndarray
    convective_detrainment_kg_m2_s: np.ndarray
    convective_precip_prod_kg_kg_s: np.ndarray
    convective_precip_reevap_kg_kg_s: np.ndarray
    convective_ice_flux_kg_m2_s: np.ndarray
    convective_liquid_flux_kg_m2_s: np.ndarray
    convective_precip_mm_day: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    vertical_mapping: str
    a1_path: Path
    a3dyn_path: Path
    a3mstc_path: Path
    a3mste_path: Path
    i3_path: Path


@dataclass(frozen=True)
class _A1Fields:
    pblh: np.ndarray
    hflux: np.ndarray
    eflux: np.ndarray
    ustar: np.ndarray
    precccon: np.ndarray
    path: Path


@dataclass(frozen=True)
class _A3Fields:
    u: np.ndarray
    v: np.ndarray
    omega: np.ndarray
    dtrain: np.ndarray
    dqrcu: np.ndarray
    reevapcn: np.ndarray
    cmfmc: np.ndarray
    pficu: np.ndarray
    pflcu: np.ndarray
    a3dyn_path: Path
    a3mstc_path: Path
    a3mste_path: Path


@dataclass(frozen=True)
class _I3Fields:
    surface_pressure: np.ndarray
    qv: np.ndarray
    temperature: np.ndarray
    path: Path


ForcingRecordCache = dict[tuple[str, datetime, int], Any]


def load_transport_forcing(
    met_root: str | Path,
    timestamp: datetime,
    grid: TransportGrid,
    *,
    time_index: int = 0,
) -> TransportForcing:
    """Load MERRA2 forcing for one day and map 72 met levels to 47 levels."""

    met_root = Path(met_root)
    timestamp = datetime(timestamp.year, timestamp.month, timestamp.day)
    a1 = _load_a1_fields(met_root, timestamp, grid, int(time_index) * 3, None)
    a3 = _load_a3_fields(met_root, timestamp, grid, int(time_index), None)
    i3 = _load_i3_fields(met_root, timestamp, grid, int(time_index), None)
    return _assemble_transport_forcing(
        a1,
        a3,
        surface_pressure_start=i3.surface_pressure,
        surface_pressure_end=i3.surface_pressure,
        restart_surface_pressure=i3.surface_pressure,
        dry_surface_pressure_start=dry_surface_pressure_hpa(i3.surface_pressure, i3.qv, grid.hyai_hpa, grid.hybi),
        dry_surface_pressure_end=dry_surface_pressure_hpa(i3.surface_pressure, i3.qv, grid.hyai_hpa, grid.hybi),
        restart_dry_surface_pressure=dry_surface_pressure_hpa(i3.surface_pressure, i3.qv, grid.hyai_hpa, grid.hybi),
        specific_humidity=i3.qv,
        restart_specific_humidity=i3.qv,
        temperature=i3.temperature,
        restart_temperature=i3.temperature,
        i3_path=i3.path,
        grid=grid,
        vertical_mapping=MERRA2_72_TO_47_MAPPING,
    )


def load_transport_forcing_for_step(
    met_root: str | Path,
    start: datetime,
    current: datetime,
    grid: TransportGrid,
    *,
    dt_s: float,
    initial_met_time_index: int = 0,
    cache: ForcingRecordCache | None = None,
) -> TransportForcing:
    """Load GEOS-Chem-timed forcing for one transport step.

    A1 records are hourly averages, A3 records are held for 3 hours, and I3
    endpoint fields are interpolated to the dynamic timestep.
    """

    if dt_s <= 0:
        raise ValueError("dt_s must be positive")
    elapsed_s = (current - start).total_seconds()
    if elapsed_s < 0:
        raise ValueError("current must be at or after start")

    met_root = Path(met_root)
    start_day = datetime(start.year, start.month, start.day)
    hour_index = int(initial_met_time_index) * 3 + int(elapsed_s // 3600.0)
    i3_start_index = int(initial_met_time_index) + int(elapsed_s // 10800.0)
    i3_end_index = i3_start_index + 1
    seconds_into_i3_window = elapsed_s % 10800.0

    a1_day, a1_index = _record_day_and_index(start_day, hour_index, 24)
    a3_day, a3_index = _record_day_and_index(start_day, i3_start_index, 8)
    i3_start_day, i3_start_time_index = _record_day_and_index(start_day, i3_start_index, 8)
    i3_end_day, i3_end_time_index = _record_day_and_index(start_day, i3_end_index, 8)
    restart_day, restart_time_index = _record_day_and_index(
        start_day,
        int(initial_met_time_index) + int((elapsed_s + dt_s) // 10800.0),
        8,
    )

    a1 = _load_a1_fields(met_root, a1_day, grid, a1_index, cache)
    a3 = _load_a3_fields(met_root, a3_day, grid, a3_index, cache)
    i3_start = _load_i3_fields(met_root, i3_start_day, grid, i3_start_time_index, cache)
    i3_end = _load_i3_fields(met_root, i3_end_day, grid, i3_end_time_index, cache)
    i3_restart = _load_i3_fields(met_root, restart_day, grid, restart_time_index, cache)

    start_fraction = seconds_into_i3_window / 10800.0
    end_fraction = (seconds_into_i3_window + float(dt_s)) / 10800.0
    midpoint_fraction = (seconds_into_i3_window + float(dt_s) / 2.0) / 10800.0
    dry_surface_start_endpoint = dry_surface_pressure_hpa(i3_start.surface_pressure, i3_start.qv, grid.hyai_hpa, grid.hybi)
    dry_surface_end_endpoint = dry_surface_pressure_hpa(i3_end.surface_pressure, i3_end.qv, grid.hyai_hpa, grid.hybi)
    restart_dry_surface = dry_surface_pressure_hpa(
        i3_restart.surface_pressure,
        i3_restart.qv,
        grid.hyai_hpa,
        grid.hybi,
    )
    return _assemble_transport_forcing(
        a1,
        a3,
        surface_pressure_start=_interpolate(i3_start.surface_pressure, i3_end.surface_pressure, start_fraction),
        surface_pressure_end=_interpolate(i3_start.surface_pressure, i3_end.surface_pressure, end_fraction),
        restart_surface_pressure=i3_restart.surface_pressure,
        dry_surface_pressure_start=_interpolate(dry_surface_start_endpoint, dry_surface_end_endpoint, start_fraction),
        dry_surface_pressure_end=_interpolate(dry_surface_start_endpoint, dry_surface_end_endpoint, end_fraction),
        restart_dry_surface_pressure=restart_dry_surface,
        specific_humidity=_interpolate(i3_start.qv, i3_end.qv, midpoint_fraction),
        restart_specific_humidity=i3_restart.qv,
        temperature=_interpolate(i3_start.temperature, i3_end.temperature, midpoint_fraction),
        restart_temperature=i3_restart.temperature,
        i3_path=i3_start.path,
        grid=grid,
        vertical_mapping=MERRA2_72_TO_47_MAPPING,
    )


def prune_forcing_record_cache(cache: ForcingRecordCache, *, keep: int = 8) -> None:
    """Bound raw forcing record cache size without assuming collection cadence."""

    while len(cache) > keep:
        oldest = next(iter(cache))
        del cache[oldest]


def _assemble_transport_forcing(
    a1: _A1Fields,
    a3: _A3Fields,
    *,
    surface_pressure_start: np.ndarray,
    surface_pressure_end: np.ndarray,
    restart_surface_pressure: np.ndarray,
    dry_surface_pressure_start: np.ndarray,
    dry_surface_pressure_end: np.ndarray,
    restart_dry_surface_pressure: np.ndarray,
    specific_humidity: np.ndarray,
    restart_specific_humidity: np.ndarray,
    temperature: np.ndarray,
    restart_temperature: np.ndarray,
    i3_path: Path,
    grid: TransportGrid,
    vertical_mapping: str,
) -> TransportForcing:

    wet_surface_pressure_start = wet_surface_pressure_hpa(surface_pressure_start)
    wet_surface_pressure_end = wet_surface_pressure_hpa(surface_pressure_end)
    restart_wet_surface_pressure = wet_surface_pressure_hpa(restart_surface_pressure)
    return TransportForcing(
        u_m_s=a3.u,
        v_m_s=a3.v,
        omega_pa_s=a3.omega,
        surface_pressure_start_pa=surface_pressure_start,
        surface_pressure_pa=surface_pressure_end,
        restart_surface_pressure_pa=restart_surface_pressure,
        wet_surface_pressure_start_hpa=wet_surface_pressure_start,
        wet_surface_pressure_hpa=wet_surface_pressure_end,
        restart_wet_surface_pressure_hpa=restart_wet_surface_pressure,
        dry_surface_pressure_start_hpa=dry_surface_pressure_start,
        dry_surface_pressure_hpa=dry_surface_pressure_end,
        restart_dry_surface_pressure_hpa=restart_dry_surface_pressure,
        specific_humidity_kg_kg=specific_humidity,
        restart_specific_humidity_kg_kg=restart_specific_humidity,
        temperature_k=temperature,
        restart_temperature_k=restart_temperature,
        pbl_height_m=a1.pblh,
        sensible_heat_flux_w_m2=a1.hflux,
        latent_heat_flux_w_m2=a1.eflux,
        friction_velocity_m_s=a1.ustar,
        convective_mass_flux_kg_m2_s=a3.cmfmc[np.newaxis, 1:, :, :],
        convective_detrainment_kg_m2_s=a3.dtrain,
        convective_precip_prod_kg_kg_s=a3.dqrcu,
        convective_precip_reevap_kg_kg_s=a3.reevapcn,
        convective_ice_flux_kg_m2_s=a3.pficu[np.newaxis, 1:, :, :],
        convective_liquid_flux_kg_m2_s=a3.pflcu[np.newaxis, 1:, :, :],
        convective_precip_mm_day=a1.precccon * 86400.0,
        lat_deg=grid.lat_deg,
        lon_deg=grid.lon_deg,
        vertical_mapping=vertical_mapping,
        a1_path=a1.path.resolve(),
        a3dyn_path=a3.a3dyn_path.resolve(),
        a3mstc_path=a3.a3mstc_path.resolve(),
        a3mste_path=a3.a3mste_path.resolve(),
        i3_path=i3_path.resolve(),
    )


def _record_day_and_index(start_day: datetime, absolute_index: int, records_per_day: int) -> tuple[datetime, int]:
    if absolute_index < 0:
        raise ValueError(f"met record index must be nonnegative, got {absolute_index}")
    return start_day + timedelta(days=absolute_index // records_per_day), absolute_index % records_per_day


def _met_path(met_root: Path, timestamp: datetime, collection: str) -> Path:
    day_dir = met_root / f"{timestamp.year:04d}" / f"{timestamp.month:02d}"
    return day_dir / MERRA2_FILENAME.format(date=timestamp.strftime("%Y%m%d"), collection=collection)


def _load_a1_fields(
    met_root: Path,
    timestamp: datetime,
    grid: TransportGrid,
    time_index: int,
    cache: ForcingRecordCache | None,
) -> _A1Fields:
    key = ("A1", datetime(timestamp.year, timestamp.month, timestamp.day), int(time_index))
    if cache is not None and key in cache:
        return cache[key]
    a1_path = _met_path(met_root, timestamp, "A1")
    with netCDF4.Dataset(a1_path) as a1:
        fields = _A1Fields(
            pblh=_read_2d_time_slice(a1, "PBLH", time_index),
            hflux=_read_2d_time_slice(a1, "HFLUX", time_index),
            eflux=_read_2d_time_slice(a1, "EFLUX", time_index),
            ustar=_read_2d_time_slice(a1, "USTAR", time_index),
            precccon=np.asarray(a1.variables["PRECCON"][time_index : time_index + 1], dtype=np.float64),
            path=a1_path,
        )
    if cache is not None:
        cache[key] = fields
    return fields


def _load_a3_fields(
    met_root: Path,
    timestamp: datetime,
    grid: TransportGrid,
    time_index: int,
    cache: ForcingRecordCache | None,
) -> _A3Fields:
    key = ("A3", datetime(timestamp.year, timestamp.month, timestamp.day), int(time_index))
    if cache is not None and key in cache:
        return cache[key]
    a3dyn_path = _met_path(met_root, timestamp, "A3dyn")
    a3mstc_path = _met_path(met_root, timestamp, "A3mstC")
    a3mste_path = _met_path(met_root, timestamp, "A3mstE")
    with (
        netCDF4.Dataset(a3dyn_path) as a3dyn,
        netCDF4.Dataset(a3mstc_path) as a3mstc,
        netCDF4.Dataset(a3mste_path) as a3mste,
    ):
        fields = _A3Fields(
            u=_map_met_levels_to_47(_read_3d_time_slice(a3dyn, "U", time_index)),
            v=_map_met_levels_to_47(_read_3d_time_slice(a3dyn, "V", time_index)),
            omega=_map_met_levels_to_47(_read_3d_time_slice(a3dyn, "OMEGA", time_index)),
            dtrain=_map_met_levels_to_47(_read_3d_time_slice(a3dyn, "DTRAIN", time_index)),
            dqrcu=_map_met_levels_to_47(_read_3d_time_slice(a3mstc, "DQRCU", time_index)),
            reevapcn=_map_met_levels_to_47(_read_3d_time_slice(a3mstc, "REEVAPCN", time_index)),
            cmfmc=_map_met_edges_to_48(np.asarray(a3mste.variables["CMFMC"][time_index], dtype=np.float64)),
            pficu=_map_met_edges_to_48(np.asarray(a3mste.variables["PFICU"][time_index], dtype=np.float64)),
            pflcu=_map_met_edges_to_48(np.asarray(a3mste.variables["PFLCU"][time_index], dtype=np.float64)),
            a3dyn_path=a3dyn_path,
            a3mstc_path=a3mstc_path,
            a3mste_path=a3mste_path,
        )
    if cache is not None:
        cache[key] = fields
    return fields


def _load_i3_fields(
    met_root: Path,
    timestamp: datetime,
    grid: TransportGrid,
    time_index: int,
    cache: ForcingRecordCache | None,
) -> _I3Fields:
    key = ("I3", datetime(timestamp.year, timestamp.month, timestamp.day), int(time_index))
    if cache is not None and key in cache:
        return cache[key]
    i3_path = _met_path(met_root, timestamp, "I3")
    with netCDF4.Dataset(i3_path) as i3:
        fields = _I3Fields(
            surface_pressure=np.asarray(i3.variables["PS"][time_index : time_index + 1], dtype=np.float64),
            qv=_map_met_levels_to_47(_read_3d_time_slice(i3, "QV", time_index)),
            temperature=_map_met_levels_to_47(_read_3d_time_slice(i3, "T", time_index)),
            path=i3_path,
        )
    if cache is not None:
        cache[key] = fields
    return fields


def _interpolate(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    start_array = np.asarray(start, dtype=np.float64)
    end_array = np.asarray(end, dtype=np.float64)
    return start_array + (end_array - start_array) * float(fraction)


def _read_3d_time_slice(dataset: netCDF4.Dataset, variable_name: str, time_index: int) -> np.ndarray:
    return np.asarray(dataset.variables[variable_name][time_index : time_index + 1], dtype=np.float64)

def _read_2d_time_slice(dataset: netCDF4.Dataset, variable_name: str, time_index: int) -> np.ndarray:
    return np.asarray(dataset.variables[variable_name][time_index : time_index + 1], dtype=np.float64)

def _map_met_levels_to_47(data: np.ndarray) -> np.ndarray:
    if data.shape[1] == FIXED_GRID["lev"]:
        return data
    if data.shape[1] == 72:
        mapped = np.empty((data.shape[0], FIXED_GRID["lev"], data.shape[2], data.shape[3]), dtype=np.float64)
        mapped[:, :36, :, :] = data[:, :36, :, :]
        for target_level, (start, end) in enumerate(MERRA2_72_TO_47_GROUPS, start=36):
            weights = MERRA2_72_AP_HPA[start:end] - MERRA2_72_AP_HPA[start + 1 : end + 1]
            mapped[:, target_level, :, :] = np.sum(
                data[:, start:end, :, :] * weights[np.newaxis, :, np.newaxis, np.newaxis],
                axis=1,
            ) / np.sum(weights)
        return mapped
    raise ValueError(f"cannot map {data.shape[1]} met levels to {FIXED_GRID['lev']}")


def _map_met_edges_to_48(data: np.ndarray) -> np.ndarray:
    edges = np.asarray(data, dtype=np.float64)
    if edges.ndim != 3:
        raise ValueError(f"edge field must be 3-D (edge, lat, lon), found {edges.shape}")
    if edges.shape[0] == FIXED_GRID["lev"] + 1:
        return edges
    if edges.shape[0] != 73:
        raise ValueError(f"cannot map {edges.shape[0]} met edges to {FIXED_GRID['lev'] + 1} target edges")
    target_indices = np.array(
        list(range(37)) + [38, 40, 42, 44, 48, 52, 56, 60, 64, 68, 72],
        dtype=np.int64,
    )
    return edges[target_indices]
