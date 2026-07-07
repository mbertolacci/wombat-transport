from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

import netCDF4
import numpy as np

from wombat_transport.io import initialize_tracers
from wombat_transport.run_config import load_run_config
from wombat_transport.transport import (
    load_transport_forcing,
    pjc_mass_flux_hpa,
    run_vdiffdr_one_step,
)
from wombat_transport.transport.tpcore import analyze_tpcore_branches, run_tpcore_one_step, setup_tpcore_terms
from wombat_transport.transport.tpcore import trace_tpcore_one_step


CONFIG_TIME_FORMAT = "%Y-%m-%d %H:%M"
PJC_INPUT_VERSION = "pjc-pfix-input-v1"
PJC_OUTPUT_VERSION = "pjc-pfix-output-v1"
PJC_SNAPSHOT_VERSION = "pjc-pfix-snapshot-v1"
TRANSPORT_INPUT_VERSION = "transport-step-input-v1"
TRANSPORT_OUTPUT_VERSION = "transport-step-output-v1"
VDIFF_INPUT_VERSION = "vdiffdr-input-v1"
VDIFF_OUTPUT_VERSION = "vdiffdr-output-v1"
TPCORE_TRACE_VERSION = "tpcore-trace-v1"
TPCORE_SNAPSHOT_VERSION = "tpcore-step-snapshot-v1"
SNAPSHOT_INPUT_NAME = "pjc_input.nc"
SNAPSHOT_OUTPUT_NAME = "pjc_output.nc"
TPCORE_SNAPSHOT_INPUT_NAME = "tpcore_input.nc"
TPCORE_SNAPSHOT_OUTPUT_NAME = "tpcore_output.nc"
VDIFF_SNAPSHOT_INPUT_NAME = "vdiff_input.nc"
VDIFF_SNAPSHOT_OUTPUT_NAME = "vdiff_output.nc"
SNAPSHOT_METADATA_NAME = "metadata.json"
TPCORE_BRANCH_SCENARIOS = ("x_fxppm_low_courant", "x_large_courant_polar")
LARGE_ORACLE_MANIFEST_NAME = "manifest.json"
PYTHON_TPCORE_TRACE_NAME = "python_tpcore_trace.nc"
ORACLE_TPCORE_TRACE_NAME = "oracle_tpcore_trace.nc"
BASE_INITIAL_TPCORE_FIXTURE_ID = "base_initial_tpcore_v1"
RESIDUAL_INITIAL_TPCORE_FIXTURE_ID = "residual_initial_tpcore_v1"
FULLGRID_SYNTHETIC_LOW_COURANT_TPCORE_FIXTURE_ID = "fullgrid_synthetic_low_courant_tpcore_v1"
LARGE_ORACLE_FIXTURE_IDS = (
    BASE_INITIAL_TPCORE_FIXTURE_ID,
    RESIDUAL_INITIAL_TPCORE_FIXTURE_ID,
    FULLGRID_SYNTHETIC_LOW_COURANT_TPCORE_FIXTURE_ID,
)

GEOS_47_AP_HPA = np.array(
    [
        0.0,
        0.04804826,
        6.593752,
        13.1348,
        19.61311,
        26.09201,
        32.57081,
        38.98201,
        45.33901,
        51.69611,
        58.05321,
        64.36264,
        70.62198,
        78.83422,
        89.09992,
        99.36521,
        109.1817,
        118.9586,
        128.6959,
        142.91,
        156.26,
        169.609,
        181.619,
        193.097,
        203.259,
        212.15,
        218.776,
        223.898,
        224.363,
        216.865,
        201.192,
        176.93,
        150.393,
        127.837,
        108.663,
        92.36572,
        78.51231,
        56.38791,
        40.17541,
        28.36781,
        19.7916,
        9.292942,
        4.076571,
        1.65079,
        0.6167791,
        0.211349,
        0.06600001,
        0.01,
    ],
    dtype=np.float64,
)

GEOS_47_BP = np.array(
    [
        1.0,
        0.984952,
        0.963406,
        0.941865,
        0.920387,
        0.898908,
        0.877429,
        0.856018,
        0.8346609,
        0.8133039,
        0.7919469,
        0.7706375,
        0.7493782,
        0.721166,
        0.6858999,
        0.6506349,
        0.6158184,
        0.5810415,
        0.5463042,
        0.4945902,
        0.4437402,
        0.3928911,
        0.3433811,
        0.2944031,
        0.2467411,
        0.2003501,
        0.1562241,
        0.1136021,
        0.06372006,
        0.02801004,
        0.006960025,
        8.175413e-09,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class PjcComparison:
    xmass_max_abs_error_hpa: float
    xmass_mean_abs_error_hpa: float
    ymass_max_abs_error_hpa: float
    ymass_mean_abs_error_hpa: float


@dataclass(frozen=True)
class TransportStepComparison:
    xmass_max_abs_error_hpa: float
    xmass_mean_abs_error_hpa: float
    ymass_max_abs_error_hpa: float
    ymass_mean_abs_error_hpa: float
    tracer_max_abs_change: float
    tracer_min_after: float
    tracer_max_after: float
    negative_count_after: int
    surface_pressure_min_hpa: float
    surface_pressure_max_hpa: float


@dataclass(frozen=True)
class PythonTpcoreComparison:
    xmass_max_abs_error_hpa: float
    xmass_mean_abs_error_hpa: float
    ymass_max_abs_error_hpa: float
    ymass_mean_abs_error_hpa: float
    surface_pressure_max_abs_error_hpa: float
    surface_pressure_mean_abs_error_hpa: float
    tracer_max_abs_error: float
    tracer_mean_abs_error: float
    negative_count_after: int
    max_abs_cx: float
    max_abs_cy: float


@dataclass(frozen=True)
class VdiffOutput:
    tracer_conc_after: np.ndarray
    specific_humidity_after: np.ndarray
    kvh_m2_s: np.ndarray
    kvm_m2_s: np.ndarray
    pbl_top_m: np.ndarray
    tpert_k: np.ndarray
    qpert_kg_kg: np.ndarray
    negative_count_before_clip: int
    negative_count_after_clip: int
    initial_tracer_mass: np.ndarray
    final_tracer_mass: np.ndarray


@dataclass(frozen=True)
class VdiffComparison:
    tracer_max_abs_error: float
    tracer_mean_abs_error: float
    specific_humidity_max_abs_error: float
    kvh_max_abs_error: float
    kvm_max_abs_error: float
    pbl_top_max_abs_error_m: float
    tpert_max_abs_error: float
    qpert_max_abs_error: float
    negative_count_before_clip_expected: int
    negative_count_before_clip_actual: int
    negative_count_after_clip_expected: int
    negative_count_after_clip_actual: int
    final_mass_max_abs_error: float


@dataclass(frozen=True)
class TransportStepOutput:
    tracer_conc_after: np.ndarray
    xmass_hpa: np.ndarray
    ymass_hpa: np.ndarray
    surface_pressure_hpa: np.ndarray


@dataclass(frozen=True)
class TpcoreInput:
    lat_deg: np.ndarray
    hyai_hpa: np.ndarray
    hybi: np.ndarray
    area_m2: np.ndarray
    p1_hpa: np.ndarray
    p2_hpa: np.ndarray
    u_m_s: np.ndarray
    v_m_s: np.ndarray
    tracer_conc: np.ndarray
    dt_s: float


@dataclass(frozen=True)
class LargeOracleFixturePaths:
    fixture_id: str
    directory: Path
    input_path: Path
    output_path: Path
    manifest_path: Path
    definition_path: Path


@dataclass(frozen=True)
class LargeOracleFixtureCheck:
    fixture_id: str
    manifest_path: Path
    missing_files: tuple[str, ...]
    checksum_failures: tuple[str, ...]
    unchecked_files: tuple[str, ...]

    @property
    def is_available(self) -> bool:
        return not self.missing_files and not self.checksum_failures and not self.unchecked_files


def write_pjc_input_from_config(
    run_config_path: str | Path,
    output_path: str | Path,
    *,
    time_index: int = 0,
    dt_s: float | None = None,
) -> Path:
    config = load_run_config(run_config_path)
    if dt_s is None:
        dt_s = float(config.transport.get("dt_s", 600.0))
    forcing = load_transport_forcing(
        _resolve_config_value(config.root, config.transport["met_root"]),
        datetime.strptime(config.transport["start"], CONFIG_TIME_FORMAT),
        config.grid_template,
        time_index=time_index,
    )
    with netCDF4.Dataset(config.grid_template) as template:
        hyai = np.asarray(template.variables["hyai"][:], dtype=np.float64)
        hybi = np.asarray(template.variables["hybi"][:], dtype=np.float64)
        area = np.asarray(template.variables["AREA"][:], dtype=np.float64)
    p1_hpa = forcing.surface_pressure_pa[0] / 100.0
    p2_hpa = p1_hpa.copy()
    return write_pjc_input(
        output_path,
        lat_deg=forcing.lat_deg,
        lon_deg=forcing.lon_deg,
        area_m2=area,
        hyai_hpa=hyai,
        hybi=hybi,
        p1_hpa=p1_hpa,
        p2_hpa=p2_hpa,
        u_m_s=forcing.u_m_s[0],
        v_m_s=forcing.v_m_s[0],
        dt_s=dt_s,
    )


def write_transport_step_input_from_config(
    run_config_path: str | Path,
    output_path: str | Path,
    *,
    time_index: int = 0,
    tracer_time_index: int = 0,
    dt_s: float | None = None,
    max_tracers: int | None = None,
) -> Path:
    config = load_run_config(run_config_path)
    tracers = initialize_tracers(
        config.initial_restart,
        config.species_database,
        template_path=config.grid_template,
    )
    tracer_data = np.asarray(tracers.data[:, tracer_time_index, :, :, :], dtype=np.float64)
    tracer_names = tracers.names
    if max_tracers is not None:
        tracer_data = tracer_data[:max_tracers]
        tracer_names = tracer_names[:max_tracers]

    forcing_path = write_pjc_input_from_config(
        run_config_path,
        output_path,
        time_index=time_index,
        dt_s=dt_s,
    )
    append_transport_step_tracers(forcing_path, tracer_data, tracer_names=tracer_names)
    return forcing_path


def write_fullgrid_synthetic_tpcore_input_from_config(
    run_config_path: str | Path,
    output_path: str | Path,
    *,
    dt_s: float | None = None,
    ntracer: int = 2,
) -> Path:
    """Write a full-grid smooth low-Courant TPCORE input from a grid template."""

    if ntracer <= 0:
        raise ValueError("ntracer must be positive")
    config = load_run_config(run_config_path)
    if dt_s is None:
        dt_s = float(config.transport.get("dt_s", 600.0))
    with netCDF4.Dataset(config.grid_template) as template:
        lat = np.asarray(template.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(template.variables["lon"][:], dtype=np.float64)
        hyai = np.asarray(template.variables["hyai"][:], dtype=np.float64)
        hybi = np.asarray(template.variables["hybi"][:], dtype=np.float64)
        area = np.asarray(template.variables["AREA"][:], dtype=np.float64)

    nlev = hyai.size - 1
    level = np.arange(nlev, dtype=np.float64)[:, np.newaxis, np.newaxis]
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lat_2d = lat_rad[:, np.newaxis]
    lon_2d = lon_rad[np.newaxis, :]

    p1 = 965.0
    p1 = p1 + 22.0 * np.cos(lat_2d) ** 2
    p1 = p1 + 2.0 * np.sin(lon_2d) * np.cos(lat_2d)
    p2 = p1 + 0.25 * np.cos(2.0 * lon_2d) * np.cos(lat_2d)

    lat_3d = lat_rad[np.newaxis, :, np.newaxis]
    lon_3d = lon_rad[np.newaxis, np.newaxis, :]
    vertical_wave = np.sin((level + 1.0) / float(nlev) * np.pi)
    u = 5.0 * vertical_wave * np.cos(lat_3d)
    u = u * (1.0 + 0.15 * np.cos(lon_3d))
    v = 0.35 * np.cos((level + 1.0) / float(nlev) * np.pi) * np.sin(2.0 * lon_3d)
    v = v * np.cos(lat_3d)

    path = write_pjc_input(
        output_path,
        lat_deg=lat,
        lon_deg=lon,
        area_m2=area,
        hyai_hpa=hyai,
        hybi=hybi,
        p1_hpa=p1,
        p2_hpa=p2,
        u_m_s=u,
        v_m_s=v,
        dt_s=dt_s,
    )

    tracer_index = np.arange(ntracer, dtype=np.float64)[:, np.newaxis, np.newaxis, np.newaxis]
    lev_index = np.arange(nlev, dtype=np.float64)[np.newaxis, :, np.newaxis, np.newaxis]
    lat_wave = np.sin(lat_rad)[np.newaxis, np.newaxis, :, np.newaxis]
    lon_wave = np.cos(lon_rad)[np.newaxis, np.newaxis, np.newaxis, :]
    tracer = 4.0e-4
    tracer = tracer + (tracer_index + 1.0) * 1.0e-7
    tracer = tracer + 2.5e-8 * lev_index / max(float(nlev - 1), 1.0)
    tracer = tracer + 1.5e-8 * lat_wave + 7.5e-9 * lon_wave
    names = tuple(f"fullgrid_synthetic_{index + 1:02d}" for index in range(ntracer))
    return append_transport_step_tracers(path, tracer, tracer_names=names)


def write_synthetic_pjc_snapshot_input(path: str | Path, *, dt_s: float = 600.0) -> Path:
    """Write a compact deterministic 47-level PJC oracle input fixture."""

    lat = np.array([-89.5, -60.0, -30.0, 0.0, 30.0, 60.0, 89.5], dtype=np.float64)
    lon = np.arange(8, dtype=np.float64) * 45.0
    area = _spherical_band_area(lat, lon.size)
    level = np.arange(GEOS_47_AP_HPA.size - 1, dtype=np.float64)[:, np.newaxis, np.newaxis]
    j = np.arange(lat.size, dtype=np.float64)[np.newaxis, :, np.newaxis]
    i = np.arange(lon.size, dtype=np.float64)[np.newaxis, np.newaxis, :]

    p1 = 970.0
    p1 = p1 + 18.0 * np.cos(np.deg2rad(lat))[:, np.newaxis]
    p1 = p1 + 1.5 * np.sin(2.0 * np.pi * np.arange(lon.size, dtype=np.float64)[np.newaxis, :] / lon.size)
    p2 = p1
    p2 = p2 + 0.8 * np.cos(2.0 * np.pi * np.arange(lon.size, dtype=np.float64)[np.newaxis, :] / lon.size)
    p2 = p2 - 0.5 * np.sin(np.deg2rad(lat))[:, np.newaxis]

    u = 8.0 * np.sin((level + 1.0) / 47.0 * np.pi) * np.cos(np.deg2rad(lat))[np.newaxis, :, np.newaxis]
    u = u + 0.2 * (i - 3.5)
    v = 2.0 * np.cos((level + 1.0) / 47.0 * np.pi) * np.sin(np.deg2rad(lat))[np.newaxis, :, np.newaxis]
    v = v + 0.1 * np.sin(2.0 * np.pi * i / lon.size)
    v = v + 0.015 * (j - 3.0)

    return write_pjc_input(
        path,
        lat_deg=lat,
        lon_deg=lon,
        area_m2=area,
        hyai_hpa=GEOS_47_AP_HPA,
        hybi=GEOS_47_BP,
        p1_hpa=p1,
        p2_hpa=p2,
        u_m_s=u,
        v_m_s=v,
        dt_s=dt_s,
    )


def write_synthetic_tpcore_snapshot_input(path: str | Path, *, dt_s: float = 600.0, ntracer: int = 2) -> Path:
    """Write a compact deterministic 47-level one-step TPCORE oracle input."""

    if ntracer <= 0:
        raise ValueError("ntracer must be positive")
    path = write_synthetic_pjc_snapshot_input(path, dt_s=dt_s)
    with netCDF4.Dataset(path) as dataset:
        nlev = len(dataset.dimensions["lev"])
        nlat = len(dataset.dimensions["lat"])
        nlon = len(dataset.dimensions["lon"])
        lat = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(dataset.variables["lon"][:], dtype=np.float64)

    tracer_index = np.arange(ntracer, dtype=np.float64)[:, np.newaxis, np.newaxis, np.newaxis]
    lev_index = np.arange(nlev, dtype=np.float64)[np.newaxis, :, np.newaxis, np.newaxis]
    lat_wave = np.sin(np.deg2rad(lat))[np.newaxis, np.newaxis, :, np.newaxis]
    lon_wave = np.cos(np.deg2rad(lon))[np.newaxis, np.newaxis, np.newaxis, :]
    tracer = 4.0e-4
    tracer = tracer + (tracer_index + 1.0) * 1.0e-7
    tracer = tracer + 2.5e-8 * lev_index / max(float(nlev - 1), 1.0)
    tracer = tracer + 1.0e-8 * lat_wave + 5.0e-9 * lon_wave
    names = tuple(f"synthetic_{index + 1:02d}" for index in range(ntracer))
    return append_transport_step_tracers(path, tracer, tracer_names=names)


def write_synthetic_vdiff_input(path: str | Path, *, dt_s: float = 600.0, ntracer: int = 2) -> Path:
    """Write a compact deterministic 47-level VDIFFDR oracle input."""

    if ntracer <= 0:
        raise ValueError("ntracer must be positive")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lat = np.array([-45.0, 0.0, 45.0], dtype=np.float64)
    lon = np.arange(4, dtype=np.float64) * 90.0
    nlev = GEOS_47_AP_HPA.size - 1
    lev = np.arange(nlev, dtype=np.float64)[:, np.newaxis, np.newaxis]
    lat_term = np.sin(np.deg2rad(lat))[np.newaxis, :, np.newaxis]
    lon_term = np.cos(np.deg2rad(lon))[np.newaxis, np.newaxis, :]
    tracer_index = np.arange(ntracer, dtype=np.float64)[:, np.newaxis, np.newaxis, np.newaxis]

    pedge_profile = np.linspace(1000.0, 50.0, nlev + 1, dtype=np.float64)
    pedge = np.broadcast_to(pedge_profile[:, np.newaxis, np.newaxis], (nlev + 1, lat.size, lon.size)).copy()
    pmid = 0.5 * (pedge[:-1] + pedge[1:])
    temperature = 289.0 - 0.45 * lev + 1.5 * lat_term + 0.2 * lon_term
    sphu = 0.010 * np.exp(-lev / 18.0) * (1.0 + 0.03 * lat_term) * np.ones((1, 1, lon.size))
    tv = temperature * (1.0 + 0.61 * sphu)
    bxheight = np.full((nlev, lat.size, lon.size), 125.0, dtype=np.float64)
    dry_mass = (pedge[:-1] - pedge[1:]) * 100.0 / 9.80665
    u = (4.0 + 0.05 * lev + 0.2 * lon_term) * np.ones((1, lat.size, 1), dtype=np.float64)
    v = (0.3 * np.sin((lev + 1.0) / nlev * np.pi) + 0.02 * lat_term) * np.ones((1, 1, lon.size))
    tracer = 4.0e-4 + 1.0e-7 * tracer_index + 4.0e-9 * lev + 2.0e-9 * lat_term + 1.0e-9 * lon_term

    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("tracer", ntracer)
        dataset.createDimension("lev", nlev)
        dataset.createDimension("ilev", nlev + 1)
        dataset.createDimension("lat", lat.size)
        dataset.createDimension("lon", lon.size)
        dataset.harness = VDIFF_INPUT_VERSION
        dataset.dt_s = float(dt_s)
        dataset.createVariable("lon", "f8", ("lon",))[:] = lon
        dataset.createVariable("lat", "f8", ("lat",))[:] = lat
        dataset.createVariable("tracer_conc", "f8", ("tracer", "lev", "lat", "lon"))[:] = tracer
        dataset.createVariable("u_m_s", "f8", ("lev", "lat", "lon"))[:] = u
        dataset.createVariable("v_m_s", "f8", ("lev", "lat", "lon"))[:] = v
        dataset.createVariable("temperature_k", "f8", ("lev", "lat", "lon"))[:] = temperature
        dataset.createVariable("specific_humidity_kg_kg", "f8", ("lev", "lat", "lon"))[:] = sphu
        dataset.createVariable("pmid_hpa", "f8", ("lev", "lat", "lon"))[:] = pmid
        dataset.createVariable("pedge_hpa", "f8", ("ilev", "lat", "lon"))[:] = pedge
        dataset.createVariable("virtual_temperature_k", "f8", ("lev", "lat", "lon"))[:] = tv
        dataset.createVariable("bxheight_m", "f8", ("lev", "lat", "lon"))[:] = bxheight
        dataset.createVariable("dry_air_mass_kg", "f8", ("lev", "lat", "lon"))[:] = dry_mass
        dataset.createVariable("pbl_top_m", "f8", ("lat", "lon"))[:] = np.full((lat.size, lon.size), 950.0)
        dataset.createVariable("hflux_w_m2", "f8", ("lat", "lon"))[:] = np.full((lat.size, lon.size), 65.0)
        dataset.createVariable("eflux_w_m2", "f8", ("lat", "lon"))[:] = np.full((lat.size, lon.size), 90.0)
        dataset.createVariable("ustar_m_s", "f8", ("lat", "lon"))[:] = np.full((lat.size, lon.size), 0.35)
        dataset.createVariable("area_m2", "f8", ("lat", "lon"))[:] = np.ones((lat.size, lon.size))
        dataset.createVariable("surface_flux_kg_m2_s", "f8", ("tracer", "lat", "lon"))[:] = 0.0
    return path


def write_synthetic_tpcore_branch_input(
    path: str | Path,
    *,
    scenario: str,
    dt_s: float = 600.0,
    ntracer: int = 2,
) -> Path:
    """Write a compact TPCORE input that isolates one horizontal branch set."""

    if scenario == "x_fxppm_low_courant":
        return _write_synthetic_tpcore_branch_input(
            path,
            lat=np.linspace(-89.5, 89.5, 11, dtype=np.float64),
            lon=np.arange(12, dtype=np.float64) * 30.0,
            wind_scale=1.0,
            dt_s=dt_s,
            ntracer=ntracer,
            tracer_prefix=scenario,
        )
    if scenario == "x_large_courant_polar":
        return _write_synthetic_tpcore_branch_input(
            path,
            lat=np.array([-89.5, -60.0, -30.0, 0.0, 30.0, 60.0, 89.5], dtype=np.float64),
            lon=np.arange(8, dtype=np.float64) * 45.0,
            wind_scale=1600.0,
            dt_s=dt_s,
            ntracer=ntracer,
            tracer_prefix=scenario,
        )
    raise ValueError(f"unknown TPCORE branch scenario {scenario!r}; expected one of {TPCORE_BRANCH_SCENARIOS}")


def _write_synthetic_tpcore_branch_input(
    path: str | Path,
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    wind_scale: float,
    dt_s: float,
    ntracer: int,
    tracer_prefix: str,
) -> Path:
    if ntracer <= 0:
        raise ValueError("ntracer must be positive")
    area = _spherical_band_area(lat, lon.size)
    level = np.arange(GEOS_47_AP_HPA.size - 1, dtype=np.float64)[:, np.newaxis, np.newaxis]
    j = np.arange(lat.size, dtype=np.float64)[np.newaxis, :, np.newaxis]
    i = np.arange(lon.size, dtype=np.float64)[np.newaxis, np.newaxis, :]

    p1 = 970.0
    p1 = p1 + 18.0 * np.cos(np.deg2rad(lat))[:, np.newaxis]
    p1 = p1 + 1.5 * np.sin(2.0 * np.pi * np.arange(lon.size, dtype=np.float64)[np.newaxis, :] / lon.size)
    p2 = p1
    p2 = p2 + 0.8 * np.cos(2.0 * np.pi * np.arange(lon.size, dtype=np.float64)[np.newaxis, :] / lon.size)
    p2 = p2 - 0.5 * np.sin(np.deg2rad(lat))[:, np.newaxis]

    u = wind_scale * 8.0 * np.sin((level + 1.0) / 47.0 * np.pi) * np.cos(np.deg2rad(lat))[np.newaxis, :, np.newaxis]
    u = u + wind_scale * 0.2 * (i - 0.5 * (lon.size - 1))
    v = 2.0 * np.cos((level + 1.0) / 47.0 * np.pi) * np.sin(np.deg2rad(lat))[np.newaxis, :, np.newaxis]
    v = v + 0.1 * np.sin(2.0 * np.pi * i / lon.size)
    v = v + 0.015 * (j - 0.5 * (lat.size - 1))

    path = write_pjc_input(
        path,
        lat_deg=lat,
        lon_deg=lon,
        area_m2=area,
        hyai_hpa=GEOS_47_AP_HPA,
        hybi=GEOS_47_BP,
        p1_hpa=p1,
        p2_hpa=p2,
        u_m_s=u,
        v_m_s=v,
        dt_s=dt_s,
    )
    tracer_index = np.arange(ntracer, dtype=np.float64)[:, np.newaxis, np.newaxis, np.newaxis]
    lev_index = np.arange(GEOS_47_AP_HPA.size - 1, dtype=np.float64)[np.newaxis, :, np.newaxis, np.newaxis]
    lat_wave = np.sin(np.deg2rad(lat))[np.newaxis, np.newaxis, :, np.newaxis]
    lon_wave = np.cos(np.deg2rad(lon))[np.newaxis, np.newaxis, np.newaxis, :]
    tracer = 4.0e-4
    tracer = tracer + (tracer_index + 1.0) * 1.0e-7
    tracer = tracer + 2.5e-8 * lev_index / 46.0
    tracer = tracer + 1.0e-8 * lat_wave + 5.0e-9 * lon_wave
    names = tuple(f"{tracer_prefix}_{index + 1:02d}" for index in range(ntracer))
    return append_transport_step_tracers(path, tracer, tracer_names=names)


def snapshot_pjc_oracle(
    output_dir: str | Path,
    *,
    executable: str | Path,
    dt_s: float = 600.0,
    repo_root: str | Path = ".",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = write_synthetic_pjc_snapshot_input(output_dir / SNAPSHOT_INPUT_NAME, dt_s=dt_s)
    output_path = output_dir / SNAPSHOT_OUTPUT_NAME
    run_pjc_harness(executable, input_path, output_path)
    metadata = _pjc_snapshot_metadata(input_path, output_path, executable=Path(executable), repo_root=Path(repo_root))
    with (output_dir / SNAPSHOT_METADATA_NAME).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_dir


def snapshot_tpcore_oracle(
    output_dir: str | Path,
    *,
    executable: str | Path,
    dt_s: float = 600.0,
    ntracer: int = 2,
    repo_root: str | Path = ".",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = write_synthetic_tpcore_snapshot_input(
        output_dir / TPCORE_SNAPSHOT_INPUT_NAME,
        dt_s=dt_s,
        ntracer=ntracer,
    )
    output_path = output_dir / TPCORE_SNAPSHOT_OUTPUT_NAME
    run_pjc_harness(executable, input_path, output_path)
    metadata = _tpcore_snapshot_metadata(input_path, output_path, executable=Path(executable), repo_root=Path(repo_root))
    with (output_dir / SNAPSHOT_METADATA_NAME).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_dir


def snapshot_tpcore_branch_oracle(
    output_dir: str | Path,
    *,
    scenario: str,
    executable: str | Path,
    dt_s: float = 600.0,
    ntracer: int = 2,
    repo_root: str | Path = ".",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = write_synthetic_tpcore_branch_input(
        output_dir / TPCORE_SNAPSHOT_INPUT_NAME,
        scenario=scenario,
        dt_s=dt_s,
        ntracer=ntracer,
    )
    output_path = output_dir / TPCORE_SNAPSHOT_OUTPUT_NAME
    run_pjc_harness(executable, input_path, output_path)
    metadata = _tpcore_snapshot_metadata(input_path, output_path, executable=Path(executable), repo_root=Path(repo_root))
    metadata["scenario"] = scenario
    with netCDF4.Dataset(input_path) as dataset:
        setup = setup_tpcore_terms(
            p1_hpa=np.asarray(dataset.variables["p1_hpa"][:], dtype=np.float64),
            p2_hpa=np.asarray(dataset.variables["p2_hpa"][:], dtype=np.float64),
            u_m_s=np.asarray(dataset.variables["u_m_s"][:], dtype=np.float64),
            v_m_s=np.asarray(dataset.variables["v_m_s"][:], dtype=np.float64),
            area_m2=np.asarray(dataset.variables["area_m2"][:], dtype=np.float64),
            hyai_hpa=np.asarray(dataset.variables["hyai"][:], dtype=np.float64),
            hybi=np.asarray(dataset.variables["hybi"][:], dtype=np.float64),
            lat_deg=np.asarray(dataset.variables["lat"][:], dtype=np.float64),
            dt_s=float(dataset.dt_s),
        )
    report = analyze_tpcore_branches(setup)
    metadata["branch_report"] = {
        "shape": list(report.shape),
        "max_abs_cx": report.max_abs_cx,
        "max_abs_cy": report.max_abs_cy,
        "has_large_cx": report.has_large_cx,
        "has_large_cy": report.has_large_cy,
        "needs_fxppm": report.needs_fxppm,
        "x_ffsl_active": report.x_ffsl_active,
        "x_ffsl_endpoint_active": report.x_ffsl_endpoint_active,
        "x_near_pole_vanleer_active": report.x_near_pole_vanleer_active,
        "is_supported": report.is_supported,
        "unsupported_reasons": list(report.unsupported_reasons),
    }
    with (output_dir / SNAPSHOT_METADATA_NAME).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_dir


def large_oracle_fixture_paths(
    fixture_id: str,
    *,
    cache_dir: str | Path = "oracle_data",
    manifest_dir: str | Path | None = None,
) -> LargeOracleFixturePaths:
    if fixture_id not in LARGE_ORACLE_FIXTURE_IDS:
        raise ValueError(f"unknown large oracle fixture {fixture_id!r}; expected one of {LARGE_ORACLE_FIXTURE_IDS}")
    cache = Path(cache_dir)
    definitions = Path(manifest_dir) if manifest_dir is not None else Path("oracle_data") / "manifests"
    directory = cache / fixture_id
    return LargeOracleFixturePaths(
        fixture_id=fixture_id,
        directory=directory,
        input_path=directory / "transport_step_input.nc",
        output_path=directory / "transport_step_output.nc",
        manifest_path=directory / LARGE_ORACLE_MANIFEST_NAME,
        definition_path=definitions / f"{fixture_id}.json",
    )


def generate_large_oracle_fixture(
    fixture_id: str,
    *,
    cache_dir: str | Path = "oracle_data",
    manifest_dir: str | Path | None = None,
    run_config: str | Path | None = None,
    executable: str | Path = "tools/gc_harness/build/pjc_pfix_harness",
    time_index: int | None = None,
    tracer_time_index: int | None = None,
    max_tracers: int | None = None,
    dt_s: float | None = None,
    repo_root: str | Path = ".",
) -> Path:
    if fixture_id not in LARGE_ORACLE_FIXTURE_IDS:
        raise ValueError(f"no generator is registered for {fixture_id!r}")
    paths = large_oracle_fixture_paths(fixture_id, cache_dir=cache_dir, manifest_dir=manifest_dir)
    definition = _load_large_oracle_definition(paths.definition_path)
    source = dict(definition.get("source", {}))
    run_config_path = Path(run_config or source.get("run_config", "base_wombat/run.yml"))
    paths.directory.mkdir(parents=True, exist_ok=True)
    fixture_dt_s = float(source["dt_s"]) if dt_s is None and "dt_s" in source else dt_s
    if fixture_id in {BASE_INITIAL_TPCORE_FIXTURE_ID, RESIDUAL_INITIAL_TPCORE_FIXTURE_ID}:
        write_transport_step_input_from_config(
            run_config_path,
            paths.input_path,
            time_index=int(source.get("time_index", 0) if time_index is None else time_index),
            tracer_time_index=int(source.get("tracer_time_index", 0) if tracer_time_index is None else tracer_time_index),
            max_tracers=int(source.get("max_tracers", 1) if max_tracers is None else max_tracers),
            dt_s=fixture_dt_s,
        )
    elif fixture_id == FULLGRID_SYNTHETIC_LOW_COURANT_TPCORE_FIXTURE_ID:
        write_fullgrid_synthetic_tpcore_input_from_config(
            run_config_path,
            paths.input_path,
            dt_s=fixture_dt_s,
            ntracer=int(source.get("ntracer", 2) if max_tracers is None else max_tracers),
        )
    else:
        raise AssertionError(f"unhandled fixture generator {fixture_id}")
    run_pjc_harness(executable, paths.input_path, paths.output_path)
    _write_generated_large_oracle_manifest(
        paths,
        definition=definition,
        executable=Path(executable),
        run_config=run_config_path,
        repo_root=Path(repo_root),
    )
    return paths.directory


def fetch_large_oracle_fixture(
    fixture_id: str,
    *,
    cache_dir: str | Path = "oracle_data",
    manifest_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    paths = large_oracle_fixture_paths(fixture_id, cache_dir=cache_dir, manifest_dir=manifest_dir)
    definition = _load_large_oracle_definition(paths.definition_path)
    paths.directory.mkdir(parents=True, exist_ok=True)
    for entry in _large_oracle_file_entries(definition):
        url = entry.get("url")
        if not url:
            raise ValueError(
                f"{paths.definition_path} does not define a download URL for {entry['name']}; "
                "generate the fixture locally or update the manifest with hosted artifact URLs"
            )
        if not entry.get("sha256"):
            raise ValueError(f"{paths.definition_path} does not define a SHA256 checksum for {entry['name']}")
        target = paths.directory / str(entry["name"])
        if target.exists() and not overwrite:
            continue
        parsed = urlparse(str(url))
        if parsed.scheme not in {"http", "https", "file"}:
            raise ValueError(f"unsupported URL scheme for {entry['name']}: {url}")
        urlretrieve(str(url), target)
    check = check_large_oracle_fixture(fixture_id, cache_dir=cache_dir, manifest_dir=manifest_dir)
    if not check.is_available:
        raise ValueError(format_large_oracle_fixture_check(check))
    return paths.directory


def check_large_oracle_fixture(
    fixture_id: str,
    *,
    cache_dir: str | Path = "oracle_data",
    manifest_dir: str | Path | None = None,
) -> LargeOracleFixtureCheck:
    paths = large_oracle_fixture_paths(fixture_id, cache_dir=cache_dir, manifest_dir=manifest_dir)
    manifest_path = paths.manifest_path if paths.manifest_path.exists() else paths.definition_path
    manifest = _load_large_oracle_definition(manifest_path)
    missing: list[str] = []
    failures: list[str] = []
    unchecked: list[str] = []
    for entry in _large_oracle_file_entries(manifest):
        filename = str(entry["name"])
        path = paths.directory / filename
        if not path.exists():
            missing.append(filename)
            continue
        expected_size = entry.get("size_bytes")
        if expected_size is not None and path.stat().st_size != int(expected_size):
            failures.append(f"{filename}: size {path.stat().st_size} != {expected_size}")
            continue
        expected_hash = entry.get("sha256")
        if expected_hash:
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                failures.append(f"{filename}: sha256 {actual_hash} != {expected_hash}")
        else:
            unchecked.append(filename)
    return LargeOracleFixtureCheck(
        fixture_id=fixture_id,
        manifest_path=manifest_path,
        missing_files=tuple(missing),
        checksum_failures=tuple(failures),
        unchecked_files=tuple(unchecked),
    )


def compare_large_oracle_fixture(
    fixture_id: str,
    *,
    cache_dir: str | Path = "oracle_data",
    manifest_dir: str | Path | None = None,
) -> str:
    check = check_large_oracle_fixture(fixture_id, cache_dir=cache_dir, manifest_dir=manifest_dir)
    if not check.is_available:
        raise FileNotFoundError(format_large_oracle_fixture_check(check))
    paths = large_oracle_fixture_paths(fixture_id, cache_dir=cache_dir, manifest_dir=manifest_dir)
    transport = compare_transport_step_output(paths.input_path, paths.output_path)
    setup = _setup_tpcore_from_input(paths.input_path)
    branch_report = analyze_tpcore_branches(setup)
    tracer_names = _read_transport_step_tracer_names(paths.input_path)
    rows = [
        "metric,value",
        f"tracer_count,{len(tracer_names)}",
        f"tracer_names,{' | '.join(tracer_names)}",
        f"xmass_max_abs_error_hpa,{transport.xmass_max_abs_error_hpa:.8e}",
        f"xmass_mean_abs_error_hpa,{transport.xmass_mean_abs_error_hpa:.8e}",
        f"ymass_max_abs_error_hpa,{transport.ymass_max_abs_error_hpa:.8e}",
        f"ymass_mean_abs_error_hpa,{transport.ymass_mean_abs_error_hpa:.8e}",
        f"tracer_max_abs_change,{transport.tracer_max_abs_change:.8e}",
        f"tracer_min_after,{transport.tracer_min_after:.8e}",
        f"tracer_max_after,{transport.tracer_max_after:.8e}",
        f"negative_count_after,{transport.negative_count_after}",
        f"surface_pressure_min_hpa,{transport.surface_pressure_min_hpa:.8e}",
        f"surface_pressure_max_hpa,{transport.surface_pressure_max_hpa:.8e}",
        f"tpcore_shape,{branch_report.shape}",
        f"max_abs_cx,{branch_report.max_abs_cx:.8e}",
        f"max_abs_cy,{branch_report.max_abs_cy:.8e}",
        f"tpcore_supported,{branch_report.is_supported}",
        f"tpcore_unsupported_reasons,{' | '.join(branch_report.unsupported_reasons)}",
    ]
    if branch_report.is_supported:
        tpcore = compare_python_tpcore_output(paths.input_path, paths.output_path)
        rows.extend(
            [
                f"surface_pressure_max_abs_error_hpa,{tpcore.surface_pressure_max_abs_error_hpa:.8e}",
                f"surface_pressure_mean_abs_error_hpa,{tpcore.surface_pressure_mean_abs_error_hpa:.8e}",
                f"tracer_max_abs_error,{tpcore.tracer_max_abs_error:.8e}",
                f"tracer_mean_abs_error,{tpcore.tracer_mean_abs_error:.8e}",
                f"python_negative_count_after,{tpcore.negative_count_after}",
            ]
        )
    return "\n".join(rows)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_pjc_input(
    path: str | Path,
    *,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    area_m2: np.ndarray,
    hyai_hpa: np.ndarray,
    hybi: np.ndarray,
    p1_hpa: np.ndarray,
    p2_hpa: np.ndarray,
    u_m_s: np.ndarray,
    v_m_s: np.ndarray,
    dt_s: float,
) -> Path:
    """Write a NetCDF fixture for the GEOS-Chem PJC pressure-fixer harness."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = np.asarray(lon_deg, dtype=np.float64)
    hyai = np.asarray(hyai_hpa, dtype=np.float64)
    hybi_arr = np.asarray(hybi, dtype=np.float64)
    area = np.asarray(area_m2, dtype=np.float64)
    p1 = np.asarray(p1_hpa, dtype=np.float64)
    p2 = np.asarray(p2_hpa, dtype=np.float64)
    u = np.asarray(u_m_s, dtype=np.float64)
    v = np.asarray(v_m_s, dtype=np.float64)
    _assert_pjc_shapes(lat, lon, area, hyai, hybi_arr, p1, p2, u, v)

    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("lon", lon.size)
        dataset.createDimension("lat", lat.size)
        dataset.createDimension("lev", hyai.size - 1)
        dataset.createDimension("ilev", hyai.size)
        dataset.harness = PJC_INPUT_VERSION
        dataset.dt_s = float(dt_s)

        dataset.createVariable("lon", "f8", ("lon",))[:] = lon
        dataset.createVariable("lat", "f8", ("lat",))[:] = lat
        dataset.createVariable("hyai", "f8", ("ilev",))[:] = hyai
        dataset.createVariable("hybi", "f8", ("ilev",))[:] = hybi_arr
        dataset.createVariable("area_m2", "f8", ("lat", "lon"))[:] = area
        dataset.createVariable("p1_hpa", "f8", ("lat", "lon"))[:] = p1
        dataset.createVariable("p2_hpa", "f8", ("lat", "lon"))[:] = p2
        dataset.createVariable("u_m_s", "f8", ("lev", "lat", "lon"))[:] = u
        dataset.createVariable("v_m_s", "f8", ("lev", "lat", "lon"))[:] = v
    return path


def append_transport_step_tracers(
    path: str | Path,
    tracer_conc: np.ndarray,
    *,
    tracer_names: tuple[str, ...] | list[str] | None = None,
) -> Path:
    path = Path(path)
    tracers = np.asarray(tracer_conc, dtype=np.float64)
    if tracers.ndim != 4:
        raise ValueError(f"tracer_conc must have shape (tracer, lev, lat, lon), found {tracers.shape}")
    if tracer_names is None:
        names = tuple(f"tracer_{index + 1:03d}" for index in range(tracers.shape[0]))
    else:
        names = tuple(tracer_names)
    if len(names) != tracers.shape[0]:
        raise ValueError("tracer_names length must match tracer_conc first dimension")

    with netCDF4.Dataset(path, "a") as dataset:
        if getattr(dataset, "harness", "") != PJC_INPUT_VERSION:
            raise ValueError(f"{path} is not a {PJC_INPUT_VERSION} file")
        expected = (
            len(dataset.dimensions["lev"]),
            len(dataset.dimensions["lat"]),
            len(dataset.dimensions["lon"]),
        )
        if tracers.shape[1:] != expected:
            raise ValueError(f"tracer_conc grid must have shape {expected}, found {tracers.shape[1:]}")
        dataset.harness = TRANSPORT_INPUT_VERSION
        dataset.createDimension("tracer", tracers.shape[0])
        name_length = max(max((len(name) for name in names), default=1), 1)
        dataset.createDimension("name_strlen", name_length)
        dataset.createVariable("tracer_conc", "f8", ("tracer", "lev", "lat", "lon"))[:] = tracers
        name_var = dataset.createVariable("tracer_name", "S1", ("tracer", "name_strlen"))
        encoded = np.asarray([name.encode("ascii", errors="replace") for name in names], dtype=f"S{name_length}")
        name_var[:] = netCDF4.stringtochar(encoded)
    return path


def read_pjc_output(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with netCDF4.Dataset(path) as dataset:
        if getattr(dataset, "harness", "") != PJC_OUTPUT_VERSION:
            raise ValueError(f"{path} is not a {PJC_OUTPUT_VERSION} file")
        xmass = np.asarray(dataset.variables["xmass_hpa"][:], dtype=np.float64)
        ymass = np.asarray(dataset.variables["ymass_hpa"][:], dtype=np.float64)
    return xmass, ymass


def read_transport_step_output(path: str | Path) -> TransportStepOutput:
    with netCDF4.Dataset(path) as dataset:
        if getattr(dataset, "harness", "") != TRANSPORT_OUTPUT_VERSION:
            raise ValueError(f"{path} is not a {TRANSPORT_OUTPUT_VERSION} file")
        return TransportStepOutput(
            tracer_conc_after=np.asarray(dataset.variables["tracer_conc_after"][:], dtype=np.float64),
            xmass_hpa=np.asarray(dataset.variables["xmass_hpa"][:], dtype=np.float64),
            ymass_hpa=np.asarray(dataset.variables["ymass_hpa"][:], dtype=np.float64),
            surface_pressure_hpa=np.asarray(dataset.variables["surface_pressure_hpa"][:], dtype=np.float64),
        )


def read_vdiff_output(path: str | Path) -> VdiffOutput:
    with netCDF4.Dataset(path) as dataset:
        if getattr(dataset, "harness", "") != VDIFF_OUTPUT_VERSION:
            raise ValueError(f"{path} is not a {VDIFF_OUTPUT_VERSION} file")
        return VdiffOutput(
            tracer_conc_after=np.asarray(dataset.variables["tracer_conc_after"][:], dtype=np.float64),
            specific_humidity_after=np.asarray(dataset.variables["specific_humidity_after"][:], dtype=np.float64),
            kvh_m2_s=np.asarray(dataset.variables["kvh_m2_s"][:], dtype=np.float64),
            kvm_m2_s=np.asarray(dataset.variables["kvm_m2_s"][:], dtype=np.float64),
            pbl_top_m=np.asarray(dataset.variables["pbl_top_m"][:], dtype=np.float64),
            tpert_k=np.asarray(dataset.variables["tpert_k"][:], dtype=np.float64),
            qpert_kg_kg=np.asarray(dataset.variables["qpert_kg_kg"][:], dtype=np.float64),
            negative_count_before_clip=int(getattr(dataset, "negative_count_before_clip")),
            negative_count_after_clip=int(getattr(dataset, "negative_count_after_clip")),
            initial_tracer_mass=np.asarray(dataset.variables["initial_tracer_mass"][:], dtype=np.float64),
            final_tracer_mass=np.asarray(dataset.variables["final_tracer_mass"][:], dtype=np.float64),
        )


def write_python_vdiff_output(input_path: str | Path, output_path: str | Path) -> Path:
    with netCDF4.Dataset(input_path) as dataset:
        if getattr(dataset, "harness", "") != VDIFF_INPUT_VERSION:
            raise ValueError(f"{input_path} is not a {VDIFF_INPUT_VERSION} file")
        result = run_vdiffdr_one_step(
            tracer_conc=np.asarray(dataset.variables["tracer_conc"][:], dtype=np.float64),
            u_m_s=np.asarray(dataset.variables["u_m_s"][:], dtype=np.float64),
            v_m_s=np.asarray(dataset.variables["v_m_s"][:], dtype=np.float64),
            temperature_k=np.asarray(dataset.variables["temperature_k"][:], dtype=np.float64),
            specific_humidity_kg_kg=np.asarray(dataset.variables["specific_humidity_kg_kg"][:], dtype=np.float64),
            pmid_hpa=np.asarray(dataset.variables["pmid_hpa"][:], dtype=np.float64),
            pedge_hpa=np.asarray(dataset.variables["pedge_hpa"][:], dtype=np.float64),
            virtual_temperature_k=np.asarray(dataset.variables["virtual_temperature_k"][:], dtype=np.float64),
            bxheight_m=np.asarray(dataset.variables["bxheight_m"][:], dtype=np.float64),
            dry_air_mass_kg=np.asarray(dataset.variables["dry_air_mass_kg"][:], dtype=np.float64),
            pbl_top_m=np.asarray(dataset.variables["pbl_top_m"][:], dtype=np.float64),
            hflux_w_m2=np.asarray(dataset.variables["hflux_w_m2"][:], dtype=np.float64),
            eflux_w_m2=np.asarray(dataset.variables["eflux_w_m2"][:], dtype=np.float64),
            ustar_m_s=np.asarray(dataset.variables["ustar_m_s"][:], dtype=np.float64),
            area_m2=np.asarray(dataset.variables["area_m2"][:], dtype=np.float64),
            dt_s=float(dataset.dt_s),
            surface_flux_kg_m2_s=np.asarray(dataset.variables["surface_flux_kg_m2_s"][:], dtype=np.float64),
        )
    return write_vdiff_output(output_path, result)


def write_vdiff_output(path: str | Path, result) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w") as dataset:
        ntracer, nlev, nlat, nlon = result.tracer_conc.shape
        dataset.createDimension("tracer", ntracer)
        dataset.createDimension("lev", nlev)
        dataset.createDimension("ilev", nlev + 1)
        dataset.createDimension("lat", nlat)
        dataset.createDimension("lon", nlon)
        dataset.harness = VDIFF_OUTPUT_VERSION
        dataset.negative_count_before_clip = int(result.negative_count_before_clip)
        dataset.negative_count_after_clip = int(result.negative_count_after_clip)
        dataset.createVariable("tracer_conc_after", "f8", ("tracer", "lev", "lat", "lon"))[:] = result.tracer_conc
        dataset.createVariable("specific_humidity_after", "f8", ("lev", "lat", "lon"))[:] = (
            result.specific_humidity_kg_kg
        )
        dataset.createVariable("kvh_m2_s", "f8", ("ilev", "lat", "lon"))[:] = result.kvh_m2_s
        dataset.createVariable("kvm_m2_s", "f8", ("ilev", "lat", "lon"))[:] = result.kvm_m2_s
        dataset.createVariable("pbl_top_m", "f8", ("lat", "lon"))[:] = result.pbl_top_m
        dataset.createVariable("tpert_k", "f8", ("lat", "lon"))[:] = result.tpert_k
        dataset.createVariable("qpert_kg_kg", "f8", ("lat", "lon"))[:] = result.qpert_kg_kg
        dataset.createVariable("initial_tracer_mass", "f8", ("tracer",))[:] = result.initial_tracer_mass
        dataset.createVariable("final_tracer_mass", "f8", ("tracer",))[:] = result.final_tracer_mass
    return path


def compare_vdiff_output(
    input_path: str | Path,
    expected_output_path: str | Path,
    *,
    python_output_path: str | Path | None = None,
) -> VdiffComparison:
    output_path = Path(expected_output_path)
    python_path = Path(python_output_path) if python_output_path is not None else output_path.with_name(f"python_{output_path.name}")
    write_python_vdiff_output(input_path, python_path)
    expected = read_vdiff_output(output_path)
    actual = read_vdiff_output(python_path)
    tracer_error = np.abs(actual.tracer_conc_after - expected.tracer_conc_after)
    sphu_error = np.abs(actual.specific_humidity_after - expected.specific_humidity_after)
    kvh_error = np.abs(actual.kvh_m2_s - expected.kvh_m2_s)
    kvm_error = np.abs(actual.kvm_m2_s - expected.kvm_m2_s)
    return VdiffComparison(
        tracer_max_abs_error=float(np.max(tracer_error)),
        tracer_mean_abs_error=float(np.mean(tracer_error)),
        specific_humidity_max_abs_error=float(np.max(sphu_error)),
        kvh_max_abs_error=float(np.max(kvh_error)),
        kvm_max_abs_error=float(np.max(kvm_error)),
        pbl_top_max_abs_error_m=float(np.max(np.abs(actual.pbl_top_m - expected.pbl_top_m))),
        tpert_max_abs_error=float(np.max(np.abs(actual.tpert_k - expected.tpert_k))),
        qpert_max_abs_error=float(np.max(np.abs(actual.qpert_kg_kg - expected.qpert_kg_kg))),
        negative_count_before_clip_expected=expected.negative_count_before_clip,
        negative_count_before_clip_actual=actual.negative_count_before_clip,
        negative_count_after_clip_expected=expected.negative_count_after_clip,
        negative_count_after_clip_actual=actual.negative_count_after_clip,
        final_mass_max_abs_error=float(np.max(np.abs(actual.final_tracer_mass - expected.final_tracer_mass))),
    )


def read_tpcore_input(path: str | Path) -> TpcoreInput:
    with netCDF4.Dataset(path) as dataset:
        if getattr(dataset, "harness", "") != TRANSPORT_INPUT_VERSION:
            raise ValueError(f"{path} is not a {TRANSPORT_INPUT_VERSION} file")
        return TpcoreInput(
            lat_deg=np.asarray(dataset.variables["lat"][:], dtype=np.float64),
            hyai_hpa=np.asarray(dataset.variables["hyai"][:], dtype=np.float64),
            hybi=np.asarray(dataset.variables["hybi"][:], dtype=np.float64),
            area_m2=np.asarray(dataset.variables["area_m2"][:], dtype=np.float64),
            p1_hpa=np.asarray(dataset.variables["p1_hpa"][:], dtype=np.float64),
            p2_hpa=np.asarray(dataset.variables["p2_hpa"][:], dtype=np.float64),
            u_m_s=np.asarray(dataset.variables["u_m_s"][:], dtype=np.float64),
            v_m_s=np.asarray(dataset.variables["v_m_s"][:], dtype=np.float64),
            tracer_conc=np.asarray(dataset.variables["tracer_conc"][:], dtype=np.float64),
            dt_s=float(getattr(dataset, "dt_s")),
        )


def _read_transport_step_tracer_names(path: str | Path) -> tuple[str, ...]:
    with netCDF4.Dataset(path) as dataset:
        if "tracer_name" not in dataset.variables:
            return tuple(f"tracer_{index + 1:03d}" for index in range(len(dataset.dimensions["tracer"])))
        decoded = netCDF4.chartostring(dataset.variables["tracer_name"][:])
        return tuple(str(value).strip() for value in decoded)


def write_python_tpcore_trace(input_path: str | Path, output_path: str | Path) -> Path:
    """Write Python TPCORE stage checkpoints for one transport fixture."""

    fixture = read_tpcore_input(input_path)
    state, trace = trace_tpcore_one_step(
        tracer_conc=fixture.tracer_conc,
        p1_hpa=fixture.p1_hpa,
        p2_hpa=fixture.p2_hpa,
        u_m_s=fixture.u_m_s,
        v_m_s=fixture.v_m_s,
        area_m2=fixture.area_m2,
        hyai_hpa=fixture.hyai_hpa,
        hybi=fixture.hybi,
        lat_deg=fixture.lat_deg,
        dt_s=fixture.dt_s,
    )
    setup = setup_tpcore_terms(
        p1_hpa=fixture.p1_hpa,
        p2_hpa=fixture.p2_hpa,
        u_m_s=fixture.u_m_s,
        v_m_s=fixture.v_m_s,
        area_m2=fixture.area_m2,
        hyai_hpa=fixture.hyai_hpa,
        hybi=fixture.hybi,
        lat_deg=fixture.lat_deg,
        dt_s=fixture.dt_s,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ntracer, nlev, nlat, nlon = trace.tracer_conc_after.shape
    with netCDF4.Dataset(output, "w") as dataset:
        dataset.harness = TPCORE_TRACE_VERSION
        dataset.source_input = str(input_path)
        dataset.dt_s = fixture.dt_s
        dataset.createDimension("tracer", ntracer)
        dataset.createDimension("lev", nlev)
        dataset.createDimension("lat", nlat)
        dataset.createDimension("lon", nlon)
        dataset.createVariable("q_after_pole_average", "f8", ("tracer", "lev", "lat", "lon"))[:] = (
            trace.q_after_pole_average
        )
        dataset.createVariable("dq_after_init_hpa", "f8", ("tracer", "lev", "lat", "lon"))[:] = trace.dq_after_init
        dataset.createVariable("q_after_cross_terms", "f8", ("tracer", "lev", "lat", "lon"))[:] = (
            trace.q_after_cross_terms
        )
        dataset.createVariable("dq_after_xtp_hpa", "f8", ("tracer", "lev", "lat", "lon"))[:] = trace.dq_after_xtp
        dataset.createVariable("dq_after_ytp_hpa", "f8", ("tracer", "lev", "lat", "lon"))[:] = trace.dq_after_ytp
        dataset.createVariable("dq_after_fzppm_hpa", "f8", ("tracer", "lev", "lat", "lon"))[:] = trace.dq_after_fzppm
        dataset.createVariable("dq_after_fill_hpa", "f8", ("tracer", "lev", "lat", "lon"))[:] = trace.dq_after_fill
        dataset.createVariable("tracer_conc_after", "f8", ("tracer", "lev", "lat", "lon"))[:] = (
            trace.tracer_conc_after
        )
        dataset.createVariable("cx", "f8", ("lev", "lat", "lon"))[:] = setup.cx
        dataset.createVariable("cy", "f8", ("lev", "lat", "lon"))[:] = setup.cy
        dataset.createVariable("vertical_mass_flux_hpa", "f8", ("lev", "lat", "lon"))[:] = setup.vertical_mass_flux_hpa
        dataset.createVariable("delp1_hpa", "f8", ("lev", "lat", "lon"))[:] = setup.delp1_hpa
        dataset.createVariable("delp2_hpa", "f8", ("lev", "lat", "lon"))[:] = setup.delp2_hpa
        dataset.createVariable("surface_pressure_hpa", "f8", ("lat", "lon"))[:] = state.surface_pressure_hpa
    return output


def compare_pjc_output(input_path: str | Path, output_path: str | Path) -> PjcComparison:
    with netCDF4.Dataset(input_path) as dataset:
        lat = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
        hyai = np.asarray(dataset.variables["hyai"][:], dtype=np.float64)
        hybi = np.asarray(dataset.variables["hybi"][:], dtype=np.float64)
        area = np.asarray(dataset.variables["area_m2"][:], dtype=np.float64)
        p1 = np.asarray(dataset.variables["p1_hpa"][:], dtype=np.float64)
        p2 = np.asarray(dataset.variables["p2_hpa"][:], dtype=np.float64)
        u = np.asarray(dataset.variables["u_m_s"][:], dtype=np.float64)
        v = np.asarray(dataset.variables["v_m_s"][:], dtype=np.float64)
        dt_s = float(getattr(dataset, "dt_s"))
    expected_x, expected_y = pjc_mass_flux_hpa(
        p1_hpa=p1,
        p2_hpa=p2,
        u_m_s=u,
        v_m_s=v,
        area_m2=area,
        hyai_hpa=hyai,
        hybi=hybi,
        lat_deg=lat,
        dt_s=dt_s,
    )
    observed_x, observed_y = read_pjc_output(output_path)
    x_error = np.abs(observed_x - expected_x)
    y_error = np.abs(observed_y - expected_y)
    return PjcComparison(
        xmass_max_abs_error_hpa=float(np.max(x_error)),
        xmass_mean_abs_error_hpa=float(np.mean(x_error)),
        ymass_max_abs_error_hpa=float(np.max(y_error)),
        ymass_mean_abs_error_hpa=float(np.mean(y_error)),
    )


def compare_transport_step_output(input_path: str | Path, output_path: str | Path) -> TransportStepComparison:
    with netCDF4.Dataset(input_path) as dataset:
        if getattr(dataset, "harness", "") != TRANSPORT_INPUT_VERSION:
            raise ValueError(f"{input_path} is not a {TRANSPORT_INPUT_VERSION} file")
        lat = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
        hyai = np.asarray(dataset.variables["hyai"][:], dtype=np.float64)
        hybi = np.asarray(dataset.variables["hybi"][:], dtype=np.float64)
        area = np.asarray(dataset.variables["area_m2"][:], dtype=np.float64)
        p1 = np.asarray(dataset.variables["p1_hpa"][:], dtype=np.float64)
        p2 = np.asarray(dataset.variables["p2_hpa"][:], dtype=np.float64)
        u = np.asarray(dataset.variables["u_m_s"][:], dtype=np.float64)
        v = np.asarray(dataset.variables["v_m_s"][:], dtype=np.float64)
        tracer_before = np.asarray(dataset.variables["tracer_conc"][:], dtype=np.float64)
        dt_s = float(getattr(dataset, "dt_s"))
    expected_x, expected_y = pjc_mass_flux_hpa(
        p1_hpa=p1,
        p2_hpa=p2,
        u_m_s=u,
        v_m_s=v,
        area_m2=area,
        hyai_hpa=hyai,
        hybi=hybi,
        lat_deg=lat,
        dt_s=dt_s,
    )
    output = read_transport_step_output(output_path)
    x_error = np.abs(output.xmass_hpa - expected_x)
    y_error = np.abs(output.ymass_hpa - expected_y)
    tracer_change = np.abs(output.tracer_conc_after - tracer_before)
    return TransportStepComparison(
        xmass_max_abs_error_hpa=float(np.max(x_error)),
        xmass_mean_abs_error_hpa=float(np.mean(x_error)),
        ymass_max_abs_error_hpa=float(np.max(y_error)),
        ymass_mean_abs_error_hpa=float(np.mean(y_error)),
        tracer_max_abs_change=float(np.max(tracer_change)),
        tracer_min_after=float(np.min(output.tracer_conc_after)),
        tracer_max_after=float(np.max(output.tracer_conc_after)),
        negative_count_after=int(np.count_nonzero(output.tracer_conc_after < 0.0)),
        surface_pressure_min_hpa=float(np.min(output.surface_pressure_hpa)),
        surface_pressure_max_hpa=float(np.max(output.surface_pressure_hpa)),
    )


def compare_python_tpcore_output(input_path: str | Path, output_path: str | Path) -> PythonTpcoreComparison:
    with netCDF4.Dataset(input_path) as dataset:
        if getattr(dataset, "harness", "") != TRANSPORT_INPUT_VERSION:
            raise ValueError(f"{input_path} is not a {TRANSPORT_INPUT_VERSION} file")
        lat = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
        hyai = np.asarray(dataset.variables["hyai"][:], dtype=np.float64)
        hybi = np.asarray(dataset.variables["hybi"][:], dtype=np.float64)
        area = np.asarray(dataset.variables["area_m2"][:], dtype=np.float64)
        p1 = np.asarray(dataset.variables["p1_hpa"][:], dtype=np.float64)
        p2 = np.asarray(dataset.variables["p2_hpa"][:], dtype=np.float64)
        u = np.asarray(dataset.variables["u_m_s"][:], dtype=np.float64)
        v = np.asarray(dataset.variables["v_m_s"][:], dtype=np.float64)
        tracer = np.asarray(dataset.variables["tracer_conc"][:], dtype=np.float64)
        dt_s = float(getattr(dataset, "dt_s"))
    expected = read_transport_step_output(output_path)
    actual = run_tpcore_one_step(
        tracer_conc=tracer,
        p1_hpa=p1,
        p2_hpa=p2,
        u_m_s=u,
        v_m_s=v,
        area_m2=area,
        hyai_hpa=hyai,
        hybi=hybi,
        lat_deg=lat,
        dt_s=dt_s,
    )
    setup = setup_tpcore_terms(
        p1_hpa=p1,
        p2_hpa=p2,
        u_m_s=u,
        v_m_s=v,
        area_m2=area,
        hyai_hpa=hyai,
        hybi=hybi,
        lat_deg=lat,
        dt_s=dt_s,
    )
    x_error = np.abs(actual.xmass_hpa - expected.xmass_hpa)
    y_error = np.abs(actual.ymass_hpa - expected.ymass_hpa)
    ps_error = np.abs(actual.surface_pressure_hpa - expected.surface_pressure_hpa)
    tracer_error = np.abs(actual.tracer_conc_after - expected.tracer_conc_after)
    return PythonTpcoreComparison(
        xmass_max_abs_error_hpa=float(np.max(x_error)),
        xmass_mean_abs_error_hpa=float(np.mean(x_error)),
        ymass_max_abs_error_hpa=float(np.max(y_error)),
        ymass_mean_abs_error_hpa=float(np.mean(y_error)),
        surface_pressure_max_abs_error_hpa=float(np.max(ps_error)),
        surface_pressure_mean_abs_error_hpa=float(np.mean(ps_error)),
        tracer_max_abs_error=float(np.max(tracer_error)),
        tracer_mean_abs_error=float(np.mean(tracer_error)),
        negative_count_after=int(np.count_nonzero(actual.tracer_conc_after < 0.0)),
        max_abs_cx=float(np.max(np.abs(setup.cx))),
        max_abs_cy=float(np.max(np.abs(setup.cy))),
    )


def attribute_python_tpcore_error(input_path: str | Path, output_path: str | Path) -> str:
    """Summarize where Python-vs-oracle final tracer error is concentrated."""

    fixture = read_tpcore_input(input_path)
    expected = read_transport_step_output(output_path)
    actual = run_tpcore_one_step(
        tracer_conc=fixture.tracer_conc,
        p1_hpa=fixture.p1_hpa,
        p2_hpa=fixture.p2_hpa,
        u_m_s=fixture.u_m_s,
        v_m_s=fixture.v_m_s,
        area_m2=fixture.area_m2,
        hyai_hpa=fixture.hyai_hpa,
        hybi=fixture.hybi,
        lat_deg=fixture.lat_deg,
        dt_s=fixture.dt_s,
    )
    setup = setup_tpcore_terms(
        p1_hpa=fixture.p1_hpa,
        p2_hpa=fixture.p2_hpa,
        u_m_s=fixture.u_m_s,
        v_m_s=fixture.v_m_s,
        area_m2=fixture.area_m2,
        hyai_hpa=fixture.hyai_hpa,
        hybi=fixture.hybi,
        lat_deg=fixture.lat_deg,
        dt_s=fixture.dt_s,
    )
    error = np.abs(actual.tracer_conc_after - expected.tracer_conc_after)
    rows = ["section,key,max_abs,mean_abs,count,extra"]
    rows.append(_error_row("global", "all", error))
    rows.append(_top_cell_row(error))
    rows.extend(_top_axis_rows("level", error, axis=1, top_n=8))
    rows.extend(_top_axis_rows("latitude", error, axis=2, top_n=8))
    rows.extend(_top_axis_rows("longitude", error, axis=3, top_n=8))
    rows.extend(_bin_rows("abs_cx", error, np.abs(setup.cx), (0.0, 0.1, 0.5, 1.0, np.inf)))
    rows.extend(_bin_rows("abs_cy", error, np.abs(setup.cy), (0.0, 0.05, 0.1, 0.5, 1.0, np.inf)))
    rows.extend(_bin_rows("abs_wz_hpa", error, np.abs(setup.vertical_mass_flux_hpa), (0.0, 0.01, 0.1, 1.0, 10.0, np.inf)))
    rows.extend(_bin_rows("initial_gradient", error, _tracer_gradient_magnitude(fixture.tracer_conc), (0.0, 1e-8, 1e-7, 1e-6, np.inf)))
    return "\n".join(rows)


def compare_tpcore_trace_files(expected_path: str | Path, actual_path: str | Path) -> str:
    """Compare two trace NetCDF files with the same checkpoint contract."""

    stage_names = (
        "delp1_hpa",
        "delp2_hpa",
        "cx",
        "cy",
        "vertical_mass_flux_hpa",
        "surface_pressure_hpa",
        "q_after_pole_average",
        "dq_after_init_hpa",
        "q_after_cross_terms",
        "dq_after_xtp_hpa",
        "dq_after_ytp_hpa",
        "dq_after_fzppm_hpa",
        "dq_after_fill_hpa",
        "tracer_conc_after",
    )
    rows = ["stage,max_abs,mean_abs,top_index"]
    with netCDF4.Dataset(expected_path) as expected, netCDF4.Dataset(actual_path) as actual:
        if getattr(expected, "harness", "") != TPCORE_TRACE_VERSION:
            raise ValueError(f"{expected_path} is not a {TPCORE_TRACE_VERSION} file")
        if getattr(actual, "harness", "") != TPCORE_TRACE_VERSION:
            raise ValueError(f"{actual_path} is not a {TPCORE_TRACE_VERSION} file")
        for stage in stage_names:
            expected_values = np.asarray(expected.variables[stage][:], dtype=np.float64)
            actual_values = np.asarray(actual.variables[stage][:], dtype=np.float64)
            err = np.abs(actual_values - expected_values)
            top = tuple(int(index) for index in np.unravel_index(int(np.argmax(err)), err.shape))
            rows.append(f"{stage},{float(np.max(err)):.8e},{float(np.mean(err)):.8e},{top}")
    return "\n".join(rows)


def generate_large_oracle_tpcore_trace(
    fixture_id: str,
    *,
    cache_dir: str | Path = "oracle_data",
    manifest_dir: str | Path | None = None,
    executable: str | Path = "tools/gc_harness/build/pjc_pfix_harness_trace",
) -> Path:
    """Run the instrumented GEOS-Chem harness and write oracle_tpcore_trace.nc."""

    check = check_large_oracle_fixture(fixture_id, cache_dir=cache_dir, manifest_dir=manifest_dir)
    if not check.is_available:
        raise FileNotFoundError(format_large_oracle_fixture_check(check))
    paths = large_oracle_fixture_paths(fixture_id, cache_dir=cache_dir, manifest_dir=manifest_dir)
    run_pjc_harness(executable, paths.input_path, paths.output_path, paths.directory / ORACLE_TPCORE_TRACE_NAME)
    return paths.directory / ORACLE_TPCORE_TRACE_NAME


def trace_compare_large_oracle_fixture(
    fixture_id: str,
    *,
    cache_dir: str | Path = "oracle_data",
    manifest_dir: str | Path | None = None,
) -> str:
    """Write Python trace for a large fixture and compare oracle trace if present."""

    check = check_large_oracle_fixture(fixture_id, cache_dir=cache_dir, manifest_dir=manifest_dir)
    if not check.is_available:
        raise FileNotFoundError(format_large_oracle_fixture_check(check))
    paths = large_oracle_fixture_paths(fixture_id, cache_dir=cache_dir, manifest_dir=manifest_dir)
    python_trace = write_python_tpcore_trace(paths.input_path, paths.directory / PYTHON_TPCORE_TRACE_NAME)
    oracle_trace = paths.directory / ORACLE_TPCORE_TRACE_NAME
    rows = [
        "metric,value",
        f"fixture_id,{fixture_id}",
        f"python_trace,{python_trace}",
        f"oracle_trace,{oracle_trace}",
        f"oracle_trace_available,{oracle_trace.exists()}",
    ]
    if oracle_trace.exists():
        rows.append("")
        rows.append(compare_tpcore_trace_files(oracle_trace, python_trace))
    else:
        rows.append("")
        rows.append("# final-error-attribution")
        rows.append(attribute_python_tpcore_error(paths.input_path, paths.output_path))
    return "\n".join(rows)


def _error_row(section: str, key: str, error: np.ndarray, *, extra: str = "") -> str:
    return f"{section},{key},{float(np.max(error)):.8e},{float(np.mean(error)):.8e},{error.size},{extra}"


def _top_cell_row(error: np.ndarray) -> str:
    top = np.unravel_index(int(np.argmax(error)), error.shape)
    extra = f"tracer={top[0]} lev={top[1]} lat={top[2]} lon={top[3]}"
    return _error_row("top_cell", "max", error[top], extra=extra)


def _top_axis_rows(section: str, error: np.ndarray, *, axis: int, top_n: int) -> list[str]:
    max_by_axis = np.max(error, axis=tuple(idx for idx in range(error.ndim) if idx != axis))
    mean_by_axis = np.mean(error, axis=tuple(idx for idx in range(error.ndim) if idx != axis))
    order = np.argsort(max_by_axis)[::-1][:top_n]
    rows = []
    for idx in order:
        rows.append(
            f"{section},{int(idx)},{float(max_by_axis[idx]):.8e},{float(mean_by_axis[idx]):.8e},"
            f"{error.size // error.shape[axis]},"
        )
    return rows


def _bin_rows(section: str, error: np.ndarray, values: np.ndarray, edges: tuple[float, ...]) -> list[str]:
    rows = []
    value_4d = values[np.newaxis, :, :, :] if values.ndim == 3 else values
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (value_4d >= low) & (value_4d < high)
        mask = np.broadcast_to(mask, error.shape)
        if not np.any(mask):
            rows.append(f"{section},[{low:.3e},{high:.3e}),nan,nan,0,")
            continue
        err = error[mask]
        rows.append(f"{section},[{low:.3e},{high:.3e}),{float(np.max(err)):.8e},{float(np.mean(err)):.8e},{err.size},")
    return rows


def _tracer_gradient_magnitude(tracer: np.ndarray) -> np.ndarray:
    grad = np.zeros_like(tracer, dtype=np.float64)
    for axis in (1, 2, 3):
        grad += np.gradient(tracer, axis=axis) ** 2
    return np.sqrt(grad)


def run_pjc_harness(
    executable: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    trace_output_path: str | Path | None = None,
) -> None:
    executable = Path(executable)
    if not executable.exists():
        raise FileNotFoundError(
            f"GEOS-Chem harness executable not found: {executable}. "
            "Build tools/gc_harness/pjc_pfix_harness.F90 first."
        )
    command = [str(executable), str(input_path), str(output_path)]
    if trace_output_path is not None:
        command.append(str(trace_output_path))
    subprocess.run(command, check=True)


def _load_large_oracle_definition(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if "fixture_id" not in data:
        raise ValueError(f"{path} is missing fixture_id")
    if "files" not in data:
        raise ValueError(f"{path} is missing files")
    return data


def _large_oracle_file_entries(manifest: dict[str, object]) -> tuple[dict[str, object], ...]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("large oracle manifest files must be a list")
    entries: list[dict[str, object]] = []
    for entry in files:
        if not isinstance(entry, dict) or not entry.get("name"):
            raise ValueError("large oracle manifest file entries must include name")
        entries.append(entry)
    return tuple(entries)


def _write_generated_large_oracle_manifest(
    paths: LargeOracleFixturePaths,
    *,
    definition: dict[str, object],
    executable: Path,
    run_config: Path,
    repo_root: Path,
) -> None:
    setup = _setup_tpcore_from_input(paths.input_path)
    report = analyze_tpcore_branches(setup)
    manifest = {
        "fixture_id": paths.fixture_id,
        "description": definition.get("description"),
        "definition_file": str(paths.definition_path),
        "input_harness": TRANSPORT_INPUT_VERSION,
        "output_harness": TRANSPORT_OUTPUT_VERSION,
        "files": [
            _large_oracle_file_record(paths.input_path),
            _large_oracle_file_record(paths.output_path),
        ],
        "source": {
            **dict(definition.get("source", {})),
            "run_config": str(run_config),
        },
        "executable": str(executable),
        "gcclassic_head": _git_head(repo_root / "GCClassic"),
        "branch_report": {
            "shape": list(report.shape),
            "max_abs_cx": report.max_abs_cx,
            "max_abs_cy": report.max_abs_cy,
            "has_large_cx": report.has_large_cx,
            "has_large_cy": report.has_large_cy,
            "needs_fxppm": report.needs_fxppm,
            "is_supported": report.is_supported,
            "unsupported_reasons": list(report.unsupported_reasons),
        },
    }
    with paths.manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _large_oracle_file_record(path: Path) -> dict[str, object]:
    return {
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "url": None,
    }


def _setup_tpcore_from_input(input_path: str | Path):
    with netCDF4.Dataset(input_path) as dataset:
        if getattr(dataset, "harness", "") != TRANSPORT_INPUT_VERSION:
            raise ValueError(f"{input_path} is not a {TRANSPORT_INPUT_VERSION} file")
        return setup_tpcore_terms(
            p1_hpa=np.asarray(dataset.variables["p1_hpa"][:], dtype=np.float64),
            p2_hpa=np.asarray(dataset.variables["p2_hpa"][:], dtype=np.float64),
            u_m_s=np.asarray(dataset.variables["u_m_s"][:], dtype=np.float64),
            v_m_s=np.asarray(dataset.variables["v_m_s"][:], dtype=np.float64),
            area_m2=np.asarray(dataset.variables["area_m2"][:], dtype=np.float64),
            hyai_hpa=np.asarray(dataset.variables["hyai"][:], dtype=np.float64),
            hybi=np.asarray(dataset.variables["hybi"][:], dtype=np.float64),
            lat_deg=np.asarray(dataset.variables["lat"][:], dtype=np.float64),
            dt_s=float(getattr(dataset, "dt_s")),
        )


def _spherical_band_area(lat_deg: np.ndarray, nlon: int) -> np.ndarray:
    from wombat_transport.constants import EARTH_RADIUS_M

    lat = np.asarray(lat_deg, dtype=np.float64)
    edges = np.empty(lat.size + 1, dtype=np.float64)
    edges[0] = -90.0
    edges[-1] = 90.0
    edges[1:-1] = 0.5 * (lat[:-1] + lat[1:])
    band_area = (2.0 * np.pi / float(nlon)) * (EARTH_RADIUS_M**2)
    band_area = band_area * (np.sin(np.deg2rad(edges[1:])) - np.sin(np.deg2rad(edges[:-1])))
    return np.broadcast_to(band_area[:, np.newaxis], (lat.size, nlon)).copy()


def _pjc_snapshot_metadata(
    input_path: Path,
    output_path: Path,
    *,
    executable: Path,
    repo_root: Path,
) -> dict[str, object]:
    with netCDF4.Dataset(input_path) as dataset:
        nlev = len(dataset.dimensions["lev"])
        nlat = len(dataset.dimensions["lat"])
        nlon = len(dataset.dimensions["lon"])
        dt_s = float(getattr(dataset, "dt_s"))
    return {
        "snapshot": PJC_SNAPSHOT_VERSION,
        "input_harness": PJC_INPUT_VERSION,
        "output_harness": PJC_OUTPUT_VERSION,
        "input_file": input_path.name,
        "output_file": output_path.name,
        "shape": {"lev": nlev, "lat": nlat, "lon": nlon},
        "dt_s": dt_s,
        "executable": str(executable),
        "gcclassic_head": _git_head(repo_root / "GCClassic"),
    }


def _tpcore_snapshot_metadata(
    input_path: Path,
    output_path: Path,
    *,
    executable: Path,
    repo_root: Path,
) -> dict[str, object]:
    with netCDF4.Dataset(input_path) as dataset:
        nlev = len(dataset.dimensions["lev"])
        nlat = len(dataset.dimensions["lat"])
        nlon = len(dataset.dimensions["lon"])
        ntracer = len(dataset.dimensions["tracer"])
        dt_s = float(getattr(dataset, "dt_s"))
    return {
        "snapshot": TPCORE_SNAPSHOT_VERSION,
        "input_harness": TRANSPORT_INPUT_VERSION,
        "output_harness": TRANSPORT_OUTPUT_VERSION,
        "input_file": input_path.name,
        "output_file": output_path.name,
        "shape": {"tracer": ntracer, "lev": nlev, "lat": nlat, "lon": nlon},
        "dt_s": dt_s,
        "executable": str(executable),
        "gcclassic_head": _git_head(repo_root / "GCClassic"),
    }


def _git_head(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def summarize_transport_step_output(output: TransportStepOutput) -> str:
    return "\n".join(
        [
            "metric,value",
            f"tracer_min,{float(np.min(output.tracer_conc_after)):.8e}",
            f"tracer_max,{float(np.max(output.tracer_conc_after)):.8e}",
            f"xmass_min_hpa,{float(np.min(output.xmass_hpa)):.8e}",
            f"xmass_max_hpa,{float(np.max(output.xmass_hpa)):.8e}",
            f"ymass_min_hpa,{float(np.min(output.ymass_hpa)):.8e}",
            f"ymass_max_hpa,{float(np.max(output.ymass_hpa)):.8e}",
            f"surface_pressure_min_hpa,{float(np.min(output.surface_pressure_hpa)):.8e}",
            f"surface_pressure_max_hpa,{float(np.max(output.surface_pressure_hpa)):.8e}",
        ]
    )


def format_pjc_comparison(comparison: PjcComparison) -> str:
    return "\n".join(
        [
            "metric,value",
            f"xmass_max_abs_error_hpa,{comparison.xmass_max_abs_error_hpa:.8e}",
            f"xmass_mean_abs_error_hpa,{comparison.xmass_mean_abs_error_hpa:.8e}",
            f"ymass_max_abs_error_hpa,{comparison.ymass_max_abs_error_hpa:.8e}",
            f"ymass_mean_abs_error_hpa,{comparison.ymass_mean_abs_error_hpa:.8e}",
        ]
    )


def format_vdiff_comparison(comparison: VdiffComparison) -> str:
    return "\n".join(
        [
            "metric,value",
            f"tracer_max_abs_error,{comparison.tracer_max_abs_error:.8e}",
            f"tracer_mean_abs_error,{comparison.tracer_mean_abs_error:.8e}",
            f"specific_humidity_max_abs_error,{comparison.specific_humidity_max_abs_error:.8e}",
            f"kvh_max_abs_error,{comparison.kvh_max_abs_error:.8e}",
            f"kvm_max_abs_error,{comparison.kvm_max_abs_error:.8e}",
            f"pbl_top_max_abs_error_m,{comparison.pbl_top_max_abs_error_m:.8e}",
            f"tpert_max_abs_error,{comparison.tpert_max_abs_error:.8e}",
            f"qpert_max_abs_error,{comparison.qpert_max_abs_error:.8e}",
            f"negative_count_before_clip_expected,{comparison.negative_count_before_clip_expected}",
            f"negative_count_before_clip_actual,{comparison.negative_count_before_clip_actual}",
            f"negative_count_after_clip_expected,{comparison.negative_count_after_clip_expected}",
            f"negative_count_after_clip_actual,{comparison.negative_count_after_clip_actual}",
            f"final_mass_max_abs_error,{comparison.final_mass_max_abs_error:.8e}",
        ]
    )


def format_transport_step_comparison(comparison: TransportStepComparison) -> str:
    return "\n".join(
        [
            "metric,value",
            f"xmass_max_abs_error_hpa,{comparison.xmass_max_abs_error_hpa:.8e}",
            f"xmass_mean_abs_error_hpa,{comparison.xmass_mean_abs_error_hpa:.8e}",
            f"ymass_max_abs_error_hpa,{comparison.ymass_max_abs_error_hpa:.8e}",
            f"ymass_mean_abs_error_hpa,{comparison.ymass_mean_abs_error_hpa:.8e}",
            f"tracer_max_abs_change,{comparison.tracer_max_abs_change:.8e}",
            f"tracer_min_after,{comparison.tracer_min_after:.8e}",
            f"tracer_max_after,{comparison.tracer_max_after:.8e}",
            f"negative_count_after,{comparison.negative_count_after}",
            f"surface_pressure_min_hpa,{comparison.surface_pressure_min_hpa:.8e}",
            f"surface_pressure_max_hpa,{comparison.surface_pressure_max_hpa:.8e}",
        ]
    )


def format_python_tpcore_comparison(comparison: PythonTpcoreComparison) -> str:
    return "\n".join(
        [
            "metric,value",
            f"xmass_max_abs_error_hpa,{comparison.xmass_max_abs_error_hpa:.8e}",
            f"xmass_mean_abs_error_hpa,{comparison.xmass_mean_abs_error_hpa:.8e}",
            f"ymass_max_abs_error_hpa,{comparison.ymass_max_abs_error_hpa:.8e}",
            f"ymass_mean_abs_error_hpa,{comparison.ymass_mean_abs_error_hpa:.8e}",
            f"surface_pressure_max_abs_error_hpa,{comparison.surface_pressure_max_abs_error_hpa:.8e}",
            f"surface_pressure_mean_abs_error_hpa,{comparison.surface_pressure_mean_abs_error_hpa:.8e}",
            f"tracer_max_abs_error,{comparison.tracer_max_abs_error:.8e}",
            f"tracer_mean_abs_error,{comparison.tracer_mean_abs_error:.8e}",
            f"python_negative_count_after,{comparison.negative_count_after}",
            f"max_abs_cx,{comparison.max_abs_cx:.8e}",
            f"max_abs_cy,{comparison.max_abs_cy:.8e}",
        ]
    )


def format_large_oracle_fixture_check(check: LargeOracleFixtureCheck) -> str:
    rows = [
        "metric,value",
        f"fixture_id,{check.fixture_id}",
        f"manifest,{check.manifest_path}",
        f"available,{check.is_available}",
        f"missing_files,{' | '.join(check.missing_files)}",
        f"checksum_failures,{' | '.join(check.checksum_failures)}",
        f"unchecked_files,{' | '.join(check.unchecked_files)}",
    ]
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare and compare GEOS-Chem operator harness fixtures.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    write_parser = subparsers.add_parser("write-pjc-input")
    write_parser.add_argument("run_config", type=Path)
    write_parser.add_argument("output", type=Path)
    write_parser.add_argument("--time-index", type=int, default=0)
    write_parser.add_argument("--dt-s", type=float, default=None)

    run_parser = subparsers.add_parser("pjc-pfix")
    run_parser.add_argument("run_config", type=Path)
    run_parser.add_argument("--work-dir", type=Path, default=Path("tools/gc_harness/work"))
    run_parser.add_argument("--executable", type=Path, default=Path("tools/gc_harness/build/pjc_pfix_harness"))
    run_parser.add_argument("--time-index", type=int, default=0)

    transport_parser = subparsers.add_parser("transport-step")
    transport_parser.add_argument("run_config", type=Path)
    transport_parser.add_argument("--work-dir", type=Path, default=Path("tools/gc_harness/work"))
    transport_parser.add_argument("--executable", type=Path, default=Path("tools/gc_harness/build/pjc_pfix_harness"))
    transport_parser.add_argument("--time-index", type=int, default=0)
    transport_parser.add_argument("--tracer-time-index", type=int, default=0)
    transport_parser.add_argument("--max-tracers", type=int, default=1)

    compare_parser = subparsers.add_parser("compare-pjc-output")
    compare_parser.add_argument("input", type=Path)
    compare_parser.add_argument("output", type=Path)

    compare_transport_parser = subparsers.add_parser("compare-transport-step-output")
    compare_transport_parser.add_argument("input", type=Path)
    compare_transport_parser.add_argument("output", type=Path)

    write_vdiff_parser = subparsers.add_parser("write-synthetic-vdiff-input")
    write_vdiff_parser.add_argument("output", type=Path)
    write_vdiff_parser.add_argument("--dt-s", type=float, default=600.0)
    write_vdiff_parser.add_argument("--ntracer", type=int, default=2)

    python_vdiff_parser = subparsers.add_parser("python-vdiff-output")
    python_vdiff_parser.add_argument("input", type=Path)
    python_vdiff_parser.add_argument("output", type=Path)

    compare_vdiff_parser = subparsers.add_parser("compare-vdiff-output")
    compare_vdiff_parser.add_argument("input", type=Path)
    compare_vdiff_parser.add_argument("output", type=Path)

    compare_python_tpcore_parser = subparsers.add_parser("compare-python-tpcore-output")
    compare_python_tpcore_parser.add_argument("input", type=Path)
    compare_python_tpcore_parser.add_argument("output", type=Path)

    trace_python_tpcore_parser = subparsers.add_parser("trace-python-tpcore")
    trace_python_tpcore_parser.add_argument("input", type=Path)
    trace_python_tpcore_parser.add_argument("output", type=Path)

    attribute_python_tpcore_parser = subparsers.add_parser("attribute-python-tpcore-error")
    attribute_python_tpcore_parser.add_argument("input", type=Path)
    attribute_python_tpcore_parser.add_argument("output", type=Path)

    compare_tpcore_trace_parser = subparsers.add_parser("compare-tpcore-trace")
    compare_tpcore_trace_parser.add_argument("expected", type=Path)
    compare_tpcore_trace_parser.add_argument("actual", type=Path)

    snapshot_parser = subparsers.add_parser("snapshot-pjc")
    snapshot_parser.add_argument("output_dir", type=Path)
    snapshot_parser.add_argument("--executable", type=Path, default=Path("tools/gc_harness/build/pjc_pfix_harness"))
    snapshot_parser.add_argument("--dt-s", type=float, default=600.0)

    tpcore_snapshot_parser = subparsers.add_parser("snapshot-tpcore")
    tpcore_snapshot_parser.add_argument("output_dir", type=Path)
    tpcore_snapshot_parser.add_argument("--executable", type=Path, default=Path("tools/gc_harness/build/pjc_pfix_harness"))
    tpcore_snapshot_parser.add_argument("--dt-s", type=float, default=600.0)
    tpcore_snapshot_parser.add_argument("--ntracer", type=int, default=2)

    tpcore_branch_snapshot_parser = subparsers.add_parser("snapshot-tpcore-branch")
    tpcore_branch_snapshot_parser.add_argument("scenario", choices=TPCORE_BRANCH_SCENARIOS)
    tpcore_branch_snapshot_parser.add_argument("output_dir", type=Path)
    tpcore_branch_snapshot_parser.add_argument("--executable", type=Path, default=Path("tools/gc_harness/build/pjc_pfix_harness"))
    tpcore_branch_snapshot_parser.add_argument("--dt-s", type=float, default=600.0)
    tpcore_branch_snapshot_parser.add_argument("--ntracer", type=int, default=2)

    generate_oracle_parser = subparsers.add_parser("oracle-fixture-generate")
    generate_oracle_parser.add_argument("fixture_id", choices=LARGE_ORACLE_FIXTURE_IDS)
    generate_oracle_parser.add_argument("--cache-dir", type=Path, default=Path("oracle_data"))
    generate_oracle_parser.add_argument("--manifest-dir", type=Path, default=None)
    generate_oracle_parser.add_argument("--run-config", type=Path, default=None)
    generate_oracle_parser.add_argument("--executable", type=Path, default=Path("tools/gc_harness/build/pjc_pfix_harness"))
    generate_oracle_parser.add_argument("--time-index", type=int, default=None)
    generate_oracle_parser.add_argument("--tracer-time-index", type=int, default=None)
    generate_oracle_parser.add_argument("--max-tracers", type=int, default=None)
    generate_oracle_parser.add_argument("--dt-s", type=float, default=None)

    fetch_oracle_parser = subparsers.add_parser("oracle-fixture-fetch")
    fetch_oracle_parser.add_argument("fixture_id", choices=LARGE_ORACLE_FIXTURE_IDS)
    fetch_oracle_parser.add_argument("--cache-dir", type=Path, default=Path("oracle_data"))
    fetch_oracle_parser.add_argument("--manifest-dir", type=Path, default=None)
    fetch_oracle_parser.add_argument("--overwrite", action="store_true")

    check_oracle_parser = subparsers.add_parser("oracle-fixture-check")
    check_oracle_parser.add_argument("fixture_id", choices=LARGE_ORACLE_FIXTURE_IDS)
    check_oracle_parser.add_argument("--cache-dir", type=Path, default=Path("oracle_data"))
    check_oracle_parser.add_argument("--manifest-dir", type=Path, default=None)

    compare_oracle_parser = subparsers.add_parser("oracle-fixture-compare")
    compare_oracle_parser.add_argument("fixture_id", choices=LARGE_ORACLE_FIXTURE_IDS)
    compare_oracle_parser.add_argument("--cache-dir", type=Path, default=Path("oracle_data"))
    compare_oracle_parser.add_argument("--manifest-dir", type=Path, default=None)

    trace_compare_oracle_parser = subparsers.add_parser("oracle-fixture-trace-compare")
    trace_compare_oracle_parser.add_argument("fixture_id", choices=LARGE_ORACLE_FIXTURE_IDS)
    trace_compare_oracle_parser.add_argument("--cache-dir", type=Path, default=Path("oracle_data"))
    trace_compare_oracle_parser.add_argument("--manifest-dir", type=Path, default=None)

    trace_generate_oracle_parser = subparsers.add_parser("oracle-fixture-trace-generate")
    trace_generate_oracle_parser.add_argument("fixture_id", choices=LARGE_ORACLE_FIXTURE_IDS)
    trace_generate_oracle_parser.add_argument("--cache-dir", type=Path, default=Path("oracle_data"))
    trace_generate_oracle_parser.add_argument("--manifest-dir", type=Path, default=None)
    trace_generate_oracle_parser.add_argument(
        "--executable",
        type=Path,
        default=Path("tools/gc_harness/build/pjc_pfix_harness_trace"),
    )

    args = parser.parse_args(argv)
    if args.command == "write-pjc-input":
        path = write_pjc_input_from_config(args.run_config, args.output, time_index=args.time_index, dt_s=args.dt_s)
        print(f"wrote_pjc_input: {path}")
        return 0
    if args.command == "pjc-pfix":
        args.work_dir.mkdir(parents=True, exist_ok=True)
        input_path = args.work_dir / "pjc_input.nc"
        output_path = args.work_dir / "pjc_output.nc"
        write_pjc_input_from_config(args.run_config, input_path, time_index=args.time_index)
        run_pjc_harness(args.executable, input_path, output_path)
        print(format_pjc_comparison(compare_pjc_output(input_path, output_path)))
        return 0
    if args.command == "transport-step":
        args.work_dir.mkdir(parents=True, exist_ok=True)
        input_path = args.work_dir / "transport_step_input.nc"
        output_path = args.work_dir / "transport_step_output.nc"
        write_transport_step_input_from_config(
            args.run_config,
            input_path,
            time_index=args.time_index,
            tracer_time_index=args.tracer_time_index,
            max_tracers=args.max_tracers,
        )
        run_pjc_harness(args.executable, input_path, output_path)
        print(summarize_transport_step_output(read_transport_step_output(output_path)))
        return 0
    if args.command == "compare-pjc-output":
        print(format_pjc_comparison(compare_pjc_output(args.input, args.output)))
        return 0
    if args.command == "compare-transport-step-output":
        print(format_transport_step_comparison(compare_transport_step_output(args.input, args.output)))
        return 0
    if args.command == "write-synthetic-vdiff-input":
        path = write_synthetic_vdiff_input(args.output, dt_s=args.dt_s, ntracer=args.ntracer)
        print(f"wrote_vdiff_input: {path}")
        return 0
    if args.command == "python-vdiff-output":
        path = write_python_vdiff_output(args.input, args.output)
        print(f"wrote_vdiff_output: {path}")
        return 0
    if args.command == "compare-vdiff-output":
        print(format_vdiff_comparison(compare_vdiff_output(args.input, args.output)))
        return 0
    if args.command == "compare-python-tpcore-output":
        print(format_python_tpcore_comparison(compare_python_tpcore_output(args.input, args.output)))
        return 0
    if args.command == "trace-python-tpcore":
        print(f"wrote_python_tpcore_trace: {write_python_tpcore_trace(args.input, args.output)}")
        return 0
    if args.command == "attribute-python-tpcore-error":
        print(attribute_python_tpcore_error(args.input, args.output))
        return 0
    if args.command == "compare-tpcore-trace":
        print(compare_tpcore_trace_files(args.expected, args.actual))
        return 0
    if args.command == "snapshot-pjc":
        output_dir = snapshot_pjc_oracle(args.output_dir, executable=args.executable, dt_s=args.dt_s)
        print(f"wrote_pjc_snapshot: {output_dir}")
        return 0
    if args.command == "snapshot-tpcore":
        output_dir = snapshot_tpcore_oracle(
            args.output_dir,
            executable=args.executable,
            dt_s=args.dt_s,
            ntracer=args.ntracer,
        )
        print(f"wrote_tpcore_snapshot: {output_dir}")
        return 0
    if args.command == "snapshot-tpcore-branch":
        output_dir = snapshot_tpcore_branch_oracle(
            args.output_dir,
            scenario=args.scenario,
            executable=args.executable,
            dt_s=args.dt_s,
            ntracer=args.ntracer,
        )
        print(f"wrote_tpcore_branch_snapshot: {output_dir}")
        return 0
    if args.command == "oracle-fixture-generate":
        output_dir = generate_large_oracle_fixture(
            args.fixture_id,
            cache_dir=args.cache_dir,
            manifest_dir=args.manifest_dir,
            run_config=args.run_config,
            executable=args.executable,
            time_index=args.time_index,
            tracer_time_index=args.tracer_time_index,
            max_tracers=args.max_tracers,
            dt_s=args.dt_s,
        )
        print(f"wrote_oracle_fixture: {output_dir}")
        return 0
    if args.command == "oracle-fixture-fetch":
        output_dir = fetch_large_oracle_fixture(
            args.fixture_id,
            cache_dir=args.cache_dir,
            manifest_dir=args.manifest_dir,
            overwrite=args.overwrite,
        )
        print(f"fetched_oracle_fixture: {output_dir}")
        return 0
    if args.command == "oracle-fixture-check":
        print(
            format_large_oracle_fixture_check(
                check_large_oracle_fixture(args.fixture_id, cache_dir=args.cache_dir, manifest_dir=args.manifest_dir)
            )
        )
        return 0
    if args.command == "oracle-fixture-compare":
        print(compare_large_oracle_fixture(args.fixture_id, cache_dir=args.cache_dir, manifest_dir=args.manifest_dir))
        return 0
    if args.command == "oracle-fixture-trace-compare":
        print(
            trace_compare_large_oracle_fixture(
                args.fixture_id,
                cache_dir=args.cache_dir,
                manifest_dir=args.manifest_dir,
            )
        )
        return 0
    if args.command == "oracle-fixture-trace-generate":
        path = generate_large_oracle_tpcore_trace(
            args.fixture_id,
            cache_dir=args.cache_dir,
            manifest_dir=args.manifest_dir,
            executable=args.executable,
        )
        print(f"wrote_oracle_tpcore_trace: {path}")
        return 0
    raise AssertionError(f"unhandled command {args.command}")


def _assert_pjc_shapes(
    lat: np.ndarray,
    lon: np.ndarray,
    area: np.ndarray,
    hyai: np.ndarray,
    hybi: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> None:
    nlev = hyai.size - 1
    horizontal = (lat.size, lon.size)
    if hybi.shape != hyai.shape:
        raise ValueError("hyai and hybi must have matching edge dimensions")
    if area.shape != horizontal:
        raise ValueError(f"area_m2 must have shape {horizontal}, found {area.shape}")
    if p1.shape != horizontal or p2.shape != horizontal:
        raise ValueError("p1_hpa and p2_hpa must have shape (lat, lon)")
    expected_3d = (nlev, lat.size, lon.size)
    if u.shape != expected_3d or v.shape != expected_3d:
        raise ValueError(f"u_m_s and v_m_s must have shape {expected_3d}")


def _resolve_config_value(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


if __name__ == "__main__":
    raise SystemExit(main())
