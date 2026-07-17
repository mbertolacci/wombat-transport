from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse
from urllib.request import urlretrieve

import netCDF4
import numpy as np

from wombat_transport.constants import (
    AIRMW_G_PER_MOL,
    G0_M_PER_S2,
    H2OMW_G_PER_MOL,
    RD_J_PER_KG_K,
)
from wombat_transport.fields import (
    TracerField,
    canonical_time_slice,
    transport_tracer_to_canonical,
)
from wombat_transport.grid import TransportGrid, load_transport_grid
from wombat_transport.io import initialize_tracers
from wombat_transport.output import (
    HistoryOutputManager,
    OutputCollectionConfig,
    OutputSnapshot,
    OutputStorageConfig,
    parse_history_interval,
)
from wombat_transport.run_config import (
    load_run_config,
    meteorology_initial_time_index,
    meteorology_root,
    simulation_start,
    transport_timestep_s,
)
from wombat_transport.transport import (
    merra2_filename,
    dry_air_mass_from_pressure,
    dry_pressure_edges_from_thickness_hpa,
    dry_pressure_thickness_from_surface_hpa,
    dry_pressure_thickness_hpa,
    dry_surface_pressure_hpa,
    load_transport_forcing,
    pjc_mass_flux_hpa,
    run_cloud_convection_one_step,
    run_vdiffdr_one_step,
    wet_surface_pressure_hpa,
    _map_met_levels_to_47,
)
from wombat_transport.transport.driver import _build_convection_input_after_vdiff
from wombat_transport.transport.driver import _build_vdiff_input_after_tpcore
from wombat_transport.transport.driver import compute_transport_stage_masses
from wombat_transport.transport.driver import trace_transport_one_step
from wombat_transport.transport.forcing import _load_i3_fields, _map_met_edges_to_48, _record_day_and_index
from wombat_transport.transport.pjc import _pjc_horizontal_geometry
from wombat_transport.transport.pbl import ZVIR
from wombat_transport.transport.tpcore import (
    analyze_tpcore_branches,
    run_tpcore_one_step,
    setup_tpcore_terms,
    trace_tpcore_one_step,
)
from wombat_transport.transport.tpcore._reference import _average_poles_in_place, _calc_divergence


PJC_INPUT_VERSION = "pjc-pfix-input-v1"
PJC_OUTPUT_VERSION = "pjc-pfix-output-v1"
PJC_SNAPSHOT_VERSION = "pjc-pfix-snapshot-v1"
TRANSPORT_INPUT_VERSION = "transport-step-input-v2"
TRANSPORT_OUTPUT_VERSION = "transport-step-output-v2"
TRANSPORT_CHAIN_OUTPUT_VERSION = "transport-chain-output-v2"
VDIFF_INPUT_VERSION = "vdiffdr-input-v2"
VDIFF_OUTPUT_VERSION = "vdiffdr-output-v2"
CONVECTION_INPUT_VERSION = "convection-input-v2"
CONVECTION_OUTPUT_VERSION = "convection-output-v2"
DRY_PRESSURE_INPUT_VERSION = "dry-pressure-input-v1"
DRY_PRESSURE_OUTPUT_VERSION = "dry-pressure-output-v1"
DRY_PRESSURE_SNAPSHOT_VERSION = "dry-pressure-snapshot-v1"
MET_AIRQNT_OUTPUT_VERSION = "met-airqnt-output-v1"
MET_AIRQNT_SNAPSHOT_VERSION = "met-airqnt-snapshot-v1"
HISTORY_HARNESS_VERSION = "history-harness-v1"
HISTORY_WOMBAT_OUTPUT_VERSION = "history-wombat-output-v1"
TPCORE_TRACE_VERSION = "tpcore-trace-v2"
TPCORE_SNAPSHOT_VERSION = "tpcore-step-snapshot-v2"
SNAPSHOT_INPUT_NAME = "pjc_input.nc"
SNAPSHOT_OUTPUT_NAME = "pjc_output.nc"
TPCORE_SNAPSHOT_INPUT_NAME = "tpcore_input.nc"
TPCORE_SNAPSHOT_OUTPUT_NAME = "tpcore_output.nc"
VDIFF_SNAPSHOT_INPUT_NAME = "vdiff_input.nc"
VDIFF_SNAPSHOT_OUTPUT_NAME = "vdiff_output.nc"
CONVECTION_SNAPSHOT_INPUT_NAME = "convection_input.nc"
CONVECTION_SNAPSHOT_OUTPUT_NAME = "convection_output.nc"
DRY_PRESSURE_SNAPSHOT_INPUT_NAME = "dry_pressure_input.nc"
DRY_PRESSURE_SNAPSHOT_OUTPUT_NAME = "dry_pressure_output.nc"
MET_AIRQNT_SNAPSHOT_INPUT_NAME = "met_airqnt_input.nc"
MET_AIRQNT_SNAPSHOT_OUTPUT_NAME = "met_airqnt_output.nc"
SNAPSHOT_METADATA_NAME = "metadata.json"
TPCORE_BRANCH_SCENARIOS = ("x_fxppm_low_courant", "x_large_courant_polar")
VDIFF_SCENARIOS = ("zero_surface_flux", "nonzero_surface_flux", "negative_clipping")
CONVECTION_SCENARIOS = ("no_cloud", "active_cloud", "multi_tracer")
REAL_CONVECTION_MODES = ("sampled-columns", "full-grid")
LARGE_ORACLE_MANIFEST_NAME = "manifest.json"
HISTORY_HARNESS_OUTPUT_NAME = "OutputDir/GEOSChem.SpeciesConcThreeHourly.20140901_0000z.nc4"
HISTORY_FIXTURE_SCENARIOS = ("default", "six_hour_groups")
PYTHON_TPCORE_TRACE_NAME = "python_tpcore_trace.nc"
ORACLE_TPCORE_TRACE_NAME = "oracle_tpcore_trace.nc"
BASE_INITIAL_TPCORE_FIXTURE_ID = "base_initial_tpcore_v3"
RESIDUAL_INITIAL_TPCORE_FIXTURE_ID = "residual_initial_tpcore_v3"
FULLGRID_SYNTHETIC_LOW_COURANT_TPCORE_FIXTURE_ID = "fullgrid_synthetic_low_courant_tpcore_v2"
BASE_INITIAL_TRANSPORT_CHAIN_FIXTURE_ID = "base_initial_transport_chain_v3"
BASE_INITIAL_VDIFF_AFTER_TPCORE_FIXTURE_ID = "base_initial_vdiff_after_tpcore_v3"
BASE_INITIAL_CONVECTION_FULLGRID_FIXTURE_ID = "base_initial_convection_fullgrid_v3"
LARGE_ORACLE_FIXTURE_IDS = (
    BASE_INITIAL_TPCORE_FIXTURE_ID,
    RESIDUAL_INITIAL_TPCORE_FIXTURE_ID,
    FULLGRID_SYNTHETIC_LOW_COURANT_TPCORE_FIXTURE_ID,
    BASE_INITIAL_TRANSPORT_CHAIN_FIXTURE_ID,
    BASE_INITIAL_VDIFF_AFTER_TPCORE_FIXTURE_ID,
    BASE_INITIAL_CONVECTION_FULLGRID_FIXTURE_ID,
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


def _fixed_width_strings_to_chars(values: np.ndarray) -> np.ndarray:
    """Return the character-array representation accepted by netCDF4."""
    if values.dtype.kind != "S":
        raise TypeError("fixed-width byte strings are required")
    return values.view("S1").reshape(values.shape + (values.dtype.itemsize,))


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
class TransportChainComparison:
    tracer_max_abs_error: float
    tracer_mean_abs_error: float
    negative_count_expected: int
    negative_count_actual: int
    common_basis_initial_mass_max_abs_error: float
    common_basis_final_mass_max_abs_error: float
    common_basis_mass_change_max_abs_error: float
    common_basis_python_mass_change_max_abs: float
    common_basis_oracle_mass_change_max_abs: float
    common_basis_tpcore_stage_mass_change_max_abs: float
    common_basis_vdiff_stage_mass_change_max_abs: float
    common_basis_convection_stage_mass_change_max_abs: float
    reported_final_mass_max_abs_error: float
    reported_python_mass_change_max_abs: float
    reported_oracle_mass_change_max_abs: float
    reported_tpcore_stage_mass_change_max_abs: float
    reported_vdiff_stage_mass_change_max_abs: float
    reported_convection_stage_mass_change_max_abs: float


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
class ConvectionOutput:
    tracer_conc_after: np.ndarray
    diag14_mass_flux: np.ndarray
    negative_count_before: int
    negative_count_after: int
    initial_tracer_mass: np.ndarray
    final_tracer_mass: np.ndarray
    internal_steps: int
    internal_dt_s: float


@dataclass(frozen=True)
class DryPressureOutput:
    ps1_wet_hpa: np.ndarray
    ps2_wet_hpa: np.ndarray
    ps1_dry_hpa: np.ndarray
    ps2_dry_hpa: np.ndarray
    psc2_wet_hpa: np.ndarray
    psc2_dry_hpa: np.ndarray
    delp_dry_hpa: np.ndarray
    specific_humidity_kg_kg: np.ndarray
    temperature_k: np.ndarray


@dataclass(frozen=True)
class MetAirQntOutput:
    ps1_wet_hpa: np.ndarray
    ps2_wet_hpa: np.ndarray
    ps1_dry_hpa: np.ndarray
    ps2_dry_hpa: np.ndarray
    psc2_wet_hpa: np.ndarray
    psc2_dry_hpa: np.ndarray
    wet_pressure_edges_hpa: np.ndarray
    wet_pressure_mid_hpa: np.ndarray
    wet_pressure_thickness_hpa: np.ndarray
    dry_partial_pressure_edges_hpa: np.ndarray
    dry_partial_pressure_mid_hpa: np.ndarray
    delp_dry_hpa: np.ndarray
    specific_humidity_kg_kg: np.ndarray
    temperature_k: np.ndarray
    water_vapor_vv_dry: np.ndarray
    virtual_temperature_k: np.ndarray
    bxheight_m: np.ndarray
    dry_air_mass_kg: np.ndarray
    air_volume_m3: np.ndarray


@dataclass(frozen=True)
class DryPressureComparison:
    ps1_wet_max_abs_error_hpa: float
    ps2_wet_max_abs_error_hpa: float
    ps1_dry_max_abs_error_hpa: float
    ps2_dry_max_abs_error_hpa: float
    psc2_wet_max_abs_error_hpa: float
    psc2_dry_max_abs_error_hpa: float
    delp_dry_max_abs_error_hpa: float
    delp_dry_mean_abs_error_hpa: float
    specific_humidity_max_abs_error: float
    temperature_max_abs_error: float


@dataclass(frozen=True)
class MetAirQntComparison:
    max_abs_errors: dict[str, float]
    mean_abs_errors: dict[str, float]


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
    common_basis_initial_mass_max_abs_error: float
    common_basis_final_mass_max_abs_error: float
    common_basis_mass_change_max_abs_error: float
    reported_initial_mass_max_abs_error: float
    reported_final_mass_max_abs_error: float


@dataclass(frozen=True)
class ConvectionComparison:
    tracer_max_abs_error: float
    tracer_mean_abs_error: float
    diag14_max_abs_error: float
    diag14_mean_abs_error: float
    negative_count_before_expected: int
    negative_count_before_actual: int
    negative_count_after_expected: int
    negative_count_after_actual: int
    common_basis_initial_mass_max_abs_error: float
    common_basis_final_mass_max_abs_error: float
    common_basis_mass_change_max_abs_error: float
    common_basis_python_mass_change_max_abs: float
    common_basis_oracle_mass_change_max_abs: float
    reported_initial_mass_max_abs_error: float
    reported_final_mass_max_abs_error: float
    reported_python_mass_change_max_abs: float
    reported_oracle_mass_change_max_abs: float
    top_error_tracer: int
    top_error_level: int
    top_error_lat: int
    top_error_lon: int
    internal_steps_expected: int
    internal_steps_actual: int


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


@dataclass(frozen=True)
class HistoryHarnessComparison:
    output_path: Path
    n_records: int
    n_tracers: int
    max_abs_error: float
    max_time_error_min: float
    first_record_expected: float
    first_record_actual: float
    boundary_included_in_previous: bool


@dataclass(frozen=True)
class HistoryHarnessWombatComparison:
    reference_path: Path
    wombat_path: Path
    n_records: int
    n_tracers: int
    max_abs_error: float
    max_time_error_min: float
    max_coord_error: float


def write_pjc_input_from_config(
    run_config_path: str | Path,
    output_path: str | Path,
    *,
    time_index: int = 0,
    dt_s: float | None = None,
) -> Path:
    config = load_run_config(run_config_path)
    if dt_s is None:
        dt_s = transport_timestep_s(config)
    grid = load_transport_grid(config.grid_template)
    forcing = load_transport_forcing(
        meteorology_root(config),
        simulation_start(config),
        grid,
        time_index=time_index,
    )
    p1_hpa = forcing.dry_surface_pressure_start_hpa[0]
    p2_hpa = forcing.dry_surface_pressure_hpa[0]
    return write_pjc_input(
        output_path,
        lat_deg=forcing.lat_deg,
        lon_deg=forcing.lon_deg,
        area_m2=grid.area_m2,
        hyai_hpa=grid.hyai_hpa,
        hybi=grid.hybi,
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
    tracer_data = canonical_time_slice(tracers.to_canonical(), tracer_time_index)
    tracer_names = tracers.names
    if max_tracers is not None:
        tracer_data = tracer_data[..., :max_tracers]
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
        dt_s = transport_timestep_s(config)
    grid = load_transport_grid(config.grid_template)
    lat = grid.lat_deg
    lon = grid.lon_deg
    hyai = grid.hyai_hpa
    hybi = grid.hybi
    area = grid.area_m2

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

    tracer_index = np.arange(ntracer, dtype=np.float64)[np.newaxis, np.newaxis, np.newaxis, :]
    lev_index = np.arange(nlev, dtype=np.float64)[:, np.newaxis, np.newaxis, np.newaxis]
    lat_wave = np.sin(lat_rad)[np.newaxis, :, np.newaxis, np.newaxis]
    lon_wave = np.cos(lon_rad)[np.newaxis, np.newaxis, :, np.newaxis]
    tracer = 4.0e-4
    tracer = tracer + (tracer_index + 1.0) * 1.0e-7
    tracer = tracer + 2.5e-8 * lev_index / max(float(nlev - 1), 1.0)
    tracer = tracer + 1.5e-8 * lat_wave + 7.5e-9 * lon_wave
    names = tuple(f"fullgrid_synthetic_{index + 1:02d}" for index in range(ntracer))
    return append_transport_step_tracers(path, tracer, tracer_names=names)


def write_dry_pressure_input_from_config(
    run_config_path: str | Path,
    output_path: str | Path,
    *,
    step_index: int = 0,
    dt_s: float | None = None,
) -> Path:
    """Write raw I3 endpoint fields for the GEOS-Chem dry-pressure harness."""

    config = load_run_config(run_config_path)
    fixture_dt_s = transport_timestep_s(config) if dt_s is None else float(dt_s)
    if fixture_dt_s <= 0:
        raise ValueError("dt_s must be positive")
    if step_index < 0:
        raise ValueError("step_index must be nonnegative")

    grid = load_transport_grid(config.grid_template)
    start = simulation_start(config)
    start_day = datetime(start.year, start.month, start.day)
    met_root = meteorology_root(config)
    elapsed_s = float(step_index) * fixture_dt_s
    initial_i3 = meteorology_initial_time_index(config)
    i3_start_index = initial_i3 + int(elapsed_s // 10800.0)
    i3_end_index = i3_start_index + 1
    seconds_into_i3_window = int(elapsed_s % 10800.0)
    i3_start_day, i3_start_time_index = _record_day_and_index(start_day, i3_start_index, 8)
    i3_end_day, i3_end_time_index = _record_day_and_index(start_day, i3_end_index, 8)
    i3_start = _load_i3_fields(met_root, i3_start_day, grid, i3_start_time_index, None)
    i3_end = _load_i3_fields(met_root, i3_end_day, grid, i3_end_time_index, None)
    return write_dry_pressure_input(
        output_path,
        lat_deg=grid.lat_deg,
        lon_deg=grid.lon_deg,
        area_m2=grid.area_m2,
        hyai_hpa=grid.hyai_hpa,
        hybi=grid.hybi,
        ps1_wet_hpa=i3_start.surface_pressure[0] / 100.0,
        ps2_wet_hpa=i3_end.surface_pressure[0] / 100.0,
        sphu1_kg_kg=i3_start.qv[0],
        sphu2_kg_kg=i3_end.qv[0],
        tmpu1_k=i3_start.temperature[0],
        tmpu2_k=i3_end.temperature[0],
        ntime0_s=0,
        ntime1_s=seconds_into_i3_window,
        ntdt_s=int(fixture_dt_s),
    )


def write_synthetic_dry_pressure_input(path: str | Path, *, ntime1_s: int = 3600, ntdt_s: int = 600) -> Path:
    """Write a compact deterministic dry-pressure oracle input."""

    lat = np.array([-89.5, -60.0, -30.0, 0.0, 30.0, 60.0, 89.5], dtype=np.float64)
    lon = np.arange(8, dtype=np.float64) * 45.0
    area = _spherical_band_area(lat, lon.size)
    nlev = GEOS_47_AP_HPA.size - 1
    lev = np.arange(nlev, dtype=np.float64)[:, np.newaxis, np.newaxis]
    lat_term = np.sin(np.deg2rad(lat))[np.newaxis, :, np.newaxis]
    lon_term = np.cos(np.deg2rad(lon))[np.newaxis, np.newaxis, :]
    lat_2d = lat_term[0]
    lon_2d = lon_term[0]

    ps1 = 970.0 + 18.0 * np.cos(np.deg2rad(lat))[:, np.newaxis] ** 2 + 1.7 * lon_2d
    ps2 = ps1 + 1.4 + 0.8 * np.sin(np.deg2rad(lon))[np.newaxis, :] - 0.35 * lat_2d
    q1 = 0.012 * np.exp(-lev / 17.0) * (1.0 + 0.04 * lat_term + 0.01 * lon_term)
    q2 = q1 * 0.985 + 1.0e-5 * np.cos((lev + 1.0) / nlev * np.pi)
    tmp1 = 289.0 - 0.55 * lev + 1.1 * lat_term + 0.2 * lon_term
    tmp2 = tmp1 + 0.4 - 0.03 * lat_term
    return write_dry_pressure_input(
        path,
        lat_deg=lat,
        lon_deg=lon,
        area_m2=area,
        hyai_hpa=GEOS_47_AP_HPA,
        hybi=GEOS_47_BP,
        ps1_wet_hpa=ps1,
        ps2_wet_hpa=ps2,
        sphu1_kg_kg=q1,
        sphu2_kg_kg=q2,
        tmpu1_k=tmp1,
        tmpu2_k=tmp2,
        ntime0_s=0,
        ntime1_s=ntime1_s,
        ntdt_s=ntdt_s,
    )


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

    lev_index = np.arange(nlev, dtype=np.float64)[:, np.newaxis, np.newaxis, np.newaxis]
    lat_wave = np.sin(np.deg2rad(lat))[np.newaxis, :, np.newaxis, np.newaxis]
    lon_wave = np.cos(np.deg2rad(lon))[np.newaxis, np.newaxis, :, np.newaxis]
    tracer_index = np.arange(ntracer, dtype=np.float64)[np.newaxis, np.newaxis, np.newaxis, :]
    tracer = 4.0e-4
    tracer = tracer + (tracer_index + 1.0) * 1.0e-7
    tracer = tracer + 2.5e-8 * lev_index / max(float(nlev - 1), 1.0)
    tracer = tracer + 1.0e-8 * lat_wave + 5.0e-9 * lon_wave
    names = tuple(f"synthetic_{index + 1:02d}" for index in range(ntracer))
    return append_transport_step_tracers(path, tracer, tracer_names=names)


def write_synthetic_vdiff_input(
    path: str | Path,
    *,
    dt_s: float = 600.0,
    ntracer: int = 2,
    scenario: str = "zero_surface_flux",
) -> Path:
    """Write a compact deterministic 47-level VDIFFDR oracle input."""

    if ntracer <= 0:
        raise ValueError("ntracer must be positive")
    if scenario not in VDIFF_SCENARIOS:
        raise ValueError(f"unknown VDIFF scenario {scenario!r}; expected one of {VDIFF_SCENARIOS}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lat = np.array([-45.0, 0.0, 45.0], dtype=np.float64)
    lon = np.arange(4, dtype=np.float64) * 90.0
    nlev = GEOS_47_AP_HPA.size - 1
    lev = np.arange(nlev, dtype=np.float64)[:, np.newaxis, np.newaxis]
    lat_term = np.sin(np.deg2rad(lat))[np.newaxis, :, np.newaxis]
    lon_term = np.cos(np.deg2rad(lon))[np.newaxis, np.newaxis, :]
    tracer_index = np.arange(ntracer, dtype=np.float64)[np.newaxis, np.newaxis, np.newaxis, :]

    pedge_profile = np.linspace(50.0, 1000.0, nlev + 1, dtype=np.float64)
    pedge = np.broadcast_to(pedge_profile[:, np.newaxis, np.newaxis], (nlev + 1, lat.size, lon.size)).copy()
    pmid = 0.5 * (pedge[:-1] + pedge[1:])
    temperature = 289.0 - 0.45 * lev + 1.5 * lat_term + 0.2 * lon_term
    sphu = 0.010 * np.exp(-lev / 18.0) * (1.0 + 0.03 * lat_term) * np.ones((1, 1, lon.size))
    tv = temperature * (1.0 + ZVIR * sphu)
    bxheight = np.full((nlev, lat.size, lon.size), 125.0, dtype=np.float64)
    dry_mass = (pedge[1:] - pedge[:-1]) * 100.0 / G0_M_PER_S2
    u = (4.0 + 0.05 * lev + 0.2 * lon_term) * np.ones((1, lat.size, 1), dtype=np.float64)
    v = (0.3 * np.sin((lev + 1.0) / nlev * np.pi) + 0.02 * lat_term) * np.ones((1, 1, lon.size))
    tracer = 4.0e-4 + 1.0e-7 * tracer_index + 4.0e-9 * lev[..., np.newaxis]
    tracer = tracer + 2.0e-9 * lat_term[..., np.newaxis] + 1.0e-9 * lon_term[..., np.newaxis]
    surface_flux = np.zeros((lat.size, lon.size, ntracer), dtype=np.float64)
    if scenario == "nonzero_surface_flux":
        lat_scale = 1.0 + 0.15 * np.arange(lat.size, dtype=np.float64)[:, np.newaxis, np.newaxis]
        lon_scale = 1.0 + 0.05 * np.arange(lon.size, dtype=np.float64)[np.newaxis, :, np.newaxis]
        tracer_scale = np.arange(1, ntracer + 1, dtype=np.float64)[np.newaxis, np.newaxis, :]
        surface_flux = 1.0e-12 * tracer_scale * lat_scale * lon_scale
    elif scenario == "negative_clipping":
        tracer[0, 1, 2, 0] = -1.0e-3
        tracer[1, 2, 3, min(1, ntracer - 1)] = -5.0e-4

    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("tracer", ntracer)
        dataset.createDimension("lev", nlev)
        dataset.createDimension("ilev", nlev + 1)
        dataset.createDimension("lat", lat.size)
        dataset.createDimension("lon", lon.size)
        dataset.harness = VDIFF_INPUT_VERSION
        dataset.dt_s = float(dt_s)
        dataset.scenario = scenario
        dataset.createVariable("lon", "f8", ("lon",))[:] = lon
        dataset.createVariable("lat", "f8", ("lat",))[:] = lat
        dataset.createVariable("tracer_conc", "f8", ("lev", "lat", "lon", "tracer"))[:] = tracer
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
        dataset.createVariable("surface_flux_kg_m2_s", "f8", ("lat", "lon", "tracer"))[:] = surface_flux
    return path


def write_synthetic_convection_input(
    path: str | Path,
    *,
    dt_s: float = 600.0,
    ntracer: int = 2,
    scenario: str = "active_cloud",
) -> Path:
    """Write a compact deterministic 47-level cloud-convection oracle input."""

    if scenario not in CONVECTION_SCENARIOS:
        raise ValueError(f"unknown convection scenario {scenario!r}; expected one of {CONVECTION_SCENARIOS}")
    if scenario == "multi_tracer":
        ntracer = max(ntracer, 3)
    if ntracer <= 0:
        raise ValueError("ntracer must be positive")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lat = np.array([-30.0, 10.0], dtype=np.float64)
    lon = np.arange(3, dtype=np.float64) * 120.0
    nlev = GEOS_47_AP_HPA.size - 1
    lev = np.arange(nlev, dtype=np.float64)[:, np.newaxis, np.newaxis]
    lat_term = np.sin(np.deg2rad(lat))[np.newaxis, :, np.newaxis]
    lon_term = np.cos(np.deg2rad(lon))[np.newaxis, np.newaxis, :]
    tracer_index = np.arange(ntracer, dtype=np.float64)[np.newaxis, np.newaxis, np.newaxis, :]

    delp_profile = np.linspace(62.0, 8.0, nlev, dtype=np.float64)
    delp_dry = np.broadcast_to(delp_profile[:, np.newaxis, np.newaxis], (nlev, lat.size, lon.size)).copy()
    delp_dry *= 1.0 + 0.01 * lat_term + 0.005 * lon_term
    delp = delp_dry * 1.01
    tracer = 4.0e-4 + 2.0e-7 * tracer_index
    tracer = tracer + 3.0e-9 * lev[..., np.newaxis]
    tracer = tracer + 2.0e-9 * lat_term[..., np.newaxis] + 1.0e-9 * lon_term[..., np.newaxis]
    area = _spherical_band_area(lat, lon.size)
    bxheight = 150.0 + 3.0 * lev + np.zeros((1, lat.size, lon.size), dtype=np.float64)
    temperature = 288.0 - 0.55 * lev + 0.8 * lat_term + 0.1 * lon_term
    cmfmc = np.zeros((nlev, lat.size, lon.size), dtype=np.float64)
    dtrain = np.zeros_like(cmfmc)
    dqrcu = np.zeros_like(cmfmc)
    reevapcn = np.zeros_like(cmfmc)
    pficu = np.zeros_like(cmfmc)
    pflcu = np.zeros_like(cmfmc)
    precccon = np.zeros((lat.size, lon.size), dtype=np.float64)

    if scenario in ("active_cloud", "multi_tracer"):
        plume = np.zeros(nlev, dtype=np.float64)
        plume[3:13] = np.linspace(0.010, 0.002, 10)
        cmfmc[:] = plume[:, np.newaxis, np.newaxis]
        cmfmc *= 1.0 + 0.08 * np.arange(lat.size, dtype=np.float64)[np.newaxis, :, np.newaxis]
        cmfmc *= 1.0 + 0.03 * np.arange(lon.size, dtype=np.float64)[np.newaxis, np.newaxis, :]
        dtrain[4:13] = 0.0015
        dqrcu[4:13] = 1.0e-8
        precccon[:] = 2.5

    names = tuple(f"conv_{index + 1:03d}" for index in range(ntracer))
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("tracer", ntracer)
        dataset.createDimension("lev", nlev)
        dataset.createDimension("lat", lat.size)
        dataset.createDimension("lon", lon.size)
        name_length = max(max((len(name) for name in names), default=1), 1)
        dataset.createDimension("name_strlen", name_length)
        dataset.harness = CONVECTION_INPUT_VERSION
        dataset.dt_s = float(dt_s)
        dataset.scenario = scenario
        dataset.reconstruct_conv_precip_flux = 0
        dataset.createVariable("lon", "f8", ("lon",))[:] = lon
        dataset.createVariable("lat", "f8", ("lat",))[:] = lat
        dataset.createVariable("tracer_conc", "f8", ("lev", "lat", "lon", "tracer"))[:] = tracer
        name_var = dataset.createVariable("tracer_name", "S1", ("tracer", "name_strlen"))
        encoded = np.asarray([name.encode("ascii", errors="replace") for name in names], dtype=f"S{name_length}")
        name_var[:] = _fixed_width_strings_to_chars(encoded)
        dataset.createVariable("cmfmc_kg_m2_s", "f8", ("lev", "lat", "lon"))[:] = cmfmc
        dataset.createVariable("dtrain_kg_m2_s", "f8", ("lev", "lat", "lon"))[:] = dtrain
        dataset.createVariable("dqrcu_kg_kg_s", "f8", ("lev", "lat", "lon"))[:] = dqrcu
        dataset.createVariable("reevapcn_kg_kg_s", "f8", ("lev", "lat", "lon"))[:] = reevapcn
        dataset.createVariable("delp_dry_hpa", "f8", ("lev", "lat", "lon"))[:] = delp_dry
        dataset.createVariable("delp_hpa", "f8", ("lev", "lat", "lon"))[:] = delp
        dataset.createVariable("area_m2", "f8", ("lat", "lon"))[:] = area
        dataset.createVariable("bxheight_m", "f8", ("lev", "lat", "lon"))[:] = bxheight
        dataset.createVariable("pficu_kg_m2_s", "f8", ("lev", "lat", "lon"))[:] = pficu
        dataset.createVariable("pflcu_kg_m2_s", "f8", ("lev", "lat", "lon"))[:] = pflcu
        dataset.createVariable("temperature_k", "f8", ("lev", "lat", "lon"))[:] = temperature
        dataset.createVariable("precccon_mm_day", "f8", ("lat", "lon"))[:] = precccon
    return path


def write_real_convection_input_from_config(
    run_config_path: str | Path,
    output_path: str | Path,
    *,
    mode: str = "sampled-columns",
    time_index: int | None = None,
    tracer_time_index: int = 0,
    dt_s: float | None = None,
    max_tracers: int | None = None,
    active_columns: int = 6,
) -> Path:
    """Write a 47-level convection fixture from local MERRA2 and tracer config."""

    if mode not in REAL_CONVECTION_MODES:
        raise ValueError(f"unknown real convection mode {mode!r}; expected one of {REAL_CONVECTION_MODES}")
    if active_columns <= 0:
        raise ValueError("active_columns must be positive")

    config = load_run_config(run_config_path)
    fixture_dt_s = transport_timestep_s(config) if dt_s is None else float(dt_s)
    fixture_time_index = meteorology_initial_time_index(config) if time_index is None else int(time_index)
    start = simulation_start(config)
    met_root = meteorology_root(config)

    tracers = initialize_tracers(
        config.initial_restart,
        config.species_database,
        template_path=config.grid_template,
    )
    tracer_data = canonical_time_slice(tracers.to_canonical(), tracer_time_index)
    tracer_names = tracers.names
    if max_tracers is not None:
        tracer_data = tracer_data[..., :max_tracers]
        tracer_names = tracer_names[:max_tracers]

    grid = load_transport_grid(config.grid_template)
    real_met = _load_real_convection_met(
        met_root,
        start,
        grid,
        time_index=fixture_time_index,
    )
    if mode == "sampled-columns":
        rows, cols = _select_real_convection_columns(real_met["cmfmc_kg_m2_s"], active_columns=active_columns)
        packed = _pack_convection_columns(
            tracer_data=tracer_data,
            met=real_met,
            rows=rows,
            cols=cols,
        )
        fixture_mode = "sampled-columns"
    else:
        nlat, nlon = real_met["area_m2"].shape
        rows = np.repeat(np.arange(nlat, dtype=np.int64), nlon)
        cols = np.tile(np.arange(nlon, dtype=np.int64), nlat)
        packed = {
            "tracer_conc": tracer_data,
            **real_met,
            "source_lat_index": rows.reshape(nlat, nlon),
            "source_lon_index": cols.reshape(nlat, nlon),
        }
        fixture_mode = "full-grid"

    return _write_convection_input_file(
        output_path,
        tracer_conc=packed["tracer_conc"],
        tracer_names=tracer_names,
        lon=np.asarray(packed["lon"], dtype=np.float64),
        lat=np.asarray(packed["lat"], dtype=np.float64),
        cmfmc=np.asarray(packed["cmfmc_kg_m2_s"], dtype=np.float64),
        dtrain=np.asarray(packed["dtrain_kg_m2_s"], dtype=np.float64),
        dqrcu=np.asarray(packed["dqrcu_kg_kg_s"], dtype=np.float64),
        reevapcn=np.asarray(packed["reevapcn_kg_kg_s"], dtype=np.float64),
        delp_dry=np.asarray(packed["delp_dry_hpa"], dtype=np.float64),
        delp=np.asarray(packed["delp_hpa"], dtype=np.float64),
        area=np.asarray(packed["area_m2"], dtype=np.float64),
        bxheight=np.asarray(packed["bxheight_m"], dtype=np.float64),
        pficu=np.asarray(packed["pficu_kg_m2_s"], dtype=np.float64),
        pflcu=np.asarray(packed["pflcu_kg_m2_s"], dtype=np.float64),
        temperature=np.asarray(packed["temperature_k"], dtype=np.float64),
        precccon=np.asarray(packed["precccon_mm_day"], dtype=np.float64),
        dt_s=fixture_dt_s,
        scenario=f"real-{fixture_mode}",
        source_lat_index=np.asarray(packed["source_lat_index"], dtype=np.int32),
        source_lon_index=np.asarray(packed["source_lon_index"], dtype=np.int32),
        source_run_config=str(Path(run_config_path)),
        source_timestamp=start.strftime("%Y-%m-%d %H:%M"),
        source_time_index=fixture_time_index,
        vertical_mapping="native_72_to_47_center_and_73_to_48_edge",
    )


def _load_real_convection_met(
    met_root: str | Path,
    timestamp: datetime,
    grid: TransportGrid,
    *,
    time_index: int,
) -> dict[str, np.ndarray]:
    met_root = Path(met_root)
    day_dir = met_root / f"{timestamp.year:04d}" / f"{timestamp.month:02d}"
    a1_path = day_dir / merra2_filename(timestamp, "A1", grid)
    a3dyn_path = day_dir / merra2_filename(timestamp, "A3dyn", grid)
    a3mstc_path = day_dir / merra2_filename(timestamp, "A3mstC", grid)
    a3mste_path = day_dir / merra2_filename(timestamp, "A3mstE", grid)
    i3_path = day_dir / merra2_filename(timestamp, "I3", grid)
    a1_time_index = int(time_index) * 3

    with (
        netCDF4.Dataset(a1_path) as a1,
        netCDF4.Dataset(a3dyn_path) as a3dyn,
        netCDF4.Dataset(a3mstc_path) as a3mstc,
        netCDF4.Dataset(a3mste_path) as a3mste,
        netCDF4.Dataset(i3_path) as i3,
    ):
        lat = grid.lat_deg
        lon = grid.lon_deg
        area = grid.area_m2
        hyai = grid.hyai_hpa
        hybi = grid.hybi
        ps = np.asarray(i3.variables["PS"][time_index : time_index + 1], dtype=np.float64)
        sphu = _map_met_levels_to_47(_read_met_3d_time_slice(i3, "QV", time_index))[0]
        center = {
            "dtrain_kg_m2_s": _map_met_levels_to_47(_read_met_3d_time_slice(a3dyn, "DTRAIN", time_index))[0],
            "dqrcu_kg_kg_s": _map_met_levels_to_47(_read_met_3d_time_slice(a3mstc, "DQRCU", time_index))[0],
            "reevapcn_kg_kg_s": _map_met_levels_to_47(_read_met_3d_time_slice(a3mstc, "REEVAPCN", time_index))[0],
            "temperature_k": _map_met_levels_to_47(_read_met_3d_time_slice(i3, "T", time_index))[0],
            "specific_humidity_kg_kg": sphu,
        }
        cmfmc_edges = _map_met_edges_to_48(np.asarray(a3mste.variables["CMFMC"][time_index], dtype=np.float64))
        pficu_edges = _map_met_edges_to_48(np.asarray(a3mste.variables["PFICU"][time_index], dtype=np.float64))
        pflcu_edges = _map_met_edges_to_48(np.asarray(a3mste.variables["PFLCU"][time_index], dtype=np.float64))
        delp = np.abs(
            (hyai[:-1, np.newaxis, np.newaxis] + hybi[:-1, np.newaxis, np.newaxis] * ps[0] / 100.0)
            - (hyai[1:, np.newaxis, np.newaxis] + hybi[1:, np.newaxis, np.newaxis] * ps[0] / 100.0)
        )
        temperature = center["temperature_k"]
        pedge = GEOS_47_AP_HPA[:, np.newaxis, np.newaxis] + GEOS_47_BP[:, np.newaxis, np.newaxis] * ps[0] / 100.0
        bxheight = _hydrostatic_box_height_from_temperature(pedge, temperature * (1.0 + ZVIR * sphu))
        precccon = np.asarray(a1.variables["PRECCON"][a1_time_index], dtype=np.float64) * 86400.0

    return {
        "lat": lat,
        "lon": lon,
        "area_m2": area,
        "cmfmc_kg_m2_s": cmfmc_edges[1:],
        "pficu_kg_m2_s": pficu_edges[1:],
        "pflcu_kg_m2_s": pflcu_edges[1:],
        "delp_dry_hpa": delp,
        "delp_hpa": delp.copy(),
        "bxheight_m": bxheight,
        "precccon_mm_day": precccon,
        **center,
    }


def _read_met_3d_time_slice(dataset: netCDF4.Dataset, variable_name: str, time_index: int) -> np.ndarray:
    return np.asarray(dataset.variables[variable_name][time_index : time_index + 1], dtype=np.float64)


def _hydrostatic_box_height_from_temperature(pedge_hpa: np.ndarray, temperature_k: np.ndarray) -> np.ndarray:
    return (RD_J_PER_KG_K / G0_M_PER_S2) * temperature_k * np.log(pedge_hpa[:-1] / pedge_hpa[1:])


def _select_real_convection_columns(cmfmc_upper: np.ndarray, *, active_columns: int) -> tuple[np.ndarray, np.ndarray]:
    column_max = np.max(np.abs(cmfmc_upper), axis=0)
    flat_order = np.argsort(column_max.ravel())[::-1]
    active = flat_order[:active_columns]
    zero_candidates = np.flatnonzero(column_max.ravel() == 0.0)
    if zero_candidates.size:
        chosen = np.concatenate([active, zero_candidates[:1]])
    else:
        chosen = np.concatenate([active, flat_order[-1:]])
    rows, cols = np.unravel_index(chosen, column_max.shape)
    return rows.astype(np.int64), cols.astype(np.int64)


def _pack_convection_columns(
    *,
    tracer_data: np.ndarray,
    met: dict[str, np.ndarray],
    rows: np.ndarray,
    cols: np.ndarray,
) -> dict[str, np.ndarray]:
    ncol = rows.size
    packed: dict[str, np.ndarray] = {
        "lat": met["lat"][rows],
        "lon": np.array([0.0], dtype=np.float64),
        "source_lat_index": rows[:, np.newaxis].astype(np.int32),
        "source_lon_index": cols[:, np.newaxis].astype(np.int32),
        "tracer_conc": tracer_data[:, rows, cols, :][:, :, np.newaxis, :],
    }
    for name in (
        "cmfmc_kg_m2_s",
        "dtrain_kg_m2_s",
        "dqrcu_kg_kg_s",
        "reevapcn_kg_kg_s",
        "delp_dry_hpa",
        "delp_hpa",
        "bxheight_m",
        "pficu_kg_m2_s",
        "pflcu_kg_m2_s",
        "temperature_k",
    ):
        packed[name] = met[name][:, rows, cols][:, :, np.newaxis]
    for name in ("area_m2", "precccon_mm_day"):
        packed[name] = met[name][rows, cols][:, np.newaxis]
    if packed["area_m2"].shape != (ncol, 1):
        raise AssertionError("packed column area has unexpected shape")
    return packed


def _write_convection_input_file(
    path: str | Path,
    *,
    tracer_conc: np.ndarray,
    tracer_names: tuple[str, ...],
    lon: np.ndarray,
    lat: np.ndarray,
    cmfmc: np.ndarray,
    dtrain: np.ndarray,
    dqrcu: np.ndarray,
    reevapcn: np.ndarray,
    delp_dry: np.ndarray,
    delp: np.ndarray,
    area: np.ndarray,
    bxheight: np.ndarray,
    pficu: np.ndarray,
    pflcu: np.ndarray,
    temperature: np.ndarray,
    precccon: np.ndarray,
    dt_s: float,
    scenario: str,
    source_lat_index: np.ndarray | None = None,
    source_lon_index: np.ndarray | None = None,
    source_run_config: str | None = None,
    source_timestamp: str | None = None,
    source_time_index: int | None = None,
    vertical_mapping: str | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tracers = np.asarray(tracer_conc, dtype=np.float64)
    nlev, nlat, nlon, ntracer = tracers.shape
    if len(tracer_names) != ntracer:
        raise ValueError("tracer_names length must match tracer_conc last dimension")
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("tracer", ntracer)
        dataset.createDimension("lev", nlev)
        dataset.createDimension("lat", nlat)
        dataset.createDimension("lon", nlon)
        name_length = max(max((len(name) for name in tracer_names), default=1), 1)
        dataset.createDimension("name_strlen", name_length)
        dataset.harness = CONVECTION_INPUT_VERSION
        dataset.dt_s = float(dt_s)
        dataset.scenario = scenario
        dataset.reconstruct_conv_precip_flux = 0
        if source_run_config is not None:
            dataset.source_run_config = source_run_config
        if source_timestamp is not None:
            dataset.source_timestamp = source_timestamp
        if source_time_index is not None:
            dataset.source_time_index = int(source_time_index)
        if vertical_mapping is not None:
            dataset.vertical_mapping = vertical_mapping
        dataset.createVariable("lon", "f8", ("lon",))[:] = lon
        dataset.createVariable("lat", "f8", ("lat",))[:] = lat
        dataset.createVariable("tracer_conc", "f8", ("lev", "lat", "lon", "tracer"))[:] = tracers
        name_var = dataset.createVariable("tracer_name", "S1", ("tracer", "name_strlen"))
        encoded = np.asarray([name.encode("ascii", errors="replace") for name in tracer_names], dtype=f"S{name_length}")
        name_var[:] = _fixed_width_strings_to_chars(encoded)
        dataset.createVariable("cmfmc_kg_m2_s", "f8", ("lev", "lat", "lon"))[:] = cmfmc
        dataset.createVariable("dtrain_kg_m2_s", "f8", ("lev", "lat", "lon"))[:] = dtrain
        dataset.createVariable("dqrcu_kg_kg_s", "f8", ("lev", "lat", "lon"))[:] = dqrcu
        dataset.createVariable("reevapcn_kg_kg_s", "f8", ("lev", "lat", "lon"))[:] = reevapcn
        dataset.createVariable("delp_dry_hpa", "f8", ("lev", "lat", "lon"))[:] = delp_dry
        dataset.createVariable("delp_hpa", "f8", ("lev", "lat", "lon"))[:] = delp
        dataset.createVariable("area_m2", "f8", ("lat", "lon"))[:] = area
        dataset.createVariable("bxheight_m", "f8", ("lev", "lat", "lon"))[:] = bxheight
        dataset.createVariable("pficu_kg_m2_s", "f8", ("lev", "lat", "lon"))[:] = pficu
        dataset.createVariable("pflcu_kg_m2_s", "f8", ("lev", "lat", "lon"))[:] = pflcu
        dataset.createVariable("temperature_k", "f8", ("lev", "lat", "lon"))[:] = temperature
        dataset.createVariable("precccon_mm_day", "f8", ("lat", "lon"))[:] = precccon
        if source_lat_index is not None:
            dataset.createVariable("source_lat_index", "i4", ("lat", "lon"))[:] = source_lat_index
        if source_lon_index is not None:
            dataset.createVariable("source_lon_index", "i4", ("lat", "lon"))[:] = source_lon_index
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
    lev_index = np.arange(GEOS_47_AP_HPA.size - 1, dtype=np.float64)[:, np.newaxis, np.newaxis, np.newaxis]
    lat_wave = np.sin(np.deg2rad(lat))[np.newaxis, :, np.newaxis, np.newaxis]
    lon_wave = np.cos(np.deg2rad(lon))[np.newaxis, np.newaxis, :, np.newaxis]
    tracer_index = np.arange(ntracer, dtype=np.float64)[np.newaxis, np.newaxis, np.newaxis, :]
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
    input_name, output_name = _large_oracle_payload_names(fixture_id)
    return LargeOracleFixturePaths(
        fixture_id=fixture_id,
        directory=directory,
        input_path=directory / input_name,
        output_path=directory / output_name,
        manifest_path=directory / LARGE_ORACLE_MANIFEST_NAME,
        definition_path=definitions / f"{fixture_id}.json",
    )


def _large_oracle_payload_names(fixture_id: str) -> tuple[str, str]:
    if fixture_id == BASE_INITIAL_TRANSPORT_CHAIN_FIXTURE_ID:
        return "transport_chain_input.nc", "transport_chain_output.nc"
    if fixture_id == BASE_INITIAL_VDIFF_AFTER_TPCORE_FIXTURE_ID:
        return "vdiff_input.nc", "vdiff_output.nc"
    if fixture_id == BASE_INITIAL_CONVECTION_FULLGRID_FIXTURE_ID:
        return "convection_input.nc", "convection_output.nc"
    return "transport_step_input.nc", "transport_step_output.nc"


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
    run_config_path = Path(run_config or source.get("run_config", "validation_runs/cases/realistic_restart_noemis_2x25/wombat/main/run.yml"))
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
    elif fixture_id == BASE_INITIAL_TRANSPORT_CHAIN_FIXTURE_ID:
        return generate_transport_chain_oracle_fixture(
            paths,
            definition=definition,
            run_config=run_config_path,
            tpcore_executable=Path(executable),
            vdiff_executable=Path("tools/gc_harness/build/vdiff_harness"),
            convection_executable=Path("tools/gc_harness/build/convection_harness"),
            time_index=int(source.get("time_index", 0) if time_index is None else time_index),
            tracer_time_index=int(source.get("tracer_time_index", 0) if tracer_time_index is None else tracer_time_index),
            max_tracers=int(source.get("max_tracers", 1) if max_tracers is None else max_tracers),
            dt_s=fixture_dt_s,
            repo_root=Path(repo_root),
        )
    elif fixture_id == BASE_INITIAL_VDIFF_AFTER_TPCORE_FIXTURE_ID:
        return generate_vdiff_after_tpcore_oracle_fixture(
            paths,
            definition=definition,
            run_config=run_config_path,
            tpcore_executable=Path(executable),
            vdiff_executable=Path("tools/gc_harness/build/vdiff_harness"),
            time_index=int(source.get("time_index", 0) if time_index is None else time_index),
            tracer_time_index=int(source.get("tracer_time_index", 0) if tracer_time_index is None else tracer_time_index),
            max_tracers=int(source.get("max_tracers", 1) if max_tracers is None else max_tracers),
            dt_s=fixture_dt_s,
            repo_root=Path(repo_root),
        )
    elif fixture_id == BASE_INITIAL_CONVECTION_FULLGRID_FIXTURE_ID:
        return generate_convection_fullgrid_oracle_fixture(
            paths,
            definition=definition,
            run_config=run_config_path,
            tpcore_executable=Path(executable),
            vdiff_executable=Path("tools/gc_harness/build/vdiff_harness"),
            convection_executable=Path("tools/gc_harness/build/convection_harness"),
            time_index=int(source.get("time_index", 0) if time_index is None else time_index),
            tracer_time_index=int(source.get("tracer_time_index", 0) if tracer_time_index is None else tracer_time_index),
            max_tracers=int(source.get("max_tracers", 1) if max_tracers is None else max_tracers),
            dt_s=fixture_dt_s,
            repo_root=Path(repo_root),
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


def generate_transport_chain_oracle_fixture(
    paths: LargeOracleFixturePaths,
    *,
    definition: dict[str, object],
    run_config: Path,
    tpcore_executable: Path,
    vdiff_executable: Path,
    convection_executable: Path,
    time_index: int,
    tracer_time_index: int,
    max_tracers: int,
    dt_s: float | None,
    repo_root: Path,
) -> Path:
    paths.directory.mkdir(parents=True, exist_ok=True)
    tpcore_output = paths.directory / "tpcore_output.nc"
    vdiff_input = paths.directory / "vdiff_input.nc"
    vdiff_output = paths.directory / "vdiff_output.nc"
    convection_input = paths.directory / "convection_input.nc"
    convection_output = paths.directory / "convection_output.nc"

    write_transport_step_input_from_config(
        run_config,
        paths.input_path,
        time_index=time_index,
        tracer_time_index=tracer_time_index,
        max_tracers=max_tracers,
        dt_s=dt_s,
    )
    run_pjc_harness(tpcore_executable, paths.input_path, tpcore_output)
    _write_chain_vdiff_input(run_config, paths.input_path, tpcore_output, vdiff_input, time_index=time_index)
    run_operator_harness(vdiff_executable, vdiff_input, vdiff_output)
    _write_chain_convection_input(
        run_config,
        paths.input_path,
        tpcore_output,
        vdiff_output,
        convection_input,
        time_index=time_index,
    )
    run_operator_harness(convection_executable, convection_input, convection_output)
    _write_transport_chain_output(paths.output_path, paths.input_path, tpcore_output, vdiff_output, convection_output)
    _write_generated_transport_chain_manifest(
        paths,
        definition=definition,
        run_config=run_config,
        tpcore_executable=tpcore_executable,
        vdiff_executable=vdiff_executable,
        convection_executable=convection_executable,
        repo_root=repo_root,
    )
    return paths.directory


def generate_vdiff_after_tpcore_oracle_fixture(
    paths: LargeOracleFixturePaths,
    *,
    definition: dict[str, object],
    run_config: Path,
    tpcore_executable: Path,
    vdiff_executable: Path,
    time_index: int,
    tracer_time_index: int,
    max_tracers: int,
    dt_s: float | None,
    repo_root: Path,
) -> Path:
    paths.directory.mkdir(parents=True, exist_ok=True)
    tpcore_input = paths.directory / "transport_step_input.nc"
    tpcore_output = paths.directory / "tpcore_output.nc"
    write_transport_step_input_from_config(
        run_config,
        tpcore_input,
        time_index=time_index,
        tracer_time_index=tracer_time_index,
        max_tracers=max_tracers,
        dt_s=dt_s,
    )
    run_pjc_harness(tpcore_executable, tpcore_input, tpcore_output)
    _write_chain_vdiff_input(run_config, tpcore_input, tpcore_output, paths.input_path, time_index=time_index)
    run_operator_harness(vdiff_executable, paths.input_path, paths.output_path)
    _write_generated_operator_oracle_manifest(
        paths,
        definition=definition,
        run_config=run_config,
        input_harness=VDIFF_INPUT_VERSION,
        output_harness=VDIFF_OUTPUT_VERSION,
        executables={
            "tpcore": str(tpcore_executable),
            "vdiff": str(vdiff_executable),
        },
        repo_root=repo_root,
        tpcore_input_path=tpcore_input,
    )
    return paths.directory


def generate_convection_fullgrid_oracle_fixture(
    paths: LargeOracleFixturePaths,
    *,
    definition: dict[str, object],
    run_config: Path,
    tpcore_executable: Path,
    vdiff_executable: Path,
    convection_executable: Path,
    time_index: int,
    tracer_time_index: int,
    max_tracers: int,
    dt_s: float | None,
    repo_root: Path,
) -> Path:
    paths.directory.mkdir(parents=True, exist_ok=True)
    tpcore_input = paths.directory / "transport_step_input.nc"
    tpcore_output = paths.directory / "tpcore_output.nc"
    vdiff_input = paths.directory / "vdiff_input.nc"
    vdiff_output = paths.directory / "vdiff_output.nc"
    write_transport_step_input_from_config(
        run_config,
        tpcore_input,
        time_index=time_index,
        tracer_time_index=tracer_time_index,
        max_tracers=max_tracers,
        dt_s=dt_s,
    )
    run_pjc_harness(tpcore_executable, tpcore_input, tpcore_output)
    _write_chain_vdiff_input(run_config, tpcore_input, tpcore_output, vdiff_input, time_index=time_index)
    run_operator_harness(vdiff_executable, vdiff_input, vdiff_output)
    _write_chain_convection_input(
        run_config,
        tpcore_input,
        tpcore_output,
        vdiff_output,
        paths.input_path,
        time_index=time_index,
    )
    run_operator_harness(convection_executable, paths.input_path, paths.output_path)
    _write_generated_operator_oracle_manifest(
        paths,
        definition=definition,
        run_config=run_config,
        input_harness=CONVECTION_INPUT_VERSION,
        output_harness=CONVECTION_OUTPUT_VERSION,
        executables={
            "tpcore": str(tpcore_executable),
            "vdiff": str(vdiff_executable),
            "convection": str(convection_executable),
        },
        repo_root=repo_root,
        tpcore_input_path=tpcore_input,
    )
    return paths.directory


def _write_chain_vdiff_input(
    run_config_path: Path,
    chain_input_path: Path,
    tpcore_output_path: Path,
    output_path: Path,
    *,
    time_index: int,
) -> Path:
    config = load_run_config(run_config_path)
    grid = load_transport_grid(config.grid_template)
    forcing = load_transport_forcing(
        meteorology_root(config),
        simulation_start(config),
        grid,
        time_index=time_index,
    )
    with netCDF4.Dataset(chain_input_path) as source:
        lat = np.asarray(source.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(source.variables["lon"][:], dtype=np.float64)
        area = np.asarray(source.variables["area_m2"][:], dtype=np.float64)
        hyai = np.asarray(source.variables["hyai"][:], dtype=np.float64)
        hybi = np.asarray(source.variables["hybi"][:], dtype=np.float64)
        dt_s = float(source.dt_s)
        tracer_names = _read_transport_step_tracer_names(chain_input_path)
    tpcore = read_transport_step_output(tpcore_output_path)
    delp = dry_pressure_thickness_from_surface_hpa(forcing.dry_surface_pressure_hpa, hyai, hybi)
    dry_mass = dry_air_mass_from_pressure(delp, area)[0]
    tpcore_state = TracerField(
        names=tracer_names,
        data=transport_tracer_to_canonical(tpcore.tracer_conc_after),
        units=tuple("mol mol-1 dry" for _ in tracer_names),
        coords={},
    )
    vdiff_input = _build_vdiff_input_after_tpcore(
        tpcore_state,
        forcing,
        dry_mass[np.newaxis, :, :, :],
        delp,
        area,
        hyai_hpa=hyai,
        hybi=hybi,
        top_edge_hpa=float(hyai[-1]),
        dt_s=dt_s,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ntracer = vdiff_input.tracer_conc.shape[-1]
    with netCDF4.Dataset(output, "w") as dataset:
        dataset.createDimension("tracer", ntracer)
        dataset.createDimension("lev", vdiff_input.tracer_conc.shape[0])
        dataset.createDimension("ilev", vdiff_input.tracer_conc.shape[0] + 1)
        dataset.createDimension("lat", lat.size)
        dataset.createDimension("lon", lon.size)
        dataset.harness = VDIFF_INPUT_VERSION
        dataset.dt_s = dt_s
        dataset.createVariable("lon", "f8", ("lon",))[:] = lon
        dataset.createVariable("lat", "f8", ("lat",))[:] = lat
        dataset.createVariable("tracer_conc", "f8", ("lev", "lat", "lon", "tracer"))[:] = vdiff_input.tracer_conc
        dataset.createVariable("u_m_s", "f8", ("lev", "lat", "lon"))[:] = vdiff_input.u_m_s
        dataset.createVariable("v_m_s", "f8", ("lev", "lat", "lon"))[:] = vdiff_input.v_m_s
        dataset.createVariable("temperature_k", "f8", ("lev", "lat", "lon"))[:] = vdiff_input.temperature_k
        dataset.createVariable("specific_humidity_kg_kg", "f8", ("lev", "lat", "lon"))[:] = (
            vdiff_input.specific_humidity_kg_kg
        )
        dataset.createVariable("pmid_hpa", "f8", ("lev", "lat", "lon"))[:] = vdiff_input.pmid_hpa
        dataset.createVariable("pedge_hpa", "f8", ("ilev", "lat", "lon"))[:] = vdiff_input.pedge_hpa
        dataset.createVariable("virtual_temperature_k", "f8", ("lev", "lat", "lon"))[:] = (
            vdiff_input.virtual_temperature_k
        )
        dataset.createVariable("bxheight_m", "f8", ("lev", "lat", "lon"))[:] = vdiff_input.bxheight_m
        dataset.createVariable("dry_air_mass_kg", "f8", ("lev", "lat", "lon"))[:] = vdiff_input.dry_air_mass_kg
        dataset.createVariable("pbl_top_m", "f8", ("lat", "lon"))[:] = vdiff_input.pbl_top_m
        dataset.createVariable("hflux_w_m2", "f8", ("lat", "lon"))[:] = vdiff_input.hflux_w_m2
        dataset.createVariable("eflux_w_m2", "f8", ("lat", "lon"))[:] = vdiff_input.eflux_w_m2
        dataset.createVariable("ustar_m_s", "f8", ("lat", "lon"))[:] = vdiff_input.ustar_m_s
        dataset.createVariable("area_m2", "f8", ("lat", "lon"))[:] = vdiff_input.area_m2
        dataset.createVariable("surface_flux_kg_m2_s", "f8", ("lat", "lon", "tracer"))[:] = (
            vdiff_input.surface_flux_kg_m2_s
        )
    return output


def _write_chain_convection_input(
    run_config_path: Path,
    chain_input_path: Path,
    tpcore_output_path: Path,
    vdiff_output_path: Path,
    output_path: Path,
    *,
    time_index: int,
) -> Path:
    config = load_run_config(run_config_path)
    grid = load_transport_grid(config.grid_template)
    forcing = load_transport_forcing(
        meteorology_root(config),
        simulation_start(config),
        grid,
        time_index=time_index,
    )
    with netCDF4.Dataset(chain_input_path) as source:
        lat = np.asarray(source.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(source.variables["lon"][:], dtype=np.float64)
        area = np.asarray(source.variables["area_m2"][:], dtype=np.float64)
        hyai = np.asarray(source.variables["hyai"][:], dtype=np.float64)
        hybi = np.asarray(source.variables["hybi"][:], dtype=np.float64)
        dt_s = float(source.dt_s)
        tracer_names = _read_transport_step_tracer_names(chain_input_path)
    tpcore = read_transport_step_output(tpcore_output_path)
    vdiff = read_vdiff_output(vdiff_output_path)
    delp = dry_pressure_thickness_from_surface_hpa(forcing.dry_surface_pressure_hpa, hyai, hybi)
    vdiff_state = TracerField(
        names=tracer_names,
        data=transport_tracer_to_canonical(vdiff.tracer_conc_after),
        units=tuple("mol mol-1 dry" for _ in tracer_names),
        coords={},
    )
    convection_input = _build_convection_input_after_vdiff(
        vdiff_state,
        forcing,
        delp,
        area,
        hyai_hpa=hyai,
        hybi=hybi,
        top_edge_hpa=float(hyai[-1]),
        dt_s=dt_s,
        specific_humidity_top=vdiff.specific_humidity_after,
    )
    return _write_convection_input_file(
        output_path,
        tracer_conc=convection_input.tracer_conc,
        tracer_names=tracer_names,
        lon=lon,
        lat=lat,
        cmfmc=convection_input.cmfmc_kg_m2_s,
        dtrain=convection_input.dtrain_kg_m2_s,
        dqrcu=convection_input.dqrcu_kg_kg_s,
        reevapcn=convection_input.reevapcn_kg_kg_s,
        delp_dry=convection_input.delp_dry_hpa,
        delp=convection_input.delp_hpa,
        area=convection_input.area_m2,
        bxheight=convection_input.bxheight_m,
        pficu=convection_input.pficu_kg_m2_s,
        pflcu=convection_input.pflcu_kg_m2_s,
        temperature=convection_input.temperature_k,
        precccon=convection_input.precccon_mm_day,
        dt_s=convection_input.dt_s,
        scenario="transport-chain-full-grid",
        source_run_config=str(run_config_path),
        source_timestamp=simulation_start(config).strftime("%Y-%m-%d %H:%M"),
        source_time_index=time_index,
        vertical_mapping="native_72_to_47_center_and_73_to_48_edge",
    )


def _write_transport_chain_output(
    output_path: Path,
    chain_input_path: Path,
    tpcore_output_path: Path,
    vdiff_output_path: Path,
    convection_output_path: Path,
) -> Path:
    with netCDF4.Dataset(chain_input_path) as source:
        tracer0 = np.asarray(source.variables["tracer_conc"][:], dtype=np.float64)
        area = np.asarray(source.variables["area_m2"][:], dtype=np.float64)
        hyai = np.asarray(source.variables["hyai"][:], dtype=np.float64)
        hybi = np.asarray(source.variables["hybi"][:], dtype=np.float64)
        p1 = np.asarray(source.variables["p1_hpa"][:], dtype=np.float64)
    tpcore = read_transport_step_output(tpcore_output_path)
    vdiff = read_vdiff_output(vdiff_output_path)
    convection = read_convection_output(convection_output_path)
    initial_delp = dry_pressure_thickness_hpa(p1[np.newaxis, :, :] * 100.0, hyai, hybi)
    final_delp = dry_pressure_thickness_hpa(tpcore.surface_pressure_hpa[np.newaxis, :, :] * 100.0, hyai, hybi)
    initial_mass = dry_air_mass_from_pressure(initial_delp, area)
    final_mass = dry_air_mass_from_pressure(final_delp, area)
    initial_tracer_mass = _tracer_mass_for_chain(tracer0, initial_mass[:, ::-1, :, :])
    tpcore_tracer_mass = _tracer_mass_for_chain(tpcore.tracer_conc_after, final_mass[:, ::-1, :, :])
    vdiff_tracer_mass = _tracer_mass_for_chain(vdiff.tracer_conc_after, final_mass[:, ::-1, :, :])
    convection_tracer_mass = _tracer_mass_for_chain(convection.tracer_conc_after, final_mass[:, ::-1, :, :])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    nlev, nlat, nlon, ntracer = convection.tracer_conc_after.shape
    with netCDF4.Dataset(output, "w") as dataset:
        dataset.harness = TRANSPORT_CHAIN_OUTPUT_VERSION
        dataset.createDimension("tracer", ntracer)
        dataset.createDimension("lev", nlev)
        dataset.createDimension("lat", nlat)
        dataset.createDimension("lon", nlon)
        dataset.negative_count_after_tpcore = int(np.count_nonzero(tpcore.tracer_conc_after < 0.0))
        dataset.negative_count_after_vdiff = int(vdiff.negative_count_after_clip)
        dataset.negative_count_after_convection = int(convection.negative_count_after)
        dataset.createVariable("tracer_conc_after", "f8", ("lev", "lat", "lon", "tracer"))[:] = (
            convection.tracer_conc_after
        )
        dataset.createVariable("tpcore_tracer_conc_after", "f8", ("lev", "lat", "lon", "tracer"))[:] = (
            tpcore.tracer_conc_after
        )
        dataset.createVariable("vdiff_tracer_conc_after", "f8", ("lev", "lat", "lon", "tracer"))[:] = (
            vdiff.tracer_conc_after
        )
        dataset.createVariable("initial_tracer_mass", "f8", ("tracer",))[:] = initial_tracer_mass
        dataset.createVariable("tpcore_tracer_mass", "f8", ("tracer",))[:] = tpcore_tracer_mass
        dataset.createVariable("vdiff_tracer_mass", "f8", ("tracer",))[:] = vdiff_tracer_mass
        dataset.createVariable("convection_tracer_mass", "f8", ("tracer",))[:] = convection_tracer_mass
        dataset.createVariable("diag14_mass_flux", "f8", ("lev", "lat", "lon", "tracer"))[:] = (
            convection.diag14_mass_flux
        )
    return output


def _tracer_mass_for_chain(tracer: np.ndarray, dry_air_mass: np.ndarray) -> np.ndarray:
    tracer_array = np.asarray(tracer, dtype=np.float64)
    mass_array = np.asarray(dry_air_mass, dtype=np.float64)
    if mass_array.ndim == 4 and mass_array.shape[0] == 1:
        mass_array = mass_array[0]
    return np.sum(tracer_array * mass_array[:, :, :, np.newaxis], axis=(0, 1, 2))


def _tracer_mass_common_basis(tracer: np.ndarray, dry_air_mass: np.ndarray) -> np.ndarray:
    """Compute per-tracer scalar mass from a shared mass field for comparisons."""

    tracer_array = np.asarray(tracer, dtype=np.float64)
    mass_array = np.asarray(dry_air_mass, dtype=np.float64)
    if tracer_array.ndim != 4:
        raise ValueError(f"tracer must have shape (lev, lat, lon, tracer), found {tracer_array.shape}")
    if mass_array.ndim == 4 and mass_array.shape[0] == 1:
        mass_array = mass_array[0]
    if mass_array.shape != tracer_array.shape[:3]:
        raise ValueError(f"dry_air_mass must have shape {tracer_array.shape[:3]}, found {mass_array.shape}")
    return np.sum(tracer_array * mass_array[:, :, :, np.newaxis], axis=(0, 1, 2))


def _convection_common_dry_air_mass(delp_dry_hpa: np.ndarray, area_m2: np.ndarray) -> np.ndarray:
    mass = dry_air_mass_from_pressure(np.asarray(delp_dry_hpa, dtype=np.float64)[np.newaxis, :, :, :], area_m2)
    return mass[0]


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
    if fixture_id == BASE_INITIAL_TRANSPORT_CHAIN_FIXTURE_ID:
        return format_transport_chain_comparison(
            compare_transport_chain_oracle_fixture(fixture_id, cache_dir=cache_dir, manifest_dir=manifest_dir)
        )
    if fixture_id == BASE_INITIAL_VDIFF_AFTER_TPCORE_FIXTURE_ID:
        paths = large_oracle_fixture_paths(fixture_id, cache_dir=cache_dir, manifest_dir=manifest_dir)
        return format_vdiff_comparison(compare_vdiff_output(paths.input_path, paths.output_path))
    if fixture_id == BASE_INITIAL_CONVECTION_FULLGRID_FIXTURE_ID:
        paths = large_oracle_fixture_paths(fixture_id, cache_dir=cache_dir, manifest_dir=manifest_dir)
        return format_convection_comparison(compare_convection_output(paths.input_path, paths.output_path))
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


def compare_transport_chain_oracle_fixture(
    fixture_id: str = BASE_INITIAL_TRANSPORT_CHAIN_FIXTURE_ID,
    *,
    cache_dir: str | Path = "oracle_data",
    manifest_dir: str | Path | None = None,
) -> TransportChainComparison:
    if fixture_id != BASE_INITIAL_TRANSPORT_CHAIN_FIXTURE_ID:
        raise ValueError(f"{fixture_id!r} is not a transport-chain oracle fixture")
    paths = large_oracle_fixture_paths(fixture_id, cache_dir=cache_dir, manifest_dir=manifest_dir)
    source = _large_oracle_source(paths)
    config = load_run_config(source.get("run_config", "validation_runs/cases/realistic_restart_noemis_2x25/wombat/main/run.yml"))
    tracer_names = _read_transport_step_tracer_names(paths.input_path)
    with netCDF4.Dataset(paths.input_path) as dataset:
        tracer0 = np.asarray(dataset.variables["tracer_conc"][:], dtype=np.float64)
        area = np.asarray(dataset.variables["area_m2"][:], dtype=np.float64)
        hyai = np.asarray(dataset.variables["hyai"][:], dtype=np.float64)
        hybi = np.asarray(dataset.variables["hybi"][:], dtype=np.float64)
        p1_hpa = np.asarray(dataset.variables["p1_hpa"][:], dtype=np.float64)
        dt_s = float(dataset.dt_s)
    grid = load_transport_grid(config.grid_template)
    forcing = load_transport_forcing(
        meteorology_root(config),
        simulation_start(config),
        grid,
        time_index=int(source.get("time_index", 0)),
    )
    field = TracerField(
        names=tracer_names,
        data=transport_tracer_to_canonical(tracer0),
        units=tuple("mol mol-1 dry" for _ in tracer_names),
        coords={},
    )
    trace = trace_transport_one_step(field, forcing, grid, dt_s=dt_s)
    result = trace.result
    stage_masses = compute_transport_stage_masses(trace, field, area)
    initial_scalar_mass = stage_masses[0].initial_scalar_mass
    tpcore_scalar_mass = stage_masses[0].final_scalar_mass
    vdiff_scalar_mass = stage_masses[1].final_scalar_mass
    convection_scalar_mass = stage_masses[2].final_scalar_mass
    with netCDF4.Dataset(paths.output_path) as dataset:
        expected_tracer = np.asarray(dataset.variables["tracer_conc_after"][:], dtype=np.float64)
        expected_tpcore_tracer = np.asarray(dataset.variables["tpcore_tracer_conc_after"][:], dtype=np.float64)
        expected_vdiff_tracer = np.asarray(dataset.variables["vdiff_tracer_conc_after"][:], dtype=np.float64)
        initial_mass = np.asarray(dataset.variables["initial_tracer_mass"][:], dtype=np.float64)
        tpcore_mass = np.asarray(dataset.variables["tpcore_tracer_mass"][:], dtype=np.float64)
        vdiff_mass = np.asarray(dataset.variables["vdiff_tracer_mass"][:], dtype=np.float64)
        convection_mass = np.asarray(dataset.variables["convection_tracer_mass"][:], dtype=np.float64)
        negative_count = int(getattr(dataset, "negative_count_after_convection"))
    initial_delp = dry_pressure_thickness_hpa(p1_hpa[np.newaxis, :, :] * 100.0, hyai, hybi)
    initial_dry_mass = dry_air_mass_from_pressure(initial_delp, area)[0][::-1]
    final_dry_mass = result.dry_air_mass_kg[0][::-1]
    common_oracle_initial_mass = _tracer_mass_common_basis(tracer0, initial_dry_mass)
    common_oracle_tpcore_mass = _tracer_mass_common_basis(expected_tpcore_tracer, final_dry_mass)
    common_oracle_vdiff_mass = _tracer_mass_common_basis(expected_vdiff_tracer, final_dry_mass)
    common_oracle_convection_mass = _tracer_mass_common_basis(expected_tracer, final_dry_mass)
    common_actual_convection_mass = convection_scalar_mass
    actual_tracer = canonical_time_slice(result.state.to_canonical())
    error = np.abs(actual_tracer - expected_tracer)
    return TransportChainComparison(
        tracer_max_abs_error=float(np.max(error)),
        tracer_mean_abs_error=float(np.mean(error)),
        negative_count_expected=negative_count,
        negative_count_actual=int(np.count_nonzero(actual_tracer < 0.0)),
        common_basis_initial_mass_max_abs_error=float(
            np.max(np.abs(initial_scalar_mass - common_oracle_initial_mass))
        ),
        common_basis_final_mass_max_abs_error=float(
            np.max(np.abs(common_actual_convection_mass - common_oracle_convection_mass))
        ),
        common_basis_mass_change_max_abs_error=float(
            np.max(
                np.abs(
                    (common_actual_convection_mass - initial_scalar_mass)
                    - (common_oracle_convection_mass - common_oracle_initial_mass)
                )
            )
        ),
        common_basis_python_mass_change_max_abs=float(
            np.max(np.abs(common_actual_convection_mass - initial_scalar_mass))
        ),
        common_basis_oracle_mass_change_max_abs=float(
            np.max(np.abs(common_oracle_convection_mass - common_oracle_initial_mass))
        ),
        common_basis_tpcore_stage_mass_change_max_abs=float(
            np.max(np.abs(common_oracle_tpcore_mass - common_oracle_initial_mass))
        ),
        common_basis_vdiff_stage_mass_change_max_abs=float(
            np.max(np.abs(common_oracle_vdiff_mass - common_oracle_tpcore_mass))
        ),
        common_basis_convection_stage_mass_change_max_abs=float(
            np.max(np.abs(common_oracle_convection_mass - common_oracle_vdiff_mass))
        ),
        reported_final_mass_max_abs_error=float(np.max(np.abs(convection_scalar_mass - convection_mass))),
        reported_python_mass_change_max_abs=float(np.max(np.abs(convection_scalar_mass - initial_scalar_mass))),
        reported_oracle_mass_change_max_abs=float(np.max(np.abs(convection_mass - initial_mass))),
        reported_tpcore_stage_mass_change_max_abs=float(np.max(np.abs(tpcore_mass - initial_mass))),
        reported_vdiff_stage_mass_change_max_abs=float(np.max(np.abs(vdiff_mass - tpcore_mass))),
        reported_convection_stage_mass_change_max_abs=float(np.max(np.abs(convection_mass - vdiff_mass))),
    )


def compare_transport_chain_handoffs(
    fixture_id: str = BASE_INITIAL_TRANSPORT_CHAIN_FIXTURE_ID,
    *,
    cache_dir: str | Path = "oracle_data",
    manifest_dir: str | Path | None = None,
) -> str:
    if fixture_id != BASE_INITIAL_TRANSPORT_CHAIN_FIXTURE_ID:
        raise ValueError(f"{fixture_id!r} is not a transport-chain oracle fixture")
    for required in (
        BASE_INITIAL_TRANSPORT_CHAIN_FIXTURE_ID,
        BASE_INITIAL_VDIFF_AFTER_TPCORE_FIXTURE_ID,
        BASE_INITIAL_CONVECTION_FULLGRID_FIXTURE_ID,
    ):
        check = check_large_oracle_fixture(required, cache_dir=cache_dir, manifest_dir=manifest_dir)
        if not check.is_available:
            raise FileNotFoundError(format_large_oracle_fixture_check(check))

    chain_paths = large_oracle_fixture_paths(fixture_id, cache_dir=cache_dir, manifest_dir=manifest_dir)
    vdiff_paths = large_oracle_fixture_paths(
        BASE_INITIAL_VDIFF_AFTER_TPCORE_FIXTURE_ID,
        cache_dir=cache_dir,
        manifest_dir=manifest_dir,
    )
    convection_paths = large_oracle_fixture_paths(
        BASE_INITIAL_CONVECTION_FULLGRID_FIXTURE_ID,
        cache_dir=cache_dir,
        manifest_dir=manifest_dir,
    )
    diagnostics = _trace_transport_chain_fixture(chain_paths)
    rows = ["section,field,max_abs,mean_abs,actual_shape,expected_shape"]

    with netCDF4.Dataset(vdiff_paths.input_path) as dataset:
        _append_vdiff_input_handoff_rows(rows, diagnostics.vdiff_input, dataset)
    expected_vdiff = read_vdiff_output(vdiff_paths.output_path)
    _append_vdiff_output_handoff_rows(rows, diagnostics.vdiff_output, expected_vdiff)

    with netCDF4.Dataset(convection_paths.input_path) as dataset:
        _append_convection_input_handoff_rows(rows, diagnostics.convection_input, dataset)
    expected_convection = read_convection_output(convection_paths.output_path)
    _append_convection_output_handoff_rows(rows, diagnostics.convection_output, expected_convection)

    with netCDF4.Dataset(chain_paths.output_path) as dataset:
        _append_array_error_row(
            rows,
            "final_chain_output",
            "tracer_conc_after",
            canonical_time_slice(diagnostics.result.state.to_canonical()),
            np.asarray(dataset.variables["tracer_conc_after"][:], dtype=np.float64),
        )
        _append_array_error_row(
            rows,
            "final_chain_output",
            "diag14_mass_flux",
            diagnostics.convection_output.diag14_mass_flux,
            np.asarray(dataset.variables["diag14_mass_flux"][:], dtype=np.float64),
        )
    return "\n".join(rows)


def _trace_transport_chain_fixture(paths: LargeOracleFixturePaths):
    source = _large_oracle_source(paths)
    config = load_run_config(source.get("run_config", "validation_runs/cases/realistic_restart_noemis_2x25/wombat/main/run.yml"))
    tracer_names = _read_transport_step_tracer_names(paths.input_path)
    with netCDF4.Dataset(paths.input_path) as dataset:
        tracer0 = np.asarray(dataset.variables["tracer_conc"][:], dtype=np.float64)
        dt_s = float(dataset.dt_s)
    grid = load_transport_grid(config.grid_template)
    forcing = load_transport_forcing(
        meteorology_root(config),
        simulation_start(config),
        grid,
        time_index=int(source.get("time_index", 0)),
    )
    field = TracerField(
        names=tracer_names,
        data=transport_tracer_to_canonical(tracer0),
        units=tuple("mol mol-1 dry" for _ in tracer_names),
        coords={},
    )
    return trace_transport_one_step(field, forcing, grid, dt_s=dt_s)


def _append_vdiff_input_handoff_rows(rows: list[str], actual, dataset: netCDF4.Dataset) -> None:
    fields = (
        ("tracer_conc", "tracer_conc"),
        ("u_m_s", "u_m_s"),
        ("v_m_s", "v_m_s"),
        ("temperature_k", "temperature_k"),
        ("specific_humidity_kg_kg", "specific_humidity_kg_kg"),
        ("pmid_hpa", "pmid_hpa"),
        ("pedge_hpa", "pedge_hpa"),
        ("virtual_temperature_k", "virtual_temperature_k"),
        ("bxheight_m", "bxheight_m"),
        ("dry_air_mass_kg", "dry_air_mass_kg"),
        ("pbl_top_m", "pbl_top_m"),
        ("hflux_w_m2", "hflux_w_m2"),
        ("eflux_w_m2", "eflux_w_m2"),
        ("ustar_m_s", "ustar_m_s"),
        ("area_m2", "area_m2"),
        ("surface_flux_kg_m2_s", "surface_flux_kg_m2_s"),
    )
    for attr, variable in fields:
        _append_array_error_row(
            rows,
            "tpcore_to_vdiff_input",
            variable,
            getattr(actual, attr),
            np.asarray(dataset.variables[variable][:], dtype=np.float64),
        )


def _append_vdiff_output_handoff_rows(rows: list[str], actual, expected: VdiffOutput) -> None:
    comparisons = (
        ("tracer_conc_after", actual.tracer_conc, expected.tracer_conc_after),
        ("specific_humidity_after", actual.specific_humidity_kg_kg, expected.specific_humidity_after),
        ("kvh_m2_s", actual.kvh_m2_s, expected.kvh_m2_s),
        ("kvm_m2_s", actual.kvm_m2_s, expected.kvm_m2_s),
        ("pbl_top_m", actual.pbl_top_m, expected.pbl_top_m),
        ("tpert_k", actual.tpert_k, expected.tpert_k),
        ("qpert_kg_kg", actual.qpert_kg_kg, expected.qpert_kg_kg),
    )
    for field, actual_values, expected_values in comparisons:
        _append_array_error_row(rows, "vdiff_output", field, actual_values, expected_values)


def _append_convection_input_handoff_rows(rows: list[str], actual, dataset: netCDF4.Dataset) -> None:
    fields = (
        ("tracer_conc", "tracer_conc"),
        ("cmfmc_kg_m2_s", "cmfmc_kg_m2_s"),
        ("dtrain_kg_m2_s", "dtrain_kg_m2_s"),
        ("dqrcu_kg_kg_s", "dqrcu_kg_kg_s"),
        ("reevapcn_kg_kg_s", "reevapcn_kg_kg_s"),
        ("delp_dry_hpa", "delp_dry_hpa"),
        ("delp_hpa", "delp_hpa"),
        ("area_m2", "area_m2"),
        ("bxheight_m", "bxheight_m"),
        ("pficu_kg_m2_s", "pficu_kg_m2_s"),
        ("pflcu_kg_m2_s", "pflcu_kg_m2_s"),
        ("temperature_k", "temperature_k"),
        ("precccon_mm_day", "precccon_mm_day"),
    )
    for attr, variable in fields:
        _append_array_error_row(
            rows,
            "vdiff_to_convection_input",
            variable,
            getattr(actual, attr),
            np.asarray(dataset.variables[variable][:], dtype=np.float64),
        )


def _append_convection_output_handoff_rows(rows: list[str], actual, expected: ConvectionOutput) -> None:
    _append_array_error_row(
        rows,
        "convection_output",
        "tracer_conc_after",
        actual.tracer_conc,
        expected.tracer_conc_after,
    )
    _append_array_error_row(
        rows,
        "convection_output",
        "diag14_mass_flux",
        actual.diag14_mass_flux,
        expected.diag14_mass_flux,
    )


def _append_array_error_row(
    rows: list[str],
    section: str,
    field: str,
    actual_values: np.ndarray,
    expected_values: np.ndarray,
) -> None:
    actual = np.asarray(actual_values, dtype=np.float64)
    expected = np.asarray(expected_values, dtype=np.float64)
    actual_shape = "x".join(str(value) for value in actual.shape)
    expected_shape = "x".join(str(value) for value in expected.shape)
    if actual.shape != expected.shape:
        rows.append(f"{section},{field},nan,nan,{actual_shape},{expected_shape}")
        return
    error = np.abs(actual - expected)
    rows.append(
        f"{section},{field},{float(np.max(error)):.8e},{float(np.mean(error)):.8e},{actual_shape},{expected_shape}"
    )


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


def write_dry_pressure_input(
    path: str | Path,
    *,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    area_m2: np.ndarray,
    hyai_hpa: np.ndarray,
    hybi: np.ndarray,
    ps1_wet_hpa: np.ndarray,
    ps2_wet_hpa: np.ndarray,
    sphu1_kg_kg: np.ndarray,
    sphu2_kg_kg: np.ndarray,
    tmpu1_k: np.ndarray,
    tmpu2_k: np.ndarray,
    ntime0_s: int,
    ntime1_s: int,
    ntdt_s: int,
) -> Path:
    """Write a NetCDF fixture for the GEOS-Chem dry-pressure harness."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = np.asarray(lon_deg, dtype=np.float64)
    hyai = np.asarray(hyai_hpa, dtype=np.float64)
    hybi_arr = np.asarray(hybi, dtype=np.float64)
    area = np.asarray(area_m2, dtype=np.float64)
    ps1 = np.asarray(ps1_wet_hpa, dtype=np.float64)
    ps2 = np.asarray(ps2_wet_hpa, dtype=np.float64)
    q1 = np.asarray(sphu1_kg_kg, dtype=np.float64)
    q2 = np.asarray(sphu2_kg_kg, dtype=np.float64)
    t1 = np.asarray(tmpu1_k, dtype=np.float64)
    t2 = np.asarray(tmpu2_k, dtype=np.float64)
    _assert_dry_pressure_shapes(lat, lon, area, hyai, hybi_arr, ps1, ps2, q1, q2, t1, t2)

    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("lon", lon.size)
        dataset.createDimension("lat", lat.size)
        dataset.createDimension("lev", hyai.size - 1)
        dataset.createDimension("ilev", hyai.size)
        dataset.harness = DRY_PRESSURE_INPUT_VERSION
        dataset.ntime0_s = int(ntime0_s)
        dataset.ntime1_s = int(ntime1_s)
        dataset.ntdt_s = int(ntdt_s)
        dataset.createVariable("lon", "f8", ("lon",))[:] = lon
        dataset.createVariable("lat", "f8", ("lat",))[:] = lat
        dataset.createVariable("hyai", "f8", ("ilev",))[:] = hyai
        dataset.createVariable("hybi", "f8", ("ilev",))[:] = hybi_arr
        dataset.createVariable("area_m2", "f8", ("lat", "lon"))[:] = area
        dataset.createVariable("ps1_wet_hpa", "f8", ("lat", "lon"))[:] = ps1
        dataset.createVariable("ps2_wet_hpa", "f8", ("lat", "lon"))[:] = ps2
        dataset.createVariable("sphu1_kg_kg", "f8", ("lev", "lat", "lon"))[:] = q1
        dataset.createVariable("sphu2_kg_kg", "f8", ("lev", "lat", "lon"))[:] = q2
        dataset.createVariable("tmpu1_k", "f8", ("lev", "lat", "lon"))[:] = t1
        dataset.createVariable("tmpu2_k", "f8", ("lev", "lat", "lon"))[:] = t2
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
        raise ValueError(f"tracer_conc must have shape (lev, lat, lon, tracer), found {tracers.shape}")
    if tracer_names is None:
        names = tuple(f"tracer_{index + 1:03d}" for index in range(tracers.shape[-1]))
    else:
        names = tuple(tracer_names)
    if len(names) != tracers.shape[-1]:
        raise ValueError("tracer_names length must match tracer_conc last dimension")

    with netCDF4.Dataset(path, "a") as dataset:
        if getattr(dataset, "harness", "") != PJC_INPUT_VERSION:
            raise ValueError(f"{path} is not a {PJC_INPUT_VERSION} file")
        expected = (
            len(dataset.dimensions["lev"]),
            len(dataset.dimensions["lat"]),
            len(dataset.dimensions["lon"]),
        )
        if tracers.shape[:3] != expected:
            raise ValueError(f"tracer_conc grid must have shape {expected}, found {tracers.shape[:3]}")
        dataset.harness = TRANSPORT_INPUT_VERSION
        dataset.createDimension("tracer", tracers.shape[-1])
        name_length = max(max((len(name) for name in names), default=1), 1)
        dataset.createDimension("name_strlen", name_length)
        dataset.createVariable("tracer_conc", "f8", ("lev", "lat", "lon", "tracer"))[:] = tracers
        name_var = dataset.createVariable("tracer_name", "S1", ("tracer", "name_strlen"))
        encoded = np.asarray([name.encode("ascii", errors="replace") for name in names], dtype=f"S{name_length}")
        name_var[:] = _fixed_width_strings_to_chars(encoded)
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


def read_dry_pressure_output(path: str | Path) -> DryPressureOutput:
    with netCDF4.Dataset(path) as dataset:
        if getattr(dataset, "harness", "") != DRY_PRESSURE_OUTPUT_VERSION:
            raise ValueError(f"{path} is not a {DRY_PRESSURE_OUTPUT_VERSION} file")
        return DryPressureOutput(
            ps1_wet_hpa=np.asarray(dataset.variables["ps1_wet_hpa"][:], dtype=np.float64),
            ps2_wet_hpa=np.asarray(dataset.variables["ps2_wet_hpa"][:], dtype=np.float64),
            ps1_dry_hpa=np.asarray(dataset.variables["ps1_dry_hpa"][:], dtype=np.float64),
            ps2_dry_hpa=np.asarray(dataset.variables["ps2_dry_hpa"][:], dtype=np.float64),
            psc2_wet_hpa=np.asarray(dataset.variables["psc2_wet_hpa"][:], dtype=np.float64),
            psc2_dry_hpa=np.asarray(dataset.variables["psc2_dry_hpa"][:], dtype=np.float64),
            delp_dry_hpa=np.asarray(dataset.variables["delp_dry_hpa"][:], dtype=np.float64),
            specific_humidity_kg_kg=np.asarray(dataset.variables["specific_humidity_kg_kg"][:], dtype=np.float64),
            temperature_k=np.asarray(dataset.variables["temperature_k"][:], dtype=np.float64),
        )


def read_met_airqnt_output(path: str | Path) -> MetAirQntOutput:
    with netCDF4.Dataset(path) as dataset:
        if getattr(dataset, "harness", "") != MET_AIRQNT_OUTPUT_VERSION:
            raise ValueError(f"{path} is not a {MET_AIRQNT_OUTPUT_VERSION} file")
        return MetAirQntOutput(
            ps1_wet_hpa=np.asarray(dataset.variables["ps1_wet_hpa"][:], dtype=np.float64),
            ps2_wet_hpa=np.asarray(dataset.variables["ps2_wet_hpa"][:], dtype=np.float64),
            ps1_dry_hpa=np.asarray(dataset.variables["ps1_dry_hpa"][:], dtype=np.float64),
            ps2_dry_hpa=np.asarray(dataset.variables["ps2_dry_hpa"][:], dtype=np.float64),
            psc2_wet_hpa=np.asarray(dataset.variables["psc2_wet_hpa"][:], dtype=np.float64),
            psc2_dry_hpa=np.asarray(dataset.variables["psc2_dry_hpa"][:], dtype=np.float64),
            wet_pressure_edges_hpa=np.asarray(dataset.variables["wet_pressure_edges_hpa"][:], dtype=np.float64),
            wet_pressure_mid_hpa=np.asarray(dataset.variables["wet_pressure_mid_hpa"][:], dtype=np.float64),
            wet_pressure_thickness_hpa=np.asarray(
                dataset.variables["wet_pressure_thickness_hpa"][:], dtype=np.float64
            ),
            dry_partial_pressure_edges_hpa=np.asarray(
                dataset.variables["dry_partial_pressure_edges_hpa"][:], dtype=np.float64
            ),
            dry_partial_pressure_mid_hpa=np.asarray(
                dataset.variables["dry_partial_pressure_mid_hpa"][:], dtype=np.float64
            ),
            delp_dry_hpa=np.asarray(dataset.variables["delp_dry_hpa"][:], dtype=np.float64),
            specific_humidity_kg_kg=np.asarray(dataset.variables["specific_humidity_kg_kg"][:], dtype=np.float64),
            temperature_k=np.asarray(dataset.variables["temperature_k"][:], dtype=np.float64),
            water_vapor_vv_dry=np.asarray(dataset.variables["water_vapor_vv_dry"][:], dtype=np.float64),
            virtual_temperature_k=np.asarray(dataset.variables["virtual_temperature_k"][:], dtype=np.float64),
            bxheight_m=np.asarray(dataset.variables["bxheight_m"][:], dtype=np.float64),
            dry_air_mass_kg=np.asarray(dataset.variables["dry_air_mass_kg"][:], dtype=np.float64),
            air_volume_m3=np.asarray(dataset.variables["air_volume_m3"][:], dtype=np.float64),
        )


def write_python_dry_pressure_output(input_path: str | Path, output_path: str | Path) -> Path:
    with netCDF4.Dataset(input_path) as dataset:
        if getattr(dataset, "harness", "") != DRY_PRESSURE_INPUT_VERSION:
            raise ValueError(f"{input_path} is not a {DRY_PRESSURE_INPUT_VERSION} file")
        hyai = np.asarray(dataset.variables["hyai"][:], dtype=np.float64)
        hybi = np.asarray(dataset.variables["hybi"][:], dtype=np.float64)
        area = np.asarray(dataset.variables["area_m2"][:], dtype=np.float64)
        ps1_wet_raw = np.asarray(dataset.variables["ps1_wet_hpa"][:], dtype=np.float64)
        ps2_wet_raw = np.asarray(dataset.variables["ps2_wet_hpa"][:], dtype=np.float64)
        q1 = np.asarray(dataset.variables["sphu1_kg_kg"][:], dtype=np.float64)
        q2 = np.asarray(dataset.variables["sphu2_kg_kg"][:], dtype=np.float64)
        t1 = np.asarray(dataset.variables["tmpu1_k"][:], dtype=np.float64)
        t2 = np.asarray(dataset.variables["tmpu2_k"][:], dtype=np.float64)
        ntime0_s = int(dataset.ntime0_s)
        ntime1_s = int(dataset.ntime1_s)
        ntdt_s = int(dataset.ntdt_s)

    ps1_wet = wet_surface_pressure_hpa(ps1_wet_raw[np.newaxis, :, :] * 100.0, area_m2=area)[0]
    ps2_wet = wet_surface_pressure_hpa(ps2_wet_raw[np.newaxis, :, :] * 100.0, area_m2=area)[0]
    ps1_dry = dry_surface_pressure_hpa(ps1_wet_raw[np.newaxis, :, :] * 100.0, q1[np.newaxis, :, :, :], hyai, hybi, area_m2=area)[
        0
    ]
    ps2_dry = dry_surface_pressure_hpa(ps2_wet_raw[np.newaxis, :, :] * 100.0, q2[np.newaxis, :, :, :], hyai, hybi, area_m2=area)[
        0
    ]
    tm = (float(ntime1_s) + float(ntdt_s) / 2.0 - float(ntime0_s)) / 10800.0
    tc2 = (float(ntime1_s) + float(ntdt_s) - float(ntime0_s)) / 10800.0
    if tm > 1.0:
        tm -= 1.0
        tc2 -= 1.0
    psc2_wet = ps1_wet + (ps2_wet - ps1_wet) * tc2
    psc2_dry = ps1_dry + (ps2_dry - ps1_dry) * tc2
    sphu = q1 + (q2 - q1) * tm
    temperature = t1 + (t2 - t1) * tm
    delp_dry = dry_pressure_thickness_from_surface_hpa(psc2_dry[np.newaxis, :, :], hyai, hybi)[0]

    return write_dry_pressure_output(
        output_path,
        DryPressureOutput(
            ps1_wet_hpa=ps1_wet,
            ps2_wet_hpa=ps2_wet,
            ps1_dry_hpa=ps1_dry,
            ps2_dry_hpa=ps2_dry,
            psc2_wet_hpa=psc2_wet,
            psc2_dry_hpa=psc2_dry,
            delp_dry_hpa=delp_dry,
            specific_humidity_kg_kg=sphu,
            temperature_k=temperature,
        ),
    )


def write_python_met_airqnt_output(input_path: str | Path, output_path: str | Path) -> Path:
    with netCDF4.Dataset(input_path) as dataset:
        if getattr(dataset, "harness", "") != DRY_PRESSURE_INPUT_VERSION:
            raise ValueError(f"{input_path} is not a {DRY_PRESSURE_INPUT_VERSION} file")
        hyai = np.asarray(dataset.variables["hyai"][:], dtype=np.float64)
        hybi = np.asarray(dataset.variables["hybi"][:], dtype=np.float64)
        area = np.asarray(dataset.variables["area_m2"][:], dtype=np.float64)
        ps1_wet_raw = np.asarray(dataset.variables["ps1_wet_hpa"][:], dtype=np.float64)
        ps2_wet_raw = np.asarray(dataset.variables["ps2_wet_hpa"][:], dtype=np.float64)
        q1 = np.asarray(dataset.variables["sphu1_kg_kg"][:], dtype=np.float64)
        q2 = np.asarray(dataset.variables["sphu2_kg_kg"][:], dtype=np.float64)
        t1 = np.asarray(dataset.variables["tmpu1_k"][:], dtype=np.float64)
        t2 = np.asarray(dataset.variables["tmpu2_k"][:], dtype=np.float64)
        ntime0_s = int(dataset.ntime0_s)
        ntime1_s = int(dataset.ntime1_s)
        ntdt_s = int(dataset.ntdt_s)

    ps1_wet = wet_surface_pressure_hpa(ps1_wet_raw[np.newaxis, :, :] * 100.0, area_m2=area)[0]
    ps2_wet = wet_surface_pressure_hpa(ps2_wet_raw[np.newaxis, :, :] * 100.0, area_m2=area)[0]
    ps1_dry = dry_surface_pressure_hpa(ps1_wet_raw[np.newaxis, :, :] * 100.0, q1[np.newaxis, :, :, :], hyai, hybi, area_m2=area)[
        0
    ]
    ps2_dry = dry_surface_pressure_hpa(ps2_wet_raw[np.newaxis, :, :] * 100.0, q2[np.newaxis, :, :, :], hyai, hybi, area_m2=area)[
        0
    ]
    tm = (float(ntime1_s) + float(ntdt_s) / 2.0 - float(ntime0_s)) / 10800.0
    tc2 = (float(ntime1_s) + float(ntdt_s) - float(ntime0_s)) / 10800.0
    if tm > 1.0:
        tm -= 1.0
        tc2 -= 1.0

    psc2_wet = ps1_wet + (ps2_wet - ps1_wet) * tc2
    psc2_dry = ps1_dry + (ps2_dry - ps1_dry) * tc2
    sphu = q1 + (q2 - q1) * tm
    temperature = t1 + (t2 - t1) * tm

    wet_edges = _hybrid_pressure_edges_from_surface_hpa(psc2_wet, hyai, hybi)
    wet_pressure_thickness = wet_edges[:-1] - wet_edges[1:]
    wet_pressure_mid = 0.5 * (wet_edges[:-1] + wet_edges[1:])
    delp_dry = dry_pressure_thickness_from_surface_hpa(psc2_dry[np.newaxis, :, :], hyai, hybi)[0]
    water_vapor_vv_dry = AIRMW_G_PER_MOL * sphu / (H2OMW_G_PER_MOL * (1.0 - sphu))
    xh2o = water_vapor_vv_dry / (1.0 + water_vapor_vv_dry)
    virtual_temperature = temperature / (1.0 - xh2o * (1.0 - H2OMW_G_PER_MOL / AIRMW_G_PER_MOL))
    bxheight = (RD_J_PER_KG_K / G0_M_PER_S2) * virtual_temperature * np.log(wet_edges[:-1] / wet_edges[1:])
    dry_partial_edges = np.empty_like(wet_edges)
    dry_partial_edges[:-1] = wet_edges[:-1] * (1.0 - xh2o)
    dry_partial_edges[-1] = wet_edges[-1] * (1.0 - xh2o[-1])
    dry_partial_mid = wet_pressure_mid * (1.0 - xh2o)
    dry_air_mass = delp_dry * 100.0 / G0_M_PER_S2 * area[np.newaxis, :, :]
    air_volume = bxheight * area[np.newaxis, :, :]

    return write_met_airqnt_output(
        output_path,
        MetAirQntOutput(
            ps1_wet_hpa=ps1_wet,
            ps2_wet_hpa=ps2_wet,
            ps1_dry_hpa=ps1_dry,
            ps2_dry_hpa=ps2_dry,
            psc2_wet_hpa=psc2_wet,
            psc2_dry_hpa=psc2_dry,
            wet_pressure_edges_hpa=wet_edges,
            wet_pressure_mid_hpa=wet_pressure_mid,
            wet_pressure_thickness_hpa=wet_pressure_thickness,
            dry_partial_pressure_edges_hpa=dry_partial_edges,
            dry_partial_pressure_mid_hpa=dry_partial_mid,
            delp_dry_hpa=delp_dry,
            specific_humidity_kg_kg=sphu,
            temperature_k=temperature,
            water_vapor_vv_dry=water_vapor_vv_dry,
            virtual_temperature_k=virtual_temperature,
            bxheight_m=bxheight,
            dry_air_mass_kg=dry_air_mass,
            air_volume_m3=air_volume,
        ),
    )


def _hybrid_pressure_edges_from_surface_hpa(surface_pressure_hpa: np.ndarray, hyai_hpa: np.ndarray, hybi: np.ndarray) -> np.ndarray:
    ps = np.asarray(surface_pressure_hpa, dtype=np.float64)
    hyai = np.asarray(hyai_hpa, dtype=np.float64)
    hyb = np.asarray(hybi, dtype=np.float64)
    return hyai[:, np.newaxis, np.newaxis] + hyb[:, np.newaxis, np.newaxis] * ps[np.newaxis, :, :]


def write_dry_pressure_output(path: str | Path, output: DryPressureOutput) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nlev, nlat, nlon = output.delp_dry_hpa.shape
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("lev", nlev)
        dataset.createDimension("lat", nlat)
        dataset.createDimension("lon", nlon)
        dataset.harness = DRY_PRESSURE_OUTPUT_VERSION
        dataset.createVariable("ps1_wet_hpa", "f8", ("lat", "lon"))[:] = output.ps1_wet_hpa
        dataset.createVariable("ps2_wet_hpa", "f8", ("lat", "lon"))[:] = output.ps2_wet_hpa
        dataset.createVariable("ps1_dry_hpa", "f8", ("lat", "lon"))[:] = output.ps1_dry_hpa
        dataset.createVariable("ps2_dry_hpa", "f8", ("lat", "lon"))[:] = output.ps2_dry_hpa
        dataset.createVariable("psc2_wet_hpa", "f8", ("lat", "lon"))[:] = output.psc2_wet_hpa
        dataset.createVariable("psc2_dry_hpa", "f8", ("lat", "lon"))[:] = output.psc2_dry_hpa
        dataset.createVariable("delp_dry_hpa", "f8", ("lev", "lat", "lon"))[:] = output.delp_dry_hpa
        dataset.createVariable("specific_humidity_kg_kg", "f8", ("lev", "lat", "lon"))[:] = (
            output.specific_humidity_kg_kg
        )
        dataset.createVariable("temperature_k", "f8", ("lev", "lat", "lon"))[:] = output.temperature_k
    return path


def write_met_airqnt_output(path: str | Path, output: MetAirQntOutput) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nlev, nlat, nlon = output.delp_dry_hpa.shape
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("lev", nlev)
        dataset.createDimension("ilev", nlev + 1)
        dataset.createDimension("lat", nlat)
        dataset.createDimension("lon", nlon)
        dataset.harness = MET_AIRQNT_OUTPUT_VERSION
        dataset.createVariable("ps1_wet_hpa", "f8", ("lat", "lon"))[:] = output.ps1_wet_hpa
        dataset.createVariable("ps2_wet_hpa", "f8", ("lat", "lon"))[:] = output.ps2_wet_hpa
        dataset.createVariable("ps1_dry_hpa", "f8", ("lat", "lon"))[:] = output.ps1_dry_hpa
        dataset.createVariable("ps2_dry_hpa", "f8", ("lat", "lon"))[:] = output.ps2_dry_hpa
        dataset.createVariable("psc2_wet_hpa", "f8", ("lat", "lon"))[:] = output.psc2_wet_hpa
        dataset.createVariable("psc2_dry_hpa", "f8", ("lat", "lon"))[:] = output.psc2_dry_hpa
        dataset.createVariable("wet_pressure_edges_hpa", "f8", ("ilev", "lat", "lon"))[:] = output.wet_pressure_edges_hpa
        dataset.createVariable("wet_pressure_mid_hpa", "f8", ("lev", "lat", "lon"))[:] = output.wet_pressure_mid_hpa
        dataset.createVariable("wet_pressure_thickness_hpa", "f8", ("lev", "lat", "lon"))[:] = (
            output.wet_pressure_thickness_hpa
        )
        dataset.createVariable("dry_partial_pressure_edges_hpa", "f8", ("ilev", "lat", "lon"))[:] = (
            output.dry_partial_pressure_edges_hpa
        )
        dataset.createVariable("dry_partial_pressure_mid_hpa", "f8", ("lev", "lat", "lon"))[:] = (
            output.dry_partial_pressure_mid_hpa
        )
        dataset.createVariable("delp_dry_hpa", "f8", ("lev", "lat", "lon"))[:] = output.delp_dry_hpa
        dataset.createVariable("specific_humidity_kg_kg", "f8", ("lev", "lat", "lon"))[:] = (
            output.specific_humidity_kg_kg
        )
        dataset.createVariable("temperature_k", "f8", ("lev", "lat", "lon"))[:] = output.temperature_k
        dataset.createVariable("water_vapor_vv_dry", "f8", ("lev", "lat", "lon"))[:] = output.water_vapor_vv_dry
        dataset.createVariable("virtual_temperature_k", "f8", ("lev", "lat", "lon"))[:] = output.virtual_temperature_k
        dataset.createVariable("bxheight_m", "f8", ("lev", "lat", "lon"))[:] = output.bxheight_m
        dataset.createVariable("dry_air_mass_kg", "f8", ("lev", "lat", "lon"))[:] = output.dry_air_mass_kg
        dataset.createVariable("air_volume_m3", "f8", ("lev", "lat", "lon"))[:] = output.air_volume_m3
    return path


def compare_dry_pressure_output(
    input_path: str | Path,
    expected_output_path: str | Path,
    *,
    python_output_path: str | Path | None = None,
) -> DryPressureComparison:
    output_path = Path(expected_output_path)
    python_path = Path(python_output_path) if python_output_path is not None else output_path.with_name(f"python_{output_path.name}")
    write_python_dry_pressure_output(input_path, python_path)
    expected = read_dry_pressure_output(output_path)
    actual = read_dry_pressure_output(python_path)
    delp_error = np.abs(actual.delp_dry_hpa - expected.delp_dry_hpa)
    return DryPressureComparison(
        ps1_wet_max_abs_error_hpa=float(np.max(np.abs(actual.ps1_wet_hpa - expected.ps1_wet_hpa))),
        ps2_wet_max_abs_error_hpa=float(np.max(np.abs(actual.ps2_wet_hpa - expected.ps2_wet_hpa))),
        ps1_dry_max_abs_error_hpa=float(np.max(np.abs(actual.ps1_dry_hpa - expected.ps1_dry_hpa))),
        ps2_dry_max_abs_error_hpa=float(np.max(np.abs(actual.ps2_dry_hpa - expected.ps2_dry_hpa))),
        psc2_wet_max_abs_error_hpa=float(np.max(np.abs(actual.psc2_wet_hpa - expected.psc2_wet_hpa))),
        psc2_dry_max_abs_error_hpa=float(np.max(np.abs(actual.psc2_dry_hpa - expected.psc2_dry_hpa))),
        delp_dry_max_abs_error_hpa=float(np.max(delp_error)),
        delp_dry_mean_abs_error_hpa=float(np.mean(delp_error)),
        specific_humidity_max_abs_error=float(
            np.max(np.abs(actual.specific_humidity_kg_kg - expected.specific_humidity_kg_kg))
        ),
        temperature_max_abs_error=float(np.max(np.abs(actual.temperature_k - expected.temperature_k))),
    )


def compare_met_airqnt_output(
    input_path: str | Path,
    expected_output_path: str | Path,
    *,
    python_output_path: str | Path | None = None,
) -> MetAirQntComparison:
    output_path = Path(expected_output_path)
    python_path = Path(python_output_path) if python_output_path is not None else output_path.with_name(f"python_{output_path.name}")
    write_python_met_airqnt_output(input_path, python_path)
    expected = read_met_airqnt_output(output_path)
    actual = read_met_airqnt_output(python_path)
    max_abs: dict[str, float] = {}
    mean_abs: dict[str, float] = {}
    for name in MetAirQntOutput.__dataclass_fields__:
        expected_values = getattr(expected, name)
        actual_values = getattr(actual, name)
        error = np.abs(actual_values - expected_values)
        max_abs[name] = float(np.max(error))
        mean_abs[name] = float(np.mean(error))
    return MetAirQntComparison(max_abs_errors=max_abs, mean_abs_errors=mean_abs)


def format_met_airqnt_comparison(comparison: MetAirQntComparison) -> str:
    rows = ["field,max_abs,mean_abs"]
    for name in comparison.max_abs_errors:
        rows.append(f"{name},{comparison.max_abs_errors[name]:.8e},{comparison.mean_abs_errors[name]:.8e}")
    return "\n".join(rows)


def format_dry_pressure_comparison(comparison: DryPressureComparison) -> str:
    return "\n".join(
        [
            "metric,value",
            f"ps1_wet_max_abs_error_hpa,{comparison.ps1_wet_max_abs_error_hpa:.8e}",
            f"ps2_wet_max_abs_error_hpa,{comparison.ps2_wet_max_abs_error_hpa:.8e}",
            f"ps1_dry_max_abs_error_hpa,{comparison.ps1_dry_max_abs_error_hpa:.8e}",
            f"ps2_dry_max_abs_error_hpa,{comparison.ps2_dry_max_abs_error_hpa:.8e}",
            f"psc2_wet_max_abs_error_hpa,{comparison.psc2_wet_max_abs_error_hpa:.8e}",
            f"psc2_dry_max_abs_error_hpa,{comparison.psc2_dry_max_abs_error_hpa:.8e}",
            f"delp_dry_max_abs_error_hpa,{comparison.delp_dry_max_abs_error_hpa:.8e}",
            f"delp_dry_mean_abs_error_hpa,{comparison.delp_dry_mean_abs_error_hpa:.8e}",
            f"specific_humidity_max_abs_error,{comparison.specific_humidity_max_abs_error:.8e}",
            f"temperature_max_abs_error,{comparison.temperature_max_abs_error:.8e}",
        ]
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
        nlev, nlat, nlon, ntracer = result.tracer_conc.shape
        dataset.createDimension("tracer", ntracer)
        dataset.createDimension("lev", nlev)
        dataset.createDimension("ilev", nlev + 1)
        dataset.createDimension("lat", nlat)
        dataset.createDimension("lon", nlon)
        dataset.harness = VDIFF_OUTPUT_VERSION
        dataset.negative_count_before_clip = int(result.negative_count_before_clip)
        dataset.negative_count_after_clip = int(result.negative_count_after_clip)
        dataset.createVariable("tracer_conc_after", "f8", ("lev", "lat", "lon", "tracer"))[:] = result.tracer_conc
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
    with netCDF4.Dataset(input_path) as dataset:
        initial_tracer = np.asarray(dataset.variables["tracer_conc"][:], dtype=np.float64)
        dry_air_mass = np.asarray(dataset.variables["dry_air_mass_kg"][:], dtype=np.float64)
    common_actual_initial_mass = _tracer_mass_common_basis(initial_tracer, dry_air_mass)
    common_expected_initial_mass = common_actual_initial_mass
    common_actual_final_mass = _tracer_mass_common_basis(actual.tracer_conc_after, dry_air_mass)
    common_expected_final_mass = _tracer_mass_common_basis(expected.tracer_conc_after, dry_air_mass)
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
        common_basis_initial_mass_max_abs_error=float(
            np.max(np.abs(common_actual_initial_mass - common_expected_initial_mass))
        ),
        common_basis_final_mass_max_abs_error=float(
            np.max(np.abs(common_actual_final_mass - common_expected_final_mass))
        ),
        common_basis_mass_change_max_abs_error=float(
            np.max(
                np.abs(
                    (common_actual_final_mass - common_actual_initial_mass)
                    - (common_expected_final_mass - common_expected_initial_mass)
                )
            )
        ),
        reported_initial_mass_max_abs_error=float(
            np.max(np.abs(actual.initial_tracer_mass - expected.initial_tracer_mass))
        ),
        reported_final_mass_max_abs_error=float(np.max(np.abs(actual.final_tracer_mass - expected.final_tracer_mass))),
    )


def read_convection_output(path: str | Path) -> ConvectionOutput:
    with netCDF4.Dataset(path) as dataset:
        if getattr(dataset, "harness", "") != CONVECTION_OUTPUT_VERSION:
            raise ValueError(f"{path} is not a {CONVECTION_OUTPUT_VERSION} file")
        return ConvectionOutput(
            tracer_conc_after=np.asarray(dataset.variables["tracer_conc_after"][:], dtype=np.float64),
            diag14_mass_flux=np.asarray(dataset.variables["diag14_mass_flux"][:], dtype=np.float64),
            negative_count_before=int(getattr(dataset, "negative_count_before")),
            negative_count_after=int(getattr(dataset, "negative_count_after")),
            initial_tracer_mass=np.asarray(dataset.variables["initial_tracer_mass"][:], dtype=np.float64),
            final_tracer_mass=np.asarray(dataset.variables["final_tracer_mass"][:], dtype=np.float64),
            internal_steps=int(getattr(dataset, "internal_steps")),
            internal_dt_s=float(getattr(dataset, "internal_dt_s")),
        )


def write_python_convection_output(input_path: str | Path, output_path: str | Path) -> Path:
    with netCDF4.Dataset(input_path) as dataset:
        if getattr(dataset, "harness", "") != CONVECTION_INPUT_VERSION:
            raise ValueError(f"{input_path} is not a {CONVECTION_INPUT_VERSION} file")
        result = run_cloud_convection_one_step(
            tracer_conc=np.asarray(dataset.variables["tracer_conc"][:], dtype=np.float64),
            cmfmc_kg_m2_s=np.asarray(dataset.variables["cmfmc_kg_m2_s"][:], dtype=np.float64),
            dtrain_kg_m2_s=np.asarray(dataset.variables["dtrain_kg_m2_s"][:], dtype=np.float64),
            dqrcu_kg_kg_s=np.asarray(dataset.variables["dqrcu_kg_kg_s"][:], dtype=np.float64),
            reevapcn_kg_kg_s=np.asarray(dataset.variables["reevapcn_kg_kg_s"][:], dtype=np.float64),
            delp_dry_hpa=np.asarray(dataset.variables["delp_dry_hpa"][:], dtype=np.float64),
            delp_hpa=np.asarray(dataset.variables["delp_hpa"][:], dtype=np.float64),
            area_m2=np.asarray(dataset.variables["area_m2"][:], dtype=np.float64),
            bxheight_m=np.asarray(dataset.variables["bxheight_m"][:], dtype=np.float64),
            pficu_kg_m2_s=np.asarray(dataset.variables["pficu_kg_m2_s"][:], dtype=np.float64),
            pflcu_kg_m2_s=np.asarray(dataset.variables["pflcu_kg_m2_s"][:], dtype=np.float64),
            temperature_k=np.asarray(dataset.variables["temperature_k"][:], dtype=np.float64),
            precccon_mm_day=np.asarray(dataset.variables["precccon_mm_day"][:], dtype=np.float64),
            dt_s=float(dataset.dt_s),
            reconstruct_conv_precip_flux=bool(getattr(dataset, "reconstruct_conv_precip_flux", 0)),
        )
    return write_convection_output(output_path, result)


def write_convection_output(path: str | Path, result) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w") as dataset:
        nlev, nlat, nlon, ntracer = result.tracer_conc.shape
        dataset.createDimension("tracer", ntracer)
        dataset.createDimension("lev", nlev)
        dataset.createDimension("lat", nlat)
        dataset.createDimension("lon", nlon)
        dataset.harness = CONVECTION_OUTPUT_VERSION
        dataset.negative_count_before = int(result.negative_count_before)
        dataset.negative_count_after = int(result.negative_count_after)
        dataset.internal_steps = int(result.internal_steps)
        dataset.internal_dt_s = float(result.internal_dt_s)
        dataset.createVariable("tracer_conc_after", "f8", ("lev", "lat", "lon", "tracer"))[:] = result.tracer_conc
        dataset.createVariable("diag14_mass_flux", "f8", ("lev", "lat", "lon", "tracer"))[:] = (
            result.diag14_mass_flux
        )
        dataset.createVariable("initial_tracer_mass", "f8", ("tracer",))[:] = result.initial_tracer_mass
        dataset.createVariable("final_tracer_mass", "f8", ("tracer",))[:] = result.final_tracer_mass
    return path


def compare_convection_output(
    input_path: str | Path,
    expected_output_path: str | Path,
    *,
    python_output_path: str | Path | None = None,
) -> ConvectionComparison:
    output_path = Path(expected_output_path)
    python_path = Path(python_output_path) if python_output_path is not None else output_path.with_name(f"python_{output_path.name}")
    write_python_convection_output(input_path, python_path)
    expected = read_convection_output(output_path)
    actual = read_convection_output(python_path)
    with netCDF4.Dataset(input_path) as dataset:
        initial_tracer = np.asarray(dataset.variables["tracer_conc"][:], dtype=np.float64)
        common_dry_mass = _convection_common_dry_air_mass(
            np.asarray(dataset.variables["delp_dry_hpa"][:], dtype=np.float64),
            np.asarray(dataset.variables["area_m2"][:], dtype=np.float64),
        )
    common_actual_initial_mass = _tracer_mass_common_basis(initial_tracer, common_dry_mass)
    common_expected_initial_mass = common_actual_initial_mass
    common_actual_final_mass = _tracer_mass_common_basis(actual.tracer_conc_after, common_dry_mass)
    common_expected_final_mass = _tracer_mass_common_basis(expected.tracer_conc_after, common_dry_mass)
    tracer_error = np.abs(actual.tracer_conc_after - expected.tracer_conc_after)
    diag14_error = np.abs(actual.diag14_mass_flux - expected.diag14_mass_flux)
    top_error_index = np.unravel_index(int(np.argmax(tracer_error)), tracer_error.shape)
    return ConvectionComparison(
        tracer_max_abs_error=float(np.max(tracer_error)),
        tracer_mean_abs_error=float(np.mean(tracer_error)),
        diag14_max_abs_error=float(np.max(diag14_error)),
        diag14_mean_abs_error=float(np.mean(diag14_error)),
        negative_count_before_expected=expected.negative_count_before,
        negative_count_before_actual=actual.negative_count_before,
        negative_count_after_expected=expected.negative_count_after,
        negative_count_after_actual=actual.negative_count_after,
        common_basis_initial_mass_max_abs_error=float(
            np.max(np.abs(common_actual_initial_mass - common_expected_initial_mass))
        ),
        common_basis_final_mass_max_abs_error=float(
            np.max(np.abs(common_actual_final_mass - common_expected_final_mass))
        ),
        common_basis_mass_change_max_abs_error=float(
            np.max(
                np.abs(
                    (common_actual_final_mass - common_actual_initial_mass)
                    - (common_expected_final_mass - common_expected_initial_mass)
                )
            )
        ),
        common_basis_python_mass_change_max_abs=float(
            np.max(np.abs(common_actual_final_mass - common_actual_initial_mass))
        ),
        common_basis_oracle_mass_change_max_abs=float(
            np.max(np.abs(common_expected_final_mass - common_expected_initial_mass))
        ),
        reported_initial_mass_max_abs_error=float(
            np.max(np.abs(actual.initial_tracer_mass - expected.initial_tracer_mass))
        ),
        reported_final_mass_max_abs_error=float(
            np.max(np.abs(actual.final_tracer_mass - expected.final_tracer_mass))
        ),
        reported_python_mass_change_max_abs=float(np.max(np.abs(actual.final_tracer_mass - actual.initial_tracer_mass))),
        reported_oracle_mass_change_max_abs=float(
            np.max(np.abs(expected.final_tracer_mass - expected.initial_tracer_mass))
        ),
        top_error_level=int(top_error_index[0]),
        top_error_lat=int(top_error_index[1]),
        top_error_lon=int(top_error_index[2]),
        top_error_tracer=int(top_error_index[3]),
        internal_steps_expected=expected.internal_steps,
        internal_steps_actual=actual.internal_steps,
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
    nlev, nlat, nlon, ntracer = trace.tracer_conc_after.shape
    with netCDF4.Dataset(output, "w") as dataset:
        dataset.harness = TPCORE_TRACE_VERSION
        dataset.source_input = str(input_path)
        dataset.dt_s = fixture.dt_s
        dataset.createDimension("tracer", ntracer)
        dataset.createDimension("lev", nlev)
        dataset.createDimension("lat", nlat)
        dataset.createDimension("lon", nlon)
        dataset.createVariable("q_after_pole_average", "f8", ("lev", "lat", "lon", "tracer"))[:] = (
            trace.q_after_pole_average
        )
        dataset.createVariable("dq_after_init_hpa", "f8", ("lev", "lat", "lon", "tracer"))[:] = trace.dq_after_init
        dataset.createVariable("q_after_cross_terms", "f8", ("lev", "lat", "lon", "tracer"))[:] = (
            trace.q_after_cross_terms
        )
        dataset.createVariable("dq_after_xtp_hpa", "f8", ("lev", "lat", "lon", "tracer"))[:] = trace.dq_after_xtp
        dataset.createVariable("dq_after_ytp_hpa", "f8", ("lev", "lat", "lon", "tracer"))[:] = trace.dq_after_ytp
        dataset.createVariable("dq_after_fzppm_hpa", "f8", ("lev", "lat", "lon", "tracer"))[:] = trace.dq_after_fzppm
        dataset.createVariable("dq_after_fill_hpa", "f8", ("lev", "lat", "lon", "tracer"))[:] = trace.dq_after_fill
        dataset.createVariable("tracer_conc_after", "f8", ("lev", "lat", "lon", "tracer"))[:] = (
            trace.tracer_conc_after
        )
        dataset.createVariable("cx", "f8", ("lev", "lat", "lon"))[:] = setup.cx
        dataset.createVariable("cy", "f8", ("lev", "lat", "lon"))[:] = setup.cy
        dataset.createVariable("vertical_mass_flux_hpa", "f8", ("lev", "lat", "lon"))[:] = (
            setup.vertical_mass_flux_hpa
        )
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
    x_error = np.abs(output.xmass_hpa - expected_x[::-1])
    y_error = np.abs(output.ymass_hpa - expected_y[::-1])
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
    rows.extend(_top_axis_rows("level", error, axis=0, top_n=8))
    rows.extend(_top_axis_rows("latitude", error, axis=1, top_n=8))
    rows.extend(_top_axis_rows("longitude", error, axis=2, top_n=8))
    rows.extend(_bin_rows("abs_cx", error, np.abs(setup.cx), (0.0, 0.1, 0.5, 1.0, np.inf)))
    rows.extend(_bin_rows("abs_cy", error, np.abs(setup.cy), (0.0, 0.05, 0.1, 0.5, 1.0, np.inf)))
    rows.extend(
        _bin_rows(
            "abs_wz_hpa",
            error,
            np.abs(setup.vertical_mass_flux_hpa),
            (0.0, 0.01, 0.1, 1.0, 10.0, np.inf),
        )
    )
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

    if fixture_id in {
        BASE_INITIAL_TRANSPORT_CHAIN_FIXTURE_ID,
        BASE_INITIAL_VDIFF_AFTER_TPCORE_FIXTURE_ID,
        BASE_INITIAL_CONVECTION_FULLGRID_FIXTURE_ID,
    }:
        raise ValueError("TPCORE trace generation is not defined for this oracle fixture")
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

    if fixture_id in {
        BASE_INITIAL_TRANSPORT_CHAIN_FIXTURE_ID,
        BASE_INITIAL_VDIFF_AFTER_TPCORE_FIXTURE_ID,
        BASE_INITIAL_CONVECTION_FULLGRID_FIXTURE_ID,
    }:
        raise ValueError("TPCORE trace comparison is not defined for this oracle fixture")
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
    extra = f"lev={top[0]} lat={top[1]} lon={top[2]} tracer={top[3]}"
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
    value_4d = values[:, :, :, np.newaxis] if values.ndim == 3 else values
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
    for axis in (0, 1, 2):
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


def run_operator_harness(executable: str | Path, input_path: str | Path, output_path: str | Path) -> None:
    executable = Path(executable)
    if not executable.exists():
        raise FileNotFoundError(f"GEOS-Chem harness executable not found: {executable}")
    subprocess.run([str(executable), str(input_path), str(output_path)], check=True)


def write_history_harness_run_directory(
    work_dir: str | Path,
    *,
    ntracer: int = 2,
    frequency: str = "00000000 030000",
    duration: str = "00000001 000000",
    acc_interval: str | None = None,
) -> Path:
    if ntracer < 1:
        raise ValueError("ntracer must be positive")
    run_dir = Path(work_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "OutputDir").mkdir(exist_ok=True)
    (run_dir / "HISTORY.rc").write_text(
        _history_harness_history_rc(frequency=frequency, duration=duration, acc_interval=acc_interval),
        encoding="utf-8",
    )
    (run_dir / "species_database.yml").write_text(_history_harness_species_database(ntracer), encoding="utf-8")
    (run_dir / "geoschem_config.yml").write_text(_history_harness_geoschem_config(ntracer), encoding="utf-8")
    return run_dir


def run_history_harness(
    executable: str | Path,
    work_dir: str | Path,
    *,
    ntracer: int = 2,
    nsteps: int = 144,
    dt_s: int = 600,
    frequency: str = "00000000 030000",
    duration: str = "00000001 000000",
    acc_interval: str | None = None,
) -> Path:
    executable_path = Path(executable)
    if not executable_path.exists():
        raise FileNotFoundError(
            f"GEOS-Chem HISTORY harness executable not found: {executable_path}. "
            "Build tools/gc_harness/history_harness.F90 first."
        )
    run_dir = write_history_harness_run_directory(
        work_dir,
        ntracer=ntracer,
        frequency=frequency,
        duration=duration,
        acc_interval=acc_interval,
    )
    output_path = run_dir / HISTORY_HARNESS_OUTPUT_NAME
    if output_path.exists():
        output_path.unlink()
    subprocess.run(
        [str(executable_path.resolve()), "HISTORY.rc", "species_database.yml", str(nsteps), str(int(dt_s))],
        cwd=run_dir,
        check=True,
    )
    if not output_path.exists():
        raise FileNotFoundError(f"HISTORY harness did not produce expected output {output_path}")
    return output_path


def compare_history_harness_output(
    output_path: str | Path,
    *,
    ntracer: int = 2,
    nsteps: int = 144,
    dt_s: int = 600,
    frequency_s: int = 10800,
    acc_interval_s: int | None = None,
) -> HistoryHarnessComparison:
    path = Path(output_path)
    update_s = int(acc_interval_s or dt_s)
    if frequency_s % update_s != 0:
        raise ValueError("frequency_s must be an integer multiple of the update interval")
    if update_s % dt_s != 0:
        raise ValueError("update interval must be an integer multiple of dt_s")
    steps_per_frequency = frequency_s // dt_s
    update_stride = update_s // dt_s
    expected_records = nsteps // steps_per_frequency
    max_abs = 0.0
    first_expected = np.nan
    first_actual = np.nan
    with netCDF4.Dataset(path) as dataset:
        times = np.asarray(dataset.variables["time"][:], dtype=np.float64)
        expected_times = np.arange(expected_records, dtype=np.float64) * (frequency_s / 60.0)
        if times.size != expected_records:
            raise ValueError(f"expected {expected_records} time records, found {times.size}")
        max_time = float(np.max(np.abs(times - expected_times))) if times.size else 0.0
        for tracer_index in range(1, ntracer + 1):
            name = f"SpeciesConcVV_hist_{tracer_index:03d}"
            if name not in dataset.variables:
                raise ValueError(f"missing HISTORY output variable {name}")
            values = np.asarray(dataset.variables[name][:], dtype=np.float64)
            expected_time = _history_harness_expected_time_values(
                tracer_index=tracer_index,
                n_records=expected_records,
                steps_per_frequency=steps_per_frequency,
                update_stride=update_stride,
            )
            expected = expected_time[:, None, None, None] + _history_harness_geos_spatial_offsets(*values.shape[1:4])[
                None, :, :, :
            ]
            max_abs = max(max_abs, float(np.max(np.abs(values - expected))))
            if tracer_index == 1:
                first_expected = float(expected[0, 0, 0, 0])
                first_actual = float(values[0, 0, 0, 0])
    return HistoryHarnessComparison(
        output_path=path,
        n_records=expected_records,
        n_tracers=ntracer,
        max_abs_error=max_abs,
        max_time_error_min=max_time,
        first_record_expected=first_expected,
        first_record_actual=first_actual,
        boundary_included_in_previous=abs(first_actual - first_expected) < 1.0e-4,
    )


def _history_harness_expected_time_values(
    *,
    tracer_index: int,
    n_records: int,
    steps_per_frequency: int,
    update_stride: int,
) -> np.ndarray:
    expected = np.empty(n_records, dtype=np.float64)
    for record in range(n_records):
        start = record * steps_per_frequency + update_stride
        stop = (record + 1) * steps_per_frequency
        steps = np.arange(start, stop + 1, update_stride, dtype=np.float64)
        expected[record] = float(tracer_index * 1000) + float(np.mean(steps))
    return expected


def _history_harness_geos_spatial_offsets(nlev: int, nlat: int, nlon: int) -> np.ndarray:
    lev = np.arange(1, nlev + 1, dtype=np.float64)[:, None, None]
    lat = np.arange(1, nlat + 1, dtype=np.float64)[None, :, None]
    lon = np.arange(1, nlon + 1, dtype=np.float64)[None, None, :]
    return lev * 0.125 + lat * 0.01 + lon * 0.001


def _history_harness_history_rc(*, frequency: str, duration: str, acc_interval: str | None) -> str:
    acc_line = f"SpeciesConcThreeHourly.acc_interval: {acc_interval},\n" if acc_interval else ""
    return (
        "EXPID: OutputDir/GEOSChem\n\n"
        "COLLECTIONS: 'SpeciesConcThreeHourly',\n"
        "::\n"
        "SpeciesConcThreeHourly.template: '%y4%m2%d2_%h2%n2z.nc4',\n"
        "SpeciesConcThreeHourly.format: 'CFIO',\n"
        f"SpeciesConcThreeHourly.frequency: {frequency},\n"
        f"SpeciesConcThreeHourly.duration: {duration},\n"
        f"{acc_line}"
        "SpeciesConcThreeHourly.mode: 'time-averaged',\n"
        "SpeciesConcThreeHourly.fields: 'SpeciesConcVV_?ADV?', 'GIGCchem',\n"
        "::\n"
    )


def _history_interval_seconds(value: str) -> int:
    pieces = value.split()
    if len(pieces) != 2:
        raise ValueError(f"invalid HISTORY interval {value!r}")
    date, clock = pieces
    if len(date) != 8 or len(clock) != 6 or not date.isdigit() or not clock.isdigit():
        raise ValueError(f"invalid HISTORY interval {value!r}")
    days = int(date[6:8])
    hours = int(clock[0:2])
    minutes = int(clock[2:4])
    seconds = int(clock[4:6])
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _history_harness_species_database(ntracer: int) -> str:
    chunks = []
    for tracer_index in range(1, ntracer + 1):
        name = f"hist_{tracer_index:03d}"
        chunks.append(
            f"{name}:\n"
            f"  Formula: {name}\n"
            "  Is_Gas: true\n"
            "  MW_g: 28.97\n"
            "  Src_Mode: HEMCO\n"
            "  Is_Tracer: true\n"
            "  Background_VV: 0.0\n"
            f"  FullName: HISTORY harness tracer {tracer_index}\n"
        )
    return "\n".join(chunks)


def _history_harness_geoschem_config(ntracer: int) -> str:
    species = ", ".join(f"hist_{tracer_index:03d}" for tracer_index in range(1, ntracer + 1))
    return (
        "simulation:\n"
        "  name: TransportTracers\n"
        "  debug_printout: false\n"
        "operations:\n"
        "  dry_deposition:\n"
        "    diag_alt_above_sfc_in_m: 10\n"
        "  rrtmg_rad_transfer_model:\n"
        "    aod_wavelengths_in_nm: [550]\n"
        "  transport:\n"
        f"    transported_species: {species}\n"
    )


def history_harness_scenario_config(scenario: str) -> dict[str, object]:
    if scenario == "default":
        return {
            "ntracer": 2,
            "nsteps": 144,
            "dt_s": 600,
            "frequency": "00000000 030000",
            "duration": "00000001 000000",
        }
    if scenario == "six_hour_groups":
        return {
            "ntracer": 3,
            "nsteps": 36,
            "dt_s": 600,
            "frequency": "00000000 010000",
            "duration": "00000000 060000",
        }
    raise ValueError(f"unknown HISTORY fixture scenario {scenario!r}; expected one of {HISTORY_FIXTURE_SCENARIOS}")


def generate_history_harness_fixture(
    scenario: str,
    fixture_dir: str | Path,
    *,
    executable: str | Path = Path("tools/gc_harness/build/history_harness"),
) -> Path:
    config = history_harness_scenario_config(scenario)
    output_path = run_history_harness(
        executable,
        fixture_dir,
        ntracer=int(config["ntracer"]),
        nsteps=int(config["nsteps"]),
        dt_s=int(config["dt_s"]),
        frequency=str(config["frequency"]),
        duration=str(config["duration"]),
    )
    _write_history_harness_fixture_metadata(Path(fixture_dir), scenario, output_path, config)
    return output_path


def write_wombat_history_harness_output(
    output_root: str | Path,
    template_path: str | Path,
    *,
    ntracer: int,
    nsteps: int,
    dt_s: int,
    frequency: str,
    duration: str,
) -> Path:
    root = Path(output_root)
    template = Path(template_path)
    with netCDF4.Dataset(template) as dataset:
        nlev = len(dataset.dimensions["lev"])
        nlat = len(dataset.dimensions["lat"])
        nlon = len(dataset.dimensions["lon"])
    start = datetime(2014, 9, 1)
    manager = HistoryOutputManager(
        root=root,
        template_path=template,
        expid="OutputDir/GEOSChem",
        collections=(
            OutputCollectionConfig(
                name="SpeciesConcThreeHourly",
                filename=None,
                template="%y4%m2%d2_%h2%n2z.nc4",
                frequency=parse_history_interval(frequency),
                duration=parse_history_interval(duration),
                mode="time-averaged",
                fields=("SpeciesConcVV_?ADV?",),
                storage=OutputStorageConfig(dtype="float32"),
            ),
        ),
        start=start,
    )
    delp = np.ones((1, nlev, nlat, nlon), dtype=np.float64)
    forcing = SimpleNamespace(
        surface_pressure_pa=np.full((1, nlat, nlon), 101000.0, dtype=np.float64),
        specific_humidity_kg_kg=np.zeros((1, nlev, nlat, nlon), dtype=np.float64),
        temperature_k=np.zeros((1, nlev, nlat, nlon), dtype=np.float64),
    )
    for step in range(1, nsteps + 1):
        timestamp = start + timedelta(seconds=step * dt_s)
        manager.record_step(
            OutputSnapshot(
                timestamp=timestamp,
                state=_history_harness_canonical_field(step, ntracer, nlev, nlat, nlon),
                delp_dry_hpa=delp,
                forcing=forcing,  # type: ignore[arg-type]
            )
        )
    manager.close()
    output_path = root / HISTORY_HARNESS_OUTPUT_NAME
    if not output_path.exists():
        raise FileNotFoundError(f"Wombat HISTORY replay did not produce expected output {output_path}")
    return output_path


def compare_history_harness_to_wombat(
    reference_path: str | Path,
    work_dir: str | Path,
    *,
    ntracer: int,
    nsteps: int,
    dt_s: int,
    frequency: str,
    duration: str,
) -> HistoryHarnessWombatComparison:
    reference = Path(reference_path)
    wombat = write_wombat_history_harness_output(
        work_dir,
        reference,
        ntracer=ntracer,
        nsteps=nsteps,
        dt_s=dt_s,
        frequency=frequency,
        duration=duration,
    )
    max_abs = 0.0
    max_time = 0.0
    max_coord = 0.0
    with netCDF4.Dataset(reference) as expected, netCDF4.Dataset(wombat) as actual:
        expected_time = np.asarray(expected.variables["time"][:], dtype=np.float64)
        actual_time = np.asarray(actual.variables["time"][:], dtype=np.float64)
        max_time = float(np.max(np.abs(actual_time - expected_time))) if expected_time.size else 0.0
        for name in ("lev", "ilev", "lat", "lon", "lat_bnds", "lon_bnds", "hyam", "hybm", "hyai", "hybi", "AREA"):
            if name in expected.variables and name in actual.variables:
                max_coord = max(
                    max_coord,
                    float(
                        np.max(
                            np.abs(
                                np.asarray(actual.variables[name][:], dtype=np.float64)
                                - np.asarray(expected.variables[name][:], dtype=np.float64)
                            )
                        )
                    ),
                )
        for tracer_index in range(1, ntracer + 1):
            name = f"SpeciesConcVV_hist_{tracer_index:03d}"
            if name not in expected.variables:
                raise ValueError(f"missing reference HISTORY variable {name}")
            if name not in actual.variables:
                raise ValueError(f"missing Wombat HISTORY variable {name}")
            max_abs = max(
                max_abs,
                float(
                    np.max(
                        np.abs(
                            np.asarray(actual.variables[name][:], dtype=np.float64)
                            - np.asarray(expected.variables[name][:], dtype=np.float64)
                        )
                    )
                ),
            )
    return HistoryHarnessWombatComparison(
        reference_path=reference,
        wombat_path=wombat,
        n_records=int(expected_time.size),
        n_tracers=ntracer,
        max_abs_error=max_abs,
        max_time_error_min=max_time,
        max_coord_error=max_coord,
    )


def _history_harness_canonical_field(step: int, ntracer: int, nlev: int, nlat: int, nlon: int) -> TracerField:
    geos_offsets = _history_harness_geos_spatial_offsets(nlev, nlat, nlon)
    data = np.empty((1, nlev, nlat, nlon, ntracer), dtype=np.float64)
    for tracer_index in range(1, ntracer + 1):
        geos_values = float(tracer_index * 1000 + step) + geos_offsets
        data[0, :, :, :, tracer_index - 1] = geos_values[::-1, :, :]
    names = tuple(f"hist_{tracer_index:03d}" for tracer_index in range(1, ntracer + 1))
    return TracerField(
        names=names,
        data=data,
        units=tuple("mol mol-1 dry" for _ in names),
        coords={},
    )


def _write_history_harness_fixture_metadata(
    fixture_dir: Path,
    scenario: str,
    output_path: Path,
    config: dict[str, object],
) -> None:
    metadata = {
        "fixture": f"history_{scenario}_v1",
        "scenario": scenario,
        "version": HISTORY_HARNESS_VERSION,
        "output": _large_oracle_file_record(output_path),
        "config": config,
        "source": {
            "build_command": "tools/gc_harness/build_history_harness.sh",
            "fixture_command": (
                "python -m wombat_transport.gc_harness history-fixture-generate "
                f"{scenario} {fixture_dir}"
            ),
        },
    }
    (fixture_dir / SNAPSHOT_METADATA_NAME).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_large_oracle_definition(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if "fixture_id" not in data:
        raise ValueError(f"{path} is missing fixture_id")
    if "files" not in data:
        raise ValueError(f"{path} is missing files")
    return data


def _large_oracle_source(paths: LargeOracleFixturePaths) -> dict[str, object]:
    """Return canonical tracked provenance, independent of a stale local cache manifest."""

    definition = _load_large_oracle_definition(paths.definition_path)
    source = definition.get("source", {})
    if not isinstance(source, dict):
        raise ValueError(f"{paths.definition_path} source must be a mapping")
    return dict(source)


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


def _write_generated_transport_chain_manifest(
    paths: LargeOracleFixturePaths,
    *,
    definition: dict[str, object],
    run_config: Path,
    tpcore_executable: Path,
    vdiff_executable: Path,
    convection_executable: Path,
    repo_root: Path,
) -> None:
    setup = _setup_tpcore_from_input(paths.input_path)
    report = analyze_tpcore_branches(setup)
    manifest = {
        "fixture_id": paths.fixture_id,
        "description": definition.get("description"),
        "definition_file": str(paths.definition_path),
        "input_harness": TRANSPORT_INPUT_VERSION,
        "output_harness": TRANSPORT_CHAIN_OUTPUT_VERSION,
        "files": [
            _large_oracle_file_record(paths.input_path),
            _large_oracle_file_record(paths.output_path),
        ],
        "source": {
            **dict(definition.get("source", {})),
            "run_config": str(run_config),
        },
        "executables": {
            "tpcore": str(tpcore_executable),
            "vdiff": str(vdiff_executable),
            "convection": str(convection_executable),
        },
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


def _write_generated_operator_oracle_manifest(
    paths: LargeOracleFixturePaths,
    *,
    definition: dict[str, object],
    run_config: Path,
    input_harness: str,
    output_harness: str,
    executables: dict[str, str],
    repo_root: Path,
    tpcore_input_path: Path,
) -> None:
    setup = _setup_tpcore_from_input(tpcore_input_path)
    report = analyze_tpcore_branches(setup)
    manifest = {
        "fixture_id": paths.fixture_id,
        "description": definition.get("description"),
        "definition_file": str(paths.definition_path),
        "input_harness": input_harness,
        "output_harness": output_harness,
        "files": [
            _large_oracle_file_record(paths.input_path),
            _large_oracle_file_record(paths.output_path),
        ],
        "source": {
            **dict(definition.get("source", {})),
            "run_config": str(run_config),
        },
        "executables": executables,
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
    pressure_branch_gap = _tpcore_pressure_branch_gap(input_path)
    return {
        "snapshot": TPCORE_SNAPSHOT_VERSION,
        "input_harness": TRANSPORT_INPUT_VERSION,
        "output_harness": TRANSPORT_OUTPUT_VERSION,
        "input_file": input_path.name,
        "output_file": output_path.name,
        "shape": {"tracer": ntracer, "lev": nlev, "lat": nlat, "lon": nlon},
        "dt_s": dt_s,
        "pressure_branch_gap_max_hpa": pressure_branch_gap,
        "executable": str(executable),
        "gcclassic_head": _git_head(repo_root / "GCClassic"),
    }


def _tpcore_pressure_branch_gap(input_path: str | Path) -> float:
    """Return max |raw-p2 branch - PJC-adjusted pressure branch| for TPCORE."""

    fixture = read_tpcore_input(input_path)
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
    rel_area, geofac, geofac_pc, _cose, _cosp = _pjc_horizontal_geometry(fixture.area_m2, fixture.lat_deg)
    p1 = fixture.p1_hpa.copy()
    _average_poles_in_place(p1, rel_area)
    ak = fixture.hyai_hpa[::-1]
    bk = fixture.hybi[::-1]
    dap = ak[1:] - ak[:-1]
    dbk = bk[1:] - bk[:-1]
    dpi = _calc_divergence(setup.xmass_hpa, setup.ymass_hpa, geofac, geofac_pc)
    pjc_adjusted_pressure = p1 + np.sum(dpi, axis=0)
    pjc_adjusted_branch = dap[:, np.newaxis, np.newaxis] + dbk[:, np.newaxis, np.newaxis] * (
        pjc_adjusted_pressure[np.newaxis, :, :]
    )
    return float(np.max(np.abs(setup.delp2_hpa - pjc_adjusted_branch)))


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
            f"common_basis_initial_mass_max_abs_error,{comparison.common_basis_initial_mass_max_abs_error:.8e}",
            f"common_basis_final_mass_max_abs_error,{comparison.common_basis_final_mass_max_abs_error:.8e}",
            f"common_basis_mass_change_max_abs_error,{comparison.common_basis_mass_change_max_abs_error:.8e}",
            f"reported_initial_mass_max_abs_error,{comparison.reported_initial_mass_max_abs_error:.8e}",
            f"reported_final_mass_max_abs_error,{comparison.reported_final_mass_max_abs_error:.8e}",
        ]
    )


def format_convection_comparison(comparison: ConvectionComparison) -> str:
    return "\n".join(
        [
            "metric,value",
            f"tracer_max_abs_error,{comparison.tracer_max_abs_error:.8e}",
            f"tracer_mean_abs_error,{comparison.tracer_mean_abs_error:.8e}",
            f"diag14_max_abs_error,{comparison.diag14_max_abs_error:.8e}",
            f"diag14_mean_abs_error,{comparison.diag14_mean_abs_error:.8e}",
            f"negative_count_before_expected,{comparison.negative_count_before_expected}",
            f"negative_count_before_actual,{comparison.negative_count_before_actual}",
            f"negative_count_after_expected,{comparison.negative_count_after_expected}",
            f"negative_count_after_actual,{comparison.negative_count_after_actual}",
            f"common_basis_initial_mass_max_abs_error,{comparison.common_basis_initial_mass_max_abs_error:.8e}",
            f"common_basis_final_mass_max_abs_error,{comparison.common_basis_final_mass_max_abs_error:.8e}",
            f"common_basis_mass_change_max_abs_error,{comparison.common_basis_mass_change_max_abs_error:.8e}",
            f"common_basis_python_mass_change_max_abs,{comparison.common_basis_python_mass_change_max_abs:.8e}",
            f"common_basis_oracle_mass_change_max_abs,{comparison.common_basis_oracle_mass_change_max_abs:.8e}",
            f"reported_initial_mass_max_abs_error,{comparison.reported_initial_mass_max_abs_error:.8e}",
            f"reported_final_mass_max_abs_error,{comparison.reported_final_mass_max_abs_error:.8e}",
            f"reported_python_mass_change_max_abs,{comparison.reported_python_mass_change_max_abs:.8e}",
            f"reported_oracle_mass_change_max_abs,{comparison.reported_oracle_mass_change_max_abs:.8e}",
            f"top_error_index,{comparison.top_error_tracer}:{comparison.top_error_level}:{comparison.top_error_lat}:{comparison.top_error_lon}",
            f"internal_steps_expected,{comparison.internal_steps_expected}",
            f"internal_steps_actual,{comparison.internal_steps_actual}",
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


def format_transport_chain_comparison(comparison: TransportChainComparison) -> str:
    return "\n".join(
        [
            "metric,value",
            f"tracer_max_abs_error,{comparison.tracer_max_abs_error:.8e}",
            f"tracer_mean_abs_error,{comparison.tracer_mean_abs_error:.8e}",
            f"negative_count_expected,{comparison.negative_count_expected}",
            f"negative_count_actual,{comparison.negative_count_actual}",
            f"common_basis_initial_mass_max_abs_error,{comparison.common_basis_initial_mass_max_abs_error:.8e}",
            f"common_basis_final_mass_max_abs_error,{comparison.common_basis_final_mass_max_abs_error:.8e}",
            f"common_basis_mass_change_max_abs_error,{comparison.common_basis_mass_change_max_abs_error:.8e}",
            f"common_basis_python_mass_change_max_abs,{comparison.common_basis_python_mass_change_max_abs:.8e}",
            f"common_basis_oracle_mass_change_max_abs,{comparison.common_basis_oracle_mass_change_max_abs:.8e}",
            f"common_basis_tpcore_stage_mass_change_max_abs,{comparison.common_basis_tpcore_stage_mass_change_max_abs:.8e}",
            f"common_basis_vdiff_stage_mass_change_max_abs,{comparison.common_basis_vdiff_stage_mass_change_max_abs:.8e}",
            f"common_basis_convection_stage_mass_change_max_abs,{comparison.common_basis_convection_stage_mass_change_max_abs:.8e}",
            f"reported_final_mass_max_abs_error,{comparison.reported_final_mass_max_abs_error:.8e}",
            f"reported_python_mass_change_max_abs,{comparison.reported_python_mass_change_max_abs:.8e}",
            f"reported_oracle_mass_change_max_abs,{comparison.reported_oracle_mass_change_max_abs:.8e}",
            f"reported_tpcore_stage_mass_change_max_abs,{comparison.reported_tpcore_stage_mass_change_max_abs:.8e}",
            f"reported_vdiff_stage_mass_change_max_abs,{comparison.reported_vdiff_stage_mass_change_max_abs:.8e}",
            f"reported_convection_stage_mass_change_max_abs,{comparison.reported_convection_stage_mass_change_max_abs:.8e}",
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


def format_history_harness_comparison(comparison: HistoryHarnessComparison) -> str:
    return "\n".join(
        [
            "metric,value",
            f"output_path,{comparison.output_path}",
            f"n_records,{comparison.n_records}",
            f"n_tracers,{comparison.n_tracers}",
            f"max_abs_error,{comparison.max_abs_error:.8e}",
            f"max_time_error_min,{comparison.max_time_error_min:.8e}",
            f"first_record_expected,{comparison.first_record_expected:.8e}",
            f"first_record_actual,{comparison.first_record_actual:.8e}",
            f"boundary_included_in_previous,{comparison.boundary_included_in_previous}",
        ]
    )


def format_history_harness_wombat_comparison(comparison: HistoryHarnessWombatComparison) -> str:
    return "\n".join(
        [
            "metric,value",
            f"reference_path,{comparison.reference_path}",
            f"wombat_path,{comparison.wombat_path}",
            f"n_records,{comparison.n_records}",
            f"n_tracers,{comparison.n_tracers}",
            f"max_abs_error,{comparison.max_abs_error:.8e}",
            f"max_time_error_min,{comparison.max_time_error_min:.8e}",
            f"max_coord_error,{comparison.max_coord_error:.8e}",
        ]
    )


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
    write_vdiff_parser.add_argument("--scenario", choices=VDIFF_SCENARIOS, default="zero_surface_flux")

    python_vdiff_parser = subparsers.add_parser("python-vdiff-output")
    python_vdiff_parser.add_argument("input", type=Path)
    python_vdiff_parser.add_argument("output", type=Path)

    compare_vdiff_parser = subparsers.add_parser("compare-vdiff-output")
    compare_vdiff_parser.add_argument("input", type=Path)
    compare_vdiff_parser.add_argument("output", type=Path)

    write_convection_parser = subparsers.add_parser("write-synthetic-convection-input")
    write_convection_parser.add_argument("output", type=Path)
    write_convection_parser.add_argument("--dt-s", type=float, default=600.0)
    write_convection_parser.add_argument("--ntracer", type=int, default=2)
    write_convection_parser.add_argument("--scenario", choices=CONVECTION_SCENARIOS, default="active_cloud")

    write_real_convection_parser = subparsers.add_parser("write-real-convection-input")
    write_real_convection_parser.add_argument("output", type=Path)
    write_real_convection_parser.add_argument("--run-config", type=Path, default=Path("validation_runs/cases/residual_24tracer_emissions_1day_2x25/wombat/main/run.yml"))
    write_real_convection_parser.add_argument("--mode", choices=REAL_CONVECTION_MODES, default="sampled-columns")
    write_real_convection_parser.add_argument("--time-index", type=int, default=None)
    write_real_convection_parser.add_argument("--tracer-time-index", type=int, default=0)
    write_real_convection_parser.add_argument("--dt-s", type=float, default=None)
    write_real_convection_parser.add_argument("--max-tracers", type=int, default=None)
    write_real_convection_parser.add_argument("--active-columns", type=int, default=6)

    python_convection_parser = subparsers.add_parser("python-convection-output")
    python_convection_parser.add_argument("input", type=Path)
    python_convection_parser.add_argument("output", type=Path)

    compare_convection_parser = subparsers.add_parser("compare-convection-output")
    compare_convection_parser.add_argument("input", type=Path)
    compare_convection_parser.add_argument("output", type=Path)

    write_history_parser = subparsers.add_parser("write-history-harness")
    write_history_parser.add_argument("work_dir", type=Path)
    write_history_parser.add_argument("--ntracer", type=int, default=2)
    write_history_parser.add_argument("--frequency", default="00000000 030000")
    write_history_parser.add_argument("--duration", default="00000001 000000")
    write_history_parser.add_argument("--acc-interval", default=None)

    history_parser = subparsers.add_parser("history-harness")
    history_parser.add_argument("work_dir", type=Path)
    history_parser.add_argument("--executable", type=Path, default=Path("tools/gc_harness/build/history_harness"))
    history_parser.add_argument("--ntracer", type=int, default=2)
    history_parser.add_argument("--nsteps", type=int, default=144)
    history_parser.add_argument("--dt-s", type=int, default=600)
    history_parser.add_argument("--frequency", default="00000000 030000")
    history_parser.add_argument("--duration", default="00000001 000000")
    history_parser.add_argument("--acc-interval", default=None)

    compare_history_parser = subparsers.add_parser("compare-history-harness-output")
    compare_history_parser.add_argument("output", type=Path)
    compare_history_parser.add_argument("--ntracer", type=int, default=2)
    compare_history_parser.add_argument("--nsteps", type=int, default=144)
    compare_history_parser.add_argument("--dt-s", type=int, default=600)
    compare_history_parser.add_argument("--frequency-s", type=int, default=10800)
    compare_history_parser.add_argument("--acc-interval-s", type=int, default=None)

    history_fixture_parser = subparsers.add_parser("history-fixture-generate")
    history_fixture_parser.add_argument("scenario", choices=HISTORY_FIXTURE_SCENARIOS)
    history_fixture_parser.add_argument("fixture_dir", type=Path)
    history_fixture_parser.add_argument("--executable", type=Path, default=Path("tools/gc_harness/build/history_harness"))

    compare_history_wombat_parser = subparsers.add_parser("compare-history-fixture-wombat")
    compare_history_wombat_parser.add_argument("reference", type=Path)
    compare_history_wombat_parser.add_argument("work_dir", type=Path)
    compare_history_wombat_parser.add_argument("--scenario", choices=HISTORY_FIXTURE_SCENARIOS, default="default")

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

    handoff_compare_oracle_parser = subparsers.add_parser("oracle-fixture-handoff-compare")
    handoff_compare_oracle_parser.add_argument("fixture_id", choices=(BASE_INITIAL_TRANSPORT_CHAIN_FIXTURE_ID,))
    handoff_compare_oracle_parser.add_argument("--cache-dir", type=Path, default=Path("oracle_data"))
    handoff_compare_oracle_parser.add_argument("--manifest-dir", type=Path, default=None)

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
        path = write_synthetic_vdiff_input(args.output, dt_s=args.dt_s, ntracer=args.ntracer, scenario=args.scenario)
        print(f"wrote_vdiff_input: {path}")
        return 0
    if args.command == "python-vdiff-output":
        path = write_python_vdiff_output(args.input, args.output)
        print(f"wrote_vdiff_output: {path}")
        return 0
    if args.command == "compare-vdiff-output":
        print(format_vdiff_comparison(compare_vdiff_output(args.input, args.output)))
        return 0
    if args.command == "write-synthetic-convection-input":
        path = write_synthetic_convection_input(
            args.output,
            dt_s=args.dt_s,
            ntracer=args.ntracer,
            scenario=args.scenario,
        )
        print(f"wrote_convection_input: {path}")
        return 0
    if args.command == "write-real-convection-input":
        path = write_real_convection_input_from_config(
            args.run_config,
            args.output,
            mode=args.mode,
            time_index=args.time_index,
            tracer_time_index=args.tracer_time_index,
            dt_s=args.dt_s,
            max_tracers=args.max_tracers,
            active_columns=args.active_columns,
        )
        print(f"wrote_convection_input: {path}")
        return 0
    if args.command == "python-convection-output":
        path = write_python_convection_output(args.input, args.output)
        print(f"wrote_convection_output: {path}")
        return 0
    if args.command == "compare-convection-output":
        print(format_convection_comparison(compare_convection_output(args.input, args.output)))
        return 0
    if args.command == "write-history-harness":
        path = write_history_harness_run_directory(
            args.work_dir,
            ntracer=args.ntracer,
            frequency=args.frequency,
            duration=args.duration,
            acc_interval=args.acc_interval,
        )
        print(f"wrote_history_harness: {path}")
        return 0
    if args.command == "history-harness":
        path = run_history_harness(
            args.executable,
            args.work_dir,
            ntracer=args.ntracer,
            nsteps=args.nsteps,
            dt_s=args.dt_s,
            frequency=args.frequency,
            duration=args.duration,
            acc_interval=args.acc_interval,
        )
        print(
            format_history_harness_comparison(
                compare_history_harness_output(
                    path,
                    ntracer=args.ntracer,
                    nsteps=args.nsteps,
                    dt_s=args.dt_s,
                    frequency_s=_history_interval_seconds(args.frequency),
                    acc_interval_s=_history_interval_seconds(args.acc_interval) if args.acc_interval else None,
                )
            )
        )
        return 0
    if args.command == "compare-history-harness-output":
        print(
            format_history_harness_comparison(
                compare_history_harness_output(
                    args.output,
                    ntracer=args.ntracer,
                    nsteps=args.nsteps,
                    dt_s=args.dt_s,
                    frequency_s=args.frequency_s,
                    acc_interval_s=args.acc_interval_s,
                )
            )
        )
        return 0
    if args.command == "history-fixture-generate":
        path = generate_history_harness_fixture(args.scenario, args.fixture_dir, executable=args.executable)
        print(f"wrote_history_fixture: {path}")
        return 0
    if args.command == "compare-history-fixture-wombat":
        config = history_harness_scenario_config(args.scenario)
        print(
            format_history_harness_wombat_comparison(
                compare_history_harness_to_wombat(
                    args.reference,
                    args.work_dir,
                    ntracer=int(config["ntracer"]),
                    nsteps=int(config["nsteps"]),
                    dt_s=int(config["dt_s"]),
                    frequency=str(config["frequency"]),
                    duration=str(config["duration"]),
                )
            )
        )
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
    if args.command == "oracle-fixture-handoff-compare":
        print(compare_transport_chain_handoffs(args.fixture_id, cache_dir=args.cache_dir, manifest_dir=args.manifest_dir))
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


def _assert_dry_pressure_shapes(
    lat: np.ndarray,
    lon: np.ndarray,
    area: np.ndarray,
    hyai: np.ndarray,
    hybi: np.ndarray,
    ps1: np.ndarray,
    ps2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
) -> None:
    nlev = hyai.size - 1
    horizontal = (lat.size, lon.size)
    expected_3d = (nlev, lat.size, lon.size)
    if hybi.shape != hyai.shape:
        raise ValueError("hyai and hybi must have matching edge dimensions")
    if area.shape != horizontal:
        raise ValueError(f"area_m2 must have shape {horizontal}, found {area.shape}")
    if ps1.shape != horizontal or ps2.shape != horizontal:
        raise ValueError("ps1_wet_hpa and ps2_wet_hpa must have shape (lat, lon)")
    for name, value in (
        ("sphu1_kg_kg", q1),
        ("sphu2_kg_kg", q2),
        ("tmpu1_k", t1),
        ("tmpu2_k", t2),
    ):
        if value.shape != expected_3d:
            raise ValueError(f"{name} must have shape {expected_3d}, found {value.shape}")


def _resolve_config_value(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


if __name__ == "__main__":
    raise SystemExit(main())
