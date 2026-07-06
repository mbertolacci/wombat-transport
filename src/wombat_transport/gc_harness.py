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
)
from wombat_transport.transport.tpcore import analyze_tpcore_branches, run_tpcore_one_step, setup_tpcore_terms


CONFIG_TIME_FORMAT = "%Y-%m-%d %H:%M"
PJC_INPUT_VERSION = "pjc-pfix-input-v1"
PJC_OUTPUT_VERSION = "pjc-pfix-output-v1"
PJC_SNAPSHOT_VERSION = "pjc-pfix-snapshot-v1"
TRANSPORT_INPUT_VERSION = "transport-step-input-v1"
TRANSPORT_OUTPUT_VERSION = "transport-step-output-v1"
TPCORE_SNAPSHOT_VERSION = "tpcore-step-snapshot-v1"
SNAPSHOT_INPUT_NAME = "pjc_input.nc"
SNAPSHOT_OUTPUT_NAME = "pjc_output.nc"
TPCORE_SNAPSHOT_INPUT_NAME = "tpcore_input.nc"
TPCORE_SNAPSHOT_OUTPUT_NAME = "tpcore_output.nc"
SNAPSHOT_METADATA_NAME = "metadata.json"
LARGE_ORACLE_MANIFEST_NAME = "manifest.json"
BASE_INITIAL_TPCORE_FIXTURE_ID = "base_initial_tpcore_v1"
LARGE_ORACLE_FIXTURE_IDS = (BASE_INITIAL_TPCORE_FIXTURE_ID,)

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
    max_abs_cx: float
    max_abs_cy: float


@dataclass(frozen=True)
class TransportStepOutput:
    tracer_conc_after: np.ndarray
    xmass_hpa: np.ndarray
    ymass_hpa: np.ndarray
    surface_pressure_hpa: np.ndarray


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
    if fixture_id != BASE_INITIAL_TPCORE_FIXTURE_ID:
        raise ValueError(f"no generator is registered for {fixture_id!r}")
    paths = large_oracle_fixture_paths(fixture_id, cache_dir=cache_dir, manifest_dir=manifest_dir)
    definition = _load_large_oracle_definition(paths.definition_path)
    source = dict(definition.get("source", {}))
    run_config_path = Path(run_config or source.get("run_config", "base_wombat/run.yml"))
    paths.directory.mkdir(parents=True, exist_ok=True)
    write_transport_step_input_from_config(
        run_config_path,
        paths.input_path,
        time_index=int(source.get("time_index", 0) if time_index is None else time_index),
        tracer_time_index=int(source.get("tracer_time_index", 0) if tracer_time_index is None else tracer_time_index),
        max_tracers=int(source.get("max_tracers", 1) if max_tracers is None else max_tracers),
        dt_s=float(source["dt_s"]) if dt_s is None and "dt_s" in source else dt_s,
    )
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
    rows = [
        "metric,value",
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
        max_abs_cx=float(np.max(np.abs(setup.cx))),
        max_abs_cy=float(np.max(np.abs(setup.cy))),
    )


def run_pjc_harness(executable: str | Path, input_path: str | Path, output_path: str | Path) -> None:
    executable = Path(executable)
    if not executable.exists():
        raise FileNotFoundError(
            f"GEOS-Chem harness executable not found: {executable}. "
            "Build tools/gc_harness/pjc_pfix_harness.F90 first."
        )
    subprocess.run([str(executable), str(input_path), str(output_path)], check=True)


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

    compare_python_tpcore_parser = subparsers.add_parser("compare-python-tpcore-output")
    compare_python_tpcore_parser.add_argument("input", type=Path)
    compare_python_tpcore_parser.add_argument("output", type=Path)

    snapshot_parser = subparsers.add_parser("snapshot-pjc")
    snapshot_parser.add_argument("output_dir", type=Path)
    snapshot_parser.add_argument("--executable", type=Path, default=Path("tools/gc_harness/build/pjc_pfix_harness"))
    snapshot_parser.add_argument("--dt-s", type=float, default=600.0)

    tpcore_snapshot_parser = subparsers.add_parser("snapshot-tpcore")
    tpcore_snapshot_parser.add_argument("output_dir", type=Path)
    tpcore_snapshot_parser.add_argument("--executable", type=Path, default=Path("tools/gc_harness/build/pjc_pfix_harness"))
    tpcore_snapshot_parser.add_argument("--dt-s", type=float, default=600.0)
    tpcore_snapshot_parser.add_argument("--ntracer", type=int, default=2)

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
    if args.command == "compare-python-tpcore-output":
        print(format_python_tpcore_comparison(compare_python_tpcore_output(args.input, args.output)))
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
