from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import netCDF4
import numpy as np

from wombat_transport.io import initialize_tracers
from wombat_transport.run_config import load_run_config
from wombat_transport.transport import (
    dry_pressure_thickness_hpa,
    horizontal_mass_flux_hpa,
    load_transport_forcing,
)


CONFIG_TIME_FORMAT = "%Y-%m-%d %H:%M"
PJC_INPUT_VERSION = "pjc-pfix-input-v1"
PJC_OUTPUT_VERSION = "pjc-pfix-output-v1"
TRANSPORT_INPUT_VERSION = "transport-step-input-v1"
TRANSPORT_OUTPUT_VERSION = "transport-step-output-v1"


@dataclass(frozen=True)
class PjcComparison:
    xmass_max_abs_error_hpa: float
    xmass_mean_abs_error_hpa: float
    ymass_max_abs_error_hpa: float
    ymass_mean_abs_error_hpa: float


@dataclass(frozen=True)
class TransportStepOutput:
    tracer_conc_after: np.ndarray
    xmass_hpa: np.ndarray
    ymass_hpa: np.ndarray
    surface_pressure_hpa: np.ndarray


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
        p1 = np.asarray(dataset.variables["p1_hpa"][:], dtype=np.float64)
        u = np.asarray(dataset.variables["u_m_s"][:], dtype=np.float64)
        v = np.asarray(dataset.variables["v_m_s"][:], dtype=np.float64)
        dt_s = float(getattr(dataset, "dt_s"))
    surface_pressure_pa = p1[np.newaxis, :, :] * 100.0
    delp = dry_pressure_thickness_hpa(surface_pressure_pa, hyai, hybi)
    expected_x, expected_y = horizontal_mass_flux_hpa(
        delp,
        u[np.newaxis, :, :, :],
        v[np.newaxis, :, :, :],
        lat,
        dt_s=dt_s,
    )
    observed_x, observed_y = read_pjc_output(output_path)
    x_error = np.abs(observed_x - expected_x[0])
    y_error = np.abs(observed_y - expected_y[0])
    return PjcComparison(
        xmass_max_abs_error_hpa=float(np.max(x_error)),
        xmass_mean_abs_error_hpa=float(np.mean(x_error)),
        ymass_max_abs_error_hpa=float(np.max(y_error)),
        ymass_mean_abs_error_hpa=float(np.mean(y_error)),
    )


def run_pjc_harness(executable: str | Path, input_path: str | Path, output_path: str | Path) -> None:
    executable = Path(executable)
    if not executable.exists():
        raise FileNotFoundError(
            f"GEOS-Chem harness executable not found: {executable}. "
            "Build tools/gc_harness/pjc_pfix_harness.F90 first."
        )
    subprocess.run([str(executable), str(input_path), str(output_path)], check=True)


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
