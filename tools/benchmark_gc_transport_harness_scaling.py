from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TextIO

import netCDF4
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.benchmark_convection_scaling import _build_synthetic_convection_inputs
from tools.benchmark_vdiff_scaling import _build_synthetic_vdiff_inputs
from wombat_transport.gc_harness import (
    CONVECTION_INPUT_VERSION,
    TRANSPORT_INPUT_VERSION,
    VDIFF_INPUT_VERSION,
    write_pjc_input,
)
from wombat_transport.run_config import load_run_config


DEFAULT_COUNTS = (1, 24, 96, 256, 512)
DEFAULT_DT_S = 600.0
DEFAULT_REPEAT = 3
CSV_FIELDS = (
    "tracer_count",
    "operator",
    "status",
    "repeat",
    "best_wall_s",
    "mean_wall_s",
    "output_mode",
    "omp_num_threads",
    "reason",
)
OPERATOR_NAMES = ("tpcore", "vdiff", "convection")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    thread_env = _single_thread_env()
    rows: list[dict[str, str]] = []

    with tempfile.TemporaryDirectory(prefix="wombat_gc_transport_bench_") as temp_name:
        temp_dir = Path(temp_name)
        executables = _build_nowrite_harnesses(temp_dir, args)
        data_dir = temp_dir / "data"
        data_dir.mkdir()

        for tracer_count in args.counts:
            count_dir = data_dir / f"n{tracer_count}"
            count_dir.mkdir()
            inputs = {
                "tpcore": count_dir / "tpcore_input.nc",
                "vdiff": count_dir / "vdiff_input.nc",
                "convection": count_dir / "convection_input.nc",
            }
            _write_tpcore_input(inputs["tpcore"], args.run_config, tracer_count, dt_s=args.dt_s)
            _write_vdiff_input(inputs["vdiff"], args.run_config, tracer_count, dt_s=args.dt_s)
            _write_convection_input(inputs["convection"], args.run_config, tracer_count, dt_s=args.dt_s)

            best_sum = 0.0
            mean_sum = 0.0
            all_ok = True
            for name in OPERATOR_NAMES:
                row = _time_operator(
                    name,
                    executable=executables[name],
                    input_path=inputs[name],
                    work_dir=count_dir,
                    repeat=args.repeat,
                    env=thread_env,
                )
                row.update(
                    {
                        "tracer_count": str(tracer_count),
                        "operator": name,
                        "repeat": str(args.repeat),
                        "output_mode": "suppressed",
                        "omp_num_threads": thread_env["OMP_NUM_THREADS"],
                    }
                )
                rows.append(row)
                if row["status"] == "completed":
                    best_sum += float(row["best_wall_s"])
                    mean_sum += float(row["mean_wall_s"])
                else:
                    all_ok = False

            rows.append(
                {
                    "tracer_count": str(tracer_count),
                    "operator": "total_sum_of_stage_bests",
                    "status": "completed" if all_ok else "failed",
                    "repeat": str(args.repeat),
                    "best_wall_s": f"{best_sum:.8f}" if all_ok else "",
                    "mean_wall_s": f"{mean_sum:.8f}" if all_ok else "",
                    "output_mode": "suppressed",
                    "omp_num_threads": thread_env["OMP_NUM_THREADS"],
                    "reason": "" if all_ok else "one or more operator rows failed",
                }
            )

    _write_rows(rows, args.output)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark single-thread GEOS-Chem operator harness scaling for one synthetic "
            "full-grid TPCORE + VDIFF + convection transport step."
        )
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        default=Path("base_wombat/run.yml"),
        help="Run config used to locate the grid template. Defaults to base_wombat/run.yml.",
    )
    parser.add_argument("--counts", type=_positive_int, nargs="+", default=list(DEFAULT_COUNTS))
    parser.add_argument("--repeat", type=_positive_int, default=DEFAULT_REPEAT)
    parser.add_argument("--dt-s", type=float, default=DEFAULT_DT_S)
    parser.add_argument(
        "--fc",
        default=os.environ.get("FC", "/usr/bin/f95"),
        help="Fortran compiler for temporary no-output harnesses. Defaults to FC or /usr/bin/f95.",
    )
    parser.add_argument(
        "--netcdf-prefix",
        type=Path,
        default=Path(os.environ.get("NETCDF_PREFIX", "/home/mgnb/miniconda3/envs/wombat-v3-forward")),
        help="NetCDF Fortran install prefix used by the GEOS-Chem harness build.",
    )
    parser.add_argument(
        "--build-root",
        type=Path,
        default=Path("base/build"),
        help="GEOS-Chem build directory with compiled libraries. Defaults to base/build.",
    )
    parser.add_argument(
        "--harness-build-dir",
        type=Path,
        default=Path("tools/gc_harness/build"),
        help="Directory containing generated harness objects. Defaults to tools/gc_harness/build.",
    )
    parser.add_argument("--output", type=Path, help="Optional CSV output path. Defaults to stdout.")
    args = parser.parse_args(argv)
    if args.dt_s <= 0.0:
        parser.error("--dt-s must be positive")
    return args


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _single_thread_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return env


def _build_nowrite_harnesses(temp_dir: Path, args: argparse.Namespace) -> dict[str, Path]:
    build_root = _resolve(args.build_root)
    harness_build = _resolve(args.harness_build_dir)
    netcdf_prefix = _resolve(args.netcdf_prefix)
    generated_objects = {
        "tpcore": (harness_build / "tpcore_trace_mod.o",),
        "vdiff": (harness_build / "vdiff_trace_mod.o", harness_build / "vdiff_mod.harness.o"),
        "convection": (harness_build / "convection_mod.harness.o",),
    }
    sources = {
        "tpcore": REPO_ROOT / "tools/gc_harness/pjc_pfix_harness.F90",
        "vdiff": REPO_ROOT / "tools/gc_harness/vdiff_harness.F90",
        "convection": REPO_ROOT / "tools/gc_harness/convection_harness.F90",
    }
    common_flags = [
        "-O2",
        f"-J{temp_dir}",
        f"-I{temp_dir}",
        f"-I{harness_build}",
        f"-I{build_root / 'mod'}",
        f"-I{netcdf_prefix / 'include'}",
    ]
    link_flags = [
        "-Wl,-O2",
        "-Wl,--sort-common",
        "-Wl,--as-needed",
        "-Wl,-z,relro",
        "-Wl,-z,now",
        "-Wl,--disable-new-dtags",
        "-Wl,--gc-sections",
        "-Wl,--allow-shlib-undefined",
        f"-Wl,-rpath,{netcdf_prefix / 'lib'}",
        f"-Wl,-rpath-link,{netcdf_prefix / 'lib'}",
        f"-L{netcdf_prefix / 'lib'}",
        f"-L{netcdf_prefix / 'targets/x86_64-linux/lib'}",
        f"-L{netcdf_prefix / 'targets/x86_64-linux/lib/stubs'}",
    ]
    gc_libs = _gc_libraries(build_root)

    executables: dict[str, Path] = {}
    for name in OPERATOR_NAMES:
        for required in (*generated_objects[name], *gc_libs):
            if not required.exists():
                raise FileNotFoundError(
                    f"Required GEOS-Chem harness object/library is missing: {required}. "
                    "Run the corresponding tools/gc_harness/build_*_harness.sh scripts first."
                )
        patched_source = temp_dir / f"{name}_nowrite.F90"
        _write_output_suppressed_source(sources[name], patched_source)
        executable = temp_dir / f"{name}_nowrite"
        command = [
            args.fc,
            *common_flags,
            str(patched_source),
            *(str(path) for path in generated_objects[name]),
            *link_flags,
            "-Wl,--start-group",
            *(str(path) for path in gc_libs),
            "-Wl,--end-group",
            "-fopenmp",
            str(netcdf_prefix / "lib/libnetcdff.so"),
            str(netcdf_prefix / "lib/libnetcdf.so"),
            "-o",
            str(executable),
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        executables[name] = executable
    return executables


def _resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _write_output_suppressed_source(source: Path, output: Path) -> None:
    lines = []
    replaced = False
    for line in source.read_text(encoding="utf-8").splitlines():
        if "call write_output(trim(output_path)" in line:
            lines.append("  continue  ! Output suppressed for timing benchmark.")
            replaced = True
        else:
            lines.append(line)
    if not replaced:
        raise ValueError(f"Could not find write_output call in {source}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _gc_libraries(build_root: Path) -> tuple[Path, ...]:
    return (
        build_root / "src/GEOS-Chem/GeosCore/CMakeFiles/GeosCore.dir/cleanup.F90.o",
        build_root / "src/GEOS-Chem/GeosCore/libGeosCore.a",
        build_root / "src/GEOS-Chem/ObsPack/libObsPack.a",
        build_root / "src/GEOS-Chem/ObsOperator/libObsOperator.a",
        build_root / "src/GEOS-Chem/History/libHistory.a",
        build_root / "src/GEOS-Chem/KPP/fullchem/libKPP.a",
        build_root / "src/GEOS-Chem/GeosUtil/libGeosUtil.a",
        build_root / "src/GEOS-Chem/NcdfUtil/libNcdfUtil.a",
        build_root / "src/GEOS-Chem/GeosUtil/libJulDay.a",
        build_root / "src/GEOS-Chem/Headers/libHeaders.a",
        build_root / "src/GEOS-Chem/KPP/fullchem/libKPP_FirstPass.a",
        build_root / "src/Cloud-J/src/Core/libCloudJ_Core.a",
        build_root / "src/HETP/src/Core/libHETP_core.a",
        build_root / "src/HEMCO/src/Interfaces/Shared/libHCOI_Shared.a",
        build_root / "src/HEMCO/src/Extensions/libHCOX.a",
        build_root / "src/HEMCO/src/Core/libHCO.a",
        build_root / "src/HEMCO/src/Shared/GeosUtil/libGeosUtilHco.a",
        build_root / "src/HEMCO/src/Shared/NcdfUtil/libNcdfUtilHco.a",
        build_root / "src/HEMCO/src/Shared/GeosUtil/libJulDayHco.a",
        build_root / "src/HEMCO/src/Shared/Headers/libHeadersHco.a",
    )


def _grid_template_values(run_config_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    config = load_run_config(run_config_path)
    with netCDF4.Dataset(config.grid_template) as template:
        return (
            np.asarray(template.variables["lat"][:], dtype=np.float64),
            np.asarray(template.variables["lon"][:], dtype=np.float64),
            np.asarray(template.variables["hyai"][:], dtype=np.float64),
            np.asarray(template.variables["hybi"][:], dtype=np.float64),
            np.asarray(template.variables["AREA"][:], dtype=np.float64),
        )


def _write_tpcore_input(path: Path, run_config: Path, ntracer: int, *, dt_s: float) -> None:
    lat, lon, hyai, hybi, area = _grid_template_values(run_config)
    nlev = hyai.size - 1
    level = np.arange(nlev, dtype=np.float64)[:, np.newaxis, np.newaxis]
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lat_2d = lat_rad[:, np.newaxis]
    lon_2d = lon_rad[np.newaxis, :]

    p1 = 965.0 + 22.0 * np.cos(lat_2d) ** 2
    p1 = p1 + 2.0 * np.sin(lon_2d) * np.cos(lat_2d)
    p2 = p1 + 0.25 * np.cos(2.0 * lon_2d) * np.cos(lat_2d)

    lat_3d = lat_rad[np.newaxis, :, np.newaxis]
    lon_3d = lon_rad[np.newaxis, np.newaxis, :]
    vertical_wave = np.sin((level + 1.0) / float(nlev) * np.pi)
    u = 5.0 * vertical_wave * np.cos(lat_3d)
    u = u * (1.0 + 0.15 * np.cos(lon_3d))
    v = 0.35 * np.cos((level + 1.0) / float(nlev) * np.pi) * np.sin(2.0 * lon_3d)
    v = v * np.cos(lat_3d)

    tracer_index = np.arange(ntracer, dtype=np.float64)[np.newaxis, np.newaxis, np.newaxis, :]
    lev_index = np.arange(nlev, dtype=np.float64)[:, np.newaxis, np.newaxis, np.newaxis]
    tracer = 4.0e-4 + (tracer_index + 1.0) * 1.0e-7
    tracer = tracer + 2.5e-8 * lev_index / max(float(nlev - 1), 1.0)
    tracer = tracer + 1.5e-8 * np.sin(lat_rad)[np.newaxis, :, np.newaxis, np.newaxis]
    tracer = tracer + 7.5e-9 * np.cos(lon_rad)[np.newaxis, np.newaxis, :, np.newaxis]

    write_pjc_input(
        path,
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
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.harness = TRANSPORT_INPUT_VERSION
        dataset.createDimension("tracer", ntracer)
        _write_names(dataset, tuple(f"tracer_{index + 1:03d}" for index in range(ntracer)))
        dataset.createVariable("tracer_conc", "f8", ("tracer", "lev", "lat", "lon"))[:] = np.transpose(
            tracer, (3, 0, 1, 2)
        )


def _write_vdiff_input(path: Path, run_config: Path, ntracer: int, *, dt_s: float) -> None:
    lat, lon, *_ = _grid_template_values(run_config)
    data = _build_synthetic_vdiff_inputs(run_config, ntracer, dt_s=dt_s)
    nlev, nlat, nlon, _ = data.tracer_conc.shape
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("tracer", ntracer)
        dataset.createDimension("lev", nlev)
        dataset.createDimension("ilev", nlev + 1)
        dataset.createDimension("lat", nlat)
        dataset.createDimension("lon", nlon)
        dataset.harness = VDIFF_INPUT_VERSION
        dataset.dt_s = float(data.dt_s)
        dataset.scenario = "fullgrid-synthetic-zero-surface-flux"
        dataset.createVariable("lon", "f8", ("lon",))[:] = lon
        dataset.createVariable("lat", "f8", ("lat",))[:] = lat
        dataset.createVariable("tracer_conc", "f8", ("tracer", "lev", "lat", "lon"))[:] = np.transpose(
            data.tracer_conc, (3, 0, 1, 2)
        )
        dataset.createVariable("surface_flux_kg_m2_s", "f8", ("tracer", "lat", "lon"))[:] = np.transpose(
            data.surface_flux_kg_m2_s, (2, 0, 1)
        )
        _write_variable(dataset, "u_m_s", data.u_m_s, ("lev", "lat", "lon"))
        _write_variable(dataset, "v_m_s", data.v_m_s, ("lev", "lat", "lon"))
        _write_variable(dataset, "temperature_k", data.temperature_k, ("lev", "lat", "lon"))
        _write_variable(dataset, "specific_humidity_kg_kg", data.specific_humidity_kg_kg, ("lev", "lat", "lon"))
        _write_variable(dataset, "pmid_hpa", data.pmid_hpa, ("lev", "lat", "lon"))
        _write_variable(dataset, "pedge_hpa", data.pedge_hpa, ("ilev", "lat", "lon"))
        _write_variable(dataset, "virtual_temperature_k", data.virtual_temperature_k, ("lev", "lat", "lon"))
        _write_variable(dataset, "bxheight_m", data.bxheight_m, ("lev", "lat", "lon"))
        _write_variable(dataset, "dry_air_mass_kg", data.dry_air_mass_kg, ("lev", "lat", "lon"))
        _write_variable(dataset, "pbl_top_m", data.pbl_top_m, ("lat", "lon"))
        _write_variable(dataset, "hflux_w_m2", data.hflux_w_m2, ("lat", "lon"))
        _write_variable(dataset, "eflux_w_m2", data.eflux_w_m2, ("lat", "lon"))
        _write_variable(dataset, "ustar_m_s", data.ustar_m_s, ("lat", "lon"))
        _write_variable(dataset, "area_m2", data.area_m2, ("lat", "lon"))


def _write_convection_input(path: Path, run_config: Path, ntracer: int, *, dt_s: float) -> None:
    lat, lon, *_ = _grid_template_values(run_config)
    data = _build_synthetic_convection_inputs(run_config, ntracer, dt_s=dt_s)
    nlev, nlat, nlon, _ = data.tracer_conc.shape
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("tracer", ntracer)
        dataset.createDimension("lev", nlev)
        dataset.createDimension("lat", nlat)
        dataset.createDimension("lon", nlon)
        _write_names(dataset, tuple(f"tracer_{index + 1:03d}" for index in range(ntracer)))
        dataset.harness = CONVECTION_INPUT_VERSION
        dataset.dt_s = float(data.dt_s)
        dataset.scenario = "fullgrid-synthetic-active-cloud"
        dataset.reconstruct_conv_precip_flux = int(data.reconstruct_conv_precip_flux)
        dataset.createVariable("lon", "f8", ("lon",))[:] = lon
        dataset.createVariable("lat", "f8", ("lat",))[:] = lat
        dataset.createVariable("tracer_conc", "f8", ("tracer", "lev", "lat", "lon"))[:] = np.transpose(
            data.tracer_conc, (3, 0, 1, 2)
        )
        _write_variable(dataset, "cmfmc_kg_m2_s", data.cmfmc_kg_m2_s, ("lev", "lat", "lon"))
        _write_variable(dataset, "dtrain_kg_m2_s", data.dtrain_kg_m2_s, ("lev", "lat", "lon"))
        _write_variable(dataset, "dqrcu_kg_kg_s", data.dqrcu_kg_kg_s, ("lev", "lat", "lon"))
        _write_variable(dataset, "reevapcn_kg_kg_s", data.reevapcn_kg_kg_s, ("lev", "lat", "lon"))
        _write_variable(dataset, "delp_dry_hpa", data.delp_dry_hpa, ("lev", "lat", "lon"))
        _write_variable(dataset, "delp_hpa", data.delp_hpa, ("lev", "lat", "lon"))
        _write_variable(dataset, "area_m2", data.area_m2, ("lat", "lon"))
        _write_variable(dataset, "bxheight_m", data.bxheight_m, ("lev", "lat", "lon"))
        _write_variable(dataset, "pficu_kg_m2_s", data.pficu_kg_m2_s, ("lev", "lat", "lon"))
        _write_variable(dataset, "pflcu_kg_m2_s", data.pflcu_kg_m2_s, ("lev", "lat", "lon"))
        _write_variable(dataset, "temperature_k", data.temperature_k, ("lev", "lat", "lon"))
        _write_variable(dataset, "precccon_mm_day", data.precccon_mm_day, ("lat", "lon"))


def _write_names(dataset: netCDF4.Dataset, names: tuple[str, ...]) -> None:
    name_length = max(max((len(name) for name in names), default=1), 1)
    dataset.createDimension("name_strlen", name_length)
    name_var = dataset.createVariable("tracer_name", "S1", ("tracer", "name_strlen"))
    encoded = np.asarray([name.encode("ascii", errors="replace") for name in names], dtype=f"S{name_length}")
    name_var[:] = netCDF4.stringtochar(encoded)


def _write_variable(dataset: netCDF4.Dataset, name: str, values: np.ndarray, dimensions: tuple[str, ...]) -> None:
    dataset.createVariable(name, "f8", dimensions)[:] = values


def _time_operator(
    name: str,
    *,
    executable: Path,
    input_path: Path,
    work_dir: Path,
    repeat: int,
    env: dict[str, str],
) -> dict[str, str]:
    elapsed_values: list[float] = []
    for run_index in range(repeat):
        output_path = work_dir / f"{name}_output_{run_index}.nc"
        start = time.perf_counter()
        try:
            subprocess.run(
                [str(executable), str(input_path), str(output_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
        except subprocess.CalledProcessError as error:
            return {
                "status": "failed",
                "best_wall_s": "",
                "mean_wall_s": "",
                "reason": f"exit status {error.returncode}",
            }
        elapsed_values.append(time.perf_counter() - start)
        output_path.unlink(missing_ok=True)

    return {
        "status": "completed",
        "best_wall_s": f"{min(elapsed_values):.8f}",
        "mean_wall_s": f"{statistics.mean(elapsed_values):.8f}",
        "reason": "",
    }


def _write_rows(rows: list[dict[str, str]], output: Path | None) -> None:
    if output is None:
        _write_csv(rows, sys.stdout)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        _write_csv(rows, handle)


def _write_csv(rows: list[dict[str, str]], handle: TextIO) -> None:
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


if __name__ == "__main__":
    raise SystemExit(main())
