from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np

from wombat_transport.cuda import CudaRuntime
from wombat_transport.transport.pbl import LATVAP_J_PER_KG
from wombat_transport.transport.pbl import run_vdiffdr_one_step
from wombat_transport.transport.pbl._cuda import CudaVdiffExecutor
from wombat_transport.transport.pbl._plan import prepare_vdiff_met_plan


INPUT_VARIABLES = (
    "tracer_conc",
    "u_m_s",
    "v_m_s",
    "temperature_k",
    "specific_humidity_kg_kg",
    "pmid_hpa",
    "pedge_hpa",
    "virtual_temperature_k",
    "bxheight_m",
    "dry_air_mass_kg",
    "pbl_top_m",
    "hflux_w_m2",
    "eflux_w_m2",
    "ustar_m_s",
    "area_m2",
    "surface_flux_kg_m2_s",
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    os.environ["WOMBAT_NUMBA"] = "1"
    os.environ["WOMBAT_NUMBA_THREADS"] = str(args.cpu_threads)
    values, dt_s = _load_input(args.input)
    geos_output = _load_geos_output(args.oracle_output)
    runtime = CudaRuntime(args.device)
    cupy = runtime.array_module
    plan_start = time.perf_counter()
    host_plan = _prepare_plan(values, dt_s, args.cpu_threads)
    plan_wall_s = time.perf_counter() - plan_start

    metadata = {
        "input": str(args.input.resolve()),
        "oracle_output": (
            str(args.oracle_output.resolve()) if args.oracle_output is not None else None
        ),
        "grid_shape": list(values["tracer_conc"].shape[:3]),
        "device": runtime.device_info.name,
        "compute_capability": runtime.device_info.compute_capability,
        "cupy": cupy.__version__,
        "cuda_driver_version": int(cupy.cuda.runtime.driverGetVersion()),
        "cuda_runtime_version": int(cupy.cuda.runtime.runtimeGetVersion()),
        "strict_compiler_options": [
            "--fmad=false",
            "--prec-div=true",
            "--prec-sqrt=true",
        ],
        "cpu_threads": args.cpu_threads,
        "host_plan_wall_s": plan_wall_s,
        "warmup": args.warmup,
        "repeat": args.repeat,
    }
    print(json.dumps({"metadata": metadata}, sort_keys=True))

    for tracer_count in args.counts:
        tracer, surface_flux, factors = _expand_tracers(values, tracer_count)
        cpu_start = time.perf_counter()
        expected = run_vdiffdr_one_step(
            tracer_conc=tracer,
            u_m_s=values["u_m_s"],
            v_m_s=values["v_m_s"],
            temperature_k=values["temperature_k"],
            specific_humidity_kg_kg=values["specific_humidity_kg_kg"],
            pmid_hpa=values["pmid_hpa"],
            pedge_hpa=values["pedge_hpa"],
            virtual_temperature_k=values["virtual_temperature_k"],
            bxheight_m=values["bxheight_m"],
            dry_air_mass_kg=values["dry_air_mass_kg"],
            pbl_top_m=values["pbl_top_m"],
            hflux_w_m2=values["hflux_w_m2"],
            eflux_w_m2=values["eflux_w_m2"],
            ustar_m_s=values["ustar_m_s"],
            area_m2=values["area_m2"],
            surface_flux_kg_m2_s=surface_flux,
            dt_s=dt_s,
            diagnostics=False,
            reuse_output=True,
        )
        cpu_full_operator_wall_s = time.perf_counter() - cpu_start

        for dtype in args.dtypes:
            report = _benchmark_dtype(
                runtime,
                host_plan,
                tracer,
                surface_flux,
                expected,
                geos_output,
                factors,
                dtype=np.dtype(dtype),
                block_width=args.block_width,
                warmup=args.warmup,
                repeat=args.repeat,
            )
            report.update(
                {
                    "tracer_count": tracer_count,
                    "cpu_full_operator_wall_s": cpu_full_operator_wall_s,
                }
            )
            print(json.dumps({"result": report}, sort_keys=True))
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and benchmark the resident CUDA VDIFF tracer application. "
            "GPU timing excludes host plan preparation and all transfers."
        )
    )
    parser.add_argument("input", type=Path, help="VDIFF harness input NetCDF")
    parser.add_argument(
        "--oracle-output",
        type=Path,
        help="Optional matching GEOS-Chem VDIFF harness output",
    )
    parser.add_argument("--counts", type=_positive_int, nargs="+", default=[1, 24])
    parser.add_argument(
        "--dtypes",
        choices=("float64", "float32"),
        nargs="+",
        default=["float64", "float32"],
    )
    parser.add_argument("--warmup", type=_nonnegative_int, default=3)
    parser.add_argument("--repeat", type=_positive_int, default=100)
    parser.add_argument("--block-width", type=_positive_int, default=8)
    parser.add_argument("--cpu-threads", type=_positive_int, default=1)
    parser.add_argument("--device", type=_nonnegative_int, default=0)
    return parser.parse_args(argv)


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def _load_input(path: Path) -> tuple[dict[str, np.ndarray], float]:
    with netCDF4.Dataset(path) as dataset:
        values = {
            name: np.ascontiguousarray(dataset.variables[name][:], dtype=np.float64)
            for name in INPUT_VARIABLES
        }
        return values, float(dataset.dt_s)


def _load_geos_output(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    with netCDF4.Dataset(path) as dataset:
        return np.ascontiguousarray(
            dataset.variables["tracer_conc_after"][:],
            dtype=np.float64,
        )


def _prepare_plan(
    values: dict[str, np.ndarray],
    dt_s: float,
    workers: int,
):
    return prepare_vdiff_met_plan(
        u_top=values["u_m_s"],
        v_top=values["v_m_s"],
        temperature_top=values["temperature_k"],
        sphu_top=values["specific_humidity_kg_kg"],
        pmid_hpa=values["pmid_hpa"],
        pint_hpa=values["pedge_hpa"],
        virtual_temperature_top=values["virtual_temperature_k"],
        bxheight_top=values["bxheight_m"],
        dry_mass_top=values["dry_air_mass_kg"],
        pblh_m=values["pbl_top_m"],
        hflux_w_m2=values["hflux_w_m2"],
        water_flux_kg_m2_s=values["eflux_w_m2"] / LATVAP_J_PER_KG,
        ustar_m_s=values["ustar_m_s"],
        area_m2=values["area_m2"],
        dt_s=dt_s,
        workers=workers,
    )


def _expand_tracers(
    values: dict[str, np.ndarray],
    tracer_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_tracers = values["tracer_conc"].shape[-1]
    if source_tracers != 1 and tracer_count != source_tracers:
        raise ValueError(
            "tracer expansion requires a one-tracer input or the original tracer count"
        )
    if tracer_count == source_tracers:
        factors = np.ones(tracer_count, dtype=np.float64)
        return (
            values["tracer_conc"],
            values["surface_flux_kg_m2_s"],
            factors,
        )
    factors = np.linspace(0.5, 1.5, tracer_count, dtype=np.float64)
    return (
        np.ascontiguousarray(values["tracer_conc"][..., :1] * factors),
        np.ascontiguousarray(values["surface_flux_kg_m2_s"][..., :1] * factors),
        factors,
    )


def _benchmark_dtype(
    runtime: CudaRuntime,
    host_plan: Any,
    tracer: np.ndarray,
    surface_flux: np.ndarray,
    expected: Any,
    geos_output: np.ndarray | None,
    factors: np.ndarray,
    *,
    dtype: np.dtype[Any],
    block_width: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    cupy = runtime.array_module
    tracer_blocks, flux_blocks = _to_blocks(tracer, surface_flux, block_width)
    runtime.reset_transfer_stats()
    executor = CudaVdiffExecutor(runtime, dtype=dtype)
    device_plan = executor.upload_plan(host_plan)
    device_tracer = runtime.to_device(tracer_blocks, dtype=dtype)
    device_flux = runtime.to_device(flux_blocks, dtype=dtype)
    output = runtime.empty(device_tracer.shape, dtype=dtype)
    setup_transfers = runtime.transfer_stats
    runtime.reset_transfer_stats()
    has_flux = bool(np.any(surface_flux != 0.0))

    for _ in range(warmup):
        executor.apply_blocks(
            device_tracer,
            device_plan,
            device_flux,
            has_flux=has_flux,
            tracer_count=tracer.shape[-1],
            output=output,
        )
    runtime.synchronize()
    runtime.reset_transfer_stats()

    start = cupy.cuda.Event()
    stop = cupy.cuda.Event()
    start.record()
    result = None
    for _ in range(repeat):
        result = executor.apply_blocks(
            device_tracer,
            device_plan,
            device_flux,
            has_flux=has_flux,
            tracer_count=tracer.shape[-1],
            output=output,
        )
    stop.record()
    stop.synchronize()
    kernel_ms = float(cupy.cuda.get_elapsed_time(start, stop)) / repeat
    apply_transfers = runtime.transfer_stats
    assert result is not None
    actual_blocks = runtime.to_host(output)
    actual_native = _from_blocks(actual_blocks, tracer.shape[-1])
    negative_count = int(runtime.to_host(result.negative_count_before_clip))
    actual = actual_native.astype(np.float64)
    cpu_reference = np.asarray(expected.tracer_conc, dtype=np.float64)
    report: dict[str, Any] = {
        "dtype": dtype.name,
        "block_width": block_width,
        "block_count": int(tracer_blocks.shape[0]),
        "padded_tracer_capacity": int(
            tracer_blocks.shape[0] * tracer_blocks.shape[-1]
        ),
        "kernel_ms": kernel_ms,
        "gridcell_tracers_per_second": float(actual.size) / (kernel_ms * 1.0e-3),
        "cpu_max_abs_error": float(np.max(np.abs(actual - cpu_reference))),
        "cpu_max_relative_error": _max_relative_error(actual, cpu_reference),
        "cpu_max_ulp_error": _max_ulp_error(actual_native, cpu_reference),
        "cpu_mass_max_relative_error": _mass_max_relative_error(
            actual,
            cpu_reference,
            host_plan.dry_mass,
        ),
        "negative_count_before_clip": negative_count,
        "negative_count_expected": int(expected.negative_count_before_clip),
        "setup_host_to_device_count": setup_transfers.host_to_device_count,
        "setup_host_to_device_bytes": setup_transfers.host_to_device_bytes,
        "timed_host_to_device_count": apply_transfers.host_to_device_count,
        "timed_device_to_host_count": apply_transfers.device_to_host_count,
        "state_bytes": int(tracer_blocks.size * dtype.itemsize),
        "tracer_workspace_bytes": int(
            2 * tracer_blocks.size * dtype.itemsize
        ),
    }
    if geos_output is not None:
        if geos_output.shape[-1] == factors.size:
            geos_reference = geos_output
        elif geos_output.shape[-1] == 1:
            geos_reference = geos_output * factors
        else:
            raise ValueError("GEOS-Chem output tracer count cannot be expanded")
        report.update(
            {
                "geos_max_abs_error": float(
                    np.max(np.abs(actual - geos_reference))
                ),
                "geos_max_relative_error": _max_relative_error(
                    actual,
                    geos_reference,
                ),
                "cpu_geos_max_abs_error": float(
                    np.max(np.abs(cpu_reference - geos_reference))
                ),
            }
        )
    return report


def _to_blocks(
    tracer: np.ndarray,
    surface_flux: np.ndarray,
    block_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    nlev, nlat, nlon, ntracer = tracer.shape
    nblock = (ntracer + block_width - 1) // block_width
    tracer_blocks = np.zeros(
        (nblock, nlev, nlat, nlon, block_width),
        dtype=tracer.dtype,
    )
    flux_blocks = np.zeros(
        (nblock, nlat, nlon, block_width),
        dtype=surface_flux.dtype,
    )
    for index in range(ntracer):
        block, lane = divmod(index, block_width)
        tracer_blocks[block, ..., lane] = tracer[..., index]
        flux_blocks[block, ..., lane] = surface_flux[..., index]
    return tracer_blocks, flux_blocks


def _from_blocks(blocks: np.ndarray, tracer_count: int) -> np.ndarray:
    block_width = blocks.shape[-1]
    canonical = np.empty((*blocks.shape[1:-1], tracer_count), dtype=blocks.dtype)
    for index in range(tracer_count):
        block, lane = divmod(index, block_width)
        canonical[..., index] = blocks[block, ..., lane]
    return canonical


def _max_relative_error(actual: np.ndarray, reference: np.ndarray) -> float:
    scale = np.maximum(np.abs(reference), np.finfo(reference.dtype).tiny)
    return float(np.max(np.abs(actual - reference) / scale))


def _mass_max_relative_error(
    actual: np.ndarray,
    reference: np.ndarray,
    dry_mass: np.ndarray,
) -> float:
    actual_mass = np.sum(actual * dry_mass[..., None], axis=(0, 1, 2))
    reference_mass = np.sum(reference * dry_mass[..., None], axis=(0, 1, 2))
    scale = np.maximum(np.abs(reference_mass), np.finfo(np.float64).tiny)
    return float(np.max(np.abs(actual_mass - reference_mass) / scale))


def _max_ulp_error(actual: np.ndarray, reference: np.ndarray) -> int:
    rounded = reference.astype(actual.dtype)
    if actual.dtype == np.float32:
        actual_bits = actual.view(np.int32).astype(np.int64)
        reference_bits = rounded.view(np.int32).astype(np.int64)
    elif actual.dtype == np.float64:
        actual_bits = actual.view(np.int64)
        reference_bits = rounded.view(np.int64)
    else:
        raise TypeError(f"unsupported ULP dtype {actual.dtype}")
    if np.any(actual < 0.0) or np.any(rounded < 0.0):
        return -1
    return int(np.max(np.abs(actual_bits - reference_bits)))


if __name__ == "__main__":
    raise SystemExit(main())
