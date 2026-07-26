from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import netCDF4
import numpy as np

from wombat_transport.cuda import CudaRuntime
from wombat_transport.cuda.transport_step import CudaTransportStepExecutor
from wombat_transport.cuda.transport_step import CudaTransportStepPlans
from wombat_transport.fields import TracerField
from wombat_transport.transport._executor import TransportExecutor
from wombat_transport.transport._executor import apply_transport
from wombat_transport.transport.convection import G0_100
from wombat_transport.transport.pbl import LATVAP_J_PER_KG
from wombat_transport.transport.pbl._plan import prepare_vdiff_met_plan
from wombat_transport.transport.tpcore._plan import prepare_tpcore_met_plan


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    os.environ["WOMBAT_NUMBA"] = "1"
    os.environ["WOMBAT_NUMBA_THREADS"] = str(args.cpu_threads)
    host = _load_chain(args.oracle_dir)
    host_plans = _prepare_host_plans(host, workers=args.cpu_threads)
    runtime = CudaRuntime(args.device)
    cupy = runtime.array_module
    metadata = {
        "oracle_dir": str(args.oracle_dir.resolve()),
        "device": runtime.device_info.name,
        "compute_capability": runtime.device_info.compute_capability,
        "cupy": cupy.__version__,
        "cuda_driver_version": int(cupy.cuda.runtime.driverGetVersion()),
        "cuda_runtime_version": int(cupy.cuda.runtime.runtimeGetVersion()),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "timing_includes_device_state_reset": True,
        "process_cpu_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
    }
    print(json.dumps({"metadata": metadata}, sort_keys=True))

    for tracer_count in args.counts:
        factors = np.linspace(0.5, 1.5, tracer_count, dtype=np.float64)
        initial = np.ascontiguousarray(host["initial"][..., :1] * factors)
        surface_flux = np.ascontiguousarray(
            host["surface_flux"][..., :1] * factors
        )
        references = {
            name: np.ascontiguousarray(values[..., :1] * factors)
            for name, values in host["references"].items()
        }
        width = args.block_width or min(32, tracer_count)
        initial_blocks = _to_blocks(initial, width)
        flux_blocks = _to_blocks_horizontal(surface_flux, width)

        if args.cpu_baseline:
            cpu_report = _benchmark_cpu(
                host,
                host_plans,
                initial_blocks,
                surface_flux,
                tracer_count=tracer_count,
                block_width=width,
                workers=args.cpu_threads,
                warmup=args.warmup,
                repeat=args.repeat,
            )
            print(json.dumps({"cpu_baseline": cpu_report}, sort_keys=True))

        for dtype_name in args.dtypes:
            dtype = np.dtype(dtype_name)
            executor = CudaTransportStepExecutor(runtime, dtype=dtype)
            plans = _upload_plans(
                runtime,
                executor,
                host,
                host_plans,
                flux_blocks,
                dtype=dtype,
            )
            pristine = runtime.to_device(initial_blocks, dtype=dtype)
            state = runtime.to_device(initial_blocks, dtype=dtype)

            for _ in range(args.warmup):
                cupy.copyto(state, pristine)
                executor.apply(state, plans, tracer_count=tracer_count)
            runtime.synchronize()
            runtime.reset_transfer_stats()

            start = cupy.cuda.Event()
            stop = cupy.cuda.Event()
            start.record()
            for _ in range(args.repeat):
                cupy.copyto(state, pristine)
                executor.apply(state, plans, tracer_count=tracer_count)
            stop.record()
            stop.synchronize()
            step_ms = float(cupy.cuda.get_elapsed_time(start, stop)) / args.repeat
            timed_transfers = runtime.transfer_stats

            cupy.copyto(state, pristine)
            result = executor.apply(
                state,
                plans,
                tracer_count=tracer_count,
                capture_vdiff_handoff=True,
            )
            actual_tpcore = _from_blocks(
                runtime.to_host(result.tpcore_tracer_blocks),
                tracer_count,
            ).astype(np.float64)
            actual_vdiff = _from_blocks(
                runtime.to_host(result.vdiff_tracer_blocks),
                tracer_count,
            ).astype(np.float64)
            actual_final = _from_blocks(
                runtime.to_host(result.tracer_blocks),
                tracer_count,
            ).astype(np.float64)
            negative_count = int(
                runtime.to_host(result.negative_count_before_vdiff_clip)
            )
            report = {
                "dtype": dtype.name,
                "tracer_count": tracer_count,
                "block_width": width,
                "block_count": int(initial_blocks.shape[0]),
                "step_ms": step_ms,
                "active_gridcell_tracers_per_second": float(initial.size)
                / (step_ms * 1.0e-3),
                "timed_host_to_device_count": (
                    timed_transfers.host_to_device_count
                ),
                "timed_device_to_host_count": (
                    timed_transfers.device_to_host_count
                ),
                "negative_count_before_vdiff_clip": negative_count,
                "tpcore": _error_metrics(
                    actual_tpcore,
                    references["tpcore"],
                ),
                "vdiff": _error_metrics(
                    actual_vdiff,
                    references["vdiff"],
                ),
                "final": _error_metrics(
                    actual_final,
                    references["final"],
                ),
                "final_mass_max_relative_error": _mass_error(
                    actual_final,
                    references["final"],
                    host["dry_mass"],
                ),
            }
            print(json.dumps({"result": report}, sort_keys=True))
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and time one resident CuPy TPCORE -> VDIFF -> convection "
            "step against a full transport-chain oracle directory."
        )
    )
    parser.add_argument("oracle_dir", type=Path)
    parser.add_argument("--counts", type=_positive_int, nargs="+", default=[1, 24])
    parser.add_argument(
        "--dtypes",
        choices=("float64", "float32"),
        nargs="+",
        default=["float64", "float32"],
    )
    parser.add_argument(
        "--block-width",
        type=_positive_int,
        help="Defaults to min(32, tracer count)",
    )
    parser.add_argument("--warmup", type=_nonnegative_int, default=2)
    parser.add_argument("--repeat", type=_positive_int, default=20)
    parser.add_argument("--cpu-threads", type=_positive_int, default=1)
    parser.add_argument(
        "--cpu-baseline",
        action="store_true",
        help=(
            "Also time the prepared-plan float64 Numba chain with the same "
            "state reset, inputs, and block layout."
        ),
    )
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


def _load_variables(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    with netCDF4.Dataset(path) as dataset:
        return {
            name: np.ascontiguousarray(dataset.variables[name][:], dtype=np.float64)
            for name in names
        }


def _load_chain(directory: Path) -> dict[str, Any]:
    chain = _load_variables(
        directory / "transport_chain_input.nc",
        (
            "lat",
            "hyai",
            "hybi",
            "area_m2",
            "p1_hpa",
            "p2_hpa",
            "u_m_s",
            "v_m_s",
            "tracer_conc",
        ),
    )
    with netCDF4.Dataset(directory / "transport_chain_input.nc") as dataset:
        dt_s = float(dataset.dt_s)
    vdiff = _load_variables(
        directory / "vdiff_input.nc",
        (
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
        ),
    )
    convection = _load_variables(
        directory / "convection_input.nc",
        (
            "cmfmc_kg_m2_s",
            "dtrain_kg_m2_s",
            "dqrcu_kg_kg_s",
            "reevapcn_kg_kg_s",
            "delp_dry_hpa",
            "delp_hpa",
            "area_m2",
        ),
    )
    with netCDF4.Dataset(directory / "convection_input.nc") as dataset:
        reconstruct = bool(dataset.reconstruct_conv_precip_flux)
    with netCDF4.Dataset(directory / "transport_chain_output.nc") as dataset:
        references = {
            "tpcore": np.ascontiguousarray(
                dataset.variables["tpcore_tracer_conc_after"][:],
                dtype=np.float64,
            ),
            "vdiff": np.ascontiguousarray(
                dataset.variables["vdiff_tracer_conc_after"][:],
                dtype=np.float64,
            ),
            "final": np.ascontiguousarray(
                dataset.variables["tracer_conc_after"][:],
                dtype=np.float64,
            ),
        }
    return {
        "chain": chain,
        "vdiff": vdiff,
        "convection": convection,
        "initial": chain["tracer_conc"],
        "surface_flux": vdiff["surface_flux_kg_m2_s"],
        "dry_mass": vdiff["dry_air_mass_kg"],
        "references": references,
        "dt_s": dt_s,
        "reconstruct": reconstruct,
    }


def _upload_plans(
    runtime: CudaRuntime,
    executor: CudaTransportStepExecutor,
    host: dict[str, Any],
    host_plans: tuple[Any, Any],
    flux_blocks: np.ndarray,
    *,
    dtype: np.dtype[Any],
) -> CudaTransportStepPlans:
    convection = host["convection"]
    tpcore_host, vdiff_host = host_plans
    convection_plan = executor.convection.upload_plan(
        cmfmc_kg_m2_s=convection["cmfmc_kg_m2_s"],
        dtrain_kg_m2_s=convection["dtrain_kg_m2_s"],
        dqrcu_kg_kg_s=convection["dqrcu_kg_kg_s"],
        reevapcn_kg_kg_s=convection["reevapcn_kg_kg_s"],
        delp_dry_hpa=convection["delp_dry_hpa"],
        delp_hpa=convection["delp_hpa"],
        area_m2=convection["area_m2"],
        dt_s=host["dt_s"],
        reconstruct_conv_precip_flux=host["reconstruct"],
    )
    return CudaTransportStepPlans(
        tpcore=executor.tpcore.upload_plan(tpcore_host),
        vdiff=executor.vdiff.upload_plan(vdiff_host),
        convection=convection_plan,
        surface_flux_blocks=runtime.to_device(
            flux_blocks,
            dtype=dtype,
        ),
        has_surface_flux=bool(np.any(flux_blocks != 0.0)),
    )


def _prepare_host_plans(
    host: dict[str, Any],
    *,
    workers: int,
) -> tuple[Any, Any]:
    chain = host["chain"]
    vdiff = host["vdiff"]
    tpcore_host = prepare_tpcore_met_plan(
        p1_hpa=chain["p1_hpa"],
        p2_hpa=chain["p2_hpa"],
        u_m_s=chain["u_m_s"],
        v_m_s=chain["v_m_s"],
        area_m2=chain["area_m2"],
        hyai_hpa=chain["hyai"],
        hybi=chain["hybi"],
        lat_deg=chain["lat"],
        dt_s=host["dt_s"],
    )
    vdiff_host = prepare_vdiff_met_plan(
        u_top=vdiff["u_m_s"],
        v_top=vdiff["v_m_s"],
        temperature_top=vdiff["temperature_k"],
        sphu_top=vdiff["specific_humidity_kg_kg"],
        pmid_hpa=vdiff["pmid_hpa"],
        pint_hpa=vdiff["pedge_hpa"],
        virtual_temperature_top=vdiff["virtual_temperature_k"],
        bxheight_top=vdiff["bxheight_m"],
        dry_mass_top=vdiff["dry_air_mass_kg"],
        pblh_m=vdiff["pbl_top_m"],
        hflux_w_m2=vdiff["hflux_w_m2"],
        water_flux_kg_m2_s=vdiff["eflux_w_m2"] / LATVAP_J_PER_KG,
        ustar_m_s=vdiff["ustar_m_s"],
        area_m2=vdiff["area_m2"],
        dt_s=host["dt_s"],
        workers=workers,
    )
    return tpcore_host, vdiff_host


def _benchmark_cpu(
    host: dict[str, Any],
    host_plans: tuple[Any, Any],
    initial_blocks: np.ndarray,
    surface_flux: np.ndarray,
    *,
    tracer_count: int,
    block_width: int,
    workers: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    nblock, nlev, nlat, nlon, _ = initial_blocks.shape
    field = TracerField(
        names=tuple(f"tracer_{index + 1:03d}" for index in range(tracer_count)),
        data=initial_blocks.copy()[None, ...],
        units=tuple("mol mol-1 dry" for _ in range(tracer_count)),
        coords={},
    )
    executor = TransportExecutor.create(field)
    if executor.workspace.workers != workers:
        raise AssertionError("CPU transport executor worker count changed")
    tpcore_plan, vdiff_plan = host_plans
    convection = host["convection"]
    delp_dry = convection["delp_dry_hpa"]
    internal_steps = max(int(host["dt_s"]) // 300, 1)

    def apply() -> int:
        np.copyto(executor.workspace.tpcore.state_a, initial_blocks)
        return apply_transport(
            tpcore_plan=tpcore_plan,
            vdiff_plan=vdiff_plan,
            workspace=executor.workspace,
            surface_flux_kg_m2_s=surface_flux,
            cmfmc=convection["cmfmc_kg_m2_s"],
            dtrain=convection["dtrain_kg_m2_s"],
            delp_hpa=convection["delp_hpa"],
            delp_dry=delp_dry,
            bmass=delp_dry * G0_100,
            dqrcu=convection["dqrcu_kg_kg_s"],
            reevapcn=convection["reevapcn_kg_kg_s"],
            reconstruct_conv_precip_flux=host["reconstruct"],
            internal_steps=internal_steps,
            internal_dt_s=host["dt_s"] / internal_steps,
            execution="blocks",
        )

    for _ in range(warmup):
        apply()
    timings: list[float] = []
    negative_count = 0
    for _ in range(repeat):
        start = time.perf_counter()
        negative_count = apply()
        timings.append(time.perf_counter() - start)
    best_s = min(timings)
    mean_s = sum(timings) / len(timings)
    active_values = nlev * nlat * nlon * tracer_count
    return {
        "dtype": "float64",
        "tracer_count": tracer_count,
        "block_width": block_width,
        "block_count": nblock,
        "workers": workers,
        "process_cpu_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "execution": "blocks",
        "timing_includes_host_state_reset": True,
        "best_step_ms": best_s * 1.0e3,
        "mean_step_ms": mean_s * 1.0e3,
        "active_gridcell_tracers_per_second": active_values / best_s,
        "negative_count_before_vdiff_clip": negative_count,
    }


def _to_blocks(values: np.ndarray, width: int) -> np.ndarray:
    nlev, nlat, nlon, ntracer = values.shape
    blocks = np.zeros(
        ((ntracer + width - 1) // width, nlev, nlat, nlon, width),
        dtype=values.dtype,
    )
    for tracer in range(ntracer):
        block, lane = divmod(tracer, width)
        blocks[block, ..., lane] = values[..., tracer]
    return blocks


def _to_blocks_horizontal(values: np.ndarray, width: int) -> np.ndarray:
    nlat, nlon, ntracer = values.shape
    blocks = np.zeros(
        ((ntracer + width - 1) // width, nlat, nlon, width),
        dtype=values.dtype,
    )
    for tracer in range(ntracer):
        block, lane = divmod(tracer, width)
        blocks[block, ..., lane] = values[..., tracer]
    return blocks


def _from_blocks(values: np.ndarray, tracer_count: int) -> np.ndarray:
    output = np.empty((*values.shape[1:-1], tracer_count), dtype=values.dtype)
    for tracer in range(tracer_count):
        block, lane = divmod(tracer, values.shape[-1])
        output[..., tracer] = values[block, ..., lane]
    return output


def _error_metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    error = np.abs(actual - reference)
    relative = error / np.maximum(
        np.abs(reference),
        np.finfo(np.float64).tiny,
    )
    return {
        "max_abs_error": float(np.max(error)),
        "mean_abs_error": float(np.mean(error)),
        "max_relative_error": float(np.max(relative)),
    }


def _mass_error(
    actual: np.ndarray,
    reference: np.ndarray,
    dry_mass: np.ndarray,
) -> float:
    actual_mass = np.sum(actual * dry_mass[..., None], axis=(0, 1, 2))
    reference_mass = np.sum(reference * dry_mass[..., None], axis=(0, 1, 2))
    return float(
        np.max(
            np.abs(actual_mass - reference_mass)
            / np.maximum(np.abs(reference_mass), np.finfo(np.float64).tiny)
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
