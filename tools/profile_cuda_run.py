"""Profile one ordinary resident CUDA run with host and CUDA-event regions."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import wombat_transport.runner as runner_module
from wombat_transport.cuda.executor import CudaRunExecutor
from wombat_transport.cuda.forcing import CudaForcingChunks
from wombat_transport.cuda.preparation import CudaPlanPreparation
from wombat_transport.cuda.runtime import CudaRuntime
from wombat_transport.emissions import EmissionsOperator
from wombat_transport.run_config import load_run_config
from wombat_transport.transport.convection._cuda import (
    CudaConvectionExecutor,
)
from wombat_transport.transport.forcing import TransportForcingProvider
from wombat_transport.transport.pbl._cuda import CudaVdiffExecutor
from wombat_transport.transport.tpcore._cuda import CudaTpcoreExecutor


class RunProfiler:
    """Collect nested host timings and asynchronous CUDA event pairs."""

    def __init__(self, cupy: Any, *, nvtx: bool) -> None:
        self.cupy = cupy
        self.nvtx = nvtx
        self.capturing = False
        self.host_seconds: defaultdict[str, float] = defaultdict(float)
        self.host_counts: defaultdict[str, int] = defaultdict(int)
        self.transfer_bytes: defaultdict[str, int] = defaultdict(int)
        self.device_events: defaultdict[str, list[tuple[Any, Any]]] = (
            defaultdict(list)
        )
        self.kernel_attributes: dict[str, dict[str, Any]] = {}

    def range(self, label: str) -> Any:
        if not self.nvtx or not self.capturing:
            return nullcontext()
        from cupyx.profiler import time_range

        return time_range(label)

    def timed_method(
        self,
        cls: type[Any],
        name: str,
        label: str,
        *,
        device: bool = False,
        byte_arg: int | None = None,
    ) -> None:
        original = getattr(cls, name)

        def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if not self.capturing:
                return original(instance, *args, **kwargs)
            start_wall = perf_counter()
            start_event = None
            end_event = None
            if device:
                start_event = self.cupy.cuda.Event()
                end_event = self.cupy.cuda.Event()
                start_event.record()
            with self.range(label):
                result = original(instance, *args, **kwargs)
            if end_event is not None:
                end_event.record()
                self.device_events[label].append((start_event, end_event))
            self.host_seconds[label] += perf_counter() - start_wall
            self.host_counts[label] += 1
            if byte_arg is not None and len(args) > byte_arg:
                self.transfer_bytes[label] += int(args[byte_arg].nbytes)
            return result

        setattr(cls, name, wrapped)

    def timed_runner_function(self, name: str, label: str) -> None:
        original = getattr(runner_module, name)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if not self.capturing:
                return original(*args, **kwargs)
            start = perf_counter()
            with self.range(label):
                result = original(*args, **kwargs)
            self.host_seconds[label] += perf_counter() - start
            self.host_counts[label] += 1
            return result

        setattr(runner_module, name, wrapped)

    def instrument_kernel_attributes(
        self,
        cls: type[Any],
        attributes: tuple[tuple[str, str], ...],
    ) -> None:
        original = cls.__init__
        profiler = self

        class TimedKernel:
            def __init__(self, kernel: Any, label: str) -> None:
                self.kernel = kernel
                self.label = label
                profiler.kernel_attributes[label] = {
                    key: _json_scalar(value)
                    for key, value in kernel.attributes.items()
                }

            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                if not profiler.capturing:
                    return self.kernel(*args, **kwargs)
                start = profiler.cupy.cuda.Event()
                end = profiler.cupy.cuda.Event()
                start.record()
                with profiler.range(self.label):
                    result = self.kernel(*args, **kwargs)
                end.record()
                profiler.device_events[self.label].append((start, end))
                return result

            @property
            def attributes(self) -> Any:
                return self.kernel.attributes

        def wrapped(instance: Any, *args: Any, **kwargs: Any) -> None:
            original(instance, *args, **kwargs)
            for attribute, label in attributes:
                kernel = getattr(instance, attribute)
                setattr(instance, attribute, TimedKernel(kernel, label))

        cls.__init__ = wrapped

    def report_regions(self) -> dict[str, Any]:
        host = [
            {
                "label": label,
                "calls": self.host_counts[label],
                "total_ms": self.host_seconds[label] * 1000.0,
                "bytes": self.transfer_bytes.get(label, 0),
            }
            for label in sorted(self.host_seconds)
        ]
        device = []
        for label in sorted(self.device_events):
            events = self.device_events[label]
            total_ms = sum(
                self.cupy.cuda.get_elapsed_time(start, end)
                for start, end in events
            )
            device.append(
                {
                    "label": label,
                    "calls": len(events),
                    "total_ms": total_ms,
                }
            )
        return {
            "host": host,
            "device": device,
            "kernel_attributes": self.kernel_attributes,
        }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    os.environ["WOMBAT_BACKEND"] = "cuda"
    os.environ["WOMBAT_CUDA_DTYPE"] = args.dtype
    os.environ["WOMBAT_TRANSPORT_BLOCK_WIDTH"] = str(args.block_width)
    runtime = CudaRuntime(args.device)
    cupy = runtime.array_module
    profiler = RunProfiler(cupy, nvtx=args.nvtx)
    _install_instrumentation(profiler)
    config = load_run_config(args.config)

    if args.warmup_steps:
        with TemporaryDirectory(prefix="wombat-cuda-profile-warmup-") as work:
            warm_config = replace(
                config,
                name=f"{config.name}_profile_warmup",
                output_dir=Path(work),
            )
            warm = runner_module.run_tracer_simulation(
                warm_config,
                max_steps=args.warmup_steps,
            )
        del warm
        gc.collect()
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()

    profiler.capturing = True
    whole_start = cupy.cuda.Event()
    whole_end = cupy.cuda.Event()
    whole_start.record()
    start_wall = perf_counter()
    with profiler.range("run.total"):
        result = runner_module.run_tracer_simulation(
            config,
            max_steps=args.steps,
        )
    whole_end.record()
    whole_end.synchronize()
    wall_seconds = perf_counter() - start_wall
    profiler.capturing = False

    memory_pool = cupy.get_default_memory_pool()
    pinned_pool = cupy.get_default_pinned_memory_pool()
    report = {
        "metadata": {
            "config": str(args.config.resolve()),
            "dtype": args.dtype,
            "block_width": args.block_width,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "device_id": args.device,
            "device": runtime.device_info.name,
            "compute_capability": runtime.device_info.compute_capability,
            "cupy": cupy.__version__,
            "cuda_driver_version": int(
                cupy.cuda.runtime.driverGetVersion()
            ),
            "cuda_runtime_version": int(
                cupy.cuda.runtime.runtimeGetVersion()
            ),
            "nvtx": args.nvtx,
            "process_cpu_affinity": (
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
            "nested_device_regions_overlap_parent_regions": True,
        },
        "summary": {
            "wall_seconds": wall_seconds,
            "device_span_ms": cupy.cuda.get_elapsed_time(
                whole_start,
                whole_end,
            ),
            "final_state_bytes": int(result.state.block_data.nbytes),
            "device_pool_used_bytes": int(memory_pool.used_bytes()),
            "device_pool_total_bytes": int(memory_pool.total_bytes()),
            "pinned_pool_free_blocks": int(pinned_pool.n_free_blocks()),
        },
        "regions": profiler.report_regions(),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{text}\n", encoding="utf-8")
        print(args.output)
    return 0


def _install_instrumentation(profiler: RunProfiler) -> None:
    for cls, name, label in (
        (CudaRunExecutor, "__init__", "cuda.initialize"),
        (
            TransportForcingProvider,
            "chunks_for_step",
            "host.forcing_select",
        ),
        (
            EmissionsOperator,
            "evaluate_surface_flux",
            "host.emissions",
        ),
    ):
        profiler.timed_method(cls, name, label)

    for cls, name, label, byte_arg in (
        (CudaRuntime, "to_device", "cuda.h2d_allocate", 0),
        (CudaRuntime, "copy_to_device", "cuda.h2d_refresh", 1),
        (CudaRuntime, "to_host", "cuda.d2h", 0),
        (CudaForcingChunks, "select", "cuda.forcing_select", None),
        (
            CudaPlanPreparation,
            "prepare_tpcore_step",
            "cuda.prepare_tpcore",
            None,
        ),
        (
            CudaPlanPreparation,
            "prepare_vdiff_and_convection",
            "cuda.prepare_vdiff_convection",
            None,
        ),
        (
            CudaTpcoreExecutor,
            "apply_blocks",
            "cuda.transport_tpcore",
            None,
        ),
        (
            CudaVdiffExecutor,
            "apply_blocks",
            "cuda.transport_vdiff",
            None,
        ),
        (
            CudaConvectionExecutor,
            "apply_blocks",
            "cuda.transport_convection",
            None,
        ),
    ):
        profiler.timed_method(
            cls,
            name,
            label,
            device=True,
            byte_arg=byte_arg,
        )

    profiler.instrument_kernel_attributes(
        CudaTpcoreExecutor,
        (
            ("_horizontal", "kernel.tpcore_horizontal"),
            ("_vertical", "kernel.tpcore_vertical"),
        ),
    )
    profiler.instrument_kernel_attributes(
        CudaVdiffExecutor,
        (("_kernel", "kernel.vdiff"),),
    )
    profiler.instrument_kernel_attributes(
        CudaConvectionExecutor,
        (("_kernel", "kernel.convection"),),
    )
    profiler.instrument_kernel_attributes(
        CudaPlanPreparation,
        tuple(
            (f"_{name}", f"kernel.prepare_{name}")
            for name in (
                "prepare_surface_endpoints",
                "average_surface_poles",
                "interpolate_meteorology",
                "tpcore_pressure_delta_terms",
                "tpcore_pressure_delta_sum",
                "tpcore_apply_pressure_fix",
                "tpcore_average_pressure",
                "tpcore_mass_flux",
                "tpcore_divergence_interior",
                "tpcore_divergence_poles",
                "tpcore_sum_vertical",
                "tpcore_pressure_rows",
                "tpcore_meridional_correction",
                "tpcore_zonal_correction",
                "tpcore_apply_correction",
                "tpcore_copy_pressure",
                "tpcore_pressure_terms",
                "tpcore_vertical_flux",
                "tpcore_cross_terms",
                "tpcore_initialize_jn_js",
                "cast_plan",
                "compute_vdiff_start",
                "prepare_vdiff",
                "prepare_convection",
            )
        ),
    )

    for name, label in (
        ("initialize_tracers", "host.initialize_tracers"),
        ("load_transport_grid", "host.load_grid"),
        ("build_tpcore_static_terms", "host.build_static"),
        ("_load_emissions_operator", "host.load_emissions"),
        ("_load_simulation_forcing", "host.initial_forcing"),
    ):
        profiler.timed_runner_function(name, label)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile one ordinary Wombat CUDA run with nested host and "
            "CUDA-event regions."
        )
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--steps", type=_positive_int, default=18)
    parser.add_argument("--warmup-steps", type=_nonnegative_int, default=1)
    parser.add_argument("--block-width", type=_positive_int, default=32)
    parser.add_argument("--device", type=_nonnegative_int, default=0)
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
